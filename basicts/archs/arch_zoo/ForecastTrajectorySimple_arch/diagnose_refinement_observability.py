"""Held-out diagnosis of inference-observable refinement signals.

This program is deliberately separate from policy training.  It may inspect
realized TRAIN/VALID route losses to ask whether a representation is capable of
predicting marginal route benefit, but it never writes a policy checkpoint and
none of its labels are consumed by ``run_online_sequential_rl``.

The diagnostic decision is made after the real native START->3 transition.
Consequently it compares only legal continuations of that reached state:
3->6->12 (reference), 3->12, and 3->4->6->12.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .progressive_selector import history_state_features
from .run_online_sequential_rl import TRAJECTORIES
from .run_pipeline import (
    DEFAULT_DATA_DIR,
    WindowDataset,
    load_data,
    make_loader,
    per_sample_mae,
    prepare_batch,
    route_name,
    seed_everything,
    to_physical,
)
from .run_selector import DEFAULT_BRIDGE_CHECKPOINT, build_frozen_forecaster
from .sequential_budget_policy import compact_forecast_state_features


REFERENCE = (3, 6, 12)
CONTINUATIONS = (REFERENCE, (3, 12), (3, 4, 6, 12))


class _NativeStepCapture:
    """Capture tensors that were already computed while producing native Z3."""

    def __init__(self, model):
        step = model.f2f.temporal_steps[0]
        self.outputs: dict[str, torch.Tensor] = {}
        modules = {
            "patch": step.patch_encoder,
            "down": step.downsamp_encoder,
            "linear": step.residual,
            "patch_hidden": step.patch_encoder.temporal_encoder,
            "down_hidden": step.downsamp_encoder.temporal_encoder,
        }
        self.handles = [
            module.register_forward_hook(self._hook(name))
            for name, module in modules.items()
        ]

    def _hook(self, name):
        def save(_module, _inputs, output):
            self.outputs[name] = output.detach()

        return save

    def clear(self):
        self.outputs.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def _moments(values: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.stack(
        (
            values.mean(dim=dim),
            values.std(dim=dim, unbiased=False),
            values.abs().mean(dim=dim),
            values.square().mean(dim=dim).sqrt(),
            values.amax(dim=dim),
            values.amin(dim=dim),
        ),
        dim=1,
    )


def _global_moments(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten(1)
    quantiles = torch.quantile(flat, torch.tensor([0.1, 0.5, 0.9], device=flat.device), dim=1).T
    return torch.cat(
        (
            flat.mean(1, keepdim=True),
            flat.std(1, unbiased=False, keepdim=True),
            flat.abs().mean(1, keepdim=True),
            flat.square().mean(1, keepdim=True).sqrt(),
            flat.amax(1, keepdim=True),
            flat.amin(1, keepdim=True),
            quantiles,
        ),
        dim=1,
    )


def _history_frequency_features(history: torch.Tensor) -> torch.Tensor:
    flow = history[..., 0]
    centered = flow - flow.mean(dim=1, keepdim=True)
    power = torch.fft.rfft(centered, dim=1).abs().square()[:, 1:]
    normalized = power / power.sum(dim=1, keepdim=True).clamp_min(1e-8)
    entropy = -(normalized * normalized.clamp_min(1e-8).log()).sum(dim=1)
    entropy = entropy / np.log(max(power.shape[1], 2))
    high_frequency = normalized[:, power.shape[1] // 2 :].sum(dim=1)
    dominant = normalized.amax(dim=1)
    first = flow[:, 1:] - flow[:, :-1]
    second = first[:, 1:] - first[:, :-1]
    node_signals = torch.stack(
        (
            entropy,
            high_frequency,
            dominant,
            first.std(dim=1, unbiased=False),
            first.abs().mean(dim=1),
            second.std(dim=1, unbiased=False),
            second.abs().mean(dim=1),
        ),
        dim=1,
    )
    # Retain sensor identity for linear probes and append permutation-stable
    # moments for nonlinear probes.
    return torch.cat(
        (node_signals.flatten(1), _global_moments(node_signals)), dim=1
    )


def _forecast_consistency_features(
    history: torch.Tensor, forecast: torch.Tensor
) -> torch.Tensor:
    flow = history[..., 0]
    z = forecast[..., 0]
    last = flow[:, -1]
    velocity = flow[:, -1] - flow[:, -2]
    steps = torch.arange(1, z.shape[1] + 1, device=z.device, dtype=z.dtype)
    linear = last[:, None, :] + steps[None, :, None] * velocity[:, None, :]
    persistence_error = z - last[:, None, :]
    linear_error = z - linear
    curvature = z[:, 2:] - 2.0 * z[:, 1:-1] + z[:, :-2]
    node_signals = torch.cat(
        (
            _moments(persistence_error, 1),
            _moments(linear_error, 1),
            _moments(curvature, 1),
        ),
        dim=1,
    )
    return torch.cat(
        (node_signals.flatten(1), _global_moments(node_signals)), dim=1
    )


def _branch_and_latent_features(captured: dict[str, torch.Tensor]) -> torch.Tensor:
    branches = torch.stack(
        (captured["patch"][..., 0], captured["down"][..., 0], captured["linear"][..., 0]),
        dim=1,
    )
    disagreement = branches.std(dim=1, unbiased=False)
    cancellation = branches.abs().sum(dim=1) / branches.sum(dim=1).abs().clamp_min(1e-4)
    pairwise = torch.stack(
        (
            (branches[:, 0] - branches[:, 1]).abs(),
            (branches[:, 0] - branches[:, 2]).abs(),
            (branches[:, 1] - branches[:, 2]).abs(),
        ),
        dim=1,
    )
    branch_nodes = torch.cat(
        (
            _moments(disagreement, 1),
            _moments(cancellation.clamp_max(100.0), 1),
            pairwise.mean(dim=2),
            pairwise.std(dim=2, unbiased=False),
        ),
        dim=1,
    )

    latent_parts = []
    for name in ("patch_hidden", "down_hidden"):
        hidden = captured[name]
        # [B, D, patches, N] -> inference-available activation energy by node.
        flattened = hidden.flatten(1, 2)
        latent_parts.extend(
            (
                flattened.mean(dim=1),
                flattened.std(dim=1, unbiased=False),
                flattened.abs().mean(dim=1),
                flattened.square().mean(dim=1).sqrt(),
            )
        )
    latent_nodes = torch.stack(latent_parts, dim=1)
    all_nodes = torch.cat((branch_nodes, latent_nodes), dim=1)
    return torch.cat((all_nodes.flatten(1), _global_moments(all_nodes)), dim=1)


@torch.inference_mode()
def build_split(model, loader, device, mean, std, max_batches=None):
    current, enhanced, losses = [], [], []
    capture = _NativeStepCapture(model)
    route_indices = [TRAJECTORIES.index(route) for route in CONTINUATIONS]
    try:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            history, target = prepare_batch(batch, device)
            target_raw = to_physical(target, mean, std)
            capture.clear()
            z3 = model.execute_transition(history, None, 3, None)
            captured = dict(capture.outputs)
            if len(captured) != 5:
                raise RuntimeError(f"Missing native-step captures: {captured.keys()}")
            base = compact_forecast_state_features(history, z3)
            proposed = torch.cat(
                (
                    base,
                    _history_frequency_features(history),
                    _forecast_consistency_features(history, z3),
                    _branch_and_latent_features(captured),
                ),
                dim=1,
            )
            route_losses = []
            for route in CONTINUATIONS:
                prediction = model.execute_trajectory(history, route)["pred"]
                route_losses.append(
                    per_sample_mae(to_physical(prediction, mean, std), target_raw, 0.0)
                )
            current.append(base.cpu())
            enhanced.append(proposed.cpu())
            losses.append(torch.stack(route_losses, dim=1).cpu())
    finally:
        capture.close()
    return {
        "current": torch.cat(current).numpy(),
        "enhanced": torch.cat(enhanced).numpy(),
        "losses": torch.cat(losses).numpy(),
        "routes": [TRAJECTORIES[index] for index in route_indices],
    }


def _correlation(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _ranking_metrics(true_gain, predicted_gain):
    positive = true_gain > 0.0
    auc = None
    if positive.any() and (~positive).any():
        auc = float(roc_auc_score(positive, predicted_gain))
    count = max(1, int(round(0.10 * len(true_gain))))
    true_top = np.argsort(true_gain)[-count:]
    pred_top = np.argsort(predicted_gain)[-count:]
    overlap = len(set(true_top.tolist()).intersection(pred_top.tolist()))
    return {
        "pearson": _correlation(true_gain, predicted_gain),
        "spearman": float(spearmanr(true_gain, predicted_gain).statistic),
        "benefit_auc": auc,
        "top10_precision": float(overlap / count),
        "top10_recall": float(overlap / count),
        "positive_fraction": float(positive.mean()),
    }


def evaluate_probe(name, estimator, train_x, valid_x, train_losses, valid_losses):
    train_delta = train_losses[:, 1:] - train_losses[:, :1]
    estimator.fit(train_x, train_delta)
    predicted_delta = np.asarray(estimator.predict(valid_x))
    if predicted_delta.ndim == 1:
        predicted_delta = predicted_delta[:, None]
    predicted_all = np.concatenate(
        (np.zeros((len(predicted_delta), 1), dtype=predicted_delta.dtype), predicted_delta),
        axis=1,
    )
    selected = predicted_all.argmin(axis=1)
    rows = np.arange(len(selected))
    selected_loss = valid_losses[rows, selected]
    true_gain = valid_losses[:, 0] - valid_losses.min(axis=1)
    predicted_gain = -predicted_all.min(axis=1)
    counts = np.bincount(selected, minlength=valid_losses.shape[1])
    best_alternative = predicted_delta.argmin(axis=1) + 1
    best_alternative_delta = predicted_delta.min(axis=1)
    ranking = np.argsort(best_alternative_delta)
    selective = {}
    for fraction in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50):
        count = max(1, int(round(fraction * len(ranking))))
        chosen_rows = ranking[:count]
        chosen_routes = best_alternative[chosen_rows]
        selective_loss = valid_losses[:, 0].copy()
        selective_loss[chosen_rows] = valid_losses[chosen_rows, chosen_routes]
        selective[str(fraction)] = {
            "selected_valid_mae": float(selective_loss.mean()),
            "gain_vs_canonical": float(
                valid_losses[:, 0].mean() - selective_loss.mean()
            ),
        }
    per_action_ranking = {}
    for index in range(predicted_delta.shape[1]):
        actual_delta = valid_losses[:, index + 1] - valid_losses[:, 0]
        beneficial = actual_delta < 0.0
        score = -predicted_delta[:, index]
        action_auc = None
        if beneficial.any() and (~beneficial).any():
            action_auc = float(roc_auc_score(beneficial, score))
        top_count = max(1, int(round(0.10 * len(score))))
        top = np.argsort(score)[-top_count:]
        per_action_ranking[route_name(CONTINUATIONS[index + 1])] = {
            "benefit_auc": action_auc,
            "positive_fraction": float(beneficial.mean()),
            "top10_positive_precision": float(beneficial[top].mean()),
            "top10_mean_true_gain": float((-actual_delta[top]).mean()),
        }

    # A target-trained uncertainty head is allowed by the methodology, but at
    # inference it sees only the same observable representation.  Diagnose the
    # strongest simple version: predict canonical absolute error, then ask if
    # high predicted error actually ranks refinement gain.
    confidence_estimator = clone(estimator)
    confidence_estimator.fit(train_x, train_losses[:, 0])
    predicted_error = np.asarray(confidence_estimator.predict(valid_x)).reshape(-1)
    return {
        "estimator": name,
        "canonical_valid_mae": float(valid_losses[:, 0].mean()),
        "continuation_oracle_valid_mae": float(valid_losses.min(axis=1).mean()),
        "selected_valid_mae": float(selected_loss.mean()),
        "gain_vs_canonical": float(valid_losses[:, 0].mean() - selected_loss.mean()),
        "route_counts": {
            route_name(route): int(counts[index])
            for index, route in enumerate(CONTINUATIONS)
        },
        "ranking": _ranking_metrics(true_gain, predicted_gain),
        "selective_refinement_frontier": selective,
        "per_action_ranking": per_action_ranking,
        "error_confidence": {
            "canonical_error_pearson": _correlation(
                valid_losses[:, 0], predicted_error
            ),
            "refinement_gain_pearson": _correlation(true_gain, predicted_error),
            "refinement_gain_spearman": float(
                spearmanr(true_gain, predicted_error).statistic
            ),
        },
        "per_action_delta_correlation": {
            route_name(CONTINUATIONS[index + 1]): {
                "pearson": _correlation(
                    valid_losses[:, index + 1] - valid_losses[:, 0],
                    predicted_delta[:, index],
                ),
                "spearman": float(
                    spearmanr(
                        valid_losses[:, index + 1] - valid_losses[:, 0],
                        predicted_delta[:, index],
                    ).statistic
                ),
            }
            for index in range(predicted_delta.shape[1])
        },
    }


def probes(train, valid):
    results = {}
    for feature_name in ("current", "enhanced"):
        train_x, valid_x = train[feature_name], valid[feature_name]
        estimators = {
            "ridge": make_pipeline(
                StandardScaler(), Ridge(alpha=100.0)
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=160,
                min_samples_leaf=20,
                max_features=0.05 if feature_name == "enhanced" else 0.5,
                n_jobs=-1,
                random_state=17,
            ),
        }
        # Histogram boosting is useful on the compact global representation,
        # but sklearn implements it as one scalar target at a time.
        for estimator_name, estimator in estimators.items():
            key = f"{feature_name}_{estimator_name}"
            print(f"[probe] {key} train={train_x.shape} valid={valid_x.shape}")
            results[key] = evaluate_probe(
                key, estimator, train_x, valid_x, train["losses"], valid["losses"]
            )
            print(json.dumps(results[key], sort_keys=True))
    return results


def parse_args():
    from .run_online_sequential_rl import parse_args as online_parse_args

    parser = argparse.ArgumentParser(
        description="Diagnose target-free observability of marginal refinement benefit."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--f2f-checkpoint", type=Path)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reuse forecaster construction defaults without duplicating model options.
    import sys

    old = sys.argv
    try:
        sys.argv = [old[0]]
        defaults = online_parse_args()
    finally:
        sys.argv = old
    for key in ("config", "f2f_checkpoint"):
        if getattr(args, key) is None:
            setattr(args, key, getattr(defaults, key))
    args.bridge_correction_limit = defaults.bridge_correction_limit
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    data, indices, mean, std = load_data(args.data_dir)
    datasets = {split: WindowDataset(data, indices[split]) for split in ("train", "valid")}

    def loader(split):
        return make_loader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=False,
            workers=args.workers,
            device=device,
            seed=args.seed,
        )

    model = build_frozen_forecaster(args, device)
    train = build_split(
        model, loader("train"), device, mean, std, args.max_train_batches
    )
    valid = build_split(
        model, loader("valid"), device, mean, std, args.max_eval_batches
    )
    print(
        f"[features] current={train['current'].shape[1]} "
        f"enhanced={train['enhanced'].shape[1]}"
    )
    result = {
        "purpose": "diagnosis only; never policy supervision",
        "decision_state": "after actually executing native START->3",
        "target_available_at_inference": False,
        "routes": [route_name(route) for route in CONTINUATIONS],
        "train_samples": len(train["losses"]),
        "valid_samples": len(valid["losses"]),
        "probes": probes(train, valid),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
