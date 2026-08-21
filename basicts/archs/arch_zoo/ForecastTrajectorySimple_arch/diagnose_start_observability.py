"""Diagnosis-only test of target-free signals available before the first edge.

Unlike the online learner, this script is allowed to use realized route losses
as held-out diagnostic labels.  The inputs contain history only: raw online
history summaries, temporal-frequency statistics, and disagreement among four
cheap deterministic trend extrapolations.  No future target or forecast from
an unexecuted graph edge is an input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .diagnose_refinement_observability import (
    _global_moments,
    _history_frequency_features,
)
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


REFERENCE = (3, 6, 12)
ROUTES = (REFERENCE,) + tuple(route for route in TRAJECTORIES if route != REFERENCE)


def cheap_view_disagreement(history: torch.Tensor) -> torch.Tensor:
    """Disagreement among persistence and multi-window trend extrapolations."""
    flow = history[..., 0]
    steps = torch.arange(1, 13, device=flow.device, dtype=flow.dtype)[None, :, None]
    views = [flow[:, -1:, :].expand(-1, 12, -1)]
    for window in (2, 4, 8):
        used = min(window, flow.shape[1] - 1)
        slope = (flow[:, -1] - flow[:, -1 - used]) / float(used)
        views.append(flow[:, -1:, :] + steps * slope[:, None, :])
    stacked = torch.stack(views, dim=1)
    disagreement = stacked.std(dim=1, unbiased=False)
    range_ = stacked.amax(dim=1) - stacked.amin(dim=1)
    node_signals = torch.stack(
        (
            disagreement.mean(dim=1),
            disagreement.std(dim=1, unbiased=False),
            disagreement.amax(dim=1),
            range_.mean(dim=1),
            range_.amax(dim=1),
        ),
        dim=1,
    )
    return torch.cat((node_signals.flatten(1), _global_moments(node_signals)), dim=1)


@torch.inference_mode()
def build_split(model, loader, device, mean, std, max_batches=None):
    current, enhanced, losses = [], [], []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target = prepare_batch(batch, device)
        target_raw = to_physical(target, mean, std)
        base = history_state_features(history)
        proposed = torch.cat(
            (
                base,
                _history_frequency_features(history),
                cheap_view_disagreement(history),
            ),
            dim=1,
        )
        route_losses = []
        for route in ROUTES:
            prediction = model.execute_trajectory(history, route)["pred"]
            route_losses.append(
                per_sample_mae(to_physical(prediction, mean, std), target_raw, 0.0)
            )
        current.append(base.cpu())
        enhanced.append(proposed.cpu())
        losses.append(torch.stack(route_losses, dim=1).cpu())
    return {
        "current": torch.cat(current).numpy(),
        "enhanced": torch.cat(enhanced).numpy(),
        "losses": torch.cat(losses).numpy(),
    }


def correlation(left, right):
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def evaluate(estimator, train_x, valid_x, train_losses, valid_losses):
    train_delta = train_losses[:, 1:] - train_losses[:, :1]
    estimator.fit(train_x, train_delta)
    predicted_delta = np.asarray(estimator.predict(valid_x))
    predicted = np.concatenate(
        (np.zeros((len(predicted_delta), 1), dtype=predicted_delta.dtype), predicted_delta),
        axis=1,
    )
    selected = predicted.argmin(axis=1)
    rows = np.arange(len(selected))
    actual = valid_losses[rows, selected]
    true_gain = valid_losses[:, 0] - valid_losses.min(axis=1)
    predicted_gain = -predicted.min(axis=1)
    positive = true_gain > 0
    auc = float(roc_auc_score(positive, predicted_gain)) if positive.any() and (~positive).any() else None
    count = max(1, round(0.1 * len(rows)))
    true_top = set(np.argsort(true_gain)[-count:].tolist())
    predicted_top = set(np.argsort(predicted_gain)[-count:].tolist())
    counts = np.bincount(selected, minlength=len(ROUTES))

    best_alternative = predicted_delta.argmin(axis=1) + 1
    ranking = np.argsort(predicted_delta.min(axis=1))
    selective = {}
    for fraction in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50):
        selected_count = max(1, round(fraction * len(rows)))
        chosen_rows = ranking[:selected_count]
        chosen = best_alternative[chosen_rows]
        losses = valid_losses[:, 0].copy()
        losses[chosen_rows] = valid_losses[chosen_rows, chosen]
        selective[str(fraction)] = {
            "mae": float(losses.mean()),
            "gain": float(valid_losses[:, 0].mean() - losses.mean()),
        }
    return {
        "canonical_mae": float(valid_losses[:, 0].mean()),
        "oracle_mae": float(valid_losses.min(axis=1).mean()),
        "selected_mae": float(actual.mean()),
        "gain_vs_canonical": float(valid_losses[:, 0].mean() - actual.mean()),
        "gain_pearson": correlation(true_gain, predicted_gain),
        "gain_spearman": float(spearmanr(true_gain, predicted_gain).statistic),
        "gain_auc": auc,
        "top10_overlap": float(len(true_top.intersection(predicted_top)) / count),
        "route_counts": {
            route_name(route): int(counts[index]) for index, route in enumerate(ROUTES)
        },
        "selective_frontier": selective,
    }


def parse_args():
    from .run_online_sequential_rl import parse_args as online_parse_args
    import sys

    parser = argparse.ArgumentParser()
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
            datasets[split], batch_size=args.batch_size, shuffle=False,
            workers=args.workers, device=device, seed=args.seed,
        )

    model = build_frozen_forecaster(args, device)
    train = build_split(model, loader("train"), device, mean, std, args.max_train_batches)
    valid = build_split(model, loader("valid"), device, mean, std, args.max_eval_batches)
    result = {
        "purpose": "diagnosis only; never policy supervision",
        "target_available_at_inference": False,
        "routes": [route_name(route) for route in ROUTES],
        "train_samples": len(train["losses"]),
        "valid_samples": len(valid["losses"]),
        "feature_dimensions": {
            "current": train["current"].shape[1],
            "enhanced": train["enhanced"].shape[1],
        },
        "probes": {},
    }
    for features in ("current", "enhanced"):
        estimators = {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=160, min_samples_leaf=20,
                max_features=0.05, n_jobs=-1, random_state=17,
            ),
        }
        for name, estimator in estimators.items():
            key = f"{features}_{name}"
            print(f"[probe] {key} train={train[features].shape}")
            result["probes"][key] = evaluate(
                estimator, train[features], valid[features], train["losses"], valid["losses"]
            )
            print(json.dumps(result["probes"][key], sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
