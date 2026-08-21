#!/usr/bin/env python3
"""TRAIN/VALID-only full-route cache, cost, oracle, and probe pipeline.

The selected route-complete forecaster is frozen. No TEST dataset is ever
constructed. FLOPs are the primary normalized cost; measured batch-1 CUDA
latency is reported as deployment calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.metrics import masked_mae
from scripts.f2f_cot_resolution_native_v1_experiment import model_args
from scripts.f2f_cot_runtime import (
    NULL_VAL,
    load_rescale,
    make_loader,
    per_sample_mae,
    select_batch,
)
from scripts.train_f2f_cot_resolution_native_route_complete import (
    CONTAINMENT_TOLERANCE,
    EXPECTED_CANONICAL_ROUTE,
    EXPECTED_ROUTES,
    assert_exact_shared_prefixes,
    call_shared_rollout,
    dump_json,
    load_architecture,
    route_edges,
    route_key,
    seed_all,
)


EXPERIMENT = "f2f_cot_resolution_native_route_complete_oracle"
DECISION_RESOLUTIONS = (0, 2, 3)
# Prefix order matches the sequential policy LEGAL_NEXT / PREFIXES contract.
POLICY_PREFIXES = (
    (),
    (2,),
    (2, 4),
    (2, 6),
    (3,),
    (3, 6),
    (4,),
    (6,),
)
PREFIX_FEATURE_SOURCE = {
    (): ((12,), None),
    (2,): ((2, 12), 0),
    (2, 4): ((2, 4, 12), 1),
    (2, 6): ((2, 6, 12), 1),
    (3,): ((3, 12), 0),
    (3, 6): ((3, 6, 12), 1),
    (4,): ((4, 12), 0),
    (6,): ((6, 12), 0),
}
ORACLE_HEADROOM_GATE = 0.03
RECOVERY_GATE = 0.25
AUC_GATE = 0.60
BALANCED_ACCURACY_GATE = 0.55


def _summary(value: torch.Tensor) -> torch.Tensor:
    flat = value.reshape(value.shape[0], -1)
    return torch.stack(
        (
            flat.mean(1),
            flat.std(1, unbiased=False),
            flat.abs().mean(1),
            flat.square().mean(1).sqrt(),
            flat.amax(1),
            flat.amin(1),
        ),
        dim=1,
    )


def _evidence_tokens(output: dict) -> torch.Tensor:
    evidence = getattr(output["state"], "evidence", None)
    tokens = getattr(evidence, "tokens", None)
    if tokens is None:
        raise AttributeError(
            "route output state.evidence.tokens is required for inference-safe features"
        )
    return tokens


def _step_vector(
    history: torch.Tensor,
    output: dict,
    rescale,
    resolution: int,
    forecast: torch.Tensor | None,
    diagnostics: Mapping[str, Any] | None,
) -> torch.Tensor:
    """Fixed-width features using only X and already-executed state."""
    batch = history.shape[0]
    raw_history = rescale(history[..., 0:1])
    last_delta = raw_history[:, -1] - raw_history[:, -2]
    evidence = _evidence_tokens(output)
    zero_summary = history.new_zeros((batch, 6))
    if forecast is None:
        forecast_summary = projected_summary = gap_summary = zero_summary
    else:
        forecast_raw = rescale(forecast)
        projected = forecast_raw.repeat_interleave(12 // int(resolution), dim=1)
        persistence = raw_history[:, -1:].expand_as(projected)
        forecast_summary = _summary(forecast_raw)
        projected_summary = _summary(projected)
        gap_summary = _summary(projected - persistence)

    diagnostic_parts = []
    for key in (
        "raw_correction",
        "low_frequency_correction",
        "detail_correction",
    ):
        value = None if diagnostics is None else diagnostics.get(key)
        diagnostic_parts.append(_summary(value) if torch.is_tensor(value) else zero_summary)
    branch = None if diagnostics is None else diagnostics.get("branch_scale")
    if not torch.is_tensor(branch):
        branch = history.new_zeros((batch, 3))
    branch = branch.reshape(batch, -1)
    if branch.shape[1] < 3:
        branch = torch.nn.functional.pad(branch, (0, 3 - branch.shape[1]))
    branch = branch[:, :3]
    low_gain = None if diagnostics is None else diagnostics.get("low_frequency_gain")
    if not torch.is_tensor(low_gain):
        low_gain = history.new_zeros((batch, 1))
    low_gain = low_gain.reshape(batch, -1)[:, :1]
    resolution_code = history.new_zeros((batch, len(DECISION_RESOLUTIONS)))
    if int(resolution) in DECISION_RESOLUTIONS:
        resolution_code[:, DECISION_RESOLUTIONS.index(int(resolution))] = 1.0
    return torch.cat(
        (
            _summary(raw_history),
            _summary(last_delta),
            _summary(evidence),
            forecast_summary,
            projected_summary,
            gap_summary,
            *diagnostic_parts,
            branch,
            low_gain,
            resolution_code,
        ),
        dim=1,
    )


def inference_safe_features(history, outputs, rescale):
    representative = next(iter(outputs.values()))
    root = _step_vector(history, representative, rescale, 0, None, None)
    decisions = {0: root}
    for resolution in (2, 3):
        output = next(
            value for route, value in outputs.items() if route[0] == resolution
        )
        decisions[resolution] = _step_vector(
            history,
            output,
            rescale,
            resolution,
            output["forecasts"][0],
            output["steps"][0],
        )
    route_features = []
    for route in EXPECTED_ROUTES:
        output = outputs[route]
        if len(route) == 1:
            route_features.append(root)
        else:
            index = len(route) - 2
            resolution = route[index]
            # All possible preterminal resolutions are 2/3/4/6. Feature code
            # reserves one-hot slots for branching states 0/2/3; 4/6 retain a
            # zero one-hot because no route choice exists there.
            feature = _step_vector(
                history,
                output,
                rescale,
                resolution,
                output["forecasts"][index],
                output["steps"][index],
            )
            route_features.append(feature)
    prefix_features = []
    for prefix in POLICY_PREFIXES:
        representative_route, step_index = PREFIX_FEATURE_SOURCE[prefix]
        if step_index is None:
            prefix_features.append(root)
            continue
        output = outputs[representative_route]
        prefix_features.append(
            _step_vector(
                history,
                output,
                rescale,
                representative_route[step_index],
                output["forecasts"][step_index],
                output["steps"][step_index],
            )
        )
    return (
        torch.stack([decisions[value] for value in DECISION_RESOLUTIONS], dim=1),
        torch.stack(route_features, dim=1),
        torch.stack(prefix_features, dim=1),
    )


@torch.inference_mode()
def build_split_cache(model, loader, split, device, rescale, max_batches=None):
    model.eval()
    chunks = {
        "indices": [],
        "mae": [],
        "decision_features": [],
        "route_features": [],
        "state_features": [],
    }
    batch_mae = {route: [] for route in EXPECTED_ROUTES}
    identity = None
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target, indices = select_batch(batch, device)
        raw, outputs = call_shared_rollout(model, history)
        current_identity = assert_exact_shared_prefixes(model, raw, outputs)
        identity = identity or current_identity
        target_raw = rescale(target)
        losses = []
        for route in EXPECTED_ROUTES:
            prediction = rescale(outputs[route]["pred"])
            losses.append(per_sample_mae(prediction, target_raw))
            batch_mae[route].append(
                float(masked_mae(prediction, target_raw, NULL_VAL))
            )
        decision_features, route_features, state_features = inference_safe_features(
            history, outputs, rescale
        )
        chunks["indices"].append(indices.cpu())
        chunks["mae"].append(torch.stack(losses, dim=1).cpu())
        chunks["decision_features"].append(decision_features.cpu())
        chunks["route_features"].append(route_features.cpu())
        chunks["state_features"].append(state_features.cpu())
    if not chunks["indices"]:
        raise RuntimeError(f"{split} cache received zero batches")
    arrays = {name: torch.cat(values).numpy() for name, values in chunks.items()}
    order = np.argsort(arrays["indices"], kind="stable")
    arrays = {name: value[order] for name, value in arrays.items()}
    if len(np.unique(arrays["indices"])) != len(arrays["indices"]):
        raise RuntimeError(f"{split} cache contains duplicate sample indices")
    report = {
        "split": split,
        "samples": int(len(arrays["indices"])),
        "routes": [list(route) for route in EXPECTED_ROUTES],
        "route_batch_mean_MAE": {
            route_key(route): float(np.mean(batch_mae[route]))
            for route in EXPECTED_ROUTES
        },
        "per_sample_route_mean_MAE": {
            route_key(route): float(arrays["mae"][:, index].mean())
            for index, route in enumerate(EXPECTED_ROUTES)
        },
        "decision_feature_shape": list(arrays["decision_features"].shape),
        "route_feature_shape": list(arrays["route_features"].shape),
        "state_feature_shape": list(arrays["state_features"].shape),
        "policy_prefixes": [list(prefix) for prefix in POLICY_PREFIXES],
        "decision_feature_resolutions": list(DECISION_RESOLUTIONS),
        "uses_target_in_features": False,
        "shared_prefix_identity": identity,
    }
    return arrays, report


def save_cache(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, default=str)),
    )


def load_cache(path: Path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            name: data[name]
            for name in (
                "indices",
                "mae",
                "decision_features",
                "route_features",
                "state_features",
            )
        }
        metadata = json.loads(str(data["metadata_json"]))
    return arrays, metadata


def _percentile_interval(values) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap_indices(n: int, repeats: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(repeats, n))


def _best_fixed_mixture(means: np.ndarray, costs: np.ndarray, budget: float):
    best = (math.inf, None)
    for index, (mae, cost) in enumerate(zip(means, costs)):
        if cost <= budget + 1e-12 and mae < best[0]:
            best = (float(mae), {index: 1.0})
    for left in range(len(costs)):
        for right in range(left + 1, len(costs)):
            low, high = (
                (left, right) if costs[left] <= costs[right] else (right, left)
            )
            if costs[low] > budget or costs[high] <= costs[low]:
                continue
            high_weight = np.clip(
                (budget - costs[low]) / (costs[high] - costs[low]), 0.0, 1.0
            )
            value = (1.0 - high_weight) * means[low] + high_weight * means[high]
            if value < best[0]:
                best = (
                    float(value),
                    {low: float(1.0 - high_weight), high: float(high_weight)},
                )
    return best


def unconstrained_oracle(
    losses: np.ndarray, boot: np.ndarray
) -> dict[str, Any]:
    means = losses.mean(0)
    best = int(means.argmin())
    oracle_loss = losses.min(1)
    headroom = float(means[best] - oracle_loss.mean())
    boot_means = losses[boot].mean(1)
    boot_oracle = losses[boot].min(2).mean(1)
    boot_headroom = boot_means.min(1) - boot_oracle
    route_ci = {
        route_key(route): _percentile_interval(boot_means[:, index])
        for index, route in enumerate(EXPECTED_ROUTES)
    }
    choices = losses.argmin(1)
    return {
        "route_mean_MAE": {
            route_key(route): float(means[index])
            for index, route in enumerate(EXPECTED_ROUTES)
        },
        "route_mean_MAE_bootstrap_95pct_CI": route_ci,
        "best_fixed_route": list(EXPECTED_ROUTES[best]),
        "best_fixed_MAE": float(means[best]),
        "sample_oracle_MAE": float(oracle_loss.mean()),
        "sample_oracle_MAE_bootstrap_95pct_CI": _percentile_interval(boot_oracle),
        "oracle_headroom": headroom,
        "oracle_headroom_bootstrap_95pct_CI": _percentile_interval(boot_headroom),
        "oracle_route_histogram": {
            route_key(EXPECTED_ROUTES[index]): int(count)
            for index, count in sorted(Counter(choices.tolist()).items())
        },
    }


def hard_budget_frontier(
    losses: np.ndarray, costs: np.ndarray, boot: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for budget in sorted(set(float(value) for value in costs)):
        feasible = np.flatnonzero(costs <= budget + 1e-12)
        subset = losses[:, feasible]
        means = subset.mean(0)
        fixed_local = int(means.argmin())
        fixed_index = int(feasible[fixed_local])
        oracle = subset.min(1)
        choices = feasible[subset.argmin(1)]
        boot_subset = subset[boot]
        boot_fixed = boot_subset.mean(1).min(1)
        boot_oracle = boot_subset.min(2).mean(1)
        rows.append(
            {
                "budget_normalized_FLOPs": budget,
                "feasible_routes": [list(EXPECTED_ROUTES[index]) for index in feasible],
                "best_fixed_route": list(EXPECTED_ROUTES[fixed_index]),
                "best_fixed_MAE": float(losses[:, fixed_index].mean()),
                "sample_oracle_MAE": float(oracle.mean()),
                "oracle_headroom": float(losses[:, fixed_index].mean() - oracle.mean()),
                "oracle_headroom_bootstrap_95pct_CI": _percentile_interval(
                    boot_fixed - boot_oracle
                ),
                "oracle_mean_cost": float(costs[choices].mean()),
            }
        )
    return rows


def matched_expected_cost_frontier(
    losses: np.ndarray,
    costs: np.ndarray,
    boot: np.ndarray,
    lambdas: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if lambdas is None:
        lambdas = np.concatenate(([0.0], np.logspace(-4, 4, 49)))
    rows = []
    seen = set()
    route_means = losses.mean(0)
    for penalty in lambdas:
        chosen = (losses + float(penalty) * costs[None]).argmin(1)
        signature = tuple(np.bincount(chosen, minlength=len(costs)).tolist())
        if signature in seen:
            continue
        seen.add(signature)
        selected_loss = losses[np.arange(len(losses)), chosen]
        mean_cost = float(costs[chosen].mean())
        fixed_mae, mixture = _best_fixed_mixture(route_means, costs, mean_cost)
        boot_headroom = []
        for sample_indices in boot:
            sampled_losses = losses[sample_indices]
            sampled_chosen = chosen[sample_indices]
            adaptive_mae = float(
                sampled_losses[
                    np.arange(len(sample_indices)), sampled_chosen
                ].mean()
            )
            sampled_means = sampled_losses.mean(0)
            sampled_cost = float(costs[sampled_chosen].mean())
            sampled_fixed, _ = _best_fixed_mixture(
                sampled_means, costs, sampled_cost
            )
            boot_headroom.append(sampled_fixed - adaptive_mae)
        rows.append(
            {
                "lambda": float(penalty),
                "mean_normalized_FLOPs": mean_cost,
                "sample_oracle_penalized_MAE": float(selected_loss.mean()),
                "matched_fixed_mixture_MAE": float(fixed_mae),
                "matched_fixed_mixture": {
                    route_key(EXPECTED_ROUTES[index]): weight
                    for index, weight in (mixture or {}).items()
                    if weight > 1e-9
                },
                "matched_cost_oracle_headroom": float(
                    fixed_mae - selected_loss.mean()
                ),
                "matched_cost_headroom_bootstrap_95pct_CI": _percentile_interval(
                    boot_headroom
                ),
                "oracle_route_histogram": {
                    route_key(EXPECTED_ROUTES[index]): int(count)
                    for index, count in sorted(Counter(chosen.tolist()).items())
                },
            }
        )
    return sorted(rows, key=lambda row: row["mean_normalized_FLOPs"])


def analyze_split(
    arrays: Mapping[str, np.ndarray],
    normalized_costs: np.ndarray,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    losses = arrays["mae"].astype(np.float64)
    boot = bootstrap_indices(len(losses), bootstrap_repeats, seed)
    return {
        "samples": int(len(losses)),
        "unconstrained": unconstrained_oracle(losses, boot),
        "hard_budget_frontier": hard_budget_frontier(losses, normalized_costs, boot),
        "matched_expected_cost_frontier": matched_expected_cost_frontier(
            losses, normalized_costs, boot
        ),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def train_valid_stability(train_losses, valid_losses, train_report, valid_report):
    train_means = train_losses.mean(0)
    valid_means = valid_losses.mean(0)
    train_share = np.bincount(train_losses.argmin(1), minlength=8) / len(train_losses)
    valid_share = np.bincount(valid_losses.argmin(1), minlength=8) / len(valid_losses)
    pearson = float(np.corrcoef(train_means, valid_means)[0, 1])
    spearman = float(np.corrcoef(rankdata(train_means), rankdata(valid_means))[0, 1])
    train_headroom = train_report["unconstrained"]["oracle_headroom"]
    valid_headroom = valid_report["unconstrained"]["oracle_headroom"]
    return {
        "route_mean_MAE_delta_VALID_minus_TRAIN": {
            route_key(route): float(valid_means[index] - train_means[index])
            for index, route in enumerate(EXPECTED_ROUTES)
        },
        "route_mean_pearson": pearson,
        "route_rank_spearman": spearman,
        "oracle_route_share_TRAIN": train_share.tolist(),
        "oracle_route_share_VALID": valid_share.tolist(),
        "oracle_route_share_total_variation": float(
            0.5 * np.abs(train_share - valid_share).sum()
        ),
        "oracle_headroom_TRAIN": float(train_headroom),
        "oracle_headroom_VALID": float(valid_headroom),
        "oracle_headroom_VALID_minus_TRAIN": float(valid_headroom - train_headroom),
    }


def _safe_auc(y_true, probabilities, classes) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return 0.5


def _fit_classifier(x_train, y_train, x_valid, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classes = np.unique(y_train)
    if len(classes) < 2:
        return None, classes
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
        ),
    )
    classifier.fit(x_train, y_train)
    return (
        classifier.predict(x_valid),
        classifier.predict_proba(x_valid),
        classifier[-1].classes_,
    ), classes


def _selection_metrics(valid_losses, selected, oracle=None):
    selected_loss = valid_losses[np.arange(len(valid_losses)), selected]
    means = valid_losses.mean(0)
    best_fixed = float(means.min())
    oracle_loss = valid_losses.min(1) if oracle is None else oracle
    headroom = best_fixed - float(oracle_loss.mean())
    recovery = (
        0.0
        if headroom <= 1e-12
        else (best_fixed - float(selected_loss.mean())) / headroom
    )
    return {
        "selected_MAE": float(selected_loss.mean()),
        "best_fixed_VALID_MAE": best_fixed,
        "oracle_MAE": float(oracle_loss.mean()),
        "oracle_headroom": float(headroom),
        "oracle_headroom_recovery": float(recovery),
    }


def multiclass_probe(train_arrays, valid_arrays, target: str, seed: int):
    from sklearn.metrics import balanced_accuracy_score

    train_losses = train_arrays["mae"]
    valid_losses = valid_arrays["mae"]
    x_train = train_arrays["decision_features"][:, 0]
    x_valid = valid_arrays["decision_features"][:, 0]
    oracle_route_train = train_losses.argmin(1)
    oracle_route_valid = valid_losses.argmin(1)
    if target == "route":
        y_train, y_valid = oracle_route_train, oracle_route_valid
        fitted, classes = _fit_classifier(x_train, y_train, x_valid, seed)
        if fitted is None:
            return {"skipped": True, "reason": "degenerate TRAIN labels"}
        prediction, probability, model_classes = fitted
        selected_routes = prediction.astype(int)
        selection = _selection_metrics(valid_losses, selected_routes)
    elif target == "top_action":
        actions = sorted({route[0] for route in EXPECTED_ROUTES})
        action_to_id = {action: index for index, action in enumerate(actions)}
        route_action = np.asarray([action_to_id[route[0]] for route in EXPECTED_ROUTES])
        y_train = route_action[oracle_route_train]
        y_valid = route_action[oracle_route_valid]
        fitted, classes = _fit_classifier(x_train, y_train, x_valid, seed)
        if fitted is None:
            return {"skipped": True, "reason": "degenerate TRAIN labels"}
        prediction, probability, model_classes = fitted
        representative = {}
        means = train_losses.mean(0)
        for action_id in range(len(actions)):
            members = np.flatnonzero(route_action == action_id)
            representative[action_id] = int(members[np.argmin(means[members])])
        selected_routes = np.asarray(
            [representative[int(value)] for value in prediction], dtype=int
        )
        selection = _selection_metrics(valid_losses, selected_routes)
    else:
        raise ValueError(target)
    return {
        "skipped": False,
        "target": target,
        "TRAIN_classes": [int(value) for value in classes],
        "VALID_balanced_accuracy": float(
            balanced_accuracy_score(y_valid, prediction)
        ),
        "VALID_macro_ROC_AUC_OVR": _safe_auc(
            y_valid, probability, model_classes
        ),
        "selection": selection,
    }


def binary_decision_probe(
    train_arrays,
    valid_arrays,
    decision_resolution: int,
    jump_route_ids: Sequence[int],
    continue_route_ids: Sequence[int],
    seed: int,
):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    train_losses = train_arrays["mae"]
    valid_losses = valid_arrays["mae"]
    decision_index = DECISION_RESOLUTIONS.index(decision_resolution)
    x_train = train_arrays["decision_features"][:, decision_index]
    x_valid = valid_arrays["decision_features"][:, decision_index]
    train_means = train_losses.mean(0)
    jump_id = int(min(jump_route_ids, key=lambda value: train_means[value]))
    continue_id = int(min(continue_route_ids, key=lambda value: train_means[value]))
    pair = np.asarray([jump_id, continue_id])
    y_train = (train_losses[:, continue_id] < train_losses[:, jump_id]).astype(int)
    y_valid = (valid_losses[:, continue_id] < valid_losses[:, jump_id]).astype(int)
    fitted, classes = _fit_classifier(x_train, y_train, x_valid, seed)
    if fitted is None:
        return {
            "skipped": True,
            "reason": "degenerate TRAIN labels",
            "jump_route": list(EXPECTED_ROUTES[jump_id]),
            "continue_route": list(EXPECTED_ROUTES[continue_id]),
        }
    prediction, probability, model_classes = fitted
    selected = pair[prediction.astype(int)]
    pair_valid = valid_losses[:, pair]
    pair_selected = valid_losses[np.arange(len(valid_losses)), selected]
    pair_fixed = float(pair_valid.mean(0).min())
    pair_oracle = pair_valid.min(1)
    headroom = pair_fixed - float(pair_oracle.mean())
    recovery = (
        0.0
        if headroom <= 1e-12
        else (pair_fixed - float(pair_selected.mean())) / headroom
    )
    auc = (
        float(roc_auc_score(y_valid, probability[:, list(model_classes).index(1)]))
        if len(np.unique(y_valid)) == 2 and 1 in model_classes
        else 0.5
    )
    return {
        "skipped": False,
        "decision_resolution": decision_resolution,
        "question": "continue versus jump-to-12",
        "jump_route": list(EXPECTED_ROUTES[jump_id]),
        "continue_route": list(EXPECTED_ROUTES[continue_id]),
        "VALID_ROC_AUC": auc,
        "VALID_balanced_accuracy": float(
            balanced_accuracy_score(y_valid, prediction)
        ),
        "VALID_positive_fraction": float(y_valid.mean()),
        "selected_MAE": float(pair_selected.mean()),
        "best_fixed_pair_MAE": pair_fixed,
        "pair_oracle_MAE": float(pair_oracle.mean()),
        "oracle_headroom": float(headroom),
        "oracle_headroom_recovery": float(recovery),
    }


def observability_probes(train_arrays, valid_arrays, seed: int):
    route_probe = multiclass_probe(train_arrays, valid_arrays, "route", seed)
    top_probe = multiclass_probe(train_arrays, valid_arrays, "top_action", seed + 1)
    root_binary = binary_decision_probe(
        train_arrays, valid_arrays, 0, [0], list(range(1, 8)), seed + 2
    )
    at_2 = binary_decision_probe(
        train_arrays, valid_arrays, 2, [1], [2, 3], seed + 3
    )
    at_3 = binary_decision_probe(
        train_arrays, valid_arrays, 3, [4], [5], seed + 4
    )
    return {
        "features": {
            "fit_split": "TRAIN",
            "evaluation_split": "VALID",
            "uses_target_Y": False,
            "decision_resolutions": list(DECISION_RESOLUTIONS),
            "feature_dimension": int(train_arrays["decision_features"].shape[-1]),
        },
        "route_multiclass": route_probe,
        "top_action_multiclass": top_probe,
        "continue_jump_binary": {
            "root": root_binary,
            "after_2": at_2,
            "after_3": at_3,
        },
    }


def _profiler_flops(function, device) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities, record_shapes=True, with_flops=True
    ) as profiler:
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return int(
        sum(
            event.flops
            for event in profiler.key_averages()
            if event.flops is not None
        )
    )


def _latency_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "mean_ms": float(values.mean()),
    }


@torch.inference_mode()
def profile_costs(model, example, warmup: int, repeats: int):
    if example.shape[0] != 1:
        raise ValueError("cost profiling requires real batch_size=1")
    device = example.device
    model.eval()
    all_edges = sorted(set().union(*(route_edges(route) for route in EXPECTED_ROUTES)))

    def make_source_state(source: int):
        state = model.begin_reasoning(example)
        if source:
            state, _ = model.reason_step(example, state, source)
        return state

    edge_states = {(src, dst): make_source_state(src) for src, dst in all_edges}

    def edge_call(edge):
        src, dst = edge
        return model.reason_step(example, edge_states[edge], dst)

    def timed(function):
        for _ in range(warmup):
            function()
        torch.cuda.synchronize(device)
        values = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
        return _latency_summary(values)

    route_flops = {
        route: _profiler_flops(lambda route=route: model.rollout(example, route), device)
        for route in EXPECTED_ROUTES
    }
    edge_flops = {
        edge: _profiler_flops(lambda edge=edge: edge_call(edge), device)
        for edge in all_edges
    }
    encode_flops = _profiler_flops(lambda: model.begin_reasoning(example), device)

    if device.type == "cuda":
        route_latency = {
            route: timed(lambda route=route: model.rollout(example, route))
            for route in EXPECTED_ROUTES
        }
        edge_latency = {
            edge: timed(lambda edge=edge: edge_call(edge)) for edge in all_edges
        }
        encode_latency = timed(lambda: model.begin_reasoning(example))
        flops_vector = np.asarray(
            [route_flops[route] for route in EXPECTED_ROUTES], dtype=np.float64
        )
        latency_vector = np.asarray(
            [route_latency[route]["median_ms"] for route in EXPECTED_ROUTES],
            dtype=np.float64,
        )
        slope, intercept = np.polyfit(flops_vector, latency_vector, 1)
        predicted = slope * flops_vector + intercept
        calibration = {
            "linear_latency_ms_per_FLOP": float(slope),
            "intercept_ms": float(intercept),
            "pearson_r": float(np.corrcoef(flops_vector, latency_vector)[0, 1]),
            "median_absolute_residual_ms": float(
                np.median(np.abs(latency_vector - predicted))
            ),
        }
    else:
        route_latency = edge_latency = {}
        encode_latency = None
        calibration = {"available": False, "reason": "CUDA is required for latency"}

    minimum, maximum = min(route_flops.values()), max(route_flops.values())
    span = max(maximum - minimum, 1)
    normalized = {
        route: (route_flops[route] - minimum) / span for route in EXPECTED_ROUTES
    }
    return {
        "primary_cost": "normalized torch.profiler FLOPs",
        "batch_size": 1,
        "device": str(device),
        "warmup": warmup,
        "repeats": repeats,
        "profiler_note": (
            "torch.profiler counts supported operators; use for within-model "
            "relative cost, with measured CUDA latency as calibration"
        ),
        "history_encode": {
            "flops": encode_flops,
            "latency": encode_latency,
        },
        "routes": {
            route_key(route): {
                "flops": route_flops[route],
                "normalized_flops": normalized[route],
                "latency": route_latency.get(route),
            }
            for route in EXPECTED_ROUTES
        },
        "edges": {
            f"{edge[0]}->{edge[1]}": {
                "flops": edge_flops[edge],
                "latency": edge_latency.get(edge),
            }
            for edge in all_edges
        },
        "latency_calibration": calibration,
    }


def normalized_cost_vector(cost_report):
    return np.asarray(
        [
            cost_report["routes"][route_key(route)]["normalized_flops"]
            for route in EXPECTED_ROUTES
        ],
        dtype=np.float64,
    )


def gate_report(
    valid_analysis,
    probes,
    stability,
    canonical_containment: bool,
):
    headroom = valid_analysis["unconstrained"]["oracle_headroom"]
    route_probe = probes["route_multiclass"]
    top_probe = probes["top_action_multiclass"]
    binary = probes["continue_jump_binary"]
    recoveries = []
    auc_and_bacc = []
    for probe in (route_probe, top_probe):
        if not probe.get("skipped", False):
            recoveries.append(probe["selection"]["oracle_headroom_recovery"])
            auc_and_bacc.append(
                (
                    probe["VALID_macro_ROC_AUC_OVR"],
                    probe["VALID_balanced_accuracy"],
                )
            )
    for probe in binary.values():
        if not probe.get("skipped", False):
            recoveries.append(probe["oracle_headroom_recovery"])
            auc_and_bacc.append(
                (probe["VALID_ROC_AUC"], probe["VALID_balanced_accuracy"])
            )
    best_recovery = max(recoveries, default=0.0)
    observable = any(
        auc >= AUC_GATE and bacc >= BALANCED_ACCURACY_GATE
        for auc, bacc in auc_and_bacc
    )
    gates = {
        "canonical_containment": bool(canonical_containment),
        "VALID_oracle_headroom_at_least_0.03": bool(
            headroom >= ORACLE_HEADROOM_GATE
        ),
        "probe_recovers_at_least_25pct": bool(best_recovery >= RECOVERY_GATE),
        "probe_discrimination_nontrivial": bool(observable),
        "TRAIN_VALID_route_rank_stable": bool(
            stability["route_rank_spearman"] >= 0.5
        ),
    }
    gates["proceed_to_policy_learning"] = all(gates.values())
    return {
        "thresholds": {
            "VALID_oracle_headroom": ORACLE_HEADROOM_GATE,
            "oracle_headroom_recovery": RECOVERY_GATE,
            "ROC_AUC": AUC_GATE,
            "balanced_accuracy": BALANCED_ACCURACY_GATE,
            "route_rank_spearman": 0.5,
        },
        "observed": {
            "VALID_oracle_headroom": float(headroom),
            "best_probe_recovery": float(best_recovery),
            "TRAIN_VALID_route_rank_spearman": stability["route_rank_spearman"],
        },
        "gates": gates,
    }


def synthetic_self_test(seed: int = 7):
    rng = np.random.default_rng(seed)
    n_train, n_valid, feature_dim = 240, 120, 16
    costs = np.linspace(0, 1, 8)

    def make(n):
        x = rng.normal(size=(n, feature_dim))
        route = np.digitize(x[:, 0], [-0.8, -0.3, 0.0, 0.3, 0.8])
        route = np.clip(route, 0, 7)
        losses = 1.0 + 0.03 * rng.normal(size=(n, 8))
        losses[np.arange(n), route] -= 0.12
        decision = np.stack((x, x + 0.01, x - 0.01), axis=1)
        route_features = np.repeat(x[:, None], 8, axis=1)
        return {
            "indices": np.arange(n),
            "mae": losses,
            "decision_features": decision,
            "route_features": route_features,
            "state_features": route_features.copy(),
        }

    train, valid = make(n_train), make(n_valid)
    train_analysis = analyze_split(train, costs, 30, seed)
    valid_analysis = analyze_split(valid, costs, 30, seed + 1)
    probes = observability_probes(train, valid, seed)
    stability = train_valid_stability(
        train["mae"], valid["mae"], train_analysis, valid_analysis
    )
    gates = gate_report(valid_analysis, probes, stability, True)
    assert valid_analysis["unconstrained"]["oracle_headroom"] > 0
    assert len(valid_analysis["hard_budget_frontier"]) == 8
    assert "proceed_to_policy_learning" in gates["gates"]
    print("route-complete oracle/probe synthetic smoke passed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tag", default="selected_v1")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=50)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        synthetic_self_test(args.seed)
        return
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")
    if args.smoke:
        args.batch_size = min(args.batch_size, 2)
        args.workers = 0
        args.bootstrap = min(args.bootstrap, 30)
        args.latency_warmup = 2
        args.latency_repeats = 5
        args.max_batches = 2
        args.tag = "smoke"
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    seed_all(args.seed)
    model_class, _, _ = load_architecture()
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_class(**checkpoint.get("model_args", model_args())).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    result_dir = ROOT / "results" / EXPERIMENT / f"{args.tag}_seed{args.seed}"
    cache_dir = result_dir / "cache"
    result_dir.mkdir(parents=True, exist_ok=True)
    rescale = load_rescale()
    split_arrays = {}
    split_metadata = {}
    for split in ("train", "valid"):
        cache_path = cache_dir / f"{split}_8route_cache.npz"
        if args.reuse_cache and cache_path.is_file():
            arrays, metadata = load_cache(cache_path)
        else:
            # shuffle=False is required so indices and cached rows are stable.
            loader = make_loader(
                split, args.batch_size, False, args.workers
            )
            arrays, metadata = build_split_cache(
                model,
                loader,
                split.upper(),
                device,
                rescale,
                args.max_batches,
            )
            metadata["cost_unit"] = "FLOPs"
            metadata["forecaster_checkpoint"] = str(args.checkpoint.resolve())
            save_cache(cache_path, arrays, metadata)
        split_arrays[split] = arrays
        split_metadata[split] = metadata

    valid_loader = make_loader("valid", 1, False, 0)
    example = select_batch(next(iter(valid_loader)), device)[0][:1]
    costs = profile_costs(
        model, example, args.latency_warmup, args.latency_repeats
    )
    dump_json(result_dir / "cost_profile.json", costs)
    normalized_costs = normalized_cost_vector(costs)

    train_analysis = analyze_split(
        split_arrays["train"], normalized_costs, args.bootstrap, args.seed + 101
    )
    valid_analysis = analyze_split(
        split_arrays["valid"], normalized_costs, args.bootstrap, args.seed + 202
    )
    stability = train_valid_stability(
        split_arrays["train"]["mae"],
        split_arrays["valid"]["mae"],
        train_analysis,
        valid_analysis,
    )
    probes = observability_probes(
        split_arrays["train"], split_arrays["valid"], args.seed
    )
    protocol = checkpoint.get("protocol", {})
    formal_valid = protocol.get("formal_VALID_MAE")
    canonical_index = EXPECTED_ROUTES.index(EXPECTED_CANONICAL_ROUTE)
    selected_canonical = float(
        split_arrays["valid"]["mae"][:, canonical_index].mean()
    )
    containment = (
        formal_valid is not None
        and selected_canonical <= float(formal_valid) + CONTAINMENT_TOLERANCE
    )
    gates = gate_report(valid_analysis, probes, stability, containment)

    report = {
        "method": "route-complete frozen forecaster oracle/cost/observability",
        "selected_checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best": checkpoint.get("best"),
        "routes": [list(route) for route in EXPECTED_ROUTES],
        "cache": split_metadata,
        "cost": costs,
        "TRAIN": train_analysis,
        "VALID": valid_analysis,
        "TRAIN_VALID_stability": stability,
        "observability_probes": probes,
        "canonical_containment": {
            "formal_VALID_MAE": formal_valid,
            "tolerance": CONTAINMENT_TOLERANCE,
            "selected_per_sample_VALID_MAE": selected_canonical,
            "pass": containment,
        },
        "gate": gates,
        "test": None,
        "policy_trained": False,
    }
    dump_json(result_dir / "route_complete_oracle_report.json", report)
    print(
        f"[done] VALID headroom="
        f"{valid_analysis['unconstrained']['oracle_headroom']:.4f} "
        f"recovery={gates['observed']['best_probe_recovery']:.3f} "
        f"proceed={gates['gates']['proceed_to_policy_learning']} "
        f"report={result_dir / 'route_complete_oracle_report.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
