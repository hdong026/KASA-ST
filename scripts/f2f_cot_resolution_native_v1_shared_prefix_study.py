"""Shared-prefix reasoning-depth study for frozen ResolutionNative V1.

Scientific object: after the same executed Z3, is one extra genuine
resolution-native step at r=6 useful before the final Z12 forecast?

    short: X -> Z3 -> Z12
    long:  X -> Z3 -> Z6 -> Z12

The common prefix is computed once and both continuations fork from that
exact state.  Independent forwards that happen to both produce a Z3 are never
used for the comparison.

3->12 is not a trained V1 transition.  Zero-shot 3->12 is recorded only as
documentation.  The scientific comparison uses a small continuation-training
run that exposes the SAME shared reasoner to both routes while protecting
canonical 3->6->12.  The frozen V1 architecture and formal checkpoint are
not modified.  TEST is not used for any methodology decision.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1 import (
    F2FCoTResolutionNativeV1Net,
    repeat_to_resolution,
    temporal_mean_pool,
)
from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1_shared_prefix import (
    F2FCoTResolutionNativeV1SharedPrefixNet,
)
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from scripts.f2f_cot_depth_study import paired_comparison
from scripts.f2f_cot_resolution_native_v1_experiment import model_args
from scripts.f2f_cot_runtime import (
    NULL_VAL,
    load_rescale,
    make_loader,
    per_sample_mae,
    select_batch,
)
from scripts.f2f_cot_shared_prefix_study import (
    controller_gates,
    crossover_and_oracle,
    extra_call_diagnostics,
    save_arrays,
    split_stability,
)


EXPERIMENT = "f2f_cot_resolution_native_v1_shared_prefix"
FORMAL_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "PEMS04"
    / "H12"
    / "f2f_cot_resolution_native_v1_formal"
    / "formal_basicts_v1_seed1"
    / "resolution_native_v1_formal_best_val_MAE.pt"
)
FORMAL_VALID_MAE = 17.865122015231123
CONTAINMENT_TOLERANCE = 0.10
CANONICAL_ROUTE = (3, 6, 12)
SHORT_ROUTE = (3, 12)
CANONICAL_WEIGHTS = (0.2, 0.3, 1.0)
SHORT_FINAL_WEIGHT = 1.0
PAIRED_FRACTION = 0.40
ORACLE_HEADROOM_FOR_PROBE = 0.03

FEATURE_GROUPS = (
    "history",
    "last_delta",
    "z3",
    "z3_projected",
    "z3_vs_persistence",
    "evidence_tokens",
    "prefix_raw_correction",
    "prefix_low_frequency_correction",
    "prefix_detail_correction",
    "prefix_branch_scale",
    "prefix_low_frequency_gain",
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.flatten(1)
    right = right.flatten(1)
    numerator = (left * right).sum(1)
    denominator = left.square().sum(1).sqrt() * right.square().sum(1).sqrt()
    return numerator / denominator.clamp_min(1e-8)


def _summaries(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(1)
    return torch.stack(
        (
            flat.mean(1),
            flat.std(1),
            flat.abs().mean(1),
            flat.square().mean(1).sqrt(),
            flat.amax(1),
            flat.amin(1),
        ),
        dim=1,
    )


def trained_transition_audit() -> dict:
    frozen_legal = {(0, 3), (3, 6), (6, 12)}
    return {
        "frozen_architecture": "F2FCoTResolutionNativeV1Net",
        "formal_checkpoint": str(FORMAL_CHECKPOINT),
        "formal_best_VALID_MAE": FORMAL_VALID_MAE,
        "formal_best_epoch": 93,
        "trained_transitions": sorted(list(frozen_legal)),
        "3_to_12_in_trained_set": (3, 12) in frozen_legal,
        "3_to_12_architecturally_encodable": True,
        "encodability_reason": (
            "Conditioner embeddings exist for {0,3,6,12} and 12 is a nested "
            "refinement of 3, but _validate_transition and rollout never "
            "exposed 3->12 during the 100-epoch formal run."
        ),
        "scientific_status": (
            "3->12 is not a legitimate trained transition. Zero-shot 3->12 "
            "must not be used to conclude whether the extra Z6 step has value."
        ),
        "continuation_training_required": True,
    }


def z3_decision_features(history, output, rescale) -> torch.Tensor:
    """Inference-safe features available immediately after the shared Z3."""
    z3 = output["prefix"]["forecast"]
    diagnostics = output["prefix"]["diagnostics"]
    evidence = output["prefix"]["state"].evidence
    traffic = history[..., 0]
    last_delta = traffic[:, -1] - traffic[:, -2]
    z3_proj = repeat_to_resolution(z3, 12)
    persistence = traffic[:, -1:, :, None].expand_as(z3_proj)
    persistence_gap = z3_proj - persistence
    low_gain = diagnostics["low_frequency_gain"].reshape(history.shape[0], 1)
    return torch.cat(
        (
            _summaries(traffic),
            _summaries(last_delta),
            _summaries(rescale(z3)),
            _summaries(rescale(z3_proj)),
            _summaries(rescale(persistence_gap)),
            _summaries(evidence.tokens),
            _summaries(diagnostics["raw_correction"]),
            _summaries(diagnostics["low_frequency_correction"]),
            _summaries(diagnostics["detail_correction"]),
            diagnostics["branch_scale"],
            low_gain,
        ),
        dim=1,
    )


def canonical_loss(model, history, target, rescale):
    output = model.rollout(history, CANONICAL_ROUTE)
    loss = history.new_zeros(())
    for resolution, prediction, weight in zip(
        CANONICAL_ROUTE, output["forecasts"], CANONICAL_WEIGHTS
    ):
        target_state = temporal_mean_pool(target, int(resolution))
        loss = loss + weight * masked_mae(
            rescale(prediction), rescale(target_state), NULL_VAL
        )
    return loss, output


def shared_prefix_pair_loss(model, history, target, rescale):
    output = model.rollout_shared_prefix_pair(history)
    z3 = output["prefix"]["forecast"]
    z6 = output["long"]["forecasts"][1]
    z12_long = output["pred_long"]
    z12_short = output["pred_short"]
    loss = (
        CANONICAL_WEIGHTS[0]
        * masked_mae(
            rescale(z3), rescale(temporal_mean_pool(target, 3)), NULL_VAL
        )
        + CANONICAL_WEIGHTS[1]
        * masked_mae(
            rescale(z6), rescale(temporal_mean_pool(target, 6)), NULL_VAL
        )
        + CANONICAL_WEIGHTS[2]
        * masked_mae(rescale(z12_long), rescale(target), NULL_VAL)
        + SHORT_FINAL_WEIGHT
        * masked_mae(rescale(z12_short), rescale(target), NULL_VAL)
    )
    return loss, output


def epoch_assignments(num_batches: int, seed: int, epoch: int) -> list[str]:
    paired = int(round(PAIRED_FRACTION * num_batches))
    names = ["paired"] * paired + ["canonical"] * (num_batches - paired)
    rng = random.Random(seed * 100003 + epoch)
    rng.shuffle(names)
    return names


def _transition_stats(previous, current, target_raw):
    correction = current - previous
    residual = target_raw - previous
    previous_mae = per_sample_mae(previous, target_raw)
    current_mae = per_sample_mae(current, target_raw)
    return {
        "abs_update": correction.abs().mean((1, 2, 3)),
        "forecast_cosine": _safe_cosine(previous, current),
        "correction_residual_cosine": _safe_cosine(correction, residual),
        "projected_gain": previous_mae - current_mae,
        "previous_mae": previous_mae,
        "current_mae": current_mae,
    }


@torch.inference_mode()
def evaluate_shared_prefix(
    model, loader, device, rescale, max_batches=None
):
    model.eval()
    batch_metrics = {
        "short": {"MAE": [], "RMSE": [], "MAPE": []},
        "long": {"MAE": [], "RMSE": [], "MAPE": []},
    }
    chunks = {
        "indices": [],
        "mae_short": [],
        "mae_long": [],
        "features": [],
        "z3_state_mae": [],
        "z6_state_mae": [],
        "z3_projected_mae": [],
        "z6_projected_mae": [],
        "z3_to_z6_abs_update": [],
        "z3_to_z6_cosine": [],
        "z3_to_z6_alignment": [],
        "z3_to_z6_projected_gain": [],
        "z6_low_frequency_abs": [],
        "z6_detail_abs": [],
        "final_abs_update": [],
        "final_cosine": [],
        "final_alignment": [],
        "z12_pool3_abs_diff": [],
        "z12_pool6_abs_diff": [],
        "short_parent_coherence": [],
        "long_z6_coherence": [],
        "prefix_identity": {
            "z3_same_object": 0,
            "z3_is_prefix_object": 0,
            "z3_torch_equal": 0,
            "evidence_same_object": 0,
            "latest_forecast_same_object": 0,
            "batches": 0,
        },
    }
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target, sample_index = select_batch(batch, device)
        target_raw = rescale(target)
        output = model.rollout_shared_prefix_pair(history)
        identity = output["shared_prefix"]
        chunks["prefix_identity"]["batches"] += 1
        for key in (
            "z3_same_object",
            "z3_is_prefix_object",
            "z3_torch_equal",
            "evidence_same_object",
            "latest_forecast_same_object",
        ):
            chunks["prefix_identity"][key] += int(bool(identity[key]))

        pred_short = rescale(output["pred_short"])
        pred_long = rescale(output["pred_long"])
        z3 = rescale(output["prefix"]["forecast"])
        z6 = rescale(output["long"]["forecasts"][1])
        z3_proj = repeat_to_resolution(z3, 12)
        z6_proj = repeat_to_resolution(z6, 12)
        y3 = rescale(temporal_mean_pool(target, 3))
        y6 = rescale(temporal_mean_pool(target, 6))
        z6_step = output["long"]["steps"][0]

        batch_metrics["short"]["MAE"].append(
            float(masked_mae(pred_short, target_raw, NULL_VAL))
        )
        batch_metrics["short"]["RMSE"].append(
            float(masked_rmse(pred_short, target_raw, NULL_VAL))
        )
        batch_metrics["short"]["MAPE"].append(
            float(masked_mape(pred_short, target_raw, NULL_VAL))
        )
        batch_metrics["long"]["MAE"].append(
            float(masked_mae(pred_long, target_raw, NULL_VAL))
        )
        batch_metrics["long"]["RMSE"].append(
            float(masked_rmse(pred_long, target_raw, NULL_VAL))
        )
        batch_metrics["long"]["MAPE"].append(
            float(masked_mape(pred_long, target_raw, NULL_VAL))
        )

        local = _transition_stats(z3_proj, z6_proj, target_raw)
        final = _transition_stats(pred_short, pred_long, target_raw)
        chunks["indices"].append(sample_index.cpu())
        chunks["mae_short"].append(per_sample_mae(pred_short, target_raw).cpu())
        chunks["mae_long"].append(per_sample_mae(pred_long, target_raw).cpu())
        chunks["features"].append(z3_decision_features(history, output, rescale).cpu())
        chunks["z3_state_mae"].append(per_sample_mae(z3, y3).cpu())
        chunks["z6_state_mae"].append(per_sample_mae(z6, y6).cpu())
        chunks["z3_projected_mae"].append(per_sample_mae(z3_proj, target_raw).cpu())
        chunks["z6_projected_mae"].append(per_sample_mae(z6_proj, target_raw).cpu())
        chunks["z3_to_z6_abs_update"].append(local["abs_update"].cpu())
        chunks["z3_to_z6_cosine"].append(local["forecast_cosine"].cpu())
        chunks["z3_to_z6_alignment"].append(local["correction_residual_cosine"].cpu())
        chunks["z3_to_z6_projected_gain"].append(local["projected_gain"].cpu())
        chunks["z6_low_frequency_abs"].append(
            z6_step["low_frequency_correction"].abs().mean((1, 2, 3)).cpu()
        )
        chunks["z6_detail_abs"].append(
            z6_step["detail_correction"].abs().mean((1, 2, 3)).cpu()
        )
        chunks["final_abs_update"].append(final["abs_update"].cpu())
        chunks["final_cosine"].append(final["forecast_cosine"].cpu())
        chunks["final_alignment"].append(final["correction_residual_cosine"].cpu())
        chunks["z12_pool3_abs_diff"].append(
            (temporal_mean_pool(pred_long, 3) - temporal_mean_pool(pred_short, 3))
            .abs()
            .mean((1, 2, 3))
            .cpu()
        )
        chunks["z12_pool6_abs_diff"].append(
            (temporal_mean_pool(pred_long, 6) - temporal_mean_pool(pred_short, 6))
            .abs()
            .mean((1, 2, 3))
            .cpu()
        )
        chunks["short_parent_coherence"].append(
            (
                temporal_mean_pool(output["pred_short"], 3)
                - output["short"]["steps"][0]["corrected_parent"]
            )
            .abs()
            .amax((1, 2, 3))
            .cpu()
        )
        chunks["long_z6_coherence"].append(
            (
                temporal_mean_pool(output["pred_long"], 6)
                - output["long"]["steps"][1]["corrected_parent"]
            )
            .abs()
            .amax((1, 2, 3))
            .cpu()
        )
        samples += len(history)

    arrays = {
        key: torch.cat(value).numpy()
        for key, value in chunks.items()
        if key not in {"prefix_identity"}
    }
    identity = chunks["prefix_identity"]
    n_batches = max(identity["batches"], 1)
    identity_report = {
        "batches": identity["batches"],
        "construction": (
            "X->Z3 is executed once; 3->12 and 3->6->12 continue from that "
            "same ResolutionNativeReasoningState, so Z3 and post-Z3 evidence "
            "are the same objects rather than separately recomputed prefixes."
        ),
        "z3_same_object_fraction": identity["z3_same_object"] / n_batches,
        "z3_is_prefix_object_fraction": identity["z3_is_prefix_object"] / n_batches,
        "z3_torch_equal_fraction": identity["z3_torch_equal"] / n_batches,
        "evidence_same_object_fraction": identity["evidence_same_object"] / n_batches,
        "latest_forecast_same_object_fraction": (
            identity["latest_forecast_same_object"] / n_batches
        ),
        "short_route": list(SHORT_ROUTE),
        "long_route": list(CANONICAL_ROUTE),
        "extra_reasoning_calls": 1,
        "verified_common_z3_numerically_identical": (
            identity["z3_torch_equal"] == identity["batches"]
        ),
    }
    report = {
        "samples": samples,
        "shared_prefix": identity_report,
        "short_3_12": {
            metric: float(np.mean(values))
            for metric, values in batch_metrics["short"].items()
        },
        "long_3_6_12": {
            metric: float(np.mean(values))
            for metric, values in batch_metrics["long"].items()
        },
        "per_sample_MAE_mean_short": float(arrays["mae_short"].mean()),
        "per_sample_MAE_mean_long": float(arrays["mae_long"].mean()),
    }
    return report, arrays


def z6_change_diagnostics(arrays: dict) -> dict:
    extra = extra_call_diagnostics(arrays)
    final = extra["final_z12_change_from_extra_z6_call"]
    local = extra["local_z3_to_z6_reasoning_state"]
    detail = arrays["z6_detail_abs"]
    low = arrays["z6_low_frequency_abs"]
    redundant = float(arrays["final_cosine"].mean()) >= 0.9999
    extra["native_z6_detail"] = {
        "mean_abs_low_frequency_correction_at_z6": float(low.mean()),
        "mean_abs_detail_correction_at_z6": float(detail.mean()),
        "detail_over_low_frequency_ratio": float(
            detail.mean() / max(low.mean(), 1e-8)
        ),
        "mean_abs_Z12_difference_pooled_to_r3": float(
            arrays["z12_pool3_abs_diff"].mean()
        ),
        "mean_abs_Z12_difference_pooled_to_r6": float(
            arrays["z12_pool6_abs_diff"].mean()
        ),
        "max_short_3to12_parent_coherence_violation": float(
            arrays["short_parent_coherence"].max()
        ),
        "max_long_6to12_parent_coherence_violation": float(
            arrays["long_z6_coherence"].max()
        ),
        "final_forecast_appears_redundant": redundant,
        "z6_adds_nontrivial_mid_frequency_content": (
            float(detail.mean()) > 0.05
            and float(arrays["z12_pool6_abs_diff"].mean()) > 0.05
            and not redundant
        ),
        "interpretation": (
            "If the extra Z6 step were redundant computation, the two final "
            "Z12 forecasts would be nearly identical (cosine ~ 1) and the "
            "Z6 detail correction would be negligible. Nontrivial pooled "
            "r=6 disagreement plus a detail/low-frequency ratio away from "
            "zero indicates genuine mid-frequency revision after the shared Z3."
        ),
    }
    extra["required_user_items"] = {
        "change_from_direct_continuation": {
            "mean_abs_forecast_update": final["mean_abs_forecast_update"],
            "forecast_cosine_similarity": final["forecast_cosine_similarity"],
        },
        "correction_magnitude": final["mean_abs_forecast_update"],
        "correction_target_residual_alignment": final[
            "correction_target_residual_cosine"
        ],
        "local_z3_to_z6_correction_alignment": local[
            "correction_target_residual_cosine"
        ],
    }
    return extra


def _selected_mae(mae_short, mae_long, take_long: np.ndarray) -> float:
    return float(np.where(take_long, mae_long, mae_short).mean())


def _reliability(y_true, probability, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (
            (probability >= left) & (probability <= right)
            if right == 1.0
            else (probability >= left) & (probability < right)
        )
        if not mask.any():
            continue
        rows.append(
            {
                "bin": [float(left), float(right)],
                "count": int(mask.sum()),
                "mean_predicted": float(probability[mask].mean()),
                "empirical_positive_rate": float(y_true[mask].mean()),
            }
        )
    return rows


def observability_probe(train_arrays: dict, valid_arrays: dict) -> dict:
    x_train = train_arrays["features"]
    x_valid = valid_arrays["features"]
    gain_train = train_arrays["mae_short"] - train_arrays["mae_long"]
    gain_valid = valid_arrays["mae_short"] - valid_arrays["mae_long"]
    y_train = (gain_train > 0).astype(np.int64)
    y_valid = (gain_valid > 0).astype(np.int64)
    if y_train.min() == y_train.max() or len(y_train) < 8:
        return {
            "question": (
                "Given the current Z3 state, can we predict whether spending "
                "the extra Z6 reasoning step will improve the final Z12 forecast "
                "enough to justify its cost?"
            ),
            "skipped": True,
            "reason": "TRAIN labels are degenerate; probe requires both classes",
            "VALID_ROC_AUC": 0.5,
            "VALID_balanced_accuracy_at_0.5": 0.5,
            "operating_point_TRAIN_tuned_VALID": {
                "fraction_of_oracle_headroom_recovered": 0.0,
                "selected_per_sample_MAE": float(valid_arrays["mae_long"].mean()),
            },
            "deployed_controller": False,
        }

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2000, random_state=1
        ),
    )
    classifier.fit(x_train, y_train)
    p_train = classifier.predict_proba(x_train)[:, 1]
    p_valid = classifier.predict_proba(x_valid)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = min(
        (
            _selected_mae(
                train_arrays["mae_short"],
                train_arrays["mae_long"],
                p_train >= threshold,
            ),
            float(threshold),
        )
        for threshold in thresholds
    )[1]

    def operating_point(mae_short, mae_long, probability, threshold, y_true):
        take_long = probability >= threshold
        selected = _selected_mae(mae_short, mae_long, take_long)
        best_fixed = min(float(mae_short.mean()), float(mae_long.mean()))
        oracle = np.minimum(mae_short, mae_long).mean()
        headroom = best_fixed - float(oracle)
        recovered = 0.0 if abs(headroom) < 1e-12 else (best_fixed - selected) / headroom
        return {
            "threshold": float(threshold),
            "continue_fraction": float(take_long.mean()),
            "selected_per_sample_MAE": selected,
            "best_fixed_per_sample_MAE": best_fixed,
            "oracle_per_sample_MAE": float(oracle),
            "oracle_headroom": float(headroom),
            "fraction_of_oracle_headroom_recovered": float(recovered),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, take_long)),
            "expected_remaining_calls": float(1.0 + take_long.mean()),
            "expected_total_calls": float(2.0 + take_long.mean()),
        }

    regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    regressor.fit(x_train, gain_train)
    pred_gain_valid = regressor.predict(x_valid)
    spearman = spearmanr(gain_valid, pred_gain_valid)
    auc = (
        float(roc_auc_score(y_valid, p_valid))
        if y_valid.min() != y_valid.max()
        else 0.5
    )
    return {
        "question": (
            "Given the current Z3 state, can we predict whether spending the "
            "extra Z6 reasoning step will improve the final Z12 forecast "
            "enough to justify its cost?"
        ),
        "skipped": False,
        "features": {
            "groups": list(FEATURE_GROUPS),
            "dimension": int(x_train.shape[1]),
            "uses_target_Y": False,
            "decision_point": "after executed Z3, before choosing 3->12 or 3->6->12",
            "available_information": [
                "X",
                "Z3",
                "current latent/context (evidence tokens, prefix corrections)",
                "resolution code / branch_scale / low_frequency_gain",
                "compute budget (remaining nested hop to 12)",
            ],
        },
        "TRAIN_positive_fraction": float(y_train.mean()),
        "VALID_positive_fraction": float(y_valid.mean()),
        "VALID_ROC_AUC": auc,
        "VALID_average_precision": float(average_precision_score(y_valid, p_valid)),
        "VALID_balanced_accuracy_at_0.5": float(
            balanced_accuracy_score(y_valid, p_valid >= 0.5)
        ),
        "VALID_log_loss": float(log_loss(y_valid, np.clip(p_valid, 1e-6, 1 - 1e-6))),
        "VALID_brier": float(brier_score_loss(y_valid, p_valid)),
        "VALID_reliability_bins": _reliability(y_valid, p_valid),
        "gain_regression": {
            "VALID_spearman_r": float(spearman.statistic),
            "VALID_spearman_pvalue": float(spearman.pvalue),
            "VALID_r2": float(
                1.0
                - np.square(gain_valid - pred_gain_valid).sum()
                / np.square(gain_valid - gain_valid.mean()).sum()
            ),
        },
        "TRAIN_selected_threshold": float(best_threshold),
        "operating_point_at_0.5_VALID": operating_point(
            valid_arrays["mae_short"],
            valid_arrays["mae_long"],
            p_valid,
            0.5,
            y_valid,
        ),
        "operating_point_TRAIN_tuned_VALID": operating_point(
            valid_arrays["mae_short"],
            valid_arrays["mae_long"],
            p_valid,
            best_threshold,
            y_valid,
        ),
        "deployed_controller": False,
    }


def _profile_flops(callable_fn, device) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities, record_shapes=True, with_flops=True
    ) as profiler:
        callable_fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return int(
        sum(
            event.flops
            for event in profiler.key_averages()
            if event.flops is not None
        )
    )


def _summarize_ms(values):
    ordered = sorted(float(value) for value in values)
    return {
        "median_ms": float(statistics.median(ordered)),
        "mean_ms": float(statistics.mean(ordered)),
        "p90_ms": float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]),
    }


@torch.inference_mode()
def profile_shared_prefix_cost(model, example, warmup: int, repeats: int) -> dict:
    device = example.device
    model.eval()
    if device.type != "cuda":
        return {"available": False, "device": str(device)}

    def timed(fn, n):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(n):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return _summarize_ms(samples)

    def run_prefix():
        model.evidence_encoder.reset_encode_count()
        model.reasoner.reset_diagnostics()
        state = model.begin_reasoning(example)
        state, _ = model.reason_step(example, state, 3)
        return state

    encode_ms = timed(lambda: model.begin_reasoning(example), repeats)
    prefix_ms = timed(run_prefix, repeats)
    prefix_state = run_prefix()
    short_ms = timed(
        lambda: model.continue_from(example, prefix_state, (12,)), repeats
    )
    long_ms = timed(
        lambda: model.continue_from(example, prefix_state, (6, 12)), repeats
    )
    step_state = run_prefix()
    step_3_12_ms = timed(
        lambda: model.reason_step(example, step_state, 12), repeats
    )
    step_3_6_ms = timed(lambda: model.reason_step(example, step_state, 6), repeats)
    after_z6, _ = model.reason_step(example, step_state, 6)
    step_6_12_ms = timed(
        lambda: model.reason_step(example, after_z6, 12), repeats
    )

    encode_flops = _profile_flops(
        lambda: model.begin_reasoning(example), device
    )
    start_for_03 = model.begin_reasoning(example)
    flops_0_3 = _profile_flops(
        lambda s=start_for_03: model.reason_step(example, s, 3), device
    )
    start = run_prefix()
    flops_3_12 = _profile_flops(
        lambda s=start: model.reason_step(example, s, 12), device
    )
    flops_3_6 = _profile_flops(
        lambda s=start: model.reason_step(example, s, 6), device
    )
    after_z6, _ = model.reason_step(example, start, 6)
    flops_6_12 = _profile_flops(
        lambda s=after_z6: model.reason_step(example, s, 12), device
    )

    short_total_flops = encode_flops + flops_0_3 + flops_3_12
    long_total_flops = encode_flops + flops_0_3 + flops_3_6 + flops_6_12
    return {
        "available": True,
        "device": str(device),
        "batch_size": int(example.shape[0]),
        "warmup": warmup,
        "repeats": repeats,
        "latency_ms": {
            "history_encode": encode_ms,
            "shared_prefix_X_to_Z3": prefix_ms,
            "short_continuation_3_to_12": short_ms,
            "long_continuation_3_to_6_to_12": long_ms,
            "step_3_to_12": step_3_12_ms,
            "step_3_to_6": step_3_6_ms,
            "step_6_to_12": step_6_12_ms,
            "short_total_prefix_plus_continuation_median_ms": (
                prefix_ms["median_ms"] + short_ms["median_ms"]
            ),
            "long_total_prefix_plus_continuation_median_ms": (
                prefix_ms["median_ms"] + long_ms["median_ms"]
            ),
            "extra_z6_path_median_ms": long_ms["median_ms"] - short_ms["median_ms"],
        },
        "flops": {
            "method": "torch.profiler.with_flops",
            "note": (
                "Counts profiler-supported operators; intended for within-model "
                "relative comparison of the two shared-prefix continuations."
            ),
            "history_encode": encode_flops,
            "0->3": flops_0_3,
            "3->12": flops_3_12,
            "3->6": flops_3_6,
            "6->12": flops_6_12,
            "short_total_X_Z3_Z12": short_total_flops,
            "long_total_X_Z3_Z6_Z12": long_total_flops,
            "extra_z6_path_flops": long_total_flops - short_total_flops,
        },
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, history, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best": best,
            "history": history,
            "model_args": model_args(),
            "method": "F2FCoTResolutionNativeV1SharedPrefixNet",
            "parent_architecture": "F2FCoTResolutionNativeV1Net",
            "new_forecasting_parameters": 0,
            "formal_parent_checkpoint": str(FORMAL_CHECKPOINT),
            "training_protocol": {
                "paired_batch_fraction": PAIRED_FRACTION,
                "canonical_weights": list(CANONICAL_WEIGHTS),
                "short_final_weight": SHORT_FINAL_WEIGHT,
                "learning_rate": args.learning_rate,
                "weight_decay": 1e-4,
                "selection_uses_TEST": False,
                "shared_prefix_training": True,
            },
        },
        path,
    )


def train(
    model, train_loader, valid_loader, device, rescale, out_dir, result_dir, args
):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.5
    )
    last_path = out_dir / "shared_prefix_last.pt"
    best_path = out_dir / "shared_prefix_best.pt"
    best = {
        "selection_score": math.inf,
        "epoch": 0,
        "long_MAE": math.inf,
        "short_MAE": math.inf,
        "eligible": False,
    }
    history_rows = []
    start_epoch = 1
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history_rows = list(checkpoint.get("history", []))
        best = dict(checkpoint["best"])
        start_epoch = int(checkpoint["epoch"]) + 1

    containment_limit = FORMAL_VALID_MAE + (
        10.0 if args.smoke else CONTAINMENT_TOLERANCE
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        assignments = epoch_assignments(len(train_loader), args.seed, epoch)
        losses = []
        used = Counter()
        start = time.perf_counter()
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            history, target, _ = select_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            name = assignments[batch_index]
            if name == "paired":
                loss, _ = shared_prefix_pair_loss(model, history, target, rescale)
            else:
                loss, _ = canonical_loss(model, history, target, rescale)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch={epoch} batch={batch_index}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            used[name] += 1
        scheduler.step()

        full_validation = (
            epoch == 1 or epoch % args.valid_every == 0 or epoch == args.epochs
        )
        if full_validation:
            valid, _ = evaluate_shared_prefix(
                model,
                valid_loader,
                device,
                rescale,
                max_batches=args.max_valid_batches,
            )
        else:
            valid = {"short_3_12": {"MAE": math.nan}, "long_3_6_12": {"MAE": math.nan}}
            model.eval()
            mae_batches = []
            with torch.inference_mode():
                for batch_index, batch in enumerate(valid_loader):
                    if (
                        args.max_valid_batches is not None
                        and batch_index >= args.max_valid_batches
                    ):
                        break
                    history, target, _ = select_batch(batch, device)
                    prediction = rescale(
                        model.rollout(history, CANONICAL_ROUTE)["pred"]
                    )
                    mae_batches.append(
                        float(masked_mae(prediction, rescale(target), NULL_VAL))
                    )
            valid["long_3_6_12"] = {"MAE": float(np.mean(mae_batches))}
            valid["short_3_12"] = {"MAE": math.nan}

        long_mae = valid["long_3_6_12"]["MAE"]
        short_mae = valid["short_3_12"]["MAE"]
        eligible = long_mae <= containment_limit
        selection_score = math.inf
        if full_validation and eligible:
            selection_score = 0.5 * long_mae + 0.5 * short_mae
            if selection_score < best["selection_score"]:
                best = {
                    "selection_score": float(selection_score),
                    "epoch": epoch,
                    "long_MAE": float(long_mae),
                    "short_MAE": float(short_mae),
                    "eligible": True,
                }
                save_checkpoint(
                    best_path, model, optimizer, scheduler, epoch, best, history_rows, args
                )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "route_batches": dict(used),
            "learning_rate_used": learning_rate_used,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "full_validation": full_validation,
            "eligible": bool(eligible),
            "selection_score": float(selection_score),
            "valid_long_MAE": float(long_mae),
            "valid_short_MAE": None if math.isnan(short_mae) else float(short_mae),
            "epoch_seconds": time.perf_counter() - start,
        }
        history_rows.append(row)
        save_checkpoint(
            last_path, model, optimizer, scheduler, epoch, best, history_rows, args
        )
        dump_json(result_dir / "training_history.json", history_rows)
        print(
            f"[rn-shared-prefix] epoch={epoch:03d} "
            f"lr={learning_rate_used:.8f} loss={row['train_loss']:.4f} "
            f"long={long_mae:.4f} "
            f"short={short_mae if not math.isnan(short_mae) else float('nan'):.4f} "
            f"eligible={eligible} best={best['selection_score']:.4f} "
            f"seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )
    if not best_path.is_file():
        raise RuntimeError(
            "no shared-prefix checkpoint satisfied canonical 3->6->12 containment"
        )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return best, history_rows


def split_report(name: str, metrics: dict, arrays: dict, seed: int) -> dict:
    analysis = crossover_and_oracle(arrays, seed)
    diag = z6_change_diagnostics(arrays)
    gain = arrays["mae_short"] - arrays["mae_long"]
    comparison = analysis["paired_long_minus_short"]
    return {
        "split": name,
        "final_MAE_3_12": metrics["short_3_12"]["MAE"],
        "final_MAE_3_6_12": metrics["long_3_6_12"]["MAE"],
        "per_sample_gain_definition": "gain_i = MAE_i(3->12) - MAE_i(3->6->12); positive means extra Z6 helps",
        "fraction_where_extra_Z6_helps": comparison["deeper_better_fraction"],
        "mean_gain_when_helpful": comparison["mean_gain_when_helpful"],
        "mean_harm_when_harmful": comparison["mean_harm_when_harmful"],
        "net_average_gain": comparison["net_per_sample_MAE_gain"],
        "paired_bootstrap_95pct_CI": comparison["paired_bootstrap_95pct_CI"],
        "sample_wise_oracle_MAE": analysis["sample_oracle_MAE"],
        "oracle_gain_over_better_fixed_continuation": analysis[
            "oracle_gain_vs_best_fixed"
        ],
        "best_fixed_continuation": analysis["best_fixed"],
        "shared_prefix_identity": metrics["shared_prefix"],
        "paired_comparison": comparison,
        "crossover_and_oracle": analysis,
        "z6_diagnostics": diag,
        "gain_mean": float(gain.mean()),
        "gain_std": float(gain.std()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    parser.add_argument("--milestones", type=int, nargs="+", default=[10, 15, 18])
    parser.add_argument("--valid-every", type=int, default=1)
    parser.add_argument("--tag", default="formal_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--zero-shot-only", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-valid-batches", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        args.workers = 0
        args.valid_every = 1
        args.max_train_batches = 2
        args.max_valid_batches = 2
        args.tag = "smoke"
        args.milestones = [1]
    seed_all(args.seed)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    result_dir = ROOT / "results" / EXPERIMENT / f"{args.tag}_seed{args.seed}"
    checkpoint_dir = (
        ROOT
        / "checkpoints"
        / "PEMS04"
        / "H12"
        / EXPERIMENT
        / f"{args.tag}_seed{args.seed}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    audit = trained_transition_audit()
    dump_json(result_dir / "transition_audit.json", audit)
    print(f"[audit] 3->12 trained={audit['3_to_12_in_trained_set']}", flush=True)
    if not FORMAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing formal V1 checkpoint: {FORMAL_CHECKPOINT}")

    frozen = F2FCoTResolutionNativeV1Net(**model_args()).to(device)
    wrapper = F2FCoTResolutionNativeV1SharedPrefixNet(**model_args()).to(device)
    if sum(p.numel() for p in frozen.parameters()) != sum(
        p.numel() for p in wrapper.parameters()
    ):
        raise RuntimeError("shared-prefix wrapper must add zero parameters")

    formal = torch.load(FORMAL_CHECKPOINT, map_location="cpu", weights_only=False)
    missing, unexpected = wrapper.load_state_dict(
        formal["model_state_dict"], strict=True
    )
    del missing, unexpected
    frozen_reject = None
    frozen.load_state_dict(formal["model_state_dict"], strict=True)
    example_cpu_check = torch.zeros(1)
    del example_cpu_check
    try:
        dummy = torch.zeros(1, 12, wrapper.node_size, 4, device=device)
        dummy[..., 1] = 0.1
        frozen.rollout(dummy, SHORT_ROUTE)
        frozen_reject = False
    except ValueError as error:
        frozen_reject = str(error)
    audit["frozen_v1_rejects_3_12"] = frozen_reject
    dump_json(result_dir / "transition_audit.json", audit)
    del frozen

    rescale = load_rescale()
    valid_loader = make_loader("valid", args.batch_size, False, args.workers)
    breakdown = wrapper.parameter_breakdown()
    zero_shot, zero_arrays = evaluate_shared_prefix(
        wrapper, valid_loader, device, rescale, max_batches=args.max_valid_batches
    )
    zero_analysis = crossover_and_oracle(zero_arrays, args.seed)
    example = select_batch(next(iter(valid_loader)), device)[0][:1]
    zero_cost = profile_shared_prefix_cost(
        wrapper, example, warmup=3 if args.smoke else 10, repeats=5 if args.smoke else 50
    )
    zero_report = {
        "status": "documentation_only_untrained_transition",
        "do_not_use_for_scientific_conclusion": True,
        "checkpoint": str(FORMAL_CHECKPOINT),
        "checkpoint_epoch": formal.get("epoch"),
        "frozen_v1_rejects_3_12": frozen_reject,
        "valid": zero_shot,
        "analysis": zero_analysis,
        "cost": zero_cost,
        "diagnostics": z6_change_diagnostics(zero_arrays),
    }
    dump_json(result_dir / "zero_shot_valid.json", zero_report)
    save_arrays(result_dir / "zero_shot_valid_arrays.npz", zero_arrays)
    print(
        f"[zero-shot-untrained] short={zero_shot['short_3_12']['MAE']:.4f} "
        f"long={zero_shot['long_3_6_12']['MAE']:.4f} "
        f"(not a scientific comparison)",
        flush=True,
    )
    if args.zero_shot_only:
        return

    train_loader = make_loader("train", args.batch_size, True, args.workers)
    best, history_rows = train(
        wrapper,
        train_loader,
        valid_loader,
        device,
        rescale,
        checkpoint_dir,
        result_dir,
        args,
    )
    dump_json(result_dir / "training_history.json", history_rows)
    selected_path = checkpoint_dir / "shared_prefix_best.pt"
    selected_valid, valid_arrays = evaluate_shared_prefix(
        wrapper, valid_loader, device, rescale, max_batches=args.max_valid_batches
    )
    save_arrays(result_dir / "selected_valid_arrays.npz", valid_arrays)
    selected_cost = profile_shared_prefix_cost(
        wrapper, example, warmup=3 if args.smoke else 10, repeats=5 if args.smoke else 50
    )

    train_eval_loader = make_loader("train", args.batch_size, False, args.workers)
    selected_train, train_arrays = evaluate_shared_prefix(
        wrapper,
        train_eval_loader,
        device,
        rescale,
        max_batches=args.max_valid_batches if args.smoke else None,
    )
    save_arrays(result_dir / "selected_train_arrays.npz", train_arrays)

    train_pack = split_report("TRAIN", selected_train, train_arrays, args.seed + 23)
    valid_pack = split_report("VALID", selected_valid, valid_arrays, args.seed + 17)
    stability = split_stability(train_arrays, valid_arrays)
    oracle_gain = valid_pack["oracle_gain_over_better_fixed_continuation"]
    probe_justified = bool(oracle_gain >= ORACLE_HEADROOM_FOR_PROBE)
    if probe_justified:
        probe = observability_probe(train_arrays, valid_arrays)
        probe["ran_because_oracle_headroom_clearly_meaningful"] = True
        probe["oracle_headroom_threshold"] = ORACLE_HEADROOM_FOR_PROBE
        gates = controller_gates(
            selected_valid,
            valid_pack["crossover_and_oracle"],
            valid_pack["z6_diagnostics"],
            probe,
            stability,
        )
    else:
        probe = {
            "skipped": True,
            "reason": (
                "Shared-prefix oracle headroom on VALID is not clearly "
                f"meaningful ({oracle_gain:.4f} < {ORACLE_HEADROOM_FOR_PROBE}). "
                "A target-free observability probe would not be justified."
            ),
            "VALID_ROC_AUC": None,
            "VALID_balanced_accuracy_at_0.5": None,
            "operating_point_TRAIN_tuned_VALID": {
                "fraction_of_oracle_headroom_recovered": 0.0,
                "selected_per_sample_MAE": valid_pack["final_MAE_3_6_12"]
                if valid_pack["best_fixed_continuation"] == "long_3_6_12"
                else valid_pack["final_MAE_3_12"],
            },
            "deployed_controller": False,
        }
        gates = {
            "A_shared_prefix_oracle_headroom_meaningful": False,
            "B_z3_context_predicts_extra_call_value": False,
            "justify_dynamic_stop_continue": False,
            "observed": {"oracle_gain": oracle_gain},
        }

    report = {
        "method": "ResolutionNative V1 shared-prefix reasoning depth",
        "architecture_frozen": True,
        "backbone_file_untouched": str(
            ROOT
            / "basicts"
            / "archs"
            / "arch_zoo"
            / "F2FCoTResolutionNative_arch"
            / "f2f_cot_resolution_native_v1.py"
        ),
        "formal_checkpoint_untouched": str(FORMAL_CHECKPOINT),
        "new_forecasting_parameters": 0,
        "parameter_breakdown": breakdown,
        "trained_transition_audit": audit,
        "shared_prefix_construction": selected_valid["shared_prefix"],
        "training_protocol": {
            "warm_start": str(FORMAL_CHECKPOINT),
            "weights_only_new_optimizer": True,
            "paired_batch_fraction": PAIRED_FRACTION,
            "canonical_only_batch_fraction": 1.0 - PAIRED_FRACTION,
            "canonical_loss_weights": list(CANONICAL_WEIGHTS),
            "short_final_weight": SHORT_FINAL_WEIGHT,
            "paired_batches_share_executed_Z3": True,
            "raw_scale_masked_MAE": True,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-4,
            "epochs": args.epochs,
            "milestones": args.milestones,
            "gradient_clip_norm": 5.0,
            "VALID_only_selection": True,
            "TEST_used_for_methodology": False,
            "selection_score": "0.5 * long_3_6_12_MAE + 0.5 * short_3_12_MAE",
            "containment": (
                f"long 3->6->12 VALID MAE within +{CONTAINMENT_TOLERANCE} of "
                f"formal V1 {FORMAL_VALID_MAE}"
            ),
        },
        "zero_shot_before_training_documentation_only": zero_report,
        "selected_checkpoint": str(selected_path),
        "best": best,
        "cost": selected_cost,
        "TRAIN": train_pack,
        "VALID": valid_pack,
        "TRAIN_VALID_gain_stability": stability,
        "observability_probe": probe,
        "controller_gates": gates,
        "dynamic_controller_implemented": False,
        "canonical_containment": {
            "formal_V1_VALID_MAE": FORMAL_VALID_MAE,
            "tolerance": CONTAINMENT_TOLERANCE,
            "selected_long_VALID_MAE": selected_valid["long_3_6_12"]["MAE"],
            "pass": selected_valid["long_3_6_12"]["MAE"]
            <= FORMAL_VALID_MAE + CONTAINMENT_TOLERANCE,
        },
        "test": None,
    }
    dump_json(result_dir / "shared_prefix_report.json", report)
    print(
        f"[selected] short={selected_valid['short_3_12']['MAE']:.4f} "
        f"long={selected_valid['long_3_6_12']['MAE']:.4f} "
        f"oracle_gain={oracle_gain:.4f} probe={probe_justified} "
        f"dynamic={gates['justify_dynamic_stop_continue']}",
        flush=True,
    )
    print(f"[done] {result_dir / 'shared_prefix_report.json'}", flush=True)


if __name__ == "__main__":
    main()
