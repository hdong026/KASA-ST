"""Shared-prefix continuation-depth study: 3->12 vs 3->6->12.

The scientific object is the value of ONE extra shared-core call after the
same executed Z_3.  Independent rollouts are never used for the comparison.
TEST is not constructed unless --evaluate-test is passed after methodology
decisions are frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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

from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)
from basicts.archs.arch_zoo.F2FCoT_arch.f2f_cot import pool_forecast
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from scripts.f2f_cot_depth_study import (
    dump_json,
    load_model,
    paired_comparison,
    profile_latency,
    seed_all,
)
from scripts.f2f_cot_runtime import (
    NULL_VAL,
    cot_args,
    load_rescale,
    make_loader,
    per_sample_mae,
    select_batch,
)


STAGE3_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "PEMS04"
    / "H12"
    / "f2f_cot_depth"
    / "formal_v1_seed1"
    / "multidepth_best.pt"
)
PROTECTED_COT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "PEMS04"
    / "H12"
    / "f2f_cot"
    / "formal_v1_seed1"
    / "extra_best.pt"
)
PROTECTED_F2FNET_VALID_MAE = 17.9391
PROTECTED_COT_VALID_MAE = 17.945135967753757
STAGE3_CANONICAL_VALID_MAE = 17.9678
CONTAINMENT_TOLERANCE = 0.10
CANONICAL_ROUTE = (3, 6, 12)
SHORT_ROUTE = (3, 12)
CANONICAL_WEIGHTS = (0.2, 0.3, 1.0)
SHORT_FINAL_WEIGHT = 1.0
PAIRED_FRACTION = 0.40

# Stage III same-resolution 12->12 refinement, for qualitative comparison only.
STAGE3_REFINE_12_TO_12 = {
    "mean_abs_forecast_update": 0.5590,
    "forecast_cosine_similarity": 0.999988,
    "correction_target_residual_cosine": -0.1066,
    "mean_projected_MAE_gain": -0.0249,
    "improve_fraction": 0.3397,
}


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


def z3_decision_features(history, output, rescale) -> torch.Tensor:
    """Inference-safe features available at the Z_3 stop/continue decision."""
    z3 = output["prefix"]["forecast"]
    memory = output["prefix"]["state"].memory
    branches = output["prefix"]["diagnostics"]["branches"]
    traffic = history[..., 0]
    last_obs = traffic[:, -1:, :, None]
    last_delta = traffic[:, -1] - traffic[:, -2]
    z3_proj = interpolate_forecast(z3, 12)
    persistence = last_obs.expand_as(z3_proj)
    persistence_gap = z3_proj - persistence
    patch = branches["patch"]
    downsample = branches["downsample"]
    linear = branches["linear"]
    return torch.cat(
        (
            _summaries(traffic),
            _summaries(last_delta),
            _summaries(rescale(z3)),
            _summaries(rescale(z3_proj)),
            _summaries(memory),
            _summaries(rescale(persistence_gap)),
            _summaries((patch - downsample).abs()),
            _summaries((patch - linear).abs()),
            _summaries((downsample - linear).abs()),
            branches["branch_scale"],
        ),
        dim=1,
    )


FEATURE_GROUPS = (
    "history",
    "last_delta",
    "z3",
    "z3_projected",
    "prefix_memory",
    "z3_vs_persistence",
    "patch_vs_downsample",
    "patch_vs_linear",
    "downsample_vs_linear",
    "branch_scale",
)


def canonical_loss(model, history, target, rescale):
    output = model.rollout(history, CANONICAL_ROUTE)
    loss = history.new_zeros(())
    for resolution, prediction, weight in zip(
        CANONICAL_ROUTE, output["forecasts"], CANONICAL_WEIGHTS
    ):
        target_state = pool_forecast(target, int(resolution))
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
        * masked_mae(rescale(z3), rescale(pool_forecast(target, 3)), NULL_VAL)
        + CANONICAL_WEIGHTS[1]
        * masked_mae(rescale(z6), rescale(pool_forecast(target, 6)), NULL_VAL)
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
    gain = previous_mae - current_mae
    return {
        "abs_update": correction.abs().mean((1, 2, 3)),
        "forecast_cosine": _safe_cosine(previous, current),
        "correction_residual_cosine": _safe_cosine(correction, residual),
        "projected_gain": gain,
        "previous_mae": previous_mae,
        "current_mae": current_mae,
    }


@torch.inference_mode()
def evaluate_shared_prefix(
    model, loader, device, rescale, max_batches=None, diagnostics=True
):
    model.eval()
    batch_metrics = {
        "short": {"MAE": [], "RMSE": [], "MAPE": []},
        "long": {"MAE": [], "RMSE": [], "MAPE": []},
        "coarse_reference": {"MAE": [], "RMSE": [], "MAPE": []},
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
        "final_abs_update": [],
        "final_cosine": [],
        "final_alignment": [],
        "prefix_identity": {
            "z3_same_object": 0,
            "z3_is_prefix_object": 0,
            "z3_torch_equal": 0,
            "prefix_memory_same_object": 0,
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
            "prefix_memory_same_object",
        ):
            chunks["prefix_identity"][key] += int(bool(identity[key]))

        pred_short = rescale(output["pred_short"])
        pred_long = rescale(output["pred_long"])
        z3 = rescale(output["prefix"]["forecast"])
        z6 = rescale(output["long"]["forecasts"][1])
        z3_proj = interpolate_forecast(z3, 12)
        z6_proj = interpolate_forecast(z6, 12)
        y3 = rescale(pool_forecast(target, 3))
        y6 = rescale(pool_forecast(target, 6))

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

        if diagnostics:
            coarse = model.rollout(history, (6, 12))
            coarse_pred = rescale(coarse["pred"])
            batch_metrics["coarse_reference"]["MAE"].append(
                float(masked_mae(coarse_pred, target_raw, NULL_VAL))
            )
            batch_metrics["coarse_reference"]["RMSE"].append(
                float(masked_rmse(coarse_pred, target_raw, NULL_VAL))
            )
            batch_metrics["coarse_reference"]["MAPE"].append(
                float(masked_mape(coarse_pred, target_raw, NULL_VAL))
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
        chunks["final_abs_update"].append(final["abs_update"].cpu())
        chunks["final_cosine"].append(final["forecast_cosine"].cpu())
        chunks["final_alignment"].append(final["correction_residual_cosine"].cpu())
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
            "X->Z_3 is executed once; 3->12 and 3->6->12 continue from that "
            "same ForecastReasoningState, so Z_3 and prefix memory are the "
            "same objects rather than separately recomputed prefixes."
        ),
        "z3_same_object_fraction": identity["z3_same_object"] / n_batches,
        "z3_is_prefix_object_fraction": identity["z3_is_prefix_object"] / n_batches,
        "z3_torch_equal_fraction": identity["z3_torch_equal"] / n_batches,
        "prefix_memory_same_object_fraction": (
            identity["prefix_memory_same_object"] / n_batches
        ),
        "short_route": list(SHORT_ROUTE),
        "long_route": list(CANONICAL_ROUTE),
        "extra_reasoning_calls": 1,
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
        "coarse_6_12_reference": {
            metric: float(np.mean(values))
            for metric, values in batch_metrics["coarse_reference"].items()
            if values
        },
        "per_sample_MAE_mean_short": float(arrays["mae_short"].mean()),
        "per_sample_MAE_mean_long": float(arrays["mae_long"].mean()),
    }
    return report, arrays


def extra_call_diagnostics(arrays: dict) -> dict:
    gain = arrays["mae_short"] - arrays["mae_long"]
    helpful = gain > 0
    harmful = gain < 0
    local_gain = arrays["z3_to_z6_projected_gain"]
    return {
        "final_z12_change_from_extra_z6_call": {
            "mean_abs_forecast_update": float(arrays["final_abs_update"].mean()),
            "forecast_cosine_similarity": float(arrays["final_cosine"].mean()),
            "correction_target_residual_cosine": float(
                arrays["final_alignment"].mean()
            ),
            "mean_per_sample_MAE_gain": float(gain.mean()),
            "improve_fraction": float(helpful.mean()),
            "gain_when_helpful": float(gain[helpful].mean()) if helpful.any() else 0.0,
            "harm_when_harmful": float((-gain[harmful]).mean()) if harmful.any() else 0.0,
        },
        "local_z3_to_z6_reasoning_state": {
            "mean_abs_forecast_update": float(arrays["z3_to_z6_abs_update"].mean()),
            "forecast_cosine_similarity": float(arrays["z3_to_z6_cosine"].mean()),
            "correction_target_residual_cosine": float(
                arrays["z3_to_z6_alignment"].mean()
            ),
            "mean_projected_MAE_gain": float(local_gain.mean()),
            "improve_fraction": float((local_gain > 0).mean()),
            "z3_state_MAE": float(arrays["z3_state_mae"].mean()),
            "z6_state_MAE": float(arrays["z6_state_mae"].mean()),
            "z3_projected_MAE": float(arrays["z3_projected_mae"].mean()),
            "z6_projected_MAE": float(arrays["z6_projected_mae"].mean()),
        },
        "stage3_12_to_12_reference": STAGE3_REFINE_12_TO_12,
        "extra_call_vs_12_to_12": {
            "final_cosine_much_less_redundant": float(arrays["final_cosine"].mean())
            < 0.9999,
            "final_alignment_more_corrective": float(arrays["final_alignment"].mean())
            > STAGE3_REFINE_12_TO_12["correction_target_residual_cosine"],
            "helps_more_often": float(helpful.mean())
            > STAGE3_REFINE_12_TO_12["improve_fraction"] + 0.05,
            "net_gain_stronger": float(gain.mean())
            > STAGE3_REFINE_12_TO_12["mean_projected_MAE_gain"] + 0.02,
        },
    }


def crossover_and_oracle(arrays: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    comparison = paired_comparison(arrays["mae_short"], arrays["mae_long"], rng)
    stacked = np.stack([arrays["mae_short"], arrays["mae_long"]], axis=1)
    oracle = stacked.min(axis=1)
    means = {
        "short_3_12": float(arrays["mae_short"].mean()),
        "long_3_6_12": float(arrays["mae_long"].mean()),
    }
    best_fixed_name = min(means, key=means.get)
    choices = stacked.argmin(axis=1)
    return {
        "paired_long_minus_short": comparison,
        "best_fixed": best_fixed_name,
        "best_fixed_per_sample_MAE": means[best_fixed_name],
        "sample_oracle_MAE": float(oracle.mean()),
        "oracle_gain_vs_best_fixed": float(means[best_fixed_name] - oracle.mean()),
        "oracle_choice_fraction": {
            "short_3_12": float((choices == 0).mean()),
            "long_3_6_12": float((choices == 1).mean()),
        },
    }


def split_stability(train_arrays: dict, valid_arrays: dict) -> dict:
    train_gain = train_arrays["mae_short"] - train_arrays["mae_long"]
    valid_gain = valid_arrays["mae_short"] - valid_arrays["mae_long"]
    ks = ks_2samp(train_gain, valid_gain)
    return {
        "TRAIN_help_fraction": float((train_gain > 0).mean()),
        "VALID_help_fraction": float((valid_gain > 0).mean()),
        "help_fraction_absolute_shift": float(
            abs((train_gain > 0).mean() - (valid_gain > 0).mean())
        ),
        "TRAIN_net_gain": float(train_gain.mean()),
        "VALID_net_gain": float(valid_gain.mean()),
        "gain_distribution_KS_statistic": float(ks.statistic),
        "gain_distribution_KS_pvalue": float(ks.pvalue),
        "gain_wasserstein_distance": float(
            wasserstein_distance(train_gain, valid_gain)
        ),
    }


def _selected_mae(mae_short, mae_long, take_long: np.ndarray) -> float:
    return float(np.where(take_long, mae_long, mae_short).mean())


def _reliability(y_true, probability, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (probability >= left) & (probability <= right)
        else:
            mask = (probability >= left) & (probability < right)
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
        constant = float(y_train.mean())
        dummy = {
            "threshold": 0.5,
            "continue_fraction": constant,
            "selected_per_sample_MAE": float(
                valid_arrays["mae_long"].mean()
                if constant >= 0.5
                else valid_arrays["mae_short"].mean()
            ),
            "best_fixed_per_sample_MAE": float(
                min(valid_arrays["mae_short"].mean(), valid_arrays["mae_long"].mean())
            ),
            "oracle_per_sample_MAE": float(
                np.minimum(valid_arrays["mae_short"], valid_arrays["mae_long"]).mean()
            ),
            "oracle_headroom": 0.0,
            "fraction_of_oracle_headroom_recovered": 0.0,
            "balanced_accuracy": 0.5,
            "expected_remaining_calls": 1.0 + constant,
            "expected_total_calls": 2.0 + constant,
        }
        return {
            "question": (
                "Given inference-safe Z_3 context, is the extra Z_6 reasoning "
                "step likely to improve the final Z_12 forecast?"
            ),
            "skipped": True,
            "reason": "TRAIN labels are degenerate; probe requires both classes",
            "TRAIN_positive_fraction": float(y_train.mean()),
            "VALID_positive_fraction": float(y_valid.mean()),
            "VALID_ROC_AUC": 0.5,
            "VALID_average_precision": float(max(y_valid.mean(), 1e-8)),
            "VALID_balanced_accuracy_at_0.5": 0.5,
            "VALID_log_loss": None,
            "VALID_brier": None,
            "VALID_reliability_bins": [],
            "gain_regression": {
                "VALID_spearman_r": 0.0,
                "VALID_spearman_pvalue": 1.0,
                "VALID_r2": 0.0,
            },
            "TRAIN_selected_threshold": 0.5,
            "operating_point_at_0.5_VALID": dummy,
            "operating_point_TRAIN_tuned_VALID": dummy,
            "deployed_controller": False,
        }

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            random_state=1,
        ),
    )
    classifier.fit(x_train, y_train)
    p_train = classifier.predict_proba(x_train)[:, 1]
    p_valid = classifier.predict_proba(x_valid)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 19)
    train_scores = []
    for threshold in thresholds:
        selected = _selected_mae(
            train_arrays["mae_short"], train_arrays["mae_long"], p_train >= threshold
        )
        train_scores.append((selected, float(threshold), float((p_train >= threshold).mean())))
    best_threshold = min(train_scores, key=lambda row: row[0])[1]

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

    auc = float(roc_auc_score(y_valid, p_valid)) if y_valid.min() != y_valid.max() else 0.5
    return {
        "question": (
            "Given inference-safe Z_3 context, is the extra Z_6 reasoning "
            "step likely to improve the final Z_12 forecast?"
        ),
        "features": {
            "groups": list(FEATURE_GROUPS),
            "dimension": int(x_train.shape[1]),
            "uses_target_Y": False,
            "decision_point": "after executed Z_3, before choosing 3->12 or 3->6->12",
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


def controller_gates(valid_report, valid_analysis, valid_diag, probe, stability) -> dict:
    oracle_gain = valid_analysis["oracle_gain_vs_best_fixed"]
    extra = valid_diag["final_z12_change_from_extra_z6_call"]
    vs_refine = valid_diag["extra_call_vs_12_to_12"]
    recovered = probe["operating_point_TRAIN_tuned_VALID"][
        "fraction_of_oracle_headroom_recovered"
    ]
    selected = probe["operating_point_TRAIN_tuned_VALID"]["selected_per_sample_MAE"]
    best_fixed = valid_analysis["best_fixed_per_sample_MAE"]
    headroom_ok = (
        oracle_gain >= 0.03
        and extra["improve_fraction"] >= 0.40
        and extra["improve_fraction"] <= 0.85
        and stability["help_fraction_absolute_shift"] <= 0.12
        and bool(vs_refine["final_cosine_much_less_redundant"] or vs_refine["net_gain_stronger"])
    )
    observable_ok = (
        probe["VALID_ROC_AUC"] >= 0.68
        and recovered >= 0.25
        and selected <= best_fixed + 0.005
    )
    return {
        "A_shared_prefix_oracle_headroom_meaningful": bool(headroom_ok),
        "B_z3_context_predicts_extra_call_value": bool(observable_ok),
        "justify_dynamic_stop_continue": bool(headroom_ok and observable_ok),
        "thresholds": {
            "oracle_gain_min": 0.03,
            "help_fraction_band": [0.40, 0.85],
            "help_shift_max": 0.12,
            "auc_min": 0.68,
            "recovered_headroom_min": 0.25,
        },
        "observed": {
            "oracle_gain": oracle_gain,
            "help_fraction": extra["improve_fraction"],
            "help_shift": stability["help_fraction_absolute_shift"],
            "auc": probe["VALID_ROC_AUC"],
            "recovered_headroom": recovered,
            "selected_minus_best_fixed": selected - best_fixed,
        },
    }


def save_arrays(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            key: value
            for key, value in arrays.items()
            if isinstance(value, np.ndarray)
        },
    )


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
            "model_args": cot_args(),
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


def train(model, train_loader, valid_loader, device, rescale, out_dir, args):
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

    containment_limit = PROTECTED_COT_VALID_MAE + (
        10.0 if args.smoke else CONTAINMENT_TOLERANCE
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        assignments = epoch_assignments(len(train_loader), args.seed, epoch)
        losses = []
        used = Counter()
        start = time.perf_counter()
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
                raise RuntimeError(f"non-finite loss at epoch={epoch} batch={batch_index}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            used[name] += 1
        scheduler.step()

        full_validation = epoch == 1 or epoch % args.valid_every == 0 or epoch == args.epochs
        if full_validation:
            valid, _ = evaluate_shared_prefix(
                model,
                valid_loader,
                device,
                rescale,
                max_batches=args.max_valid_batches,
                diagnostics=False,
            )
        else:
            valid = {"short_3_12": {"MAE": math.nan}, "long_3_6_12": {"MAE": math.nan}}
            model.eval()
            mae_batches = []
            with torch.inference_mode():
                for batch_index, batch in enumerate(valid_loader):
                    if args.max_valid_batches is not None and batch_index >= args.max_valid_batches:
                        break
                    history, target, _ = select_batch(batch, device)
                    prediction = rescale(model.rollout(history, CANONICAL_ROUTE)["pred"])
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
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "full_validation": full_validation,
            "eligible": bool(eligible),
            "selection_score": float(selection_score),
            "valid_long_MAE": float(long_mae),
            "valid_short_MAE": None if math.isnan(short_mae) else float(short_mae),
            "epoch_seconds": time.perf_counter() - start,
        }
        history_rows.append(row)
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best, history_rows, args)
        dump_json(out_dir / "training_history.json", history_rows)
        print(
            f"[shared-prefix] epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"long={long_mae:.4f} short={short_mae if not math.isnan(short_mae) else float('nan'):.4f} "
            f"eligible={eligible} best={best['selection_score']:.4f} "
            f"seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )
    if not best_path.is_file():
        raise RuntimeError("no shared-prefix checkpoint satisfied canonical containment")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return best, history_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    parser.add_argument("--milestones", type=int, nargs="+", default=[20, 30, 36])
    parser.add_argument("--valid-every", type=int, default=2)
    parser.add_argument("--tag", default="formal_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--zero-shot-only", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--eval-test-only",
        action="store_true",
        help="Load the already selected checkpoint and evaluate TEST once.",
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-valid-batches", type=int)
    parser.add_argument(
        "--warm-start",
        choices=["stage3", "protected_cot"],
        default="stage3",
    )
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
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    result_dir = ROOT / "results" / "f2f_cot_shared_prefix" / f"{args.tag}_seed{args.seed}"
    checkpoint_dir = (
        ROOT
        / "checkpoints"
        / "PEMS04"
        / "H12"
        / "f2f_cot_shared_prefix"
        / f"{args.tag}_seed{args.seed}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_test_only:
        selected_path = checkpoint_dir / "shared_prefix_best.pt"
        report_path = result_dir / "shared_prefix_report.json"
        if not selected_path.is_file() or not report_path.is_file():
            raise FileNotFoundError("selected checkpoint/report missing; train first")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("dynamic_controller_implemented"):
            raise RuntimeError("TEST eval must follow a frozen methodology")
        model, _ = load_model(device, selected_path)
        rescale = load_rescale()
        test_loader = make_loader("test", args.batch_size, False, args.workers)
        test, test_arrays = evaluate_shared_prefix(model, test_loader, device, rescale)
        report["test"] = {
            "metrics": test,
            "analysis": crossover_and_oracle(test_arrays, args.seed + 31),
            "diagnostics": extra_call_diagnostics(test_arrays),
            "evaluated_after_methodology_fixed": True,
            "dynamic_controller_implemented": False,
        }
        save_arrays(result_dir / "selected_test_arrays.npz", test_arrays)
        dump_json(report_path, report)
        print(
            f"[test] short={test['short_3_12']['MAE']:.4f} "
            f"long={test['long_3_6_12']['MAE']:.4f} "
            f"oracle_gain={report['test']['analysis']['oracle_gain_vs_best_fixed']:.4f}",
            flush=True,
        )
        print(f"[done] {report_path}", flush=True)
        return

    warm_start = (
        STAGE3_CHECKPOINT if args.warm_start == "stage3" else PROTECTED_COT_CHECKPOINT
    )
    rescale = load_rescale()
    valid_loader = make_loader("valid", args.batch_size, False, args.workers)
    model, warm_ckpt = load_model(device, warm_start)
    breakdown = model.parameter_breakdown()

    zero_shot, zero_arrays = evaluate_shared_prefix(
        model, valid_loader, device, rescale, max_batches=args.max_valid_batches
    )
    zero_analysis = crossover_and_oracle(zero_arrays, args.seed)
    example = select_batch(next(iter(valid_loader)), device)[0][:1]
    zero_latency = profile_latency(
        model,
        example,
        device,
        schedules={"short_3_12": SHORT_ROUTE, "long_3_6_12": CANONICAL_ROUTE},
        repeats=10 if args.smoke else 100,
    )
    zero_report = {
        "checkpoint": str(warm_start),
        "checkpoint_epoch": warm_ckpt.get("epoch"),
        "zero_shot_3_12_not_in_stage3_curriculum": True,
        "valid": zero_shot,
        "analysis": zero_analysis,
        "latency": zero_latency,
        "diagnostics": extra_call_diagnostics(zero_arrays),
    }
    dump_json(result_dir / "zero_shot_valid.json", zero_report)
    save_arrays(result_dir / "zero_shot_valid_arrays.npz", zero_arrays)
    print(
        f"[zero-shot] short={zero_shot['short_3_12']['MAE']:.4f} "
        f"long={zero_shot['long_3_6_12']['MAE']:.4f} "
        f"oracle_gain={zero_analysis['oracle_gain_vs_best_fixed']:.4f}",
        flush=True,
    )
    if args.zero_shot_only:
        return

    train_loader = make_loader("train", args.batch_size, True, args.workers)
    best, _ = train(
        model, train_loader, valid_loader, device, rescale, checkpoint_dir, args
    )
    selected_path = checkpoint_dir / "shared_prefix_best.pt"
    selected_valid, valid_arrays = evaluate_shared_prefix(
        model, valid_loader, device, rescale, max_batches=args.max_valid_batches
    )
    valid_analysis = crossover_and_oracle(valid_arrays, args.seed + 17)
    valid_diag = extra_call_diagnostics(valid_arrays)
    selected_latency = profile_latency(
        model,
        example,
        device,
        schedules={"short_3_12": SHORT_ROUTE, "long_3_6_12": CANONICAL_ROUTE},
        repeats=10 if args.smoke else 100,
    )
    save_arrays(result_dir / "selected_valid_arrays.npz", valid_arrays)

    train_eval_loader = make_loader("train", args.batch_size, False, args.workers)
    selected_train, train_arrays = evaluate_shared_prefix(
        model,
        train_eval_loader,
        device,
        rescale,
        max_batches=args.max_valid_batches if args.smoke else None,
    )
    train_analysis = crossover_and_oracle(train_arrays, args.seed + 23)
    save_arrays(result_dir / "selected_train_arrays.npz", train_arrays)
    stability = split_stability(train_arrays, valid_arrays)
    probe = observability_probe(train_arrays, valid_arrays)
    gates = controller_gates(
        selected_valid, valid_analysis, valid_diag, probe, stability
    )

    report = {
        "method": "F2FCoT shared-prefix continuation depth",
        "starting_checkpoint": str(warm_start),
        "new_training_run": True,
        "same_shared_core": True,
        "new_forecasting_parameters": 0,
        "parameter_breakdown": breakdown,
        "protected_artifacts_untouched": [
            str(PROTECTED_COT_CHECKPOINT),
            str(STAGE3_CHECKPOINT),
        ],
        "shared_prefix_construction": selected_valid["shared_prefix"],
        "training_protocol": {
            "warm_start": str(warm_start),
            "paired_batch_fraction": PAIRED_FRACTION,
            "canonical_only_batch_fraction": 1.0 - PAIRED_FRACTION,
            "canonical_loss_weights": list(CANONICAL_WEIGHTS),
            "short_final_weight": SHORT_FINAL_WEIGHT,
            "paired_batches_share_executed_Z3": True,
            "raw_scale_masked_MAE": True,
            "PEMS04_original_split_and_targets": True,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-4,
            "VALID_only_selection": True,
            "TEST_loaded_during_selection": False,
            "selection_score": "0.5 * long_3_6_12_MAE + 0.5 * short_3_12_MAE",
            "containment": "long 3->6->12 VALID MAE within +0.10 of protected F2FCoT",
        },
        "zero_shot_before_training": zero_report,
        "selected_checkpoint": str(selected_path),
        "best": best,
        "selected_valid": selected_valid,
        "selected_valid_analysis": valid_analysis,
        "selected_valid_diagnostics": valid_diag,
        "selected_train": selected_train,
        "selected_train_analysis": train_analysis,
        "TRAIN_VALID_crossover_stability": stability,
        "latency": selected_latency,
        "observability_probe": probe,
        "controller_gates": gates,
        "dynamic_controller_implemented": False,
        "canonical_containment": {
            "protected_F2FNet_VALID_MAE": PROTECTED_F2FNET_VALID_MAE,
            "protected_shared_F2FCoT_VALID_MAE": PROTECTED_COT_VALID_MAE,
            "stage3_canonical_VALID_MAE": STAGE3_CANONICAL_VALID_MAE,
            "tolerance": CONTAINMENT_TOLERANCE,
            "selected_long_VALID_MAE": selected_valid["long_3_6_12"]["MAE"],
            "pass": selected_valid["long_3_6_12"]["MAE"]
            <= PROTECTED_COT_VALID_MAE + CONTAINMENT_TOLERANCE,
        },
        "test": None,
    }
    dump_json(result_dir / "shared_prefix_report.json", report)
    print(
        f"[selected] short={selected_valid['short_3_12']['MAE']:.4f} "
        f"long={selected_valid['long_3_6_12']['MAE']:.4f} "
        f"oracle_gain={valid_analysis['oracle_gain_vs_best_fixed']:.4f} "
        f"auc={probe['VALID_ROC_AUC']:.3f} "
        f"dynamic={gates['justify_dynamic_stop_continue']}",
        flush=True,
    )
    print(f"[done] {result_dir / 'shared_prefix_report.json'}", flush=True)

    if args.evaluate_test and not args.smoke:
        test_loader = make_loader("test", args.batch_size, False, args.workers)
        test, test_arrays = evaluate_shared_prefix(model, test_loader, device, rescale)
        report["test"] = {
            "metrics": test,
            "analysis": crossover_and_oracle(test_arrays, args.seed + 31),
            "diagnostics": extra_call_diagnostics(test_arrays),
        }
        save_arrays(result_dir / "selected_test_arrays.npz", test_arrays)
        dump_json(result_dir / "shared_prefix_report.json", report)


if __name__ == "__main__":
    main()
