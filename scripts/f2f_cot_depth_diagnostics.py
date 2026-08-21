"""Post-selection TRAIN/VALID diagnostics for the F2FCoT depth study.

No TEST data are loaded here.  Predictive probes are diagnostic classifiers,
not deployed routing controllers, and consume only target-free summaries of the
canonical reasoning trace.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoT_arch import F2FCoTMultiDepthNet
from basicts.metrics import masked_mae
from scripts.f2f_cot_depth_study import (
    CANONICAL_NAME,
    SCHEDULES,
    load_model,
)
from scripts.f2f_cot_runtime import NULL_VAL, load_rescale, make_loader, select_batch


def load_arrays(path: Path):
    packed = np.load(path)
    losses, features = {}, {}
    for key in packed.files:
        if key.startswith("loss__"):
            losses[key.removeprefix("loss__")] = packed[key]
        elif key.startswith("features__"):
            features[key.removeprefix("features__")] = packed[key]
    return {"losses": losses, "features": features}


def help_distribution(short, deep):
    gain = np.asarray(short) - np.asarray(deep)
    positive = gain[gain > 0]
    negative = -gain[gain < 0]
    return {
        "help_fraction": float((gain > 0).mean()),
        "mean_gain_when_helpful": float(positive.mean()) if len(positive) else 0.0,
        "mean_harm_when_harmful": float(negative.mean()) if len(negative) else 0.0,
        "net_gain": float(gain.mean()),
        "gain_std": float(gain.std()),
    }, gain


def split_stability(train_arrays, valid_arrays):
    report = {}
    pairs = []
    for name, route in SCHEDULES.items():
        if len(route) > len(SCHEDULES[CANONICAL_NAME]):
            pairs.append((CANONICAL_NAME, name))
    pairs.extend((("coarse_d2", CANONICAL_NAME), ("direct_d1", "coarse_d2")))
    for short_name, deep_name in pairs:
        train_summary, train_gain = help_distribution(
            train_arrays["losses"][short_name], train_arrays["losses"][deep_name]
        )
        valid_summary, valid_gain = help_distribution(
            valid_arrays["losses"][short_name], valid_arrays["losses"][deep_name]
        )
        ks = ks_2samp(train_gain, valid_gain)
        report[f"{short_name}__vs__{deep_name}"] = {
            "TRAIN": train_summary,
            "VALID": valid_summary,
            "help_fraction_absolute_shift": abs(
                train_summary["help_fraction"] - valid_summary["help_fraction"]
            ),
            "gain_distribution_KS_statistic": float(ks.statistic),
            "gain_distribution_KS_pvalue": float(ks.pvalue),
            "gain_wasserstein_distance": float(
                wasserstein_distance(train_gain, valid_gain)
            ),
        }
    return report


def binary_probe(train_arrays, valid_arrays, deep_name):
    short = CANONICAL_NAME
    x_train = train_arrays["features"][short]
    x_valid = valid_arrays["features"][short]
    y_train = (
        train_arrays["losses"][deep_name] < train_arrays["losses"][short]
    ).astype(np.int64)
    y_valid = (
        valid_arrays["losses"][deep_name] < valid_arrays["losses"][short]
    ).astype(np.int64)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=1,
        ),
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_valid)[:, 1]
    prediction = probability >= 0.5
    return {
        "target": f"does {deep_name} beat {short}?",
        "features": "30 target-free summary features from canonical history/forecast/correction/memory",
        "TRAIN_positive_fraction": float(y_train.mean()),
        "VALID_positive_fraction": float(y_valid.mean()),
        "VALID_ROC_AUC": float(roc_auc_score(y_valid, probability)),
        "VALID_average_precision": float(average_precision_score(y_valid, probability)),
        "VALID_balanced_accuracy_at_0.5": float(
            balanced_accuracy_score(y_valid, prediction)
        ),
        "VALID_log_loss": float(log_loss(y_valid, probability)),
        "deployed_controller": False,
    }


def multiclass_next_program_probe(train_arrays, valid_arrays):
    candidates = [
        CANONICAL_NAME,
        "coupled_d4",
        "refine_d4",
        "dense_d5",
        "refine_d5",
    ]
    x_train = train_arrays["features"][CANONICAL_NAME]
    x_valid = valid_arrays["features"][CANONICAL_NAME]
    y_train = np.stack(
        [train_arrays["losses"][name] for name in candidates], axis=1
    ).argmin(1)
    y_valid = np.stack(
        [valid_arrays["losses"][name] for name in candidates], axis=1
    ).argmin(1)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=1,
        ),
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_valid)
    train_counts = np.bincount(y_train, minlength=len(candidates)) / len(y_train)
    valid_counts = np.bincount(y_valid, minlength=len(candidates)) / len(y_valid)
    return {
        "candidates": candidates,
        "TRAIN_oracle_choice_fraction": {
            name: float(train_counts[index]) for index, name in enumerate(candidates)
        },
        "VALID_oracle_choice_fraction": {
            name: float(valid_counts[index]) for index, name in enumerate(candidates)
        },
        "VALID_balanced_accuracy": float(
            balanced_accuracy_score(y_valid, prediction)
        ),
        "VALID_top1_accuracy": float((prediction == y_valid).mean()),
        "deployed_controller": False,
    }


def rollout_with_context_ablation(model, history, route, mode):
    state = model.begin_reasoning(history)
    for step_index, next_resolution in enumerate(route):
        if step_index:
            if mode in {"latest_only", "no_trace"}:
                state = replace(state, memory=torch.zeros_like(state.memory))
            if mode in {"memory_only", "no_trace"}:
                state = replace(state, latest_forecast=None)
        state, _ = model.reason_step(history, state, next_resolution)
    return state.latest_forecast


@torch.inference_mode()
def context_ablation(model, valid_loader, device, rescale):
    model.eval()
    names = [CANONICAL_NAME, "coupled_d4", "refine_d4", "dense_d5", "refine_d5"]
    modes = ["full", "latest_only", "memory_only", "no_trace"]
    metrics = {name: {mode: [] for mode in modes} for name in names}
    for batch in valid_loader:
        history, target, _ = select_batch(batch, device)
        target_raw = rescale(target)
        for name in names:
            route = SCHEDULES[name]
            for mode in modes:
                prediction = (
                    model.rollout(history, route)["pred"]
                    if mode == "full"
                    else rollout_with_context_ablation(model, history, route, mode)
                )
                metrics[name][mode].append(
                    float(masked_mae(rescale(prediction), target_raw, NULL_VAL))
                )
    report = {}
    for name in names:
        means = {mode: float(np.mean(values)) for mode, values in metrics[name].items()}
        means["delta_without_memory"] = means["latest_only"] - means["full"]
        means["delta_without_latest"] = means["memory_only"] - means["full"]
        means["delta_without_any_trace"] = means["no_trace"] - means["full"]
        report[name] = means
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tag", default="formal_v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = ROOT / "results" / "f2f_cot_depth" / f"{args.tag}_seed{args.seed}"
    checkpoint_path = (
        ROOT
        / "checkpoints"
        / "PEMS04"
        / "H12"
        / "f2f_cot_depth"
        / f"{args.tag}_seed{args.seed}"
        / "multidepth_best.pt"
    )
    train_arrays = load_arrays(result_dir / "selected_train_arrays.npz")
    valid_arrays = load_arrays(result_dir / "selected_valid_arrays.npz")
    probes = {
        name: binary_probe(train_arrays, valid_arrays, name)
        for name in ("coupled_d4", "refine_d4", "dense_d5", "refine_d5")
    }
    probes["multiclass_next_program"] = multiclass_next_program_probe(
        train_arrays, valid_arrays
    )
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(device, checkpoint_path)
    valid_loader = make_loader("valid", args.batch_size, False, args.workers)
    report = {
        "TRAIN_VALID_crossover_stability": split_stability(
            train_arrays, valid_arrays
        ),
        "target_free_benefit_predictability": probes,
        "forecast_context_ablation_VALID": context_ablation(
            model, valid_loader, device, load_rescale()
        ),
        "TEST_loaded": False,
        "controller_implemented": False,
    }
    output = result_dir / "depth_diagnostics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

