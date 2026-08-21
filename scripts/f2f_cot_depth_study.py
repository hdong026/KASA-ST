"""Stage III study of computation depth in the shared F2FCoT reasoner.

The script preserves the PEMS04 data, raw-scale masked-MAE objective, explicit
resolution-matched supervision, optimizer family, and validation-only model
selection used by the successful F2FCoT containment run.  It never writes to
that run's checkpoint directory.
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
from typing import Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)
from basicts.archs.arch_zoo.F2FCoT_arch import F2FCoTMultiDepthNet
from basicts.archs.arch_zoo.F2FCoT_arch.f2f_cot import ForecastReasoningState, pool_forecast
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from scripts.f2f_cot_runtime import (
    NULL_VAL,
    cot_args,
    load_rescale,
    make_loader,
    per_sample_mae,
    select_batch,
)


PROTECTED_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "PEMS04"
    / "H12"
    / "f2f_cot"
    / "formal_v1_seed1"
    / "extra_best.pt"
)
PROTECTED_CANONICAL_VALID_MAE = 17.945135967753757
CONTAINMENT_TOLERANCE = 0.10

# One representative per call count, plus organization controls at depths 4/5.
SCHEDULES = {
    "direct_d1": (12,),
    "coarse_d2": (6, 12),
    "canonical_d3": (3, 6, 12),
    "coupled_d4": (3, 4, 6, 12),
    "refine_d4": (3, 6, 12, 12),
    "dense_d5": (2, 3, 4, 6, 12),
    "refine_d5": (3, 6, 12, 12, 12),
}
CANONICAL_NAME = "canonical_d3"
CANONICAL_ROUTE = SCHEDULES[CANONICAL_NAME]
TRAIN_ALTERNATIVES = tuple(name for name in SCHEDULES if name != CANONICAL_NAME)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def step_label(index: int, resolution: int, route: Sequence[int]) -> str:
    occurrence = sum(value == resolution for value in route[: index + 1])
    total = sum(value == resolution for value in route)
    return f"s{index + 1}:r{resolution}" + (f"^{occurrence}" if total > 1 else "")


def route_loss_weights(route: Sequence[int]) -> tuple[float, ...]:
    """Keep canonical supervision exact; fix auxiliary mass at 0.5 elsewhere."""
    route = tuple(route)
    if route == CANONICAL_ROUTE:
        return (0.2, 0.3, 1.0)
    intermediate = len(route) - 1
    if intermediate == 0:
        return (1.0,)
    denominator = intermediate * (intermediate + 1) / 2.0
    auxiliary = tuple(0.5 * index / denominator for index in range(1, intermediate + 1))
    return (*auxiliary, 1.0)


def multidepth_loss(model, history, target, route, rescale):
    output = model.rollout(history, route)
    loss = history.new_zeros(())
    for resolution, prediction, weight in zip(
        route, output["forecasts"], route_loss_weights(route)
    ):
        target_state = pool_forecast(target, int(resolution))
        loss = loss + weight * masked_mae(
            rescale(prediction), rescale(target_state), NULL_VAL
        )
    return loss, output


def balanced_epoch_schedule(num_batches: int, seed: int, epoch: int) -> list[str]:
    """60% canonical batches; remaining exposure is balanced across depths."""
    canonical_count = int(round(0.60 * num_batches))
    alternative_count = num_batches - canonical_count
    names = [CANONICAL_NAME] * canonical_count
    names.extend(
        TRAIN_ALTERNATIVES[index % len(TRAIN_ALTERNATIVES)]
        for index in range(alternative_count)
    )
    rng = random.Random(seed * 100003 + epoch)
    rng.shuffle(names)
    return names


def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.flatten(1)
    right = right.flatten(1)
    numerator = (left * right).sum(1)
    denominator = left.square().sum(1).sqrt() * right.square().sum(1).sqrt()
    return numerator / denominator.clamp_min(1e-8)


def _observable_features(history, output, rescale) -> torch.Tensor:
    """Target-free summary used only to diagnose benefit predictability."""
    forecasts = [rescale(interpolate_forecast(z, 12)) for z in output["forecasts"]]
    current = forecasts[-1]
    previous = forecasts[-2] if len(forecasts) > 1 else current
    correction = current - previous
    memory = output["state"].memory
    traffic = history[..., 0]
    last_delta = traffic[:, -1] - traffic[:, -2]

    def summaries(value):
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

    return torch.cat(
        (
            summaries(current),
            summaries(correction),
            summaries(memory),
            summaries(traffic),
            summaries(last_delta),
        ),
        dim=1,
    )


@torch.inference_mode()
def evaluate_schedules(
    model,
    loader,
    device,
    rescale,
    schedules=SCHEDULES,
    max_batches=None,
    return_arrays=False,
):
    """Step-indexed evaluation supporting repeated explicit resolutions."""
    model.eval()
    metric_batches = {
        name: {"MAE": [], "RMSE": [], "MAPE": []} for name in schedules
    }
    sample_losses = {name: [] for name in schedules}
    features = {name: [] for name in schedules}
    trace = {
        name: {
            "state_mae": [[] for _ in route],
            "projected_mae": [[] for _ in route],
            "update_abs": [[] for _ in route[1:]],
            "forecast_cosine": [[] for _ in route[1:]],
            "correction_residual_cosine": [[] for _ in route[1:]],
            "projected_gain": [[] for _ in route[1:]],
        }
        for name, route in schedules.items()
    }
    indices = []
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target, sample_index = select_batch(batch, device)
        target_raw = rescale(target)
        indices.append(sample_index.cpu())
        for name, route in schedules.items():
            output = model.rollout(history, route)
            prediction_raw = rescale(output["pred"])
            metric_batches[name]["MAE"].append(
                float(masked_mae(prediction_raw, target_raw, NULL_VAL))
            )
            metric_batches[name]["RMSE"].append(
                float(masked_rmse(prediction_raw, target_raw, NULL_VAL))
            )
            metric_batches[name]["MAPE"].append(
                float(masked_mape(prediction_raw, target_raw, NULL_VAL))
            )
            sample_losses[name].append(per_sample_mae(prediction_raw, target_raw).cpu())
            features[name].append(_observable_features(history, output, rescale).cpu())

            projected = []
            for step_index, (resolution, forecast) in enumerate(
                zip(route, output["forecasts"])
            ):
                target_state = pool_forecast(target, resolution)
                own = per_sample_mae(rescale(forecast), rescale(target_state)).cpu()
                trace[name]["state_mae"][step_index].append(own)
                canvas = rescale(interpolate_forecast(forecast, 12))
                projected.append(canvas)
                projected_loss = per_sample_mae(canvas, target_raw).cpu()
                trace[name]["projected_mae"][step_index].append(projected_loss)
            for step_index in range(1, len(projected)):
                previous, current = projected[step_index - 1], projected[step_index]
                correction = current - previous
                residual = target_raw - previous
                trace[name]["update_abs"][step_index - 1].append(
                    correction.abs().mean((1, 2, 3)).cpu()
                )
                trace[name]["forecast_cosine"][step_index - 1].append(
                    _safe_cosine(previous, current).cpu()
                )
                trace[name]["correction_residual_cosine"][step_index - 1].append(
                    _safe_cosine(correction, residual).cpu()
                )
                before = trace[name]["projected_mae"][step_index - 1][-1]
                after = trace[name]["projected_mae"][step_index][-1]
                trace[name]["projected_gain"][step_index - 1].append(before - after)
        samples += len(history)

    arrays = {
        "indices": torch.cat(indices).numpy(),
        "losses": {name: torch.cat(chunks).numpy() for name, chunks in sample_losses.items()},
        "features": {name: torch.cat(chunks).numpy() for name, chunks in features.items()},
    }
    report = {"samples": samples, "schedules": {}}
    for name, route in schedules.items():
        route_report = {
            metric: float(np.mean(values))
            for metric, values in metric_batches[name].items()
        }
        route_report.update(
            {
                "trajectory": list(route),
                "reasoning_calls": len(route),
                "loss_weights": list(route_loss_weights(route)),
                "per_sample_MAE_mean": float(arrays["losses"][name].mean()),
                "trace": [],
            }
        )
        for step_index, resolution in enumerate(route):
            step = {
                "step": step_index + 1,
                "state": step_label(step_index, resolution, route),
                "resolution": resolution,
                "state_target_MAE": float(
                    torch.cat(trace[name]["state_mae"][step_index]).mean()
                ),
                "projected_full_resolution_MAE": float(
                    torch.cat(trace[name]["projected_mae"][step_index]).mean()
                ),
            }
            if step_index:
                update = torch.cat(trace[name]["update_abs"][step_index - 1])
                similarity = torch.cat(trace[name]["forecast_cosine"][step_index - 1])
                alignment = torch.cat(
                    trace[name]["correction_residual_cosine"][step_index - 1]
                )
                gain = torch.cat(trace[name]["projected_gain"][step_index - 1])
                step["from_previous"] = {
                    "mean_abs_forecast_update": float(update.mean()),
                    "forecast_cosine_similarity": float(similarity.mean()),
                    "correction_target_residual_cosine": float(alignment.mean()),
                    "mean_projected_MAE_gain": float(gain.mean()),
                    "improve_fraction": float((gain > 0).float().mean()),
                    "gain_when_helpful": float(gain[gain > 0].mean())
                    if bool((gain > 0).any())
                    else 0.0,
                    "harm_when_harmful": float((-gain[gain < 0]).mean())
                    if bool((gain < 0).any())
                    else 0.0,
                }
            route_report["trace"].append(step)
        report["schedules"][name] = route_report
    return (report, arrays) if return_arrays else report


def paired_comparison(short, deep, rng, bootstrap_samples=2000):
    gain = np.asarray(short) - np.asarray(deep)
    helpful = gain > 0
    harmful = gain < 0
    n = len(gain)
    means = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        means[index] = gain[rng.integers(0, n, size=n)].mean()
    return {
        "deeper_better_fraction": float(helpful.mean()),
        "mean_gain_when_helpful": float(gain[helpful].mean()) if helpful.any() else 0.0,
        "mean_harm_when_harmful": float((-gain[harmful]).mean()) if harmful.any() else 0.0,
        "net_per_sample_MAE_gain": float(gain.mean()),
        "median_gain": float(np.median(gain)),
        "paired_bootstrap_95pct_CI": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def analyze_frontier(arrays, seed: int):
    losses = arrays["losses"]
    rng = np.random.default_rng(seed)
    names = list(SCHEDULES)
    comparisons = {}
    for short in names:
        for deep in names:
            if len(SCHEDULES[deep]) > len(SCHEDULES[short]):
                key = f"{short}__vs__{deep}"
                comparisons[key] = paired_comparison(
                    losses[short], losses[deep], rng
                )

    oracle = {}
    for budget in sorted({len(route) for route in SCHEDULES.values()}):
        eligible = [name for name, route in SCHEDULES.items() if len(route) <= budget]
        stacked = np.stack([losses[name] for name in eligible], axis=1)
        means = {name: float(losses[name].mean()) for name in eligible}
        best_name = min(means, key=means.get)
        oracle_losses = stacked.min(axis=1)
        choices = stacked.argmin(axis=1)
        oracle[str(budget)] = {
            "eligible": eligible,
            "best_fixed": best_name,
            "best_fixed_per_sample_MAE": means[best_name],
            "sample_oracle_MAE": float(oracle_losses.mean()),
            "oracle_gain_vs_best_fixed": float(means[best_name] - oracle_losses.mean()),
            "oracle_choice_fraction": {
                name: float((choices == index).mean())
                for index, name in enumerate(eligible)
            },
        }
    return {"paired_depth_comparisons": comparisons, "oracle_by_call_budget": oracle}


@torch.inference_mode()
def profile_latency(model, example, device, schedules=SCHEDULES, warmup=10, repeats=100):
    if device.type != "cuda":
        return {"available": False, "device": str(device)}
    model.eval()
    report = {}
    for name, route in schedules.items():
        for _ in range(warmup):
            model.rollout(example, route)
        torch.cuda.synchronize(device)
        values = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model.rollout(example, route)
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
        ordered = sorted(values)
        report[name] = {
            "calls": len(route),
            "trajectory": list(route),
            "median_ms": float(statistics.median(values)),
            "mean_ms": float(statistics.mean(values)),
            "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "median_ms_per_call": float(statistics.median(values) / len(route)),
        }
    return {
        "available": True,
        "device": str(device),
        "batch_size": int(example.shape[0]),
        "warmup": warmup,
        "repeats": repeats,
        "schedules": report,
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
            "model_args": cot_args(),
            "schedules": {name: list(route) for name, route in SCHEDULES.items()},
            "training_protocol": {
                "canonical_batch_fraction": 0.60,
                "canonical_weights": [0.2, 0.3, 1.0],
                "alternative_auxiliary_weight_mass": 0.5,
                "learning_rate": args.learning_rate,
                "weight_decay": 1e-4,
                "selection_uses_TEST": False,
            },
        },
        path,
    )


def train(
    model,
    train_loader,
    valid_loader,
    device,
    rescale,
    out_dir,
    args,
    containment_reference_mae,
):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 35, 45], gamma=0.5
    )
    last_path = out_dir / "multidepth_last.pt"
    best_path = out_dir / "multidepth_best.pt"
    best = {
        "selection_score": math.inf,
        "epoch": 0,
        "canonical_MAE": math.inf,
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

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        assignments = balanced_epoch_schedule(len(train_loader), args.seed, epoch)
        losses = []
        start = time.perf_counter()
        used = Counter()
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            name = assignments[batch_index]
            route = SCHEDULES[name]
            history, target, _ = select_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = multidepth_loss(model, history, target, route, rescale)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch} batch={batch_index}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            used[name] += 1
        scheduler.step()

        full_validation = epoch == 1 or epoch % args.valid_every == 0 or epoch == args.epochs
        validation_schedules = SCHEDULES if full_validation else {
            CANONICAL_NAME: CANONICAL_ROUTE
        }
        valid = evaluate_schedules(
            model,
            valid_loader,
            device,
            rescale,
            schedules=validation_schedules,
            max_batches=args.max_valid_batches,
        )
        canonical_mae = valid["schedules"][CANONICAL_NAME]["MAE"]
        eligible = canonical_mae <= containment_reference_mae + CONTAINMENT_TOLERANCE
        selection_score = math.inf
        if full_validation:
            alternative_mean = float(
                np.mean(
                    [
                        valid["schedules"][name]["MAE"]
                        for name in SCHEDULES
                        if name != CANONICAL_NAME
                    ]
                )
            )
            selection_score = 0.5 * canonical_mae + 0.5 * alternative_mean
            if eligible and selection_score < best["selection_score"]:
                best = {
                    "selection_score": float(selection_score),
                    "epoch": epoch,
                    "canonical_MAE": float(canonical_mae),
                    "alternative_mean_MAE": alternative_mean,
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
            "valid": valid,
            "epoch_seconds": time.perf_counter() - start,
        }
        history_rows.append(row)
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best, history_rows, args)
        dump_json(out_dir / "training_history.json", history_rows)
        route_text = " ".join(
            f"{name}={metrics['MAE']:.4f}"
            for name, metrics in valid["schedules"].items()
        )
        print(
            f"[depth] epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"{route_text} eligible={eligible} best={best['selection_score']:.4f} "
            f"seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )
    if not best_path.is_file():
        raise RuntimeError("no multi-depth checkpoint satisfied canonical containment")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return best, history_rows


def load_model(device, checkpoint_path: Path):
    model = F2FCoTMultiDepthNet(**cot_args()).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def save_arrays(path: Path, arrays: dict) -> None:
    values = {"sample_indices": arrays["indices"]}
    for name, array in arrays["losses"].items():
        values[f"loss__{name}"] = array
    for name, array in arrays["features"].items():
        values[f"features__{name}"] = array
    np.savez_compressed(path, **values)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.000125)
    parser.add_argument("--valid-every", type=int, default=2)
    parser.add_argument("--tag", default="formal_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
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
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    result_dir = ROOT / "results" / "f2f_cot_depth" / f"{args.tag}_seed{args.seed}"
    checkpoint_dir = (
        ROOT
        / "checkpoints"
        / "PEMS04"
        / "H12"
        / "f2f_cot_depth"
        / f"{args.tag}_seed{args.seed}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rescale = load_rescale()
    valid_loader = make_loader("valid", args.batch_size, False, args.workers)

    # III-A.1: characterize the protected checkpoint before any new exposure.
    model, protected = load_model(device, PROTECTED_CHECKPOINT)
    baseline_valid, baseline_arrays = evaluate_schedules(
        model,
        valid_loader,
        device,
        rescale,
        max_batches=args.max_valid_batches,
        return_arrays=True,
    )
    baseline_analysis = analyze_frontier(baseline_arrays, args.seed)
    example = select_batch(next(iter(valid_loader)), device)[0][:1]
    baseline_latency = profile_latency(
        model, example, device, repeats=10 if args.smoke else 100
    )
    baseline_report = {
        "checkpoint": str(PROTECTED_CHECKPOINT),
        "checkpoint_epoch": protected.get("epoch"),
        "zero_exposure": True,
        "valid": baseline_valid,
        "analysis": baseline_analysis,
        "latency": baseline_latency,
    }
    dump_json(result_dir / "zero_exposure_valid.json", baseline_report)
    save_arrays(result_dir / "zero_exposure_valid_arrays.npz", baseline_arrays)
    print(
        "[zero-exposure] "
        + " ".join(
            f"{name}={values['MAE']:.4f}"
            for name, values in baseline_valid["schedules"].items()
        ),
        flush=True,
    )
    if args.baseline_only:
        return

    train_loader = make_loader("train", args.batch_size, True, args.workers)
    best, training_history = train(
        model,
        train_loader,
        valid_loader,
        device,
        rescale,
        checkpoint_dir,
        args,
        baseline_valid["schedules"][CANONICAL_NAME]["MAE"]
        + (10.0 if args.smoke else 0.0),
    )
    selected_checkpoint = checkpoint_dir / "multidepth_best.pt"
    selected_valid, selected_valid_arrays = evaluate_schedules(
        model,
        valid_loader,
        device,
        rescale,
        max_batches=args.max_valid_batches,
        return_arrays=True,
    )
    selected_analysis = analyze_frontier(selected_valid_arrays, args.seed + 17)
    selected_latency = profile_latency(
        model, example, device, repeats=10 if args.smoke else 100
    )
    save_arrays(result_dir / "selected_valid_arrays.npz", selected_valid_arrays)

    # TRAIN labels are diagnostic only: stability/predictability never select weights.
    train_eval_loader = make_loader("train", args.batch_size, False, args.workers)
    train_eval, train_arrays = evaluate_schedules(
        model,
        train_eval_loader,
        device,
        rescale,
        max_batches=args.max_valid_batches if args.smoke else None,
        return_arrays=True,
    )
    train_analysis = analyze_frontier(train_arrays, args.seed + 23)
    save_arrays(result_dir / "selected_train_arrays.npz", train_arrays)

    report = {
        "method": "F2FCoTMultiDepthNet",
        "same_shared_core": True,
        "new_forecasting_parameters": 0,
        "protected_checkpoint_untouched": str(PROTECTED_CHECKPOINT),
        "schedule_rationale": {
            "one_representative_per_call_count": True,
            "organization_controls": ["coupled_d4 vs refine_d4", "dense_d5 vs refine_d5"],
            "resolution_depth_partially_decoupled": True,
            "same_resolution_states_are_explicit": True,
        },
        "training_protocol": {
            "warm_start": str(PROTECTED_CHECKPOINT),
            "canonical_batch_fraction": 0.60,
            "canonical_loss_weights": [0.2, 0.3, 1.0],
            "alternative_auxiliary_weight_mass": 0.5,
            "raw_scale_masked_MAE": True,
            "PEMS04_original_split_and_targets": True,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-4,
            "VALID_only_selection": True,
            "TEST_loaded_during_selection": False,
        },
        "zero_exposure": baseline_report,
        "selected_checkpoint": str(selected_checkpoint),
        "best": best,
        "selected_valid": selected_valid,
        "selected_valid_analysis": selected_analysis,
        "selected_train": train_eval,
        "selected_train_analysis": train_analysis,
        "latency": selected_latency,
        "canonical_containment": {
            "protected_shared_VALID_MAE": PROTECTED_CANONICAL_VALID_MAE,
            "tolerance": CONTAINMENT_TOLERANCE,
            "selected_VALID_MAE": selected_valid["schedules"][CANONICAL_NAME]["MAE"],
            "pass": selected_valid["schedules"][CANONICAL_NAME]["MAE"]
            <= PROTECTED_CANONICAL_VALID_MAE + CONTAINMENT_TOLERANCE,
        },
        "dynamic_controller_implemented": False,
        "test": None,
    }

    if args.evaluate_test and not args.smoke:
        test_loader = make_loader("test", args.batch_size, False, args.workers)
        test, test_arrays = evaluate_schedules(
            model, test_loader, device, rescale, return_arrays=True
        )
        report["test"] = {
            "metrics": test,
            "analysis": analyze_frontier(test_arrays, args.seed + 31),
        }
        save_arrays(result_dir / "selected_test_arrays.npz", test_arrays)
    dump_json(result_dir / "stage3_depth_report.json", report)
    print(f"[done] {result_dir / 'stage3_depth_report.json'}", flush=True)


if __name__ == "__main__":
    main()
