"""Minimal PEMS04 H=12 experiment for Resolution-Native F2FCoT V1.

Uses only 3->6->12, selects on VALID, never loads TEST, and writes to a new
experiment namespace.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch import (
    F2FCoTResolutionNativeV1Net,
)
from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1 import (
    FIXED_ROUTE,
    temporal_mean_pool,
)
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from scripts.f2f_cot_runtime import (
    NULL_VAL,
    cot_args,
    load_rescale,
    make_loader,
    select_batch,
)

EXPERIMENT = "f2f_cot_resolution_native_v1"
PROTECTED_F2F_COT_CHECKPOINT = (
    ROOT / "checkpoints" / "PEMS04" / "H12" / "f2f_cot"
    / "formal_v1_seed1" / "extra_best.pt"
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


def model_args() -> dict:
    values = dict(cot_args())
    values.update(
        evidence_num_layer=2,
        reasoner_num_layer=2,
        graph_rank=16,
        adp_hidden_dim=32,
        adp_topk=20,
    )
    return values


def route_loss(model, history, target, rescale):
    output = model.rollout(history, FIXED_ROUTE)
    loss = history.new_zeros(())
    for resolution, prediction, weight in zip(
        FIXED_ROUTE, output["forecasts"], (0.2, 0.3, 1.0)
    ):
        state_target = temporal_mean_pool(target, resolution)
        loss = loss + weight * masked_mae(
            rescale(prediction), rescale(state_target), NULL_VAL
        )
    return loss, output


def rollout_with_previous_ablation(model, history, mode: str):
    state = model.begin_reasoning(history)
    for step_index, resolution in enumerate(FIXED_ROUTE):
        if step_index and state.latest_forecast is not None:
            if mode == "zero":
                state = replace(
                    state, latest_forecast=torch.zeros_like(state.latest_forecast)
                )
            elif mode == "shuffle":
                state = replace(
                    state,
                    latest_forecast=state.latest_forecast.roll(shifts=1, dims=0),
                )
            elif mode != "full":
                raise ValueError(f"unknown ablation mode: {mode}")
        state, _ = model.reason_step(history, state, resolution)
    return state.latest_forecast


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    rescale,
    max_batches: Optional[int] = None,
    include_ablations: bool = False,
) -> dict:
    model.eval()
    final_metrics = {"MAE": [], "RMSE": [], "MAPE": []}
    state_mae = {resolution: [] for resolution in FIXED_ROUTE}
    coherence = {"3->6": [], "6->12": []}
    ablation_mae = {mode: [] for mode in ("full", "zero", "shuffle")}
    forecast_change = {mode: [] for mode in ("zero", "shuffle")}
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target, _ = select_batch(batch, device)
        output = model.rollout(history, FIXED_ROUTE)
        prediction = output["pred"]
        prediction_raw, target_raw = rescale(prediction), rescale(target)
        final_metrics["MAE"].append(float(masked_mae(prediction_raw, target_raw, NULL_VAL)))
        final_metrics["RMSE"].append(float(masked_rmse(prediction_raw, target_raw, NULL_VAL)))
        final_metrics["MAPE"].append(float(masked_mape(prediction_raw, target_raw, NULL_VAL)))
        for resolution, state_prediction in zip(FIXED_ROUTE, output["forecasts"]):
            state_target = temporal_mean_pool(target, resolution)
            state_mae[resolution].append(
                float(masked_mae(rescale(state_prediction), rescale(state_target), NULL_VAL))
            )
        coherence["3->6"].append(
            float(
                (
                    temporal_mean_pool(output["forecasts"][1], 3)
                    - output["steps"][1]["corrected_parent"]
                ).abs().max()
            )
        )
        coherence["6->12"].append(
            float(
                (
                    temporal_mean_pool(output["forecasts"][2], 6)
                    - output["steps"][2]["corrected_parent"]
                ).abs().max()
            )
        )
        if include_ablations:
            ablation_predictions = {"full": prediction}
            for mode in ("zero", "shuffle"):
                ablation_predictions[mode] = rollout_with_previous_ablation(
                    model, history, mode
                )
            for mode, ablated_prediction in ablation_predictions.items():
                ablation_mae[mode].append(
                    float(masked_mae(rescale(ablated_prediction), target_raw, NULL_VAL))
                )
            for mode in ("zero", "shuffle"):
                forecast_change[mode].append(
                    float((rescale(ablation_predictions[mode]) - prediction_raw).abs().mean())
                )
        samples += int(history.shape[0])
    report = {
        "samples": samples,
        **{metric: float(np.mean(values)) for metric, values in final_metrics.items()},
        "state_target_MAE": {
            str(resolution): float(np.mean(values))
            for resolution, values in state_mae.items()
        },
        "max_coherence_violation": {
            edge: float(max(values)) for edge, values in coherence.items()
        },
    }
    if include_ablations:
        means = {mode: float(np.mean(values)) for mode, values in ablation_mae.items()}
        report["previous_state_ablation"] = {
            **means,
            "zero_minus_full_MAE": means["zero"] - means["full"],
            "shuffle_minus_full_MAE": means["shuffle"] - means["full"],
            "mean_abs_forecast_change_physical": {
                mode: float(np.mean(values)) for mode, values in forecast_change.items()
            },
        }
    return report


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "best": best,
            "history": history,
            "model_args": model_args(),
            "route": FIXED_ROUTE,
            "method": "F2FCoTResolutionNativeV1Net",
        },
        path,
    )


def train(model, train_loader, valid_loader, device, rescale, out_dir, args):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    milestones = sorted(
        {
            max(1, int(args.epochs * 0.50)),
            max(1, int(args.epochs * 0.75)),
            max(1, int(args.epochs * 0.90)),
        }
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5
    )
    best_path = out_dir / "resolution_native_v1_best.pt"
    last_path = out_dir / "resolution_native_v1_last.pt"
    best = {"MAE": float("inf"), "epoch": 0}
    history_rows = []
    stale_validations = 0
    start_epoch = 1
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best = dict(checkpoint["best"])
        history_rows = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses, gradient_norms = [], []
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            history, target, _ = select_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = route_loss(model, history, target, rescale)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch} batch={batch_index}")
            loss.backward()
            gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()

        run_validation = (
            epoch == 1 or epoch % args.valid_every == 0 or epoch == args.epochs
        )
        valid, improved = None, False
        if run_validation:
            valid = evaluate(
                model, valid_loader, device, rescale, max_batches=args.max_valid_batches
            )
            if valid["MAE"] < best["MAE"]:
                best = {"MAE": float(valid["MAE"]), "epoch": int(epoch)}
                improved, stale_validations = True, 0
                save_checkpoint(
                    best_path, model, optimizer, scheduler, epoch, best, history_rows
                )
            else:
                stale_validations += 1
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "gradient_norm": float(np.mean(gradient_norms)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "valid": valid,
            "improved": improved,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history_rows.append(row)
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best, history_rows)
        dump_json(out_dir / "training_history.json", history_rows)
        valid_text = f" VALID_MAE={valid['MAE']:.4f}" if valid is not None else ""
        print(
            f"[native-v1] epoch={epoch:03d} loss={row['train_loss']:.4f}"
            f"{valid_text} best={best['MAE']:.4f}@{best['epoch']} "
            f"seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )
        if (
            not args.smoke
            and args.early_stop_validations > 0
            and stale_validations >= args.early_stop_validations
        ):
            print("[native-v1] early stopping", flush=True)
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return best, history_rows, best_path


@torch.inference_mode()
def profile_latency(model, example, warmup: int, repeats: int) -> dict:
    if example.device.type != "cuda":
        return {"available": False, "device": str(example.device)}
    model.eval()
    for _ in range(warmup):
        model.rollout(example, FIXED_ROUTE)
    torch.cuda.synchronize(example.device)
    encode_samples = []
    step_samples = {edge: [] for edge in ("0->3", "3->6", "6->12")}
    total_samples = []
    for _ in range(repeats):
        total_start, total_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        encode_start, encode_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        total_start.record()
        encode_start.record()
        state = model.begin_reasoning(example)
        encode_end.record()
        encode_end.synchronize()
        encode_samples.append(encode_start.elapsed_time(encode_end))
        for resolution in FIXED_ROUTE:
            edge = f"{state.current_resolution}->{resolution}"
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            state, _ = model.reason_step(example, state, resolution)
            end.record()
            end.synchronize()
            step_samples[edge].append(start.elapsed_time(end))
        total_end.record()
        total_end.synchronize()
        total_samples.append(total_start.elapsed_time(total_end))

    def summarize(values):
        ordered = sorted(float(value) for value in values)
        return {
            "median_ms": float(statistics.median(ordered)),
            "mean_ms": float(statistics.mean(ordered)),
            "p90_ms": float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]),
        }

    return {
        "available": True,
        "device": str(example.device),
        "batch_size": int(example.shape[0]),
        "warmup": warmup,
        "repeats": repeats,
        "history_evidence_encode": summarize(encode_samples),
        "per_reasoning_step": {
            edge: summarize(values) for edge, values in step_samples.items()
        },
        "trajectory_total": summarize(total_samples),
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


@torch.inference_mode()
def profile_flops(model, example) -> dict:
    model.eval()
    encode_flops = _profile_flops(
        lambda: model.begin_reasoning(example), example.device
    )
    state = model.begin_reasoning(example)
    step_flops = {}
    for resolution in FIXED_ROUTE:
        current, step_state = state.current_resolution, state
        step_flops[f"{current}->{resolution}"] = _profile_flops(
            lambda: model.reason_step(example, step_state, resolution),
            example.device,
        )
        state, _ = model.reason_step(example, state, resolution)
    return {
        "method": "torch.profiler.with_flops",
        "note": "Counts profiler-supported operators; intended for within-model relative comparison.",
        "batch_size": int(example.shape[0]),
        "history_evidence_encode": encode_flops,
        "per_reasoning_step": step_flops,
        "trajectory_supported_flops": encode_flops + sum(step_flops.values()),
    }


@torch.inference_mode()
def structural_report(model, example) -> dict:
    model.eval()
    parameter_ids = model.shared_reasoner_parameter_ids()
    output = model.rollout(example, FIXED_ROUTE)
    return {
        "forecast_shapes": [list(value.shape) for value in output["forecasts"]],
        "active_hidden_shapes": [
            list(step["active_hidden_shape"]) for step in output["steps"]
        ],
        "active_future_lengths": [
            int(step["active_future_length"]) for step in output["steps"]
        ],
        "history_encode_count": int(output["history_encode_count"]),
        "reasoning_calls": int(output["reasoning_calls"]),
        "created_full_horizon_canvas": bool(output["created_full_horizon_canvas"]),
        "forecast_canvas_keys_present": any(
            "forecast_canvas" in step for step in output["steps"]
        ),
        "shared_reasoner_parameter_object_count": len(parameter_ids),
        "shared_reasoner_parameter_ids_unique": len(parameter_ids) == len(set(parameter_ids)),
        "one_shared_reasoner_reused": True,
        "fixed_route": list(FIXED_ROUTE),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--valid-every", type=int, default=2)
    parser.add_argument("--early-stop-validations", type=int, default=6)
    parser.add_argument("--tag", default="probe_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-valid-batches", type=int)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--profile-repeats", type=int, default=100)
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
        args.early_stop_validations = 0
        args.profile_repeats = 5
        args.tag = "smoke"
    seed_all(args.seed)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    result_dir = ROOT / "results" / EXPERIMENT / f"{args.tag}_seed{args.seed}"
    checkpoint_dir = (
        ROOT / "checkpoints" / "PEMS04" / "H12" / EXPERIMENT
        / f"{args.tag}_seed{args.seed}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = F2FCoTResolutionNativeV1Net(**model_args()).to(device)
    warm_start = {
        "enabled": not args.no_warm_start,
        "source": str(PROTECTED_F2F_COT_CHECKPOINT),
    }
    if not args.no_warm_start:
        checkpoint = torch.load(
            PROTECTED_F2F_COT_CHECKPOINT, map_location="cpu", weights_only=False
        )
        warm_start.update(
            model.warm_start_from_f2f_cot(checkpoint["model_state_dict"])
        )
    rescale = load_rescale()
    train_loader = make_loader("train", args.batch_size, True, args.workers)
    valid_loader = make_loader("valid", args.batch_size, False, args.workers)
    example = select_batch(next(iter(valid_loader)), device)[0][:1]
    structure = structural_report(model, example)
    dump_json(result_dir / "structural_report.json", structure)
    print(f"[native-v1] structure={structure}", flush=True)
    print(
        f"[native-v1] parameters={model.parameter_breakdown()} "
        f"warm_start={warm_start}",
        flush=True,
    )
    best, training_history, best_path = train(
        model, train_loader, valid_loader, device, rescale, checkpoint_dir, args
    )
    final_valid = evaluate(
        model,
        valid_loader,
        device,
        rescale,
        max_batches=args.max_valid_batches,
        include_ablations=True,
    )
    latency = profile_latency(
        model,
        example,
        warmup=5 if args.smoke else 20,
        repeats=args.profile_repeats,
    )
    flops = profile_flops(model, example)
    report = {
        "method": "F2FCoTResolutionNativeV1Net",
        "route": list(FIXED_ROUTE),
        "seed": args.seed,
        "architecture": {
            "history_evidence_encoded_once": True,
            "shared_kasa_patch_downsample_trunks": True,
            "one_shared_resolution_native_reasoner": True,
            "explicit_forecast_state_only": True,
            "hidden_full_horizon_memory": False,
            "fixed_horizon_forecast_canvas": False,
            "spatial_node_coarsening": False,
            "dynamic_controller": False,
            "progressive_low_detail_decomposition": True,
        },
        "parameters": model.parameter_breakdown(),
        "warm_start": warm_start,
        "protected_checkpoint_untouched": str(PROTECTED_F2F_COT_CHECKPOINT),
        "selected_checkpoint": str(best_path),
        "best": best,
        "structural_verification": structure,
        "valid": final_valid,
        "latency": latency,
        "flops": flops,
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(training_history),
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-4,
            "raw_scale_masked_MAE": True,
            "loss_weights": [0.2, 0.3, 1.0],
            "VALID_only_selection": True,
            "TEST_loaded": False,
        },
    }
    dump_json(result_dir / "final_report.json", report)
    print(
        f"[native-v1] done report={result_dir / 'final_report.json'} "
        f"VALID_MAE={final_valid['MAE']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
