"""Train/evaluate the CoT-style recurrent F2F containment methodology.

Scientific order is enforced in code:
1. VALID-only canonical baseline and fixed 3->6->12 containment.
2. Only after containment passes, VALID-only 3->4->6->12 curriculum.
3. TEST is constructed/evaluated only after the methodology checkpoint is fixed.
There is intentionally no controller or RL code in this runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.ChainForecasting_arch import (
    ChainForecasting,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)
from basicts.archs.arch_zoo.F2FCoT_arch import F2FCoTNet
from basicts.archs.arch_zoo.F2FCoT_arch.f2f_cot import pool_forecast
from basicts.data import SCALER_REGISTRY
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.utils import load_pkl


DATA_DIR = ROOT / "datasets" / "PEMS04"
DATA_FILE = DATA_DIR / "data_in12_out12.pkl"
INDEX_FILE = DATA_DIR / "index_in12_out12.pkl"
SCALER_FILE = DATA_DIR / "scaler_in12_out12.pkl"
ADJ_MX = DATA_DIR / "adj_mx.pkl"
CANONICAL_CKPT = (
    ROOT
    / "checkpoints"
    / "ChainForecasting_100"
    / "cd0ad9dcc9dd855c893d064f10450546"
    / "ChainForecasting_best_val_MAE.pt"
)
FIXED_ROUTE = (3, 6, 12)
EXTRA_ROUTE = (3, 4, 6, 12)
NULL_VAL = 0.0
CONTAINMENT_TOL = 0.10


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def count_params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def canonical_args() -> dict:
    return {
        "node_size": 307,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 4,
        "main_input_dim": 3,
        "patch_len": 3,
        "stride": 4,
        "td_size": 288,
        "dw_size": 7,
        "d_td": 32,
        "d_dw": 32,
        "d_d": 32,
        "d_spa": 32,
        "if_time_in_day": True,
        "if_day_in_week": True,
        "if_spatial": True,
        "num_layer": 2,
        "spatial_scheme": "C",
        "adj_mx_path": str(ADJ_MX),
        "use_gcn": True,
        "gcn_hidden_dim": 64,
        "use_dynamic_spatial": True,
        "dyn_hidden_dim": 64,
        "dyn_topk": 20,
        "dyn_tau": 0.5,
        "dyn_static_weight": 0.2,
        "use_adaptive_adj": True,
        "adp_hidden_dim": 32,
        "adp_topk": 20,
        "adp_tau": 0.5,
        "use_hybrid_graph": True,
        "hybrid_alpha": 0.2,
        "use_patch_branch": True,
        "use_downsample_branch": True,
        "use_linear_residual_branch": True,
        "patch_embedding_mode": "serial_concat",
        "patch_data_input_mode": "all",
        "post_spatial_mode": "adaptive_only",
        "spatial_placement": "final",
        "use_pre_temporal_spatial_enhancement": False,
        "keep_output_prior_residual": False,
        "use_input_prior_enhancement": False,
        "use_graph_spectral_calibration": False,
        "use_extra_prior_input": False,
        "use_prev_condition": True,
        "chain_lengths": [3, 6, 12],
        "chain_loss_weights": [0.2, 0.3, 1.0],
    }


def cot_args() -> dict:
    return {
        "node_size": 307,
        "input_len": 12,
        "output_len": 12,
        "patch_len": 3,
        "stride": 4,
        "td_size": 288,
        "dw_size": 7,
        "d_d": 64,
        "d_td": 48,
        "d_dw": 48,
        "d_spa": 64,
        "num_layer": 4,
        "memory_dim": 16,
        "context_channels": 4,
        "resolution_dim": 32,
        "condition_channels": 8,
        "resolutions": [2, 3, 4, 6, 12],
        "spatial_scheme": "C",
        "adj_mx_path": str(ADJ_MX),
        "use_gcn": True,
        "gcn_hidden_dim": 64,
        "use_dynamic_spatial": True,
        "dyn_hidden_dim": 64,
        "dyn_topk": 20,
        "dyn_tau": 0.5,
        "dyn_static_weight": 0.2,
        "use_adaptive_adj": True,
        "adp_hidden_dim": 32,
        "adp_topk": 20,
        "adp_tau": 0.5,
        "use_hybrid_graph": True,
        "hybrid_alpha": 0.2,
        "post_spatial_mode": "adaptive_only",
        "patch_embedding_mode": "serial_concat",
        "patch_data_input_mode": "all",
    }


class ForecastSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base[self.indices[index]]


def collate_forecasts(batch):
    futures, histories, sample_indices = zip(*batch)
    return (
        torch.stack(futures, 0),
        torch.stack(histories, 0),
        torch.tensor(sample_indices, dtype=torch.long),
    )


def make_loader(split: str, batch_size: int, shuffle: bool, num_workers: int = 0):
    dataset = IndexedTimeSeriesForecastingDataset(
        str(DATA_FILE), str(INDEX_FILE), split
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_forecasts,
        drop_last=False,
        pin_memory=False,
    )


def load_rescale() -> Callable[[torch.Tensor], torch.Tensor]:
    scaler = load_pkl(str(SCALER_FILE))
    function = SCALER_REGISTRY.get(scaler["func"])

    def rescale(tensor: torch.Tensor) -> torch.Tensor:
        return function(tensor, **scaler["args"])

    return rescale


def select_batch(batch, device):
    future, history, sample_index = batch
    return (
        history.to(device)[..., [0, 1, 2, 3]],
        future.to(device)[..., [0]],
        sample_index,
    )


def per_sample_mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isclose(
        target,
        torch.tensor(NULL_VAL, device=target.device, dtype=target.dtype),
        atol=5e-5,
        rtol=0.0,
    )
    error = (prediction - target).abs() * mask
    return error.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1)


def route_key(route: Sequence[int]) -> str:
    return "-".join(str(value) for value in route)


def load_canonical(device):
    model = ChainForecasting(**canonical_args())
    checkpoint = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), {
        "parameters": count_params(model),
        "epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_metrics": checkpoint.get("best_metrics", {}),
        "checkpoint": str(CANONICAL_CKPT),
    }


@torch.inference_mode()
def evaluate_canonical(model, loader, device, rescale, max_batches=None):
    model.eval()
    mae, rmse, mape = [], [], []
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target, _ = select_batch(batch, device)
        prediction = model(history_data=history)
        pred_raw, target_raw = rescale(prediction), rescale(target)
        mae.append(float(masked_mae(pred_raw, target_raw, NULL_VAL).item()))
        rmse.append(float(masked_rmse(pred_raw, target_raw, NULL_VAL).item()))
        mape.append(float(masked_mape(pred_raw, target_raw, NULL_VAL).item()))
        sample_count += len(history)
    return {
        "samples": sample_count,
        "MAE": float(np.mean(mae)),
        "RMSE": float(np.mean(rmse)),
        "MAPE": float(np.mean(mape)),
        "trajectory": list(FIXED_ROUTE),
    }


def weighted_route_loss(model, history, target, route, rescale):
    output = model.rollout(history, route)
    if tuple(route) == FIXED_ROUTE:
        weights = (0.2, 0.3, 1.0)
    elif tuple(route) == EXTRA_ROUTE:
        weights = (0.15, 0.15, 0.3, 1.0)
    else:
        weights = tuple([0.2] * (len(route) - 1) + [1.0])
    loss = history.new_zeros(())
    for resolution, prediction, weight in zip(route, output["forecasts"], weights):
        state_target = pool_forecast(target, int(resolution))
        loss = loss + float(weight) * masked_mae(
            rescale(prediction), rescale(state_target), NULL_VAL
        )
    return loss, output


@torch.inference_mode()
def evaluate_routes(model, loader, routes, device, rescale, max_batches=None):
    model.eval()
    routes = [tuple(int(value) for value in route) for route in routes]
    final_batch_metrics = {
        route: {"MAE": [], "RMSE": [], "MAPE": []} for route in routes
    }
    state_mae = {route: {resolution: [] for resolution in route} for route in routes}
    refinement_gains = {
        route: {f"{left}->{right}": [] for left, right in zip(route, route[1:])}
        for route in routes
    }
    final_sample_losses = {route: [] for route in routes}
    route_difference = []
    sample_count = 0
    reasoning_calls = {route: set() for route in routes}

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target, _ = select_batch(batch, device)
        target_raw = rescale(target)
        batch_outputs = {}
        for route in routes:
            output = model.rollout(history, route)
            batch_outputs[route] = output
            reasoning_calls[route].add(int(output["reasoning_calls"]))
            prediction_raw = rescale(output["pred"])
            final_batch_metrics[route]["MAE"].append(
                float(masked_mae(prediction_raw, target_raw, NULL_VAL).item())
            )
            final_batch_metrics[route]["RMSE"].append(
                float(masked_rmse(prediction_raw, target_raw, NULL_VAL).item())
            )
            final_batch_metrics[route]["MAPE"].append(
                float(masked_mape(prediction_raw, target_raw, NULL_VAL).item())
            )
            final_sample_losses[route].append(
                per_sample_mae(prediction_raw, target_raw).cpu()
            )

            projected_losses = {}
            for resolution, state in zip(route, output["forecasts"]):
                own_target = pool_forecast(target, resolution)
                own_mae = masked_mae(rescale(state), rescale(own_target), NULL_VAL)
                state_mae[route][resolution].append(float(own_mae.item()))
                projected = interpolate_forecast(state, target.shape[1])
                projected_losses[resolution] = per_sample_mae(
                    rescale(projected), target_raw
                ).cpu()
            for left, right in zip(route, route[1:]):
                gain = projected_losses[left] - projected_losses[right]
                refinement_gains[route][f"{left}->{right}"].append(gain)

        if len(routes) >= 2:
            first_raw = rescale(batch_outputs[routes[0]]["pred"])
            second_raw = rescale(batch_outputs[routes[1]]["pred"])
            route_difference.append(
                (first_raw - second_raw).abs().mean(dim=(1, 2, 3)).cpu()
            )
        sample_count += len(history)

    report = {"samples": sample_count, "routes": {}}
    for route in routes:
        route_report = {
            metric: float(np.mean(values))
            for metric, values in final_batch_metrics[route].items()
        }
        route_report["reasoning_calls"] = sorted(reasoning_calls[route])
        route_report["state_target_MAE"] = {
            str(resolution): float(np.mean(values))
            for resolution, values in state_mae[route].items()
        }
        refinement = {}
        for edge, chunks in refinement_gains[route].items():
            values = torch.cat(chunks)
            refinement[edge] = {
                "mean_full_resolution_MAE_gain": float(values.mean()),
                "improve_fraction": float((values > 0).float().mean()),
                "median_gain": float(values.median()),
            }
        route_report["successive_refinement"] = refinement
        report["routes"][route_key(route)] = route_report

    if len(routes) >= 2:
        first, second = routes[:2]
        first_loss = torch.cat(final_sample_losses[first])
        second_loss = torch.cat(final_sample_losses[second])
        difference = torch.cat(route_difference)
        oracle = torch.minimum(first_loss, second_loss)
        report["route_comparison"] = {
            "reference": route_key(first),
            "extra_step": route_key(second),
            "extra_minus_reference_MAE": float(second_loss.mean() - first_loss.mean()),
            "extra_better_fraction": float((second_loss < first_loss).float().mean()),
            "mean_abs_forecast_change_physical": float(difference.mean()),
            "sample_oracle_MAE": float(oracle.mean()),
            "oracle_gain_vs_best_fixed": float(
                min(first_loss.mean(), second_loss.mean()) - oracle.mean()
            ),
        }
    return report


def save_training_checkpoint(path, model, optimizer, scheduler, epoch, best, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "best": best,
            "history": history,
            "model_args": cot_args(),
        },
        path,
    )


def train_fixed(
    model,
    train_loader,
    valid_loader,
    device,
    rescale,
    epochs,
    out_dir,
    resume,
    max_train_batches=None,
    max_valid_batches=None,
):
    # The same unit is inside its own three-step feedback loop.  A lower LR than
    # the independent-stage canonical model prevents early recurrent blow-up.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[35, 60, 80, 95], gamma=0.5
    )
    best = {"MAE": float("inf"), "epoch": 0}
    history_rows = []
    start_epoch = 1
    last_path = out_dir / "fixed_last.pt"
    best_path = out_dir / "fixed_best.pt"
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = dict(checkpoint["best"])
        history_rows = list(checkpoint.get("history", []))
        print(f"[fixed] resumed at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, int(epochs) + 1):
        model.train()
        losses = []
        grad_norms = []
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            history, target, _ = select_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = weighted_route_loss(
                model, history, target, FIXED_ROUTE, rescale
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite fixed loss at epoch {epoch}")
            loss.backward()
            grad_norms.append(
                float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        valid = evaluate_routes(
            model,
            valid_loader,
            [FIXED_ROUTE],
            device,
            rescale,
            max_batches=max_valid_batches,
        )
        valid_mae = valid["routes"][route_key(FIXED_ROUTE)]["MAE"]
        elapsed = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "grad_norm": float(np.mean(grad_norms)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "valid": valid,
            "epoch_seconds": elapsed,
        }
        history_rows.append(row)
        if valid_mae < float(best["MAE"]):
            best = {"MAE": float(valid_mae), "epoch": int(epoch)}
            save_training_checkpoint(
                best_path, model, optimizer, scheduler, epoch, best, history_rows
            )
        save_training_checkpoint(
            last_path, model, optimizer, scheduler, epoch, best, history_rows
        )
        dump_json(out_dir / "fixed_history.json", history_rows)
        print(
            f"[fixed] epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"VALID_MAE={valid_mae:.4f} best={best['MAE']:.4f}@{best['epoch']} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, best, history_rows


def train_extra(
    model,
    train_loader,
    valid_loader,
    device,
    rescale,
    epochs,
    out_dir,
    canonical_valid_mae,
    resume,
    max_train_batches=None,
    max_valid_batches=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 35, 45], gamma=0.5
    )
    best = {"score": float("inf"), "epoch": 0, "canonical_MAE": float("inf")}
    history_rows = []
    start_epoch = 1
    last_path = out_dir / "extra_last.pt"
    best_path = out_dir / "extra_best.pt"
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = dict(checkpoint["best"])
        history_rows = list(checkpoint.get("history", []))

    for epoch in range(start_epoch, int(epochs) + 1):
        model.train()
        losses = []
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            history, target, _ = select_batch(batch, device)
            route = FIXED_ROUTE if batch_index % 2 == 0 else EXTRA_ROUTE
            optimizer.zero_grad(set_to_none=True)
            loss, _ = weighted_route_loss(model, history, target, route, rescale)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite extra-step loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        valid = evaluate_routes(
            model,
            valid_loader,
            [FIXED_ROUTE, EXTRA_ROUTE],
            device,
            rescale,
            max_batches=max_valid_batches,
        )
        canonical_mae = valid["routes"][route_key(FIXED_ROUTE)]["MAE"]
        extra_mae = valid["routes"][route_key(EXTRA_ROUTE)]["MAE"]
        eligible = canonical_mae <= float(canonical_valid_mae) + CONTAINMENT_TOL
        score = 0.5 * (canonical_mae + extra_mae)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "eligible": bool(eligible),
            "selection_score": float(score),
            "valid": valid,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history_rows.append(row)
        if eligible and score < float(best["score"]):
            best = {
                "score": float(score),
                "epoch": int(epoch),
                "canonical_MAE": float(canonical_mae),
                "extra_MAE": float(extra_mae),
            }
            save_training_checkpoint(
                best_path, model, optimizer, scheduler, epoch, best, history_rows
            )
        save_training_checkpoint(
            last_path, model, optimizer, scheduler, epoch, best, history_rows
        )
        dump_json(out_dir / "extra_history.json", history_rows)
        print(
            f"[extra] epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"VALID 3-6-12={canonical_mae:.4f} 3-4-6-12={extra_mae:.4f} "
            f"eligible={eligible} best_score={best['score']:.4f}",
            flush=True,
        )
    if best_path.is_file():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, best, history_rows


@torch.inference_mode()
def profile_reasoning_steps(model, example, trajectory, warmup=20, repeats=100):
    if example.device.type != "cuda":
        return {"device": str(example.device), "available": False}
    model.eval()
    trajectory = tuple(trajectory)
    for _ in range(warmup):
        model.rollout(example, trajectory)
    torch.cuda.synchronize(example.device)
    samples = {f"{left}->{right}": [] for left, right in zip((0, *trajectory[:-1]), trajectory)}
    total_samples = []
    for _ in range(repeats):
        state = model.begin_reasoning(example)
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
        for next_resolution in trajectory:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            current = state.current_resolution
            state, _ = model.reason_step(example, state, next_resolution)
            end.record()
            end.synchronize()
            samples[f"{current}->{next_resolution}"].append(start.elapsed_time(end))
        total_end.record()
        total_end.synchronize()
        total_samples.append(total_start.elapsed_time(total_end))

    def summary(values):
        ordered = sorted(float(value) for value in values)
        return {
            "median_ms": float(statistics.median(ordered)),
            "mean_ms": float(statistics.mean(ordered)),
            "p90_ms": float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]),
        }

    return {
        "device": str(example.device),
        "batch_size": int(example.shape[0]),
        "warmup": warmup,
        "repeats": repeats,
        "per_reasoning_call": {edge: summary(values) for edge, values in samples.items()},
        "trajectory_total": summary(total_samples),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fixed-epochs", type=int, default=100)
    parser.add_argument("--extra-epochs", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-extra", action="store_true")
    parser.add_argument("--evaluate-test-if-fixed", action="store_true")
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-valid-batches", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.fixed_epochs = 1
        args.extra_epochs = 1
        args.batch_size = min(args.batch_size, 4)
        args.tag = "smoke"
        args.max_train_batches = 2
        args.max_valid_batches = 2
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results" / "f2f_cot" / f"{args.tag}_seed{args.seed}"
    checkpoint_dir = (
        ROOT / "checkpoints" / "PEMS04" / "H12" / "f2f_cot" / f"{args.tag}_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rescale = load_rescale()
    train_loader = make_loader("train", args.batch_size, True, args.workers)
    valid_loader = make_loader("valid", args.batch_size, False, args.workers)

    canonical, canonical_meta = load_canonical(device)
    canonical_valid = evaluate_canonical(
        canonical,
        valid_loader,
        device,
        rescale,
        max_batches=args.max_valid_batches,
    )
    del canonical
    torch.cuda.empty_cache() if device.type == "cuda" else None

    model = F2FCoTNet(**cot_args()).to(device)
    parameter_report = model.parameter_breakdown()
    parameter_report.update(
        {
            "original_f2f": canonical_meta["parameters"],
            "total_to_original_ratio": count_params(model)
            / float(canonical_meta["parameters"]),
            "resolution_specific_embedding_parameters": model.resolution_specific_parameter_count(),
        }
    )
    dump_json(
        out_dir / "preflight.json",
        {
            "canonical": canonical_meta,
            "canonical_VALID": canonical_valid,
            "model_args": cot_args(),
            "parameters": parameter_report,
            "fixed_route": list(FIXED_ROUTE),
            "test_loaded": False,
            "no_RL_or_controller": True,
        },
    )
    print(
        f"[preflight] canonical VALID={canonical_valid['MAE']:.4f} "
        f"params original={canonical_meta['parameters']} new={count_params(model)} "
        f"ratio={parameter_report['total_to_original_ratio']:.3f}",
        flush=True,
    )

    fixed_out = checkpoint_dir
    model, fixed_best, _ = train_fixed(
        model,
        train_loader,
        valid_loader,
        device,
        rescale,
        args.fixed_epochs,
        fixed_out,
        args.resume,
        args.max_train_batches,
        args.max_valid_batches,
    )
    fixed_valid = evaluate_routes(
        model,
        valid_loader,
        [FIXED_ROUTE],
        device,
        rescale,
        max_batches=args.max_valid_batches,
    )
    fixed_mae = fixed_valid["routes"][route_key(FIXED_ROUTE)]["MAE"]
    containment_gap = float(fixed_mae - canonical_valid["MAE"])
    containment_pass = containment_gap <= CONTAINMENT_TOL

    example_batch = next(iter(valid_loader))
    example, _, _ = select_batch(example_batch, device)
    latency = profile_reasoning_steps(model, example[:1], FIXED_ROUTE)
    fixed_report = {
        "pass": bool(containment_pass),
        "tolerance": CONTAINMENT_TOL,
        "canonical_VALID_MAE": canonical_valid["MAE"],
        "new_fixed_VALID_MAE": fixed_mae,
        "gap": containment_gap,
        "best": fixed_best,
        "valid": fixed_valid,
        "latency": latency,
        "test_evaluated": False,
    }
    dump_json(out_dir / "fixed_containment.json", fixed_report)
    print(
        "BACKBONE_CONTAINMENT_PASS" if containment_pass else "BACKBONE_CONTAINMENT_FAIL",
        flush=True,
    )

    final_model = model
    extra_report = None
    methodology_fixed = False
    if containment_pass and not args.skip_extra:
        final_model, extra_best, _ = train_extra(
            model,
            train_loader,
            valid_loader,
            device,
            rescale,
            args.extra_epochs,
            checkpoint_dir,
            canonical_valid["MAE"],
            args.resume,
            args.max_train_batches,
            args.max_valid_batches,
        )
        extra_valid = evaluate_routes(
            final_model,
            valid_loader,
            [FIXED_ROUTE, EXTRA_ROUTE],
            device,
            rescale,
            max_batches=args.max_valid_batches,
        )
        extra_report = {
            "best": extra_best,
            "valid": extra_valid,
            "canonical_containment_retained": bool(
                extra_valid["routes"][route_key(FIXED_ROUTE)]["MAE"]
                <= canonical_valid["MAE"] + CONTAINMENT_TOL
            ),
            "test_evaluated": False,
        }
        methodology_fixed = bool(
            extra_best.get("epoch", 0) > 0
            and extra_report["canonical_containment_retained"]
        )
        dump_json(out_dir / "extra_step_valid.json", extra_report)
    elif containment_pass and args.skip_extra:
        methodology_fixed = True

    test_report = None
    if methodology_fixed and args.evaluate_test_if_fixed and not args.smoke:
        # This is the first point at which a TEST loader is constructed.
        test_loader = make_loader("test", args.batch_size, False, args.workers)
        routes = [FIXED_ROUTE, EXTRA_ROUTE] if extra_report is not None else [FIXED_ROUTE]
        test_report = evaluate_routes(final_model, test_loader, routes, device, rescale)
        dump_json(out_dir / "test_once.json", test_report)

    final_report = {
        "method": "F2FCoTNet",
        "concept": "same enlarged KASA forecasting unit repeatedly applied to explicit forecast-derived context",
        "canonical": canonical_meta,
        "canonical_VALID": canonical_valid,
        "parameters": parameter_report,
        "fixed_containment": fixed_report,
        "extra_step": extra_report,
        "methodology_fixed": methodology_fixed,
        "test": test_report,
        "dynamic_reasoning_implemented": False,
        "dynamic_reasoning_gate": (
            "eligible_for_design" if methodology_fixed else "blocked_by_forecasting_validation"
        ),
    }
    dump_json(out_dir / "final_report.json", final_report)
    print(f"[done] {out_dir / 'final_report.json'}", flush=True)


if __name__ == "__main__":
    main()
