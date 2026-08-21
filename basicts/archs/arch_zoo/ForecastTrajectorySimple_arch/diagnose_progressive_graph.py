"""Measure whether frozen trajectory edges behave as progressive corrections."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from basicts.archs.arch_zoo.ChainForecasting_arch.ChainForecasting_arch import (
    ChainForecasting,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)

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


def _safe_corr(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.float().numpy()
    y = right.float().numpy()
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _summarize_edge(rows: dict[str, list[torch.Tensor]]) -> dict:
    values = {key: torch.cat(parts) for key, parts in rows.items()}
    gain = values["mae_gain"]
    return {
        "samples": len(gain),
        "anchor_mae": float(values["anchor_mae"].mean()),
        "refined_mae": float(values["refined_mae"].mean()),
        "mean_mae_gain": float(gain.mean()),
        "benefit_fraction": float((gain > 0).float().mean()),
        "harm_fraction": float((gain < 0).float().mean()),
        "mean_correction_abs": float(values["correction_abs"].mean()),
        "mean_error_abs": float(values["error_abs"].mean()),
        "positive_error_dot_fraction": float(
            (values["error_dot"] > 0).float().mean()
        ),
        "mean_error_cosine": float(values["error_cosine"].mean()),
        "correction_magnitude_vs_gain_pearson": _safe_corr(
            values["correction_abs"], gain
        ),
        "alignment_vs_gain_pearson": _safe_corr(
            values["error_cosine"], gain
        ),
    }


@torch.inference_mode()
def diagnose_split(model, loader, device, mean, std, max_batches=None):
    edges = defaultdict(lambda: defaultdict(list))
    previous_to_next = defaultdict(lambda: defaultdict(list))
    route_losses = defaultdict(list)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target = prepare_batch(batch, device)
        target_raw = to_physical(target, mean, std)
        for route in TRAJECTORIES:
            trace = model.execute_trajectory(history, route)
            route_losses[route].append(
                per_sample_mae(
                    to_physical(trace["pred"], mean, std), target_raw, 0.0
                ).cpu()
            )
            prior_edge_metrics = None
            source = None
            previous = None
            for target_resolution in route:
                current = trace["state_forecasts"][target_resolution]
                anchor = (
                    history[:, -1:, :, :1].expand(
                        -1, target_resolution, -1, -1
                    )
                    if previous is None
                    else interpolate_forecast(previous, target_resolution)
                )
                # Compare every reached state under the same full-resolution
                # stopping semantics. Pooled low-resolution targets otherwise
                # make restoring high-frequency detail look spuriously harmful.
                anchor_full = model.finalize_forecast(
                    interpolate_forecast(anchor, model.output_len), history
                )
                refined_full = model.finalize_forecast(
                    interpolate_forecast(current, model.output_len), history
                )
                desired = target - anchor_full
                correction = refined_full - anchor_full
                anchor_mae = per_sample_mae(
                    to_physical(anchor_full, mean, std),
                    target_raw,
                    0.0,
                )
                refined_mae = per_sample_mae(
                    to_physical(refined_full, mean, std),
                    target_raw,
                    0.0,
                )
                flat_correction = correction.flatten(1)
                flat_desired = desired.flatten(1)
                dot = (flat_correction * flat_desired).sum(1)
                cosine = dot / (
                    flat_correction.square().sum(1).sqrt()
                    * flat_desired.square().sum(1).sqrt()
                ).clamp_min(1e-8)
                metrics = {
                    "anchor_mae": anchor_mae.cpu(),
                    "refined_mae": refined_mae.cpu(),
                    "mae_gain": (anchor_mae - refined_mae).cpu(),
                    "correction_abs": correction.abs().mean((1, 2, 3)).cpu(),
                    "error_abs": desired.abs().mean((1, 2, 3)).cpu(),
                    "error_dot": dot.cpu(),
                    "error_cosine": cosine.cpu(),
                }
                context = f"{route_name(route)}::{source}->{target_resolution}"
                for key, value in metrics.items():
                    edges[context][key].append(value)
                if prior_edge_metrics is not None:
                    pair = f"{route_name(route)}::{prior_edge_metrics['name']}=>{source}->{target_resolution}"
                    previous_to_next[pair]["previous_gain"].append(
                        prior_edge_metrics["mae_gain"]
                    )
                    previous_to_next[pair]["previous_alignment"].append(
                        prior_edge_metrics["error_cosine"]
                    )
                    previous_to_next[pair]["previous_correction_abs"].append(
                        prior_edge_metrics["correction_abs"]
                    )
                    previous_to_next[pair]["next_gain"].append(metrics["mae_gain"])
                prior_edge_metrics = dict(metrics)
                prior_edge_metrics["name"] = f"{source}->{target_resolution}"
                previous = current
                source = target_resolution

    route_matrix = {route: torch.cat(parts) for route, parts in route_losses.items()}
    canonical = route_matrix[(3, 6, 12)]
    route_report = {
        route_name(route): {
            "mae": float(loss.mean()),
            "better_than_canonical_fraction": float((loss < canonical).float().mean()),
            "mean_gain_vs_canonical": float((canonical - loss).mean()),
        }
        for route, loss in route_matrix.items()
    }
    pair_report = {}
    for name, rows in previous_to_next.items():
        values = {key: torch.cat(parts) for key, parts in rows.items()}
        pair_report[name] = {
            "previous_gain_vs_next_gain_pearson": _safe_corr(
                values["previous_gain"], values["next_gain"]
            ),
            "previous_alignment_vs_next_gain_pearson": _safe_corr(
                values["previous_alignment"], values["next_gain"]
            ),
            "previous_correction_magnitude_vs_next_gain_pearson": _safe_corr(
                values["previous_correction_abs"], values["next_gain"]
            ),
        }
    return {
        "routes": route_report,
        "edges": {name: _summarize_edge(rows) for name, rows in edges.items()},
        "previous_change_predicts_next": pair_report,
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
    report = {
        "interpretation": (
            "Positive edge gain means the executed transition improved over its "
            "interpolated previous forecast (or persistence at START)."
        ),
        "train": diagnose_split(
            model, loader("train"), device, mean, std, args.max_train_batches
        ),
        "valid": diagnose_split(
            model, loader("valid"), device, mean, std, args.max_eval_batches
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
