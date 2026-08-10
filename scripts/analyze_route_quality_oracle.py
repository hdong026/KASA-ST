#!/usr/bin/env python3
"""Offline analysis of train/valid route-loss oracles (no model execution)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budget_from_intensity,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    feasible_mask_from_budget,
    oracle_best_feasible_route,
)
from basicts.data.route_quality_dataset import dedupe_route_loss_records, load_oracle_json


def _entropy(counts: Counter, n: int) -> float:
    if n <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    return float(ent)


def analyze_split(path: Path, deltas: list[float], intensities: list[float]) -> dict[str, Any]:
    oracle = load_oracle_json(path)
    packed = dedupe_route_loss_records(oracle)
    routes = packed["candidate_routes"]
    costs = torch.tensor(packed["route_costs"], dtype=torch.float32)
    losses = torch.tensor(
        [packed["route_losses"][i] for i in packed["sample_indices"]],
        dtype=torch.float32,
    )
    n, r = losses.shape
    mean_mae = losses.mean(dim=0).tolist()
    best_ids = losses.argmin(dim=-1)
    hist = Counter(int(x) for x in best_ids.tolist())
    # Pairwise preference: fraction where route i beats j
    pair = {}
    for i in range(r):
        for j in range(i + 1, r):
            pair[f"{i}<{j}"] = float((losses[:, i] < losses[:, j]).float().mean().item())

    intensity_reports = {}
    for eta in intensities:
        bval = budget_from_intensity(float(eta), costs.tolist())
        feas = feasible_mask_from_budget(costs, torch.full((n,), bval))
        for delta in deltas:
            ora = oracle_best_feasible_route(
                losses, costs, feas, delta_abs=float(delta), delta_rel=0.0
            )
            sel = ora["oracle_route_id"]
            sel_hist = Counter(int(x) for x in sel.tolist())
            sel_cost = costs.gather(0, sel)
            best_loss = ora["oracle_best_feasible_loss"]
            # strict best feasible (delta=0) for regret reference
            strict = oracle_best_feasible_route(
                losses, costs, feas, delta_abs=0.0, delta_rel=0.0
            )
            regret = ora["oracle_selected_loss"] - strict["oracle_best_feasible_loss"]
            key = f"eta={eta:.2f}|delta={delta:.3f}"
            intensity_reports[key] = {
                "budget": bval,
                "route_histogram": {str(k): v for k, v in sorted(sel_hist.items())},
                "entropy": _entropy(sel_hist, n),
                "unique_routes": len(sel_hist),
                "avg_selected_cost": float(sel_cost.mean().item()),
                "avg_stage_count": float(
                    sum(len(routes[int(i)]) for i in sel.tolist()) / n
                ),
                "avg_selected_true_mae": float(ora["oracle_selected_loss"].mean().item()),
                "avg_best_feasible_mae": float(best_loss.mean().item()),
                "mean_tolerance_regret_vs_strict": float(regret.mean().item()),
            }

    # Pareto: mean MAE vs cost
    pareto = [
        {
            "route_id": i,
            "route": routes[i],
            "cost": float(costs[i].item()),
            "mean_mae": float(mean_mae[i]),
            "best_count": int(hist.get(i, 0)),
        }
        for i in range(r)
    ]
    return {
        "path": str(path),
        "split": packed["metadata"].get("split"),
        "n_samples": n,
        "n_records_raw": packed["metadata"].get("n_records"),
        "candidate_routes": routes,
        "route_costs": costs.tolist(),
        "mean_route_mae": mean_mae,
        "unconstrained_best_histogram": {str(k): v for k, v in sorted(hist.items())},
        "unconstrained_best_entropy": _entropy(hist, n),
        "pairwise_preference": pair,
        "intensity_tolerance": intensity_reports,
        "pareto": pareto,
        "checkpoint_hash": packed["metadata"].get("checkpoint_hash"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument(
        "--deltas", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1, 0.2]
    )
    parser.add_argument(
        "--intensities",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = {
        "train": analyze_split(Path(args.train), args.deltas, args.intensities),
        "valid": analyze_split(Path(args.valid), args.deltas, args.intensities),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
