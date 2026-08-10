#!/usr/bin/env python3
"""Offline analysis of forecast-refinement gains from route-loss oracles (no model)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budget_from_intensity,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_gain_loss import (
    compute_pair_imbalance_weights,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    build_refinement_route_index_map,
    gains_from_route_losses,
    route_scores_from_gains,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    feasible_mask_from_budget,
    oracle_best_feasible_route,
)
from basicts.data.route_quality_dataset import dedupe_route_loss_records, load_oracle_json


def _stats(x: torch.Tensor) -> dict:
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "median": float(x.median().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "positive_rate": float((x > 0).float().mean().item()),
        "negative_rate": float((x < 0).float().mean().item()),
    }


def analyze_split(path: Path, delta_abs: float) -> dict:
    oracle = load_oracle_json(path)
    packed = dedupe_route_loss_records(oracle)
    routes = packed["candidate_routes"]
    costs = torch.tensor(packed["route_costs"], dtype=torch.float32)
    index_map = build_refinement_route_index_map(routes, int(packed["metadata"]["horizon"]))
    losses = torch.tensor(
        [packed["route_losses"][i] for i in packed["sample_indices"]], dtype=torch.float32
    )
    g3, g6, g36 = [], [], []
    for row in losses:
        by = {
            "direct": float(row[index_map["direct"]]),
            "half": float(row[index_map["half"]]),
            "quarter": float(row[index_map["quarter"]]),
            "progressive": float(row[index_map["progressive"]]),
        }
        g = gains_from_route_losses(by)
        g3.append(g["g3"])
        g6.append(g["g6"])
        g36.append(g["g36"])
    g3_t = torch.tensor(g3)
    g6_t = torch.tensor(g6)
    g36_t = torch.tensor(g36)
    gains = torch.stack([g3_t, g6_t, g36_t], dim=-1)
    corr = torch.corrcoef(gains.T)
    scores = route_scores_from_gains(
        g3_t, g6_t, g36_t, index_map=index_map, n_routes=len(routes)
    )
    pair_w, pair_report = compute_pair_imbalance_weights(scores)

    # Route-prior baseline: mean train gains applied to every sample (caller may pass)
    mean_gains = gains.mean(dim=0)
    prior_scores = route_scores_from_gains(
        mean_gains[0].expand(losses.shape[0]),
        mean_gains[1].expand(losses.shape[0]),
        mean_gains[2].expand(losses.shape[0]),
        index_map=index_map,
        n_routes=len(routes),
    )

    intensities = [0.0, 0.25, 0.5, 0.75, 1.0]
    eta_reports = {}
    for eta in intensities:
        bval = budget_from_intensity(eta, costs.tolist())
        feas = feasible_mask_from_budget(costs, torch.full((losses.shape[0],), bval))
        # Prior decision via score tolerance
        from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_decision import (
            select_routes_from_scores,
        )

        prior_dec = select_routes_from_scores(
            prior_scores, costs, float(eta), delta_abs=float(delta_abs)
        )
        strict = oracle_best_feasible_route(losses, costs, feas, delta_abs=0.0)
        tol = oracle_best_feasible_route(
            losses, costs, feas, delta_abs=float(delta_abs), delta_rel=0.0
        )
        sel = prior_dec["selected_route_id"]
        sel_true = losses.gather(1, sel.unsqueeze(1)).squeeze(1)
        regret = sel_true - strict["oracle_best_feasible_loss"]
        hist = Counter(int(x) for x in sel.tolist())
        eta_reports[str(eta)] = {
            "budget": bval,
            "prior_route_histogram": {str(k): v for k, v in sorted(hist.items())},
            "prior_mean_regret": float(regret.mean().item()),
            "prior_avg_cost": float(prior_dec["selected_cost"].mean().item()),
            "strict_oracle_mae": float(strict["oracle_best_feasible_loss"].mean().item()),
            "tolerance_oracle_mae": float(tol["oracle_selected_loss"].mean().item()),
            "tolerance_oracle_cost": float(
                costs.gather(0, tol["oracle_route_id"]).mean().item()
            ),
        }

    return {
        "path": str(path),
        "split": packed["metadata"].get("split"),
        "n_samples": int(losses.shape[0]),
        "target_scale": "raw_physical_mae_gain",
        "index_map": index_map,
        "candidate_routes": routes,
        "route_costs": costs.tolist(),
        "G3": _stats(g3_t),
        "G6": _stats(g6_t),
        "G36": _stats(g36_t),
        "gain_correlation_matrix": corr.tolist(),
        "sign_consistency": {
            "g3_pos": float((g3_t > 0).float().mean().item()),
            "g6_pos": float((g6_t > 0).float().mean().item()),
            "g36_pos": float((g36_t > 0).float().mean().item()),
        },
        "pair_imbalance_weights": pair_report,
        "mean_gains": mean_gains.tolist(),
        "eta_reports": eta_reports,
        "checkpoint_hash": packed["metadata"].get("checkpoint_hash"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--delta-abs", type=float, default=0.05)
    parser.add_argument("--out", default="results/pems04_forecast_refinement_targets.json")
    args = parser.parse_args()

    train = analyze_split(Path(args.train), args.delta_abs)
    valid = analyze_split(Path(args.valid), args.delta_abs)
    shift = {
        k: {
            "mean_shift": valid[k]["mean"] - train[k]["mean"],
            "std_shift": valid[k]["std"] - train[k]["std"],
        }
        for k in ("G3", "G6", "G36")
    }
    report = {
        "train": train,
        "valid": valid,
        "train_valid_gain_shift": shift,
        "notes": {
            "route_prior_baseline": "eta_reports.*.prior_mean_regret uses train/valid mean gains within split",
            "delta_abs": float(args.delta_abs),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train_G": {k: train[k] for k in ("G3", "G6", "G36")},
                      "valid_G": {k: valid[k] for k in ("G3", "G6", "G36")},
                      "shift": shift,
                      "train_pair_weights": train["pair_imbalance_weights"],
                      "valid_eta0.75_prior_regret": valid["eta_reports"]["0.75"]["prior_mean_regret"],
                      }, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
