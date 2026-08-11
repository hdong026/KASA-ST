#!/usr/bin/env python3
"""Compare in-sample TRAIN vs temporal holdout vs official VALID gain statistics.

Plan A pilot analysis. Prints CROSS_FIT HYPOTHESIS SUPPORTED / NOT SUPPORTED.
Does not auto-start full cross-fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    build_refinement_route_index_map,
    gains_from_route_losses,
)
from basicts.data.route_quality_dataset import dedupe_route_loss_records, load_oracle_json


def _gain_stats(g: torch.Tensor) -> dict:
    return {
        "mean": float(g.mean().item()),
        "std": float(g.std(unbiased=False).item()),
        "median": float(g.median().item()),
        "min": float(g.min().item()),
        "max": float(g.max().item()),
        "positive_rate": float((g > 0).float().mean().item()),
        "negative_rate": float((g < 0).float().mean().item()),
    }


def _extract_gains(oracle_path: str | Path, horizon: int = 12) -> dict:
    oracle = load_oracle_json(oracle_path)
    # holdout oracle may already store G3/G6/G36 on records
    records = oracle.get("records", [])
    if records and "G3" in records[0]:
        g3 = torch.tensor([float(r["G3"]) for r in records])
        g6 = torch.tensor([float(r["G6"]) for r in records])
        g36 = torch.tensor([float(r["G36"]) for r in records])
        routes = oracle["metadata"].get("candidate_routes")
        return {
            "n": len(records),
            "G3": _gain_stats(g3),
            "G6": _gain_stats(g6),
            "G36": _gain_stats(g36),
            "gains": torch.stack([g3, g6, g36], dim=-1),
            "routes": routes,
        }

    packed = dedupe_route_loss_records(oracle)
    routes = packed["candidate_routes"]
    index_map = build_refinement_route_index_map(routes, horizon)
    g3, g6, g36 = [], [], []
    for si in packed["sample_indices"]:
        losses = packed["route_losses"][si]
        by_name = {
            "direct": losses[index_map["direct"]],
            "half": losses[index_map["half"]],
            "quarter": losses[index_map["quarter"]],
            "progressive": losses[index_map["progressive"]],
        }
        g = gains_from_route_losses(by_name)
        g3.append(g["g3"])
        g6.append(g["g6"])
        g36.append(g["g36"])
    g3_t = torch.tensor(g3)
    g6_t = torch.tensor(g6)
    g36_t = torch.tensor(g36)
    return {
        "n": len(g3),
        "G3": _gain_stats(g3_t),
        "G6": _gain_stats(g6_t),
        "G36": _gain_stats(g36_t),
        "gains": torch.stack([g3_t, g6_t, g36_t], dim=-1),
        "routes": routes,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-oracle", required=True)
    p.add_argument("--holdout-oracle", required=True)
    p.add_argument("--valid-oracle", required=True)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--out", default="results/pems04_temporal_holdout_refinement_analysis.json")
    args = p.parse_args()

    train = _extract_gains(args.train_oracle, args.horizon)
    hold = _extract_gains(args.holdout_oracle, args.horizon)
    valid = _extract_gains(args.valid_oracle, args.horizon)

    table = {}
    distances = {}
    for name in ("G3", "G6", "G36"):
        table[name] = {
            "train_in_sample_mean": train[name]["mean"],
            "temporal_holdout_mean": hold[name]["mean"],
            "valid_mean": valid[name]["mean"],
            "train_pos": train[name]["positive_rate"],
            "holdout_pos": hold[name]["positive_rate"],
            "valid_pos": valid[name]["positive_rate"],
        }
        d_hv = abs(hold[name]["mean"] - valid[name]["mean"])
        d_tv = abs(train[name]["mean"] - valid[name]["mean"])
        distances[name] = {
            "abs_holdout_minus_valid": d_hv,
            "abs_train_minus_valid": d_tv,
            "holdout_closer": d_hv < d_tv,
        }

    closer_count = sum(1 for k in distances if distances[k]["holdout_closer"])
    supported = closer_count >= 2  # majority of gain means closer to valid
    verdict = (
        "CROSS_FIT HYPOTHESIS SUPPORTED"
        if supported
        else "CROSS_FIT HYPOTHESIS NOT SUPPORTED"
    )

    report = {
        "table": table,
        "distances": distances,
        "closer_count": closer_count,
        "verdict": verdict,
        "n_train": train["n"],
        "n_holdout": hold["n"],
        "n_valid": valid["n"],
        "detail": {
            "train": {k: train[k] for k in ("G3", "G6", "G36")},
            "holdout": {k: hold[k] for k in ("G3", "G6", "G36")},
            "valid": {k: valid[k] for k in ("G3", "G6", "G36")},
        },
        "note": "Do not auto-start full cross-fitting from this script.",
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== Plan A Pilot: mean gains ===")
    print(f"{'':18} {'train-in-sample':>16} {'temporal-holdout':>16} {'valid':>12}")
    for name in ("G3", "G6", "G36"):
        print(
            f"mean {name:12} "
            f"{train[name]['mean']:16.4f} "
            f"{hold[name]['mean']:16.4f} "
            f"{valid[name]['mean']:12.4f}"
        )
    print("\n=== positive rates ===")
    for name in ("G3", "G6", "G36"):
        print(
            f"pos% {name:12} "
            f"{100*train[name]['positive_rate']:15.2f}% "
            f"{100*hold[name]['positive_rate']:15.2f}% "
            f"{100*valid[name]['positive_rate']:11.2f}%"
        )
    print("\n=== distances to valid ===")
    for name, d in distances.items():
        print(
            f"{name}: |holdout-valid|={d['abs_holdout_minus_valid']:.4f} "
            f"|train-valid|={d['abs_train_minus_valid']:.4f} "
            f"holdout_closer={d['holdout_closer']}"
        )
    print("\n" + verdict)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
