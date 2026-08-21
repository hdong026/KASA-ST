#!/usr/bin/env python3
"""Aggregate frozen finalized-router VALID/TEST reports across formal seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROUTES = ("12", "2-12", "2-4-12", "2-6-12", "3-12", "3-6-12", "4-12", "6-12")
SCALAR_KEYS = (
    "policy_MAE",
    "matched_fixed_mixture_MAE",
    "gain_over_matched_fixed_mixture",
    "mean_actual_FLOPs",
    "mean_latency_ms",
    "budget_oracle_MAE",
    "budget_oracle_headroom",
)


def mean_std(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def aggregate_reports(paths: list[Path]) -> dict[str, object]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    splits = sorted({str(report["evaluation_split"]) for report in reports})
    output: dict[str, object] = {
        "method": "FullDAGRichCounterfactualQualityAnchoredGRPO_RLOO",
        "uses_TEST_for_selection": False,
        "seed_reports": [str(path) for path in paths],
        "seeds": [int(report["evaluation_seed"]) for report in reports],
        "splits": splits,
        "aggregates": {},
    }
    for split in splits:
        split_reports = [report for report in reports if report["evaluation_split"] == split]
        budgets = sorted({budget for report in split_reports for budget in report["budgets"]}, key=float)
        split_payload: dict[str, object] = {"seed_count": len(split_reports), "budgets": {}}
        for budget in budgets:
            rows = [report["budgets"][budget] for report in split_reports if budget in report["budgets"]]
            scalar = {key: mean_std([float(row[key]) for row in rows]) for key in SCALAR_KEYS}
            route_share = {
                route: mean_std([float(row["route_share"].get(route, 0.0)) for row in rows])
                for route in ROUTES
            }
            split_payload["budgets"][budget] = {
                "metrics": scalar,
                "route_share": route_share,
            }
        output["aggregates"][split] = split_payload
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    missing = [str(path) for path in args.reports if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing reports: {missing}")
    payload = aggregate_reports(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[done] aggregate={args.output}")


if __name__ == "__main__":
    main()
