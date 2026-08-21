#!/usr/bin/env python3
"""TRAIN/VALID all-route oracle decomposition for the complete DAG.

This is a target-using diagnostic only: route losses are counterfactual
forecast errors from the frozen forecaster, never router inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROUTES = ((12,), (2, 12), (2, 4, 12), (2, 6, 12), (3, 12), (3, 6, 12), (4, 12), (6, 12))


def route_name(route):
    return "-".join(map(str, route))


def decompose(path: Path) -> dict[str, object]:
    losses = np.load(path, allow_pickle=False)["mae"].astype(np.float64)
    best = losses.argmin(1)
    long = losses[:, 5]
    oracle = losses.min(1)
    result = {
        "samples": int(len(losses)),
        "route_mean_MAE": {route_name(r): float(losses[:, i].mean()) for i, r in enumerate(ROUTES)},
        "oracle_best_share": {route_name(r): float((best == i).mean()) for i, r in enumerate(ROUTES)},
        "always_long_MAE": float(long.mean()),
        "oracle_MAE": float(oracle.mean()),
        "oracle_gain_beyond_always_long": float((long - oracle).mean()),
        "oracle_gain_quantiles": [float(x) for x in np.quantile(long - oracle, (0.25, 0.50, 0.75, 0.90, 0.95))],
        "oracle_top1_top2_gap_mean": float((np.sort(losses, axis=1)[:, 1] - np.sort(losses, axis=1)[:, 0]).mean()),
        "route_best_conditional_gain_vs_long": {
            route_name(r): float((long[best == i] - losses[best == i, i]).mean()) if bool((best == i).any()) else None
            for i, r in enumerate(ROUTES)
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--valid-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-arrays", type=Path, help="optional VALID npz with losses, choices, and oracle arrays")
    args = parser.parse_args()
    payload = {"method": "FrozenCompleteDAGAllRouteOracleDecomposition", "uses_TEST": False, "TRAIN": decompose(args.train_cache), "VALID": decompose(args.valid_cache)}
    if args.policy_arrays is not None:
        arrays = np.load(args.policy_arrays, allow_pickle=False)
        losses = arrays["losses"].astype(np.float64)
        choices = arrays["choices"].astype(int)
        oracle = losses.argmin(1)
        long = losses[:, 5]
        selected = losses[np.arange(len(losses)), choices]
        payload["current_policy_VALID"] = {
            "route_share": {route_name(r): float((choices == i).mean()) for i, r in enumerate(ROUTES)},
            "oracle_route_match": float((choices == oracle).mean()),
            "policy_MAE": float(selected.mean()),
            "always_long_MAE": float(long.mean()),
            "oracle_MAE": float(losses.min(1).mean()),
            "oracle_gain_recovered_fraction": float((long.mean() - selected.mean()) / max(long.mean() - losses.min(1).mean(), 1e-12)),
            "oracle_to_policy_route_share": {
                route_name(r): {route_name(s): float((choices[oracle == i] == j).mean()) for j, s in enumerate(ROUTES) if bool((choices[oracle == i] == j).any())}
                for i, r in enumerate(ROUTES) if bool((oracle == i).any())
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[done] report={args.output}")


if __name__ == "__main__":
    main()
