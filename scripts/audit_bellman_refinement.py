#!/usr/bin/env python3
"""Audits for Budgeted Bellman Plan B (cost, dataset, residual, frontier)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.bellman_refinement_dataset import (
    BellmanOOFCache,
    audit_dataset_ordering,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budgeted_bellman_refinement import (
    cost_audit_dict,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="results/planB_bellman_oof_cache")
    args = ap.parse_args()

    audit = cost_audit_dict(12)
    Path("results/planB_bellman_cost_audit.json").write_text(json.dumps(audit, indent=2))
    print("cost additive:", audit.get("additive"))

    ds_audit = {"available": False}
    if Path(args.cache_dir).is_dir() and (Path(args.cache_dir) / "manifest.json").is_file():
        cache = BellmanOOFCache(args.cache_dir)
        ds_audit = {
            "available": True,
            "n_samples": len(cache),
            "fold_counts": cache.fold_counts(),
            "global_return_scale": cache.manifest.get("global_return_scale"),
            "ordering": audit_dataset_ordering(cache, max_check=min(1024, len(cache))),
        }
    Path("results/planB_bellman_dataset_audit.json").write_text(json.dumps(ds_audit, indent=2))

    # frontier comparison from existing stored results if present
    frontier = {"methods": [], "note": "uses existing stored PlanA/Bv2 results when available"}
    for path, name in [
        ("results/planB_v2_policy_eval.json", "PlanB-v2"),
        ("results/pems04_crossfit_accuracy_cost_summary.json", "PlanA_and_fixed"),
        ("results/planB_bellman_valid_eval.json", "Bellman_valid"),
        ("results/planB_bellman_test_eval.json", "Bellman_test"),
    ]:
        if Path(path).is_file():
            frontier["methods"].append({"name": name, "path": path, "available": True})
        else:
            frontier["methods"].append({"name": name, "path": path, "available": False})
    Path("results/planB_bellman_frontier_comparison.json").write_text(json.dumps(frontier, indent=2))
    print("wrote audits")


if __name__ == "__main__":
    main()
