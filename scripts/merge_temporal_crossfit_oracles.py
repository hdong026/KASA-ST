#!/usr/bin/env python3
"""Merge per-fold temporal cross-fit oracles into one non-duplicated file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    build_refinement_route_index_map,
    gains_from_route_losses,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold-oracles", nargs="+", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", default="results/pems04_temporal_crossfit_refinement_oracle.json")
    args = p.parse_args()

    merged = {}
    meta_folds = []
    candidate_routes = None
    route_costs = None
    dataset = None
    horizon = None
    for path in args.fold_oracles:
        data = json.loads(Path(path).read_text())
        md = data.get("metadata", {})
        fold = md.get("teacher_fold", md.get("fold"))
        teacher_hash = md.get("teacher_hash") or md.get("checkpoint_hash")
        routes = md.get("candidate_routes")
        if candidate_routes is None:
            candidate_routes = routes
        elif routes is not None and routes != candidate_routes:
            raise RuntimeError(f"route order mismatch in {path}")
        costs = md.get("route_costs")
        if costs is None and data.get("records"):
            # Fallback: recover costs from first record entries.
            costs = [float(e["cost"]) for e in data["records"][0]["route_final_losses"]]
        if route_costs is None:
            route_costs = [float(c) for c in costs]
        elif [float(c) for c in costs] != [float(c) for c in route_costs]:
            raise RuntimeError(f"route_costs mismatch in {path}: {costs} vs {route_costs}")
        if dataset is None:
            dataset = md.get("dataset", "PEMS04")
        elif md.get("dataset") is not None and str(md.get("dataset")) != str(dataset):
            raise RuntimeError(f"dataset mismatch in {path}")
        h = int(md.get("horizon", 12))
        if horizon is None:
            horizon = h
        elif h != horizon:
            raise RuntimeError(f"horizon mismatch in {path}")
        index_map = build_refinement_route_index_map(candidate_routes, horizon)
        for rec in data["records"]:
            si = int(rec["sample_index"])
            if si in merged:
                raise RuntimeError(f"duplicate sample_index {si} across folds")
            losses = rec["route_final_losses"]
            by_name = {
                name: next(
                    float(e["final_mae"])
                    for e in losses
                    if list(e["route"])
                    == list(candidate_routes[index_map[name]])
                )
                for name in ("direct", "half", "quarter", "progressive")
            }
            g = gains_from_route_losses(by_name)
            merged[si] = {
                "sample_index": si,
                "teacher_fold": fold,
                "teacher_checkpoint_hash": teacher_hash,
                "true_route_losses": [float(e["final_mae"]) for e in losses],
                "route_final_losses": losses,
                "G3": g["g3"],
                "G6": g["g6"],
                "G36": g["g36"],
            }
        meta_folds.append({"path": path, "fold": fold, "n": len(data["records"])})

    if not route_costs or len(route_costs) != len(candidate_routes):
        raise RuntimeError(
            f"merged route_costs invalid: costs={route_costs} routes={candidate_routes}"
        )

    records = [merged[k] for k in sorted(merged.keys())]
    out_meta = {
        "dataset": str(dataset or "PEMS04"),
        "horizon": int(horizon or 12),
        "scheme": "rolling_origin_temporal_crossfit",
        "split_type": "temporal_crossfit_merged",
        "loss_scale": "raw_physical_scale",
        "candidate_routes": candidate_routes,
        "route_costs": list(map(float, route_costs)),
        "route_cost_type": "normalized_static_cost",
        "n_samples": len(records),
        "sample_count": len(records),
        "folds": meta_folds,
        "manifest": args.manifest,
        "merged_hash": hashlib.sha1(
            json.dumps([r["sample_index"] for r in records]).encode()
        ).hexdigest()[:16],
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metadata": out_meta, "records": records}, indent=2))
    print(f"merged {len(records)} samples -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
