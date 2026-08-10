#!/usr/bin/env python3
"""Evaluate Route Quality Estimator + frozen F2F supernet over intensity sweeps."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_conditioned_route_quality_f2f import (
    BudgetConditionedRouteQualityF2FNet,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    default_candidate_routes,
    parse_candidate_routes,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    select_batch_route_from_quality,
    select_route_ids_from_quality,
)
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from scripts.train_budget_route_quality_estimator import _build_model


@torch.no_grad()
def feature_ablation(model, histories: torch.Tensor, routes, costs, eta: float):
    """Evaluation-only: normal / permute / zero / route-prior-only."""
    out = {}
    q = model.estimate_route_quality(histories)["predicted_route_losses"]
    out["normal"] = select_route_ids_from_quality(q, costs, eta)["selected_route_id"]

    perm = histories[torch.randperm(histories.shape[0], device=histories.device)]
    qp = model.estimate_route_quality(perm)["predicted_route_losses"]
    out["permuted"] = select_route_ids_from_quality(qp, costs, eta)["selected_route_id"]

    qz = model.estimate_route_quality(torch.zeros_like(histories))["predicted_route_losses"]
    out["zeroed"] = select_route_ids_from_quality(qz, costs, eta)["selected_route_id"]

    # Prior-only: ignore history embedding by using batch-mean predicted losses
    prior = q.mean(dim=0, keepdim=True).expand_as(q)
    out["route_prior"] = select_route_ids_from_quality(prior, costs, eta)["selected_route_id"]
    return {k: v.cpu().tolist() for k, v in out.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--supernet-checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--data-dir", default="datasets/PEMS04")
    parser.add_argument("--candidate-routes", nargs="+", default=None)
    parser.add_argument("--route-cost-source", default="static")
    parser.add_argument("--route-cost-file", default=None)
    parser.add_argument(
        "--intensities", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument(
        "--route-granularity", default="sample", choices=["sample", "batch"]
    )
    parser.add_argument("--delta-abs", type=float, default=0.05)
    parser.add_argument("--delta-rel", type=float, default=0.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--spatial-query-count", type=int, default=4)
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--profile-batches", type=int, default=5)
    parser.add_argument("--feature-ablation", action="store_true")
    parser.add_argument("--out", default="results/pems04_budget_route_quality_eval.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    routes = (
        parse_candidate_routes(args.candidate_routes, args.horizon)
        if args.candidate_routes
        else default_candidate_routes(args.horizon)
    )
    # Minimal namespace compatible with _build_model
    class NS:
        pass

    ns = NS()
    ns.cfg = args.cfg
    ns.horizon = args.horizon
    ns.route_cost_source = args.route_cost_source
    ns.route_cost_file = args.route_cost_file
    ns.delta_abs = args.delta_abs
    ns.delta_rel = args.delta_rel
    ns.d_model = args.d_model
    ns.temporal_layers = args.temporal_layers
    ns.spatial_query_count = args.spatial_query_count
    model = _build_model(ns, routes, device)
    model.route_granularity = args.route_granularity
    model.route_selection_mode = args.route_granularity
    model.set_training_phase("eval")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    if args.supernet_checkpoint:
        s2 = torch.load(args.supernet_checkpoint, map_location="cpu")
        if isinstance(s2, dict) and "model_state_dict" in s2:
            s2 = s2["model_state_dict"]
        model.load_state_dict(s2, strict=False)
    model.freeze_backbone(True)
    model.eval()

    data_file = str(Path(args.data_dir) / f"data_in12_out{args.horizon}.pkl")
    index_file = str(Path(args.data_dir) / f"index_in12_out{args.horizon}.pkl")
    ds = IndexedTimeSeriesForecastingDataset(data_file, index_file, args.split)
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=2)

    report = {
        "split": args.split,
        "route_granularity": args.route_granularity,
        "per_eta": {},
        "profiling": {},
        "feature_ablation": None,
    }

    # Profiling estimator vs route execution on a few batches
    est_ms = []
    exec_ms = []
    total_ms = []
    n_prof = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            _fut, history, _si = batch
        else:
            _fut, history = batch
        history = history.to(device)
        t0 = time.perf_counter()
        q = model.estimate_route_quality(history)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        dec = select_route_ids_from_quality(
            q["predicted_route_losses"],
            model.route_costs,
            0.5,
            delta_abs=float(args.delta_abs),
            delta_rel=float(args.delta_rel),
        )
        _ = model._execute_routes_bucketed(history, dec["selected_route_id"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        est_ms.append((t1 - t0) * 1000.0)
        exec_ms.append((t2 - t1) * 1000.0)
        total_ms.append((t2 - t0) * 1000.0)
        n_prof += 1
        if n_prof >= int(args.profile_batches):
            break
    report["profiling"] = {
        "estimator_latency_ms_mean": float(sum(est_ms) / max(len(est_ms), 1)),
        "route_execution_latency_ms_mean": float(sum(exec_ms) / max(len(exec_ms), 1)),
        "total_adaptive_latency_ms_mean": float(sum(total_ms) / max(len(total_ms), 1)),
        "note": "total = quality estimator + selected F2F route execution",
        "estimator_params": model.route_quality_estimator.count_parameters(),
    }

    for eta in args.intensities:
        hist = Counter()
        costs = []
        stages = []
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                _fut, history, _si = batch
            else:
                _fut, history = batch
            history = history.to(device)
            pred = model.estimate_route_quality(history)["predicted_route_losses"]
            if args.route_granularity == "batch":
                dec = select_batch_route_from_quality(
                    pred,
                    model.route_costs,
                    float(eta),
                    delta_abs=float(args.delta_abs),
                    delta_rel=float(args.delta_rel),
                )
            else:
                dec = select_route_ids_from_quality(
                    pred,
                    model.route_costs,
                    float(eta),
                    delta_abs=float(args.delta_abs),
                    delta_rel=float(args.delta_rel),
                )
            sel = dec["selected_route_id"]
            # Execute to obtain executed_route_id (must match proposed under this decision)
            executed = model._execute_routes_bucketed(history, sel)
            exec_ids = executed["executed_route_id"]
            hist.update(int(i) for i in exec_ids.tolist())
            costs.extend(float(x) for x in executed["selected_cost"].tolist())
            stages.extend(len(model.candidate_routes[int(i)]) for i in exec_ids.tolist())
        n = sum(hist.values())
        ent = 0.0
        for c in hist.values():
            p = c / max(n, 1)
            ent -= p * math.log(p + 1e-12)
        warn = ""
        if float(eta) in {0.5, 0.75, 1.0} and len(hist) <= 1:
            warn = "ROUTE COLLAPSE WARNING"
            print(f"[ROUTE COLLAPSE WARNING] eta={eta} hist={dict(hist)}")
        report["per_eta"][str(eta)] = {
            "route_histogram_executed": {str(k): v for k, v in sorted(hist.items())},
            "entropy": ent,
            "unique_routes": len(hist),
            "avg_selected_cost": float(sum(costs) / max(len(costs), 1)),
            "avg_stage_count": float(sum(stages) / max(len(stages), 1)),
            "warning": warn,
        }

    if args.feature_ablation:
        # One batch diagnostic
        batch = next(iter(loader))
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            _fut, history, _si = batch
        else:
            _fut, history = batch
        report["feature_ablation"] = feature_ablation(
            model, history.to(device), routes, model.route_costs, eta=0.5
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["profiling"], indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
