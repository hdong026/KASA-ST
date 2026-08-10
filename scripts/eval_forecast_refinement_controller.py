#!/usr/bin/env python3
"""Evaluate Adaptive Forecast Refinement Route Controller + frozen supernet."""

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

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    default_candidate_routes,
    parse_candidate_routes,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_decision import (
    select_batch_routes_from_scores,
    select_routes_from_scores,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    route_scores_from_gains,
)
from basicts.data.forecast_refinement_gain_dataset import ForecastRefinementGainDataset
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from basicts.data.route_quality_dataset import dedupe_route_loss_records, load_oracle_json
from scripts.train_forecast_refinement_controller import _build_model, _load_supernet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--supernet-checkpoint", required=True)
    parser.add_argument("--controller-checkpoint", required=True)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--valid-oracle", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--data-dir", default="datasets/PEMS04")
    parser.add_argument("--candidate-routes", nargs="+", default=None)
    parser.add_argument("--route-cost-file", default=None)
    parser.add_argument(
        "--intensities", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--delta-abs", type=float, default=0.05)
    parser.add_argument(
        "--route-granularity", default="sample", choices=["sample", "batch"]
    )
    parser.add_argument("--controller-dim", type=int, default=128)
    parser.add_argument("--pooling-queries", type=int, default=4)
    parser.add_argument("--profile-batches", type=int, default=5)
    parser.add_argument("--out", default="results/pems04_forecast_refinement_eval.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    routes = (
        parse_candidate_routes(args.candidate_routes, args.horizon)
        if args.candidate_routes
        else default_candidate_routes(args.horizon)
    )

    class NS:
        pass

    ns = NS()
    for k, v in vars(args).items():
        setattr(ns, k, v)
    model = _build_model(ns, routes, device)
    _load_supernet(model, Path(args.supernet_checkpoint))
    ckpt = torch.load(args.controller_checkpoint, map_location="cpu")
    # Compatibility checks
    cfg = ckpt.get("controller_config") or {}
    if cfg.get("candidate_routes") and cfg["candidate_routes"] != routes:
        raise RuntimeError(
            f"candidate_routes mismatch: ckpt={cfg['candidate_routes']} vs {routes}"
        )
    if cfg.get("horizon") is not None and int(cfg["horizon"]) != int(args.horizon):
        raise RuntimeError("horizon mismatch with controller checkpoint")
    if "controller_state_dict" in ckpt:
        model.gain_controller.load_state_dict(ckpt["controller_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.set_training_phase("eval")
    model.route_granularity = args.route_granularity
    model.route_selection_mode = args.route_granularity
    model.eval()

    data_file = str(Path(args.data_dir) / f"data_in12_out{args.horizon}.pkl")
    index_file = str(Path(args.data_dir) / f"index_in12_out{args.horizon}.pkl")
    ds = IndexedTimeSeriesForecastingDataset(data_file, index_file, args.split)
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=2)

    # Profiling
    shared_ms, ctrl_ms, route_ms, total_ms = [], [], [], []
    n_prof = 0
    for batch in loader:
        if len(batch) == 3:
            _f, history, _si = batch
        else:
            _f, history = batch
        history = history.to(device)
        t0 = time.perf_counter()
        h_shared = model.extract_pre_route_context(history, detach=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        g = model.gain_controller(h_shared)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        scores = route_scores_from_gains(
            g["g3_hat"], g["g6_hat"], g["g36_hat"],
            index_map=model.index_map, n_routes=len(routes),
        )
        dec = select_routes_from_scores(
            scores, model.route_costs, 0.5, delta_abs=float(args.delta_abs)
        )
        _ = model._execute_routes_bucketed(history, dec["selected_route_id"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        shared_ms.append((t1 - t0) * 1000)
        ctrl_ms.append((t2 - t1) * 1000)
        route_ms.append((t3 - t2) * 1000)
        total_ms.append((t3 - t0) * 1000)
        n_prof += 1
        if n_prof >= int(args.profile_batches):
            break

    report = {
        "split": args.split,
        "profiling": {
            "shared_feature_extraction_ms_mean": float(sum(shared_ms) / max(len(shared_ms), 1)),
            "incremental_controller_latency_ms_mean": float(sum(ctrl_ms) / max(len(ctrl_ms), 1)),
            "selected_route_execution_ms_mean": float(sum(route_ms) / max(len(route_ms), 1)),
            "total_adaptive_ms_mean": float(sum(total_ms) / max(len(total_ms), 1)),
            "note": "shared feature is Priority-B tap of H-stage patch_encoder; not a second backbone",
            "controller_params": model.gain_controller.count_parameters(),
        },
        "per_eta": {},
    }

    oracle_losses = None
    if args.split == "valid" and args.valid_oracle:
        packed = dedupe_route_loss_records(load_oracle_json(args.valid_oracle))
        oracle_losses = packed["route_losses"]

    for eta in args.intensities:
        hist = Counter()
        costs, stages = [], []
        gain_rows = []
        for batch in loader:
            if len(batch) == 3:
                _f, history, sample_index = batch
                sis = sample_index.tolist()
            else:
                _f, history = batch
                sis = None
            history = history.to(device)
            q = model.estimate_refinement_gains(history)
            gain_rows.append(q["predicted_gains"].cpu())
            scores = q["route_scores"]
            if args.route_granularity == "batch":
                dec = select_batch_routes_from_scores(
                    scores, model.route_costs, float(eta), delta_abs=float(args.delta_abs)
                )
            else:
                dec = select_routes_from_scores(
                    scores, model.route_costs, float(eta), delta_abs=float(args.delta_abs)
                )
            executed = model._execute_routes_bucketed(history, dec["selected_route_id"])
            exec_ids = executed["executed_route_id"]
            hist.update(int(i) for i in exec_ids.tolist())
            costs.extend(float(x) for x in executed["selected_cost"].tolist())
            stages.extend(len(model.candidate_routes[int(i)]) for i in exec_ids.tolist())
        gains = torch.cat(gain_rows, dim=0)
        n = sum(hist.values())
        ent = 0.0
        for c in hist.values():
            p = c / max(n, 1)
            ent -= p * math.log(p + 1e-12)
        report["per_eta"][str(eta)] = {
            "route_histogram_executed": {str(k): v for k, v in sorted(hist.items())},
            "entropy": ent,
            "unique_routes": len(hist),
            "avg_selected_cost": float(sum(costs) / max(len(costs), 1)),
            "avg_stage_count": float(sum(stages) / max(len(stages), 1)),
            "predicted_gain_stats": {
                "mean": gains.mean(0).tolist(),
                "std": gains.std(0, unbiased=False).tolist(),
            },
            "note_metrics": "MAE/RMSE/MAPE require BasicTS runner metric path; this script focuses on routing/profile",
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["profiling"], indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
