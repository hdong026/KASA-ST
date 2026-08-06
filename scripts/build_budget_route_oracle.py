#!/usr/bin/env python3
"""Build offline oracle route labels for planner imitation.

Implement only — do not run against PEMS in the agent task.
Uses train/val indices only; never test labels for training oracles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--route-cost-file", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--intensities", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--delta", type=float, default=0.0, help="tolerance on best MAE")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import sys
    import torch
    from torch.utils.data import DataLoader

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from easytorch.config import import_config
    from basicts.data import TimeSeriesForecastingDataset
    from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
        budget_from_intensity,
        load_route_costs,
        parse_route,
    )

    cfg = import_config(args.cfg)
    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM)
    state = torch.load(args.checkpoint, map_location="cpu")
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(sd, strict=False)
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    model = model.to(device).eval()

    data_dir = Path(cfg.TRAIN.DATA.DIR if args.split == "train" else cfg.VAL.DATA.DIR)
    h = int(cfg.DATASET_OUTPUT_LEN)
    p = int(cfg.DATASET_INPUT_LEN)
    data_file = data_dir / f"data_in{p}_out{h}.pkl"
    index_file = data_dir / f"index_in{p}_out{h}.pkl"
    ds = TimeSeriesForecastingDataset(
        data_file_path=str(data_file),
        index_file_path=str(index_file),
        mode=args.split,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    routes = list(model.candidate_routes)
    costs = load_route_costs(args.route_cost_file, routes, h, cost_type=str(cfg.MODEL.PARAM.get("route_cost_type", "normalized_static_cost")))

    records = []
    with torch.no_grad():
        for i, (future, history) in enumerate(loader):
            if args.max_samples is not None and i >= args.max_samples:
                break
            history = history.to(device)
            future = future.to(device)
            y = future[..., : model.output_dim]
            route_losses = []
            for rid, route in enumerate(routes):
                model.set_forced_route(route)
                model.route_selection_mode = "forced"
                pred = model(history_data=history, train=False, return_all=False)
                mae = (pred - y).abs().mean().item()
                route_losses.append({"route_id": rid, "route": route, "final_mae": mae, "cost": costs[rid]})
            for eta in args.intensities:
                bud = budget_from_intensity(eta, costs)
                feasible = [e for e in route_losses if e["cost"] <= bud + 1e-8]
                if not feasible:
                    feasible = [min(route_losses, key=lambda e: e["cost"])]
                best = min(feasible, key=lambda e: e["final_mae"])
                # tolerance: cheapest among near-best
                near = [e for e in feasible if e["final_mae"] <= best["final_mae"] + args.delta]
                oracle = min(near, key=lambda e: e["cost"])
                records.append(
                    {
                        "sample_index": i,
                        "split": args.split,
                        "intensity": eta,
                        "budget": bud,
                        "feasible_routes": [e["route"] for e in feasible],
                        "route_final_losses": route_losses,
                        "oracle_route_id": oracle["route_id"],
                        "oracle_route": oracle["route"],
                        "oracle_route_cost": oracle["cost"],
                        "best_route_loss": best["final_mae"],
                    }
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"records": records, "routes": routes, "costs": costs}, indent=2), encoding="utf-8")
    print("wrote", out, "n=", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
