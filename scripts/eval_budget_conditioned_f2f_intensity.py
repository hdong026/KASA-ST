#!/usr/bin/env python3
"""Evaluate a budget-conditioned F2F checkpoint across inference intensities.

Does not train. User runs this after joint fine-tuning.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--intensities", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", required=True)
    parser.add_argument("--also-forced-full", action="store_true")
    args = parser.parse_args()

    import sys
    import time

    import torch
    from torch.utils.data import DataLoader

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from easytorch.config import import_config
    from basicts.data import TimeSeriesForecastingDataset
    from basicts.metrics import masked_mae, masked_mape, masked_rmse

    cfg = import_config(args.cfg)
    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM)
    state = torch.load(args.checkpoint, map_location="cpu")
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    model = model.to(device).eval()

    data_dir = Path(cfg.TEST.DATA.DIR if args.split == "test" else cfg.VAL.DATA.DIR)
    h = int(cfg.DATASET_OUTPUT_LEN)
    p = int(cfg.DATASET_INPUT_LEN)
    ds = TimeSeriesForecastingDataset(
        data_file_path=str(data_dir / f"data_in{p}_out{h}.pkl"),
        index_file_path=str(data_dir / f"index_in{p}_out{h}.pkl"),
        mode=args.split,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    def run_once(tag: str, set_forced_full: bool = False) -> dict:
        preds, tgts = [], []
        routes = []
        costs = []
        budgets = []
        stage_counts = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for future, history in loader:
                history = history.to(device)
                future = future.to(device)
                if set_forced_full:
                    model.set_forced_route([h // 4, h // 2, h] if h % 4 == 0 else [h])
                    model.route_selection_mode = "forced"
                else:
                    model.set_forced_route(None)
                    model.route_selection_mode = "batch"
                out = model(history_data=history, train=False, return_all=True)
                y = future[..., : out["pred"].shape[-1]]
                preds.append(out["pred"].cpu())
                tgts.append(y.cpu())
                rid = out["selected_route_id"]
                routes.extend(rid.detach().cpu().tolist())
                costs.append(out["selected_cost"].detach().cpu())
                budgets.append(out["budget"].detach().cpu())
                stage_counts.append(float(len(out["chain_resolutions"])))
                if device.type == "cuda":
                    torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        pred = torch.cat(preds, dim=0)
        tgt = torch.cat(tgts, dim=0)
        cost = torch.cat(costs, dim=0)
        bud = torch.cat(budgets, dim=0)
        hist = Counter(routes)
        return {
            "tag": tag,
            "mae": float(masked_mae(pred, tgt).item()),
            "rmse": float(masked_rmse(pred, tgt).item()),
            "mape": float(masked_mape(pred, tgt).item()),
            "route_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
            "avg_stage_count": float(sum(stage_counts) / max(len(stage_counts), 1)),
            "avg_selected_cost": float(cost.mean()),
            "budget_violation_rate": float((cost > bud + 1e-8).float().mean()),
            "wall_time_sec": elapsed,
            "n_batches": len(stage_counts),
        }

    rows = []
    for eta in args.intensities:
        model.inference_intensity = float(eta)
        rows.append(run_once(f"eta={eta}"))
    if args.also_forced_full:
        rows.append(run_once("forced_full", set_forced_full=True))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
