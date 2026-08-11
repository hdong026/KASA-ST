#!/usr/bin/env python3
"""Smoke-capable / full-run-locked holdout supernet training wrapper (Plan A).

Uses Indexed subset of official TRAIN via temporal holdout manifest.
Does NOT invent a new forecasting engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_conditioned_adaptive_f2f import (
    BudgetConditionedAdaptiveF2FNet,
)
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from scripts.budget_f2f_synth_kwargs import synthetic_budget_f2f_kwargs


def _require_safety(args) -> None:
    if args.smoke_test:
        return
    if not args.confirm_full_run:
        raise RuntimeError(
            "Full training is disabled. Pass --confirm-full-run manually "
            "(or --smoke-test for a non-scientific code path)."
        )


def _build_pems_kwargs(horizon: int = 12) -> dict:
    # Match stable supernet geometry (not synth).
    return {
        "node_size": 307,
        "input_len": 12,
        "output_len": horizon,
        "input_dim": 4,
        "output_dim": 1,
        "patch_len": 3,
        "stride": 4,
        "td_size": 288,
        "dw_size": 7,
        "d_td": 32,
        "d_dw": 32,
        "d_d": 32,
        "d_spa": 32,
        "num_layer": 2,
        "if_time_in_day": True,
        "if_day_in_week": True,
        "if_spatial": True,
        "spatial_scheme": "C",
        "adj_mx_path": "datasets/PEMS04/adj_mx.pkl",
        "use_gcn": True,
        "gcn_hidden_dim": 64,
        "use_dynamic_spatial": True,
        "dyn_hidden_dim": 64,
        "dyn_topk": 20,
        "use_adaptive_adj": True,
        "adp_hidden_dim": 32,
        "adp_topk": 20,
        "use_hybrid_graph": True,
        "hybrid_alpha": 0.2,
        "use_patch_branch": True,
        "use_downsample_branch": True,
        "use_linear_residual_branch": True,
        "patch_embedding_mode": "serial_concat",
        "patch_data_input_mode": "all",
        "post_spatial_mode": "adaptive_only",
        "spatial_placement": "interleaved_progressive",
        "use_prev_condition": True,
        "progressive_spatial_ratios": [0.25, 0.5, 1.0],
        "progressive_spatial_topks": [8, 16, 32],
        "progressive_spatial_alphas": [0.03, 0.06, 0.10],
        "use_forecast_state_adapter": True,
        "forecast_state_adapter_mode": "condition_only",
        "forecast_state_adapter_hidden_dim": 16,
        "forecast_state_adapter_epsilon": 0.02,
        "candidate_routes": [[12], [6, 12], [3, 12], [3, 6, 12]],
        "training_phase": "supernet",
        "route_sampling": "sandwich",
        "route_selection_mode": "batch",
        "loss_mode": "dynamic_fair",
        "freeze_forecasting_backbone": False,
        "dataset_name": "PEMS04",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--confirm-full-run", action="store_true")
    p.add_argument("--out", default="checkpoints/PEMS04/H12/budget_f2f/holdout_supernet.pt")
    args = p.parse_args()
    _require_safety(args)

    manifest = json.loads(Path(args.manifest).read_text())
    idxs = list(manifest["supernet_train_samples"])
    if args.smoke_test:
        idxs = idxs[:64]
        args.batch_size = min(args.batch_size, 16)
        max_batches = 2
        out = Path("/tmp/kasa_planA_smoke.pt")
        use_synth = True  # tiny CPU-friendly architecture for smoke
    else:
        max_batches = None
        out = Path(args.out)
        use_synth = False

    data_file = "datasets/PEMS04/data_in12_out12.pkl"
    index_file = "datasets/PEMS04/index_in12_out12.pkl"
    base = IndexedTimeSeriesForecastingDataset(data_file, index_file, "train")
    ds = Subset(base, idxs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device(args.device)
    if use_synth:
        kw = synthetic_budget_f2f_kwargs(
            node_size=7, training_phase="supernet", route_selection_mode="forced",
            forced_route=[3, 6, 12],
        )
        kw["use_gcn"] = False
        kw["use_dynamic_spatial"] = False
        kw["freeze_forecasting_backbone"] = False
        # smoke: random tiny tensors instead of PEMS batch shapes
        model = BudgetConditionedAdaptiveF2FNet(**kw).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.train()
        losses = []
        for step in range(max_batches):
            h = torch.randn(4, 12, 7, 4, device=device)
            y = torch.randn(4, 12, 7, 1, device=device)
            out_d = model(history_data=h, train=True, return_all=True)
            pred = out_d["pred"]
            loss = (pred - y).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().item()))
            print(f"[smoke A2] step={step} loss={losses[-1]:.4f}")
        torch.save({"model_state_dict": model.state_dict(), "smoke": True}, out)
        print("SMOKE TEST ONLY - NOT A SCIENTIFIC RESULT")
        print(f"Wrote {out}")
        return 0

    # Full path: build real model (user must pass --confirm-full-run)
    kw = _build_pems_kwargs()
    model = BudgetConditionedAdaptiveF2FNet(**kw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(
        "[full] This wrapper only demonstrates subset DataLoader wiring. "
        "Prefer scripts/run_budget_conditioned_f2f.py with a filtered index for "
        "production holdout supernet training."
    )
    # One-epoch skeleton (user should use official runner for real experiments)
    model.train()
    n_done = 0
    for batch in loader:
        future, history, _si = batch
        history = history.to(device)
        future = future.to(device)
        out_d = model(history_data=history, train=True, return_all=True)
        pred = out_d["pred"]
        # target channel 0
        loss = (pred - future[..., :1]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        n_done += 1
        print(f"[full skeleton] batch={n_done} loss={float(loss):.4f}")
        if max_batches is not None and n_done >= max_batches:
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
