#!/usr/bin/env python3
"""Evaluate frozen ForecastTrajectory transition + policy (VALID or TEST)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ForecastTrajectory_arch.online_trajectory_policy import (
    OnlineTrajectoryPolicy,
)
from scripts.forecast_trajectory_runtime import (
    VALIDATION_PANEL,
    build_model,
    ckpt_dir,
    dump_json,
    evaluate_trajectories,
    load_scaler,
    make_loader,
    run_online_policy,
    seed_all,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--split", choices=["valid", "test"], default="valid")
    args = p.parse_args()
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    cdir = ckpt_dir(args.seed, "formal")
    tblob = torch.load(cdir / "transition_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(tblob["state_dict"])
    policy = OnlineTrajectoryPolicy(model.graph, d_history=model.d_model).to(device)
    pblob = torch.load(cdir / "policy_best.pt", map_location=device, weights_only=False)
    policy.load_state_dict(pblob["state_dict"])
    _, rescale = load_scaler()
    lat = json.loads((ROOT / "results" / "forecast_trajectory_latency_table.json").read_text())
    extra = float(lat["lookup"].get("policy_step_median_ms") or 0.0)
    loader, _ = make_loader(args.split, None, 8, False)
    out = {
        "quality_only_lambda0_noB": run_online_policy(
            model, policy, loader, device, rescale, lat, 0.0, None, extra
        ),
        "fixed_baselines": {},
    }
    for tau in VALIDATION_PANEL:
        out["fixed_baselines"][model.graph.trajectory_key(tau)] = evaluate_trajectories(
            model, loader, [tau], device, rescale
        )
    dump_json(ROOT / "results" / f"forecast_trajectory_{args.split}_eval.json", out)


if __name__ == "__main__":
    main()
