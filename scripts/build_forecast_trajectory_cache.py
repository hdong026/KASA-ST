#!/usr/bin/env python3
"""Build ForecastTrajectory TRAIN/VALID trajectory + prefix cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.forecast_trajectory_runtime import (
    build_latency_table,
    build_model,
    build_trajectory_cache,
    ckpt_dir,
    dump_json,
    load_scaler,
    make_loader,
    run_dir,
    seed_all,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--split", choices=["train", "valid"], default="train")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--ckpt", default=None)
    args = p.parse_args()
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    ckpt = Path(args.ckpt) if args.ckpt else ckpt_dir(args.seed, "formal") / "transition_best.pt"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(blob["state_dict"])
    _, rescale = load_scaler()
    loader, _ = make_loader(args.split, None, 8, False)
    lat_path = ROOT / "results" / "forecast_trajectory_latency_table.json"
    if lat_path.is_file():
        lat = json.loads(lat_path.read_text())
    else:
        lat = build_latency_table(model, loader, device, warmup=50, iters=200)
        dump_json(lat_path, lat)
    out = run_dir(args.seed, "formal") / f"{args.split}_cache"
    man = build_trajectory_cache(
        model, loader, device, rescale, lat, out, max_samples=args.max_samples
    )
    dump_json(ROOT / "results" / f"forecast_trajectory_{args.split}_cache_manifest.json", man)


if __name__ == "__main__":
    main()
