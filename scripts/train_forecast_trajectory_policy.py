#!/usr/bin/env python3
"""Train the exact-expectation online trajectory policy (transition frozen)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from scripts.forecast_trajectory_runtime import (
    INDEX_FILE,
    build_model,
    chronological_policy_split,
    ckpt_dir,
    run_dir,
    seed_all,
    train_policy,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_cache import TrajectoryCache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    ckpt = ckpt_dir(args.seed, "formal") / "transition_best.pt"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(blob["state_dict"])
    cache_dir = Path(args.cache_dir) if args.cache_dir else run_dir(args.seed, "formal") / "train_cache"
    cache = TrajectoryCache(cache_dir)
    train_ds = IndexedTimeSeriesForecastingDataset(
        str(ROOT / "datasets/PEMS04/data_in12_out12.pkl"), str(INDEX_FILE), "train"
    )
    split = chronological_policy_split(cache.sample_indices(), train_ds.index, 0.8)
    lat = json.loads((ROOT / "results" / "forecast_trajectory_latency_table.json").read_text())
    oracle = json.loads((ROOT / "results" / "forecast_trajectory_oracle_analysis.json").read_text())
    train_policy(
        model=model,
        cache=cache,
        split=split,
        device=device,
        latency_table=lat,
        oracle=oracle,
        out_ckpt=ckpt_dir(args.seed, "formal") / "policy_best.pt",
        history_json=ROOT / "results" / "forecast_trajectory_policy_history.json",
        acceptance=False,
    )


if __name__ == "__main__":
    main()
