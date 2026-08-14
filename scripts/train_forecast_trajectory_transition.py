#!/usr/bin/env python3
"""Train the shared ForecastTrajectory transition model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.forecast_trajectory_runtime import (
    ckpt_dir,
    seed_all,
    train_transition,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--acceptance-1epoch", action="store_true")
    args = p.parse_args()
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tag = "acceptance" if args.acceptance_1epoch else "formal"
    cdir = ckpt_dir(args.seed, tag)
    train_transition(
        device=device,
        seed=args.seed,
        epochs=1 if args.acceptance_1epoch else args.epochs,
        batch_size=16 if args.acceptance_1epoch else args.batch_size,
        train_indices=list(range(1024)) if args.acceptance_1epoch else None,
        valid_indices=list(range(256)) if args.acceptance_1epoch else None,
        test_indices=list(range(256)) if args.acceptance_1epoch else None,
        out_ckpt=cdir / "transition_best.pt",
        history_json=ROOT / "results" / "forecast_trajectory_transition_history.json",
        acceptance=args.acceptance_1epoch,
        auto_extend=not args.acceptance_1epoch,
        eval_test=args.acceptance_1epoch,
    )


if __name__ == "__main__":
    main()
