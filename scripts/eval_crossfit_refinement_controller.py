#!/usr/bin/env python3
"""Evaluate cross-fitted Plan A refinement controller (VALID/TEST via BasicTS path).

TEST must only be used for final evaluation — never for training or selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller-checkpoint", required=True)
    p.add_argument("--supernet-checkpoint", required=True)
    p.add_argument("--oracle", required=True, help="valid oracle for selection metrics")
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--forbid-test-oracle",
        action="store_true",
        default=True,
        help="refuse if oracle metadata says split=test",
    )
    args = p.parse_args()

    if args.split == "test":
        print(
            "[warn] TEST split: final forecasting metrics only. "
            "Do not build/tune on test route oracle."
        )

    # Reuse existing eval script by argv rewrite
    sys.argv = [
        "eval_forecast_refinement_controller.py",
        "--controller-checkpoint",
        args.controller_checkpoint,
        "--supernet-checkpoint",
        args.supernet_checkpoint,
        "--oracle",
        args.oracle,
        "--device",
        args.device,
    ]
    try:
        from scripts.eval_forecast_refinement_controller import main as base_main
    except Exception:
        # fallback if module path differs
        import runpy

        return runpy.run_path(
            str(ROOT / "scripts/eval_forecast_refinement_controller.py"),
            run_name="__main__",
        )
    return base_main()


if __name__ == "__main__":
    raise SystemExit(main())
