#!/usr/bin/env python3
"""Deprecated thin wrapper — use run_matched_maturity_crossfit_pipeline.py --smoke."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_matched_maturity_crossfit_pipeline import run_engineering_smoke, write_preflight_v2


def main() -> int:
    gpu = "1"
    if len(sys.argv) > 1:
        gpu = sys.argv[1]
    rep = run_engineering_smoke(gpu=gpu, seed=1)
    write_preflight_v2(rep)
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
