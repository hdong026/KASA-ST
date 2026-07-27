#!/usr/bin/env python3
"""Thin wrapper: Protocol A horizon runner for PEMS07."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Ensure --dataset is set unless user already passed it.
if "--dataset" not in sys.argv:
    sys.argv.insert(1, "--dataset")
    sys.argv.insert(2, "PEMS07")

TARGET = Path(__file__).resolve().parent / "run_chain_forecasting_horizon.py"
runpy.run_path(str(TARGET), run_name="__main__")
