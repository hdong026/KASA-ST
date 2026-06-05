#!/usr/bin/env python3
"""Load baseline PeMS04 configs and instantiate models (no training)."""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE_DIR = ROOT / "examples" / "baselines"


def load_cfg(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CFG


def main():
    configs = sorted(BASELINE_DIR.glob("*/*_PEMS04.py"))
    if not configs:
        print("No baseline configs found under examples/baselines/")
        return 1

    print(f"{'Model':<14} {'Status':<8} Detail")
    print("-" * 72)
    failed = 0
    for cfg_path in configs:
        model = cfg_path.parent.name
        try:
            cfg = load_cfg(cfg_path)
            arch = cfg.MODEL.ARCH
            param = dict(cfg.MODEL.PARAM)
            _ = arch(**param)
            print(f"{model:<14} {'OK':<8}")
        except Exception as e:
            failed += 1
            msg = f"{type(e).__name__}: {e}"
            print(f"{model:<14} {'FAIL':<8} {msg}")
            traceback.print_exc()
            print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
