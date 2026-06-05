#!/usr/bin/env python3
"""Smoke test for previously failed PeMS04 baselines (no full training)."""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

import torch
from easytorch.config.utils import convert_config
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.data import TimeSeriesForecastingDataset

MODELS = ["DCRNN", "DGCRN", "GWNet", "GTS", "STNorm"]


def load_cfg(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CFG


def get_batch(cfg_dict):
    dn = cfg_dict["DATASET_NAME"]
    il, ol = cfg_dict["DATASET_INPUT_LEN"], cfg_dict["DATASET_OUTPUT_LEN"]
    ds = TimeSeriesForecastingDataset(
        f"datasets/{dn}/data_in{il}_out{ol}.pkl",
        f"datasets/{dn}/index_in{il}_out{ol}.pkl",
        "train",
    )
    return next(iter(DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)))


def main() -> int:
    print(f"{'Model':<10} {'Status':<6} Detail")
    print("-" * 72)
    failed = 0
    for name in MODELS:
        cfg_path = ROOT / "examples" / "baselines" / name / f"{name}_PEMS04.py"
        try:
            raw_cfg = load_cfg(cfg_path)
            cfg = convert_config(raw_cfg)
            runner_cls = cfg["RUNNER"]
            print(f"\n[{name}] runner={runner_cls.__name__} arch={cfg['MODEL']['ARCH'].__name__}")

            model = cfg["MODEL"]["ARCH"](**cfg["MODEL"]["PARAM"])
            print(f"  model instantiated: {type(model).__name__}")

            runner = runner_cls(cfg)
            runner.build_model(cfg)
            print(f"  runner instantiated: {runner_cls.__name__}")

            batch = get_batch(cfg)
            future, history = batch
            print(f"  history_data shape: {tuple(history.shape)}")
            print(f"  future_data shape:  {tuple(future.shape)}")

            with torch.no_grad():
                out = runner.forward(batch, epoch=0, iter_num=0, train=True)
            pred, target = out[0], out[1]
            print(f"  prediction shape: {tuple(pred.shape)}")
            print(f"  target shape:     {tuple(target.shape)}")
            print(f"{name:<10} {'OK':<6}")
        except Exception as e:
            failed += 1
            print(f"{name:<10} {'FAIL':<6} {type(e).__name__}: {e}")
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
