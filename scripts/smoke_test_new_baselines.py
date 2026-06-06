#!/usr/bin/env python3
"""Smoke test for newly added PeMS04 baselines (no training)."""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = [
    ("STAEformer", ROOT / "examples" / "baselines" / "STAEformer" / "STAEformer_PEMS04.py"),
    ("STWave", ROOT / "examples" / "baselines" / "STWave" / "STWave_PEMS04.py"),
    ("STDN", ROOT / "examples" / "baselines" / "STDN" / "STDN_PEMS04.py"),
    ("HimNet", ROOT / "examples" / "baselines" / "HimNet" / "HimNet_PEMS04.py"),
]


def load_cfg(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CFG


def _traffic_dummy(B: int, L: int, N: int, C: int) -> torch.Tensor:
    """Flow + normalized ToD/DoW in [0, 1) for embedding-based models."""
    x = torch.zeros(B, L, N, C)
    x[..., 0] = torch.rand(B, L, N)
    if C > 1:
        x[..., 1] = torch.rand(B, L, N) * 0.99
    if C > 2:
        x[..., 2] = torch.rand(B, L, N) * 0.99
    return x


def mini_forward(name: str, cfg, model: torch.nn.Module) -> str:
    if name in ("STAEformer", "STWave"):
        B, L, N = 2, 12, cfg.MODEL.PARAM.get("num_nodes", 307)
        C = len(cfg.MODEL.FORWARD_FEATURES)
        hist = _traffic_dummy(B, L, N, C)
        fut = _traffic_dummy(B, L, N, C)
        with torch.no_grad():
            out = model(history_data=hist, future_data=fut, batch_seen=0, epoch=0, train=False)
        return f"forward OK shape={tuple(out.shape)}"
    if name == "STDN":
        return "instantiate OK (custom STDN forward skipped in smoke test)"
    if name == "HimNet":
        B, L, N = 2, 12, cfg.MODEL.PARAM["num_nodes"]
        x = _traffic_dummy(B, L, N, 3)
        x[..., 2] = torch.randint(0, 7, (B, L, N)).float()
        y_cov = x[..., 1:3]
        model.eval()
        with torch.no_grad():
            out = model(x, y_cov)
        return f"forward OK shape={tuple(out.shape)}"
    return "instantiate OK"


def main() -> int:
    print(f"{'Model':<12} {'Status':<6} Details")
    print("-" * 80)
    failed = 0
    for name, cfg_path in MODELS:
        try:
            compile(cfg_path.read_text(encoding="utf-8"), str(cfg_path), "exec")
            cfg = load_cfg(cfg_path)
            arch = cfg.MODEL.ARCH
            param = dict(cfg.MODEL.PARAM)
            model = arch(**param)
            runner = cfg.RUNNER.__name__ if hasattr(cfg.RUNNER, "__name__") else str(cfg.RUNNER)
            ff = cfg.MODEL.FORWARD_FEATURES
            tf = cfg.MODEL.TARGET_FEATURES
            detail = mini_forward(name, cfg, model)
            print(f"{name:<12} {'OK':<6} runner={runner} FORWARD={ff} TARGET={tf} | {detail}")
        except Exception as e:
            failed += 1
            print(f"{name:<12} {'FAIL':<6} {type(e).__name__}: {e}")
            traceback.print_exc()
            print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
