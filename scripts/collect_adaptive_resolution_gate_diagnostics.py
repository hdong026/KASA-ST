#!/usr/bin/env python3
"""Collect AdaptiveResolutionGate diagnostics from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / max(n, 1)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(values),
        "max": max(values),
        "n": n,
    }


def _find_best_ckpt(ckpt_dir: Path) -> Path | None:
    cands = sorted(ckpt_dir.glob("*/ChainForecasting_*_best_val_MAE.pt"))
    return cands[-1] if cands else None


def collect_from_cfg(
    cfg_path: Path,
    ckpt_path: Path | None,
    split: str = "test",
    max_batches: int = 20,
    device: str = "cuda:0",
    batch_size: int = 32,
) -> dict:
    from easytorch.config import get_config

    from basicts.archs import ChainForecasting
    from basicts.data import TimeSeriesForecastingDataset

    cfg = get_config(str(cfg_path))
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ChainForecasting(**dict(cfg.MODEL.PARAM)).to(device_t)
    model.eval()

    if ckpt_path is None:
        ckpt_path = _find_best_ckpt(Path(cfg.TRAIN.CKPT_SAVE_DIR))
    if ckpt_path is not None and ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=device_t)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)

    mode = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}[split]
    data_file = cfg.DATASET.PARAM["data_file_path"]
    index_file = cfg.DATASET.PARAM["index_file_path"]
    dataset = TimeSeriesForecastingDataset(data_file, index_file, mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    stage_t: dict[int, list[float]] = {}
    stage_s: dict[int, list[float]] = {}
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            # Dataset returns (future, history)
            future_data, history_data = batch
            history_data = history_data.to(device_t)
            out = model(
                history_data=history_data,
                future_data=future_data.to(device_t),
                train=False,
                return_intermediates=True,
            )
            for si, g in enumerate(out.get("temporal_detail_gates") or []):
                stage_t.setdefault(si, []).extend(g.detach().float().flatten().tolist())
            for si, g in enumerate(out.get("spatial_detail_gates") or []):
                stage_s.setdefault(si, []).extend(g.detach().float().flatten().tolist())

    result = {
        "cfg_path": str(cfg_path),
        "ckpt_path": str(ckpt_path) if ckpt_path else None,
        "split": split,
        "max_batches": max_batches,
        "stages": {},
    }
    for si in sorted(set(stage_t) | set(stage_s)):
        result["stages"][str(si)] = {
            "temporal_gate": _stats(stage_t.get(si, [])),
            "spatial_gate": _stats(stage_s.get(si, [])),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "valid", "test"])
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else None
    result = collect_from_cfg(
        Path(args.cfg),
        ckpt,
        args.split,
        args.max_batches,
        args.device,
        args.batch_size,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    for si, st in result["stages"].items():
        print(
            f"stage={si} temporal={st['temporal_gate']} spatial={st['spatial_gate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
