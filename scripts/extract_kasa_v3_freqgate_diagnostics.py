#!/usr/bin/env python3
"""Best-effort export of KASA v3-freqgate graph fusion and gate diagnostics."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_cfg_module(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("kasa_cfg", cfg_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load config: {cfg_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CFG


def find_checkpoint(ckpt: str, ckpt_dir: str) -> Path | None:
    if ckpt:
        p = Path(ckpt)
        return p if p.is_file() else None
    if ckpt_dir:
        base = Path(ckpt_dir)
        if not base.is_dir():
            return None
        for pattern in ("**/*.pt", "**/*.pth"):
            hits = sorted(base.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
            if hits:
                return hits[0]
    return None


def load_state_dict(ckpt_path: Path) -> dict:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
        if "state_dict" in obj:
            return obj["state_dict"]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract KASA v3-freqgate diagnostics.")
    parser.add_argument("--cfg", default="examples/KASAST_v3_freqgate/KASAST_v3_freqgate_PEMS04.py")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--ckpt_dir", default="")
    parser.add_argument("--out", default="results/kasa_v3_freqgate_diagnostics.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    cfg_path = ROOT / args.cfg
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}")
        return 1

    cfg = load_cfg_module(cfg_path)
    arch_cls = cfg.MODEL.ARCH
    model = arch_cls(**cfg.MODEL.PARAM)

    ckpt_path = find_checkpoint(args.ckpt, args.ckpt_dir)
    if ckpt_path is not None:
        try:
            model.load_state_dict(load_state_dict(ckpt_path), strict=False)
            print(f"Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"TODO: checkpoint load failed ({e}); exporting static params only.")
    else:
        print("TODO: no checkpoint found; exporting static params and one random forward pass.")

    model.to(args.device)
    model.eval()

    b = args.batch_size
    t = cfg.DATASET_INPUT_LEN
    h = cfg.DATASET_OUTPUT_LEN
    n = cfg.MODEL.PARAM["node_size"]
    c = cfg.MODEL.PARAM.get("input_dim", 4)

    history = torch.randn(b, t, n, c, device=args.device)
    future = torch.randn(b, h, n, c, device=args.device)
    with torch.no_grad():
        _ = model(history, future, 0, 0, False)

    diag = model.spatial_module.get_diagnostics()
    row = {
        "checkpoint": str(ckpt_path) if ckpt_path else "",
        "cfg": str(cfg_path),
        **diag,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(f"Wrote {out_path}")
    for k, v in row.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
