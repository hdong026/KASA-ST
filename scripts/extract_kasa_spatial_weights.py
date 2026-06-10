#!/usr/bin/env python3
"""Export learned hybrid graph weights from a trained KASA checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_state_dict(ckpt_obj) -> dict | None:
    if isinstance(ckpt_obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    return None


def extract_weights(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = find_state_dict(ckpt)
    if state is None:
        raise ValueError(f"Could not find model state dict in {ckpt_path}")

    logits_key = None
    alpha_key = None
    for k in state:
        if k.endswith("hybrid_logits"):
            logits_key = k
        if k.endswith("hybrid_alpha"):
            alpha_key = k

    out: dict = {"checkpoint": str(ckpt_path)}
    if logits_key is None:
        out["warning"] = "hybrid_logits not found in checkpoint"
        return out

    logits = state[logits_key].detach().float()
    weights = torch.softmax(logits, dim=0)
    out["hybrid_logits"] = logits.tolist()
    out["hybrid_weights"] = {
        "static": float(weights[0]),
        "adaptive": float(weights[1]),
        "dynamic": float(weights[2]),
    }
    if alpha_key is not None:
        out["hybrid_alpha"] = float(state[alpha_key].detach())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract KASA hybrid spatial weights.")
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    try:
        result = extract_weights(ckpt_path)
    except Exception as e:
        print(f"Failed to extract weights: {e}")
        return 1

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
