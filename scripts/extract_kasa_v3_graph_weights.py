#!/usr/bin/env python3
"""Best-effort export of KASA v3 graph fusion weights and gate diagnostics."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.KASA_arch_v3.KASA_arch import KASA_v3


def load_checkpoint(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def find_latest_ckpt(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.is_dir():
        return None
    candidates = sorted(ckpt_dir.glob("**/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(ckpt_dir.glob("**/*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def extract_static_weights(model: KASA_v3) -> dict:
    spatial = model.spatial_module
    out: dict = {
        "use_frequency_guided_graph": spatial.use_frequency_guided_graph,
        "use_cross_st_gate": spatial.use_cross_st_gate,
        "use_spectral_decomp_gate": spatial.use_spectral_decomp_gate,
    }
    if spatial.freq_hybrid_logits is not None:
        w = torch.softmax(spatial.freq_hybrid_logits, dim=0).detach().cpu().tolist()
        out["fusion_weights"] = {
            "static": w[0],
            "adaptive": w[1],
            "dynamic": w[2],
            "frequency": w[3],
        }
    elif spatial.hybrid_logits is not None:
        w = torch.softmax(spatial.hybrid_logits, dim=0).detach().cpu().tolist()
        out["fusion_weights"] = {
            "static": w[0],
            "adaptive": w[1],
            "dynamic": w[2],
            "frequency": None,
        }
    return out


def run_forward_diagnostics(model: KASA_v3, batch_size: int, device: str) -> dict:
    model.eval()
    b, t, n = batch_size, model.input_len, model.node_size
    h = model.output_len
    history = torch.randn(b, t, n, model.input_dim, device=device)
    future = torch.randn(b, h, n, model.input_dim, device=device)
    with torch.no_grad():
        _ = model(history, future, 0, 0, False)
    diag = model.spatial_module.get_last_diagnostics()
    return {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in diag.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract KASA v3 graph fusion weights.")
    parser.add_argument("--ckpt", default="", help="Path to checkpoint .pt/.pth")
    parser.add_argument("--ckpt_dir", default="", help="Checkpoint directory (uses latest)")
    parser.add_argument("--out", default="results/kasa_v3_graph_weights.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_args_json", default="", help="Optional JSON model args override")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if ckpt_path is None and args.ckpt_dir:
        ckpt_path = find_latest_ckpt(Path(args.ckpt_dir))
    if ckpt_path is None or not ckpt_path.is_file():
        print("Checkpoint not found. Provide --ckpt or --ckpt_dir.")
        return 1

    model_args = {
        "node_size": 307,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 4,
        "patch_len": 3,
        "stride": 4,
        "td_size": 288,
        "dw_size": 7,
        "d_td": 32,
        "d_dw": 32,
        "d_d": 32,
        "d_spa": 32,
        "if_time_in_day": True,
        "if_day_in_week": True,
        "if_spatial": True,
        "num_layer": 2,
        "spatial_scheme": "C",
        "use_gcn": True,
        "use_dynamic_spatial": True,
        "use_adaptive_adj": True,
        "use_hybrid_graph": True,
        "use_frequency_guided_graph": True,
        "freq_dim": 16,
        "freq_topk": 20,
        "use_cross_st_gate": False,
        "use_spectral_decomp_gate": False,
    }
    if args.model_args_json:
        model_args.update(json.loads(args.model_args_json))

    model = KASA_v3(**model_args)
    state = load_checkpoint(ckpt_path)
    model.load_state_dict(state, strict=False)
    model.to(args.device)

    result = {
        "checkpoint": str(ckpt_path),
        "static_params": extract_static_weights(model),
    }
    try:
        result["forward_diagnostics"] = run_forward_diagnostics(model, args.batch_size, args.device)
    except Exception as e:
        result["forward_diagnostics_error"] = str(e)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(result["static_params"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
