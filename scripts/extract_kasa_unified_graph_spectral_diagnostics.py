#!/usr/bin/env python3
"""Extract diagnostics from a trained KASA graph spectral checkpoint."""
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

from basicts.archs import KASA_v2


def find_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.is_dir():
        return None
    candidates = sorted(ckpt_dir.glob("**/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    candidates = sorted(ckpt_dir.glob("**/*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_state_dict(ckpt_path: Path) -> dict:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
        if "state_dict" in obj:
            return obj["state_dict"]
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")


def try_validation_batch_stats(model: KASA_v2) -> dict:
    """Best-effort one-batch spectral component statistics."""
    try:
        model.eval()
        channels = model.input_dim
        history = torch.randn(1, model.input_len, model.node_size, channels)
        future = torch.randn(1, model.output_len, model.node_size, channels)
        with torch.no_grad():
            backbone = model(history, future, batch_seen=1, epoch=0, train=False, return_backbone=True)
            refined = model.spatial_module.refine_prediction(backbone, history[..., 0])
            if model.use_graph_spectral_calibration and model.graph_spectral_calibration is not None:
                adj = model.spatial_module.get_adaptive_adj()
                calib = model.graph_spectral_calibration
                p_low = calib._projector_low(adj)
                y_flat = refined.squeeze(-1)
                y_low = torch.einsum("nm,bhm->bhn", p_low, y_flat).unsqueeze(-1)
                y_high = refined - y_low
                delta = calib.mlp(torch.cat([y_low, y_high], dim=-1))
                return {
                    "low_component_abs_mean": float(y_low.abs().mean().item()),
                    "high_component_abs_mean": float(y_high.abs().mean().item()),
                    "calibration_delta_abs_mean": float(delta.abs().mean().item()),
                }
    except Exception as exc:
        return {"validation_batch_stats_error": str(exc)}
    return {"validation_batch_stats": "skipped"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract KASA graph spectral diagnostics.")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--variant", type=str, default="unknown")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = find_checkpoint(ckpt_dir)
    if ckpt_path is None:
        print(json.dumps({"error": f"No checkpoint found under {ckpt_dir}"}, indent=2))
        return 1

    state = load_state_dict(ckpt_path)
    gamma_key = "graph_spectral_calibration.gamma"
    gamma = float(state[gamma_key].item()) if gamma_key in state else None

    report = {
        "variant": args.variant,
        "checkpoint": str(ckpt_path),
        "offline_prior_used": None,
        "online_calibration_used": gamma is not None,
        "graph_spectral_gamma": gamma,
        "graph_spectral_k": None,
        "graph_spectral_calibration_mode": None,
    }

    # TODO: instantiate model with saved config if available for richer stats.
    try:
        dummy = KASA_v2(
            node_size=307,
            input_len=12,
            output_len=12,
            input_dim=4,
            patch_len=3,
            stride=4,
            td_size=288,
            dw_size=7,
            d_td=32,
            d_dw=32,
            d_d=32,
            d_spa=32,
            if_time_in_day=True,
            if_day_in_week=True,
            if_spatial=True,
            num_layer=2,
            use_adaptive_adj=True,
            post_spatial_mode="adaptive_only",
            use_graph_spectral_calibration=gamma is not None,
            graph_spectral_calibration_mode="residual_safe_low_high" if gamma is not None else "none",
            graph_spectral_k=32,
        )
        dummy.load_state_dict(state, strict=False)
        report["graph_spectral_k"] = dummy.graph_spectral_calibration.k if dummy.graph_spectral_calibration else None
        report["graph_spectral_calibration_mode"] = (
            dummy.graph_spectral_calibration.mode if dummy.graph_spectral_calibration else "none"
        )
        report.update(try_validation_batch_stats(dummy))
        if args.variant.startswith("offline"):
            report["offline_prior_used"] = True
            report["input_prior_channel_abs_mean"] = "TODO: requires dataset batch"
            report["input_prior_channel_abs_std"] = "TODO: requires dataset batch"
        else:
            report["offline_prior_used"] = False
    except Exception as exc:
        report["model_load_error"] = str(exc)

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
