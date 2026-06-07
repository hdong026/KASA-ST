#!/usr/bin/env python3
"""Smoke test for KASA v3-freqgate architecture variants."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_cfg():
    cfg_path = ROOT / "examples" / "KASAST_v3_freqgate" / "KASAST_v3_freqgate_PEMS04.py"
    spec = importlib.util.spec_from_file_location("kasa_v3fg_cfg", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CFG


def run_variant(name: str, overrides: dict, cfg) -> None:
    params = dict(cfg.MODEL.PARAM)
    params.update(overrides)
    model = cfg.MODEL.ARCH(**params)

    b, t, h = 2, cfg.DATASET_INPUT_LEN, cfg.DATASET_OUTPUT_LEN
    n = params["node_size"]
    c = params["input_dim"]
    history = torch.randn(b, t, n, c)
    future = torch.randn(b, h, n, c)

    with torch.no_grad():
        out = model(history, future, 0, 0, False)

    assert out.shape == (b, h, n, 1), f"{name}: expected {(b, h, n, 1)}, got {out.shape}"
    diag = model.spatial_module.get_diagnostics()
    print(f"[OK] {name}: out={tuple(out.shape)}, diagnostics={diag}")


def main() -> int:
    cfg = load_cfg()
    base = {
        "use_pre_temporal_spatial_enhancement": False,
        "keep_output_prior_residual": False,
        "use_input_prior_enhancement": False,
    }

    variants = {
        "frequency_graph_only": {
            **base,
            "use_frequency_guided_graph": True,
            "use_freq_conditioned_fusion": False,
            "use_cross_st_gate": False,
            "use_spectral_decomp_gate": False,
        },
        "frequency_conditioned_fusion": {
            **base,
            "use_frequency_guided_graph": True,
            "use_freq_conditioned_fusion": True,
            "use_cross_st_gate": False,
            "use_spectral_decomp_gate": False,
        },
        "cross_gate": {
            **base,
            "use_frequency_guided_graph": True,
            "use_freq_conditioned_fusion": True,
            "use_cross_st_gate": True,
            "use_spectral_decomp_gate": False,
        },
        "spectral_gate": {
            **base,
            "use_frequency_guided_graph": True,
            "use_freq_conditioned_fusion": True,
            "use_cross_st_gate": True,
            "use_spectral_decomp_gate": True,
        },
        "gate_no_freq": {
            **base,
            "use_frequency_guided_graph": False,
            "use_freq_conditioned_fusion": False,
            "use_cross_st_gate": True,
            "use_spectral_decomp_gate": False,
        },
    }

    print("KASA v3-freqgate smoke test")
    for name, overrides in variants.items():
        run_variant(name, overrides, cfg)

    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
