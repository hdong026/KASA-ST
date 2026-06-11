#!/usr/bin/env python3
"""Smoke test for ChainForecasting architecture."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from basicts.archs import ChainForecasting
from basicts.archs.arch_zoo.ChainForecasting_arch.chain_modules import pool_target_to_length


def base_model_args(**overrides):
    args = {
        "node_size": 307,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 4,
        "main_input_dim": 3,
        "chain_lengths": [3, 6, 12],
        "d_model": 64,
        "num_heads": 4,
        "ffn_dim": 128,
        "patch_len": 3,
        "patch_stride": 3,
        "use_downsample_memory": True,
        "use_final_spatial_refine": True,
        "post_spatial_mode": "adaptive_only",
        "adj_mx_path": None,
        "td_size": 288,
        "adp_hidden_dim": 32,
        "adp_topk": 20,
        "adp_tau": 0.5,
    }
    args.update(overrides)
    return args


def test_forward_return_all_false() -> None:
    model = ChainForecasting(**base_model_args())
    history = torch.randn(2, 12, 307, 4)
    out = model(history, return_all=False)
    assert out.shape == (2, 12, 307, 1), f"expected (2,12,307,1), got {out.shape}"
    print(f"[ok] return_all=False: output shape {tuple(out.shape)}")


def test_forward_return_all_true() -> None:
    model = ChainForecasting(**base_model_args())
    history = torch.randn(2, 12, 307, 4)
    out = model(history, return_all=True)
    assert out["pred"].shape == (2, 12, 307, 1)
    assert len(out["chain_preds"]) == 3
    assert out["chain_preds"][0].shape == (2, 3, 307, 1)
    assert out["chain_preds"][1].shape == (2, 6, 307, 1)
    assert out["chain_preds"][2].shape == (2, 12, 307, 1)
    assert len(out["chain_states"]) == 3
    d_model = model.d_model
    assert out["chain_states"][0].shape == (2, 3, 307, d_model)
    assert out["chain_states"][1].shape == (2, 6, 307, d_model)
    assert out["chain_states"][2].shape == (2, 12, 307, d_model)
    print(
        f"[ok] return_all=True: pred {tuple(out['pred'].shape)}, "
        f"chain_preds {[tuple(p.shape) for p in out['chain_preds']]}, "
        f"chain_states d={d_model}"
    )


def test_chain_target_pooling() -> None:
    y = torch.randn(2, 12, 307, 1)
    t3 = ChainForecasting.build_chain_targets(y, [3, 6, 12])[0]
    t6 = ChainForecasting.build_chain_targets(y, [3, 6, 12])[1]
    assert t3.shape == (2, 3, 307, 1), f"t3 shape {t3.shape}"
    assert t6.shape == (2, 6, 307, 1), f"t6 shape {t6.shape}"
    t3_direct = pool_target_to_length(y, 3)
    t6_direct = pool_target_to_length(y, 6)
    assert torch.allclose(t3, t3_direct)
    assert torch.allclose(t6, t6_direct)
    print(f"[ok] build_chain_targets: t3 {tuple(t3.shape)}, t6 {tuple(t6.shape)}")


def test_no_spatial_no_downsample() -> None:
    model = ChainForecasting(**base_model_args(
        use_final_spatial_refine=False,
        use_downsample_memory=False,
    ))
    history = torch.randn(2, 12, 307, 4)
    out = model(history, return_all=True)
    assert out["pred"].shape == (2, 12, 307, 1)
    print("[ok] no_spatial_no_downsample variants forward ok")


def main() -> int:
    test_forward_return_all_false()
    test_forward_return_all_true()
    test_chain_target_pooling()
    test_no_spatial_no_downsample()
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
