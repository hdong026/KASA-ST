#!/usr/bin/env python3
"""Smoke checks for interleaved nested graph-resolution variant."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from easytorch.config import init_cfg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs import ChainForecasting
from scripts.run_chain_forecasting_horizon import (
    activate_dataset,
    generate_temp_config,
    load_cfg,
    variant_spec,
)

FORMAL = "chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss"
NESTED = "chain_interleaved_nested_graph_resolution_state_adapter_fixed_token_loss"
ADJ = str(ROOT / "datasets" / "PEMS04" / "adj_mx.pkl")


def _count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_hierarchy_and_shapes() -> dict:
    activate_dataset("PEMS04")
    cfg = load_cfg(generate_temp_config(12, NESTED, 1))
    assert cfg.MODEL.PARAM["spatial_placement"] == "interleaved_graph_resolution"
    assert cfg.MODEL.PARAM["clustering_seed"] == 0
    assert cfg.TRAIN.CKPT_SAVE_DIR != str(
        load_cfg(generate_temp_config(12, FORMAL, 1)).TRAIN.CKPT_SAVE_DIR
    )

    model = ChainForecasting(**dict(cfg.MODEL.PARAM))
    stack = model.graph_resolution_stack
    assert stack is not None
    meta = stack.metadata()
    sizes = list(meta["graph_resolution_sizes"])
    assert len(sizes) == 3
    assert sizes[-1] == 307
    assert meta["nested_consistency"] is True
    assert meta["clustering_seed"] == 0

    for i, m_j in enumerate(sizes):
        c = getattr(stack, f"stage{i}_C")
        p = getattr(stack, f"stage{i}_P")
        assert tuple(c.shape) == (307, m_j), (i, c.shape, m_j)
        assert tuple(p.shape) == (m_j, 307), (i, p.shape, m_j)
        row_sum = c.sum(dim=1)
        assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-5)
        assert not c.requires_grad and not p.requires_grad

    history = torch.randn(2, 12, 307, 4)
    model.train()
    out = model(history, return_all=True)
    assert out["pred"].shape == (2, 12, 307, 1)
    for i, z in enumerate(out["chain_preds"]):
        assert z.shape[-2] == 307, (i, z.shape)
    assert out.get("nested_consistency") is True
    assert out.get("graph_resolution_capacities") == [4, 2, 1]

    # condition-only: Z6_condition differs from supervised Z6_raw when adapter active
    with torch.no_grad():
        model.forecast_state_adapter.proj_out.weight.fill_(0.05)
        model.forecast_state_adapter.proj_out.bias.fill_(0.01)
    captured = {}
    real = model.temporal_steps[2].forward

    def wrap(history_data, prev_forecast=None, prev_latent=None, **kwargs):
        captured["prev"] = None if prev_forecast is None else prev_forecast.detach().clone()
        return real(
            history_data, prev_forecast=prev_forecast, prev_latent=prev_latent, **kwargs
        )

    model.temporal_steps[2].forward = wrap
    out2 = model(history, return_all=True)
    z6 = out2["chain_preds"][1]
    assert captured["prev"] is not None
    # T12 receives interpolate(condition)->12; condition != interpolate(Z6_raw)
    from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
        interpolate_forecast,
    )

    assert not torch.allclose(captured["prev"], interpolate_forecast(z6, 12), atol=1e-6)

    # grads
    target = torch.randn_like(out2["pred"])
    loss = sum((p - target[:, : p.shape[1]]).abs().mean() for p in out2["chain_preds"])
    loss.backward()
    spa_grads = [
        sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
        for m in stack.spatial_modules
    ]
    assert all(g > 0 for g in spa_grads), spa_grads
    print(
        f"[ok] hierarchy sizes={sizes} max={meta['max_cluster_sizes']} "
        f"eff_topk={meta['graph_resolution_effective_topks']} nested={meta['nested_consistency']}"
    )
    return {
        "sizes": sizes,
        "max_sizes": meta["max_cluster_sizes"],
        "eff_topk": meta["graph_resolution_effective_topks"],
        "nested": meta["nested_consistency"],
        "n_params": _count_params(model),
    }


def test_horizons_and_formal_unchanged() -> None:
    activate_dataset("PEMS04")
    formal = ChainForecasting(**dict(load_cfg(generate_temp_config(12, FORMAL, 1)).MODEL.PARAM))
    n_formal = _count_params(formal)
    for h, lens in ((12, [3, 6, 12]), (24, [6, 12, 24]), (48, [12, 24, 48])):
        cfg = load_cfg(generate_temp_config(h, NESTED, 1))
        assert cfg.MODEL.PARAM["chain_lengths"] == lens
        model = ChainForecasting(**dict(cfg.MODEL.PARAM))
        out = model(torch.randn(1, 12, 307, 4), return_all=True)
        assert out["pred"].shape == (1, h, 307, 1)
    # formal still progressive
    assert formal.spatial_placement == "interleaved_progressive"
    assert formal.graph_resolution_stack is None
    y = formal(torch.randn(1, 12, 307, 4))
    assert y.shape == (1, 12, 307, 1)
    assert _count_params(formal) == n_formal
    print(f"[ok] horizons + formal unchanged params={n_formal}")


def main() -> None:
    activate_dataset("PEMS04")
    assert NESTED in (
        "chain_interleaved_nested_graph_resolution_state_adapter_fixed_token_loss",
    )
    assert variant_spec(NESTED, 12)["spatial_placement"] == "interleaved_graph_resolution"
    stats = test_hierarchy_and_shapes()
    test_horizons_and_formal_unchanged()
    print("[ok] all nested graph-resolution checks passed")
    print("REPORT", stats)


if __name__ == "__main__":
    main()
