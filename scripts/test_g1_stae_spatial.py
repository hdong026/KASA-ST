"""Checks for G1-STAE-Spatial vs original G1 temporal chain."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basicts.archs import ChainForecasting
from basicts.losses import masked_mae
from examples.ChainForecasting.ChainForecasting_PEMS04 import CFG as G1_CFG
from examples.ChainForecasting.ChainForecasting_PEMS04_G1_STAESpatial import (
    CFG as STAE_CFG,
)


def _history(batch: int = 2) -> torch.Tensor:
    x = torch.randn(batch, 12, 307, 4)
    x[..., 1] = torch.rand(batch, 12, 307)
    x[..., 2] = torch.randint(0, 7, (batch, 12, 307)).float() / 7.0
    return x


def _copy_shared_weights(src: ChainForecasting, dst: ChainForecasting) -> None:
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    shared = {
        k: v for k, v in src_sd.items() if k in dst_sd and dst_sd[k].shape == v.shape
    }
    dst.load_state_dict(shared, strict=False)


def test_import_and_config():
    assert G1_CFG.MODEL.PARAM["post_spatial_mode"] == "adaptive_only"
    assert STAE_CFG.MODEL.PARAM["post_spatial_mode"] == "stae_spatial_attention"
    assert G1_CFG.TRAIN.CKPT_SAVE_DIR != STAE_CFG.TRAIN.CKPT_SAVE_DIR

    g1_param = dict(G1_CFG.MODEL.PARAM)
    stae_param = dict(STAE_CFG.MODEL.PARAM)
    stae_only = {
        "post_spatial_mode",
        "spatial_attn_dim",
        "spatial_attn_heads",
        "spatial_attn_ffn_dim",
        "spatial_attn_layers",
        "spatial_attn_dropout",
    }
    for k, v in g1_param.items():
        if k in stae_only:
            continue
        assert stae_param.get(k) == v, f"PARAM mismatch {k}: {stae_param.get(k)!r} vs {v!r}"

    model = ChainForecasting(**STAE_CFG.MODEL.PARAM)
    assert model.post_spatial_mode == "stae_spatial_attention"
    assert model.spatial_placement == "final"
    assert list(model.chain_lengths) == [3, 6, 12]
    assert model.spatial_module.stae_spatial_block is not None
    print("import_and_config: OK")


def test_temporal_identical_to_g1():
    torch.manual_seed(0)
    g1 = ChainForecasting(**G1_CFG.MODEL.PARAM)
    stae = ChainForecasting(**STAE_CFG.MODEL.PARAM)
    _copy_shared_weights(g1, stae)
    g1.eval()
    stae.eval()

    history = _history()
    with torch.no_grad():
        _, _, t_g1, *_ = g1._forward_chain(history)
        _, _, t_stae, *_ = stae._forward_chain(history)

    assert len(t_g1) == len(t_stae) == 3
    for i, (a, b) in enumerate(zip(t_g1, t_stae)):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        assert a.shape[1] == [3, 6, 12][i]
    print("temporal_identical_T3_T6_T12: OK")


def test_forward_shape_and_no_outer_residual():
    model = ChainForecasting(**STAE_CFG.MODEL.PARAM)
    model.eval()
    history = _history()
    with torch.no_grad():
        y_final, _, temporal_preds, *_ = model._forward_chain(history)
        y_t = temporal_preds[-1]
        block_out = model.spatial_module.refine_prediction(y_t, history[..., 0])

    assert y_final.shape == (2, 12, 307, 1)
    torch.testing.assert_close(y_final, block_out, rtol=1e-5, atol=1e-6)

    # Must not be Y_T + alpha * anything (outer residual).
    assert not torch.allclose(y_final, y_t)
    # Attention is over N tokens: reshape path uses B*T batches of N.
    block = model.spatial_module.stae_spatial_block
    assert isinstance(block.attn_layers[0], nn.MultiheadAttention)
    assert block.attn_layers[0].batch_first is True
    print("forward_shape_and_no_outer_residual: OK", tuple(y_final.shape))


def test_internal_residuals_exist():
    block = ChainForecasting(**STAE_CFG.MODEL.PARAM).spatial_module.stae_spatial_block
    assert len(block.attn_layers) == 1
    assert len(block.norm1_layers) == 1
    assert len(block.ffn_layers) == 1
    assert len(block.norm2_layers) == 1
    # Two residual sites: attn residual (norm1) and ffn residual (norm2).
    y = torch.randn(2, 12, 307, 1, requires_grad=True)
    out = block(y)
    # If either residual path were removed, gradients to y would still exist via
    # projections; instead verify both norm modules participate by checking they
    # receive gradients when backpropagating.
    out.abs().mean().backward()
    assert block.norm1_layers[0].weight.grad is not None
    assert block.norm2_layers[0].weight.grad is not None
    print("internal_residuals: OK")


def test_backward_gradients():
    model = ChainForecasting(**STAE_CFG.MODEL.PARAM)
    history = _history()
    pred = model(history_data=history, future_data=None, batch_seen=0, epoch=1, train=True)
    pred.abs().mean().backward()

    for idx, step in enumerate(model.temporal_steps):
        grads = [p.grad for p in step.parameters() if p.requires_grad]
        assert any(g is not None for g in grads), f"T stage {idx}"
        assert all(torch.isfinite(g).all() for g in grads if g is not None)
        assert sum(float(g.abs().sum()) for g in grads if g is not None) > 0

    block = model.spatial_module.stae_spatial_block
    named = {
        "input_proj": block.input_proj.weight,
        "adaptive_embedding": block.adaptive_embedding,
        "attn_in_proj": block.attn_layers[0].in_proj_weight,
        "ffn": block.ffn_layers[0][0].weight,
        "output_proj": block.output_proj.weight,
    }
    for name, param in named.items():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert float(param.grad.abs().sum()) > 0, name
    print("backward_gradients: OK")


def test_basicts_dry_run():
    model = ChainForecasting(**STAE_CFG.MODEL.PARAM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = _history(batch=4)
    target = torch.randn(4, 12, 307, 1)

    model.train()
    pred = model(history_data=history, future_data=None, batch_seen=0, epoch=1, train=True)
    loss = masked_mae(pred, target, null_val=0.0)
    assert torch.isfinite(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(
            history_data=history, future_data=None, batch_seen=0, epoch=1, train=False
        )
        mae = masked_mae(val_pred, target, null_val=0.0)
        rmse = torch.sqrt(((val_pred - target) ** 2).mean())
    assert torch.isfinite(mae) and torch.isfinite(rmse)
    print(
        "basicts_dry_run: OK",
        f"loss={float(loss.detach()):.4f}",
        f"mae={float(mae):.4f}",
        f"rmse={float(rmse):.4f}",
    )


def main():
    test_import_and_config()
    test_temporal_identical_to_g1()
    test_forward_shape_and_no_outer_residual()
    test_internal_residuals_exist()
    test_backward_gradients()
    test_basicts_dry_run()
    print("\nALL G1-STAE-Spatial CHECKS PASSED")


if __name__ == "__main__":
    main()
