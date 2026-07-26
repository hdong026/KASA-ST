"""G1-self-replace checks aligned with ChainForecasting_PEMS04 G1."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basicts.archs import ChainForecasting
from basicts.archs.arch_zoo.ChainForecasting_arch.gcn import apply_adj
from basicts.losses import masked_mae
from examples.ChainForecasting.ChainForecasting_PEMS04 import CFG as G1_CFG
from examples.ChainForecasting.ChainForecasting_PEMS04_G1_self_replace import (
    CFG as SELF_CFG,
)


def _history(batch: int = 2) -> torch.Tensor:
    # Match G1 / ChainForecasting_PEMS04: 4 input channels.
    x = torch.randn(batch, 12, 307, 4)
    x[..., 1] = torch.rand(batch, 12, 307)
    x[..., 2] = torch.randint(0, 7, (batch, 12, 307)).float() / 7.0
    return x


def _copy_shared_weights(src: ChainForecasting, dst: ChainForecasting) -> None:
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    shared = {k: v for k, v in src_sd.items() if k in dst_sd and dst_sd[k].shape == v.shape}
    missing = dst.load_state_dict(shared, strict=False)
    assert len(missing.unexpected_keys) == 0


def test_import_original_g1():
    assert G1_CFG.MODEL.PARAM["post_spatial_mode"] == "adaptive_only"
    assert SELF_CFG.MODEL.PARAM["post_spatial_mode"] == "adaptive_self_replace"
    assert G1_CFG.TRAIN.CKPT_SAVE_DIR != SELF_CFG.TRAIN.CKPT_SAVE_DIR
    g1 = ChainForecasting(**G1_CFG.MODEL.PARAM)
    assert g1.post_spatial_mode == "adaptive_only"
    assert g1.spatial_placement == "final"
    assert list(g1.chain_lengths) == [3, 6, 12]
    print("import_original_g1: OK")


def test_config_alignment():
    g1_param = dict(G1_CFG.MODEL.PARAM)
    sr_param = dict(SELF_CFG.MODEL.PARAM)
    # Only post_spatial_mode may differ among MODEL.PARAM keys that exist in G1.
    for k, v in g1_param.items():
        if k == "post_spatial_mode":
            continue
        assert sr_param.get(k) == v, f"PARAM mismatch at {k}: {sr_param.get(k)!r} vs {v!r}"
    assert G1_CFG.MODEL.FORWARD_FEATURES == SELF_CFG.MODEL.FORWARD_FEATURES
    assert G1_CFG.MODEL.TARGET_FEATURES == SELF_CFG.MODEL.TARGET_FEATURES
    assert G1_CFG.ENV.SEED == SELF_CFG.ENV.SEED
    assert G1_CFG.TRAIN.OPTIM.PARAM == SELF_CFG.TRAIN.OPTIM.PARAM
    assert G1_CFG.TRAIN.LR_SCHEDULER.PARAM == SELF_CFG.TRAIN.LR_SCHEDULER.PARAM
    print("config_alignment: OK")


def test_temporal_identical_to_g1():
    torch.manual_seed(0)
    g1 = ChainForecasting(**G1_CFG.MODEL.PARAM)
    sr = ChainForecasting(**SELF_CFG.MODEL.PARAM)
    _copy_shared_weights(g1, sr)
    g1.eval()
    sr.eval()

    history = _history()
    with torch.no_grad():
        _, _, t_g1, *_ = g1._forward_chain(history)
        _, _, t_sr, *_ = sr._forward_chain(history)

    assert len(t_g1) == len(t_sr) == 3
    for i, (a, b) in enumerate(zip(t_g1, t_sr)):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        assert a.shape[1] == [3, 6, 12][i]
    print("temporal_identical_T3_T6_T12: OK")


def test_adaptive_adj_properties():
    model = ChainForecasting(**SELF_CFG.MODEL.PARAM)
    adj = model.spatial_module._build_adaptive_adj_with_self()
    n = 307
    topk = int(SELF_CFG.MODEL.PARAM["adp_topk"])
    assert adj.shape == (n, n)
    assert torch.isfinite(adj).all()
    torch.testing.assert_close(
        adj.sum(dim=-1),
        torch.ones(n, device=adj.device),
        rtol=1e-5,
        atol=1e-6,
    )
    diag = torch.diagonal(adj)
    assert torch.all(diag > 0), float(diag.min())
    nnz = (adj > 0).sum(dim=-1)
    assert torch.all(nnz == topk), nnz.unique()
    print(
        "adaptive_adj: OK",
        f"diag_min={float(diag.min().detach()):.6f}",
        f"nnz={int(nnz[0])}",
    )


def test_output_equals_manual_apply_adj():
    model = ChainForecasting(**SELF_CFG.MODEL.PARAM)
    history = _history()
    with torch.no_grad():
        y_final, _, temporal_preds, *_ = model._forward_chain(history)
        y_t = temporal_preds[-1]
        assert y_t.shape == (2, 12, 307, 1)
        adj = model.spatial_module._build_adaptive_adj_with_self()
        manual = apply_adj(y_t.squeeze(-1), adj).unsqueeze(-1)
        refined = model.spatial_module.refine_prediction(y_t, history[..., 0])
    torch.testing.assert_close(y_final, manual, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(refined, manual, rtol=1e-5, atol=1e-6)

    old = y_t + float(model.spatial_module.hybrid_alpha) * manual
    assert not torch.allclose(y_final, old)
    print("output_equals_A_self_Y12: OK", tuple(y_final.shape))


def test_backward_gradients():
    model = ChainForecasting(**SELF_CFG.MODEL.PARAM)
    history = _history()
    pred = model(history_data=history, future_data=None, batch_seen=0, epoch=1, train=True)
    pred.abs().mean().backward()

    for idx, step in enumerate(model.temporal_steps):
        grads = [p.grad for p in step.parameters() if p.requires_grad]
        assert any(g is not None for g in grads), f"T stage {idx}"
        assert all(torch.isfinite(g).all() for g in grads if g is not None)
        assert sum(float(g.abs().sum()) for g in grads if g is not None) > 0

    assert model.spatial_module.adaptive_src.grad is not None
    assert model.spatial_module.adaptive_dst.grad is not None
    assert torch.isfinite(model.spatial_module.adaptive_src.grad).all()
    assert torch.isfinite(model.spatial_module.adaptive_dst.grad).all()
    assert float(model.spatial_module.adaptive_src.grad.abs().sum()) > 0
    assert float(model.spatial_module.adaptive_dst.grad.abs().sum()) > 0
    print("backward_gradients: OK")


def test_basicts_dry_run():
    model = ChainForecasting(**SELF_CFG.MODEL.PARAM)
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
        mape = (val_pred - target).abs().mean()  # finite check proxy
    assert torch.isfinite(mae) and torch.isfinite(rmse) and torch.isfinite(mape)
    print(
        "basicts_dry_run: OK",
        f"loss={float(loss):.4f}",
        f"mae={float(mae):.4f}",
        f"rmse={float(rmse):.4f}",
    )


def main():
    test_import_original_g1()
    test_config_alignment()
    test_temporal_identical_to_g1()
    test_adaptive_adj_properties()
    test_output_equals_manual_apply_adj()
    test_backward_gradients()
    test_basicts_dry_run()
    print("\nALL G1-self-replace ALIGNED CHECKS PASSED")


if __name__ == "__main__":
    main()
