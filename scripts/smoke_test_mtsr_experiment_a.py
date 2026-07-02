#!/usr/bin/env python3
"""Sanity checks for MTSR Experiment Group A (graph-resolution focus)."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from basicts.archs import ChainForecasting
from basicts.archs.arch_zoo.ChainForecasting_arch.graph_cluster_utils import (
    resolve_graph_resolution_sizes,
)
from basicts.losses import masked_mae
from scripts.run_mtsr_experiment_a_pems04 import VARIANT_SPECS, variant_spec

ADJ = os.path.join(ROOT, "datasets", "PEMS04", "adj_mx.pkl")

HORIZONS = [12, 48]
ACTIVE_VARIANTS = [
    "mtsr_temporal_first_final_spatial",
    "mtsr_temporal_first_node_preserving_multiscale_spatial",
    "mtsr_temporal_first_graph_resolution_spatial",
]


def model_args_for(variant: str, horizon: int) -> dict:
    spec = variant_spec(variant, horizon)
    base = {
        "node_size": 307,
        "input_len": 12,
        "output_len": horizon,
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
        "adj_mx_path": ADJ,
        "use_gcn": True,
        "gcn_hidden_dim": 64,
        "use_dynamic_spatial": True,
        "dyn_hidden_dim": 64,
        "dyn_topk": 20,
        "dyn_tau": 0.5,
        "use_adaptive_adj": True,
        "adp_hidden_dim": 32,
        "adp_topk": 20,
        "adp_tau": 0.5,
        "use_hybrid_graph": True,
        "hybrid_alpha": 0.2,
        "post_spatial_mode": "adaptive_only",
        "use_patch_branch": True,
        "use_downsample_branch": True,
        "use_linear_residual_branch": True,
        "patch_embedding_mode": "serial_concat",
        "patch_data_input_mode": "all",
        "use_prev_condition": True,
        "propagation_mode": "forecast_state",
        "dataset_name": "PEMS04",
        "clustering_seed": 0,
    }
    skip = {"is_chain"}
    for k, v in spec.items():
        if k in skip or v is None:
            continue
        base[k] = v
    return base


def compute_loss(out: dict, future: torch.Tensor, chain_lengths: list[int], chain_weights: list[float]) -> torch.Tensor:
    targets = [ChainForecasting.pool_target(future[..., :1], k) for k in chain_lengths]
    loss = torch.tensor(0.0, device=future.device)
    preds = out["chain_preds"]
    for w, p, t in zip(chain_weights[:-1], preds[:-1], targets[:-1]):
        if float(w) != 0.0:
            loss = loss + float(w) * masked_mae(p, t, null_val=0.0)
    loss = loss + float(chain_weights[-1]) * masked_mae(out["pred"], targets[-1], null_val=0.0)
    return loss


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(masked_mae(pred, target, null_val=0.0).item())


def check_variant(variant: str, horizon: int) -> dict:
    spec = variant_spec(variant, horizon)
    args = model_args_for(variant, horizon)
    model = ChainForecasting(**args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    batch, nodes, channels = 2, 307, 4
    history = torch.randn(batch, 12, nodes, channels)
    future = torch.randn(batch, horizon, nodes, channels)
    target = future[..., :1]

    out = model(history, return_all=True)
    chain_lengths = spec["chain_lengths"]

    report: dict = {
        "variant": variant,
        "horizon": horizon,
        "params": n_params,
        "spatial_organization_type": out.get("spatial_organization_type"),
        "temporal_shapes": [tuple(p.shape) for p in out["temporal_preds"]],
        "final_temporal_shape": tuple(out["final_temporal_pred"].shape),
        "final_pred_shape": tuple(out["pred"].shape),
        "spatial_stage_shapes": [tuple(p.shape) for p in out.get("spatial_stage_preds", [])],
    }

    assert out["pred"].shape == (batch, horizon, nodes, 1), report
    for i, length in enumerate(chain_lengths):
        assert out["temporal_preds"][i].shape == (batch, length, nodes, 1)
        assert out["chain_preds"][i].shape == (batch, length, nodes, 1)

    loss = compute_loss(out, future, chain_lengths, spec["chain_loss_weights"])
    assert torch.isfinite(loss)
    report["loss"] = float(loss.item())

    if variant == "mtsr_temporal_first_graph_resolution_spatial":
        meta = model.graph_resolution_stack.metadata()
        sizes = meta["graph_resolution_sizes"]
        report["graph_resolution_sizes"] = sizes
        report["clustering_methods"] = meta["clustering_methods"]
        report["cluster_cache_paths"] = meta["cluster_cache_paths"]
        report["expected_sizes"] = resolve_graph_resolution_sizes(
            nodes, spec.get("graph_resolution_ratios", [0.25, 0.50, 1.00])
        )
        assert sizes == report["expected_sizes"]

        diag = out["graph_resolution_diagnostics"]
        report["cluster_stage_shapes"] = [tuple(t.shape) for t in diag["cluster_stage_preds"]]
        report["cluster_residual_shapes"] = [tuple(t.shape) for t in diag["cluster_residuals"]]
        report["lifted_residual_shapes"] = [tuple(t.shape) for t in diag["lifted_residuals"]]
        report["residual_energy_cluster"] = diag["residual_energy_cluster"]
        report["residual_energy_lifted"] = diag["residual_energy_lifted"]

        y_temp = out["final_temporal_pred"]
        report["mae_before_graph_spatial"] = mae(y_temp, target)
        node_stages = [y_temp] + list(diag["node_stage_preds"])
        report["mae_after_graph_steps"] = [mae(u, target) for u in node_stages[1:]]
        report["mae_final"] = mae(out["pred"], target)

        for idx, m_j in enumerate(sizes):
            if m_j < nodes:
                p = getattr(model.graph_resolution_stack, f"stage{idx}_P")
                y_cluster = torch.einsum("mn,btnc->btmc", p, target)
                u_cluster = diag["cluster_stage_preds"][idx]
                report.setdefault("cluster_target_mae", []).append(mae(u_cluster, y_cluster))

    return report


def print_report(report: dict) -> None:
    print(f"\n=== {report['variant']} F={report['horizon']} ===")
    print(f"  params: {report['params']}")
    print(f"  spatial_organization_type: {report['spatial_organization_type']}")
    print(f"  temporal forecasts: {report['temporal_shapes']}")
    print(f"  final temporal:       {report['final_temporal_shape']}")
    print(f"  spatial stages:       {report.get('spatial_stage_shapes', [])}")
    print(f"  final prediction:     {report['final_pred_shape']}")
    print(f"  loss:                 {report['loss']:.6f}")
    if "graph_resolution_sizes" in report:
        print(f"  graph sizes M_j:      {report['graph_resolution_sizes']}")
        print(f"  clustering:           {report['clustering_methods']}")
        print(f"  cache paths:          {report['cluster_cache_paths'][:2]}...")
        print(f"  cluster tensors:      {report['cluster_stage_shapes']}")
        print(f"  cluster residuals:    {report['cluster_residual_shapes']}")
        print(f"  lifted residuals:     {report['lifted_residual_shapes']}")
        print(f"  MAE temp-only:        {report['mae_before_graph_spatial']:.4f}")
        print(f"  MAE after steps:      {report['mae_after_graph_steps']}")
        print(f"  residual energy:      {report['residual_energy_cluster']}")


def test_interleaved_unchanged() -> None:
    args = model_args_for("chain_interleaved_progressive_spatial", 12)
    args["chain_supervision_source"] = "spatial_chain"
    args["spatial_placement"] = "interleaved_progressive"
    args["progressive_spatial_ratios"] = [0.25, 0.5, 1.0]
    args["progressive_spatial_topks"] = [8, 16, 32]
    args["progressive_spatial_alphas"] = [0.03, 0.06, 0.10]
    model = ChainForecasting(**args)
    history = torch.randn(2, 12, 307, 4)
    out = model(history, return_all=True)
    for i in range(3):
        assert torch.equal(out["chain_preds"][i], out["spatial_preds"][i])
    print("[ok] chain_interleaved_progressive_spatial unchanged")


def main() -> int:
    reports = [check_variant(v, h) for h in HORIZONS for v in ACTIVE_VARIANTS]
    for r in reports:
        print_report(r)
    test_interleaved_unchanged()
    print("\nAll MTSR graph-resolution smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
