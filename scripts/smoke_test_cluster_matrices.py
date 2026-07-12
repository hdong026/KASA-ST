#!/usr/bin/env python3
"""Smoke tests for PAM-based graph cluster matrices and GR variants forward."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from basicts.archs import ChainForecasting
from basicts.archs.arch_zoo.ChainForecasting_arch.graph_cluster_utils import (
    build_cluster_assignment,
    cluster_capacities,
    load_or_build_cluster_assignment,
    validate_cluster_assignment,
)
from scripts.run_graph_resolution_pems04 import VARIANT_SPECS, variant_spec

NODE_SIZE = 307
M_LIST = [77, 154]
ADJ = os.path.join(ROOT, "datasets", "PEMS04", "adj_mx.pkl")
SPATIAL = os.path.join(ROOT, "datasets", "raw_data", "PEMS04", "adj_PEMS04_distance.pkl")

METHODS = [
    "pearson_balanced_pam",
    "xcorr_balanced_pam",
    "joint_pearson_spatial_balanced_pam",
    "pearson_standard_pam",
    "autocorr_feature_balanced_pam",
]

FORWARD_VARIANTS = [
    "GR7_sparse_topk",
    "GR9_pearson_balanced_pam",
    "GR10_xcorr_balanced_pam",
    "GR11_joint_pearson_spatial_pam",
    "GR12_pearson_standard_pam",
    "GR13_autocorr_feature_pam",
]


def check_matrix(meta: dict, m: int) -> None:
    val = validate_cluster_assignment(meta)
    assert val["C_shape"] == [NODE_SIZE, m], val
    assert val["P_shape"] == [m, NODE_SIZE], val
    assert val["row_one_hot_ok"], "C rows must be one-hot"
    assert val["P_row_sum_ok"], "P rows must sum to 1"
    caps = cluster_capacities(NODE_SIZE, m)
    sizes = sorted(val["cluster_sizes"], reverse=True)
    expected = sorted(caps.tolist(), reverse=True)
    if "balanced_pam" in meta.get("clustering_method", ""):
        assert sizes == expected, f"balanced sizes {sizes} != {expected}"


def smoke_cluster_methods() -> None:
    for m in M_LIST:
        for method in METHODS:
            kwargs = {
                "node_size": NODE_SIZE,
                "num_clusters": m,
                "adj_mx_path": ADJ,
                "seed": 0,
                "dataset_name": "PEMS04",
                "graph_cluster_method": method,
                "cluster_max_lag": 12,
                "cluster_lambda_s": 0.2,
                "cluster_acf_lag": 24,
            }
            if method == "joint_pearson_spatial_balanced_pam":
                kwargs["cluster_spatial_coord_path"] = SPATIAL
            meta, cache_path = load_or_build_cluster_assignment(**kwargs)
            check_matrix(meta, m)
            print(f"[ok] {method} M={m} cache={cache_path.name}")


def model_args_for(variant: str) -> dict:
    spec = variant_spec(variant)
    base = {
        "node_size": NODE_SIZE,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 4,
        "main_input_dim": 3,
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
        "chain_lengths": [3, 6, 12],
        "chain_loss_weights": [0.2, 0.3, 1.0],
        "dataset_name": "PEMS04",
    }
    for k, v in spec.items():
        if v is not None:
            base[k] = v
    return base


def smoke_forward_variants() -> None:
    b, t, n = 2, 12, NODE_SIZE
    hist = torch.randn(b, 12, n, 4)
    fut = torch.randn(b, t, n, 4)
    for variant in FORWARD_VARIANTS:
        if variant not in VARIANT_SPECS:
            raise KeyError(variant)
        model = ChainForecasting(**model_args_for(variant))
        model.eval()
        with torch.no_grad():
            out = model(hist, fut, train=False, return_all=False)
        assert out.shape == (b, t, n, 1), f"{variant}: {out.shape}"
        print(f"[ok] forward {variant} -> {tuple(out.shape)}")


def main() -> int:
    smoke_cluster_methods()
    smoke_forward_variants()
    print("\nAll cluster matrix smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
