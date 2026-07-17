"""CapDistRefine spatial stack: S1/2 (cap+dist cluster) -> S1 (full node)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.gcn import ABCDSpatialModule
from basicts.archs.arch_zoo.ChainForecasting_arch.graph_cluster_utils import (
    cluster_graph_adjacency_normalized,
    load_or_build_capdist_half_cluster,
    row_normalize_square,
    validate_cluster_assignment,
)


class CapDistRefineSpatialStack(nn.Module):
    """Two-stage spatial refinement with cluster-mixed propagation."""

    MODEL_NAME = "CapDistRefine"

    def __init__(
        self,
        node_size: int,
        input_len: int,
        output_len: int,
        chain_lengths: list[int],
        d_spa: int,
        if_spatial: bool,
        spatial_scheme: str,
        adj_mx_path: str | None,
        post_spatial_mode: str = "adaptive_cluster_mix",
        capdist_cluster_method: str = "capdist_spectral_pair",
        capdist_use_road_distance: bool = True,
        capdist_sigma_d: float = 0.5,
        capdist_lambda_d: float = 0.1,
        capdist_lambda_mix: list[float] | None = None,
        capdist_alphas: list[float] | None = None,
        capdist_topks: list[int] | None = None,
        clustering_seed: int = 0,
        dataset_name: str = "PEMS04",
        cluster_cache_dir: str | Path | None = None,
        cluster_road_distance_path: str | Path | None = None,
        unified_aux_loss_mode: str = "none",
        adp_hidden_dim: int = 32,
        adp_tau: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.node_size = int(node_size)
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.chain_lengths = list(chain_lengths)
        self.post_spatial_mode = str(post_spatial_mode).lower()
        self.capdist_cluster_method = str(capdist_cluster_method)
        self.capdist_use_road_distance = bool(capdist_use_road_distance)
        self.capdist_sigma_d = float(capdist_sigma_d)
        self.capdist_lambda_d = float(capdist_lambda_d)
        self.clustering_seed = int(clustering_seed)
        self.dataset_name = dataset_name
        self.unified_aux_loss_mode = str(unified_aux_loss_mode).lower()

        self.stage_tags = ["S12", "S1"]
        self.stage_capacities = [2, 1]
        self.graph_resolution_sizes = [
            int(np.ceil(self.node_size / 2)),
            self.node_size,
        ]
        self.capdist_lambda_mix = list(capdist_lambda_mix or [0.5, 0.3])
        self.capdist_alphas = list(capdist_alphas or [0.08, 0.08])
        self.capdist_topks = list(capdist_topks or [8, 16])
        self.capdist_betas = [1.0, 1.0]

        half_meta, half_cache = load_or_build_capdist_half_cluster(
            node_size=self.node_size,
            adj_mx_path=adj_mx_path,
            seed=self.clustering_seed,
            dataset_name=dataset_name,
            cache_dir=cluster_cache_dir,
            cluster_road_distance_path=cluster_road_distance_path,
            cluster_sigma_d=self.capdist_sigma_d,
            cluster_lambda_d=self.capdist_lambda_d,
            use_road_distance=self.capdist_use_road_distance,
        )
        half_meta["resolution_tag"] = "S12"
        half_meta["capacity"] = 2
        half_meta["cache_path"] = str(half_cache)
        half_val = half_meta.get("validation") or validate_cluster_assignment(half_meta)

        affinity_w = half_meta["affinity_W"]
        self.register_buffer("affinity_W", torch.from_numpy(affinity_w))

        self.cluster_meta: list[dict[str, Any]] = [half_meta]
        self.cluster_cache_paths = [str(half_cache)]

        self.register_buffer("stage0_C", torch.from_numpy(half_meta["C"]))
        self.register_buffer("stage0_P", torch.from_numpy(half_meta["P"]))
        a_half = cluster_graph_adjacency_normalized(
            affinity_w, half_meta["P"], half_meta["C"]
        )
        self.register_buffer("stage0_A_cluster", torch.from_numpy(a_half))

        eye = torch.eye(self.node_size, dtype=torch.float32)
        self.register_buffer("stage1_C", eye)
        self.register_buffer("stage1_P", eye)
        a_full = row_normalize_square(affinity_w)
        self.register_buffer("stage1_A_cluster", torch.from_numpy(a_full))

        self.cluster_meta.append(
            {
                "node_size": self.node_size,
                "num_clusters": self.node_size,
                "clustering_method": "identity",
                "resolution_tag": "S1",
                "capacity": 1,
                "cache_path": "",
            }
        )
        self.cluster_cache_paths.append("")

        self.spatial_modules = nn.ModuleList()
        for stage_idx, m_j in enumerate(self.graph_resolution_sizes):
            alpha = float(self.capdist_alphas[stage_idx])
            topk = int(self.capdist_topks[stage_idx])
            mix_lam = float(self.capdist_lambda_mix[stage_idx])
            cluster_adj = getattr(self, f"stage{stage_idx}_A_cluster")
            adp_k = min(topk, max(1, m_j - 1))
            self.spatial_modules.append(
                ABCDSpatialModule(
                    node_size=m_j,
                    input_len=self.input_len,
                    d_spa=d_spa,
                    if_spatial=if_spatial,
                    spatial_scheme=spatial_scheme,
                    adj_mx_path=None,
                    use_gcn=False,
                    use_dynamic_spatial=False,
                    use_adaptive_adj=True,
                    adp_hidden_dim=max(8, adp_hidden_dim),
                    adp_topk=adp_k,
                    adp_tau=adp_tau,
                    adp_alpha=alpha,
                    use_hybrid_graph=False,
                    hybrid_alpha=alpha,
                    post_spatial_mode=self.post_spatial_mode,
                    cluster_adj=cluster_adj,
                    cluster_graph_mix_lambda=mix_lam,
                )
            )

        labels = np.asarray(half_meta["labels"])
        sizes = np.bincount(labels.astype(np.int64))
        singleton_ratio = float((sizes == 1).sum() / max(len(sizes), 1))
        print(
            f"[CapDistRefine] MODEL_NAME={self.MODEL_NAME} "
            f"input_len={self.input_len} output_len={self.output_len} "
            f"chain_lengths={self.chain_lengths}"
        )
        print(
            f"[CapDistRefine] spatial_stages=S1/2->S1 "
            f"capdist_cluster_method={self.capdist_cluster_method} "
            f"capdist_use_road_distance={self.capdist_use_road_distance} "
            f"road_distance_used={half_meta.get('road_distance_used')} "
            f"affinity_source={half_meta.get('affinity_source')} "
            f"sigma_d={self.capdist_sigma_d} lambda_d={self.capdist_lambda_d}"
        )
        print(
            f"[CapDistRefine] M_half={half_meta['num_clusters']} "
            f"max_cluster_size={half_val.get('max_cluster_size')} "
            f"min_cluster_size={half_val.get('min_cluster_size')} "
            f"singleton_ratio={singleton_ratio:.3f} "
            f"cache={half_cache}"
        )
        print(
            f"[CapDistRefine] lambda_mix_half={self.capdist_lambda_mix[0]} "
            f"lambda_mix_full={self.capdist_lambda_mix[1]} "
            f"alpha_half={self.capdist_alphas[0]} alpha_full={self.capdist_alphas[1]} "
            f"topk_half={self.capdist_topks[0]} topk_full={self.capdist_topks[1]} "
            f"post_spatial_mode={self.post_spatial_mode} "
            f"unified_aux_loss_mode={self.unified_aux_loss_mode}"
        )

    def _project_nodes(self, node_x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.einsum("mn,btnc->btmc", p, node_x)

    def _lift_clusters(self, cluster_r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nm,btmc->btnc", c, cluster_r)

    def _project_history(self, history_flow: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.einsum("mn,bln->blm", p, history_flow)

    def forward(
        self,
        forecast: torch.Tensor,
        history_flow: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        u_node = forecast
        node_stage_preds: list[torch.Tensor] = [u_node]
        node_before_preds: list[torch.Tensor] = []
        cluster_residuals: list[torch.Tensor] = []
        lifted_residuals: list[torch.Tensor] = []
        projection_matrices: list[torch.Tensor] = []
        residual_energy_cluster: list[float] = []
        residual_energy_lifted: list[float] = []
        cluster_mix_diag: list[dict[str, float]] = []

        for stage_idx, module in enumerate(self.spatial_modules):
            m_j = self.graph_resolution_sizes[stage_idx]
            beta = float(self.capdist_betas[stage_idx])
            c = getattr(self, f"stage{stage_idx}_C")
            p = getattr(self, f"stage{stage_idx}_P")
            projection_matrices.append(p)
            node_before_preds.append(u_node)

            if m_j < self.node_size:
                u_cluster = self._project_nodes(u_node, p)
                hist_cluster = self._project_history(history_flow, p)
            else:
                u_cluster = u_node
                hist_cluster = history_flow

            u_tilde = module.refine_prediction(u_cluster, hist_cluster)
            r_cluster = u_tilde - u_cluster
            cluster_residuals.append(r_cluster)

            if m_j < self.node_size:
                lifted = self._lift_clusters(r_cluster, c)
            else:
                lifted = r_cluster
            lifted_residuals.append(lifted)

            with torch.no_grad():
                residual_energy_cluster.append(float(r_cluster.abs().mean().item()))
                residual_energy_lifted.append(float(lifted.abs().mean().item()))
                if hasattr(module, "get_cluster_mix_diagnostics"):
                    cluster_mix_diag.append(module.get_cluster_mix_diagnostics())

            u_node = u_node + beta * lifted
            node_stage_preds.append(u_node)

        if not return_diagnostics:
            return u_node

        out: dict[str, Any] = {
            "pred": u_node,
            "temporal_input": forecast,
            "node_stage_preds": node_stage_preds[1:],
            "node_stage_preds_all": node_stage_preds,
            "node_before_preds": node_before_preds,
            "cluster_residuals": cluster_residuals,
            "lifted_residuals": lifted_residuals,
            "graph_projection_matrices": projection_matrices,
            "graph_ratios": self.stage_tags,
            "residual_energy_cluster": residual_energy_cluster,
            "residual_energy_lifted": residual_energy_lifted,
            "graph_resolution_sizes": self.graph_resolution_sizes,
            "graph_resolution_ratios": self.stage_tags,
            "clustering_methods": [m.get("clustering_method", "") for m in self.cluster_meta],
            "cluster_cache_paths": self.cluster_cache_paths,
            "spatial_operator_type": self.post_spatial_mode,
            "graph_resolution_alphas": self.capdist_alphas,
            "graph_resolution_topks": self.capdist_topks,
            "graph_resolution_betas": self.capdist_betas,
            "model_name": self.MODEL_NAME,
            "capdist_lambda_mix": self.capdist_lambda_mix,
        }
        if cluster_mix_diag:
            out["cluster_mix_diagnostics"] = cluster_mix_diag
            if cluster_mix_diag:
                out["a_cluster_density"] = cluster_mix_diag[0].get("a_cluster_density")
                out["a_adp_density"] = cluster_mix_diag[0].get("a_adp_density")
                out["a_cluster_adp_mean_abs_diff"] = cluster_mix_diag[0].get(
                    "a_cluster_adp_mean_abs_diff"
                )
            if len(cluster_mix_diag) > 1:
                out["a_cluster_density_full"] = cluster_mix_diag[1].get("a_cluster_density")
                out["a_adp_density_full"] = cluster_mix_diag[1].get("a_adp_density")
        return out

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.MODEL_NAME,
            "spatial_stages": "S1/2->S1",
            "graph_resolution_sizes": self.graph_resolution_sizes,
            "capdist_cluster_method": self.capdist_cluster_method,
            "capdist_use_road_distance": self.capdist_use_road_distance,
            "capdist_sigma_d": self.capdist_sigma_d,
            "capdist_lambda_d": self.capdist_lambda_d,
            "capdist_lambda_mix": self.capdist_lambda_mix,
            "capdist_alphas": self.capdist_alphas,
            "capdist_topks": self.capdist_topks,
            "clustering_methods": [m.get("clustering_method", "") for m in self.cluster_meta],
            "cluster_cache_paths": self.cluster_cache_paths,
            "post_spatial_mode": self.post_spatial_mode,
            "unified_aux_loss_mode": self.unified_aux_loss_mode,
        }
