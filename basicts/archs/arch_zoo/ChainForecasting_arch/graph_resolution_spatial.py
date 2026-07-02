"""Graph-resolution spatial residualization for temporal-first MTSR variants."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.gcn import ABCDSpatialModule
from basicts.archs.arch_zoo.ChainForecasting_arch.graph_cluster_utils import (
    load_or_build_cluster_assignment,
    resolve_graph_resolution_sizes,
)


class GraphResolutionSpatialStack(nn.Module):
    """Apply adaptive-only spatial residuals across coarse-to-fine graph resolutions.

    Layout convention (matches ChainForecasting): forecast [B, T, N, C], history_flow [B, L, N].
    Projection: ``einsum("mn,btnc->btmc", P, node_x)``
    Lifting: ``einsum("nm,btmc->btnc", C, cluster_r)``
    """

    def __init__(
        self,
        node_size: int,
        input_len: int,
        d_spa: int,
        if_spatial: bool,
        spatial_scheme: str,
        adj_mx_path: str | None,
        post_spatial_mode: str,
        graph_resolution_ratios: list[float] | None = None,
        graph_resolution_alphas: list[float] | None = None,
        graph_resolution_topks: list[int] | None = None,
        graph_resolution_betas: list[float] | None = None,
        graph_resolution_rhos: list[float] | None = None,
        adp_hidden_dim: int = 32,
        adp_tau: float = 0.5,
        clustering_seed: int = 0,
        dataset_name: str = "unknown",
        cluster_cache_dir: str | Path | None = None,
        **kwargs,
    ):
        super().__init__()
        self.node_size = int(node_size)
        self.input_len = int(input_len)
        self.post_spatial_mode = str(post_spatial_mode).lower()
        self.adj_mx_path = adj_mx_path
        self.dataset_name = dataset_name
        self.clustering_seed = int(clustering_seed)
        self.cluster_cache_dir = cluster_cache_dir

        ratios = list(graph_resolution_ratios or [0.25, 0.50, 1.00])
        self.graph_resolution_ratios = ratios
        self.graph_resolution_sizes = resolve_graph_resolution_sizes(self.node_size, ratios)

        alphas = list(graph_resolution_alphas or [0.03, 0.06, 0.10])
        topks = list(graph_resolution_topks or [8, 16, 32])
        betas = list(graph_resolution_betas or [1.0, 1.0, 1.0])
        rhos = list(graph_resolution_rhos or ratios)

        num_stages = len(self.graph_resolution_sizes)
        self.graph_resolution_alphas = self._fit_list(alphas, num_stages)
        self.graph_resolution_topks = self._fit_list(topks, num_stages)
        self.graph_resolution_betas = self._fit_list(betas, num_stages)
        self.graph_resolution_rhos = self._fit_list(rhos, num_stages)

        self.cluster_meta: list[dict[str, Any]] = []
        self.cluster_cache_paths: list[str] = []
        self.spatial_modules = nn.ModuleList()

        for stage_idx, m_j in enumerate(self.graph_resolution_sizes):
            ratio = float(self.graph_resolution_rhos[stage_idx])
            alpha = float(self.graph_resolution_alphas[stage_idx])
            topk = int(self.graph_resolution_topks[stage_idx])

            if m_j < self.node_size:
                meta, cache_path = load_or_build_cluster_assignment(
                    node_size=self.node_size,
                    num_clusters=m_j,
                    adj_mx_path=adj_mx_path,
                    seed=clustering_seed,
                    dataset_name=dataset_name,
                    cache_dir=cluster_cache_dir,
                )
                self.cluster_meta.append(meta)
                self.cluster_cache_paths.append(str(cache_path))
                self.register_buffer(f"stage{stage_idx}_C", torch.from_numpy(meta["C"]))
                self.register_buffer(f"stage{stage_idx}_P", torch.from_numpy(meta["P"]))
            else:
                self.cluster_meta.append(
                    {
                        "node_size": self.node_size,
                        "num_clusters": self.node_size,
                        "clustering_method": "identity",
                    }
                )
                self.cluster_cache_paths.append("")
                eye_c = torch.eye(self.node_size, dtype=torch.float32)
                self.register_buffer(f"stage{stage_idx}_C", eye_c)
                self.register_buffer(f"stage{stage_idx}_P", eye_c)

            adp_dim = max(8, int(adp_hidden_dim * ratio))
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
                    adp_hidden_dim=adp_dim,
                    adp_topk=adp_k,
                    adp_tau=adp_tau,
                    adp_alpha=alpha,
                    use_hybrid_graph=False,
                    hybrid_alpha=alpha,
                    post_spatial_mode=self.post_spatial_mode,
                )
            )

    @staticmethod
    def _fit_list(values: list, num_stages: int) -> list:
        values = list(values)
        if len(values) >= num_stages:
            return values[:num_stages]
        return values + [values[-1]] * (num_stages - len(values))

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
        cluster_stage_preds: list[torch.Tensor] = []
        cluster_residuals: list[torch.Tensor] = []
        lifted_residuals: list[torch.Tensor] = []
        residual_energy_cluster: list[float] = []
        residual_energy_lifted: list[float] = []

        for stage_idx, module in enumerate(self.spatial_modules):
            m_j = self.graph_resolution_sizes[stage_idx]
            beta = float(self.graph_resolution_betas[stage_idx])
            c = getattr(self, f"stage{stage_idx}_C")
            p = getattr(self, f"stage{stage_idx}_P")

            if m_j < self.node_size:
                u_cluster = self._project_nodes(u_node, p)
                hist_cluster = self._project_history(history_flow, p)
            else:
                u_cluster = u_node
                hist_cluster = history_flow

            cluster_stage_preds.append(u_cluster)
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

            u_node = u_node + beta * lifted
            node_stage_preds.append(u_node)

        if not return_diagnostics:
            return u_node

        return {
            "pred": u_node,
            "node_stage_preds": node_stage_preds[1:],
            "cluster_stage_preds": cluster_stage_preds,
            "cluster_residuals": cluster_residuals,
            "lifted_residuals": lifted_residuals,
            "residual_energy_cluster": residual_energy_cluster,
            "residual_energy_lifted": residual_energy_lifted,
            "graph_resolution_sizes": list(self.graph_resolution_sizes),
            "graph_resolution_ratios": list(self.graph_resolution_ratios),
            "clustering_methods": [m.get("clustering_method", "") for m in self.cluster_meta],
            "cluster_cache_paths": list(self.cluster_cache_paths),
            "spatial_operator_type": self.post_spatial_mode,
            "graph_resolution_alphas": list(self.graph_resolution_alphas),
            "graph_resolution_topks": list(self.graph_resolution_topks),
            "graph_resolution_betas": list(self.graph_resolution_betas),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "graph_resolution_ratios": self.graph_resolution_ratios,
            "graph_resolution_sizes": self.graph_resolution_sizes,
            "clustering_methods": [m.get("clustering_method", "") for m in self.cluster_meta],
            "cluster_cache_paths": self.cluster_cache_paths,
            "spatial_operator_type": self.post_spatial_mode,
            "graph_resolution_alphas": self.graph_resolution_alphas,
            "graph_resolution_topks": self.graph_resolution_topks,
            "graph_resolution_betas": self.graph_resolution_betas,
        }
