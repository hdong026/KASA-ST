"""Graph-resolution spatial residualization for temporal-first MTSR variants."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.gcn import ABCDSpatialModule
from basicts.archs.arch_zoo.ChainForecasting_arch.graph_cluster_utils import (
    CAPACITY_CAPDIST_METHODS,
    CAPACITY_MULTILEVEL_METHODS,
    CAPACITY_SINGLE_STAGE_METHODS,
    capacity_stage_tag,
    cluster_graph_adjacency_normalized,
    load_or_build_capdist_spectral_cluster,
    load_or_build_cluster_assignment,
    load_or_build_multilevel_cluster_assignments,
    load_raw_adj_numpy,
    num_clusters_for_capacity,
    resolve_graph_resolution_capacities,
    resolve_graph_resolution_sizes,
    row_normalize_square,
    symmetrize_adjacency,
    validate_cluster_assignment,
)


class GraphResolutionSpatialStack(nn.Module):
    """Apply spatial residuals across coarse-to-fine graph resolutions.

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
        graph_resolution_capacities: list[int] | None = None,
        graph_resolution_skip_final_identity: bool = False,
        graph_resolution_alphas: list[float] | None = None,
        graph_resolution_topks: list[int] | None = None,
        graph_resolution_betas: list[float] | None = None,
        graph_resolution_rhos: list[float] | None = None,
        adp_hidden_dim: int = 32,
        adp_tau: float = 0.5,
        clustering_seed: int = 0,
        dataset_name: str = "unknown",
        cluster_cache_dir: str | Path | None = None,
        graph_cluster_method: str = "current",
        graph_cluster_affinity: str | None = None,
        cluster_train_series_path: str | Path | None = None,
        cluster_spatial_coord_path: str | Path | None = None,
        cluster_road_distance_path: str | Path | None = None,
        cluster_sigma_d: float = 0.5,
        cluster_road_delta: float | None = None,
        cluster_delta_4: float = 0.8,
        cluster_delta_2: float = 0.5,
        cluster_max_lag: int = 12,
        cluster_lambda_s: float = 0.2,
        cluster_acf_lag: int = 24,
        cluster_graph_mix_lambda: float = 0.5,
        cluster_graph_mix_lambdas: list[float] | None = None,
        capdist_sigma_d: float | None = None,
        capdist_lambda_d: float | None = None,
        capdist_use_road_distance: bool = True,
        capdist_use_hard_cutoff: bool = False,
        data_dir: str | Path | None = None,
        variant_name: str = "",
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
        self.graph_cluster_method = str(graph_cluster_method).lower()
        self.graph_cluster_affinity = str(graph_cluster_affinity or "").lower()
        self.cluster_train_series_path = cluster_train_series_path
        self.cluster_spatial_coord_path = cluster_spatial_coord_path
        self.cluster_road_distance_path = cluster_road_distance_path
        self.cluster_sigma_d = float(cluster_sigma_d)
        self.cluster_road_delta = cluster_road_delta
        self.cluster_delta_4 = float(cluster_delta_4)
        self.cluster_delta_2 = float(cluster_delta_2)
        self.cluster_max_lag = int(cluster_max_lag)
        self.cluster_lambda_s = float(cluster_lambda_s)
        self.cluster_acf_lag = int(cluster_acf_lag)
        self.cluster_graph_mix_lambda = float(cluster_graph_mix_lambda)
        self.cluster_graph_mix_lambdas = (
            list(cluster_graph_mix_lambdas) if cluster_graph_mix_lambdas is not None else None
        )
        self.capdist_sigma_d = float(capdist_sigma_d if capdist_sigma_d is not None else cluster_sigma_d)
        self.capdist_lambda_d = float(capdist_lambda_d if capdist_lambda_d is not None else 0.05)
        self.capdist_use_road_distance = bool(capdist_use_road_distance)
        self.capdist_use_hard_cutoff = bool(capdist_use_hard_cutoff)
        if self.capdist_use_hard_cutoff:
            raise ValueError("capdist_use_hard_cutoff=True is not supported; use soft distance penalty only.")
        self.data_dir = data_dir
        self.variant_name = str(variant_name or "")

        ratios = list(graph_resolution_ratios) if graph_resolution_ratios is not None else []
        capacities = list(graph_resolution_capacities or [])
        self.graph_resolution_ratios = ratios
        self.graph_resolution_capacities = capacities
        self.graph_resolution_skip_final_identity = bool(graph_resolution_skip_final_identity)
        self.s1_stage_disabled = bool(graph_resolution_skip_final_identity)

        self.use_single_capacity_schedule = (
            self.graph_cluster_method in CAPACITY_SINGLE_STAGE_METHODS
            or (
                bool(capacities)
                and graph_resolution_skip_final_identity
                and self.graph_cluster_method not in CAPACITY_MULTILEVEL_METHODS
            )
        )
        self.use_capacity_schedule = (
            self.graph_cluster_method in CAPACITY_MULTILEVEL_METHODS
            and not self.use_single_capacity_schedule
        )
        self.use_capdist_staged_schedule = (
            self.graph_cluster_method in CAPACITY_CAPDIST_METHODS and bool(capacities)
        )

        self._adj_sym_np: np.ndarray | None = None
        self._affinity_w_np: np.ndarray | None = None
        if adj_mx_path:
            self._adj_sym_np = symmetrize_adjacency(load_raw_adj_numpy(adj_mx_path, self.node_size))

        self._stage_metas: list[dict[str, Any]] = []
        self._multilevel_cache_path = ""

        if self.use_capdist_staged_schedule:
            cap_schedule = resolve_graph_resolution_capacities(capacities or [4, 2, 1])
            self.graph_resolution_capacities = cap_schedule
            self.graph_resolution_sizes = []
            self._stage_metas = []
            for cap in cap_schedule:
                if int(cap) <= 1:
                    m_j = self.node_size
                    self.graph_resolution_sizes.append(m_j)
                    self._stage_metas.append(
                        {
                            "node_size": self.node_size,
                            "num_clusters": self.node_size,
                            "clustering_method": "identity",
                            "graph_cluster_method": self.graph_cluster_method,
                            "capacity": 1,
                            "resolution_tag": "S1",
                            "max_capacity": 1,
                        }
                    )
                    continue
                m_j = num_clusters_for_capacity(self.node_size, int(cap))
                meta, cache_path = load_or_build_capdist_spectral_cluster(
                    node_size=self.node_size,
                    max_capacity=int(cap),
                    adj_mx_path=adj_mx_path,
                    seed=clustering_seed,
                    dataset_name=dataset_name,
                    cache_dir=cluster_cache_dir,
                    cluster_road_distance_path=self.cluster_road_distance_path,
                    cluster_sigma_d=self.capdist_sigma_d,
                    cluster_lambda_d=self.capdist_lambda_d,
                    use_road_distance=self.capdist_use_road_distance,
                )
                meta["resolution_tag"] = capacity_stage_tag(int(cap))
                meta["capacity"] = int(cap)
                meta["cache_path"] = str(cache_path)
                self._stage_metas.append(meta)
                self.graph_resolution_sizes.append(m_j)
                if self._affinity_w_np is None and "affinity_W" in meta:
                    self._affinity_w_np = np.asarray(meta["affinity_W"], dtype=np.float32)
            print(
                f"[GraphResolution] capdist_staged method={self.graph_cluster_method} "
                f"capacities={cap_schedule} sizes={self.graph_resolution_sizes}"
            )
        elif self.use_single_capacity_schedule:
            cap_schedule = resolve_graph_resolution_capacities(
                capacities or [2], skip_final_identity=True
            )
            self.graph_resolution_capacities = cap_schedule
            self.graph_resolution_sizes = []
            for cap in cap_schedule:
                m_j = num_clusters_for_capacity(self.node_size, cap)
                meta, cache_path = load_or_build_cluster_assignment(
                    node_size=self.node_size,
                    num_clusters=m_j,
                    adj_mx_path=adj_mx_path,
                    seed=clustering_seed,
                    dataset_name=dataset_name,
                    cache_dir=cluster_cache_dir,
                    graph_cluster_method=self.graph_cluster_method,
                    cluster_train_series_path=self.cluster_train_series_path,
                    cluster_spatial_coord_path=self.cluster_spatial_coord_path,
                    cluster_road_distance_path=self.cluster_road_distance_path,
                    cluster_sigma_d=self.cluster_sigma_d,
                    cluster_road_delta=self.cluster_road_delta,
                    cluster_delta_4=self.cluster_delta_4,
                    cluster_delta_2=self.cluster_delta_2,
                    cluster_max_lag=self.cluster_max_lag,
                    cluster_lambda_s=self.cluster_lambda_s,
                    cluster_acf_lag=self.cluster_acf_lag,
                    data_dir=self.data_dir,
                    cluster_max_capacity=cap,
                )
                meta["resolution_tag"] = capacity_stage_tag(cap)
                meta["capacity"] = int(cap)
                meta["cache_path"] = str(cache_path)
                self._stage_metas.append(meta)
                self.graph_resolution_sizes.append(m_j)
            print(
                f"[GraphResolution] single_capacity method={self.graph_cluster_method} "
                f"capacities={cap_schedule} sizes={self.graph_resolution_sizes} "
                f"S1_disabled={self.s1_stage_disabled}"
            )
        elif self.use_capacity_schedule:
            cap_schedule = resolve_graph_resolution_capacities(capacities or [4, 2, 1])
            self.graph_resolution_capacities = cap_schedule
            self.graph_resolution_sizes = []
            stage_bundle, multilevel_cache = load_or_build_multilevel_cluster_assignments(
                node_size=self.node_size,
                capacities=cap_schedule,
                adj_mx_path=adj_mx_path,
                seed=clustering_seed,
                dataset_name=dataset_name,
                cache_dir=cluster_cache_dir,
                graph_cluster_method=self.graph_cluster_method,
                cluster_road_distance_path=self.cluster_road_distance_path,
                cluster_sigma_d=self.cluster_sigma_d,
                cluster_road_delta=self.cluster_road_delta,
            )
            self._multilevel_cache_path = str(multilevel_cache)
            self._stage_metas = stage_bundle
            self.graph_resolution_sizes = [int(st["num_clusters"]) for st in stage_bundle]
            print(
                f"[GraphResolution] method={self.graph_cluster_method} "
                f"capacities={cap_schedule} sizes={self.graph_resolution_sizes} "
                f"road_used={stage_bundle[0].get('road_distance_used', False)} "
                f"nested={stage_bundle[0].get('nested_consistency')} "
                f"cache={multilevel_cache}"
            )
        else:
            ratio_schedule = ratios or [0.25, 0.50, 1.00]
            self.graph_resolution_ratios = ratio_schedule
            self.graph_resolution_sizes = resolve_graph_resolution_sizes(
                self.node_size,
                ratio_schedule,
                skip_final_identity=graph_resolution_skip_final_identity,
            )
            self._stage_metas = []
            self._multilevel_cache_path = ""

        default_rhos = ratios if ratios else [1.0] * len(self.graph_resolution_sizes)
        rhos = list(graph_resolution_rhos or default_rhos)
        alphas = list(graph_resolution_alphas or [0.03, 0.06, 0.10])
        topks = list(graph_resolution_topks or [8, 16, 32])
        betas = list(graph_resolution_betas or [1.0, 1.0, 1.0])

        num_stages = len(self.graph_resolution_sizes)
        self.graph_resolution_alphas = self._fit_list(alphas, num_stages)
        self.graph_resolution_topks = self._fit_list(topks, num_stages)
        self.graph_resolution_betas = self._fit_list(betas, num_stages)
        self.graph_resolution_rhos = self._fit_list(rhos, num_stages)
        mix_lambdas = self._fit_list(
            self.cluster_graph_mix_lambdas
            if self.cluster_graph_mix_lambdas is not None
            else [self.cluster_graph_mix_lambda] * num_stages,
            num_stages,
        )
        self.cluster_graph_mix_lambdas = mix_lambdas

        self.cluster_meta: list[dict[str, Any]] = []
        self.cluster_cache_paths: list[str] = []
        self.spatial_modules = nn.ModuleList()

        for stage_idx, m_j in enumerate(self.graph_resolution_sizes):
            ratio = float(self.graph_resolution_rhos[stage_idx])
            alpha = float(self.graph_resolution_alphas[stage_idx])
            topk = int(self.graph_resolution_topks[stage_idx])

            cluster_adj_tensor = None
            if m_j < self.node_size:
                if self.use_capdist_staged_schedule:
                    meta = dict(self._stage_metas[stage_idx])
                    cache_path = Path(meta.get("cache_path", ""))
                elif self.use_capacity_schedule:
                    meta = dict(self._stage_metas[stage_idx])
                    cache_path = Path(self._multilevel_cache_path)
                elif self.use_single_capacity_schedule:
                    meta = dict(self._stage_metas[stage_idx])
                    cache_path = Path(meta.get("cache_path", ""))
                else:
                    stage_ratio = (
                        float(self.graph_resolution_ratios[stage_idx])
                        if stage_idx < len(self.graph_resolution_ratios)
                        else 1.0
                    )
                    meta, cache_path = load_or_build_cluster_assignment(
                        node_size=self.node_size,
                        num_clusters=m_j,
                        adj_mx_path=adj_mx_path,
                        seed=clustering_seed,
                        dataset_name=dataset_name,
                        cache_dir=cluster_cache_dir,
                        graph_cluster_method=self.graph_cluster_method,
                        cluster_train_series_path=self.cluster_train_series_path,
                        cluster_spatial_coord_path=self.cluster_spatial_coord_path,
                        cluster_road_distance_path=self.cluster_road_distance_path,
                        cluster_sigma_d=self.cluster_sigma_d,
                        cluster_road_delta=self.cluster_road_delta,
                        cluster_delta_4=self.cluster_delta_4,
                        cluster_delta_2=self.cluster_delta_2,
                        ratio=stage_ratio,
                        cluster_max_lag=self.cluster_max_lag,
                        cluster_lambda_s=self.cluster_lambda_s,
                        cluster_acf_lag=self.cluster_acf_lag,
                        data_dir=self.data_dir,
                    )
                val = validate_cluster_assignment(meta)
                meta["validation"] = val
                self.cluster_meta.append(meta)
                self.cluster_cache_paths.append(str(cache_path))
                self.register_buffer(f"stage{stage_idx}_C", torch.from_numpy(meta["C"]))
                self.register_buffer(f"stage{stage_idx}_P", torch.from_numpy(meta["P"]))

                if self.post_spatial_mode == "adaptive_cluster_mix":
                    w_np = meta.get("affinity_W")
                    if w_np is None:
                        w_np = self._affinity_w_np if self._affinity_w_np is not None else self._adj_sym_np
                    if w_np is None:
                        raise RuntimeError(
                            "adaptive_cluster_mix requires affinity_W or adj_sym for A_cluster construction."
                        )
                    if self._affinity_w_np is None:
                        self._affinity_w_np = np.asarray(w_np, dtype=np.float32)
                    a_cluster = cluster_graph_adjacency_normalized(
                        np.asarray(w_np, dtype=np.float64), meta["P"], meta["C"]
                    )
                    cluster_adj_tensor = torch.from_numpy(a_cluster)
                    self.register_buffer(f"stage{stage_idx}_A_cluster", cluster_adj_tensor)

                self._log_stage_config(
                    stage_idx, meta, val, cache_path, mix_lambdas[stage_idx]
                )
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
                if self.post_spatial_mode == "adaptive_cluster_mix":
                    w_np = self._affinity_w_np if self._affinity_w_np is not None else self._adj_sym_np
                    if w_np is None:
                        raise RuntimeError(
                            "adaptive_cluster_mix on S1 requires affinity_W or adj_sym for RowNorm(W)."
                        )
                    a_cluster = row_normalize_square(np.asarray(w_np, dtype=np.float64))
                    cluster_adj_tensor = torch.from_numpy(a_cluster)
                    self.register_buffer(f"stage{stage_idx}_A_cluster", cluster_adj_tensor)

            adp_dim = max(8, int(adp_hidden_dim * ratio))
            adp_k = min(topk, max(1, m_j - 1))
            ms_kwargs = {
                k: kwargs[k]
                for k in (
                    "adaptive_ms_topks",
                    "adaptive_ms_alpha",
                    "adaptive_ms_fusion",
                    "adaptive_ms_share_logits",
                    "adaptive_ms_init",
                )
                if k in kwargs
            }
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
                    cluster_adj=cluster_adj_tensor,
                    cluster_graph_mix_lambda=mix_lambdas[stage_idx],
                    **ms_kwargs,
                )
            )

        if self.variant_name == "GR7_capdist_mix":
            self._log_capdist_variant_startup()

    def _log_capdist_variant_startup(self) -> None:
        road_used = False
        affinity_source = "unknown"
        for meta in self.cluster_meta:
            if meta.get("clustering_method") != "identity":
                road_used = bool(meta.get("road_distance_used", road_used))
                affinity_source = str(meta.get("affinity_source", affinity_source))
        print(
            f"[GR7_capdist_mix] variant={self.variant_name} "
            f"spatial_placement=temporal_first_graph_resolution "
            f"post_spatial_mode={self.post_spatial_mode} "
            f"graph_cluster_method={self.graph_cluster_method} "
            f"graph_resolution_capacities={self.graph_resolution_capacities} "
            f"graph_resolution_topks={self.graph_resolution_topks} "
            f"graph_resolution_alphas={self.graph_resolution_alphas} "
            f"cluster_graph_mix_lambdas={self.cluster_graph_mix_lambdas} "
            f"capdist_sigma_d={self.capdist_sigma_d} "
            f"capdist_lambda_d={self.capdist_lambda_d} "
            f"capdist_use_hard_cutoff={self.capdist_use_hard_cutoff} "
            f"road_distance_used={road_used} affinity_source={affinity_source}"
        )
        for meta in self.cluster_meta:
            if meta.get("clustering_method") == "identity":
                continue
            val = meta.get("validation") or validate_cluster_assignment(meta)
            labels = np.asarray(meta.get("labels", []))
            sizes = np.bincount(labels.astype(np.int64)) if labels.size else np.array([])
            tag = meta.get("resolution_tag", capacity_stage_tag(int(meta.get("capacity", 0))))
            print(
                f"[GR7_capdist_mix] stage={tag} M={meta.get('num_clusters')} "
                f"cluster_sizes={sizes.tolist()} max_cluster_size={val.get('max_cluster_size')}"
            )

    def _log_stage_config(
        self,
        stage_idx: int,
        meta: dict[str, Any],
        val: dict[str, Any],
        cache_path: Path | str,
        mix_lambda: float,
    ) -> None:
        labels = np.asarray(meta.get("labels", []))
        sizes = np.bincount(labels.astype(np.int64)) if labels.size else np.array([])
        singleton_ratio = float((sizes == 1).sum() / max(len(sizes), 1)) if sizes.size else 0.0
        aff = meta.get("affinity_source", self.graph_cluster_affinity or "unknown")
        road = bool(meta.get("road_distance_used", False))
        cap_k = int(meta.get("max_capacity", meta.get("capacity", 0)) or 0)
        print(
            f"[GraphResolution] stage={stage_idx} "
            f"variant={self.variant_name or 'graph_resolution'} "
            f"method={meta.get('graph_cluster_method', self.graph_cluster_method)} "
            f"affinity={aff} road={road} capacity_K={cap_k} "
            f"M={meta.get('num_clusters')} max_size={val.get('max_cluster_size')} "
            f"singleton_ratio={singleton_ratio:.3f} "
            f"post_spatial_mode={self.post_spatial_mode} "
            f"cluster_graph_mix_lambda={mix_lambda} "
            f"alpha={self.graph_resolution_alphas[stage_idx]} "
            f"topk={self.graph_resolution_topks[stage_idx]} "
            f"S1_disabled={self.s1_stage_disabled} cache={cache_path}"
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

    def forward_stagewise(
        self,
        forecast: torch.Tensor,
        history_flow: torch.Tensor,
        active_stage_idx: int,
        max_stage_idx: int,
        detach_previous: bool = True,
        train_all_spatial: bool = False,
        skip_stage_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        u_node = forecast
        node_stage_preds: list[torch.Tensor] = [u_node]
        node_before_preds: list[torch.Tensor] = []
        cluster_residuals: list[torch.Tensor] = []
        lifted_residuals: list[torch.Tensor] = []
        projection_matrices: list[torch.Tensor] = []
        skip_set = set(skip_stage_indices or [])

        for stage_idx, module in enumerate(self.spatial_modules):
            if stage_idx > max_stage_idx:
                break
            if stage_idx in skip_set:
                continue
            m_j = self.graph_resolution_sizes[stage_idx]
            beta = float(self.graph_resolution_betas[stage_idx])
            c = getattr(self, f"stage{stage_idx}_C")
            p = getattr(self, f"stage{stage_idx}_P")
            projection_matrices.append(p)
            y_prev = u_node.detach() if detach_previous and stage_idx > active_stage_idx else u_node
            node_before_preds.append(y_prev)

            if m_j < self.node_size:
                u_cluster = self._project_nodes(u_node, p)
                hist_cluster = self._project_history(history_flow, p)
            else:
                u_cluster = u_node
                hist_cluster = history_flow

            trainable = train_all_spatial or stage_idx == active_stage_idx
            ctx = torch.enable_grad() if trainable else torch.no_grad()
            with ctx:
                u_tilde = module.refine_prediction(u_cluster, hist_cluster)
            r_cluster = u_tilde - u_cluster
            cluster_residuals.append(r_cluster)

            if m_j < self.node_size:
                lifted = self._lift_clusters(r_cluster, c)
            else:
                lifted = r_cluster
            lifted_residuals.append(lifted)
            u_node = u_node + beta * lifted
            node_stage_preds.append(u_node)

        return {
            "pred": u_node,
            "temporal_input": forecast,
            "node_stage_preds": node_stage_preds[1:],
            "node_stage_preds_all": node_stage_preds,
            "node_before_preds": node_before_preds,
            "cluster_residuals": cluster_residuals,
            "lifted_residuals": lifted_residuals,
            "graph_projection_matrices": projection_matrices,
            "graph_ratios": list(self.graph_resolution_rhos),
            "graph_resolution_sizes": list(self.graph_resolution_sizes),
            "graph_resolution_ratios": list(self.graph_resolution_rhos),
            "graph_resolution_alphas": list(self.graph_resolution_alphas),
            "graph_resolution_betas": list(self.graph_resolution_betas),
        }

    def forward(
        self,
        forecast: torch.Tensor,
        history_flow: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        u_node = forecast
        node_stage_preds: list[torch.Tensor] = [u_node]
        node_before_preds: list[torch.Tensor] = []
        cluster_stage_preds: list[torch.Tensor] = []
        cluster_residuals: list[torch.Tensor] = []
        lifted_residuals: list[torch.Tensor] = []
        projection_matrices: list[torch.Tensor] = []
        residual_energy_cluster: list[float] = []
        residual_energy_lifted: list[float] = []
        cluster_mix_diag: list[dict[str, float]] = []

        for stage_idx, module in enumerate(self.spatial_modules):
            m_j = self.graph_resolution_sizes[stage_idx]
            beta = float(self.graph_resolution_betas[stage_idx])
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
            "cluster_stage_preds": cluster_stage_preds,
            "cluster_residuals": cluster_residuals,
            "lifted_residuals": lifted_residuals,
            "graph_projection_matrices": projection_matrices,
            "graph_ratios": list(self.graph_resolution_rhos),
            "residual_energy_cluster": residual_energy_cluster,
            "residual_energy_lifted": residual_energy_lifted,
            "graph_resolution_sizes": list(self.graph_resolution_sizes),
            "graph_resolution_ratios": list(self.graph_resolution_rhos),
            "clustering_methods": [m.get("clustering_method", "") for m in self.cluster_meta],
            "cluster_cache_paths": list(self.cluster_cache_paths),
            "spatial_operator_type": self.post_spatial_mode,
            "graph_resolution_alphas": list(self.graph_resolution_alphas),
            "graph_resolution_topks": list(self.graph_resolution_topks),
            "graph_resolution_betas": list(self.graph_resolution_betas),
            "s1_stage_disabled": self.s1_stage_disabled,
            "cluster_graph_mix_lambda": self.cluster_graph_mix_lambda,
            "cluster_graph_mix_lambdas": list(self.cluster_graph_mix_lambdas or []),
        }
        if cluster_mix_diag:
            out["cluster_mix_diagnostics"] = cluster_mix_diag
            last = cluster_mix_diag[-1]
            out["a_cluster_density"] = last.get("a_cluster_density")
            out["a_adp_density"] = last.get("a_adp_density")
            out["a_cluster_adp_mean_abs_diff"] = last.get("a_cluster_adp_mean_abs_diff")
        return out

    def metadata(self) -> dict[str, Any]:
        return {
            "graph_resolution_ratios": self.graph_resolution_ratios,
            "graph_resolution_capacities": self.graph_resolution_capacities,
            "graph_resolution_sizes": self.graph_resolution_sizes,
            "clustering_methods": [m.get("clustering_method", "") for m in self.cluster_meta],
            "cluster_cache_paths": self.cluster_cache_paths,
            "spatial_operator_type": self.post_spatial_mode,
            "graph_resolution_alphas": self.graph_resolution_alphas,
            "graph_resolution_topks": self.graph_resolution_topks,
            "graph_resolution_betas": self.graph_resolution_betas,
            "graph_cluster_method": self.graph_cluster_method,
            "graph_cluster_affinity": self.graph_cluster_affinity,
            "cluster_road_distance_path": self.cluster_road_distance_path,
            "cluster_sigma_d": self.cluster_sigma_d,
            "cluster_graph_mix_lambda": self.cluster_graph_mix_lambda,
            "cluster_graph_mix_lambdas": list(self.cluster_graph_mix_lambdas or []),
            "s1_stage_disabled": self.s1_stage_disabled,
            "multilevel_cache_path": getattr(self, "_multilevel_cache_path", ""),
        }
