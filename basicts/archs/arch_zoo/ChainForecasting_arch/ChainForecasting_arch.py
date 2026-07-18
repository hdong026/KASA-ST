import torch
import torch.nn.functional as F
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.gcn import ABCDSpatialModule
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_hidden_step import KASAHiddenStep
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    KASATemporalStep,
    interpolate_forecast,
    interpolate_latent,
)


class ChainForecasting(nn.Module):
    """Forecast-state chain using KASA temporal operators: X -> Y3 -> Y6 -> Y12."""

    def __init__(self, **model_args):
        super().__init__()
        self.node_size = model_args["node_size"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.output_len = model_args["output_len"]
        self.patch_len = model_args["patch_len"]
        self.stride = model_args.get("stride", model_args.get("patch_stride", 4))
        self.td_size = model_args["td_size"]
        self.dw_size = model_args["dw_size"]
        self.d_td = model_args.get("d_td", 32)
        self.d_dw = model_args.get("d_dw", 32)
        self.d_d = model_args.get("d_d", model_args.get("d_model", 32))
        self.d_spa = model_args.get("d_spa", 32)
        self.num_layer = model_args.get("num_layer", 2)

        self.if_time_in_day = model_args.get("if_time_in_day", True)
        self.if_day_in_week = model_args.get("if_day_in_week", True)
        self.if_spatial = model_args.get("if_spatial", True)
        self.spatial_scheme = str(model_args.get("spatial_scheme", "C")).upper()

        self.use_patch_branch = model_args.get("use_patch_branch", True)
        self.use_downsample_branch = model_args.get("use_downsample_branch", True)
        self.use_linear_residual_branch = model_args.get("use_linear_residual_branch", True)
        self.patch_data_input_mode = model_args.get("patch_data_input_mode", "all")
        self.patch_embedding_mode = model_args.get("patch_embedding_mode", "serial_concat")
        self.patch_feature_dim = model_args.get("patch_feature_dim", None)

        self.chain_lengths = list(model_args.get("chain_lengths", [3, 6, 12]))
        self.chain_loss_weights = list(model_args.get("chain_loss_weights", [0.2, 0.3, 1.0]))
        self.use_prev_condition = model_args.get("use_prev_condition", True)
        self.architecture_mode = str(model_args.get("architecture_mode", "chain")).lower()
        if self.architecture_mode not in {"chain", "direct_matched"}:
            raise ValueError(
                f"Unsupported architecture_mode: {self.architecture_mode}. "
                "Expected 'chain' or 'direct_matched'."
            )
        self.matched_stage_lengths = list(
            model_args.get("matched_stage_lengths", self.chain_lengths)
        )
        self.matched_hidden_dim = int(
            model_args.get("matched_hidden_dim", model_args.get("latent_prop_dim", self.d_d))
        )
        self.propagation_mode = str(model_args.get("propagation_mode", "forecast_state")).lower()
        if self.propagation_mode not in {"forecast_state", "latent"}:
            raise ValueError(
                f"Unsupported propagation_mode: {self.propagation_mode}. "
                "Expected 'forecast_state' or 'latent'."
            )
        self.latent_prop_dim = int(model_args.get("latent_prop_dim", self.d_d))
        self.spatial_placement = self._resolve_spatial_placement(model_args)
        self.spatial_organization_type = self._resolve_spatial_organization_type(
            self.spatial_placement
        )
        self.chain_supervision_source = str(
            model_args.get("chain_supervision_source", "spatial_chain")
        ).lower()
        if self.chain_supervision_source not in {"spatial_chain", "temporal_chain"}:
            raise ValueError(
                f"Unsupported chain_supervision_source: {self.chain_supervision_source}. "
                "Expected 'spatial_chain' or 'temporal_chain'."
            )
        self.spatial_stage_loss_weights = list(
            model_args.get("spatial_stage_loss_weights", [0.0, 0.0, 1.0])
        )
        self.spatial_graph_loss_weights = list(
            model_args.get("spatial_graph_loss_weights", [0.0, 0.0, 0.0])
        )
        ratios_arg = model_args.get("graph_resolution_ratios")
        self.graph_resolution_ratios = list(ratios_arg) if ratios_arg is not None else None
        self.dataset_name = str(model_args.get("dataset_name", "PEMS04"))
        self.clustering_seed = int(model_args.get("clustering_seed", 0))
        self._last_graph_diagnostics = None
        self.post_spatial_mode = model_args.get("post_spatial_mode", "adaptive_only")
        self.use_pre_temporal_spatial_enhancement = model_args.get(
            "use_pre_temporal_spatial_enhancement", False
        )

        if self.architecture_mode == "direct_matched":
            if self.matched_stage_lengths[-1] != self.output_len:
                raise ValueError(
                    f"Last matched stage length {self.matched_stage_lengths[-1]} "
                    f"must equal output_len {self.output_len}"
                )
        elif self.chain_lengths[-1] != self.output_len:
            raise ValueError(
                f"Last chain length {self.chain_lengths[-1]} must equal output_len {self.output_len}"
            )

        self.td_codebook = None
        self.dw_codebook = None
        self.spa_codebook = None
        if self.if_time_in_day:
            self.td_codebook = nn.Parameter(torch.empty(self.td_size, self.d_td))
            nn.init.xavier_uniform_(self.td_codebook)
        if self.if_day_in_week:
            self.dw_codebook = nn.Parameter(torch.empty(self.dw_size, self.d_dw))
            nn.init.xavier_uniform_(self.dw_codebook)
        if self.if_spatial:
            self.spa_codebook = nn.Parameter(torch.empty(self.node_size, self.d_spa))
            nn.init.xavier_uniform_(self.spa_codebook)

        step_kwargs = dict(
            input_len=self.input_len,
            patch_len=self.patch_len,
            stride=self.stride,
            td_size=self.td_size,
            dw_size=self.dw_size,
            td_codebook=self.td_codebook,
            dw_codebook=self.dw_codebook,
            spa_codebook=self.spa_codebook,
            if_time_in_day=self.if_time_in_day,
            if_day_in_week=self.if_day_in_week,
            if_spatial=self.if_spatial,
            d_d=self.d_d,
            d_td=self.d_td,
            d_dw=self.d_dw,
            d_spa=self.d_spa,
            num_layer=self.num_layer,
            use_patch_branch=self.use_patch_branch,
            use_downsample_branch=self.use_downsample_branch,
            use_linear_residual_branch=self.use_linear_residual_branch,
            patch_data_input_mode=self.patch_data_input_mode,
            patch_embedding_mode=self.patch_embedding_mode,
            patch_feature_dim=self.patch_feature_dim,
            use_prev_condition=self.use_prev_condition,
        )

        self.temporal_steps = nn.ModuleList()
        self.hidden_steps = nn.ModuleList()
        self.final_temporal_step = None
        self.latent_encoders = nn.ModuleList()

        if self.architecture_mode == "direct_matched":
            matched_num_layer = int(model_args.get("matched_num_layer", self.num_layer))
            hidden_kwargs = dict(
                input_len=self.input_len,
                patch_len=self.patch_len,
                stride=self.stride,
                td_size=self.td_size,
                dw_size=self.dw_size,
                td_codebook=self.td_codebook,
                dw_codebook=self.dw_codebook,
                spa_codebook=self.spa_codebook,
                if_time_in_day=self.if_time_in_day,
                if_day_in_week=self.if_day_in_week,
                if_spatial=self.if_spatial,
                d_d=self.d_d,
                d_td=self.d_td,
                d_dw=self.d_dw,
                d_spa=self.d_spa,
                num_layer=matched_num_layer,
                hidden_dim=self.matched_hidden_dim,
                use_patch_branch=self.use_patch_branch,
                use_downsample_branch=self.use_downsample_branch,
                use_linear_residual_branch=False,
                patch_data_input_mode=self.patch_data_input_mode,
                patch_embedding_mode=self.patch_embedding_mode,
                patch_feature_dim=self.patch_feature_dim,
            )
            for step_idx, internal_len in enumerate(self.matched_stage_lengths[:-1]):
                self.hidden_steps.append(
                    KASAHiddenStep(
                        internal_len=internal_len,
                        latent_cond_dim=self.matched_hidden_dim if step_idx > 0 else 0,
                        **hidden_kwargs,
                    )
                )
            final_step_kwargs = dict(step_kwargs, num_layer=matched_num_layer, use_prev_condition=False)
            self.final_temporal_step = KASATemporalStep(
                output_len=self.output_len,
                latent_cond_dim=self.matched_hidden_dim if self.hidden_steps else 0,
                **final_step_kwargs,
            )
        else:
            for step_idx, k in enumerate(self.chain_lengths):
                latent_cond_dim = (
                    self.latent_prop_dim
                    if self.propagation_mode == "latent" and step_idx > 0
                    else 0
                )
                self.temporal_steps.append(
                    KASATemporalStep(
                        output_len=k,
                        latent_cond_dim=latent_cond_dim,
                        **step_kwargs,
                    )
                )

            num_transitions = max(len(self.chain_lengths) - 1, 0)
            self.latent_encoders = nn.ModuleList()
            if self.propagation_mode == "latent" and num_transitions > 0:
                for _ in range(num_transitions):
                    self.latent_encoders.append(
                        nn.Sequential(
                            nn.Linear(1, self.latent_prop_dim),
                            nn.GELU(),
                            nn.Linear(self.latent_prop_dim, self.latent_prop_dim),
                        )
                    )

        self.spatial_module = ABCDSpatialModule(
            node_size=self.node_size,
            input_len=self.input_len,
            d_spa=self.d_spa,
            if_spatial=self.if_spatial,
            **self._abcd_spatial_kwargs(model_args),
        )

        self.progressive_spatial_modules = nn.ModuleList()
        self.post_chain_spatial_modules = nn.ModuleList()
        self.graph_resolution_stack = None
        self.capdist_refine_stack = None

        spatial_stage_count = (
            len(self.matched_stage_lengths)
            if self.architecture_mode == "direct_matched"
            else len(self.chain_lengths)
        )
        if self.spatial_placement == "interleaved_progressive":
            self.progressive_spatial_modules = self._build_progressive_spatial_modules(
                model_args,
                spatial_stage_count,
            )
        elif self.spatial_placement == "temporal_first_multiscale":
            post_stage_count = len(
                model_args.get("post_chain_spatial_ratios", [0.25, 0.5, 1.0])
            )
            self.post_chain_spatial_modules = self._build_progressive_spatial_modules(
                model_args,
                post_stage_count,
                ratio_key="post_chain_spatial_ratios",
                alpha_key="post_chain_spatial_alphas",
                topk_key="post_chain_spatial_topks",
            )
        elif self.spatial_placement == "temporal_first_graph_resolution":
            from basicts.archs.arch_zoo.ChainForecasting_arch.graph_resolution_spatial import (
                GraphResolutionSpatialStack,
            )

            self.graph_resolution_stack = GraphResolutionSpatialStack(
                node_size=self.node_size,
                input_len=self.input_len,
                d_spa=self.d_spa,
                if_spatial=self.if_spatial,
                spatial_scheme=self.spatial_scheme,
                adj_mx_path=model_args.get("adj_mx_path"),
                post_spatial_mode=self.post_spatial_mode,
                graph_resolution_ratios=self.graph_resolution_ratios,
                graph_resolution_capacities=model_args.get("graph_resolution_capacities"),
                graph_resolution_skip_final_identity=model_args.get(
                    "graph_resolution_skip_final_identity", False
                ),
                graph_resolution_alphas=model_args.get(
                    "graph_resolution_alphas", [0.03, 0.06, 0.10]
                ),
                graph_resolution_topks=model_args.get(
                    "graph_resolution_topks", [8, 16, 32]
                ),
                graph_resolution_betas=model_args.get(
                    "graph_resolution_betas", [1.0, 1.0, 1.0]
                ),
                graph_resolution_rhos=model_args.get(
                    "graph_resolution_rhos", self.graph_resolution_ratios or [0.50]
                ),
                adp_hidden_dim=model_args.get("adp_hidden_dim", 32),
                adp_tau=model_args.get("adp_tau", 0.5),
                clustering_seed=self.clustering_seed,
                dataset_name=self.dataset_name,
                cluster_cache_dir=model_args.get("graph_cluster_cache_dir"),
                graph_cluster_method=model_args.get("graph_cluster_method", "current"),
                graph_cluster_affinity=model_args.get("graph_cluster_affinity"),
                cluster_train_series_path=model_args.get("cluster_train_series_path"),
                cluster_spatial_coord_path=model_args.get("cluster_spatial_coord_path"),
                cluster_road_distance_path=model_args.get("cluster_road_distance_path"),
                cluster_sigma_d=model_args.get("cluster_sigma_d", 0.5),
                cluster_road_delta=model_args.get("cluster_road_delta"),
                cluster_delta_4=model_args.get("cluster_delta_4", 0.8),
                cluster_delta_2=model_args.get("cluster_delta_2", 0.5),
                cluster_max_lag=model_args.get("cluster_max_lag", 12),
                cluster_lambda_s=model_args.get("cluster_lambda_s", 0.2),
                cluster_acf_lag=model_args.get("cluster_acf_lag", 24),
                cluster_graph_mix_lambda=model_args.get("cluster_graph_mix_lambda", 0.5),
                cluster_graph_mix_lambdas=model_args.get("cluster_graph_mix_lambdas"),
                capdist_sigma_d=model_args.get("capdist_sigma_d"),
                capdist_lambda_d=model_args.get("capdist_lambda_d"),
                capdist_use_road_distance=model_args.get("capdist_use_road_distance", True),
                capdist_use_hard_cutoff=model_args.get("capdist_use_hard_cutoff", False),
                data_dir=model_args.get("data_dir"),
                variant_name=model_args.get("variant_name", ""),
                adaptive_ms_topks=model_args.get("adaptive_ms_topks", [8, 16, 32]),
                adaptive_ms_alpha=model_args.get("adaptive_ms_alpha", 0.10),
                adaptive_ms_fusion=model_args.get("adaptive_ms_fusion", "softmax"),
                adaptive_ms_share_logits=model_args.get("adaptive_ms_share_logits", True),
                adaptive_ms_init=model_args.get("adaptive_ms_init", "favor_largest"),
            )
        elif self.spatial_placement == "temporal_first_capdist_refine":
            from basicts.archs.arch_zoo.ChainForecasting_arch.capdist_refine_spatial import (
                CapDistRefineSpatialStack,
            )

            self.capdist_refine_stack = CapDistRefineSpatialStack(
                node_size=self.node_size,
                input_len=self.input_len,
                output_len=self.output_len,
                chain_lengths=self.chain_lengths,
                d_spa=self.d_spa,
                if_spatial=self.if_spatial,
                spatial_scheme=self.spatial_scheme,
                adj_mx_path=model_args.get("adj_mx_path"),
                post_spatial_mode=self.post_spatial_mode,
                capdist_cluster_method=model_args.get(
                    "capdist_cluster_method", "capdist_spectral_pair"
                ),
                capdist_use_road_distance=model_args.get("capdist_use_road_distance", True),
                capdist_sigma_d=model_args.get("capdist_sigma_d", 0.5),
                capdist_lambda_d=model_args.get("capdist_lambda_d", 0.1),
                capdist_lambda_mix=model_args.get("capdist_lambda_mix"),
                capdist_alphas=model_args.get("capdist_alphas"),
                capdist_topks=model_args.get("capdist_topks"),
                clustering_seed=self.clustering_seed,
                dataset_name=self.dataset_name,
                cluster_cache_dir=model_args.get("graph_cluster_cache_dir"),
                cluster_road_distance_path=model_args.get("cluster_road_distance_path"),
                unified_aux_loss_mode=model_args.get("unified_aux_loss_mode", "none"),
                adp_hidden_dim=model_args.get("adp_hidden_dim", 32),
                adp_tau=model_args.get("adp_tau", 0.5),
            )

    @staticmethod
    def _abcd_spatial_kwargs(model_args: dict, **overrides) -> dict:
        kwargs = {
            "spatial_scheme": model_args.get("spatial_scheme", "C"),
            "adj_mx_path": model_args.get("adj_mx_path"),
            "use_gcn": model_args.get("use_gcn", False),
            "gcn_hidden_dim": model_args.get("gcn_hidden_dim", 64),
            "use_dynamic_spatial": model_args.get("use_dynamic_spatial", False),
            "dyn_hidden_dim": model_args.get("dyn_hidden_dim", 64),
            "dyn_topk": model_args.get("dyn_topk", 20),
            "dyn_tau": model_args.get("dyn_tau", 0.5),
            "dyn_alpha": model_args.get("dyn_alpha", 0.15),
            "dyn_static_weight": model_args.get("dyn_static_weight", 0.2),
            "use_adaptive_adj": model_args.get("use_adaptive_adj", True),
            "adp_hidden_dim": model_args.get("adp_hidden_dim", 32),
            "adp_topk": model_args.get("adp_topk", 20),
            "adp_tau": model_args.get("adp_tau", 0.5),
            "adp_alpha": model_args.get("adp_alpha", 0.1),
            "use_hybrid_graph": model_args.get("use_hybrid_graph", False),
            "hybrid_alpha": model_args.get("hybrid_alpha", 0.2),
            "use_lightweight_spatial": model_args.get("use_lightweight_spatial", False),
            "light_alpha": model_args.get("light_alpha", 0.05),
            "post_spatial_mode": model_args.get("post_spatial_mode", "adaptive_only"),
            "adaptive_ms_topks": model_args.get("adaptive_ms_topks", [8, 16, 32]),
            "adaptive_ms_alpha": model_args.get("adaptive_ms_alpha", 0.10),
            "adaptive_ms_fusion": model_args.get("adaptive_ms_fusion", "softmax"),
            "adaptive_ms_share_logits": model_args.get("adaptive_ms_share_logits", True),
            "adaptive_ms_init": model_args.get("adaptive_ms_init", "favor_largest"),
        }
        kwargs.update(overrides)
        return kwargs

    def _build_progressive_spatial_modules(
        self,
        model_args: dict,
        num_stages: int,
        ratio_key: str = "progressive_spatial_ratios",
        alpha_key: str = "progressive_spatial_alphas",
        topk_key: str = "progressive_spatial_topks",
    ) -> nn.ModuleList:
        ratios = self._fit_stage_list(model_args.get(ratio_key, [0.25, 0.5, 1.0]), num_stages)
        alphas = self._fit_stage_list(model_args.get(alpha_key, [0.03, 0.06, 0.10]), num_stages)
        topks = self._fit_stage_list(model_args.get(topk_key, [8, 16, 32]), num_stages)
        modules = nn.ModuleList()
        for ratio, alpha, topk in zip(ratios, alphas, topks):
            modules.append(
                ABCDSpatialModule(
                    node_size=self.node_size,
                    input_len=self.input_len,
                    d_spa=self.d_spa,
                    if_spatial=self.if_spatial,
                    **self._abcd_spatial_kwargs(
                        model_args,
                        dyn_hidden_dim=max(8, int(model_args.get("dyn_hidden_dim", 64) * ratio)),
                        dyn_topk=topk,
                        dyn_alpha=alpha,
                        adp_hidden_dim=max(8, int(model_args.get("adp_hidden_dim", 32) * ratio)),
                        adp_topk=topk,
                        adp_alpha=alpha,
                        hybrid_alpha=alpha,
                        light_alpha=alpha,
                    ),
                )
            )
        return modules

    @staticmethod
    def _fit_stage_list(values: list, num_stages: int) -> list:
        values = list(values)
        if num_stages <= 0:
            return []
        if not values:
            raise ValueError("Progressive spatial config lists must be non-empty.")
        if len(values) >= num_stages:
            return values[:num_stages]
        return values + [values[-1]] * (num_stages - len(values))

    @staticmethod
    def _resolve_spatial_placement(model_args: dict) -> str:
        if "spatial_placement" in model_args:
            placement = str(model_args["spatial_placement"]).lower()
        elif model_args.get("use_final_spatial_refine", True):
            placement = "final"
        else:
            placement = "none"
        if placement not in {
            "final",
            "each_level",
            "none",
            "interleaved_progressive",
            "temporal_first_multiscale",
            "temporal_first_graph_resolution",
            "temporal_first_capdist_refine",
        }:
            raise ValueError(
                f"Unsupported spatial_placement: {placement}. "
                "Expected 'final', 'each_level', 'none', 'interleaved_progressive', "
                "'temporal_first_multiscale', 'temporal_first_graph_resolution', "
                "or 'temporal_first_capdist_refine'."
            )
        return placement

    @staticmethod
    def _resolve_spatial_organization_type(spatial_placement: str) -> str:
        mapping = {
            "none": "none",
            "interleaved_progressive": "interleaved",
            "final": "final_only",
            "each_level": "each_level",
            "temporal_first_multiscale": "temporal_first_multiscale",
            "temporal_first_graph_resolution": "graph_resolution",
            "temporal_first_capdist_refine": "capdist_refine",
        }
        return mapping.get(spatial_placement, spatial_placement)

    def _uses_interleaved_spatial(self) -> bool:
        return self.spatial_placement in {"interleaved_progressive", "each_level"}

    def _propagate_temporal_only(self) -> bool:
        return self.spatial_placement in {
            "none",
            "final",
            "temporal_first_multiscale",
            "temporal_first_graph_resolution",
            "temporal_first_capdist_refine",
        }

    @staticmethod
    def pool_target(future_target: torch.Tensor, target_len: int) -> torch.Tensor:
        """Average-pool full future target [B, F, N, 1] to [B, target_len, N, 1]."""
        target = future_target[..., :1]
        batch_size, future_len, num_nodes, _ = target.shape
        if future_len % target_len == 0:
            group = future_len // target_len
            return target.reshape(batch_size, target_len, group, num_nodes, 1).mean(dim=2)
        x = target.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, 1, future_len)
        x = F.adaptive_avg_pool1d(x, target_len)
        return x.reshape(batch_size, num_nodes, 1, target_len).permute(0, 3, 1, 2)

    def _spatial_codebook(self):
        if self.use_pre_temporal_spatial_enhancement:
            return self.spatial_module.get_enhanced_spatial_embedding(self.spa_codebook)
        return self.spa_codebook

    def _apply_spatial_refine(self, forecast: torch.Tensor, history_data: torch.Tensor) -> torch.Tensor:
        history_flow = history_data[..., 0]
        return self.spatial_module.refine_prediction(forecast, history_flow)

    def _apply_progressive_spatial_refine(
        self,
        forecast: torch.Tensor,
        history_data: torch.Tensor,
        stage_idx: int,
    ) -> torch.Tensor:
        history_flow = history_data[..., 0]
        module = self.progressive_spatial_modules[stage_idx]
        return module.refine_prediction(forecast, history_flow)

    def _apply_post_chain_spatial_stages(
        self,
        forecast: torch.Tensor,
        history_data: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        history_flow = history_data[..., 0]
        stage_outputs: list[torch.Tensor] = []
        current = forecast
        for module in self.post_chain_spatial_modules:
            current = module.refine_prediction(current, history_flow)
            stage_outputs.append(current)
        return current, stage_outputs

    def _temporal_step_trainable(self, step_idx: int, stage: str) -> bool:
        stage = str(stage).upper()
        if stage == "FT":
            return True
        if stage == "T1":
            return step_idx == 0
        if stage == "T2":
            return step_idx == 1
        if stage == "T3":
            return step_idx == 2
        return False

    def _forward_temporal_stagewise(
        self,
        history_data: torch.Tensor,
        stage: str,
        detach_previous: bool = True,
    ) -> list[torch.Tensor]:
        spatial_codebook = self._spatial_codebook()
        stage = str(stage).upper()
        if stage in {"T1", "T2", "T3"}:
            max_step = {"T1": 0, "T2": 1, "T3": 2}[stage]
        else:
            max_step = len(self.temporal_steps) - 1

        temporal_preds: list[torch.Tensor] = []
        prev_forecast = None
        for step_idx in range(max_step + 1):
            target_len = self.chain_lengths[step_idx]
            prev_up = None
            if (
                self.propagation_mode == "forecast_state"
                and prev_forecast is not None
                and self.use_prev_condition
            ):
                prev_up = interpolate_forecast(prev_forecast, target_len)
                if detach_previous and not self._temporal_step_trainable(step_idx, stage):
                    prev_up = prev_up.detach()

            trainable = self._temporal_step_trainable(step_idx, stage) or stage == "FT"
            step = self.temporal_steps[step_idx]
            ctx = torch.enable_grad() if trainable else torch.no_grad()
            with ctx:
                t_k = step(
                    history_data,
                    prev_forecast=prev_up,
                    spatial_codebook=spatial_codebook,
                )
            if detach_previous and step_idx < max_step and stage in {"T1", "T2", "T3"}:
                if not self._temporal_step_trainable(step_idx + 1, stage):
                    t_k = t_k.detach()
            temporal_preds.append(t_k)
            prev_forecast = t_k
        return temporal_preds

    def _forward_chain_stagewise(
        self,
        history_data: torch.Tensor,
        stage: str,
        detach_previous: bool = True,
        stagewise_sequence: str = "full",
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict]:
        self._last_graph_diagnostics = None
        stage = str(stage).upper()
        temporal_preds = self._forward_temporal_stagewise(history_data, stage, detach_previous)
        y_temporal_final = temporal_preds[-1]
        chain_preds = list(temporal_preds)
        spatial_preds = list(temporal_preds)
        spatial_stage_preds: list[torch.Tensor] = []
        y_final = y_temporal_final
        extras: dict = {
            "pred_T_low": temporal_preds[0] if temporal_preds else None,
            "pred_T_mid": temporal_preds[1] if len(temporal_preds) > 1 else None,
            "pred_T_full": temporal_preds[-1],
        }

        if stage in {"S14", "S12", "S1", "FT"} and self.spatial_placement == "temporal_first_graph_resolution":
            history_flow = history_data[..., 0]
            seq = str(stagewise_sequence).lower()
            if seq == "final_spatial_only" and stage == "S1":
                active_spatial = 0
                max_spatial = 0
            else:
                max_spatial = {"S14": 0, "S12": 1, "S1": 2, "FT": 2}[stage]
                active_spatial = 2 if stage == "FT" else max_spatial
            skip_stage_indices: list[int] = []
            if seq == "no_s14" and stage == "S12":
                skip_stage_indices = [0]
            graph_out = self.graph_resolution_stack.forward_stagewise(
                y_temporal_final,
                history_flow,
                active_stage_idx=active_spatial,
                max_stage_idx=max_spatial,
                detach_previous=detach_previous,
                train_all_spatial=stage == "FT",
                skip_stage_indices=skip_stage_indices,
            )
            y_final = graph_out["pred"]
            spatial_stage_preds = graph_out.get("node_stage_preds") or []
            self._last_graph_diagnostics = graph_out
            node_preds = graph_out.get("node_stage_preds_all") or []
            extras["Y_after_S14"] = node_preds[1] if len(node_preds) > 1 else None
            extras["Y_after_S12"] = node_preds[2] if len(node_preds) > 2 else None
            extras["Y_after_S1"] = graph_out["pred"]

        if self.chain_supervision_source == "temporal_chain" and chain_preds:
            chain_preds[-1] = y_temporal_final

        return y_final, chain_preds, temporal_preds, spatial_preds, spatial_stage_preds, extras

    def _forward_g1_stagewise(
        self,
        history_data: torch.Tensor,
        stage: str,
        detach_previous: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict]:
        """G1_final_adaptive stagewise path: T1/T2/T3 temporal + optional final adaptive spatial (S1)."""
        self._last_graph_diagnostics = None
        stage = str(stage).upper()
        if self.spatial_placement != "final":
            raise RuntimeError(
                f"G1_stagewise requires spatial_placement='final', got {self.spatial_placement!r}"
            )
        if self.spatial_module is None:
            raise RuntimeError("G1_stagewise requires spatial_module (final adaptive spatial)")

        temporal_preds = self._forward_temporal_stagewise(history_data, stage, detach_previous)
        y_temporal_final = temporal_preds[-1]
        chain_preds = list(temporal_preds)
        spatial_preds = list(temporal_preds)
        spatial_stage_preds: list[torch.Tensor] = []
        extras: dict = {
            "pred_T_low": temporal_preds[0] if temporal_preds else None,
            "pred_T_mid": temporal_preds[1] if len(temporal_preds) > 1 else None,
            "pred_T_full": y_temporal_final,
            "Y_T": y_temporal_final,
        }
        y_final = y_temporal_final

        if stage == "S1":
            y_T = y_temporal_final.detach() if detach_previous else y_temporal_final
            y_hat = self._apply_spatial_refine(y_T, history_data)
            y_final = y_hat
            extras["pred_final"] = y_hat
            spatial_stage_preds = [y_hat]

        return y_final, chain_preds, temporal_preds, spatial_preds, spatial_stage_preds, extras

    def _forward_chain(
        self, history_data: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        self._last_graph_diagnostics = None
        spatial_codebook = self._spatial_codebook()
        chain_preds: list[torch.Tensor] = []
        temporal_preds: list[torch.Tensor] = []
        spatial_preds: list[torch.Tensor] = []
        spatial_stage_preds: list[torch.Tensor] = []
        prev_forecast = None
        prev_latent = None

        for step_idx, step in enumerate(self.temporal_steps):
            target_len = self.chain_lengths[step_idx]
            prev_up = None
            if (
                self.propagation_mode == "forecast_state"
                and prev_forecast is not None
                and self.use_prev_condition
            ):
                prev_up = interpolate_forecast(prev_forecast, target_len)

            if self.propagation_mode == "latent" and prev_latent is not None:
                t_k = step(
                    history_data,
                    prev_latent=prev_latent,
                    spatial_codebook=spatial_codebook,
                )
            else:
                t_k = step(
                    history_data,
                    prev_forecast=prev_up,
                    spatial_codebook=spatial_codebook,
                )

            if self.spatial_placement == "interleaved_progressive":
                z_k = self._apply_progressive_spatial_refine(t_k, history_data, step_idx)
            elif self.spatial_placement == "each_level":
                z_k = self._apply_spatial_refine(t_k, history_data)
            else:
                z_k = t_k

            temporal_preds.append(t_k)
            spatial_preds.append(z_k)

            if self.chain_supervision_source == "temporal_chain":
                chain_preds.append(t_k)
            else:
                chain_preds.append(z_k)

            if self.propagation_mode == "latent":
                if step_idx < len(self.latent_encoders):
                    h_k = self.latent_encoders[step_idx](t_k)
                    prev_latent = interpolate_latent(h_k, self.input_len)
                prev_forecast = None
            elif self._propagate_temporal_only():
                prev_forecast = t_k
                prev_latent = None
            else:
                prev_forecast = z_k
                prev_latent = None

        y_temporal_final = temporal_preds[-1]
        y_final = chain_preds[-1]

        if self.spatial_placement == "final":
            y_final = self._apply_spatial_refine(y_temporal_final, history_data)
            spatial_stage_preds = []
        elif self.spatial_placement == "temporal_first_multiscale":
            y_final, spatial_stage_preds = self._apply_post_chain_spatial_stages(
                y_temporal_final, history_data
            )
        elif self.spatial_placement == "temporal_first_graph_resolution":
            history_flow = history_data[..., 0]
            graph_out = self.graph_resolution_stack(
                y_temporal_final, history_flow, return_diagnostics=True
            )
            y_final = graph_out["pred"]
            spatial_stage_preds = graph_out["node_stage_preds"]
            self._last_graph_diagnostics = graph_out
        elif self.spatial_placement == "temporal_first_capdist_refine":
            history_flow = history_data[..., 0]
            capdist_out = self.capdist_refine_stack(
                y_temporal_final, history_flow, return_diagnostics=True
            )
            y_final = capdist_out["pred"]
            spatial_stage_preds = capdist_out["node_stage_preds"]
            self._last_graph_diagnostics = capdist_out

        if self.chain_supervision_source == "temporal_chain":
            chain_preds[-1] = y_temporal_final

        return y_final, chain_preds, temporal_preds, spatial_preds, spatial_stage_preds

    def _forward_direct_matched(
        self, history_data: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        spatial_codebook = self._spatial_codebook()
        prev_hidden = None
        for hidden_step in self.hidden_steps:
            prev_hidden = hidden_step(
                history_data,
                prev_latent=prev_hidden,
                spatial_codebook=spatial_codebook,
            )

        y = self.final_temporal_step(
            history_data,
            prev_latent=prev_hidden,
            spatial_codebook=spatial_codebook,
        )
        spatial_stage_preds: list[torch.Tensor] = [y]
        if self.spatial_placement == "interleaved_progressive":
            history_flow = history_data[..., 0]
            for module in self.progressive_spatial_modules:
                y = module.refine_prediction(y, history_flow)
                spatial_stage_preds.append(y)
        elif self.spatial_placement == "final":
            y = self._apply_spatial_refine(y, history_data)
            spatial_stage_preds.append(y)
        elif self.spatial_placement == "temporal_first_multiscale":
            y, spatial_stage_preds = self._apply_post_chain_spatial_stages(y, history_data)
        elif self.spatial_placement == "temporal_first_capdist_refine":
            history_flow = history_data[..., 0]
            capdist_out = self.capdist_refine_stack(y, history_flow, return_diagnostics=True)
            y = capdist_out["pred"]
            spatial_stage_preds = capdist_out["node_stage_preds"]
            self._last_graph_diagnostics = capdist_out

        return y, [y], [y], [y], spatial_stage_preds

    def _collect_adaptive_ms_diagnostics(self) -> dict:
        if self.post_spatial_mode != "adaptive_multiscale_only":
            return {}
        if self.spatial_placement == "final" and self.spatial_module is not None:
            return self.spatial_module.get_adaptive_ms_diagnostics()
        if (
            self.spatial_placement == "temporal_first_graph_resolution"
            and self.graph_resolution_stack is not None
            and self.graph_resolution_stack.spatial_modules
        ):
            return self.graph_resolution_stack.spatial_modules[-1].get_adaptive_ms_diagnostics()
        return {}

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = False,
        return_all: bool = False,
        return_intermediates: bool = False,
        stagewise_stage=None,
        detach_previous: bool = True,
        stagewise_sequence: str = "full",
        stagewise_backend: str = "gr7",
        **kwargs,
    ):
        stagewise_extras: dict = {}
        if stagewise_stage:
            if str(stagewise_backend).lower() == "g1":
                (
                    y_final,
                    chain_preds,
                    temporal_preds,
                    spatial_preds,
                    spatial_stage_preds,
                    stagewise_extras,
                ) = self._forward_g1_stagewise(
                    history_data,
                    stage=stagewise_stage,
                    detach_previous=detach_previous,
                )
            else:
                (
                    y_final,
                    chain_preds,
                    temporal_preds,
                    spatial_preds,
                    spatial_stage_preds,
                    stagewise_extras,
                ) = self._forward_chain_stagewise(
                    history_data,
                    stage=stagewise_stage,
                    detach_previous=detach_previous,
                    stagewise_sequence=stagewise_sequence,
                )
        else:
            y_final, chain_preds, temporal_preds, spatial_preds, spatial_stage_preds = (
                self._forward_direct_matched(history_data)
                if self.architecture_mode == "direct_matched"
                else self._forward_chain(history_data)
            )

        if return_all or return_intermediates:
            result = {
                "pred": y_final,
                "chain_preds": chain_preds,
                "temporal_preds": temporal_preds,
                "temporal_stage_preds": temporal_preds,
                "spatial_preds": spatial_preds,
                "spatial_stage_preds": spatial_stage_preds,
                "final_temporal_pred": temporal_preds[-1] if temporal_preds else y_final,
                "chain_lengths": list(self.chain_lengths),
                "spatial_organization_type": self.spatial_organization_type,
            }
            if temporal_preds:
                result["pred_T_low"] = temporal_preds[0]
                if len(temporal_preds) > 1:
                    result["pred_T_mid"] = temporal_preds[1]
                result["pred_T_full"] = temporal_preds[-1]
            result.update(stagewise_extras)
            if stagewise_extras.get("pred_final") is not None:
                result["pred_final"] = stagewise_extras["pred_final"]
            if self._last_graph_diagnostics is not None:
                result["graph_resolution_diagnostics"] = self._last_graph_diagnostics
                diag = self._last_graph_diagnostics
                result["graph_node_stage_preds"] = diag.get("node_stage_preds")
                result["graph_cluster_residuals"] = diag.get("cluster_residuals")
                result["graph_lifted_residuals"] = diag.get("lifted_residuals")
                result["graph_projection_matrices"] = diag.get("graph_projection_matrices")
                result["graph_ratios"] = diag.get("graph_ratios")
                if self.graph_resolution_stack is not None:
                    result["graph_resolution_metadata"] = self.graph_resolution_stack.metadata()
                if self.capdist_refine_stack is not None:
                    result["graph_resolution_metadata"] = self.capdist_refine_stack.metadata()
            result.update(self._collect_adaptive_ms_diagnostics())
            return result
        return y_final
