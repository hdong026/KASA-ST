import torch
import torch.nn.functional as F
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.gcn import ABCDSpatialModule
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    KASATemporalStep,
    interpolate_forecast,
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
        self.spatial_placement = self._resolve_spatial_placement(model_args)
        self.post_spatial_mode = model_args.get("post_spatial_mode", "adaptive_only")
        self.use_pre_temporal_spatial_enhancement = model_args.get(
            "use_pre_temporal_spatial_enhancement", False
        )

        if self.chain_lengths[-1] != self.output_len:
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

        self.temporal_steps = nn.ModuleList(
            [KASATemporalStep(output_len=k, **step_kwargs) for k in self.chain_lengths]
        )

        self.spatial_module = ABCDSpatialModule(
            node_size=self.node_size,
            input_len=self.input_len,
            d_spa=self.d_spa,
            if_spatial=self.if_spatial,
            spatial_scheme=self.spatial_scheme,
            adj_mx_path=model_args.get("adj_mx_path"),
            use_gcn=model_args.get("use_gcn", False),
            gcn_hidden_dim=model_args.get("gcn_hidden_dim", 64),
            use_dynamic_spatial=model_args.get("use_dynamic_spatial", False),
            dyn_hidden_dim=model_args.get("dyn_hidden_dim", 64),
            dyn_topk=model_args.get("dyn_topk", 20),
            dyn_tau=model_args.get("dyn_tau", 0.5),
            dyn_alpha=model_args.get("dyn_alpha", 0.15),
            dyn_static_weight=model_args.get("dyn_static_weight", 0.2),
            use_adaptive_adj=model_args.get("use_adaptive_adj", True),
            adp_hidden_dim=model_args.get("adp_hidden_dim", 32),
            adp_topk=model_args.get("adp_topk", 20),
            adp_tau=model_args.get("adp_tau", 0.5),
            adp_alpha=model_args.get("adp_alpha", 0.1),
            use_hybrid_graph=model_args.get("use_hybrid_graph", False),
            hybrid_alpha=model_args.get("hybrid_alpha", 0.2),
            use_lightweight_spatial=model_args.get("use_lightweight_spatial", False),
            light_alpha=model_args.get("light_alpha", 0.05),
            post_spatial_mode=self.post_spatial_mode,
        )

        self.progressive_spatial_modules = nn.ModuleList()
        if self.spatial_placement == "interleaved_progressive":
            ratios = self._fit_stage_list(
                model_args.get("progressive_spatial_ratios", [0.25, 0.5, 1.0]),
                len(self.chain_lengths),
            )
            alphas = self._fit_stage_list(
                model_args.get("progressive_spatial_alphas", [0.03, 0.06, 0.10]),
                len(self.chain_lengths),
            )
            topks = self._fit_stage_list(
                model_args.get("progressive_spatial_topks", [8, 16, 32]),
                len(self.chain_lengths),
            )
            for ratio, alpha, topk in zip(ratios, alphas, topks):
                self.progressive_spatial_modules.append(
                    ABCDSpatialModule(
                        node_size=self.node_size,
                        input_len=self.input_len,
                        d_spa=self.d_spa,
                        if_spatial=self.if_spatial,
                        spatial_scheme=self.spatial_scheme,
                        adj_mx_path=model_args.get("adj_mx_path"),
                        use_gcn=model_args.get("use_gcn", False),
                        gcn_hidden_dim=model_args.get("gcn_hidden_dim", 64),
                        use_dynamic_spatial=model_args.get("use_dynamic_spatial", False),
                        dyn_hidden_dim=max(8, int(model_args.get("dyn_hidden_dim", 64) * ratio)),
                        dyn_topk=topk,
                        dyn_tau=model_args.get("dyn_tau", 0.5),
                        dyn_alpha=alpha,
                        dyn_static_weight=model_args.get("dyn_static_weight", 0.2),
                        use_adaptive_adj=model_args.get("use_adaptive_adj", True),
                        adp_hidden_dim=max(8, int(model_args.get("adp_hidden_dim", 32) * ratio)),
                        adp_topk=topk,
                        adp_tau=model_args.get("adp_tau", 0.5),
                        adp_alpha=alpha,
                        use_hybrid_graph=model_args.get("use_hybrid_graph", False),
                        hybrid_alpha=alpha,
                        use_lightweight_spatial=model_args.get("use_lightweight_spatial", False),
                        light_alpha=alpha,
                        post_spatial_mode=self.post_spatial_mode,
                    )
                )

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
        if placement not in {"final", "each_level", "none", "interleaved_progressive"}:
            raise ValueError(
                f"Unsupported spatial_placement: {placement}. "
                "Expected 'final', 'each_level', 'none', or 'interleaved_progressive'."
            )
        return placement

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

    def _forward_chain(
        self, history_data: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        spatial_codebook = self._spatial_codebook()
        chain_preds: list[torch.Tensor] = []
        temporal_preds: list[torch.Tensor] = []
        spatial_preds: list[torch.Tensor] = []
        prev_forecast = None

        for step_idx, step in enumerate(self.temporal_steps):
            target_len = self.chain_lengths[step_idx]
            if prev_forecast is not None and self.use_prev_condition:
                prev_up = interpolate_forecast(prev_forecast, target_len)
            else:
                prev_up = None

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
            chain_preds.append(z_k)
            prev_forecast = z_k

        y_final = chain_preds[-1]
        if self.spatial_placement == "final":
            y_final = self._apply_spatial_refine(y_final, history_data)

        return y_final, chain_preds, temporal_preds, spatial_preds

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = False,
        return_all: bool = False,
        **kwargs,
    ):
        y_final, chain_preds, temporal_preds, spatial_preds = self._forward_chain(history_data)

        if return_all:
            return {
                "pred": y_final,
                "chain_preds": chain_preds,
                "temporal_preds": temporal_preds,
                "spatial_preds": spatial_preds,
            }
        return y_final
