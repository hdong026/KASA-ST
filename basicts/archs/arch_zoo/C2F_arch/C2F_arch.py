from math import ceil
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from basicts.archs.arch_zoo.C2F_arch.patch_emb import PatchEncoder
from basicts.archs.arch_zoo.C2F_arch.downsamp_emb import DownsampEncoder
from basicts.archs.arch_zoo.C2F_arch.gcn import ABCDSpatialModule
from basicts.archs.arch_zoo.C2F_arch.graph_spectral import GraphSpectralCalibration


class SimpleKANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5):
        super(SimpleKANLinear, self).__init__()
        self.grid_size = grid_size
        self.base_linear = nn.Linear(in_features, out_features)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        self.grid = nn.Parameter(torch.linspace(-1, 1, grid_size), requires_grad=False)
        nn.init.kaiming_uniform_(self.base_linear.weight)
        nn.init.uniform_(self.spline_weight, -0.1, 0.1)

    def forward(self, x):
        base = self.base_linear(F.silu(x))
        x_uns = x.unsqueeze(-1)
        basis = torch.exp(-((x_uns - self.grid) / (2 / (self.grid_size - 1))) ** 2)
        spline = torch.einsum("...ig,oig->...o", basis, self.spline_weight)
        return base + spline


class C2F(nn.Module):
    def __init__(self, **model_args):
        super(C2F, self).__init__()
        self.node_size = model_args["node_size"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.output_len = model_args["output_len"]
        self.patch_len = model_args["patch_len"]
        self.stride = model_args["stride"]
        self.td_size = model_args["td_size"]
        self.dw_size = model_args["dw_size"]
        self.d_td = model_args["d_td"]
        self.d_dw = model_args["d_dw"]
        self.d_d = model_args["d_d"]
        self.d_spa = model_args["d_spa"]

        self.if_time_in_day = model_args["if_time_in_day"]
        self.if_day_in_week = model_args["if_day_in_week"]
        self.if_spatial = model_args["if_spatial"]
        self.num_layer = model_args["num_layer"]
        self.spatial_scheme = str(model_args.get("spatial_scheme", "legacy")).upper()
        self.use_pre_temporal_spatial_enhancement = model_args.get(
            "use_pre_temporal_spatial_enhancement", True
        )
        self.use_input_prior_enhancement = model_args.get(
            "use_input_prior_enhancement", False
        )
        self.input_prior_mapper_type = model_args.get(
            "input_prior_mapper_type", "mlp"
        )
        self.keep_output_prior_residual = model_args.get(
            "keep_output_prior_residual",
            model_args.get("use_prior_residual", True),
        )
        self.prior_source = model_args.get("prior_source", "history")
        self.post_spatial_mode = model_args.get("post_spatial_mode", "hybrid")
        self.use_patch_branch = model_args.get("use_patch_branch", True)
        self.use_downsample_branch = model_args.get("use_downsample_branch", True)
        self.use_linear_residual_branch = model_args.get("use_linear_residual_branch", True)
        self.use_extra_prior_input = model_args.get("use_extra_prior_input", False)
        self.main_input_dim = model_args.get("main_input_dim", 3)
        self.use_graph_spectral_calibration = model_args.get(
            "use_graph_spectral_calibration", False
        )
        self.patch_data_input_mode = model_args.get("patch_data_input_mode", "all")
        self.patch_embedding_mode = model_args.get("patch_embedding_mode", "serial_concat")
        self.patch_feature_dim = model_args.get("patch_feature_dim", None)

        self.c2f_mode = str(model_args.get("c2f_mode", "none")).lower()
        self.coarse_len = int(model_args.get("coarse_len", 3))
        self.coarse_loss_weight = float(model_args.get("coarse_loss_weight", 0.1))
        self.use_coarse_loss = model_args.get("use_coarse_loss", True)
        self.c2f_upsample_mode = model_args.get("c2f_upsample_mode", "linear")
        self.residual_scale_init = float(model_args.get("residual_scale_init", 1.0))
        self.patch_residual_condition = str(
            model_args.get("patch_residual_condition", "none")
        ).lower()
        self.use_direct_patch_in_c2f = model_args.get("use_direct_patch_in_c2f", True)
        self.use_linear_residual_in_c2f = model_args.get(
            "use_linear_residual_in_c2f", True
        )

        if self.c2f_mode not in {"none", "coarse_residual"}:
            raise ValueError(
                f"Unsupported c2f_mode: {self.c2f_mode}. "
                "Expected 'none' or 'coarse_residual'."
            )

        self.latest_coarse_pred = None
        self.latest_coarse_target = None

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
            use_adaptive_adj=model_args.get("use_adaptive_adj", False),
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

        encoder_input_dim = self.main_input_dim if self.use_extra_prior_input else 3

        self.patch_encoder = PatchEncoder(
            self.td_size,
            self.dw_size,
            self.td_codebook,
            self.dw_codebook,
            self.spa_codebook,
            self.if_time_in_day,
            self.if_day_in_week,
            self.if_spatial,
            encoder_input_dim,
            self.patch_len,
            self.stride,
            self.d_d,
            self.d_td,
            self.d_dw,
            self.d_spa,
            self.output_len,
            self.num_layer,
            patch_data_input_mode=self.patch_data_input_mode,
            patch_embedding_mode=self.patch_embedding_mode,
            patch_feature_dim=self.patch_feature_dim,
        )

        downsamp_coarse_len = self.coarse_len if self.c2f_mode == "coarse_residual" else None
        self.downsamp_encoder = DownsampEncoder(
            self.td_size,
            self.dw_size,
            self.td_codebook,
            self.dw_codebook,
            self.spa_codebook,
            self.if_time_in_day,
            self.if_day_in_week,
            self.if_spatial,
            encoder_input_dim,
            self.patch_len,
            self.stride,
            self.d_d,
            self.d_td,
            self.d_dw,
            self.d_spa,
            self.output_len,
            self.num_layer,
            coarse_len=downsamp_coarse_len,
        )

        self.graph_spectral_calibration = None
        if self.use_graph_spectral_calibration:
            self.graph_spectral_calibration = GraphSpectralCalibration(**model_args)

        self.residual = nn.Conv2d(
            in_channels=self.input_len,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )

        self.linear_residual_scale = nn.Parameter(
            torch.tensor(self.residual_scale_init, dtype=torch.float32)
        )

        self.coarse_condition_mlp = None
        self.coarse_condition_scale = None
        if self.c2f_mode == "coarse_residual" and self.patch_residual_condition == "add_coarse":
            self.coarse_condition_mlp = nn.Sequential(
                nn.Linear(1, 16),
                nn.SiLU(),
                nn.Linear(16, 1),
            )
            self.coarse_condition_scale = nn.Parameter(torch.tensor(0.0))

        if self.input_dim > 3:
            prior_mapper_type = model_args.get("prior_mapper_type", "mlp")
            if prior_mapper_type == "kan":
                self.prior_mapper = SimpleKANLinear(1, 1)
            elif prior_mapper_type == "mlp":
                self.prior_mapper = nn.Sequential(
                    nn.Linear(1, 16),
                    nn.SiLU(),
                    nn.Linear(16, 1),
                )
            elif prior_mapper_type == "linear":
                self.prior_mapper = nn.Linear(1, 1)
            else:
                raise ValueError(f"Unsupported prior_mapper_type: {prior_mapper_type}")
            self.prior_mapper_type = prior_mapper_type

        if self.use_input_prior_enhancement and self.input_dim > 3:
            if self.input_prior_mapper_type == "mlp":
                self.input_prior_mapper = nn.Sequential(
                    nn.Linear(1, 16),
                    nn.SiLU(),
                    nn.Linear(16, 1),
                )
                nn.init.zeros_(self.input_prior_mapper[-1].weight)
                nn.init.zeros_(self.input_prior_mapper[-1].bias)
            elif self.input_prior_mapper_type == "linear":
                self.input_prior_mapper = nn.Linear(1, 1)
                nn.init.zeros_(self.input_prior_mapper.weight)
                nn.init.zeros_(self.input_prior_mapper.bias)
            else:
                raise ValueError(
                    f"Unsupported input_prior_mapper_type: {self.input_prior_mapper_type}"
                )

        self.slots_per_day = int(model_args.get("slots_per_day", 288))
        template = self._load_weekly_template(model_args.get("data_path"))
        if template is not None:
            self.register_buffer("weekly_spectral_template", template, persistent=False)
        else:
            self.weekly_spectral_template = None

    def _load_weekly_template(self, data_path):
        if not data_path or not os.path.exists(data_path) or not str(data_path).endswith(".npz"):
            return None
        archive = np.load(data_path)
        if "weekly_spectral_template" not in archive:
            return None
        template = torch.from_numpy(np.array(archive["weekly_spectral_template"])).float()
        if template.ndim == 2:
            template = template.unsqueeze(-1)
        if "slots_per_day" in archive:
            self.slots_per_day = int(archive["slots_per_day"])
        return template

    def get_latest_coarse_pred(self):
        return self.latest_coarse_pred

    def get_latest_coarse_target(self):
        return self.latest_coarse_target

    @staticmethod
    def build_coarse_target(future_data: torch.Tensor, coarse_len: int) -> torch.Tensor:
        """Pool future target to coarse resolution. Returns [B, coarse_len, N, 1]."""
        target = future_data[..., :1]
        batch_size, future_len, num_nodes, _ = target.shape
        if future_len % coarse_len == 0:
            group = future_len // coarse_len
            coarse = target.reshape(batch_size, coarse_len, group, num_nodes, 1).mean(dim=2)
        else:
            x = target.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, 1, future_len)
            x = F.adaptive_avg_pool1d(x, coarse_len)
            coarse = x.reshape(batch_size, num_nodes, 1, coarse_len).permute(0, 3, 1, 2)
        return coarse

    def _upsample_future(self, coarse_pred: torch.Tensor, target_len: int) -> torch.Tensor:
        """Upsample coarse prediction [B, Fc, N, 1] to [B, target_len, N, 1]."""
        batch_size, coarse_steps, num_nodes, channels = coarse_pred.shape
        mode = self.c2f_upsample_mode
        if mode != "linear":
            raise ValueError(f"Unsupported c2f_upsample_mode: {mode}")
        x = coarse_pred.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, channels, coarse_steps)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        x = x.reshape(batch_size, num_nodes, channels, target_len).permute(0, 3, 1, 2)
        return x

    def _prepare_inputs(self, history_data: torch.Tensor):
        if self.use_pre_temporal_spatial_enhancement:
            spatial_codebook_for_encoder = self.spatial_module.get_enhanced_spatial_embedding(
                self.spa_codebook
            )
        else:
            spatial_codebook_for_encoder = self.spa_codebook

        if self.use_input_prior_enhancement and self.input_dim > 3:
            prior_data_for_input = history_data[..., 3:4]
            flow_delta = self.input_prior_mapper(prior_data_for_input)
            enhanced_flow = history_data[..., 0:1] + flow_delta
            main_input = torch.cat([enhanced_flow, history_data[..., 1:3]], dim=-1)
        elif self.use_extra_prior_input:
            main_input = history_data[..., : self.main_input_dim]
        else:
            main_input = history_data[..., :3]

        in_len_add = ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if in_len_add:
            main_input_aug = torch.cat(
                (main_input[:, -1:, :, :].expand(-1, in_len_add, -1, -1), main_input), dim=1
            )
        else:
            main_input_aug = main_input

        downsamp_input = [main_input_aug[:, i :: self.stride, :, :] for i in range(self.stride)]
        downsamp_input = torch.stack(downsamp_input, dim=1)

        patch_input = main_input_aug.unfold(dimension=1, size=self.patch_len, step=self.patch_len).permute(
            0, 1, 4, 2, 3
        )

        res_input = history_data[..., 0:1].permute(0, 1, 2, 3)
        return spatial_codebook_for_encoder, patch_input, downsamp_input, res_input

    def _forward_kasa_base(
        self,
        patch_input,
        downsamp_input,
        res_input,
        spatial_codebook_for_encoder,
        kwargs,
    ):
        patch_predict = self.patch_encoder(
            patch_input, spatial_codebook=spatial_codebook_for_encoder
        )
        downsamp_predict = self.downsamp_encoder(
            downsamp_input, spatial_codebook=spatial_codebook_for_encoder
        )
        res_out = self.residual(res_input)

        if kwargs.get("return_backbone", False):
            backbone_outputs = []
            if self.use_patch_branch:
                backbone_outputs.append(patch_predict)
            if self.use_downsample_branch:
                backbone_outputs.append(downsamp_predict)
            if not backbone_outputs:
                raise ValueError("At least one temporal branch must be enabled.")
            return sum(backbone_outputs)

        branch_outputs = []
        if self.use_patch_branch:
            branch_outputs.append(patch_predict)
        if self.use_downsample_branch:
            branch_outputs.append(downsamp_predict)
        if self.use_linear_residual_branch:
            branch_outputs.append(res_out)
        if not branch_outputs:
            raise ValueError("At least one temporal branch must be enabled.")
        return sum(branch_outputs)

    def _forward_coarse_residual(
        self,
        patch_input,
        downsamp_input,
        res_input,
        spatial_codebook_for_encoder,
        future_data,
    ):
        patch_predict = None
        if self.use_patch_branch and self.use_direct_patch_in_c2f:
            patch_predict = self.patch_encoder(
                patch_input, spatial_codebook=spatial_codebook_for_encoder
            )

        coarse_pred = None
        if self.use_downsample_branch:
            coarse_pred = self.downsamp_encoder.forward_coarse(
                downsamp_input, spatial_codebook=spatial_codebook_for_encoder
            )
        else:
            raise ValueError("coarse_residual mode requires use_downsample_branch=True.")

        self.latest_coarse_pred = coarse_pred
        if future_data is not None:
            self.latest_coarse_target = self.build_coarse_target(future_data, self.coarse_len)
        else:
            self.latest_coarse_target = None

        y_coarse = self._upsample_future(coarse_pred, self.output_len)

        if patch_predict is not None:
            residual = patch_predict
        else:
            residual = torch.zeros_like(y_coarse)

        if self.use_linear_residual_in_c2f and self.use_linear_residual_branch:
            res_out = self.residual(res_input)
            residual = residual + self.linear_residual_scale * res_out

        if (
            self.patch_residual_condition == "add_coarse"
            and self.coarse_condition_mlp is not None
            and self.coarse_condition_scale is not None
        ):
            coarse_feat = y_coarse.reshape(
                y_coarse.shape[0], y_coarse.shape[1], y_coarse.shape[2]
            )
            cond = self.coarse_condition_mlp(coarse_feat.unsqueeze(-1))
            residual = residual + self.coarse_condition_scale * cond

        return y_coarse + residual

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        spatial_codebook_for_encoder, patch_input, downsamp_input, res_input = self._prepare_inputs(
            history_data
        )

        if self.c2f_mode == "none":
            self.latest_coarse_pred = None
            self.latest_coarse_target = None
            output = self._forward_kasa_base(
                patch_input,
                downsamp_input,
                res_input,
                spatial_codebook_for_encoder,
                kwargs,
            )
        elif self.c2f_mode == "coarse_residual":
            output = self._forward_coarse_residual(
                patch_input,
                downsamp_input,
                res_input,
                spatial_codebook_for_encoder,
                future_data,
            )
        else:
            raise ValueError(f"Unsupported c2f_mode: {self.c2f_mode}")

        history_flow = history_data[..., 0]
        output = self.spatial_module.refine_prediction(output, history_flow)

        if self.use_graph_spectral_calibration:
            if self.post_spatial_mode != "adaptive_only":
                raise ValueError(
                    "Graph spectral calibration is only supported with "
                    "post_spatial_mode='adaptive_only' in this experiment."
                )
            adaptive_adj = self.spatial_module.get_adaptive_adj()
            output = self.graph_spectral_calibration(output, adaptive_adj)

        if self.input_dim > 3 and self.keep_output_prior_residual:
            if self.prior_source == "history":
                prior_data = history_data[..., 3:4]
            elif self.prior_source == "future":
                if future_data is None:
                    raise ValueError("future_data is required when prior_source='future'")
                if future_data.shape[-1] <= 3:
                    raise ValueError("future_data must contain channel 3 when prior_source='future'")
                prior_data = future_data[..., 3:4]
            else:
                raise ValueError(f"Unknown prior_source: {self.prior_source}")

            prior_residual = self.prior_mapper(prior_data)
            output = output + prior_residual

        return output
