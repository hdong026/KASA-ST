from math import ceil
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from basicts.archs.arch_zoo.KASA_arch_v2.patch_emb import PatchEncoder
from basicts.archs.arch_zoo.KASA_arch_v2.downsamp_emb import DownsampEncoder
from basicts.archs.arch_zoo.KASA_arch_v2.gcn import ABCDSpatialModule

# ==========================================
# 最小限度的 KAN 定义 (局部定义，不影响其他)
# ==========================================
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

class KASA_v2(nn.Module):
    def __init__(self, **model_args):
        super(KASA_v2, self).__init__()
        # 参数保存
        self.node_size = model_args["node_size"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"] # 这里是4
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

        # All A/B/C/D spatial implementations are centralized in gcn.py.
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
        )

        # 🔥 关键修改 1: 强制传入 input_dim=3 给子模块
        # 这样 patch_emb.py 就会创建 3 通道的卷积层，完美适配 concat(0,1,2)
        # 这保证了主干网络和原版 LSTNN 100% 一致
        encoder_input_dim = 3 
        
        self.patch_encoder = PatchEncoder(self.td_size, self.dw_size, self.td_codebook, self.dw_codebook, self.spa_codebook, self.if_time_in_day, self.if_day_in_week, self.if_spatial,
                                          encoder_input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw, self.d_spa, self.output_len, self.num_layer)

        self.downsamp_encoder = DownsampEncoder(self.td_size, self.dw_size, self.td_codebook, self.dw_codebook, self.spa_codebook, self.if_time_in_day, self.if_day_in_week, self.if_spatial,
                                          encoder_input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw, self.d_spa, self.output_len, self.num_layer)

        # Main Residual (Standard LSTNN)
        self.residual = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        
        # Prior mapper on history channel 3
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

        # Optional weekly template for debug scripts (not used in forward).
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

    def _tod_dow_indices(self, tod, dow):
        if float(tod.max()) <= 1.0 + 1e-6:
            tod_idx = torch.floor(tod * self.slots_per_day).long()
        else:
            tod_idx = tod.long()
        tod_idx = torch.clamp(tod_idx, 0, self.slots_per_day - 1)

        if float(dow.max()) <= 1.0 + 1e-6:
            dow_idx = torch.floor(dow * 7).long()
        else:
            dow_idx = dow.long()
        dow_idx = torch.clamp(dow_idx, 0, 6)
        return tod_idx, dow_idx

    def _lookup_weekly_template(self, future_data):
        """Lookup train-only weekly spectral template using future ToD/DoW."""
        tod = future_data[..., 1]
        dow = future_data[..., 2]
        tod_idx, dow_idx = self._tod_dow_indices(tod, dow)
        week_idx = dow_idx * self.slots_per_day + tod_idx

        template = self.weekly_spectral_template  # [slots_per_week, N, 1]
        if template is None:
            raise ValueError("weekly_spectral_template is required for template lookup")
        node_idx = torch.arange(template.shape[1], device=future_data.device).view(1, 1, -1)
        prior = template[week_idx, node_idx, :]  # [B, F, N, 1]
        return prior.to(dtype=future_data.dtype, device=future_data.device)

    def _print_template_debug(self, future_data, prior_lookup, history_data):
        tod = future_data[..., 1]
        dow = future_data[..., 2]
        tod_idx, dow_idx = self._tod_dow_indices(tod, dow)
        week_idx = dow_idx * self.slots_per_day + tod_idx
        hist_prior = history_data[..., 3:4]

        print("=== template lookup debug ===")
        print(f"tod min/max: {tod.min().item():.6f} / {tod.max().item():.6f}")
        print(f"dow min/max: {dow.min().item():.6f} / {dow.max().item():.6f}")
        print(f"week_idx min/max: {week_idx.min().item()} / {week_idx.max().item()}")
        print(
            "prior_lookup mean/std/min/max: "
            f"{prior_lookup.mean().item():.6f} / {prior_lookup.std().item():.6f} / "
            f"{prior_lookup.min().item():.6f} / {prior_lookup.max().item():.6f}"
        )
        print(
            "history prior mean/std/min/max: "
            f"{hist_prior.mean().item():.6f} / {hist_prior.std().item():.6f} / "
            f"{hist_prior.min().item():.6f} / {hist_prior.max().item():.6f}"
        )
        if future_data.shape[-1] > 3:
            mae = torch.mean(torch.abs(prior_lookup - future_data[..., 3:4])).item()
            print(f"MAE(prior_lookup, future_data[...,3:4]): {mae:.8f}")
    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        # history_data: [B, L, N, 4]
        
        if self.use_pre_temporal_spatial_enhancement:
            spatial_codebook_for_encoder = self.spatial_module.get_enhanced_spatial_embedding(
                self.spa_codebook
            )
        else:
            spatial_codebook_for_encoder = self.spa_codebook
        
        # 1. 准备主干输入 (Flow, TOD, DOW) — channel 3 never enters temporal encoders
        if self.use_input_prior_enhancement and self.input_dim > 3:
            prior_data_for_input = history_data[..., 3:4]
            flow_delta = self.input_prior_mapper(prior_data_for_input)
            enhanced_flow = history_data[..., 0:1] + flow_delta
            main_input = torch.cat([enhanced_flow, history_data[..., 1:3]], dim=-1)
        else:
            main_input = history_data[..., :3]

        # 2. Patching (Copy from LSTNN logic)
        in_len_add = ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if in_len_add:
            main_input_aug = torch.cat((main_input[:, -1:, :, :].expand(-1, in_len_add, -1, -1), main_input), dim=1)
        else:
            main_input_aug = main_input

        # 3. Encoders Forward (Standard LSTNN)
        downsamp_input = [main_input_aug[:, i::self.stride, :, :] for i in range(self.stride)]
        downsamp_input = torch.stack(downsamp_input, dim=1)

        patch_input = main_input_aug.unfold(dimension=1, size=self.patch_len, step=self.patch_len).permute(0, 1, 4, 2, 3) 

        patch_predict = self.patch_encoder(
            patch_input, spatial_codebook=spatial_codebook_for_encoder
        )
        downsamp_predict = self.downsamp_encoder(
            downsamp_input, spatial_codebook=spatial_codebook_for_encoder
        )

        # 4. Main Residual (Standard LSTNN)
        # Only use Flow (Channel 0)
        res_input = history_data[..., 0:1].permute(0, 1, 2, 3)
        res_out = self.residual(res_input)

        # 🔥 【新增】如果是可视化模式，直接在这里返回骨干特征
        if kwargs.get("return_backbone", False):
            return patch_predict + downsamp_predict

        # Base Output (SOTA Performance Baseline)
        output = patch_predict + downsamp_predict + res_out

        # B/C/D scheme: refine prediction with spatial propagation.
        history_flow = history_data[..., 0]  # [B, L, N]
        output = self.spatial_module.refine_prediction(output, history_flow)
        
        if self.input_dim > 3 and self.keep_output_prior_residual:
            prior_data = history_data[..., 3:4]
            prior_residual = self.prior_mapper(prior_data)
            output = output + prior_residual

        return output
