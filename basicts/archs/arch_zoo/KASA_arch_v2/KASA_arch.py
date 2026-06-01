from math import ceil
import torch
from torch import nn
import torch.nn.functional as F
from basicts.archs.arch_zoo.KASA_arch_v2.kasa_components import PatchEncoder, DownsampEncoder, ABCDSpatialModule

# ==========================================
# Minimal KAN definition (local; does not affect other modules)
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
        # Save config (input_dim=4 here)
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

        # Key 1: force input_dim=3 for submodules -> 3-channel conv in patch_emb, backbone same as LSTNN
        encoder_input_dim = 3 
        
        self.patch_encoder = PatchEncoder(self.td_size, self.dw_size, self.td_codebook, self.dw_codebook, self.spa_codebook, self.if_time_in_day, self.if_day_in_week, self.if_spatial,
                                          encoder_input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw, self.d_spa, self.output_len, self.num_layer)

        self.downsamp_encoder = DownsampEncoder(self.td_size, self.dw_size, self.td_codebook, self.dw_codebook, self.spa_codebook, self.if_time_in_day, self.if_day_in_week, self.if_spatial,
                                          encoder_input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw, self.d_spa, self.output_len, self.num_layer)

        # Main Residual (Standard LSTNN)
        self.residual = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

        # Forecast-side spectral residual calibration (4th channel prior template).
        self.use_prior_residual = model_args.get("use_prior_residual", True)
        if self.input_dim > 3:
            # Localized residual mapper: [B, output_len, N, 1] -> [B, output_len, N, 1]
            self.prior_mapper = SimpleKANLinear(1, 1)
            self.prior_kan = self.prior_mapper  # backward-compat alias for old checkpoints
            # Map history prior [B, input_len, N, 1] -> forecast prior [B, output_len, N, 1]
            self.prior_time_proj = nn.Conv2d(
                in_channels=self.input_len,
                out_channels=self.output_len,
                kernel_size=(1, 1),
            )
            # Horizon-wise gate; init small so residual starts near zero after sigmoid.
            self.prior_residual_gate = nn.Parameter(
                torch.full((1, self.output_len, 1, 1), -2.0)
            )

        # Self-diffusion retention gate for retention-regulated spatial diffusion.
        self.retention_gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        nn.init.constant_(self.retention_gate[-1].bias, 1.0)

    def _get_forecast_prior(self, history_data, future_data=None):
        """Build forecast-side prior template [B, output_len, N, 1].

        Uses future known covariate channel 3 when the runner provides it;
        otherwise projects the history prior to the forecast horizon.
        """
        if future_data is not None and future_data.shape[-1] > 3:
            prior = future_data[..., 3:4]
            if prior.shape[1] == self.output_len:
                return prior.to(dtype=history_data.dtype, device=history_data.device)

        prior_hist = history_data[..., 3:4]  # [B, input_len, N, 1]
        prior = self.prior_time_proj(prior_hist)  # [B, output_len, N, 1]
        return prior

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        legacy_prefix = prefix + "prior_kan."
        mapper_prefix = prefix + "prior_mapper."
        for key in list(state_dict.keys()):
            if key.startswith(legacy_prefix):
                state_dict[mapper_prefix + key[len(legacy_prefix):]] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None,
                batch_seen: int = 0, epoch: int = 0, train: bool = True, **kwargs) -> torch.Tensor:
        # history_data: [B, L, N, 4]
        
        # A/C scheme: GCN-enhanced spatial codebook before temporal encoders.
        enhanced_spa_emb = self.spatial_module.get_enhanced_spatial_embedding(self.spa_codebook)
        
        # 1. Backbone input (first 3 channels: Flow, TOD, DOW)
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

        patch_predict = self.patch_encoder(patch_input, spatial_codebook=enhanced_spa_emb)
        downsamp_predict = self.downsamp_encoder(downsamp_input, spatial_codebook=enhanced_spa_emb)

        # 4. Main Residual (Standard LSTNN)
        # Only use Flow (Channel 0)
        res_input = history_data[..., 0:1].permute(0, 1, 2, 3)
        res_out = self.residual(res_input)

        # If visualization mode, return backbone features here
        if kwargs.get("return_backbone", False):
            return patch_predict + downsamp_predict

        # Base Output (SOTA Performance Baseline)
        output = patch_predict + downsamp_predict + res_out

        # Retention-regulated spatial diffusion with explicit self-state retention.
        history_flow = history_data[..., 0]  # [B, input_len, N]
        diff_output = self.spatial_module.diffuse_prediction(output, history_flow)  # [B, output_len, N, 1]
        diffusion_alpha = self.spatial_module.get_diffusion_alpha()
        if diffusion_alpha is not None:
            diff_output = output + diffusion_alpha * (diff_output - output)

        gate_input = torch.cat(
            [output, diff_output, torch.abs(output - diff_output)], dim=-1
        )  # [B, output_len, N, 3]
        gamma = torch.sigmoid(self.retention_gate(gate_input))  # [B, output_len, N, 1]
        output = gamma * output + (1.0 - gamma) * diff_output

        # Forecast-side spectral residual calibration (gated prior residual).
        if self.input_dim > 3 and self.use_prior_residual:
            prior_data = self._get_forecast_prior(history_data, future_data)  # [B, output_len, N, 1]
            prior_residual = self.prior_mapper(prior_data)  # [B, output_len, N, 1]
            output = output + torch.sigmoid(self.prior_residual_gate) * prior_residual

        return output
