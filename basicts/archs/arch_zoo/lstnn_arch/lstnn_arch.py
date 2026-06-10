from math import ceil

import torch
from torch import nn

from basicts.archs.arch_zoo.lstnn_arch.patch_emb import PatchEncoder
from basicts.archs.arch_zoo.lstnn_arch.downsamp_emb import DownsampEncoder


def _flow_channel(y: torch.Tensor) -> torch.Tensor:
    return y if y.shape[-1] == 1 else y[..., 0:1]


class MultiscaleMLP(nn.Module):
    def __init__(self, **model_args):
        super(MultiscaleMLP, self).__init__()
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

        if self.if_time_in_day:
            self.td_codebook = nn.Parameter(torch.empty(self.td_size, self.d_td))
            nn.init.xavier_uniform_(self.td_codebook)

        if self.if_day_in_week:
            self.dw_codebook = nn.Parameter(torch.empty(self.dw_size, self.d_dw))
            nn.init.xavier_uniform_(self.dw_codebook)

        if self.if_spatial:
            self.spa_codebook = nn.Parameter(torch.empty(self.node_size, self.d_spa))
            nn.init.xavier_uniform_(self.spa_codebook)

        self.patch_encoder = PatchEncoder(
            self.td_size, self.td_codebook, self.dw_codebook, self.spa_codebook,
            self.if_time_in_day, self.if_day_in_week, self.if_spatial,
            self.input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw,
            self.d_spa, self.output_len, self.num_layer,
        )
        self.downsamp_encoder = DownsampEncoder(
            self.td_size, self.td_codebook, self.dw_codebook, self.spa_codebook,
            self.if_time_in_day, self.if_day_in_week, self.if_spatial,
            self.input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw,
            self.d_spa, self.output_len, self.num_layer,
        )
        self.residual = nn.Conv2d(
            in_channels=self.input_len,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        input_data = history_data[..., range(self.input_dim)]

        in_len_add = ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if not in_len_add:
            input_data = torch.cat(
                (input_data[:, -1:, :, :].expand(-1, in_len_add, -1, -1), input_data),
                dim=1,
            )

        downsamp_input = [input_data[:, i :: self.stride, :, :] for i in range(self.stride)]
        downsamp_input = torch.stack(downsamp_input, dim=1)
        patch_input = (
            input_data.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
            .permute(0, 1, 4, 2, 3)
        )

        patch_predict = self.patch_encoder(patch_input)
        downsamp_predict = self.downsamp_encoder(downsamp_input)
        output = patch_predict + downsamp_predict + self.residual(input_data)
        return output


class SpectralMixLSTNN(nn.Module):
    """LSTNN + prior branch + rFFT branch-level mixer (speed-friendly)."""

    def __init__(self, **model_args):
        super().__init__()
        self.node_size = model_args["node_size"]
        self.input_len = model_args["input_len"]
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

        self.use_prior_branch = model_args.get("use_prior_branch", True)
        self.use_spectral_mixer = model_args.get("use_spectral_mixer", True)
        self.mixer_global = model_args.get("mixer_global", False)

        encoder_input_dim = 3

        if self.if_time_in_day:
            self.td_codebook = nn.Parameter(torch.empty(self.td_size, self.d_td))
            nn.init.xavier_uniform_(self.td_codebook)
        if self.if_day_in_week:
            self.dw_codebook = nn.Parameter(torch.empty(self.dw_size, self.d_dw))
            nn.init.xavier_uniform_(self.dw_codebook)
        if self.if_spatial:
            self.spa_codebook = nn.Parameter(torch.empty(self.node_size, self.d_spa))
            nn.init.xavier_uniform_(self.spa_codebook)

        self.patch_encoder = PatchEncoder(
            self.td_size, self.td_codebook, self.dw_codebook, self.spa_codebook,
            self.if_time_in_day, self.if_day_in_week, self.if_spatial,
            encoder_input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw,
            self.d_spa, self.output_len, self.num_layer,
        )
        self.downsamp_encoder = DownsampEncoder(
            self.td_size, self.td_codebook, self.dw_codebook, self.spa_codebook,
            self.if_time_in_day, self.if_day_in_week, self.if_spatial,
            encoder_input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw,
            self.d_spa, self.output_len, self.num_layer,
        )
        self.residual = nn.Conv2d(
            in_channels=self.input_len,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )
        self.prior_proj = nn.Conv2d(
            in_channels=self.input_len,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )
        self.prior_scale = nn.Parameter(torch.tensor(0.1))

        self.branch_mixer = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        last = self.branch_mixer[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, 0.0)
        with torch.no_grad():
            last.bias[3] = -1.0

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        in_len_add = ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if not in_len_add:
            return torch.cat((x[:, -1:, :, :].expand(-1, in_len_add, -1, -1), x), dim=1)
        return x

    def _mixer_features(self, flow_x: torch.Tensor) -> torch.Tensor:
        # flow_x: [B, L, N, 1] -> [B, N, 4]
        spec = torch.fft.rfft(flow_x.squeeze(-1), dim=1).abs().unsqueeze(-1)
        f_bins = spec.shape[1]
        b1 = max(1, f_bins // 3)
        b2 = max(2, 2 * f_bins // 3)

        low = spec[:, :b1].mean(dim=1)
        mid = spec[:, b1:b2].mean(dim=1)
        high = spec[:, b2:].mean(dim=1)
        total = spec.mean(dim=1) + 1e-6

        low_ratio = low / total
        mid_ratio = mid / total
        high_ratio = high / total
        log_energy = torch.log1p(total)

        return torch.cat([low_ratio, mid_ratio, high_ratio, log_energy], dim=-1)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        flow_x = history_data[..., 0:1]
        base_x = history_data[..., 0:3]
        prior_x = history_data[..., 3:4] if history_data.shape[-1] > 3 else None

        base_aug = self._pad_input(base_x)
        downsamp_input = torch.stack(
            [base_aug[:, i :: self.stride, :, :] for i in range(self.stride)],
            dim=1,
        )
        patch_input = (
            base_aug.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
            .permute(0, 1, 4, 2, 3)
        )

        patch_predict = _flow_channel(self.patch_encoder(patch_input))
        downsamp_predict = _flow_channel(self.downsamp_encoder(downsamp_input))
        residual_predict = _flow_channel(self.residual(base_aug))

        if self.use_prior_branch and prior_x is not None:
            prior_predict = self.prior_proj(self._pad_input(prior_x))
            if prior_predict.shape[-1] != 1:
                prior_predict = prior_predict[..., 0:1]
        else:
            prior_predict = torch.zeros_like(patch_predict)

        if self.use_spectral_mixer:
            mixer_feat = self._mixer_features(flow_x)
            alpha = torch.softmax(self.branch_mixer(mixer_feat), dim=-1)
            if self.mixer_global:
                alpha = alpha.mean(dim=1, keepdim=True)

            alpha_patch = alpha[..., 0:1].unsqueeze(1)
            alpha_down = alpha[..., 1:2].unsqueeze(1)
            alpha_res = alpha[..., 2:3].unsqueeze(1)
            alpha_prior = alpha[..., 3:4].unsqueeze(1)

            pred = (
                alpha_patch * patch_predict
                + alpha_down * downsamp_predict
                + alpha_res * residual_predict
                + alpha_prior * prior_predict
            )
        else:
            pred = patch_predict + downsamp_predict + residual_predict
            if self.use_prior_branch and prior_x is not None:
                pred = pred + self.prior_scale * prior_predict

        return pred
