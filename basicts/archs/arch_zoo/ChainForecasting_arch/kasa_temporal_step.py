from __future__ import annotations

from math import ceil
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.downsamp_emb import DownsampEncoder
from basicts.archs.arch_zoo.ChainForecasting_arch.patch_emb import PatchEncoder


def interpolate_forecast(forecast: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linearly resample forecast [B, F, N, C] to [B, target_len, N, C]."""
    batch_size, forecast_len, num_nodes, channels = forecast.shape
    x = forecast.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, channels, forecast_len)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.reshape(batch_size, num_nodes, channels, target_len).permute(0, 3, 1, 2)


class KASATemporalStep(nn.Module):
    """One KASA temporal forecasting step without post spatial refinement."""

    def __init__(
        self,
        output_len: int,
        input_len: int,
        patch_len: int,
        stride: int,
        td_size: int,
        dw_size: int,
        td_codebook,
        dw_codebook,
        spa_codebook,
        if_time_in_day: bool,
        if_day_in_week: bool,
        if_spatial: bool,
        d_d: int,
        d_td: int,
        d_dw: int,
        d_spa: int,
        num_layer: int,
        use_patch_branch: bool = True,
        use_downsample_branch: bool = True,
        use_linear_residual_branch: bool = True,
        patch_data_input_mode: str = "all",
        patch_embedding_mode: str = "serial_concat",
        patch_feature_dim=None,
        use_prev_condition: bool = True,
    ):
        super().__init__()
        self.output_len = output_len
        self.input_len = input_len
        self.patch_len = patch_len
        self.stride = stride
        self.use_patch_branch = use_patch_branch
        self.use_downsample_branch = use_downsample_branch
        self.use_linear_residual_branch = use_linear_residual_branch
        self.use_prev_condition = use_prev_condition
        self.base_encoder_input_dim = 3
        self.cond_encoder_input_dim = 4

        self.patch_encoder = PatchEncoder(
            td_size,
            dw_size,
            td_codebook,
            dw_codebook,
            spa_codebook,
            if_time_in_day,
            if_day_in_week,
            if_spatial,
            self.base_encoder_input_dim,
            patch_len,
            stride,
            d_d,
            d_td,
            d_dw,
            d_spa,
            output_len,
            num_layer,
            patch_data_input_mode=patch_data_input_mode,
            patch_embedding_mode=patch_embedding_mode,
            patch_feature_dim=patch_feature_dim,
        )

        self.downsamp_encoder = DownsampEncoder(
            td_size,
            dw_size,
            td_codebook,
            dw_codebook,
            spa_codebook,
            if_time_in_day,
            if_day_in_week,
            if_spatial,
            self.base_encoder_input_dim,
            patch_len,
            stride,
            d_d,
            d_td,
            d_dw,
            d_spa,
            output_len,
            num_layer,
        )

        self.patch_encoder_cond = PatchEncoder(
            td_size,
            dw_size,
            td_codebook,
            dw_codebook,
            spa_codebook,
            if_time_in_day,
            if_day_in_week,
            if_spatial,
            self.cond_encoder_input_dim,
            patch_len,
            stride,
            d_d,
            d_td,
            d_dw,
            d_spa,
            output_len,
            num_layer,
            patch_data_input_mode=patch_data_input_mode,
            patch_embedding_mode=patch_embedding_mode,
            patch_feature_dim=patch_feature_dim,
        )

        self.downsamp_encoder_cond = DownsampEncoder(
            td_size,
            dw_size,
            td_codebook,
            dw_codebook,
            spa_codebook,
            if_time_in_day,
            if_day_in_week,
            if_spatial,
            self.cond_encoder_input_dim,
            patch_len,
            stride,
            d_d,
            d_td,
            d_dw,
            d_spa,
            output_len,
            num_layer,
        )

        self.residual = nn.Conv2d(
            in_channels=input_len,
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def _build_step_input(
        self,
        history_data: torch.Tensor,
        prev_forecast: Optional[torch.Tensor],
    ) -> tuple:
        x_main = history_data[..., :3]
        if (
            prev_forecast is not None
            and self.use_prev_condition
        ):
            cond = interpolate_forecast(prev_forecast, self.input_len)
            step_input = torch.cat([x_main, cond], dim=-1)
            return step_input, True
        return x_main, False

    def forward(
        self,
        history_data: torch.Tensor,
        prev_forecast: Optional[torch.Tensor] = None,
        spatial_codebook=None,
    ) -> torch.Tensor:
        step_input, use_cond = self._build_step_input(history_data, prev_forecast)

        in_len_add = ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if in_len_add:
            main_input_aug = torch.cat(
                (step_input[:, -1:, :, :].expand(-1, in_len_add, -1, -1), step_input),
                dim=1,
            )
        else:
            main_input_aug = step_input

        downsamp_input = [main_input_aug[:, i :: self.stride, :, :] for i in range(self.stride)]
        downsamp_input = torch.stack(downsamp_input, dim=1)
        patch_input = main_input_aug.unfold(
            dimension=1, size=self.patch_len, step=self.patch_len
        ).permute(0, 1, 4, 2, 3)

        if use_cond:
            patch_encoder = self.patch_encoder_cond
            downsamp_encoder = self.downsamp_encoder_cond
        else:
            patch_encoder = self.patch_encoder
            downsamp_encoder = self.downsamp_encoder

        branch_outputs = []
        if self.use_patch_branch:
            branch_outputs.append(
                patch_encoder(patch_input, spatial_codebook=spatial_codebook)
            )
        if self.use_downsample_branch:
            branch_outputs.append(
                downsamp_encoder(downsamp_input, spatial_codebook=spatial_codebook)
            )
        if self.use_linear_residual_branch:
            res_input = history_data[..., 0:1].permute(0, 1, 2, 3)
            branch_outputs.append(self.residual(res_input))

        if not branch_outputs:
            raise ValueError("At least one temporal branch must be enabled.")
        return sum(branch_outputs)
