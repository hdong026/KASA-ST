from __future__ import annotations

from math import ceil
from typing import Optional

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.downsamp_emb import DownsampEncoder
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import interpolate_latent
from basicts.archs.arch_zoo.ChainForecasting_arch.patch_emb import PatchEncoder


class KASAHiddenStep(nn.Module):
    """Temporal backbone block that returns a hidden state, not a forecast head output."""

    def __init__(
        self,
        internal_len: int,
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
        hidden_dim: int,
        use_patch_branch: bool = True,
        use_downsample_branch: bool = True,
        use_linear_residual_branch: bool = True,
        patch_data_input_mode: str = "all",
        patch_embedding_mode: str = "serial_concat",
        patch_feature_dim=None,
        latent_cond_dim: int = 0,
    ):
        super().__init__()
        self.internal_len = internal_len
        self.input_len = input_len
        self.hidden_dim = int(hidden_dim)
        self.latent_cond_dim = int(latent_cond_dim)
        self.base_encoder_input_dim = 3
        self.cond_encoder_input_dim = 3 + (
            self.latent_cond_dim if self.latent_cond_dim > 0 else 1
        )

        enc_tail = dict(
            patch_data_input_mode=patch_data_input_mode,
            patch_embedding_mode=patch_embedding_mode,
            patch_feature_dim=patch_feature_dim,
        )
        enc_pos = (
            td_size,
            dw_size,
            td_codebook,
            dw_codebook,
            spa_codebook,
            if_time_in_day,
            if_day_in_week,
            if_spatial,
        )
        enc_dims = (
            patch_len,
            stride,
            d_d,
            d_td,
            d_dw,
            d_spa,
            internal_len,
            num_layer,
        )

        self.patch_encoder = PatchEncoder(
            *enc_pos,
            self.base_encoder_input_dim,
            *enc_dims,
            **enc_tail,
        )
        self.downsamp_encoder = DownsampEncoder(
            *enc_pos,
            self.base_encoder_input_dim,
            *enc_dims,
        )
        self.patch_encoder_cond = PatchEncoder(
            *enc_pos,
            self.cond_encoder_input_dim,
            *enc_dims,
            **enc_tail,
        )
        self.downsamp_encoder_cond = DownsampEncoder(
            *enc_pos,
            self.cond_encoder_input_dim,
            *enc_dims,
        )

        self.use_patch_branch = use_patch_branch
        self.use_downsample_branch = use_downsample_branch
        self.use_linear_residual_branch = use_linear_residual_branch
        self.patch_len = patch_len
        self.stride = stride

        self.hidden_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _build_step_input(
        self,
        history_data: torch.Tensor,
        prev_latent: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, bool]:
        x_main = history_data[..., :3]
        if prev_latent is not None and self.latent_cond_dim > 0:
            return torch.cat([x_main, prev_latent], dim=-1), True
        return x_main, False

    def forward(
        self,
        history_data: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
        spatial_codebook=None,
    ) -> torch.Tensor:
        step_input, use_cond = self._build_step_input(history_data, prev_latent)

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

        patch_encoder = self.patch_encoder_cond if use_cond else self.patch_encoder
        downsamp_encoder = self.downsamp_encoder_cond if use_cond else self.downsamp_encoder

        branch_outputs = []
        if self.use_patch_branch:
            branch_outputs.append(
                patch_encoder(patch_input, spatial_codebook=spatial_codebook)
            )
        if self.use_downsample_branch:
            branch_outputs.append(
                downsamp_encoder(downsamp_input, spatial_codebook=spatial_codebook)
            )

        if not branch_outputs:
            raise ValueError("At least one temporal branch must be enabled.")

        internal = sum(branch_outputs)
        hidden = self.hidden_proj(internal)
        return interpolate_latent(hidden, self.input_len)
