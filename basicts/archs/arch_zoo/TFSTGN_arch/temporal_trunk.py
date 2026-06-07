"""Strong temporal trunk (KASA-style patch + downsample + flow residual)."""

from math import ceil

import torch
import torch.nn as nn

from basicts.archs.arch_zoo.KASA_arch_v2.downsamp_emb import DownsampEncoder
from basicts.archs.arch_zoo.KASA_arch_v2.patch_emb import PatchEncoder


class KASATemporalTrunk(nn.Module):
    """LSTNN-style temporal encoder with ToD/DoW embeddings and flow residual."""

    def __init__(
        self,
        node_size: int,
        input_len: int,
        output_len: int,
        input_dim: int = 4,
        patch_len: int = 3,
        stride: int = 4,
        td_size: int = 288,
        dw_size: int = 7,
        d_td: int = 32,
        d_dw: int = 32,
        d_d: int = 32,
        d_spa: int = 32,
        if_time_in_day: bool = True,
        if_day_in_week: bool = True,
        if_spatial: bool = False,
        num_layer: int = 2,
        use_prior_residual: bool = True,
        prior_mapper_type: str = "mlp",
    ):
        super().__init__()
        self.node_size = node_size
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.stride = stride

        self.td_codebook = None
        self.dw_codebook = None
        self.spa_codebook = None
        if if_time_in_day:
            self.td_codebook = nn.Parameter(torch.empty(td_size, d_td))
            nn.init.xavier_uniform_(self.td_codebook)
        if if_day_in_week:
            self.dw_codebook = nn.Parameter(torch.empty(dw_size, d_dw))
            nn.init.xavier_uniform_(self.dw_codebook)
        if if_spatial:
            self.spa_codebook = nn.Parameter(torch.empty(node_size, d_spa))
            nn.init.xavier_uniform_(self.spa_codebook)

        encoder_input_dim = 3
        self.patch_encoder = PatchEncoder(
            td_size,
            dw_size,
            self.td_codebook,
            self.dw_codebook,
            self.spa_codebook,
            if_time_in_day,
            if_day_in_week,
            if_spatial,
            encoder_input_dim,
            patch_len,
            stride,
            d_d,
            d_td,
            d_dw,
            d_spa,
            output_len,
            num_layer,
        )
        self.downsamp_encoder = DownsampEncoder(
            td_size,
            dw_size,
            self.td_codebook,
            self.dw_codebook,
            self.spa_codebook,
            if_time_in_day,
            if_day_in_week,
            if_spatial,
            encoder_input_dim,
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

        self.use_prior_residual = use_prior_residual and input_dim > 3
        if self.use_prior_residual:
            if prior_mapper_type == "mlp":
                self.prior_mapper = nn.Sequential(
                    nn.Linear(1, 16),
                    nn.SiLU(),
                    nn.Linear(16, 1),
                )
            elif prior_mapper_type == "linear":
                self.prior_mapper = nn.Linear(1, 1)
            else:
                raise ValueError(f"Unsupported prior_mapper_type: {prior_mapper_type}")

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        # history_data: [B, L, N, C]
        main_input = history_data[..., :3]

        in_len_add = ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if in_len_add:
            main_input_aug = torch.cat(
                (main_input[:, -1:, :, :].expand(-1, in_len_add, -1, -1), main_input),
                dim=1,
            )
        else:
            main_input_aug = main_input

        downsamp_input = [main_input_aug[:, i :: self.stride, :, :] for i in range(self.stride)]
        downsamp_input = torch.stack(downsamp_input, dim=1)
        patch_input = (
            main_input_aug.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
            .permute(0, 1, 4, 2, 3)
        )

        patch_predict = self.patch_encoder(patch_input, spatial_codebook=self.spa_codebook)
        downsamp_predict = self.downsamp_encoder(downsamp_input, spatial_codebook=self.spa_codebook)

        res_out = self.residual(history_data[..., 0:1])

        output = patch_predict + downsamp_predict + res_out

        if self.use_prior_residual:
            prior_residual = self.prior_mapper(history_data[..., 3:4])
            output = output + prior_residual

        return output
