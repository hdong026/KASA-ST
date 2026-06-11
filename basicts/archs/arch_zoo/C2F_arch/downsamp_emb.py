from math import ceil

import torch
from einops import rearrange
from torch import nn
from basicts.archs.arch_zoo.C2F_arch.mlp import MultiLayerPerceptron


class DownsampEncoder(nn.Module):
    def __init__(
        self,
        td_size,
        dw_size,
        td_codebook,
        dw_codebook,
        spa_codebook,
        if_time_in_day,
        if_day_in_week,
        if_spatial,
        input_dim,
        patch_len,
        stride,
        d_d,
        d_td,
        d_dw,
        d_spa,
        output_len,
        num_layer,
        coarse_len=None,
    ):
        super(DownsampEncoder, self).__init__()
        self.td_codebook = td_codebook
        self.dw_codebook = dw_codebook
        self.spa_codebook = spa_codebook
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.if_spatial = if_spatial
        self.output_len = output_len
        self.td_size = td_size
        self.dw_size = dw_size
        self.stride = stride
        self.encoder_input_dim = input_dim
        self.coarse_len = coarse_len

        self.data_embedding_layer = nn.Conv2d(
            in_channels=input_dim * patch_len,
            out_channels=d_d,
            kernel_size=(1, 1),
            bias=True,
        )

        self.hidden_dim = d_d + d_dw * int(self.if_day_in_week) * 2 + d_td * int(self.if_time_in_day) * 2

        self.temporal_encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(
                    self.hidden_dim + d_spa * int(self.if_spatial),
                    self.hidden_dim + d_spa * int(self.if_spatial),
                )
                for _ in range(num_layer)
            ]
        )

        self.spatial_encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(
                    d_d + d_spa * int(self.if_spatial),
                    d_d + d_spa * int(self.if_spatial),
                )
                for _ in range(num_layer)
            ]
        )

        self.data_encoder = nn.Sequential(
            *[MultiLayerPerceptron(d_d, d_d) for _ in range(num_layer)]
        )

        proj_in_channels = (self.hidden_dim + d_spa * int(self.if_spatial)) * self.stride + d_td + d_dw
        self.projection1 = nn.Conv2d(
            in_channels=proj_in_channels,
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True,
        )

        self.coarse_projection1 = None
        if coarse_len is not None:
            self.coarse_projection1 = nn.Conv2d(
                in_channels=proj_in_channels,
                out_channels=coarse_len,
                kernel_size=(1, 1),
                bias=True,
            )

    def _forward_hidden(self, patch_input, spatial_codebook=None):
        # patch_input: [B, P, L, N, C]
        batch_size, num, _, _, _ = patch_input.shape

        if self.if_time_in_day:
            time_in_day_data = patch_input[..., 1]
            time_start_idx = torch.clamp(
                (time_in_day_data[:, :, 0, :] * self.td_size).long(), 0, self.td_size - 1
            )
            time_end_idx = torch.clamp(
                (time_in_day_data[:, :, -1, :] * self.td_size).long(), 0, self.td_size - 1
            )
            time_in_day_start_emb = self.td_codebook[time_start_idx]
            time_in_day_end_emb = self.td_codebook[time_end_idx]
            future_time_idx = (
                (time_in_day_data[:, -1, -1, :] * self.td_size + self.output_len) % self.td_size
            ).long()
            future_time_idx = torch.clamp(future_time_idx, 0, self.td_size - 1)
            future_time_in_day_emb = self.td_codebook[future_time_idx].permute(0, 2, 1).unsqueeze(-1)
        else:
            time_in_day_start_emb, time_in_day_end_emb, future_time_in_day_emb = None, None, None

        if self.if_day_in_week:
            day_in_week_data = patch_input[..., 2]
            day_start_idx = day_in_week_data[:, :, 0, :].long().clamp(0, self.dw_size - 1)
            day_end_idx = day_in_week_data[:, :, -1, :].long().clamp(0, self.dw_size - 1)
            day_in_week_start_emb = self.dw_codebook[day_start_idx]
            day_in_week_end_emb = self.dw_codebook[day_end_idx]
            future_day_in_week_emb = day_in_week_end_emb[:, -1, :, :].permute(0, 2, 1).unsqueeze(-1)
        else:
            day_in_week_start_emb, day_in_week_end_emb, future_day_in_week_emb = None, None, None

        if self.if_spatial:
            if spatial_codebook is None:
                spatial_codebook = self.spa_codebook
            spatial_emb = (
                spatial_codebook.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .unsqueeze(1)
                .expand(-1, num, -1, -1)
            )
        else:
            spatial_emb = None

        data_channels = [patch_input[..., i] for i in range(self.encoder_input_dim)]
        data_emb = self.data_embedding_layer(
            torch.concat(data_channels, dim=2).permute(0, 2, 1, 3)
        ).permute(0, 2, 3, 1)
        data_emb = self.data_encoder(data_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        if self.if_spatial:
            hidden_input = torch.concat((data_emb, spatial_emb), dim=-1)
        else:
            hidden_input = data_emb
        hidden = hidden_input.permute(0, 3, 1, 2)
        hidden = self.spatial_encoder(hidden).permute(0, 2, 3, 1)

        hidden = torch.concat(
            (time_in_day_start_emb, day_in_week_start_emb, hidden, time_in_day_end_emb, day_in_week_end_emb),
            dim=-1,
        ).permute(0, 3, 1, 2)
        hidden = self.temporal_encoder(hidden)

        hidden = rearrange(hidden, "B D P N -> B (D P) N").unsqueeze(-1)
        hidden = torch.concat((hidden, future_time_in_day_emb, future_day_in_week_emb), dim=1)
        return hidden

    def forward(self, patch_input, spatial_codebook=None):
        hidden = self._forward_hidden(patch_input, spatial_codebook=spatial_codebook)
        return self.projection1(hidden)

    def forward_coarse(self, patch_input, spatial_codebook=None):
        if self.coarse_projection1 is None:
            raise ValueError("coarse_projection1 is not configured; set coarse_len at init.")
        hidden = self._forward_hidden(patch_input, spatial_codebook=spatial_codebook)
        return self.coarse_projection1(hidden)
