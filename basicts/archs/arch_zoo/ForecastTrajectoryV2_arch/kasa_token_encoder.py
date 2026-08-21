"""Horizon-agnostic KASA patch/downsample trunks. Same MLPs as original TemporalStep."""

from __future__ import annotations

from math import ceil
from typing import Optional

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.downsamp_emb import DownsampEncoder
from basicts.archs.arch_zoo.ChainForecasting_arch.patch_emb import PatchEncoder


def _prepare_patch_inputs(history_data: torch.Tensor, input_len: int, patch_len: int, stride: int):
    x = history_data
    in_len_add = ceil(1.0 * input_len / stride) * stride - input_len
    if in_len_add:
        x = torch.cat((x[:, -1:, :, :].expand(-1, in_len_add, -1, -1), x), dim=1)
    down = torch.stack([x[:, i::stride, :, :] for i in range(stride)], dim=1)
    patch = x.unfold(dimension=1, size=patch_len, step=patch_len).permute(0, 1, 4, 2, 3)
    return patch, down


class KASATokenPatchEncoder(PatchEncoder):
    """Original PatchEncoder trunk; returns tokens [B, P, N, D] instead of horizon logits."""

    def forward_tokens(self, patch_input, spatial_codebook=None, dest_len: Optional[int] = None):
        batch_size, num, _, _, _ = patch_input.shape
        dest = int(dest_len) if dest_len is not None else int(self.output_len)

        if self.if_day_in_week:
            day_in_week_data = patch_input[..., 2]
            day_start_idx = day_in_week_data[:, :, 0, :].long().clamp(0, self.dw_size - 1)
            day_end_idx = day_in_week_data[:, :, -1, :].long().clamp(0, self.dw_size - 1)
            day_in_week_start_emb = self.dw_codebook[day_start_idx]
            day_in_week_end_emb = self.dw_codebook[day_end_idx]
        else:
            day_in_week_start_emb = day_in_week_end_emb = None

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
        else:
            time_in_day_start_emb = time_in_day_end_emb = None
        del dest  # dest length is handled by destination queries, not this trunk.

        if self.if_spatial:
            if spatial_codebook is None:
                spatial_codebook = self.spa_codebook
            spatial_emb = (
                spatial_codebook.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1).expand(-1, num, -1, -1)
            )
        else:
            spatial_emb = None

        if self.patch_embedding_mode == "serial_concat":
            data_emb = self._embed_serial_concat(patch_input)
        else:
            data_emb = self._embed_time_feature_2d(patch_input)
        data_emb = self.data_encoder(data_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        hidden_input = torch.concat((data_emb, spatial_emb), dim=-1) if self.if_spatial else data_emb
        hidden = self.spatial_encoder(hidden_input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        hidden = torch.concat(
            (time_in_day_start_emb, day_in_week_start_emb, hidden, time_in_day_end_emb, day_in_week_end_emb),
            dim=-1,
        ).permute(0, 3, 1, 2)
        hidden = self.temporal_encoder(hidden)
        return hidden.permute(0, 2, 3, 1).contiguous()  # [B, P, N, D]


class KASATokenDownsampEncoder(DownsampEncoder):
    def forward_tokens(self, patch_input, spatial_codebook=None, dest_len: Optional[int] = None):
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
        else:
            time_in_day_start_emb = time_in_day_end_emb = None
        if self.if_day_in_week:
            day_in_week_data = patch_input[..., 2]
            day_start_idx = day_in_week_data[:, :, 0, :].long().clamp(0, self.dw_size - 1)
            day_end_idx = day_in_week_data[:, :, -1, :].long().clamp(0, self.dw_size - 1)
            day_in_week_start_emb = self.dw_codebook[day_start_idx]
            day_in_week_end_emb = self.dw_codebook[day_end_idx]
        else:
            day_in_week_start_emb = day_in_week_end_emb = None
        if self.if_spatial:
            if spatial_codebook is None:
                spatial_codebook = self.spa_codebook
            spatial_emb = (
                spatial_codebook.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1).expand(-1, num, -1, -1)
            )
        else:
            spatial_emb = None
        data_channels = [patch_input[..., i] for i in range(self.encoder_input_dim)]
        data_emb = self.data_embedding_layer(
            torch.concat(data_channels, dim=2).permute(0, 2, 1, 3)
        ).permute(0, 2, 3, 1)
        data_emb = self.data_encoder(data_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        hidden_input = torch.concat((data_emb, spatial_emb), dim=-1) if self.if_spatial else data_emb
        hidden = self.spatial_encoder(hidden_input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        hidden = torch.concat(
            (time_in_day_start_emb, day_in_week_start_emb, hidden, time_in_day_end_emb, day_in_week_end_emb),
            dim=-1,
        ).permute(0, 3, 1, 2)
        hidden = self.temporal_encoder(hidden)
        return hidden.permute(0, 2, 3, 1).contiguous()
