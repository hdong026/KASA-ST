from math import ceil

import torch
from einops import rearrange
from torch import nn
from basicts.archs.arch_zoo.ChainForecasting_arch.mlp import MultiLayerPerceptron


class TimeFeature2DPatchEmbedding(nn.Module):
    """Patch embedding that preserves [N, P, C] structure before projection."""

    def __init__(self, input_dim, patch_len, d_d, patch_feature_dim):
        super().__init__()
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.patch_feature_dim = patch_feature_dim
        self.feature_mix = nn.Linear(input_dim, patch_feature_dim)
        self.temporal_projection = nn.Linear(patch_len * patch_feature_dim, d_d)
        self.norm = nn.LayerNorm(d_d)

    def forward(self, patch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tensor: [B, M, N, P, C]
        Returns:
            data_emb: [B, M, N, d_d]
        """
        h = self.feature_mix(patch_tensor)
        h = torch.nn.functional.silu(h)
        batch_size, num_patches, num_nodes, patch_len, feat_dim = h.shape
        h_flat = h.reshape(batch_size, num_patches, num_nodes, patch_len * feat_dim)
        data_emb = self.temporal_projection(h_flat)
        data_emb = self.norm(data_emb)
        return data_emb


class PatchEncoder(nn.Module):
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
        patch_data_input_mode="all",
        patch_embedding_mode="serial_concat",
        patch_feature_dim=None,
    ):
        super(PatchEncoder, self).__init__()
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
        self.patch_len = patch_len
        self.encoder_input_dim = input_dim
        self.patch_data_input_mode = str(patch_data_input_mode).lower()
        self.patch_embedding_mode = str(patch_embedding_mode).lower()
        self.d_d = d_d

        if self.patch_embedding_mode not in {"serial_concat", "time_feature_2d"}:
            raise ValueError(
                f"Unsupported patch_embedding_mode: {patch_embedding_mode}. "
                "Expected 'serial_concat' or 'time_feature_2d'."
            )

        if self.patch_data_input_mode == "flow_only":
            data_input_dim = 1
        elif self.patch_data_input_mode == "all":
            data_input_dim = input_dim
        else:
            raise ValueError(
                f"Unsupported patch_data_input_mode: {patch_data_input_mode}. "
                "Expected 'all' or 'flow_only'."
            )
        self.data_input_dim = data_input_dim

        resolved_patch_feature_dim = d_d if patch_feature_dim is None else int(patch_feature_dim)
        self.patch_feature_dim = resolved_patch_feature_dim

        self.data_embedding_layer = None
        self.time_feature_2d_embedding = None
        if self.patch_embedding_mode == "serial_concat":
            self.data_embedding_layer = nn.Conv2d(
                in_channels=data_input_dim * patch_len,
                out_channels=d_d,
                kernel_size=(1, 1),
                bias=True,
            )
        else:
            self.time_feature_2d_embedding = TimeFeature2DPatchEmbedding(
                input_dim=input_dim,
                patch_len=patch_len,
                d_d=d_d,
                patch_feature_dim=resolved_patch_feature_dim,
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
        self.projection1 = nn.Conv2d(
            in_channels=(self.hidden_dim + d_spa * int(self.if_spatial)) * self.stride + d_td + d_dw,
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def _build_patch_tensor(self, patch_input: torch.Tensor) -> torch.Tensor:
        """Return patch tensor [B, M, N, P, C] for time_feature_2d mode."""
        min_channels = max(3, self.encoder_input_dim)
        if patch_input.shape[-1] < min_channels:
            raise ValueError(
                f"PatchEncoder needs at least {min_channels} channels for data/time "
                f"embeddings, but got {patch_input.shape[-1]}"
            )
        patch_tensor = patch_input[..., : self.encoder_input_dim]
        return patch_tensor.permute(0, 1, 3, 2, 4)

    def _embed_serial_concat(self, patch_input: torch.Tensor) -> torch.Tensor:
        if self.patch_data_input_mode == "flow_only":
            data_channel_indices = [0]
        elif self.patch_data_input_mode == "all":
            data_channel_indices = list(range(self.encoder_input_dim))
        else:
            raise ValueError(
                f"Unsupported patch_data_input_mode: {self.patch_data_input_mode}"
            )

        data_channels = [patch_input[..., i] for i in data_channel_indices]
        data_emb_input = torch.concat(data_channels, dim=2)

        patch_len = patch_input.shape[2]
        expected_channels = len(data_channel_indices) * patch_len
        if data_emb_input.shape[2] != expected_channels:
            raise ValueError(
                f"PatchEncoder data embedding channel mismatch: "
                f"expected {expected_channels}, got {data_emb_input.shape[2]}"
            )

        data_emb = self.data_embedding_layer(
            data_emb_input.permute(0, 2, 1, 3)
        ).permute(0, 2, 3, 1)
        return data_emb

    def _embed_time_feature_2d(self, patch_input: torch.Tensor) -> torch.Tensor:
        patch_tensor = self._build_patch_tensor(patch_input)
        return self.time_feature_2d_embedding(patch_tensor)

    def forward(self, patch_input, spatial_codebook=None):
        # patch_input: [B, M, P, N, C]
        batch_size, num, _, _, _ = patch_input.shape

        if self.if_day_in_week:
            day_in_week_data = patch_input[..., 2]
            day_start_idx = day_in_week_data[:, :, 0, :].long().clamp(0, self.dw_size - 1)
            day_end_idx = day_in_week_data[:, :, -1, :].long().clamp(0, self.dw_size - 1)
            day_in_week_start_emb = self.dw_codebook[day_start_idx]
            day_in_week_end_emb = self.dw_codebook[day_end_idx]
            future_day_in_week_emb = day_in_week_end_emb[:, -1, :, :].permute(0, 2, 1).unsqueeze(-1)
        else:
            day_in_week_start_emb, day_in_week_end_emb, future_day_in_week_emb = None, None, None

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

        if self.if_spatial:
            if spatial_codebook is None:
                spatial_codebook = self.spa_codebook
            spatial_emb = spatial_codebook.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1).expand(-1, num, -1, -1)
        else:
            spatial_emb = None

        if self.patch_embedding_mode == "serial_concat":
            data_emb = self._embed_serial_concat(patch_input)
        elif self.patch_embedding_mode == "time_feature_2d":
            data_emb = self._embed_time_feature_2d(patch_input)
        else:
            raise ValueError(f"Unsupported patch_embedding_mode: {self.patch_embedding_mode}")

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
        predict = self.projection1(hidden)

        return predict
