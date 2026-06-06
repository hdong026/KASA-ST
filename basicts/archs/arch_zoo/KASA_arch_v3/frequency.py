"""Lightweight frequency descriptor from history flow."""

import torch
from torch import nn


class FrequencyDescriptor(nn.Module):
    """Extract per-node frequency-band energy embedding from history flow."""

    def __init__(self, freq_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, freq_dim),
            nn.SiLU(),
            nn.Linear(freq_dim, freq_dim),
        )

    def forward(self, history_flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_flow: [B, T, N]
        Returns:
            freq_emb: [B, N, freq_dim]
        """
        amp = torch.fft.rfft(history_flow, dim=1).abs()  # [B, F, N]
        n_bins = amp.shape[1]
        b1 = max(1, n_bins // 3)
        b2 = max(b1 + 1, 2 * n_bins // 3)

        low = amp[:, :b1].mean(dim=1)
        mid = amp[:, b1:b2].mean(dim=1)
        if b2 < n_bins:
            high = amp[:, b2:].mean(dim=1)
        else:
            high = mid

        freq_feat = torch.stack([low, mid, high], dim=-1)  # [B, N, 3]
        return self.mlp(freq_feat)
