"""Robust frequency descriptor from history flow (channel 0 only)."""

import torch
from torch import nn

FREQ_FEATURE_DIM = 6
FREQ_NUM_BANDS = 3


class FrequencyDescriptor(nn.Module):
    """Per-node spectral + lightweight time-domain embedding from history flow."""

    def __init__(self, freq_dim: int, num_bands: int = FREQ_NUM_BANDS, eps: float = 1e-6):
        super().__init__()
        self.num_bands = num_bands
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(FREQ_FEATURE_DIM, freq_dim),
            nn.SiLU(),
            nn.Linear(freq_dim, freq_dim),
            nn.LayerNorm(freq_dim),
        )

    def _split_bands(self, amp_no_dc: torch.Tensor) -> torch.Tensor:
        """Split non-DC amplitude bins into low/mid/high bands."""
        n_bins = amp_no_dc.shape[1]
        if n_bins == 0:
            return torch.zeros(
                amp_no_dc.shape[0],
                amp_no_dc.shape[2],
                self.num_bands,
                device=amp_no_dc.device,
                dtype=amp_no_dc.dtype,
            )

        if n_bins >= self.num_bands:
            edges = [0]
            for i in range(1, self.num_bands):
                edges.append(i * n_bins // self.num_bands)
            edges.append(n_bins)
            bands = []
            for i in range(self.num_bands):
                seg = amp_no_dc[:, edges[i] : edges[i + 1], :]
                bands.append(seg.mean(dim=1))
            return torch.stack(bands, dim=-1)

        bands = [amp_no_dc[:, i, :] for i in range(n_bins)]
        while len(bands) < self.num_bands:
            bands.append(bands[-1])
        return torch.stack(bands[: self.num_bands], dim=-1)

    def forward(self, history_flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            history_flow: [B, T, N]
        Returns:
            freq_emb: [B, N, freq_dim]
            freq_bands: [B, N, num_bands] normalized band ratios
        """
        x = history_flow - history_flow.mean(dim=1, keepdim=True)
        amp = torch.fft.rfft(x, dim=1).abs()
        amp_no_dc = amp[:, 1:, :] if amp.shape[1] > 1 else amp[:, :0, :]

        freq_bands = self._split_bands(amp_no_dc)
        band_ratios = freq_bands / (freq_bands.sum(dim=-1, keepdim=True) + self.eps)

        time_std = x.std(dim=1)
        slope = x[:, -1, :] - x[:, 0, :]
        temporal_mean = x.mean(dim=1)
        last_deviation = x[:, -1, :] - temporal_mean

        freq_feat = torch.stack(
            [
                band_ratios[..., 0],
                band_ratios[..., 1],
                band_ratios[..., 2],
                time_std,
                slope,
                last_deviation,
            ],
            dim=-1,
        )
        freq_emb = self.mlp(freq_feat)
        return freq_emb, band_ratios
