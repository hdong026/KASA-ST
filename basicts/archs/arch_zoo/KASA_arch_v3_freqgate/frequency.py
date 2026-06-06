"""Robust frequency descriptor from history flow (channel 0 only)."""

import torch
from torch import nn


class FrequencyDescriptor(nn.Module):
    """Per-node spectral distribution embedding from detrended history flow."""

    def __init__(self, freq_dim: int, num_bands: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_bands = num_bands
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(num_bands, freq_dim),
            nn.SiLU(),
            nn.Linear(freq_dim, freq_dim),
            nn.LayerNorm(freq_dim),
        )

    def _split_bands(self, amp_no_dc: torch.Tensor) -> torch.Tensor:
        """Split non-DC amplitude bins into low/mid/high bands."""
        n_bins = amp_no_dc.shape[1]
        if n_bins == 0:
            zeros = torch.zeros(
                amp_no_dc.shape[0], amp_no_dc.shape[2], self.num_bands,
                device=amp_no_dc.device, dtype=amp_no_dc.dtype,
            )
            return zeros

        if n_bins >= self.num_bands:
            edges = [0]
            for i in range(1, self.num_bands):
                edges.append(i * n_bins // self.num_bands)
            edges.append(n_bins)
            bands = []
            for i in range(self.num_bands):
                seg = amp_no_dc[:, edges[i]:edges[i + 1], :]
                bands.append(seg.mean(dim=1))
            return torch.stack(bands, dim=-1)

        # Fewer bins than bands: pad by repeating last available bin.
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
            freq_bands: [B, N, num_bands]
        """
        x = history_flow - history_flow.mean(dim=1, keepdim=True)
        amp = torch.fft.rfft(x, dim=1).abs()
        amp_no_dc = amp[:, 1:, :] if amp.shape[1] > 1 else amp[:, :0, :]

        freq_bands = self._split_bands(amp_no_dc)
        freq_bands = freq_bands / (freq_bands.sum(dim=-1, keepdim=True) + self.eps)
        freq_emb = self.mlp(freq_bands)
        return freq_emb, freq_bands
