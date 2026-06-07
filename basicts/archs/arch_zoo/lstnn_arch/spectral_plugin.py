"""Lightweight spectral residual gate for LSTNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralPlugin(nn.Module):
    """rFFT band-energy gates for residual feature enhancement."""

    def __init__(
        self,
        hidden_dim: int = 1,
        num_bands: int = 4,
        init_scale: float = 0.01,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_bands = num_bands
        self.eps = eps
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

        self.spatial_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.temporal_mlp = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def _band_energy(self, x0: torch.Tensor) -> torch.Tensor:
        # x0: [B, L, N, 1] -> band_energy [B, N, K]
        spec = torch.fft.rfft(x0.squeeze(-1), dim=1).abs()  # [B, F, N]
        f_bins = spec.shape[1]
        edges = torch.linspace(0, f_bins, self.num_bands + 1, device=spec.device).long()
        bands = []
        for k in range(self.num_bands):
            seg = spec[:, edges[k] : edges[k + 1], :]
            bands.append(seg.mean(dim=1))
        return torch.stack(bands, dim=-1)

    def _spatial_gate(self, band_energy: torch.Tensor) -> torch.Tensor:
        # band_energy: [B, N, K] -> g_s [B, N, 1]
        mu = band_energy.mean(dim=-1)
        std = band_energy.std(dim=-1, unbiased=False)
        cv = std / (mu.abs() + self.eps)
        feat_s = torch.stack([mu, std, cv], dim=-1)
        return torch.sigmoid(self.spatial_mlp(feat_s))

    def _temporal_gate(self, x0: torch.Tensor) -> torch.Tensor:
        # x0: [B, L, N, 1] -> g_t [B, L, 1]
        energy_t = x0.abs().mean(dim=(2, 3))
        diff_t = x0.diff(dim=1).abs().mean(dim=(2, 3))
        diff_t = F.pad(diff_t, (1, 0))
        feat_t = torch.stack([energy_t, diff_t], dim=-1)
        return torch.sigmoid(self.temporal_mlp(feat_t))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: history_data [B, L, N, C]
            h: hidden feature [B, T, N, D] or [B, N, D]
        Returns:
            delta: same shape as h
        """
        x0 = x[..., :1]
        band_energy = self._band_energy(x0)
        g_s = self._spatial_gate(band_energy)

        if h.dim() == 4:
            b, t, n, d = h.shape
            g_t = self._temporal_gate(x0)
            if g_t.shape[1] != t:
                g_t = F.interpolate(
                    g_t.transpose(1, 2),
                    size=t,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            gate = g_t.view(b, t, 1, 1) * g_s.view(b, 1, n, 1)
            return self.scale * h * gate

        if h.dim() == 3:
            b, n, d = h.shape
            return self.scale * h * g_s.view(b, n, 1)

        raise ValueError(f"SpectralPlugin expects h dim 3 or 4, got shape {tuple(h.shape)}")
