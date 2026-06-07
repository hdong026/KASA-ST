"""Module 3: ST-Spectral Gate — temporal and spatial gating from band_amp statistics."""

import torch
import torch.nn as nn


def _init_gate_bias_last_linear(seq: nn.Sequential, bias: float = -2.0) -> None:
    for module in reversed(list(seq.modules())):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.constant_(module.bias, bias)
            break


class STSpectralGate(nn.Module):
    """Modulate FS features with spectral-energy-driven temporal/spatial gates."""

    def __init__(
        self,
        use_spectral_gate: bool = True,
        use_temporal_gate: bool = True,
        use_spatial_gate: bool = True,
        gate_bias: float = -2.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.use_spectral_gate = use_spectral_gate
        self.use_temporal_gate = use_temporal_gate
        self.use_spatial_gate = use_spatial_gate
        self.eps = eps

        self.temporal_mlp = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.spatial_mlp = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        _init_gate_bias_last_linear(self.temporal_mlp, gate_bias)
        _init_gate_bias_last_linear(self.spatial_mlp, gate_bias)

        self.latest_temporal_gate = None
        self.latest_spatial_gate = None

    def forward(self, z: torch.Tensor, band_amp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, T_prime, N, D]
            band_amp: [B, N, C_tf, K, T_prime]
        Returns:
            z_gated: [B, T_prime, N, D]
        """
        b, t_prime, n, _ = z.shape
        k = band_amp.shape[3]

        if not self.use_spectral_gate:
            self.latest_temporal_gate = None
            self.latest_spatial_gate = None
            return z

        g_t = torch.ones(b, t_prime, 1, 1, device=z.device, dtype=z.dtype)
        if self.use_temporal_gate:
            e_t = band_amp.mean(dim=(1, 2, 3))
            high_energy = band_amp[:, :, :, k // 2 :, :].sum(dim=(1, 2, 3))
            total_energy = band_amp.sum(dim=(1, 2, 3)) + self.eps
            r_t = high_energy / total_energy
            feat_t = torch.stack([e_t, r_t], dim=-1)
            g_t = torch.sigmoid(self.temporal_mlp(feat_t)).unsqueeze(2)
        self.latest_temporal_gate = g_t.detach()

        g_s = torch.ones(b, 1, n, 1, device=z.device, dtype=z.dtype)
        if self.use_spatial_gate:
            amp = band_amp.mean(dim=2)
            amp_flat = amp.reshape(b, n, k * t_prime)
            mu = amp_flat.mean(dim=-1)
            std = amp_flat.std(dim=-1, unbiased=False)
            cv = std / (mu.abs() + self.eps)
            feat_s = torch.stack([mu, std, cv], dim=-1)
            g_s = torch.sigmoid(self.spatial_mlp(feat_s)).unsqueeze(1)
        self.latest_spatial_gate = g_s.detach()

        return z * g_t * g_s
