from __future__ import annotations

import torch
from torch import nn


def spectral_band_energy_ratios(history_flow: torch.Tensor) -> torch.Tensor:
    """Compute low/mid/high rFFT power ratios per node.

    Args:
        history_flow: historical flow `[B, T, N]`.

    Returns:
        ratios `[B, N, 3]` for (low, mid, high), summing to 1 along the last dim.
    """
    # rFFT along time: [B, N, F]
    x = history_flow.permute(0, 2, 1).contiguous()
    spec = torch.fft.rfft(x, dim=-1)
    power = spec.real.square() + spec.imag.square()
    n_freq = power.shape[-1]
    cut1 = max(n_freq // 3, 1)
    cut2 = max((2 * n_freq) // 3, cut1 + 1)
    if cut2 >= n_freq:
        cut2 = n_freq - 1
    bands = [
        power[..., :cut1].sum(dim=-1),
        power[..., cut1:cut2].sum(dim=-1),
        power[..., cut2:].sum(dim=-1),
    ]
    energy = torch.stack(bands, dim=-1)
    denom = energy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return energy / denom


class SpectralBranchRouter(nn.Module):
    """Shared spectral stage router over patch / downsample / linear branches."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, history_flow: torch.Tensor, stage_ratio: float) -> torch.Tensor:
        """Return branch probabilities `[B, N, 3]` (patch, downsample, linear)."""
        ratios = spectral_band_energy_ratios(history_flow)
        batch_size, num_nodes, _ = ratios.shape
        stage = history_flow.new_full((batch_size, num_nodes, 1), float(stage_ratio))
        router_in = torch.cat([ratios, stage], dim=-1)
        logits = self.fc2(self.act(self.fc1(router_in)))
        return torch.softmax(logits, dim=-1)
