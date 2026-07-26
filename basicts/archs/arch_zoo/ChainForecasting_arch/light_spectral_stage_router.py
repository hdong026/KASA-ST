from __future__ import annotations

import torch
from torch import nn


class LightSpectralStageRouter(nn.Module):
    """Sample-level, stage-aware light spectral router for temporal branch fusion.

    Shared across T3/T6/T12. Zero-init last layer yields coefficients == 1.
    """

    def __init__(self, hidden_dim: int = 8, max_deviation: float = 0.05):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_deviation = float(max_deviation)
        self.fc1 = nn.Linear(4, self.hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(self.hidden_dim, 3)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @staticmethod
    def build_router_input(history_flow: torch.Tensor, stage_ratio: float) -> torch.Tensor:
        """Build sample-level spectral features `[B, 4]`.

        Args:
            history_flow: `[B, T, N]` historical flow.
            stage_ratio: current_horizon / final_horizon.
        """
        flow = history_flow - history_flow.mean(dim=1, keepdim=True)
        spectrum = torch.fft.rfft(flow, dim=1)
        power = spectrum.abs().square()
        power = power[:, 1:, :]  # drop DC
        power = power.mean(dim=-1)  # [B, F]
        bands = torch.tensor_split(power, 3, dim=1)
        energy = torch.stack([band.sum(dim=1) for band in bands], dim=-1)  # [B, 3]
        energy_ratio = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        stage = history_flow.new_full((history_flow.shape[0], 1), float(stage_ratio))
        return torch.cat([energy_ratio, stage], dim=-1)

    def coefficients_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Map logits `[B, 3]` to bounded coefficients summing to 3."""
        z = torch.tanh(logits)
        z = z - z.mean(dim=-1, keepdim=True)
        z = z / z.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        return 1.0 + self.max_deviation * z

    def forward(self, history_flow: torch.Tensor, stage_ratio: float) -> torch.Tensor:
        """Return branch coefficients `[B, 3]` for (patch, downsample, linear)."""
        router_input = self.build_router_input(history_flow, stage_ratio)
        logits = self.fc2(self.act(self.fc1(router_input)))
        return self.coefficients_from_logits(logits)
