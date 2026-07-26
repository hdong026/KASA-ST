from __future__ import annotations

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)


class ForecastStateAdapter(nn.Module):
    """Shared light Forecast-State Dynamics Adapter for Z3→Z6 and Z6→Z12.

    Last projection is zero-initialized so corrected_state == current_state at init.
    """

    def __init__(self, hidden_dim: int = 16, epsilon: float = 0.05):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.epsilon = float(epsilon)
        self.proj_in = nn.Conv2d(4, self.hidden_dim, kernel_size=1)
        self.act1 = nn.SiLU()
        self.dw = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=self.hidden_dim,
        )
        self.act2 = nn.SiLU()
        self.proj_out = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        current_state: torch.Tensor,
        previous_state: torch.Tensor,
        stage_ratio: float,
    ) -> torch.Tensor:
        """Apply bounded residual correction.

        Args:
            current_state: `[B, h, N, 1]` post-spatial forecast at current stage.
            previous_state: previous-stage forecast state (any length).
            stage_ratio: current_horizon / final_horizon.
        """
        target_len = current_state.shape[1]
        prev_aligned = interpolate_forecast(previous_state, target_len)
        state_delta = current_state - prev_aligned
        stage_channel = current_state.new_full(current_state.shape, float(stage_ratio))
        adapter_input = torch.cat(
            [current_state, prev_aligned, state_delta, stage_channel],
            dim=-1,
        )  # [B, h, N, 4]

        x = adapter_input.permute(0, 3, 1, 2).contiguous()  # [B, 4, h, N]
        raw_correction = self.proj_out(self.act2(self.dw(self.act1(self.proj_in(x)))))
        raw_correction = raw_correction.permute(0, 2, 3, 1).contiguous()  # [B, h, N, 1]

        transition_scale = state_delta.abs() + state_delta.abs().mean(
            dim=(1, 2, 3), keepdim=True
        )
        correction = self.epsilon * torch.tanh(raw_correction) * transition_scale
        return current_state + correction
