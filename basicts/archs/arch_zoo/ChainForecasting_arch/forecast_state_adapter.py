from __future__ import annotations

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)


class ForecastStateAdapter(nn.Module):
    """Shared light Forecast-State Dynamics Adapter.

    Last projection is zero-initialized so corrected_state == current_state at init.

    correction_scale:
      - \"transition_delta\": legacy |Δ| + mean(|Δ|) scale (original state_adapter)
      - \"sample_scale\": condition-only sample-level abs mean scale
    delta_feature:
      - \"raw\": use state_delta as input channel
      - \"normalized\": use clamped normalized_delta as input channel
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        epsilon: float = 0.05,
        correction_scale: str = "transition_delta",
        delta_feature: str = "raw",
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.epsilon = float(epsilon)
        self.correction_scale = str(correction_scale).lower()
        self.delta_feature = str(delta_feature).lower()
        if self.correction_scale not in {"transition_delta", "sample_scale"}:
            raise ValueError(
                f"Unsupported correction_scale: {self.correction_scale}. "
                "Expected 'transition_delta' or 'sample_scale'."
            )
        if self.delta_feature not in {"raw", "normalized"}:
            raise ValueError(
                f"Unsupported delta_feature: {self.delta_feature}. "
                "Expected 'raw' or 'normalized'."
            )
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
        self._last_correction: torch.Tensor | None = None

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
        if self.delta_feature == "normalized":
            delta_denom = state_delta.detach().abs().mean(
                dim=(1, 2, 3), keepdim=True
            ).clamp_min(1e-3)
            delta_feat = (state_delta / delta_denom).clamp(-3.0, 3.0)
        else:
            delta_feat = state_delta

        stage_channel = current_state.new_full(current_state.shape, float(stage_ratio))
        adapter_input = torch.cat(
            [current_state, prev_aligned, delta_feat, stage_channel],
            dim=-1,
        )  # [B, h, N, 4]

        x = adapter_input.permute(0, 3, 1, 2).contiguous()  # [B, 4, h, N]
        raw_correction = self.proj_out(self.act2(self.dw(self.act1(self.proj_in(x)))))
        raw_correction = raw_correction.permute(0, 2, 3, 1).contiguous()  # [B, h, N, 1]

        if self.correction_scale == "sample_scale":
            sample_scale = current_state.detach().abs().mean(
                dim=(1, 2, 3), keepdim=True
            ).clamp_min(1e-3)
            correction = self.epsilon * torch.tanh(raw_correction) * sample_scale
        else:
            transition_scale = state_delta.abs() + state_delta.abs().mean(
                dim=(1, 2, 3), keepdim=True
            )
            correction = self.epsilon * torch.tanh(raw_correction) * transition_scale

        self._last_correction = correction
        return current_state + correction

