"""Lightweight adaptive-resolution gates over forwarded conditions only."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.gcn import apply_adj
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class AdaptiveResolutionController(nn.Module):
    """Sample-wise temporal/spatial detail gates on forwarded conditions.

    Does not modify supervised forecasts. Shared across stage transitions; stage
    identity is provided via ``stage_ratio`` in the feature vector.
    """

    STAT_DIM = 6

    def __init__(
        self,
        hidden_dim: int = 16,
        gate_init: float = 0.98,
        temporal_kernel: int = 3,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gate_init = float(gate_init)
        self.temporal_kernel = int(temporal_kernel)
        if self.temporal_kernel < 1 or self.temporal_kernel % 2 == 0:
            raise ValueError(
                f"adaptive_resolution_temporal_kernel must be odd positive, "
                f"got {self.temporal_kernel}"
            )
        self.mlp = nn.Sequential(
            nn.Linear(self.STAT_DIM, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, _logit(self.gate_init))
        self._logged_stages: set[int] = set()

    def _extract_stats(
        self,
        condition: torch.Tensor,
        supervised_forecast: torch.Tensor,
        previous_condition: torch.Tensor | None,
        stage_ratio: float,
    ) -> torch.Tensor:
        """Build per-sample features ``[B, 6]``."""
        # Prefer supervised forecast for distributional stats.
        forecast = supervised_forecast
        b = forecast.shape[0]
        mean = forecast.mean(dim=(1, 2, 3))
        std = forecast.std(dim=(1, 2, 3), unbiased=False)
        if forecast.shape[1] > 1:
            temporal_diff = (forecast[:, 1:] - forecast[:, :-1]).abs().mean(dim=(1, 2, 3))
        else:
            temporal_diff = forecast.new_zeros(b)
        node_mean = forecast.mean(dim=2, keepdim=True)
        spatial_dev = (forecast - node_mean).abs().mean(dim=(1, 2, 3))
        if previous_condition is None:
            discrepancy = forecast.new_zeros(b)
        else:
            prev_aligned = interpolate_forecast(previous_condition, forecast.shape[1])
            discrepancy = (forecast - prev_aligned).abs().mean(dim=(1, 2, 3))
        ratio = forecast.new_full((b,), float(stage_ratio))
        return torch.stack(
            [mean, std, temporal_diff, spatial_dev, discrepancy, ratio],
            dim=-1,
        )

    def temporal_smooth(self, condition: torch.Tensor) -> torch.Tensor:
        """Length-preserving temporal moving average with replicate padding."""
        b, t, n, c = condition.shape
        k = self.temporal_kernel
        pad = k // 2
        # [B*N*C, 1, T]
        x = condition.permute(0, 2, 3, 1).reshape(b * n * c, 1, t)
        if pad > 0:
            x = F.pad(x, (pad, pad), mode="replicate")
        smoothed = F.avg_pool1d(x, kernel_size=k, stride=1)
        return smoothed.reshape(b, n, c, t).permute(0, 3, 1, 2).contiguous()

    def spatial_smooth(
        self,
        condition: torch.Tensor,
        adaptive_adj: torch.Tensor,
    ) -> torch.Tensor:
        """Pure graph propagation ``A @ condition`` (no alpha residual)."""
        if condition.shape[-1] == 1:
            x = condition.squeeze(-1)
            y = apply_adj(x, adaptive_adj)
            return y.unsqueeze(-1)
        # Generic channel-wise propagation.
        outs = []
        for c in range(condition.shape[-1]):
            outs.append(apply_adj(condition[..., c], adaptive_adj))
        return torch.stack(outs, dim=-1)

    def forward(
        self,
        condition: torch.Tensor,
        supervised_forecast: torch.Tensor,
        previous_condition: torch.Tensor | None,
        stage_ratio: float,
        adaptive_adj: torch.Tensor,
        stage_idx: int = 0,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Gate-blend temporal then spatial smoothing on ``condition`` only."""
        stats = self._extract_stats(
            condition=condition,
            supervised_forecast=supervised_forecast,
            previous_condition=previous_condition,
            stage_ratio=stage_ratio,
        )  # [B, 6]
        gates = torch.sigmoid(self.mlp(stats))  # [B, 2]
        temporal_detail_gate = gates[:, 0].view(-1, 1, 1, 1)
        spatial_detail_gate = gates[:, 1].view(-1, 1, 1, 1)

        temporal_smoothed = self.temporal_smooth(condition)
        temporal_condition = (
            temporal_detail_gate * condition
            + (1.0 - temporal_detail_gate) * temporal_smoothed
        )
        spatial_smoothed = self.spatial_smooth(temporal_condition, adaptive_adj)
        adaptive_condition = (
            spatial_detail_gate * temporal_condition
            + (1.0 - spatial_detail_gate) * spatial_smoothed
        )

        if stage_idx not in self._logged_stages:
            self._logged_stages.add(int(stage_idx))
            t_g = temporal_detail_gate.detach().float().flatten()
            s_g = spatial_detail_gate.detach().float().flatten()
            print(
                "[AdaptiveResolutionGate] "
                f"stage={stage_idx} "
                f"controller_input_shape={tuple(stats.shape)} "
                f"temporal_gate_mean={float(t_g.mean()):.4f} "
                f"temporal_gate_std={float(t_g.std(unbiased=False)):.4f} "
                f"spatial_gate_mean={float(s_g.mean()):.4f} "
                f"spatial_gate_std={float(s_g.std(unbiased=False)):.4f} "
                f"condition_in={tuple(condition.shape)} "
                f"condition_out={tuple(adaptive_condition.shape)}"
            )

        if not return_diagnostics:
            return adaptive_condition
        diag = {
            "temporal_detail_gate": temporal_detail_gate,
            "spatial_detail_gate": spatial_detail_gate,
            "controller_input_shape": tuple(stats.shape),
            "condition_in_shape": tuple(condition.shape),
            "condition_out_shape": tuple(adaptive_condition.shape),
            "stage_idx": int(stage_idx),
            "stage_ratio": float(stage_ratio),
        }
        return adaptive_condition, diag
