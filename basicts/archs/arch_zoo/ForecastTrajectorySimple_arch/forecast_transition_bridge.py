"""Composition-stable forecast transition bridge."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    KASATemporalStep,
    interpolate_forecast,
)


class ForecastTransitionBridge(nn.Module):
    """Predict a bounded correction to an explicit forecast anchor.

    A transition state must remain a forecast because a later native F2F stage
    consumes it as one. Conditioned edges anchor on the current forecast
    resampled to the destination resolution. Start edges use last-observation
    persistence. A KASA temporal step predicts only the correction.

    All correction heads are zero-initialized, so a new bridge starts as its
    stable anchor rather than an arbitrary absolute forecast. The smooth bound
    prevents composition from turning an explicit state into an unbounded
    hidden code while retaining unit slope around zero.
    """

    def __init__(
        self,
        *,
        source_resolution: Optional[int],
        target_resolution: int,
        correction_limit: float = 2.0,
        **temporal_step_kwargs,
    ):
        super().__init__()
        if target_resolution <= 0:
            raise ValueError("target_resolution must be positive.")
        if correction_limit <= 0:
            raise ValueError("correction_limit must be positive.")
        self.source_resolution = source_resolution
        self.target_resolution = int(target_resolution)
        self.correction_limit = float(correction_limit)
        self.predictor = KASATemporalStep(
            output_len=self.target_resolution,
            use_prev_condition=True,
            **temporal_step_kwargs,
        )
        self._zero_correction_heads()

    @staticmethod
    def _zero_module(module: nn.Module) -> None:
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def _zero_correction_heads(self) -> None:
        for encoder in (
            self.predictor.patch_encoder,
            self.predictor.downsamp_encoder,
            self.predictor.patch_encoder_cond,
            self.predictor.downsamp_encoder_cond,
        ):
            self._zero_module(encoder.projection1)
        self._zero_module(self.predictor.residual)

    def anchor(
        self,
        history_data: torch.Tensor,
        previous_forecast: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if previous_forecast is not None:
            return interpolate_forecast(previous_forecast, self.target_resolution)
        last_observation = history_data[:, -1:, :, :1]
        return last_observation.expand(-1, self.target_resolution, -1, -1)

    def forward(
        self,
        history_data: torch.Tensor,
        previous_forecast: Optional[torch.Tensor],
        *,
        spatial_codebook=None,
    ) -> torch.Tensor:
        anchor = self.anchor(history_data, previous_forecast)
        condition = (
            None
            if previous_forecast is None
            else interpolate_forecast(previous_forecast, self.target_resolution)
        )
        raw_correction = self.predictor(
            history_data,
            prev_forecast=condition,
            spatial_codebook=spatial_codebook,
        )
        limit = self.correction_limit
        correction = limit * torch.tanh(raw_correction / limit)
        return anchor + correction
