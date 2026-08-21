"""Explicit learned history--forecast relation policy for online routing.

The forecaster is intentionally absent from this module.  It consumes only X
and the explicit Z_r that has already been produced by a real graph transition.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)

from .online_rl_policy import OnlineResolutionActorCritic


Resolution = Optional[int]


def xz_relation_state_features(
    history: torch.Tensor, forecast: torch.Tensor | None
) -> torch.Tensor:
    """Store ordered X and the reached Z_r in a resolution-independent layout."""
    flow = history[..., 0]
    if forecast is None:
        aligned = torch.zeros_like(flow)
    else:
        aligned = interpolate_forecast(forecast, flow.shape[1])[..., 0]
    # Calendar values are known from X and repeated over sensors in PEMS04.
    calendar = history[:, :, 0, 1:3].flatten(1)
    return torch.cat((flow.flatten(1), aligned.flatten(1), calendar), dim=1)


class LearnedHistoryForecastRelationEncoder(nn.Module):
    """Encode correspondence, not independent pooled X/Z summaries.

    A shared temporal encoder maps X and Z_r into the same latent coordinates.
    The learned matcher then receives only their contrast/interactions.  This
    makes every post-transition actor representation explicitly conditional on
    how forecast dynamics relate to historical dynamics.
    """

    def __init__(
        self,
        node_count: int,
        width: int,
        dropout: float,
        shape_dim: int = 24,
        relation_dim: int = 32,
        spatial_rank: int = 8,
    ):
        super().__init__()
        self.node_count = int(node_count)
        self.shape_dim = int(shape_dim)
        self.relation_dim = int(relation_dim)
        self.spatial_rank = min(int(spatial_rank), self.node_count)
        self.shared_temporal_encoder = nn.Sequential(
            nn.LayerNorm(12),
            nn.Linear(12, shape_dim),
            nn.SiLU(),
            nn.Linear(shape_dim, shape_dim),
            nn.SiLU(),
        )
        self.history_node_encoder = nn.Sequential(
            nn.Linear(shape_dim + 4, relation_dim), nn.SiLU()
        )
        # No plain X/Z concatenation reaches this matcher: its inputs are
        # signed/absolute latent contrast and multiplicative correspondence.
        self.relation_matcher = nn.Sequential(
            nn.LayerNorm(shape_dim * 3 + 4),
            nn.Linear(shape_dim * 3 + 4, relation_dim),
            nn.SiLU(),
            nn.Linear(relation_dim, relation_dim),
            nn.SiLU(),
        )
        self.spatial_projection = nn.Linear(
            self.node_count, self.spatial_rank, bias=False
        )
        self.calendar_encoder = nn.Sequential(
            nn.LayerNorm(24), nn.Linear(24, 16), nn.SiLU()
        )
        compressed_dim = relation_dim * (4 + self.spatial_rank) + 8 + 16
        self.normalization = nn.LayerNorm(compressed_dim)
        self.projection = nn.Linear(compressed_dim, width)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def _unpack(self, state: torch.Tensor):
        end = 12 * self.node_count
        history = state[:, :end].reshape(-1, 12, self.node_count).transpose(1, 2)
        forecast = state[:, end : 2 * end].reshape(
            -1, 12, self.node_count
        ).transpose(1, 2)
        calendar = state[:, 2 * end :]
        return history, forecast, calendar

    def _relation(self, source: Resolution, state: torch.Tensor):
        history, forecast, calendar = self._unpack(state)
        history_mean = history.mean(-1, keepdim=True)
        history_scale = history.std(-1, keepdim=True, unbiased=False).clamp_min(0.1)
        history_shape = (history - history_mean) / history_scale
        # Use X's scale for Z so discontinuity and amplitude mismatch survive.
        forecast_on_history_scale = (forecast - history_mean) / history_scale
        hx = self.shared_temporal_encoder(history_shape)
        hz = self.shared_temporal_encoder(forecast_on_history_scale)
        history_trend = history[..., -1] - history[..., 0]
        forecast_trend = forecast[..., -1] - forecast[..., 0]
        boundary = forecast[..., 0] - history[..., -1]
        volatility_delta = (
            (forecast[..., 1:] - forecast[..., :-1]).abs().mean(-1)
            - (history[..., 1:] - history[..., :-1]).abs().mean(-1)
        )
        scalar_relation = torch.stack(
            (forecast_trend - history_trend, boundary, volatility_delta,
             forecast.mean(-1) - history.mean(-1)),
            dim=-1,
        )
        if source is None:
            nodes = self.history_node_encoder(
                torch.cat((hx, torch.stack(
                    (history.mean(-1), history.std(-1, unbiased=False),
                     history_trend,
                     (history[..., 1:] - history[..., :-1]).abs().mean(-1)),
                    dim=-1)), dim=-1)
            )
        else:
            nodes = self.relation_matcher(
                torch.cat((hz - hx, (hz - hx).abs(), hz * hx, scalar_relation), dim=-1)
            )
        pooled = torch.cat(
            (nodes.mean(1), nodes.std(1, unbiased=False), nodes.amax(1), nodes.amin(1)),
            dim=1,
        )
        spatial = self.spatial_projection(nodes.transpose(1, 2)).flatten(1)
        amplitude = torch.stack(
            (
                scalar_relation[..., 0].mean(1), scalar_relation[..., 0].std(1, unbiased=False),
                scalar_relation[..., 1].mean(1), scalar_relation[..., 1].std(1, unbiased=False),
                scalar_relation[..., 2].mean(1), scalar_relation[..., 2].std(1, unbiased=False),
                scalar_relation[..., 3].mean(1), scalar_relation[..., 3].std(1, unbiased=False),
            ), dim=1,
        )
        compressed = torch.cat(
            (pooled, spatial, amplitude, self.calendar_encoder(calendar)), dim=1
        )
        diagnostics = {
            "latent_contrast_l2": (hz - hx).square().mean((1, 2)).sqrt(),
            "boundary_abs": boundary.abs().mean(1),
            "trend_mismatch_abs": (forecast_trend - history_trend).abs().mean(1),
            "volatility_mismatch_abs": volatility_delta.abs().mean(1),
            "relation_embedding_l2": nodes.square().mean((1, 2)).sqrt(),
        }
        return compressed, diagnostics

    def forward(self, source: Resolution, state: torch.Tensor) -> torch.Tensor:
        compressed, _ = self._relation(source, state)
        return self.dropout(
            self.activation(self.projection(self.normalization(compressed)))
        )

    @torch.no_grad()
    def diagnostics(self, source: Resolution, state: torch.Tensor):
        _, diagnostics = self._relation(source, state)
        return diagnostics


class OnlineXZRelationActorCritic(OnlineResolutionActorCritic):
    """Same actors, objective, feasibility, and actions; relation encoder only."""

    uses_xz_relation_state = True

    def __init__(
        self,
        node_count: int,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        budget_min_ms: float = 0.0,
        budget_max_ms: float = 1.0,
    ):
        super().__init__(
            node_count=node_count,
            hidden_dim=hidden_dim,
            dropout=dropout,
            budget_min_ms=budget_min_ms,
            budget_max_ms=budget_max_ms,
        )
        width = max(16, int(hidden_dim))
        self.encoder = LearnedHistoryForecastRelationEncoder(
            node_count, width, dropout
        )

    @staticmethod
    def state_features(history: torch.Tensor, forecast: torch.Tensor | None):
        return xz_relation_state_features(history, forecast)
