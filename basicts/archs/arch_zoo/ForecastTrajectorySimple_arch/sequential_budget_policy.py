"""Budget-aware online next-resolution policy for ForecastTrajectorySimple."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)
from .progressive_selector import history_state_features


Resolution = Optional[int]
Trajectory = tuple[int, ...]

# These are graph edges, not complete route classes. Terminal paths are derived
# from this DAG and can grow automatically when a legal edge is added.
NEXT_RESOLUTIONS: dict[Resolution, tuple[int, ...]] = {
    None: (2, 3, 4),
    2: (4,),
    3: (4, 6, 12),
    4: (6, 12),
    6: (12,),
}
DECISION_RESOLUTIONS = (None, 3, 4)


def enumerate_terminal_trajectories(
    graph: Mapping[Resolution, Sequence[int]] = NEXT_RESOLUTIONS,
    terminal: int = 12,
) -> tuple[Trajectory, ...]:
    """Enumerate all terminal paths implied by the next-resolution DAG."""
    paths: list[Trajectory] = []

    def visit(source: Resolution, prefix: Trajectory) -> None:
        for target in graph.get(source, ()):
            route = prefix + (int(target),)
            if target == terminal:
                paths.append(route)
            else:
                visit(int(target), route)

    visit(None, ())
    return tuple(paths)


TRAJECTORIES = enumerate_terminal_trajectories()


def explicit_forecast_state_features(
    history: torch.Tensor, forecast: torch.Tensor
) -> torch.Tensor:
    """Encode history plus only the explicit forecast available now.

    The feature size is resolution-independent. Forecast values are summarized
    per sensor and deterministically interpolated to four points; no later
    forecast state or target is used.
    """
    return torch.cat(
        (history_state_features(history), explicit_forecast_only_features(forecast)),
        dim=1,
    )


def explicit_forecast_only_features(forecast: torch.Tensor) -> torch.Tensor:
    """Retain sensor identity while encoding only the forecast available now."""
    values = forecast[..., 0]
    differences = values[:, 1:] - values[:, :-1]
    mean_abs_step = (
        differences.abs().mean(dim=1)
        if differences.shape[1] > 0
        else torch.zeros_like(values[:, 0])
    )
    node_summary = torch.stack(
        (
            values.mean(dim=1),
            values.std(dim=1, unbiased=False),
            values[:, 0],
            values[:, -1],
            values[:, -1] - values[:, 0],
            mean_abs_step,
        ),
        dim=1,
    ).flatten(1)
    resampled = interpolate_forecast(forecast, 4)[..., 0].flatten(1)
    return torch.cat((node_summary, resampled), dim=1)


def _pool_node_channels(values: torch.Tensor) -> torch.Tensor:
    """Deterministic sample statistics for [B, channels, sensors]."""
    return torch.cat(
        (
            values.mean(dim=2),
            values.std(dim=2, unbiased=False),
            values.amax(dim=2),
            values.amin(dim=2),
        ),
        dim=1,
    )


def compact_online_features(
    features: torch.Tensor, node_count: int, has_forecast: bool
) -> torch.Tensor:
    """Convert a legacy expanded online state to its lightweight form."""
    history_end = 10 * int(node_count)
    context = features[:, history_end : history_end + 48]
    history_nodes = features[:, :history_end].reshape(-1, 10, int(node_count))
    if has_forecast:
        forecast_nodes = features[:, history_end + 48 :].reshape(
            -1, 10, int(node_count)
        )
        nodes = torch.cat((history_nodes, forecast_nodes), dim=1)
    else:
        nodes = history_nodes
    return torch.cat((_pool_node_channels(nodes), context), dim=1)


def compact_history_state_features(history: torch.Tensor) -> torch.Tensor:
    """Direct deployment history state: 40 pooled sensor values + context."""
    flow = history[..., 0]
    differences = flow[:, 1:] - flow[:, :-1]
    nodes = torch.cat(
        (
            torch.stack(
                (
                    flow.mean(dim=1),
                    flow.std(dim=1, unbiased=False),
                    flow[:, -1],
                    flow[:, -1] - flow[:, 0],
                    flow[:, -1] - flow[:, -2],
                    differences.abs().mean(dim=1),
                ),
                dim=1,
            ),
            flow[:, -4:],
        ),
        dim=1,
    )
    context = torch.cat(
        (
            flow.mean(dim=2),
            flow.std(dim=2, unbiased=False),
            history[:, :, 0, 1:3].flatten(1),
        ),
        dim=1,
    )
    return torch.cat((_pool_node_channels(nodes), context), dim=1)


def compact_forecast_state_features(
    history: torch.Tensor, forecast: torch.Tensor
) -> torch.Tensor:
    """Direct deployment state using history and the explicit forecast now."""
    flow = history[..., 0]
    history_differences = flow[:, 1:] - flow[:, :-1]
    history_nodes = torch.cat(
        (
            torch.stack(
                (
                    flow.mean(dim=1),
                    flow.std(dim=1, unbiased=False),
                    flow[:, -1],
                    flow[:, -1] - flow[:, 0],
                    flow[:, -1] - flow[:, -2],
                    history_differences.abs().mean(dim=1),
                ),
                dim=1,
            ),
            flow[:, -4:],
        ),
        dim=1,
    )
    values = forecast[..., 0]
    forecast_differences = values[:, 1:] - values[:, :-1]
    mean_abs_step = (
        forecast_differences.abs().mean(dim=1)
        if forecast_differences.shape[1] > 0
        else torch.zeros_like(values[:, 0])
    )
    forecast_nodes = torch.cat(
        (
            torch.stack(
                (
                    values.mean(dim=1),
                    values.std(dim=1, unbiased=False),
                    values[:, 0],
                    values[:, -1],
                    values[:, -1] - values[:, 0],
                    mean_abs_step,
                ),
                dim=1,
            ),
            interpolate_forecast(forecast, 4)[..., 0],
        ),
        dim=1,
    )
    context = torch.cat(
        (
            flow.mean(dim=2),
            flow.std(dim=2, unbiased=False),
            history[:, :, 0, 1:3].flatten(1),
        ),
        dim=1,
    )
    return torch.cat(
        (_pool_node_channels(torch.cat((history_nodes, forecast_nodes), dim=1)), context),
        dim=1,
    )


class _NextResolutionValueHead(nn.Module):
    """Cheap pooled-state encoder with a value for every legal next edge.

    The previous policy learned a sensor MLP independently at every decision
    and for both constraint modes.  Besides being relatively expensive at
    batch size one, direct action classification let the majority action erase
    small but useful sample-specific loss differences.  This head contains no
    sensor-wise learned pass: deterministic moments pool the online sensor
    state and one shared trunk feeds two tiny cost-to-go regressors.
    """

    def __init__(
        self,
        node_count: int,
        has_forecast: bool,
        action_count: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.node_count = int(node_count)
        self.has_forecast = bool(has_forecast)
        node_dim = 20 if has_forecast else 10
        pooled_dim = node_dim * 4 + 48
        width = max(16, hidden_dim // 2)
        self.trunk = nn.Sequential(
            nn.Linear(pooled_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # The constraint is deliberately injected only after state encoding.
        # This makes the sample state the main signal and the budget/price a
        # preference over that state, rather than a shortcut route identity.
        self.hard_value = nn.Linear(width + 1, action_count)
        self.average_value = nn.Linear(width + 1, action_count)

    def forward(
        self, features: torch.Tensor, condition: torch.Tensor, mode: str
    ) -> torch.Tensor:
        encoded = self.trunk(features)
        conditioned = torch.cat((encoded, condition[:, None]), dim=1)
        if mode == "hard":
            return self.hard_value(conditioned)
        if mode == "average":
            return self.average_value(conditioned)
        raise ValueError(f"Unknown value mode {mode!r}.")


class SequentialBudgetPolicy(nn.Module):
    """Fitted value-of-refinement policy over online next-resolution edges.

    Two small policies share the state representation but have distinct heads:
    ``hard`` consumes a physical per-sample latency cap, while ``average``
    consumes the dual price used to meet a dataset/service-level mean cap.
    Lambda is internal; users specify milliseconds in both cases.
    """

    def __init__(
        self,
        node_count: int,
        hidden_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.node_count = int(node_count)
        self.history_dim = 88
        self.forecast_dim = 128
        self.value_heads = self._make_heads(hidden_dim, dropout)
        self.register_buffer("history_mean", torch.zeros(self.history_dim))
        self.register_buffer("history_std", torch.ones(self.history_dim))
        self.register_buffer("state3_mean", torch.zeros(self.forecast_dim))
        self.register_buffer("state3_std", torch.ones(self.forecast_dim))
        self.register_buffer("state4_mean", torch.zeros(self.forecast_dim))
        self.register_buffer("state4_std", torch.ones(self.forecast_dim))
        self.register_buffer("budget_min", torch.tensor(0.0))
        self.register_buffer("budget_max", torch.tensor(1.0))
        self.register_buffer("log_lambda_mean", torch.tensor(0.0))
        self.register_buffer("log_lambda_std", torch.tensor(1.0))

    def _make_heads(self, hidden_dim: int, dropout: float) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "start": _NextResolutionValueHead(
                    self.node_count, False, 3, hidden_dim, dropout
                ),
                "3": _NextResolutionValueHead(
                    self.node_count, True, 3, hidden_dim, dropout
                ),
                "4": _NextResolutionValueHead(
                    self.node_count, True, 2, hidden_dim, dropout
                ),
            }
        )

    @staticmethod
    def _moments(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return values.mean(dim=0), values.std(dim=0, unbiased=False).clamp_min(1e-5)

    @torch.no_grad()
    def fit_normalizers(
        self,
        history: torch.Tensor,
        state3: torch.Tensor,
        state4: torch.Tensor,
        budgets_ms: torch.Tensor,
        lambdas: torch.Tensor,
    ) -> None:
        for values, mean_buffer, std_buffer in (
            (history, self.history_mean, self.history_std),
            (state3, self.state3_mean, self.state3_std),
            (state4, self.state4_mean, self.state4_std),
        ):
            mean, std = self._moments(values)
            mean_buffer.copy_(mean)
            std_buffer.copy_(std)
        self.budget_min.copy_(budgets_ms.min())
        self.budget_max.copy_(budgets_ms.max())
        log_lambda = torch.log1p(lambdas)
        self.log_lambda_mean.copy_(log_lambda.mean())
        self.log_lambda_std.copy_(
            log_lambda.std(unbiased=False).clamp_min(1e-5)
        )

    @staticmethod
    def _key(source: Resolution) -> str:
        return "start" if source is None else str(int(source))

    def _normalize_state(self, source: Resolution, features: torch.Tensor) -> torch.Tensor:
        if source is None and features.shape[1] != self.history_dim:
            features = compact_online_features(features, self.node_count, False)
        elif source in (3, 4) and features.shape[1] != self.forecast_dim:
            features = compact_online_features(features, self.node_count, True)
        if source is None:
            return (features - self.history_mean) / self.history_std
        if source == 3:
            return (features - self.state3_mean) / self.state3_std
        if source == 4:
            return (features - self.state4_mean) / self.state4_std
        raise ValueError(f"Resolution {source} is not a learned decision state.")

    def hard_values(
        self, source: Resolution, features: torch.Tensor, budget_ms: torch.Tensor
    ) -> torch.Tensor:
        scale = (self.budget_max - self.budget_min).clamp_min(1e-5)
        condition = (budget_ms - self.budget_min) / scale
        return self.value_heads[self._key(source)](
            self._normalize_state(source, features), condition, "hard"
        )

    def average_values(
        self, source: Resolution, features: torch.Tensor, lambda_mae_per_ms: torch.Tensor
    ) -> torch.Tensor:
        condition = (
            torch.log1p(lambda_mae_per_ms) - self.log_lambda_mean
        ) / self.log_lambda_std
        return self.value_heads[self._key(source)](
            self._normalize_state(source, features), condition, "average"
        )

    # Compatibility helpers for downstream callers that only need rankings.
    # Values are minimized, while the retired API exposed maximized logits.
    def hard_logits(
        self, source: Resolution, features: torch.Tensor, budget_ms: torch.Tensor
    ) -> torch.Tensor:
        return -self.hard_values(source, features, budget_ms)

    def average_logits(
        self, source: Resolution, features: torch.Tensor, lambda_mae_per_ms: torch.Tensor
    ) -> torch.Tensor:
        return -self.average_values(source, features, lambda_mae_per_ms)


def route_extends(route: Trajectory, prefix: Sequence[int]) -> bool:
    prefix = tuple(int(value) for value in prefix)
    return route[: len(prefix)] == prefix


def feasible_action_mask(
    prefix: Sequence[int],
    actions: Sequence[int],
    budget_ms: float,
    route_costs_ms: Mapping[Trajectory, float],
) -> torch.Tensor:
    """Mask next actions that have no terminal completion within a hard cap."""
    values = []
    prefix = tuple(int(value) for value in prefix)
    for action in actions:
        next_prefix = prefix + (int(action),)
        values.append(
            any(
                route_extends(route, next_prefix) and cost <= budget_ms
                for route, cost in route_costs_ms.items()
            )
        )
    return torch.tensor(values, dtype=torch.bool)
