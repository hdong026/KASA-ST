"""Online actor--critic for real resolution-graph interaction.

The actor scores forecast actions from the sample state only.  A physical
budget expands or contracts the feasible action set, but it never forces a
different route or changes the score of an action that was already feasible.
This gives a larger hard budget the intended option semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from torch import nn
from torch.distributions import Categorical

from .sequential_budget_policy import NEXT_RESOLUTIONS


Resolution = Optional[int]


class _SharedOnlineStateEncoder(nn.Module):
    """Compress sensor identity once, then share the result at every decision.

    The raw state has ten history channels per sensor and, after a transition,
    ten current-forecast channels per sensor.  A learned low-rank projection
    over sensors preserves sample-specific spatial information without three
    separate multi-million-parameter dense input layers.  Missing forecast
    channels at START are represented by zeros, which are genuinely known at
    that point and cannot leak a future forecast.
    """

    def __init__(
        self, node_count: int, width: int, dropout: float, spatial_rank: int = 8
    ):
        super().__init__()
        self.node_count = int(node_count)
        self.spatial_rank = min(int(spatial_rank), self.node_count)
        self.spatial_projection = nn.Linear(
            self.node_count, self.spatial_rank, bias=False
        )
        compressed_dim = 20 * self.spatial_rank + 48
        self.normalization = nn.LayerNorm(compressed_dim)
        self.projection = nn.Linear(compressed_dim, width)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, source: Resolution, state: torch.Tensor) -> torch.Tensor:
        nodes = self.node_count
        history_nodes = state[:, : 10 * nodes].reshape(-1, 10, nodes)
        context = state[:, 10 * nodes : 10 * nodes + 48]
        if source is None:
            forecast_nodes = torch.zeros_like(history_nodes)
        else:
            forecast_nodes = state[:, 10 * nodes + 48 :].reshape(-1, 10, nodes)
        spatial = self.spatial_projection(
            torch.cat((history_nodes, forecast_nodes), dim=1)
        ).flatten(1)
        compressed = torch.cat((spatial, context), dim=1)
        return self.dropout(
            self.activation(self.projection(self.normalization(compressed)))
        )


class OnlineResolutionActorCritic(nn.Module):
    """Actor--critic called only at states reached by executed transitions."""

    def __init__(
        self,
        node_count: int,
        hidden_dim: int = 32,
        dropout: float = 0.05,
        budget_min_ms: float = 0.0,
        budget_max_ms: float = 1.0,
    ):
        super().__init__()
        self.node_count = int(node_count)
        width = max(16, int(hidden_dim))
        self.encoder = _SharedOnlineStateEncoder(
            self.node_count, width, dropout
        )
        self.actors = nn.ModuleDict(
            {
                "start": nn.Linear(width, 3),
                "3": nn.Linear(width, 3),
                "4": nn.Linear(width, 2),
            }
        )
        self.critics = nn.ModuleDict(
            {
                "start": nn.Linear(width + 3, 1),
                "3": nn.Linear(width + 3, 1),
                "4": nn.Linear(width + 3, 1),
            }
        )
        for actor in self.actors.values():
            nn.init.zeros_(actor.weight)
            nn.init.zeros_(actor.bias)
        self.register_buffer("budget_min_ms", torch.tensor(float(budget_min_ms)))
        self.register_buffer("budget_max_ms", torch.tensor(float(budget_max_ms)))

    @staticmethod
    def _key(source: Resolution) -> str:
        return "start" if source is None else str(int(source))

    def set_budget_range(self, minimum_ms: float, maximum_ms: float) -> None:
        with torch.no_grad():
            self.budget_min_ms.fill_(float(minimum_ms))
            self.budget_max_ms.fill_(float(maximum_ms))

    def cost_condition(
        self, budget_ms: torch.Tensor, consumed_ms: torch.Tensor
    ) -> torch.Tensor:
        scale = (self.budget_max_ms - self.budget_min_ms).clamp_min(1e-5)
        return torch.stack(
            (
                (budget_ms - self.budget_min_ms) / scale,
                consumed_ms / self.budget_max_ms.clamp_min(1e-5),
                (budget_ms - consumed_ms) / scale,
            ),
            dim=1,
        )

    def forward(
        self,
        source: Resolution,
        state: torch.Tensor,
        budget_ms: torch.Tensor,
        consumed_ms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._key(source)
        hidden = self.encoder(source, state)
        return (
            self.actors[key](hidden),
            self.critics[key](
                torch.cat(
                    (hidden, self.cost_condition(budget_ms, consumed_ms)), dim=1
                )
            ).squeeze(1),
        )

    def act(
        self,
        source: Resolution,
        state: torch.Tensor,
        budget_ms: torch.Tensor,
        consumed_ms: torch.Tensor,
        feasible_mask: torch.Tensor | None = None,
        action_costs_ms: torch.Tensor | None = None,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        # action_costs_ms remains in the API for compatibility and accounting;
        # measured cost affects feasibility/reward, not the sample-quality score.
        del action_costs_ms
        logits, value = self(source, state, budget_ms, consumed_ms)
        logits = self._masked_logits(logits, feasible_mask)
        distribution = Categorical(logits=logits)
        action_index = distribution.sample() if sample else logits.argmax(dim=1)
        return {
            "action_index": action_index,
            "log_prob": distribution.log_prob(action_index),
            "entropy": distribution.entropy(),
            "value": value,
            "logits": logits,
        }

    def greedy(
        self,
        source: Resolution,
        state: torch.Tensor,
        feasible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Low-launch deployment path: actor only, without a distribution/critic."""
        logits = self.actors[self._key(source)](self.encoder(source, state))
        return self._masked_logits(logits, feasible_mask).argmax(dim=1)

    def evaluate_actions(
        self,
        source: Resolution,
        state: torch.Tensor,
        budget_ms: torch.Tensor,
        consumed_ms: torch.Tensor,
        action_index: torch.Tensor,
        feasible_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate freshly executed actions for bounded PPO updates."""
        logits, value = self(source, state, budget_ms, consumed_ms)
        distribution = Categorical(
            logits=self._masked_logits(logits, feasible_mask)
        )
        return (
            distribution.log_prob(action_index),
            distribution.entropy(),
            value,
        )

    @staticmethod
    def _masked_logits(
        logits: torch.Tensor, feasible_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if feasible_mask is None:
            return logits
        if feasible_mask.ndim == 1:
            feasible_mask = feasible_mask[None, :].expand(len(logits), -1)
        # Minimum-budget validation and completion-feasible transitions ensure
        # that a reached state has a legal continuation. Avoid synchronizing
        # the GPU merely to repeat that invariant at every decision: dynamic
        # dispatch already requires one host synchronization for the action.
        return logits.masked_fill(~feasible_mask, -torch.inf)

    @staticmethod
    def action_values(source: Resolution, device=None) -> torch.Tensor:
        return torch.tensor(NEXT_RESOLUTIONS[source], dtype=torch.long, device=device)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def completion_feasible_mask(
    prefix: Sequence[int],
    actions: Sequence[int],
    budget_ms: float,
    route_costs_ms: Mapping[tuple[int, ...], float],
) -> torch.Tensor:
    """Hard-cap mask based only on measured costs and the reached prefix."""
    prefix = tuple(int(value) for value in prefix)
    result = []
    for action in actions:
        candidate = prefix + (int(action),)
        result.append(
            any(
                route[: len(candidate)] == candidate and cost <= float(budget_ms) + 1e-6
                for route, cost in route_costs_ms.items()
            )
        )
    return torch.tensor(result, dtype=torch.bool)
