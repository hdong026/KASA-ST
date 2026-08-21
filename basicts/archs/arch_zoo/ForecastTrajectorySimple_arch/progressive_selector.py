"""Small online selector for the five ForecastTrajectorySimple routes.

The route tree has only two genuine decisions.  Before forecasting, history
chooses the first resolution (2, 3, or 4).  If resolution 3 is chosen, its
explicit forecast is then available and chooses the next resolution (4, 6,
or 12).  All other continuations are forced by the declared route set.
"""

from __future__ import annotations

import torch
from torch import nn


ROUTES = (
    (3, 6, 12),
    (3, 12),
    (2, 4, 12),
    (4, 12),
    (3, 4, 6, 12),
)


def history_state_features(history: torch.Tensor) -> torch.Tensor:
    """Return compact features containing history only.

    ``history`` is the normal normalized model input [B, T, N, C].  Flow is
    summarized per sensor while the known calendar channels are included once
    per time step (they are repeated over sensors in PEMS04).
    """
    flow = history[..., 0]
    differences = flow[:, 1:] - flow[:, :-1]
    node_features = torch.stack(
        (
            flow.mean(dim=1),
            flow.std(dim=1, unbiased=False),
            flow[:, -1],
            flow[:, -1] - flow[:, 0],
            flow[:, -1] - flow[:, -2],
            differences.abs().mean(dim=1),
        ),
        dim=1,
    ).flatten(1)
    recent = flow[:, -4:].flatten(1)
    temporal = torch.cat(
        (flow.mean(dim=2), flow.std(dim=2, unbiased=False)), dim=1
    )
    calendar = history[:, :, 0, 1:3].flatten(1)
    return torch.cat((node_features, recent, temporal, calendar), dim=1)


def forecast_state_features(
    history: torch.Tensor, forecast_z3: torch.Tensor
) -> torch.Tensor:
    """Return the online state after the explicit three-step forecast exists."""
    base = history_state_features(history)
    z3 = forecast_z3[..., 0]
    last = history[:, -1, :, 0]
    forecast_delta = z3 - last[:, None, :]
    return torch.cat((base, z3.flatten(1), forecast_delta.flatten(1)), dim=1)


class _OnlineStateEncoder(nn.Module):
    """Permutation-stable sensor encoder matched to sample-level MAE."""

    def __init__(
        self, node_count: int, has_z3: bool, hidden_dim: int, dropout: float
    ):
        super().__init__()
        self.node_count = int(node_count)
        self.has_z3 = bool(has_z3)
        node_dim = 16 if has_z3 else 10
        sensor_dim = max(16, hidden_dim // 8)
        self.sensor_encoder = nn.Sequential(
            nn.Linear(node_dim, sensor_dim),
            nn.GELU(),
            nn.Linear(sensor_dim, sensor_dim),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(48, 32), nn.GELU(),
        )
        self.network = nn.Sequential(
            nn.Linear(sensor_dim * 4 + 32, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.sensor_head = nn.Sequential(
            nn.Linear(sensor_dim + 32, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2),
        )
        # The untrained selector is the safe fixed-canonical policy.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -4.0)
        nn.init.zeros_(self.sensor_head[-1].weight)
        nn.init.constant_(self.sensor_head[-1].bias, -4.0)

    def _encode(self, features: torch.Tensor):
        node_count = self.node_count
        node_summary = features[:, : 6 * node_count].reshape(
            -1, 6, node_count
        ).transpose(1, 2)
        recent = features[:, 6 * node_count : 10 * node_count].reshape(
            -1, 4, node_count
        ).transpose(1, 2)
        node_parts = [node_summary, recent]
        context = features[:, 10 * node_count : 10 * node_count + 48]
        if self.has_z3:
            offset = 10 * node_count + 48
            forecast = features[:, offset : offset + 3 * node_count].reshape(
                -1, 3, node_count
            ).transpose(1, 2)
            delta = features[:, offset + 3 * node_count :].reshape(
                -1, 3, node_count
            ).transpose(1, 2)
            node_parts.extend((forecast, delta))
        encoded = self.sensor_encoder(torch.cat(node_parts, dim=2))
        pooled = torch.cat(
            (
                encoded.mean(dim=1),
                encoded.std(dim=1, unbiased=False),
                encoded.amax(dim=1),
                encoded.amin(dim=1),
            ),
            dim=1,
        )
        context_encoded = self.context_encoder(context)
        return encoded, pooled, context_encoded

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, pooled, context = self._encode(features)
        return self.network(torch.cat((pooled, context), dim=1))

    def forward_sensors(self, features: torch.Tensor) -> torch.Tensor:
        encoded, _, context = self._encode(features)
        context = context[:, None, :].expand(-1, self.node_count, -1)
        return self.sensor_head(torch.cat((encoded, context), dim=2))


class ProgressiveTrajectorySelector(nn.Module):
    """Predict pairwise benefit logits at the two online decisions.

    The first output scores START->2 and START->4 against canonical START->3.
    The second scores 3->12 and 3->4 against canonical 3->6. A positive score
    is required to leave the canonical action, making the zero-information
    behavior safe and explicit.
    """

    def __init__(
        self,
        history_dim: int,
        z3_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.10,
        safety_margin: float = 0.0,
    ):
        super().__init__()
        if (history_dim - 48) % 10 != 0:
            raise ValueError("history_dim does not match the online feature layout.")
        node_count = (history_dim - 48) // 10
        if z3_dim != history_dim + 6 * node_count:
            raise ValueError("z3_dim does not match the online feature layout.")
        self.node_count = node_count
        self.initial = _OnlineStateEncoder(
            node_count, False, hidden_dim, dropout
        )
        self.after_z3 = _OnlineStateEncoder(
            node_count, True, hidden_dim, dropout
        )
        self.safety_margin = float(safety_margin)
        self.register_buffer("history_mean", torch.zeros(history_dim))
        self.register_buffer("history_std", torch.ones(history_dim))
        self.register_buffer("z3_mean", torch.zeros(z3_dim))
        self.register_buffer("z3_std", torch.ones(z3_dim))
        self.register_buffer("initial_target_mean", torch.zeros(2))
        self.register_buffer("initial_target_std", torch.ones(2))
        self.register_buffer("z3_target_mean", torch.zeros(2))
        self.register_buffer("z3_target_std", torch.ones(2))

    @torch.no_grad()
    def fit_normalizers(
        self,
        history_features: torch.Tensor,
        z3_features: torch.Tensor,
        initial_targets: torch.Tensor,
        z3_targets: torch.Tensor,
    ) -> None:
        def moments(values: torch.Tensor):
            mean = values.mean(dim=0)
            std = values.std(dim=0, unbiased=False).clamp_min(1e-5)
            return mean, std

        self.history_mean.copy_(moments(history_features)[0])
        self.history_std.copy_(moments(history_features)[1])
        self.z3_mean.copy_(moments(z3_features)[0])
        self.z3_std.copy_(moments(z3_features)[1])
        self.initial_target_mean.copy_(moments(initial_targets)[0])
        self.initial_target_std.copy_(moments(initial_targets)[1])
        self.z3_target_mean.copy_(moments(z3_targets)[0])
        self.z3_target_std.copy_(moments(z3_targets)[1])

    def normalized_predictions(
        self, history_features: torch.Tensor, z3_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial = self.initial(
            (history_features - self.history_mean) / self.history_std
        )
        after_z3 = self.after_z3((z3_features - self.z3_mean) / self.z3_std)
        return initial, after_z3

    def normalized_sensor_predictions(
        self, history_features: torch.Tensor, z3_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial = self.initial.forward_sensors(
            (history_features - self.history_mean) / self.history_std
        )
        after_z3 = self.after_z3.forward_sensors(
            (z3_features - self.z3_mean) / self.z3_std
        )
        return initial, after_z3

    def cost_differences(
        self, history_features: torch.Tensor, z3_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial, after_z3 = self.normalized_predictions(history_features, z3_features)
        initial = initial * self.initial_target_std + self.initial_target_mean
        after_z3 = after_z3 * self.z3_target_std + self.z3_target_mean
        return initial, after_z3

    def select_route_indices(
        self, history_features: torch.Tensor, z3_features: torch.Tensor
    ) -> torch.Tensor:
        """Select one of ``ROUTES`` without consulting any future target."""
        initial_logits, z3_logits = self.normalized_predictions(
            history_features, z3_features
        )
        zeros = torch.zeros_like(initial_logits[:, :1])
        # An alternative must have positive predicted benefit (plus margin).
        initial_scores = torch.cat(
            (zeros, initial_logits - self.safety_margin), dim=1
        )
        initial_action = initial_scores.argmax(dim=1)  # 0:3, 1:2, 2:4

        z3_scores = torch.cat((zeros, z3_logits - self.safety_margin), dim=1)
        z3_action = z3_scores.argmax(dim=1)  # 0:6, 1:12, 2:4

        route_index = torch.empty_like(initial_action)
        route_index[initial_action == 1] = 2  # [2,4,12]
        route_index[initial_action == 2] = 3  # [4,12]
        reached_z3 = initial_action == 0
        route_index[reached_z3] = torch.where(
            z3_action[reached_z3] == 0,
            torch.zeros_like(z3_action[reached_z3]),
            torch.where(
                z3_action[reached_z3] == 1,
                torch.ones_like(z3_action[reached_z3]),
                torch.full_like(z3_action[reached_z3], 4),
            ),
        )
        return route_index


def decision_targets(route_losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create pairwise TRAIN excess-loss targets against the canonical route."""
    if route_losses.ndim < 2 or route_losses.shape[1] != len(ROUTES):
        raise ValueError("route_losses must have shape [samples, 5, ...].")
    initial = torch.stack(
        (route_losses[:, 2] - route_losses[:, 0],
         route_losses[:, 3] - route_losses[:, 0]),
        dim=1,
    )
    after_z3 = torch.stack(
        (route_losses[:, 1] - route_losses[:, 0],
         route_losses[:, 4] - route_losses[:, 0]),
        dim=1,
    )
    return initial, after_z3
