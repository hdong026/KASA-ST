"""Constrained trajectory-level GRPO for the frozen post-Z3 route fork.

The forecasting environment has exactly one policy decision:

    X -> Z3 -> {Z12, Z6 -> Z12}

The module does not own or update forecasting parameters.  It exposes the
actual reached ResolutionNative state to a small actor and executes sampled
continuations through the frozen shared reasoner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .f2f_cot_resolution_native_v1 import (
    HistoryEvidence,
    ResolutionNativeReasoningState,
)


SHORT = 0
LONG = 1


def _select_evidence(evidence: HistoryEvidence, indices: torch.Tensor) -> HistoryEvidence:
    return HistoryEvidence(
        tokens=evidence.tokens.index_select(0, indices),
        history_data=evidence.history_data.index_select(0, indices),
        history_flow=evidence.history_flow.index_select(0, indices),
        td_codebook=evidence.td_codebook,
        dw_codebook=evidence.dw_codebook,
        spa_codebook=evidence.spa_codebook,
    )


def select_reasoning_state(
    state: ResolutionNativeReasoningState, indices: torch.Tensor
) -> ResolutionNativeReasoningState:
    """Select or repeat batch members without changing their reached prefix."""
    return ResolutionNativeReasoningState(
        evidence=_select_evidence(state.evidence, indices),
        latest_forecast=(
            None
            if state.latest_forecast is None
            else state.latest_forecast.index_select(0, indices)
        ),
        current_resolution=int(state.current_resolution),
        forecasts=tuple(value.index_select(0, indices) for value in state.forecasts),
        resolutions=tuple(int(value) for value in state.resolutions),
    )


def _select_diagnostics(
    diagnostics: Mapping[str, object], indices: torch.Tensor
) -> dict[str, object]:
    batch = int(indices.max().item()) + 1 if indices.numel() else 0
    selected: dict[str, object] = {}
    for key, value in diagnostics.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] >= batch:
            selected[key] = value.index_select(0, indices)
        else:
            selected[key] = value
    return selected


@dataclass
class PostZ3Observation:
    """Inference-safe tensors available after the real 0->3 computation."""

    history: torch.Tensor
    state: ResolutionNativeReasoningState
    diagnostics: Mapping[str, object]

    @property
    def batch_size(self) -> int:
        return int(self.history.shape[0])

    def index_select(self, indices: torch.Tensor) -> "PostZ3Observation":
        return PostZ3Observation(
            history=self.history.index_select(0, indices),
            state=select_reasoning_state(self.state, indices),
            diagnostics=_select_diagnostics(self.diagnostics, indices),
        )

    def detached(self) -> "PostZ3Observation":
        indices = torch.arange(self.batch_size, device=self.history.device)
        selected = self.index_select(indices)
        selected.history = selected.history.detach()
        selected.state.evidence.tokens = selected.state.evidence.tokens.detach()
        selected.state.evidence.history_data = (
            selected.state.evidence.history_data.detach()
        )
        selected.state.evidence.history_flow = (
            selected.state.evidence.history_flow.detach()
        )
        if selected.state.latest_forecast is not None:
            selected.state.latest_forecast = selected.state.latest_forecast.detach()
        selected.state.forecasts = tuple(value.detach() for value in selected.state.forecasts)
        selected.diagnostics = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in selected.diagnostics.items()
        }
        return selected


class FrozenPostZ3Environment:
    """Execute the common prefix once and only the selected real continuation."""

    def __init__(self, forecaster: nn.Module) -> None:
        self.forecaster = forecaster
        self.freeze_forecaster()

    def freeze_forecaster(self) -> None:
        self.forecaster.eval()
        for parameter in self.forecaster.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def begin(self, history: torch.Tensor) -> PostZ3Observation:
        self.forecaster.eval()
        self.forecaster.evidence_encoder.reset_encode_count()
        self.forecaster.reasoner.reset_diagnostics()
        root = self.forecaster.begin_reasoning(history)
        state, diagnostics = self.forecaster.reason_step(history, root, 3)
        if state.current_resolution != 3 or len(state.forecasts) != 1:
            raise RuntimeError("post-Z3 environment did not reach exactly Z3")
        if self.forecaster.evidence_encoder.encode_count != 1:
            raise RuntimeError("history evidence must be encoded once")
        return PostZ3Observation(history, state, diagnostics).detached()

    @torch.inference_mode()
    def continue_actions(
        self,
        observation: PostZ3Observation,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Execute each sampled action and return ``[B,G,12,N,1]`` forecasts."""
        if actions.ndim == 1:
            actions = actions[:, None]
        if actions.ndim != 2 or actions.shape[0] != observation.batch_size:
            raise ValueError("actions must have shape [B] or [B,G]")
        if not bool(((actions == SHORT) | (actions == LONG)).all()):
            raise ValueError("post-Z3 actions must be SHORT=0 or LONG=1")

        batch, group = actions.shape
        sample_indices = torch.arange(batch, device=actions.device).repeat_interleave(group)
        flat_actions = actions.reshape(-1)
        flat_history = observation.history.index_select(0, sample_indices)
        flat_state = select_reasoning_state(observation.state, sample_indices)
        output = flat_history.new_empty(
            (batch * group, 12, flat_history.shape[2], 1)
        )
        for action in (SHORT, LONG):
            chosen = torch.nonzero(flat_actions == action, as_tuple=False).flatten()
            if chosen.numel() == 0:
                continue
            history = flat_history.index_select(0, chosen)
            state = select_reasoning_state(flat_state, chosen)
            if action == LONG:
                state, _ = self.forecaster.reason_step(history, state, 6)
            state, _ = self.forecaster.reason_step(history, state, 12)
            output.index_copy_(0, chosen, state.latest_forecast)
        return output.reshape(batch, group, 12, flat_history.shape[2], 1)

    @torch.inference_mode()
    def forced_pair(self, observation: PostZ3Observation) -> torch.Tensor:
        actions = torch.tensor(
            [SHORT, LONG], device=observation.history.device, dtype=torch.long
        ).view(1, 2).expand(observation.batch_size, -1)
        return self.continue_actions(observation, actions)


def _temporal_node_features(history: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
    history = history[..., 0]
    z3 = z3[..., 0]
    h_diff = history[:, 1:] - history[:, :-1]
    z_diff = z3[:, 1:] - z3[:, :-1]
    h_last = history[:, -1]
    z_mean = z3.mean(1)
    return torch.stack(
        (
            history.mean(1),
            history.std(1, unbiased=False),
            h_last,
            history[:, -1] - history[:, -2],
            h_diff.abs().mean(1),
            z_mean,
            z3.std(1, unbiased=False),
            z3[:, 0],
            z3[:, -1],
            z_diff.mean(1),
            z_mean - h_last,
            (z3 - h_last[:, None]).abs().mean(1),
        ),
        dim=-1,
    )


def _evidence_node_features(tokens: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            tokens.mean(dim=(1, 3)),
            tokens.std(dim=(1, 3), unbiased=False),
            tokens.square().mean(dim=(1, 3)).sqrt(),
        ),
        dim=-1,
    )


def _global_stats(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(1)
    return torch.stack(
        (
            flat.mean(1),
            flat.std(1, unbiased=False),
            flat.abs().mean(1),
            flat.square().mean(1).sqrt(),
        ),
        dim=-1,
    )


class PostZ3BudgetRouter(nn.Module):
    """Structured X/Z3 actor with budget-matched, zero-residual initialization."""

    def __init__(
        self,
        node_size: int,
        *,
        node_hidden: int = 32,
        node_embedding_dim: int = 8,
        hidden_dim: int = 96,
        budget_epsilon: float = 0.02,
    ) -> None:
        super().__init__()
        self.node_size = int(node_size)
        self.budget_epsilon = float(budget_epsilon)
        self.node_embedding = nn.Embedding(self.node_size, node_embedding_dim)
        self.node_encoder = nn.Sequential(
            nn.LayerNorm(15 + node_embedding_dim),
            nn.Linear(15 + node_embedding_dim, node_hidden),
            nn.SiLU(),
            nn.Linear(node_hidden, node_hidden),
            nn.SiLU(),
        )
        self.attention_score = nn.Linear(node_hidden, 1)
        diagnostic_dim = 4 * 3 + 3 + 1
        pooled_dim = node_hidden * 4
        self.residual_actor = nn.Sequential(
            nn.LayerNorm(pooled_dim + diagnostic_dim + 3),
            nn.Linear(pooled_dim + diagnostic_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_actor[-1].weight)
        nn.init.zeros_(self.residual_actor[-1].bias)

    def _diagnostics(self, observation: PostZ3Observation) -> torch.Tensor:
        parts = []
        for key in (
            "raw_correction",
            "low_frequency_correction",
            "detail_correction",
        ):
            value = observation.diagnostics.get(key)
            if not torch.is_tensor(value):
                raise KeyError(f"post-Z3 diagnostics are missing {key}")
            parts.append(_global_stats(value))
        branch = observation.diagnostics.get("branch_scale")
        low_gain = observation.diagnostics.get("low_frequency_gain")
        if not torch.is_tensor(branch) or not torch.is_tensor(low_gain):
            raise KeyError("post-Z3 transition conditioning diagnostics are missing")
        parts.extend((branch.reshape(branch.shape[0], -1)[:, :3], low_gain.reshape(-1, 1)))
        return torch.cat(parts, dim=-1)

    def forward(
        self, observation: PostZ3Observation, budget: torch.Tensor
    ) -> torch.Tensor:
        z3 = observation.state.latest_forecast
        if z3 is None or z3.shape[1] != 3:
            raise ValueError("router requires an actually reached Z3 forecast")
        if observation.history.shape[2] != self.node_size:
            raise ValueError("router node count does not match observation")
        node_features = torch.cat(
            (
                _temporal_node_features(observation.history, z3),
                _evidence_node_features(observation.state.evidence.tokens),
            ),
            dim=-1,
        )
        node_ids = torch.arange(self.node_size, device=node_features.device)
        embeddings = self.node_embedding(node_ids)[None].expand(node_features.shape[0], -1, -1)
        hidden = self.node_encoder(torch.cat((node_features, embeddings), dim=-1))
        weights = torch.softmax(self.attention_score(hidden).squeeze(-1), dim=-1)
        pooled = torch.cat(
            (
                (weights[..., None] * hidden).sum(1),
                hidden.mean(1),
                hidden.std(1, unbiased=False),
                hidden.amax(1),
            ),
            dim=-1,
        )
        budget = budget.reshape(-1).to(pooled)
        if budget.shape[0] != pooled.shape[0]:
            raise ValueError("one budget is required per sample")
        clipped = budget.clamp(self.budget_epsilon, 1.0 - self.budget_epsilon)
        base_logit = torch.logit(clipped)
        budget_features = torch.stack((budget, budget.square(), base_logit), dim=-1)
        residual = self.residual_actor(
            torch.cat((pooled, self._diagnostics(observation), budget_features), dim=-1)
        ).squeeze(-1)
        return base_logit + residual

    def probabilities(
        self, observation: PostZ3Observation, budget: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(self(observation, budget))


def bernoulli_log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    logits = logits[:, None].expand_as(actions)
    return -nn.functional.binary_cross_entropy_with_logits(
        logits, actions.to(logits.dtype), reduction="none"
    )


def bernoulli_entropy(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    return -(
        probability * torch.log(probability.clamp_min(1e-8))
        + (1.0 - probability) * torch.log((1.0 - probability).clamp_min(1e-8))
    )


def bernoulli_kl(old_probability: torch.Tensor, new_logits: torch.Tensor) -> torch.Tensor:
    old = old_probability.clamp(1e-8, 1.0 - 1e-8)
    new = torch.sigmoid(new_logits).clamp(1e-8, 1.0 - 1e-8)
    return old * (old.log() - new.log()) + (1.0 - old) * (
        (1.0 - old).log() - (1.0 - new).log()
    )


def leave_one_out_advantages(returns: torch.Tensor) -> torch.Tensor:
    """RLOO centering with no per-group standard-deviation normalization."""
    if returns.ndim != 2 or returns.shape[1] < 2:
        raise ValueError("returns must have shape [B,G] with G >= 2")
    others = (returns.sum(dim=1, keepdim=True) - returns) / (returns.shape[1] - 1)
    return returns - others


def clipped_trajectory_grpo_loss(
    current_logits: torch.Tensor,
    actions: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    old_policy_probability: torch.Tensor,
    *,
    clip_ratio: float,
    entropy_coefficient: float,
    kl_coefficient: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    current_log_prob = bernoulli_log_prob(current_logits, actions)
    ratio = torch.exp(current_log_prob - behavior_log_probs)
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    entropy = bernoulli_entropy(current_logits).mean()
    kl = bernoulli_kl(old_policy_probability, current_logits).mean()
    loss = -surrogate.mean() - entropy_coefficient * entropy + kl_coefficient * kl
    return loss, {
        "surrogate": surrogate.mean(),
        "entropy": entropy,
        "kl": kl,
        "ratio_mean": ratio.mean(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_ratio).float().mean(),
    }


class BudgetDualPanel:
    """Projected dual ascent for expected extra-compute constraints."""

    def __init__(
        self,
        budgets: torch.Tensor,
        *,
        learning_rate: float,
        maximum: float = 100.0,
    ) -> None:
        self.budgets = budgets.detach().float().clone()
        self.lambdas = torch.zeros_like(self.budgets)
        self.learning_rate = float(learning_rate)
        self.maximum = float(maximum)

    def values(self, budget_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        return self.lambdas.to(device).index_select(0, budget_indices)

    @torch.no_grad()
    def update(
        self,
        budget_indices: torch.Tensor,
        expected_long_probability: torch.Tensor,
    ) -> None:
        indices = budget_indices.detach().cpu()
        values = expected_long_probability.detach().float().cpu()
        for index in indices.unique(sorted=True):
            mask = indices == index
            violation = values[mask].mean() - self.budgets[index]
            self.lambdas[index].add_(self.learning_rate * violation)
        self.lambdas.clamp_(0.0, self.maximum)

    def state_dict(self) -> dict[str, object]:
        return {
            "budgets": self.budgets.clone(),
            "lambdas": self.lambdas.clone(),
            "learning_rate": self.learning_rate,
            "maximum": self.maximum,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        budgets = torch.as_tensor(state["budgets"]).detach().float().cpu()
        if not torch.equal(budgets, self.budgets):
            raise ValueError("dual checkpoint budget panel does not match")
        self.lambdas.copy_(torch.as_tensor(state["lambdas"]).detach().float().cpu())
