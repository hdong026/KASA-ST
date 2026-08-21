"""Sequential constrained GRPO for the complete ResolutionNative route DAG.

The forecaster is an immutable environment.  A rollout starts with one real
history encoding, samples a legal next resolution at each reached prefix, and
executes that edge with the frozen reasoner.  The policy sees only summaries of
the history and already-reached forecast state; targets are used only to form
the terminal reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .f2f_cot_resolution_native_v1_route_complete import (
    ROUTES,
    LEGAL_EDGES,
    F2FCoTResolutionNativeV1RouteCompleteNet,
)
from .post_z3_constrained_grpo import (
    _select_diagnostics,
    select_reasoning_state,
)


ACTION_RESOLUTIONS = (2, 3, 4, 6, 12)
ACTION_INDEX = {value: i for i, value in enumerate(ACTION_RESOLUTIONS)}
LEGAL_NEXT = {
    (): (2, 3, 4, 6, 12),
    (2,): (4, 6, 12),
    (2, 4): (12,),
    (2, 6): (12,),
    (3,): (6, 12),
    (3, 6): (12,),
    (4,): (12,),
    (6,): (12,),
}
PREFIXES = tuple(LEGAL_NEXT)
PREFIX_INDEX = {prefix: i for i, prefix in enumerate(PREFIXES)}
ROUTE_INDEX = {route: i for i, route in enumerate(ROUTES)}


def _stats(value: torch.Tensor, width: int = 6) -> torch.Tensor:
    flat = value.reshape(value.shape[0], -1)
    values = [
        flat.mean(1),
        flat.std(1, unbiased=False),
        flat.abs().mean(1),
        flat.square().mean(1).sqrt(),
        flat.amax(1),
        flat.amin(1),
    ]
    return torch.stack(values[:width], dim=1)


def _diag_stats(value: object, batch: int, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return _stats(value)
    return reference.new_zeros((batch, 6))


def state_features(
    history: torch.Tensor,
    state: Any,
    diagnostics: Mapping[str, object] | None,
    prefix: Sequence[int],
    budget: torch.Tensor,
    consumed_cost: torch.Tensor,
    ablate_current_forecast: bool = False,
) -> torch.Tensor:
    """Build fixed-width inference-safe features from an actual reached state."""
    batch = history.shape[0]
    hist = history[..., 0]
    parts = [_stats(hist), _stats(hist[:, 1:] - hist[:, :-1])]
    evidence = getattr(getattr(state, "evidence", None), "tokens", None)
    parts.append(_stats(evidence) if torch.is_tensor(evidence) else history.new_zeros((batch, 6)))
    resolution = int(getattr(state, "current_resolution", 0))
    forecast = None if ablate_current_forecast else getattr(state, "latest_forecast", None)
    if forecast is None:
        parts.extend((history.new_zeros((batch, 6)), history.new_zeros((batch, 6))))
        resolution = 0
    else:
        forecast = forecast[..., 0]
        projected = forecast.repeat_interleave(12 // resolution, dim=1)
        persistence = hist[:, -1:].expand_as(projected)
        parts.extend((_stats(forecast), _stats(projected - persistence)))
    diagnostics = diagnostics or {}
    for key in ("raw_correction", "low_frequency_correction", "detail_correction"):
        parts.append(_diag_stats(diagnostics.get(key), batch, history))
    branch = diagnostics.get("branch_scale")
    if not torch.is_tensor(branch):
        branch = history.new_zeros((batch, 3))
    branch = branch.reshape(batch, -1)
    branch = torch.nn.functional.pad(branch, (0, max(0, 3 - branch.shape[1])))[:, :3]
    low = diagnostics.get("low_frequency_gain")
    if not torch.is_tensor(low):
        low = history.new_zeros((batch, 1))
    low = low.reshape(batch, -1)[:, :1]
    path_code = history.new_zeros((batch, len(ACTION_RESOLUTIONS)))
    for value in prefix:
        path_code[:, ACTION_INDEX[int(value)]] = 1.0
    explicit = torch.cat(
        (
            path_code,
            history.new_full((batch, 1), float(resolution) / 12.0),
            history.new_full((batch, 1), float(len(prefix)) / 3.0),
            budget.reshape(batch, 1).to(history),
            consumed_cost.reshape(batch, 1).to(history),
        ),
        dim=1,
    )
    return torch.cat((*parts, branch, low, explicit), dim=1)


def rich_state_features(
    history: torch.Tensor,
    state: Any,
    diagnostics: Mapping[str, object] | None,
    prefix: Sequence[int],
    budget: torch.Tensor,
    consumed_cost: torch.Tensor,
    *,
    ablate_current_forecast: bool = False,
    ablate_current_hidden: bool = False,
) -> torch.Tensor:
    """Node-preserving state representation modeled on the successful Z3 actor.

    The previous full-DAG representation flattened all nodes before the actor
    saw them.  Here every node retains temporal, evidence, forecast, and
    transition-correction channels; the router performs learned attention
    pooling with node embeddings.  No target-derived quantity is included.
    """
    batch, _, node_size, _ = history.shape
    hist = history[..., 0]
    diff = hist[:, 1:] - hist[:, :-1]
    node_parts = [
        hist.mean(1),
        hist.std(1, unbiased=False),
        hist[:, -1],
        diff.mean(1),
        diff.abs().mean(1),
        hist.square().mean(1).sqrt(),
    ]
    flow = getattr(getattr(state, "evidence", None), "history_flow", None)
    if torch.is_tensor(flow):
        node_parts.extend((flow.mean(1), flow.std(1, unbiased=False), flow[:, -1]))
    else:
        node_parts.extend((hist.new_zeros((batch, node_size)),) * 3)

    evidence = getattr(getattr(state, "evidence", None), "tokens", None)
    if torch.is_tensor(evidence):
        node_parts.extend(
            (
                evidence.mean(dim=(1, 3)),
                evidence.std(dim=(1, 3), unbiased=False),
                evidence.square().mean(dim=(1, 3)).sqrt(),
            )
        )
    else:
        node_parts.extend((hist.new_zeros((batch, node_size)),) * 3)

    resolution = int(getattr(state, "current_resolution", 0))
    forecast = None if ablate_current_forecast else getattr(state, "latest_forecast", None)
    if torch.is_tensor(forecast):
        forecast = forecast[..., 0]
        projected = forecast.repeat_interleave(12 // resolution, dim=1)
        gap = projected - hist[:, -1:].expand_as(projected)
        node_parts.extend(
            (
                forecast.mean(1),
                forecast.std(1, unbiased=False),
                forecast[:, -1],
                forecast[:, 0],
                gap.mean(1),
                gap.std(1, unbiased=False),
                gap.abs().mean(1),
                gap[:, -1],
            )
        )
    else:
        node_parts.extend((hist.new_zeros((batch, node_size)),) * 8)

    diagnostics = diagnostics or {}
    for key in ("raw_correction", "low_frequency_correction", "detail_correction"):
        value = diagnostics.get(key)
        if torch.is_tensor(value):
            value = value[..., 0]
            node_parts.extend((value.mean(1), value.std(1, unbiased=False), value.abs().mean(1)))
        else:
            node_parts.extend((hist.new_zeros((batch, node_size)),) * 3)

    hidden = None if ablate_current_hidden else diagnostics.get("active_hidden")
    if torch.is_tensor(hidden) and hidden.ndim == 4:
        # Preserve the reasoner's node-wise latent geometry while reducing
        # only its temporal/channel axes.  This is the exact computation
        # already performed by the frozen reasoner, exposed read-only.
        hidden_flat = hidden.reshape(batch, hidden.shape[1], node_size, -1)
        node_parts.extend(
            (
                hidden_flat.mean(dim=(1, 3)),
                hidden_flat.std(dim=(1, 3), unbiased=False),
                hidden_flat.abs().mean(dim=(1, 3)),
                hidden_flat.square().mean(dim=(1, 3)).sqrt(),
                hidden_flat.amax(dim=(1, 3)),
                hidden_flat.amin(dim=(1, 3)),
            )
        )
    else:
        node_parts.extend((hist.new_zeros((batch, node_size)),) * 6)

    branch = diagnostics.get("branch_scale")
    if not torch.is_tensor(branch):
        branch = hist.new_zeros((batch, 3))
    branch = branch.reshape(batch, -1)
    branch = torch.nn.functional.pad(branch, (0, max(0, 3 - branch.shape[1])))[:, :3]
    branch = branch[:, None, :].expand(-1, node_size, -1)
    low = diagnostics.get("low_frequency_gain")
    if not torch.is_tensor(low):
        low = hist.new_zeros((batch, 1))
    low = low.reshape(batch, -1)[:, :1][:, None, :].expand(-1, node_size, -1)

    path_code = hist.new_zeros((batch, len(ACTION_RESOLUTIONS)))
    for value in prefix:
        path_code[:, ACTION_INDEX[int(value)]] = 1.0
    explicit = torch.cat(
        (
            path_code,
            hist.new_full((batch, 1), float(resolution) / 12.0),
            hist.new_full((batch, 1), float(len(prefix)) / 3.0),
            budget.reshape(batch, 1).to(hist),
            consumed_cost.reshape(batch, 1).to(hist),
        ),
        dim=1,
    )[:, None, :].expand(-1, node_size, -1)
    return torch.cat((*[value[..., None] for value in ()], torch.stack(node_parts, dim=-1), branch, low, explicit), dim=-1)


def legal_budget_mask(
    prefix: Sequence[int], budgets: torch.Tensor, route_costs: torch.Tensor
) -> torch.Tensor:
    """Mask legal actions whose cheapest completion fits the budget."""
    key = tuple(int(x) for x in prefix)
    if key not in LEGAL_NEXT:
        raise ValueError(f"unknown prefix {key}")
    costs = route_costs.to(budgets)
    rows = []
    for budget in budgets.reshape(-1):
        allowed = []
        for action in ACTION_RESOLUTIONS:
            if action not in LEGAL_NEXT[key]:
                allowed.append(False)
                continue
            candidates = [
                costs[i]
                for i, route in enumerate(ROUTES)
                if route[: len(key)] == key and len(route) > len(key) and route[len(key)] == action
            ]
            allowed.append(bool(candidates) and float(torch.stack(candidates).min()) <= float(budget) + 1e-6)
        # Cost zero is the direct (12,) route, so this should never be empty.
        if not any(allowed):
            allowed[ACTION_INDEX[12]] = True
        rows.append(torch.tensor(allowed, dtype=torch.bool, device=budgets.device))
    return torch.stack(rows, dim=0)


def prefix_cost_lower_bound(prefix: Sequence[int], route_costs: torch.Tensor) -> float:
    key = tuple(int(x) for x in prefix)
    candidates = [route_costs[i] for i, route in enumerate(ROUTES) if route[: len(key)] == key]
    if not candidates:
        raise ValueError(f"prefix {key} has no terminal completion")
    return float(torch.stack(candidates).min().item())


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape != mask.shape or not bool(mask.any(dim=-1).all()):
        raise ValueError("invalid logits/action mask")
    masked = logits.masked_fill(~mask, float("-inf"))
    logp = torch.log_softmax(masked, dim=-1)
    return logp, logp.exp()


class FullDAGBudgetRouter(nn.Module):
    """Small prefix-conditioned categorical actor with zero-residual budget bias."""

    def __init__(self, state_dim: int = 61, hidden_dim: int = 96, prefix_dim: int = 12, action_bias: bool = False) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.state_norm = nn.LayerNorm(self.state_dim)
        self.prefix_embedding = nn.Embedding(len(PREFIXES), prefix_dim)
        self.action_embedding = nn.Embedding(len(ACTION_RESOLUTIONS) + 1, 8)
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim + prefix_dim + 8 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            # No free global action bias: route preference must be expressed
            # through the reached sample state/prefix and cannot become a
            # constant path merely because a single action has a favorable
            # TRAIN mean.
            nn.Linear(hidden_dim, len(ACTION_RESOLUTIONS), bias=bool(action_bias)),
        )
        nn.init.zeros_(self.actor[-1].weight)
        if self.actor[-1].bias is not None:
            nn.init.zeros_(self.actor[-1].bias)

    def logits_and_probs(
        self,
        features: torch.Tensor,
        prefix: Sequence[int],
        budget: torch.Tensor,
        route_costs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = tuple(int(x) for x in prefix)
        if key not in PREFIX_INDEX:
            raise ValueError(f"unknown prefix {key}")
        batch = features.shape[0]
        budget = budget.reshape(batch, 1).to(features)
        consumed = features.new_zeros((batch, 1))
        # consumed cost is already included in the final two feature slots;
        # budget and a stable slack feature make the actor budget-aware.
        slack = budget - features[:, -1:].detach()
        current = features.new_full((batch, 1), (key[-1] if key else 0) / 12.0)
        explicit = torch.cat((budget, slack, current), dim=1)
        prefix_id = torch.full((batch,), PREFIX_INDEX[key], dtype=torch.long, device=features.device)
        action_id = torch.full((batch,), 0 if not key else ACTION_INDEX[key[-1]] + 1, dtype=torch.long, device=features.device)
        encoded = torch.cat((self.state_norm(features), self.prefix_embedding(prefix_id), self.action_embedding(action_id), explicit), dim=1)
        logits = self.actor(encoded)
        mask = legal_budget_mask(key, budget.reshape(-1), route_costs)
        logp, probs = masked_log_probs(logits, mask)
        return logits, logp, probs, mask

    def log_probs_from_features(self, features, prefix, budget, route_costs):
        return self.logits_and_probs(features, prefix, budget, route_costs)[1:]


class RichFullDAGBudgetRouter(nn.Module):
    """Node-aware sequential actor retaining the post-Z3 representation idea."""

    def __init__(
        self,
        node_size: int,
        node_feature_dim: int = 48,
        node_hidden: int = 48,
        prefix_dim: int = 12,
        action_bias: bool = False,
    ) -> None:
        super().__init__()
        self.node_size = int(node_size)
        self.node_feature_dim = int(node_feature_dim)
        self.node_embedding = nn.Embedding(self.node_size, 8)
        self.node_encoder = nn.Sequential(
            nn.LayerNorm(self.node_feature_dim + 8),
            nn.Linear(self.node_feature_dim + 8, node_hidden),
            nn.SiLU(),
            nn.Linear(node_hidden, node_hidden),
            nn.SiLU(),
        )
        self.attention_score = nn.Linear(node_hidden, 1)
        self.prefix_embedding = nn.Embedding(len(PREFIXES), prefix_dim)
        self.action_embedding = nn.Embedding(len(ACTION_RESOLUTIONS) + 1, 8)
        pooled_dim = node_hidden * 4
        self.actor = nn.Sequential(
            nn.LayerNorm(pooled_dim + prefix_dim + 8 + 3),
            nn.Linear(pooled_dim + prefix_dim + 8 + 3, 96),
            nn.SiLU(),
            nn.Linear(96, len(ACTION_RESOLUTIONS), bias=bool(action_bias)),
        )
        nn.init.zeros_(self.actor[-1].weight)
        if self.actor[-1].bias is not None:
            nn.init.zeros_(self.actor[-1].bias)

    def build_features(self, history, state, diagnostics, prefix, budget, consumed_cost, *, ablate_current_forecast=False, ablate_current_hidden=False):
        return rich_state_features(
            history,
            state,
            diagnostics,
            prefix,
            budget,
            consumed_cost,
            ablate_current_forecast=ablate_current_forecast,
            ablate_current_hidden=ablate_current_hidden,
        )

    def logits_and_probs(self, features, prefix, budget, route_costs):
        if features.ndim != 3 or features.shape[1] != self.node_size or features.shape[2] != self.node_feature_dim:
            raise ValueError(
                f"rich features must be [B,{self.node_size},{self.node_feature_dim}], got {tuple(features.shape)}"
            )
        batch = features.shape[0]
        node_ids = torch.arange(self.node_size, device=features.device)
        node_ids = self.node_embedding(node_ids)[None].expand(batch, -1, -1)
        hidden = self.node_encoder(torch.cat((features, node_ids), dim=-1))
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
        budget = budget.reshape(batch, 1).to(pooled)
        consumed = features[:, :, -1:].mean(1)
        slack = budget - consumed
        current = features.new_full((batch, 1), (prefix[-1] if prefix else 0) / 12.0)
        prefix_id = torch.full((batch,), PREFIX_INDEX[tuple(prefix)], dtype=torch.long, device=features.device)
        action_id = torch.full((batch,), 0 if not prefix else ACTION_INDEX[prefix[-1]] + 1, dtype=torch.long, device=features.device)
        explicit = torch.cat((budget, slack, current), dim=1)
        encoded = torch.cat((pooled, self.prefix_embedding(prefix_id), self.action_embedding(action_id), explicit), dim=1)
        logits = self.actor(encoded)
        mask = legal_budget_mask(tuple(prefix), budget.reshape(-1), route_costs)
        logp, probs = masked_log_probs(logits, mask)
        return logits, logp, probs, mask

    def log_probs_from_features(self, features, prefix, budget, route_costs):
        return self.logits_and_probs(features, prefix, budget, route_costs)[1:]


@dataclass
class RouteObservation:
    history: torch.Tensor
    state: Any
    diagnostics: Mapping[str, object]


@dataclass
class TrajectoryRecords:
    route_ids: torch.Tensor
    route_costs: torch.Tensor
    predictions: torch.Tensor
    returns: torch.Tensor | None
    records: tuple[dict[str, Any], ...]


class FrozenCompleteDAGEnvironment:
    """Actual shared-prefix execution for sampled complete-DAG trajectories."""

    def __init__(self, forecaster: F2FCoTResolutionNativeV1RouteCompleteNet):
        self.forecaster = forecaster
        self.forecaster.eval()
        for parameter in self.forecaster.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def begin(self, history: torch.Tensor) -> RouteObservation:
        self.forecaster.evidence_encoder.reset_encode_count()
        self.forecaster.reasoner.reset_diagnostics()
        state = self.forecaster.begin_reasoning(history)
        if self.forecaster.evidence_encoder.encode_count != 1:
            raise RuntimeError("full-DAG environment encoded history more than once")
        return RouteObservation(history, state, {})

    @torch.inference_mode()
    def sample(
        self,
        observation: RouteObservation,
        budgets: torch.Tensor,
        route_costs: torch.Tensor,
        policy: FullDAGBudgetRouter,
        group_size: int,
        exploration_mix: float,
        generator: torch.Generator,
        deterministic: bool = False,
        ablate_current_forecast: bool = False,
    ) -> tuple[TrajectoryRecords, dict[str, float]]:
        batch = observation.history.shape[0]
        device = observation.history.device
        total = batch * int(group_size)
        base = torch.arange(batch, device=device).repeat_interleave(group_size)
        history = observation.history.index_select(0, base)
        root_state = select_reasoning_state(observation.state, base)
        root_obs = RouteObservation(history, root_state, {})
        budgets_flat = budgets.reshape(-1).repeat_interleave(group_size).to(device)
        active: dict[tuple[int, ...], tuple[torch.Tensor, RouteObservation]] = {(): (torch.arange(total, device=device), root_obs)}
        route_ids = torch.full((total,), -1, dtype=torch.long, device=device)
        predictions = history.new_empty((total, 12, history.shape[2], 1))
        records: list[dict[str, Any]] = []
        action_count = 0
        while active:
            next_active: dict[tuple[int, ...], list[tuple[torch.Tensor, RouteObservation]]] = {}
            for prefix, (ids, current_obs) in active.items():
                consumed = history.new_full(
                    (ids.numel(),), prefix_cost_lower_bound(prefix, route_costs)
                )
                feature_builder = getattr(policy, "build_features", state_features)
                features = feature_builder(
                    current_obs.history,
                    current_obs.state,
                    current_obs.diagnostics,
                    prefix,
                    budgets_flat.index_select(0, ids),
                    consumed,
                    ablate_current_forecast=ablate_current_forecast,
                )
                _, logp, probs, mask = policy.logits_and_probs(features, prefix, budgets_flat.index_select(0, ids), route_costs)
                uniform = mask.to(probs.dtype) / mask.sum(dim=1, keepdim=True).to(probs.dtype)
                behavior = (1.0 - float(exploration_mix)) * probs + float(exploration_mix) * uniform
                actions = (
                    behavior.argmax(dim=1)
                    if deterministic
                    else torch.multinomial(behavior, 1, generator=generator).squeeze(1)
                )
                selected_logp = behavior.gather(1, actions[:, None]).squeeze(1).log()
                # ``sample`` is inference-only because it executes the frozen
                # reasoner.  Clone tensors before handing them to the actor's
                # autograd update; inference tensors cannot be saved by a
                # backward graph.
                with torch.inference_mode(False):
                    records.append({"features": features.detach().clone(), "prefix": prefix, "budget": budgets_flat.index_select(0, ids).detach().clone(), "actions": actions.detach().clone(), "behavior_logp": selected_logp.detach().clone(), "old_probs": probs.detach().clone(), "mask": mask.detach().clone(), "ids": ids.detach().clone()})
                action_count += int(ids.numel())
                for action in torch.unique(actions, sorted=True).tolist():
                    chosen = torch.nonzero(actions == int(action), as_tuple=False).flatten()
                    chosen_ids = ids.index_select(0, chosen)
                    chosen_history = current_obs.history.index_select(0, chosen)
                    chosen_state = select_reasoning_state(current_obs.state, chosen)
                    chosen_diag = _select_diagnostics(current_obs.diagnostics, chosen)
                    nxt = ACTION_RESOLUTIONS[int(action)]
                    new_state, diagnostics = self.forecaster.reason_step(chosen_history, chosen_state, nxt)
                    if int(action) == ACTION_INDEX[12]:
                        route = (*prefix, 12)
                        route_ids.index_fill_(0, chosen_ids, ROUTE_INDEX[route])
                        predictions.index_copy_(0, chosen_ids, new_state.latest_forecast)
                        continue
                    new_prefix = (*prefix, nxt)
                    new_obs = RouteObservation(chosen_history, new_state, diagnostics)
                    next_active.setdefault(new_prefix, []).append((chosen_ids, new_obs))
            active = {}
            for prefix, chunks in next_active.items():
                ids = torch.cat([chunk[0] for chunk in chunks], dim=0)
                obs = RouteObservation(torch.cat([chunk[1].history for chunk in chunks], dim=0), select_reasoning_state(chunks[0][1].state, torch.arange(chunks[0][1].history.shape[0], device=device)), {})
                # Concatenating state objects needs explicit tensor assembly.
                states = [chunk[1].state for chunk in chunks]
                obs.state = select_reasoning_state(states[0], torch.arange(states[0].latest_forecast.shape[0], device=device))
                if len(states) > 1:
                    obs.state.evidence.tokens = torch.cat([s.evidence.tokens for s in states], dim=0)
                    obs.state.evidence.history_data = torch.cat([s.evidence.history_data for s in states], dim=0)
                    obs.state.evidence.history_flow = torch.cat([s.evidence.history_flow for s in states], dim=0)
                    obs.state.latest_forecast = torch.cat([s.latest_forecast for s in states], dim=0)
                    obs.state.forecasts = tuple(torch.cat([s.forecasts[i] for s in states], dim=0) for i in range(len(states[0].forecasts)))
                diagnostics = {}
                keys = set().union(*(chunk[1].diagnostics.keys() for chunk in chunks))
                for key in keys:
                    values = [chunk[1].diagnostics.get(key) for chunk in chunks]
                    diagnostics[key] = torch.cat(values, dim=0) if all(torch.is_tensor(v) and v.ndim > 0 for v in values) else values[0]
                obs.diagnostics = diagnostics
                active[prefix] = (ids, obs)
        if bool((route_ids < 0).any()):
            raise RuntimeError("sampled trajectory failed to reach terminal 12")
        costs = route_costs.to(device).index_select(0, route_ids)
        return TrajectoryRecords(route_ids, costs, predictions, None, tuple(records)), {"action_count": float(action_count), "history_encode_count": float(self.forecaster.evidence_encoder.encode_count)}


def leave_one_out_advantages(returns: torch.Tensor) -> torch.Tensor:
    if returns.ndim != 2 or returns.shape[1] < 2:
        raise ValueError("returns must be [B,G], G>=2")
    return returns - (returns.sum(1, keepdim=True) - returns) / (returns.shape[1] - 1)


def categorical_trajectory_grpo_loss(
    current_logp: torch.Tensor,
    behavior_logp: torch.Tensor,
    advantages: torch.Tensor,
    old_probs: torch.Tensor,
    current_probs: torch.Tensor,
    masks: torch.Tensor,
    *,
    clip_ratio: float,
    entropy_coefficient: float,
    kl_coefficient: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(current_logp - behavior_logp)
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    legal = masks.to(current_probs.dtype)
    safe_old = old_probs.clamp_min(1e-8)
    safe_cur = current_probs.clamp_min(1e-8)
    kl = (legal * safe_old * (safe_old.log() - safe_cur.log())).sum(1).mean()
    entropy = -(legal * safe_cur * safe_cur.log()).sum(1).mean()
    loss = -surrogate.mean() - entropy_coefficient * entropy + kl_coefficient * kl
    return loss, {"ratio_mean": ratio.mean(), "clip_fraction": ((ratio - 1.0).abs() > clip_ratio).float().mean(), "entropy": entropy, "kl": kl, "surrogate": surrogate.mean()}
