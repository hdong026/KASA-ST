"""Train a real on-policy sequential resolution controller.

Every sampled action in this file is followed by the corresponding frozen
forecast transition.  No complete-path prediction cache, counterfactual path
loss cache, imitation target, or offline cost-to-go target is used for policy
training.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from .online_rl_policy import (
    OnlineResolutionActorCritic,
)
from .online_xz_relation_policy import OnlineXZRelationActorCritic
from .progressive_selector import history_state_features
from .run_pipeline import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_F2F_CHECKPOINT,
    ROOT,
    WindowDataset,
    canonical_audit,
    load_data,
    make_loader,
    per_sample_mae,
    prepare_batch,
    route_name,
    seed_everything,
    to_physical,
)
from .run_selector import DEFAULT_BRIDGE_CHECKPOINT, build_frozen_forecaster
from .sequential_budget_policy import (
    NEXT_RESOLUTIONS,
    TRAJECTORIES,
    explicit_forecast_only_features,
)


ROUTE_TO_INDEX = {route: index for index, route in enumerate(TRAJECTORIES)}
DECISION_SOURCES = (None, 3, 4)
FORCED_NEXT = {2: 4, 6: 12}


def controller_state_features(policy, history, forecast):
    """Build only the state layout expected by the selected controller."""
    if getattr(policy, "uses_xz_relation_state", False):
        return policy.state_features(history, forecast)
    history_features = history_state_features(history)
    if forecast is None:
        return history_features
    return torch.cat(
        (history_features, explicit_forecast_only_features(forecast)), dim=1
    )


def _profile_callable(device, function, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(samples)
    return {
        "median_ms": float(statistics.median(samples)),
        "mean_ms": float(statistics.fmean(samples)),
        "p90_ms": float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]),
        "min_ms": float(ordered[0]),
        "repeats": int(repeats),
        "batch_size": 1,
    }


def _profile_paired_callables(
    device, fixed_function, adaptive_function, warmup: int, repeats: int
) -> tuple[dict, dict]:
    """Interleave both sides so clock/thermal drift cannot favor one system."""
    for _ in range(warmup):
        fixed_function()
        adaptive_function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = {"fixed": [], "adaptive": []}
    functions = {"fixed": fixed_function, "adaptive": adaptive_function}
    for repeat in range(repeats):
        order = ("fixed", "adaptive") if repeat % 2 == 0 else ("adaptive", "fixed")
        for name in order:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            functions[name]()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples[name].append((time.perf_counter() - started) * 1000.0)

    def summarize(values):
        ordered = sorted(values)
        return {
            "median_ms": float(statistics.median(values)),
            "mean_ms": float(statistics.fmean(values)),
            "p90_ms": float(
                ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]
            ),
            "min_ms": float(ordered[0]),
            "repeats": int(repeats),
            "batch_size": 1,
            "paired_interleaved": True,
        }

    return summarize(samples["fixed"]), summarize(samples["adaptive"])


@torch.no_grad()
def execute_forced_online_route(model, policy, history, route: Sequence[int]):
    """Execute an exact requested route while paying actor inference overhead."""
    prefix: tuple[int, ...] = ()
    source = None
    forecast = None
    budget = torch.full((len(history),), 1e3, device=history.device)
    for target in route:
        if source in DECISION_SOURCES:
            state = controller_state_features(policy, history, forecast)
            consumed = torch.zeros_like(budget)
            # Consume the result exactly as dynamic Python dispatch must. The
            # forced target below lets us profile a specified route, while
            # this host read keeps the real action synchronization in latency.
            int(policy.greedy(source, state)[0])
        forecast = model.execute_transition(history, source, int(target), forecast)
        prefix = prefix + (int(target),)
        source = int(target)
    return model.finalize_forecast(forecast, history)


@torch.no_grad()
def execute_prefix(model, policy, history, prefix: Sequence[int]):
    """Execute a real prefix for physically measured consumed-cost features."""
    source = None
    forecast = None
    budget = torch.full((len(history),), 1e3, device=history.device)
    for target in prefix:
        if source in DECISION_SOURCES:
            state = controller_state_features(policy, history, forecast)
            int(policy.greedy(source, state)[0])
        forecast = model.execute_transition(history, source, int(target), forecast)
        source = int(target)
    return forecast


def profile_physical_costs(model, policy, history, args):
    fixed_raw = {}
    adaptive_raw = {}
    for route in TRAJECTORIES:
        fixed_row, adaptive_row = _profile_paired_callables(
            history.device,
            lambda route=route: model.execute_trajectory(history, route),
            lambda route=route: execute_forced_online_route(
                model, policy, history, route
            ),
            args.latency_warmup,
            args.latency_repeats,
        )
        fixed_raw[route] = fixed_row
        adaptive_raw[route] = adaptive_row
    decision_prefixes = ((), (3,), (4,), (2, 4), (3, 4))
    prefix_raw = {(): {"p90_ms": 0.0}}
    for prefix in decision_prefixes[1:]:
        prefix_raw[prefix] = _profile_callable(
            history.device,
            lambda prefix=prefix: execute_prefix(model, policy, history, prefix),
            args.latency_warmup,
            args.latency_repeats,
        )
    fixed = {route: row["p90_ms"] for route, row in fixed_raw.items()}
    adaptive = {route: row["p90_ms"] for route, row in adaptive_raw.items()}
    prefix = {key: row["p90_ms"] for key, row in prefix_raw.items()}
    return fixed_raw, adaptive_raw, prefix_raw, fixed, adaptive, prefix


@torch.inference_mode()
def evaluate_xz_relation_states(model, policy, loader, device, max_batches=None):
    """Describe the learned inference-safe relation at actually reached states."""
    if not getattr(policy, "uses_xz_relation_state", False):
        return None
    prefixes = {
        "r2_start-2": (2,),
        "r3_start-3": (3,),
        "r4_start-4": (4,),
        "r4_via-2-4": (2, 4),
        "r4_via-3-4": (3, 4),
        "r6_via-3-6": (3, 6),
        "r6_via-3-4-6": (3, 4, 6),
    }
    pieces = {name: defaultdict(list) for name in prefixes}
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, _ = prepare_batch(batch, device)
        for name, prefix in prefixes.items():
            source = None
            forecast = None
            for target in prefix:
                forecast = model.execute_transition(
                    history, source, int(target), forecast
                )
                source = int(target)
            state = controller_state_features(policy, history, forecast)
            diagnostics = policy.encoder.diagnostics(source, state)
            for key, values in diagnostics.items():
                pieces[name][key].append(values.cpu())
    report = {}
    for name, metrics in pieces.items():
        resolution = int(name.split("r", 1)[1].split("_", 1)[0])
        report[name] = {
            "resolution": resolution,
            "legal_next_resolutions": list(NEXT_RESOLUTIONS[resolution]),
            "decision_is_forced": len(NEXT_RESOLUTIONS[resolution]) == 1,
            "metrics": {
                key: {
                    "mean": float(torch.cat(values).mean()),
                    "std_across_samples": float(
                        torch.cat(values).std(unbiased=False)
                    ),
                }
                for key, values in metrics.items()
            },
        }
    return report


def derive_budget_grid(adaptive_costs: Mapping[tuple[int, ...], float], count: int):
    values = torch.tensor(sorted(adaptive_costs.values()))
    if count <= 1:
        return [float(values.max())]
    # Do not make the default lowest *average* budget equal to the unique
    # cheapest route. That point is incompatible with the stochastic
    # exploration required by on-policy training. Explicit --budgets-ms are
    # honored exactly, including that boundary when the user requests it.
    first = 1.0 / max(len(values) - 1, 1)
    quantiles = torch.linspace(first, 1, count)
    return sorted(set(float(value) for value in torch.quantile(values, quantiles)))


def _terminal_group(
    model,
    history,
    forecast,
    indices,
    target_raw,
    mean,
    std,
    final_losses,
):
    with torch.no_grad():
        prediction = model.finalize_forecast(forecast, history)
        loss = per_sample_mae(
            to_physical(prediction, mean, std),
            target_raw.index_select(0, indices),
            null_val=0.0,
        )
    final_losses.index_copy_(0, indices, loss)


def rollout_online_batch(
    model,
    policy,
    history,
    target,
    mean: float,
    std: float,
    budget_ms: float | torch.Tensor,
    route_costs: Mapping[tuple[int, ...], float],
    prefix_costs: Mapping[tuple[int, ...], float],
    *,
    sample_actions: bool,
    hard_cap: bool,
    training_group_ids: torch.Tensor | None = None,
    reference_losses: torch.Tensor | None = None,
) -> dict:
    """Generate trajectories by sampling and immediately executing each edge."""
    device = history.device
    batch_size = len(history)
    if torch.is_tensor(budget_ms):
        budget_values = budget_ms.to(device=device, dtype=history.dtype).flatten()
        if len(budget_values) != batch_size:
            raise ValueError("Per-sample budget vector must match rollout batch size.")
    else:
        budget_values = torch.full(
            (batch_size,), float(budget_ms), device=device, dtype=history.dtype
        )
    target_raw = to_physical(target, mean, std)
    if reference_losses is None:
        reference_losses = torch.zeros(batch_size, device=device)
    else:
        reference_losses = reference_losses.to(device).detach().flatten()
        if len(reference_losses) != batch_size:
            raise ValueError("Reference-loss vector must match rollout batch size.")
    final_losses = torch.empty(batch_size, device=device)
    routes: list[tuple[int, ...] | None] = [None] * batch_size
    records = []
    executed_edges = Counter()
    sampled_decisions = 0
    chosen_transition_executions = 0

    # Each frontier cohort shares an actually reached prefix and forecast state.
    frontier = [((), torch.arange(batch_size, device=device), None)]
    while frontier:
        prefix, global_indices, forecast = frontier.pop()
        source = None if not prefix else int(prefix[-1])
        local_history = history.index_select(0, global_indices)

        if source in FORCED_NEXT:
            targets = torch.full(
                (len(global_indices),), FORCED_NEXT[source],
                dtype=torch.long, device=device,
            )
            decision = None
        else:
            if source not in DECISION_SOURCES:
                raise RuntimeError(f"Unexpected reached resolution {source}.")
            state = controller_state_features(
                policy, local_history, forecast
            )
            budget = budget_values.index_select(0, global_indices)
            consumed = torch.full(
                (len(global_indices),), float(prefix_costs.get(prefix, 0.0)),
                device=device,
            )
            actions = NEXT_RESOLUTIONS[source]
            action_costs = torch.tensor(
                [
                    min(
                        cost
                        for route, cost in route_costs.items()
                        if route[: len(prefix) + 1]
                        == prefix + (int(action),)
                    )
                    for action in actions
                ],
                device=device,
                dtype=history.dtype,
            )
            mask = None
            if hard_cap:
                mask = action_costs[None, :] <= budget[:, None] + 1e-6
            if sample_actions:
                decision = policy.act(
                    source,
                    state,
                    budget,
                    consumed,
                    feasible_mask=mask,
                    sample=True,
                )
            else:
                action_index = policy.greedy(source, state, feasible_mask=mask)
                decision = {"action_index": action_index}
            targets = policy.action_values(source, device).index_select(
                0, decision["action_index"]
            )
            if sample_actions:
                records.append(
                    {
                        "indices": global_indices,
                        "state": state.detach(),
                        "budget": budget.detach(),
                        "consumed": consumed.detach(),
                        "feasible_mask": None if mask is None else mask.detach(),
                        "action_index": decision["action_index"].detach(),
                        "old_log_prob": decision["log_prob"].detach(),
                        "old_value": decision["value"].detach(),
                        "source": source,
                        "prefix": prefix,
                    }
                )
            sampled_decisions += len(global_indices)

        for target_resolution in targets.unique(sorted=True).tolist():
            local_positions = (targets == int(target_resolution)).nonzero(
                as_tuple=False
            ).flatten()
            selected_indices = global_indices.index_select(0, local_positions)
            selected_history = history.index_select(0, selected_indices)
            selected_forecast = (
                None
                if forecast is None
                else forecast.index_select(0, local_positions)
            )
            # This is the hard acceptance point: every selected action invokes
            # the real frozen transition before another policy observation.
            with torch.no_grad():
                next_forecast = model.execute_transition(
                    selected_history,
                    source,
                    int(target_resolution),
                    selected_forecast,
                )
            executed_edges[(source, int(target_resolution))] += len(selected_indices)
            if decision is not None:
                chosen_transition_executions += len(selected_indices)
            next_prefix = prefix + (int(target_resolution),)
            if int(target_resolution) == 12:
                _terminal_group(
                    model,
                    selected_history,
                    next_forecast,
                    selected_indices,
                    target_raw,
                    mean,
                    std,
                    final_losses,
                )
                for index in selected_indices.tolist():
                    routes[index] = next_prefix
            else:
                frontier.append((next_prefix, selected_indices, next_forecast))

    if any(route is None for route in routes):
        raise RuntimeError("Online rollout did not terminate every sample at 12.")
    if sampled_decisions != chosen_transition_executions:
        raise RuntimeError(
            "A sampled policy action was not matched by a real transition execution."
        )
    route_indices = torch.tensor(
        [ROUTE_TO_INDEX[route] for route in routes], dtype=torch.long, device=device
    )
    cost_tensor = torch.tensor(
        [route_costs[route] for route in TRAJECTORIES], device=device
    )
    return {
        "losses": final_losses,
        "reference_baseline": reference_losses,
        "costs_ms": cost_tensor.index_select(0, route_indices),
        "route_indices": route_indices,
        "routes": routes,
        "records": records,
        "sampled_decisions": sampled_decisions,
        "chosen_transition_executions": chosen_transition_executions,
        "executed_edges": executed_edges,
        "training_group_ids": training_group_ids,
        "budget_ms": budget_values,
    }


def actor_critic_loss(
    policy,
    rollout: Mapping,
    lambda_mae_per_ms: float | torch.Tensor,
    budget_ms: float | torch.Tensor,
    critic_weight: float,
    entropy_weight: float,
    mae_reward_scale: float = 1.0,
    ppo_clip: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    if torch.is_tensor(lambda_mae_per_ms):
        multiplier = lambda_mae_per_ms.to(rollout["losses"])
    else:
        multiplier = torch.full_like(
            rollout["losses"], float(lambda_mae_per_ms)
        )
    if torch.is_tensor(budget_ms):
        budget = budget_ms.to(rollout["losses"])
    else:
        budget = torch.full_like(rollout["losses"], float(budget_ms))
    objective = (
        float(mae_reward_scale)
        * (rollout["losses"] - rollout["reference_baseline"])
        + multiplier * (rollout["costs_ms"] - budget)
    )
    actor_terms = []
    entropies = []
    values = []
    returns = []
    group_ids = rollout.get("training_group_ids")
    for record in rollout["records"]:
        terminal = objective.index_select(0, record["indices"]).detach()
        # Repeated on-policy rollouts of the same training example give an
        # especially strong action-independent control variate: another
        # independently sampled and fully executed trajectory for that same
        # example.  Leave the current trajectory out of its baseline so the
        # policy-gradient estimator remains valid.  At a later prefix there
        # can be only one replica left; those cases fall back to the reached
        # cohort mean, which is also action independent.
        if group_ids is not None:
            reached_ids = group_ids.index_select(0, record["indices"])
            _, inverse = torch.unique(reached_ids, return_inverse=True)
            group_sum = torch.zeros(
                int(inverse.max()) + 1, device=terminal.device, dtype=terminal.dtype
            )
            group_count = torch.zeros_like(group_sum)
            group_square_sum = torch.zeros_like(group_sum)
            group_sum.scatter_add_(0, inverse, terminal)
            group_count.scatter_add_(0, inverse, torch.ones_like(terminal))
            group_square_sum.scatter_add_(0, inverse, terminal.square())
            other_count = group_count.index_select(0, inverse) - 1.0
            leave_one_out = (
                group_sum.index_select(0, inverse) - terminal
            ) / other_count.clamp_min(1.0)
            # A singleton branch has no sibling return. Its critic sees this
            # exact reached state and is trained on the online canonical-delta
            # return, so it is the appropriate action-independent fallback.
            fallback = record["old_value"]
            paired_relative = terminal - leave_one_out
            group_mean = group_sum / group_count.clamp_min(1.0)
            group_variance = (
                group_square_sum / group_count.clamp_min(1.0)
                - group_mean.square()
            ).clamp_min(0.0)
            paired_scale = group_variance.sqrt().index_select(0, inverse).clamp_min(
                1e-3
            )
            # Normalize only among independently sampled continuations of this
            # same sample at this same reached forecast state. Unlike the old
            # cohort normalization, no unrelated sample enters the scale.
            relative = torch.where(
                other_count > 0,
                paired_relative / paired_scale,
                terminal - fallback,
            )
        else:
            relative = terminal - terminal.mean()
        log_prob, entropy, value = policy.evaluate_actions(
            record["source"],
            record["state"],
            record["budget"],
            record["consumed"],
            record["action_index"],
            record["feasible_mask"],
        )
        ratio = (log_prob - record["old_log_prob"]).exp()
        unclipped = ratio * relative
        clipped = ratio.clamp(1.0 - ppo_clip, 1.0 + ppo_clip) * relative
        actor_terms.append(torch.maximum(unclipped, clipped).mean())
        entropies.append(entropy)
        values.append(value)
        returns.append(terminal)
    entropy = torch.cat(entropies)
    value = torch.cat(values)
    terminal_return = torch.cat(returns)
    # We minimize forecasting loss, hence positive bad-outcome advantages must
    # reduce the sampled action's log probability.
    actor = torch.stack(actor_terms).mean()
    critic = F.smooth_l1_loss(value, terminal_return)
    loss = actor + float(critic_weight) * critic - float(entropy_weight) * entropy.mean()
    return loss, {
        "actor": float(actor.detach()),
        "critic": float(critic.detach()),
        "entropy": float(entropy.detach().mean()),
        "objective": float(objective.detach().mean()),
        "reference_mae": float(rollout["reference_baseline"].mean()),
    }


def _route_report(losses, costs, route_indices) -> dict:
    counts = torch.bincount(route_indices.cpu(), minlength=len(TRAJECTORIES))
    probabilities = counts.float() / counts.sum().clamp_min(1)
    positive = probabilities[probabilities > 0]
    return {
        "mae": float(losses.mean()),
        "mean_latency_ms": float(costs.mean()),
        "max_route_p90_ms": float(costs.max()),
        "route_counts": {
            route_name(route): int(counts[index])
            for index, route in enumerate(TRAJECTORIES)
        },
        "route_diversity": {
            "active_route_count": int((counts > 0).sum()),
            "dominant_route_fraction": float(probabilities.max()),
            "entropy_nats": float(-(positive * positive.log()).sum()),
        },
    }


@torch.no_grad()
def evaluate_policy(
    model,
    policy,
    loader,
    device,
    mean,
    std,
    budgets,
    route_costs,
    prefix_costs,
    max_batches,
    hard_cap=False,
    sample_actions=True,
):
    policy.eval()
    rows = []
    for budget in budgets:
        losses = []
        costs = []
        routes = []
        decisions = 0
        transitions = 0
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            history, target = prepare_batch(batch, device)
            rollout = rollout_online_batch(
                model,
                policy,
                history,
                target,
                mean,
                std,
                float(budget),
                route_costs,
                prefix_costs,
                sample_actions=sample_actions,
                hard_cap=hard_cap,
            )
            losses.append(rollout["losses"].cpu())
            costs.append(rollout["costs_ms"].cpu())
            routes.append(rollout["route_indices"].cpu())
            decisions += rollout["sampled_decisions"]
            transitions += rollout["chosen_transition_executions"]
        report = _route_report(torch.cat(losses), torch.cat(costs), torch.cat(routes))
        report.update(
            {
                "budget_ms": float(budget),
                "average_budget_violation_ms": max(
                    0.0, report["mean_latency_ms"] - float(budget)
                ),
                "sampled_decisions": int(decisions),
                "chosen_transition_executions": int(transitions),
            }
        )
        rows.append(report)
    return rows


@torch.no_grad()
def evaluate_all_routes(
    model, loader, device, mean, std, max_batches
) -> torch.Tensor:
    """Evaluation-only full route panel for fixed baselines and oracle headroom."""
    pieces = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target = prepare_batch(batch, device)
        target_raw = to_physical(target, mean, std)
        pieces.append(
            torch.stack(
                [
                    per_sample_mae(
                        to_physical(model(history, trajectory=route), mean, std),
                        target_raw,
                        null_val=0.0,
                    ).cpu()
                    for route in TRAJECTORIES
                ],
                dim=1,
            )
        )
    return torch.cat(pieces)


def best_fixed_mixture(fixed_mae, fixed_costs, budget_ms):
    candidates = []
    for index, route in enumerate(TRAJECTORIES):
        if fixed_costs[route] <= budget_ms + 1e-6:
            candidates.append((float(fixed_mae[index]), index, 1.0, None, 0.0))
    for low, low_route in enumerate(TRAJECTORIES):
        low_cost = fixed_costs[low_route]
        if low_cost >= budget_ms:
            continue
        for high, high_route in enumerate(TRAJECTORIES):
            high_cost = fixed_costs[high_route]
            if high_cost <= budget_ms:
                continue
            fraction = (budget_ms - low_cost) / (high_cost - low_cost)
            mae = (1 - fraction) * float(fixed_mae[low]) + fraction * float(
                fixed_mae[high]
            )
            candidates.append((mae, low, 1 - fraction, high, fraction))
    if not candidates:
        return None
    mae, low, low_fraction, high, high_fraction = min(candidates)
    components = [
        {"trajectory": list(TRAJECTORIES[low]), "fraction": float(low_fraction)}
    ]
    if high is not None:
        components.append(
            {"trajectory": list(TRAJECTORIES[high]), "fraction": float(high_fraction)}
        )
    return {"mae": float(mae), "components": components}


def average_oracle(losses, route_costs, budget_ms):
    cost = torch.tensor([route_costs[route] for route in TRAJECTORIES])
    breakpoints = [torch.zeros(1)]
    for left in range(len(TRAJECTORIES)):
        for right in range(left + 1, len(TRAJECTORIES)):
            dc = float(cost[right] - cost[left])
            if abs(dc) < 1e-6:
                continue
            value = (losses[:, left] - losses[:, right]) / dc
            breakpoints.append(value[value >= 0])
    values = torch.cat(breakpoints)
    if len(values) > 512:
        values = torch.quantile(values, torch.linspace(0, 1, 512))
    rows = torch.arange(len(losses))
    candidates = []
    for lam in torch.unique(values.clamp_min(0), sorted=True).tolist():
        selected = (losses + float(lam) * cost[None, :]).argmin(dim=1)
        selected_cost = cost[selected]
        if float(selected_cost.mean()) <= budget_ms + 1e-6:
            candidates.append(
                (float(losses[rows, selected].mean()), selected, selected_cost, lam)
            )
    if not candidates:
        index = int(cost.argmin())
        selected = torch.full((len(losses),), index, dtype=torch.long)
        candidates.append(
            (float(losses[:, index].mean()), selected, cost[selected], float("inf"))
        )
    _, selected, selected_cost, lam = min(candidates, key=lambda item: item[0])
    report = _route_report(losses[rows, selected], selected_cost, selected)
    report["lambda_mae_per_ms"] = float(lam)
    report["analysis_only"] = True
    return report


def attach_baselines(rows, route_losses, fixed_costs, adaptive_costs):
    fixed_mae = route_losses.mean(dim=0)
    cost = torch.tensor([adaptive_costs[route] for route in TRAJECTORIES])
    for row in rows:
        budget = row["budget_ms"]
        feasible = [
            index
            for index, route in enumerate(TRAJECTORIES)
            if fixed_costs[route] <= budget + 1e-6
        ]
        best = min(feasible, key=lambda index: float(fixed_mae[index]))
        mixture = best_fixed_mixture(fixed_mae, fixed_costs, budget)
        oracle = average_oracle(route_losses, adaptive_costs, budget)
        hard_feasible = torch.tensor(
            [adaptive_costs[route] <= budget + 1e-6 for route in TRAJECTORIES]
        )
        hard_selected = route_losses.masked_fill(
            ~hard_feasible[None, :], torch.inf
        ).argmin(dim=1)
        hard_rows = torch.arange(len(route_losses))
        hard_selected_cost = cost.index_select(0, hard_selected)
        hard_oracle = _route_report(
            route_losses[hard_rows, hard_selected],
            hard_selected_cost,
            hard_selected,
        )
        hard_oracle["analysis_only"] = True
        row["best_fixed"] = {
            "trajectory": list(TRAJECTORIES[best]),
            "mae": float(fixed_mae[best]),
            "p90_latency_ms": float(fixed_costs[TRAJECTORIES[best]]),
        }
        row["best_fixed_average_mixture"] = mixture
        row["average_oracle"] = oracle
        row["hard_cap_oracle"] = hard_oracle
        row["gain_vs_fixed_mixture"] = (
            None if mixture is None else float(mixture["mae"] - row["mae"])
        )
        row["regret_to_oracle"] = float(row["mae"] - oracle["mae"])
        row["regret_to_hard_cap_oracle"] = float(
            row["mae"] - hard_oracle["mae"]
        )
    return rows


def train_epoch(
    model,
    policy,
    loader,
    optimizer,
    device,
    mean,
    std,
    budgets,
    duals,
    route_costs,
    prefix_costs,
    args,
    epoch,
):
    policy.train()
    started = time.perf_counter()
    totals = defaultdict(float)
    route_counts = {float(budget): Counter() for budget in budgets}
    edge_counts = Counter()
    samples = 0
    sampled_decisions = 0
    transition_executions = 0
    reference_executions = 0
    decay_fraction = min(
        1.0, max(0.0, (epoch - 1) / max(args.entropy_decay_epochs, 1e-6))
    )
    entropy_weight = (
        args.entropy_weight
        + decay_fraction * (args.entropy_final_weight - args.entropy_weight)
    )
    for batch_index, batch in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        history, target = prepare_batch(batch, device)
        unique_batch_size = len(history)
        # Online action-independent control variate: execute the untouched
        # canonical forecaster now, on this batch. This is never cached and is
        # not an action label; it only turns absolute MAE into the within-sample
        # question "did the sampled route improve on canonical?".
        with torch.no_grad():
            canonical_prediction = model.f2f(history)
            canonical_loss = per_sample_mae(
                to_physical(canonical_prediction, mean, std),
                to_physical(target, mean, std),
                null_val=0.0,
            ).detach()
        reference_executions += unique_batch_size
        group_ids = torch.arange(unique_batch_size, device=device).repeat_interleave(
            args.rollouts_per_sample
        )
        history = history.repeat_interleave(args.rollouts_per_sample, dim=0)
        target = target.repeat_interleave(args.rollouts_per_sample, dim=0)
        reference_loss = canonical_loss.repeat_interleave(args.rollouts_per_sample)
        original_budget_indices = (
            torch.arange(unique_batch_size, device=device) + batch_index + epoch - 1
        ) % len(budgets)
        budget_indices = original_budget_indices.repeat_interleave(
            args.rollouts_per_sample
        )
        budget_table = torch.tensor(budgets, device=device, dtype=history.dtype)
        dual_table = torch.tensor(duals, device=device, dtype=history.dtype)
        budget_values = budget_table.index_select(0, budget_indices)
        multiplier_values = dual_table.index_select(0, budget_indices)
        rollout = rollout_online_batch(
            model,
            policy,
            history,
            target,
            mean,
            std,
            budget_values,
            route_costs,
            prefix_costs,
            sample_actions=True,
            hard_cap=args.hard_cap,
            training_group_ids=group_ids,
            reference_losses=reference_loss,
        )
        ppo_details = []
        for _ in range(args.ppo_epochs):
            loss, detail = actor_critic_loss(
                policy,
                rollout,
                multiplier_values,
                budget_values,
                args.critic_weight,
                entropy_weight,
                args.mae_reward_scale,
                args.ppo_clip,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            ppo_details.append(detail)
        detail = {
            key: statistics.fmean(row[key] for row in ppo_details)
            for key in ppo_details[0]
        }

        mean_cost = float(rollout["costs_ms"].mean())
        if epoch > args.dual_warmup_epochs:
            for budget_index, budget in enumerate(budgets):
                selected = budget_indices == budget_index
                if bool(selected.any()):
                    selected_cost = float(rollout["costs_ms"][selected].mean())
                    duals[budget_index] = max(
                        0.0,
                        min(
                            args.dual_max,
                            float(duals[budget_index])
                            + args.dual_lr * (selected_cost - float(budget)),
                        ),
                    )
        n = len(history)
        samples += n
        totals["unique_samples"] += unique_batch_size
        totals["mae"] += float(rollout["losses"].mean()) * n
        totals["cost"] += mean_cost * n
        for key, value in detail.items():
            totals[key] += value * n
        for route, budget_index in zip(rollout["routes"], budget_indices.tolist()):
            route_counts[float(budgets[budget_index])][route_name(route)] += 1
        edge_counts.update(rollout["executed_edges"])
        sampled_decisions += rollout["sampled_decisions"]
        transition_executions += rollout["chosen_transition_executions"]

    elapsed = time.perf_counter() - started
    return {
        "epoch": int(epoch),
        "seconds": float(elapsed),
        "samples": int(samples),
        "unique_samples": int(totals["unique_samples"]),
        "rollouts_per_sample": int(args.rollouts_per_sample),
        "samples_per_second": float(samples / max(elapsed, 1e-9)),
        "mae": totals["mae"] / max(samples, 1),
        "mean_latency_ms": totals["cost"] / max(samples, 1),
        "actor_loss": totals["actor"] / max(samples, 1),
        "critic_loss": totals["critic"] / max(samples, 1),
        "entropy": totals["entropy"] / max(samples, 1),
        "entropy_weight": float(entropy_weight),
        "reference_mae": totals["reference_mae"] / max(samples, 1),
        "duals": {str(float(b)): float(duals[i]) for i, b in enumerate(budgets)},
        "route_counts_by_budget": {
            str(budget): dict(counts) for budget, counts in route_counts.items()
        },
        "executed_edge_counts": {
            f"{source}->{target}": int(count)
            for (source, target), count in edge_counts.items()
        },
        "sampled_decisions": int(sampled_decisions),
        "chosen_transition_executions": int(transition_executions),
        "online_canonical_reference_executions": int(reference_executions),
        "acceptance_exact_match": sampled_decisions == transition_executions,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="On-policy actor-critic over real forecast transitions."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--f2f-checkpoint", type=Path, default=DEFAULT_F2F_CHECKPOINT)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Unique training examples per batch before on-policy replication.",
    )
    parser.add_argument(
        "--rollouts-per-sample",
        type=int,
        default=8,
        help=(
            "Independent sampled trajectories executed online per training example; "
            "used only for a leave-one-out policy-gradient baseline."
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--controller-state",
        choices=("legacy", "xz_relation"),
        default="legacy",
        help=(
            "legacy keeps independently compressed X/Z features; xz_relation "
            "uses a shared temporal encoder and explicit learned X--Z contrast."
        ),
    )
    parser.add_argument("--critic-weight", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument(
        "--mae-reward-scale",
        type=float,
        default=1.0,
        help="Positive numerical scaling of MAE return; does not change its optimum.",
    )
    parser.add_argument("--entropy-weight", type=float, default=1.0)
    parser.add_argument("--entropy-final-weight", type=float, default=0.1)
    parser.add_argument("--entropy-decay-epochs", type=float, default=15.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--dual-lr", type=float, default=0.001)
    parser.add_argument("--dual-initial", type=float, default=0.0)
    parser.add_argument("--dual-warmup-epochs", type=int, default=5)
    parser.add_argument("--dual-max", type=float, default=10.0)
    parser.add_argument("--budget-count", type=int, default=4)
    parser.add_argument("--budgets-ms", type=str)
    parser.add_argument(
        "--hard-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mask actions whose measured route completion exceeds B (default); "
            "--no-hard-cap runs the average-only ablation."
        ),
    )
    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="Sampling ablation; deployment and default evaluation use fast greedy actions.",
    )
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--bridge-correction-limit", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 2
        args.patience = 2
        args.workers = 0
        args.max_train_batches = 2
        args.max_eval_batches = 2
        args.rollouts_per_sample = min(args.rollouts_per_sample, 2)
        args.latency_warmup = 1
        args.latency_repeats = 3
    seed_everything(args.seed)
    device = torch.device(args.device)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or ROOT / (
        f"checkpoints/ForecastTrajectorySimpleOnlineRL/seed{args.seed}_{run_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"[run] output={output_dir}")
    print("[training] on-policy: no trajectory/state/loss cache is read or written")

    data, indices, mean, std = load_data(args.data_dir)
    datasets = {
        split: WindowDataset(data, indices[split]) for split in ("train", "valid")
    }

    def loader(split, batch_size=None, shuffle=False):
        return make_loader(
            datasets[split],
            batch_size=batch_size or args.batch_size,
            shuffle=shuffle,
            workers=args.workers,
            device=device,
            seed=args.seed,
        )

    model = build_frozen_forecaster(args, device)
    canonical_audit(model, loader("valid", args.eval_batch_size), device)
    policy_class = (
        OnlineXZRelationActorCritic
        if args.controller_state == "xz_relation"
        else OnlineResolutionActorCritic
    )
    policy = policy_class(
        node_count=int(data.shape[1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    latency_history, _ = prepare_batch(
        next(iter(loader("valid", 1))), device
    )
    fixed_raw, adaptive_raw, prefix_raw, fixed_costs, adaptive_costs, prefix_costs = (
        profile_physical_costs(model, policy, latency_history, args)
    )
    if args.budgets_ms:
        budgets = sorted(float(value) for value in args.budgets_ms.split(","))
    else:
        budgets = derive_budget_grid(adaptive_costs, args.budget_count)
    if args.hard_cap and min(budgets) + 1e-6 < min(adaptive_costs.values()):
        raise ValueError(
            "A hard budget is below the cheapest measured online trajectory: "
            f"B={min(budgets):.6f} ms, minimum={min(adaptive_costs.values()):.6f} ms"
        )
    policy.set_budget_range(min(budgets), max(budgets))
    print(f"[latency] average budgets_ms={[round(value, 4) for value in budgets]}")

    latency_report = {
        "physical_definition": "batch-1 synchronized CUDA p90 milliseconds",
        "fixed_forecast_only": {route_name(k): v for k, v in fixed_raw.items()},
        "online_actor_plus_forecast": {
            route_name(k): v for k, v in adaptive_raw.items()
        },
        "reached_prefix_costs": {
            route_name(k): v for k, v in prefix_raw.items()
        },
        "policy_overhead_ms": {
            route_name(route): float(adaptive_costs[route] - fixed_costs[route])
            for route in TRAJECTORIES
        },
    }
    (output_dir / "latency_profile.json").write_text(
        json.dumps(latency_report, indent=2), encoding="utf-8"
    )

    # VALID fixed-route outcomes are evaluation baselines only. They are never
    # presented to the actor as training targets.
    valid_route_losses = evaluate_all_routes(
        model,
        loader("valid", args.eval_batch_size),
        device,
        mean,
        std,
        args.max_eval_batches,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    duals = [float(args.dual_initial) for _ in budgets]
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    history_log = []
    best_path = output_dir / "online_actor_critic_best.pt"
    acceptance_totals = Counter()

    for epoch in range(1, args.epochs + 1):
        train_row = train_epoch(
            model,
            policy,
            loader("train", args.batch_size, shuffle=True),
            optimizer,
            device,
            mean,
            std,
            budgets,
            duals,
            adaptive_costs,
            prefix_costs,
            args,
            epoch,
        )
        acceptance_totals["sampled_decisions"] += train_row["sampled_decisions"]
        acceptance_totals["chosen_transition_executions"] += train_row[
            "chosen_transition_executions"
        ]
        acceptance_totals["online_canonical_reference_executions"] += train_row[
            "online_canonical_reference_executions"
        ]
        valid_rows = evaluate_policy(
            model,
            policy,
            loader("valid", args.eval_batch_size),
            device,
            mean,
            std,
            budgets,
            adaptive_costs,
            prefix_costs,
            args.max_eval_batches,
            hard_cap=args.hard_cap,
            sample_actions=args.stochastic_eval,
        )
        attach_baselines(
            valid_rows, valid_route_losses, fixed_costs, adaptive_costs
        )
        score = statistics.fmean(
            row["mae"] + 5.0 * row["average_budget_violation_ms"]
            for row in valid_rows
        )
        improved = score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "policy_state_dict": policy.state_dict(),
                    "duals": list(duals),
                    "budgets_ms": list(budgets),
                    "validation_score": score,
                    "validation_frontier": valid_rows,
                    "online_training": True,
                    "cached_supervision": False,
                },
                best_path,
            )
        else:
            stale += 1
        history_log.append(
            {
                "epoch": epoch,
                "train": train_row,
                "validation": valid_rows,
                "validation_score": score,
                "best": improved,
            }
        )
        (output_dir / "history.json").write_text(
            json.dumps(history_log, indent=2), encoding="utf-8"
        )
        print(
            f"[epoch {epoch:03d}] seconds={train_row['seconds']:.1f} "
            f"train_mae={train_row['mae']:.5f} entropy={train_row['entropy']:.4f} "
            f"valid_score={score:.5f} duals={[round(v, 4) for v in duals]} "
            f"best={improved}"
        )
        if stale >= args.patience:
            print(f"[early-stop] {args.patience} epochs without improvement")
            break

    checkpoint = torch.load(best_path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    policy.eval()
    valid_rows = evaluate_policy(
        model,
        policy,
        loader("valid", args.eval_batch_size),
        device,
        mean,
        std,
        budgets,
        adaptive_costs,
        prefix_costs,
        args.max_eval_batches,
        hard_cap=args.hard_cap,
        sample_actions=args.stochastic_eval,
    )
    attach_baselines(valid_rows, valid_route_losses, fixed_costs, adaptive_costs)

    with torch.no_grad():
        expected = model.f2f(latency_history)
        actual = execute_forced_online_route(
            model, policy, latency_history, (3, 6, 12)
        )
    canonical_equal = torch.equal(expected, actual)
    canonical_max_abs = float((expected - actual).abs().max())
    if not canonical_equal:
        raise RuntimeError(f"Canonical online route changed: {canonical_max_abs}")
    relation_state_report = evaluate_xz_relation_states(
        model,
        policy,
        loader("valid", args.eval_batch_size),
        device,
        args.max_eval_batches,
    )

    base_report = {
        "method": (
            "OnlineXZRelationConstrainedSequentialPPO_RLOO"
            if args.controller_state == "xz_relation"
            else "OnlineConstrainedSequentialPPO_RLOO"
        ),
        "controller_state": args.controller_state,
        "forecast_graph_modified": False,
        "terminal_correction_used": False,
        "xz_relation_states_valid": relation_state_report,
        "training_mechanism": (
            "on-policy sampled next edge -> real forecast transition -> newly "
            "observed explicit forecast -> next decision -> clipped PPO updates"
        ),
        "primary_constraint": (
            "hard per-sample measured p90 completion mask plus average "
            "measured p90 latency with primal-dual update"
            + (
                ""
                if args.hard_cap
                else "; hard masks disabled for this average-only ablation"
            )
        ),
        "deployment_policy": (
            "categorical sampling ablation"
            if args.stochastic_eval
            else "fast greedy actor over the physically feasible action set"
        ),
        "objective": (
            f"{args.mae_reward_scale:g} * (terminal physical-scale MAE - "
            "fresh online canonical MAE) + "
            "lambda_B * (latency_ms - B); positive MAE scaling is numerical only"
        ),
        "variance_reduction": (
            f"{args.rollouts_per_sample} independent fully executed on-policy "
            "trajectories per TRAIN sample with leave-one-out return baseline; "
            "a freshly executed canonical forecast is the action-independent "
            "within-sample control variate"
        ),
        "best_valid_epoch": int(best_epoch),
        "checkpoint": str(best_path),
        "policy_parameters": policy.parameter_count(),
        "budgets_ms": budgets,
        "canonical_exact": {
            "torch_equal": canonical_equal,
            "max_abs": canonical_max_abs,
        },
        "latency": latency_report,
        "validation": valid_rows,
        "training_epochs": [row["train"] for row in history_log],
        "hard_acceptance_test": {
            "cached_path_losses_used_for_training": False,
            "cached_state_features_used_for_training": False,
            "complete_routes_chosen_before_execution": False,
            "sampled_decisions": int(acceptance_totals["sampled_decisions"]),
            "chosen_transition_executions": int(
                acceptance_totals["chosen_transition_executions"]
            ),
            "online_canonical_reference_executions": int(
                acceptance_totals["online_canonical_reference_executions"]
            ),
            "every_sampled_action_executed_real_transition": (
                acceptance_totals["sampled_decisions"]
                == acceptance_totals["chosen_transition_executions"]
                and acceptance_totals["sampled_decisions"] > 0
            ),
            "transition_call_site": (
                "run_online_sequential_rl.py::rollout_online_batch -> "
                "model.execute_transition"
            ),
        },
        "target_usage": {
            "train": "sampled terminal reward only; no counterfactual route labels",
            "valid": "checkpoint selection, baselines, and oracle analysis",
            "test": "not evaluated" if args.skip_test else "one final evaluation",
        },
    }

    if args.skip_test:
        report_path = output_dir / "valid_report.json"
        report_path.write_text(json.dumps(base_report, indent=2), encoding="utf-8")
        print(f"[done] VALID-only report={report_path}")
        return

    datasets["test"] = WindowDataset(data, indices["test"])
    test_rows = evaluate_policy(
        model,
        policy,
        loader("test", args.eval_batch_size),
        device,
        mean,
        std,
        budgets,
        adaptive_costs,
        prefix_costs,
        args.max_eval_batches,
        hard_cap=args.hard_cap,
        sample_actions=args.stochastic_eval,
    )
    test_route_losses = evaluate_all_routes(
        model,
        loader("test", args.eval_batch_size),
        device,
        mean,
        std,
        args.max_eval_batches,
    )
    attach_baselines(test_rows, test_route_losses, fixed_costs, adaptive_costs)
    base_report["test"] = test_rows
    report_path = output_dir / "final_report.json"
    report_path.write_text(json.dumps(base_report, indent=2), encoding="utf-8")
    print(json.dumps({"test": test_rows}, indent=2))
    print(f"[done] checkpoint={best_path}")
    print(f"[done] report={report_path}")


if __name__ == "__main__":
    main()
