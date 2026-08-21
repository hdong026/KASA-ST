"""Quality-first trajectory evaluation and the later latency trade-off."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from basicts.archs.arch_zoo.ChainForecasting_arch.ChainForecasting_arch import (
    ChainForecasting,
)


def per_sample_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    null_val: float = 0.0,
) -> torch.Tensor:
    """Return one masked MAE value per sample without cross-sample normalization."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}"
        )
    valid = torch.isfinite(target)
    if null_val == null_val:  # NaN is intentionally handled by isfinite only.
        valid = valid & target.ne(null_val)
    reduce_dims = tuple(range(1, target.ndim))
    count = valid.sum(dim=reduce_dims).clamp_min(1)
    error = torch.where(valid, (prediction - target).abs(), torch.zeros_like(target))
    return error.sum(dim=reduce_dims) / count


def headroom_from_predictions(
    predictions: Mapping[Sequence[int], torch.Tensor],
    target: torch.Tensor,
    *,
    canonical_trajectory: Sequence[int] = (3, 6, 12),
    null_val: float = 0.0,
) -> dict:
    """Compare sample-wise label oracle routing with every fixed trajectory."""
    if not predictions:
        raise ValueError("predictions cannot be empty.")
    routes = tuple(tuple(int(value) for value in route) for route in predictions)
    canonical = tuple(int(value) for value in canonical_trajectory)
    if canonical not in routes:
        raise ValueError(f"Canonical trajectory {list(canonical)} is missing.")

    losses = torch.stack(
        [per_sample_mae(prediction, target, null_val) for prediction in predictions.values()],
        dim=1,
    )
    fixed_mae = losses.mean(dim=0)
    best_fixed_index = fixed_mae.argmin()
    oracle_loss, oracle_route_index = losses.min(dim=1)
    canonical_index = routes.index(canonical)

    stacked_predictions = torch.stack(list(predictions.values()), dim=1)
    view_shape = [target.shape[0], 1] + [1] * (target.ndim - 1)
    gather_index = oracle_route_index.view(*view_shape).expand(
        target.shape[0], 1, *target.shape[1:]
    )
    oracle_prediction = stacked_predictions.gather(1, gather_index).squeeze(1)

    oracle_counts = torch.bincount(oracle_route_index, minlength=len(routes))
    return {
        "routes": routes,
        "per_sample_route_mae": losses,
        "fixed_route_mae": fixed_mae,
        "canonical_mae": fixed_mae[canonical_index],
        "best_fixed_index": best_fixed_index,
        "best_fixed_trajectory": routes[int(best_fixed_index.item())],
        "best_fixed_mae": fixed_mae[best_fixed_index],
        "oracle_route_index": oracle_route_index,
        "oracle_route_counts": oracle_counts,
        "oracle_prediction": oracle_prediction,
        "oracle_mae": oracle_loss.mean(),
        "oracle_gain_vs_canonical": fixed_mae[canonical_index] - oracle_loss.mean(),
        "oracle_gain_vs_best_fixed": fixed_mae[best_fixed_index] - oracle_loss.mean(),
    }


def quality_latency_objective(
    per_sample_route_loss: torch.Tensor,
    route_times: Sequence[float] | torch.Tensor,
    preference_lambda: float | torch.Tensor,
    *,
    latency_ceiling: float | torch.Tensor | None = None,
) -> dict:
    """Compute forecast loss + lambda * measured time for each route.

    route_times must contain measured times in a consistent unit. Lambda is
    expressed in loss units per that time unit. latency_ceiling only masks
    physically infeasible routes; it never changes user preference.
    """
    if per_sample_route_loss.ndim != 2:
        raise ValueError("per_sample_route_loss must have shape [batch, routes].")
    times = torch.as_tensor(
        route_times,
        device=per_sample_route_loss.device,
        dtype=per_sample_route_loss.dtype,
    )
    if times.ndim != 1 or times.numel() != per_sample_route_loss.shape[1]:
        raise ValueError("route_times must have one measured value per route.")
    lam = torch.as_tensor(
        preference_lambda,
        device=per_sample_route_loss.device,
        dtype=per_sample_route_loss.dtype,
    )
    if lam.ndim == 1:
        if lam.numel() != per_sample_route_loss.shape[0]:
            raise ValueError("A vector preference_lambda must have one value per sample.")
        lam = lam[:, None]
    objective = per_sample_route_loss + lam * times[None, :]

    feasible = torch.ones_like(objective, dtype=torch.bool)
    if latency_ceiling is not None:
        ceiling = torch.as_tensor(
            latency_ceiling,
            device=objective.device,
            dtype=objective.dtype,
        )
        if ceiling.ndim == 1:
            if ceiling.numel() != objective.shape[0]:
                raise ValueError("A vector latency_ceiling must have one value per sample.")
            ceiling = ceiling[:, None]
        feasible = (times[None, :] <= ceiling).expand_as(objective)
        if (~feasible).all(dim=1).any():
            raise ValueError("At least one sample has no route within its latency ceiling.")
        objective = objective.masked_fill(~feasible, torch.inf)

    selected_objective, selected_route_index = objective.min(dim=1)
    return {
        "objective": objective,
        "feasible": feasible,
        "selected_route_index": selected_route_index,
        "selected_objective": selected_objective,
        "selected_time": times[selected_route_index],
    }


def trajectory_supervision_loss(
    trajectory_result: Mapping,
    target: torch.Tensor,
    *,
    null_val: float = 0.0,
    intermediate_bridge_weight: float = 0.0,
) -> torch.Tensor:
    """Supervise a bridge trajectory, with full-resolution quality primary.

    Intermediate bridge supervision is optional and off by default. Native F2F
    states are never re-supervised by this helper.
    """
    loss = per_sample_mae(trajectory_result["pred"], target, null_val).mean()
    weight = float(intermediate_bridge_weight)
    if weight <= 0.0:
        return loss
    edges = trajectory_result["trajectory_edges"]
    edge_types = trajectory_result["edge_types"]
    states = trajectory_result["state_forecasts"]
    auxiliary = []
    for (_, target_resolution), edge_type in zip(edges[:-1], edge_types[:-1]):
        if edge_type != "bridge":
            continue
        pooled = ChainForecasting.pool_target(target, target_resolution)
        auxiliary.append(per_sample_mae(states[target_resolution], pooled, null_val).mean())
    if auxiliary:
        loss = loss + weight * torch.stack(auxiliary).mean()
    return loss
