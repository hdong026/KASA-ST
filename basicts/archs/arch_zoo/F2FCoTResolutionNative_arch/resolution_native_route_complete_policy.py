"""Exact full-information policy for the ResolutionNative eight-route DAG.

This module is intentionally independent from the frozen ResolutionNative V1
forecaster.  It consumes target-free structured summaries cached by a frozen
route-complete forecaster and never receives targets or target-derived losses as
router inputs.  Losses are used only by the full-information training objective.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


ACTION_RESOLUTIONS = (2, 3, 4, 6, 12)
ACTION_INDEX = {resolution: index for index, resolution in enumerate(ACTION_RESOLUTIONS)}
ROUTES = (
    (12,),
    (2, 12),
    (2, 4, 12),
    (2, 6, 12),
    (3, 12),
    (3, 6, 12),
    (4, 12),
    (6, 12),
)
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
PREFIX_INDEX = {prefix: index for index, prefix in enumerate(PREFIXES)}
ROUTE_NAMES = tuple("-".join(map(str, route)) for route in ROUTES)

_FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "label",
    "ground_truth",
    "future_truth",
    "route_loss",
    "final_mae",
    "oracle",
    "reward",
    "advantage",
)


def legal_action_mask(
    prefix: Sequence[int],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the hard nested-DAG mask over ``ACTION_RESOLUTIONS``."""
    key = tuple(int(value) for value in prefix)
    if key not in LEGAL_NEXT:
        raise ValueError(f"unknown or terminal ResolutionNative prefix: {key}")
    allowed = set(LEGAL_NEXT[key])
    return torch.tensor(
        [resolution in allowed for resolution in ACTION_RESOLUTIONS],
        dtype=torch.bool,
        device=device,
    )


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax with an exact hard mask and no probability on illegal actions."""
    if mask.ndim == 1:
        mask = mask.view(1, -1).expand(logits.shape[0], -1)
    if logits.shape != mask.shape:
        raise ValueError(f"logits/mask mismatch: {logits.shape} != {mask.shape}")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every policy state must have at least one legal action")
    masked = logits.masked_fill(~mask, float("-inf"))
    return torch.log_softmax(masked, dim=-1)


def validate_inference_safe_feature_names(feature_names: Sequence[str]) -> None:
    """Reject obvious target leakage in cached router feature declarations."""
    violations = []
    for name in feature_names:
        lowered = str(name).lower()
        if any(token in lowered for token in _FORBIDDEN_FEATURE_TOKENS):
            violations.append(str(name))
    if violations:
        raise ValueError(
            "router state contains target-derived feature names: "
            + ", ".join(violations)
        )


def normalize_actual_flops(route_flops: torch.Tensor) -> torch.Tensor:
    """Min-max normalize measured route FLOPs while preserving route ordering."""
    costs = route_flops.float()
    if costs.ndim != 1 or costs.numel() != len(ROUTES):
        raise ValueError(f"expected {len(ROUTES)} route FLOPs, got {tuple(costs.shape)}")
    if not bool(torch.isfinite(costs).all()) or bool((costs <= 0).any()):
        raise ValueError("actual route FLOPs must be finite and positive")
    span = costs.max() - costs.min()
    if float(span) <= 0.0:
        raise ValueError("route FLOPs must contain at least two distinct costs")
    return (costs - costs.min()) / span


def robust_global_route_margin_scale(
    train_route_losses: torch.Tensor,
    *,
    split: str,
    eps: float = 1e-6,
) -> float:
    """Fit one robust scale from TRAIN-only pairwise route-loss margins.

    The median positive absolute pairwise margin is robust to a small number of
    difficult samples and, unlike per-group standardization, leaves relative
    margin magnitudes intact across samples.
    """
    if str(split).lower() != "train":
        raise ValueError("global route-margin scale may be fitted on TRAIN only")
    losses = train_route_losses.detach().float().cpu()
    if losses.ndim != 2 or losses.shape[1] != len(ROUTES):
        raise ValueError(f"route losses must have shape [N,{len(ROUTES)}]")
    margins = []
    for left, right in combinations(range(len(ROUTES)), 2):
        values = (losses[:, left] - losses[:, right]).abs()
        values = values[torch.isfinite(values) & (values > eps)]
        if values.numel():
            margins.append(values)
    if not margins:
        return 1.0
    scale = float(torch.cat(margins).median().item())
    return max(scale, float(eps))


def mean_centered_advantages(utilities: torch.Tensor) -> torch.Tensor:
    """Group-relative centering without per-sample standard-deviation scaling."""
    if utilities.ndim != 2 or utilities.shape[1] != len(ROUTES):
        raise ValueError(f"utilities must have shape [B,{len(ROUTES)}]")
    return utilities - utilities.mean(dim=-1, keepdim=True)


def trajectory_kl(
    reference_probs: torch.Tensor,
    current_probs: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean KL(reference || current) on complete terminal trajectories."""
    reference = reference_probs.clamp_min(eps)
    current = current_probs.clamp_min(eps)
    reference = reference / reference.sum(dim=-1, keepdim=True)
    current = current / current.sum(dim=-1, keepdim=True)
    return (reference * (reference.log() - current.log())).sum(dim=-1).mean()


def terminal_entropy(route_probs: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    probs = route_probs.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return -(probs * probs.log()).sum(dim=-1).mean()


def entropy_coefficient(
    epoch: int,
    total_epochs: int,
    initial: float,
    *,
    anneal_fraction: float = 0.25,
) -> float:
    """Linearly remove exploration entropy during the early training window."""
    if initial <= 0.0 or total_epochs <= 0 or anneal_fraction <= 0.0:
        return 0.0
    end = max(1.0, float(total_epochs) * float(anneal_fraction))
    return float(initial) * max(0.0, 1.0 - float(epoch) / end)


class ResolutionNativeExactRouter(nn.Module):
    """Small sequential router over target-free prefix-state summaries."""

    def __init__(
        self,
        state_dim: int,
        *,
        hidden_dim: int = 64,
        resolution_dim: int = 8,
        prefix_dim: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.state_norm = nn.LayerNorm(self.state_dim)
        self.resolution_embedding = nn.Embedding(len(ACTION_RESOLUTIONS) + 1, resolution_dim)
        self.prefix_embedding = nn.Embedding(len(PREFIXES), prefix_dim)
        explicit_dim = len(ACTION_RESOLUTIONS) + 4
        input_dim = self.state_dim + resolution_dim + prefix_dim + explicit_dim
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(ACTION_RESOLUTIONS)),
        )

    @staticmethod
    def _current_resolution_index(prefix: tuple[int, ...]) -> int:
        if not prefix:
            return 0
        return ACTION_INDEX[prefix[-1]] + 1

    def prefix_log_probs(
        self,
        state_features: torch.Tensor,
        prefix: Sequence[int],
        budget: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Emit current legal next-resolution logits with a nested-DAG hard mask."""
        key = tuple(int(value) for value in prefix)
        if key not in PREFIX_INDEX:
            raise ValueError(f"unknown prefix: {key}")
        if state_features.ndim != 2 or state_features.shape[-1] != self.state_dim:
            raise ValueError(
                f"prefix state must be [B,{self.state_dim}], got {tuple(state_features.shape)}"
            )
        batch = state_features.shape[0]
        budget = budget.reshape(batch, 1).to(state_features)
        current = 0 if not key else key[-1]
        path_multi_hot = state_features.new_zeros(batch, len(ACTION_RESOLUTIONS))
        for resolution in key:
            path_multi_hot[:, ACTION_INDEX[resolution]] = 1.0
        explicit = torch.cat(
            (
                path_multi_hot,
                state_features.new_full((batch, 1), current / 12.0),
                state_features.new_full((batch, 1), len(key) / 3.0),
                budget,
                budget - state_features.new_full((batch, 1), current / 12.0),
            ),
            dim=-1,
        )
        resolution_index = torch.full(
            (batch,),
            self._current_resolution_index(key),
            dtype=torch.long,
            device=state_features.device,
        )
        prefix_index = torch.full(
            (batch,),
            PREFIX_INDEX[key],
            dtype=torch.long,
            device=state_features.device,
        )
        encoded = torch.cat(
            (
                self.state_norm(state_features),
                self.resolution_embedding(resolution_index),
                self.prefix_embedding(prefix_index),
                explicit,
            ),
            dim=-1,
        )
        logits = self.router(encoded)
        mask = legal_action_mask(key, device=logits.device).view(1, -1).expand(batch, -1)
        log_probs = masked_log_softmax(logits, mask)
        return logits, log_probs, mask

    def terminal_distribution(
        self,
        states_by_prefix: torch.Tensor,
        budget: torch.Tensor,
    ) -> dict[str, Any]:
        """Compute exact normalized route probabilities from action products.

        ``states_by_prefix`` follows ``PREFIXES`` and has shape ``[B,8,D]``.
        Forced one-action states are retained, making factorization explicit.
        """
        if (
            states_by_prefix.ndim != 3
            or states_by_prefix.shape[1] != len(PREFIXES)
            or states_by_prefix.shape[2] != self.state_dim
        ):
            raise ValueError(
                "states_by_prefix must have shape "
                f"[B,{len(PREFIXES)},{self.state_dim}], got {tuple(states_by_prefix.shape)}"
            )
        batch = states_by_prefix.shape[0]
        budget = budget.reshape(batch).to(states_by_prefix)
        action_log_probs: dict[tuple[int, ...], torch.Tensor] = {}
        action_logits: dict[tuple[int, ...], torch.Tensor] = {}
        action_masks: dict[tuple[int, ...], torch.Tensor] = {}
        for prefix, index in PREFIX_INDEX.items():
            logits, log_probs, mask = self.prefix_log_probs(
                states_by_prefix[:, index], prefix, budget
            )
            action_logits[prefix] = logits
            action_log_probs[prefix] = log_probs
            action_masks[prefix] = mask

        route_log_products = []
        for route in ROUTES:
            prefix: tuple[int, ...] = ()
            terms = []
            for action in route:
                terms.append(action_log_probs[prefix][:, ACTION_INDEX[action]])
                prefix = (*prefix, action)
                if action == 12:
                    break
            route_log_products.append(torch.stack(terms, dim=-1).sum(dim=-1))
        route_log_products_t = torch.stack(route_log_products, dim=-1)
        raw_products = route_log_products_t.exp()
        normalizer = raw_products.sum(dim=-1, keepdim=True)
        route_probs = raw_products / normalizer.clamp_min(1e-12)
        return {
            "route_probs": route_probs,
            "raw_route_action_products": raw_products,
            "route_normalizer": normalizer,
            "route_log_action_products": route_log_products_t,
            "action_logits": action_logits,
            "action_log_probs": action_log_probs,
            "action_masks": action_masks,
        }


def exact_full_information_objective(
    route_probs: torch.Tensor,
    route_losses: torch.Tensor,
    normalized_flops: torch.Tensor,
    budgets: torch.Tensor,
    dual_lambdas: torch.Tensor,
    *,
    global_margin_scale: float,
    reference_probs: torch.Tensor | None,
    kl_coefficient: float,
    entropy_coefficient_value: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Lagrangian exact expected trajectory utility for complete 8-route groups."""
    if route_probs.shape != route_losses.shape:
        raise ValueError("route probabilities and final per-sample losses must align")
    if route_probs.shape[-1] != len(ROUTES):
        raise ValueError(f"objective requires all {len(ROUTES)} terminal routes")
    costs = normalized_flops.to(route_probs).view(1, -1)
    budgets = budgets.to(route_probs).reshape(-1, 1)
    dual_lambdas = dual_lambdas.to(route_probs).reshape(-1, 1)
    quality_utility = -route_losses / max(float(global_margin_scale), 1e-8)
    lagrangian_utility = quality_utility - dual_lambdas * (costs - budgets)
    advantages = mean_centered_advantages(lagrangian_utility)
    expected_utility_per_sample = (route_probs * advantages).sum(dim=-1)
    expected_cost_per_sample = (route_probs * costs).sum(dim=-1)
    entropy = terminal_entropy(route_probs)
    if reference_probs is None or kl_coefficient <= 0.0:
        kl = route_probs.new_zeros(())
    else:
        kl = trajectory_kl(reference_probs.detach(), route_probs)
    loss = (
        -expected_utility_per_sample.mean()
        + float(kl_coefficient) * kl
        - float(entropy_coefficient_value) * entropy
    )
    return loss, {
        "expected_utility": expected_utility_per_sample.mean(),
        "expected_cost": expected_cost_per_sample.mean(),
        "cost_violation": (expected_cost_per_sample - budgets.squeeze(1)).mean(),
        "kl": kl,
        "entropy": entropy,
        "advantages": advantages,
        "expected_cost_per_sample": expected_cost_per_sample,
    }


class PrimalDualBudgetController:
    """Projected dual ascent for a fixed panel of expected normalized-FLOPs budgets."""

    def __init__(
        self,
        budgets: Sequence[float],
        *,
        learning_rate: float = 0.05,
        max_lambda: float = 100.0,
        device: torch.device | str = "cpu",
    ) -> None:
        values = torch.tensor(list(budgets), dtype=torch.float32, device=device)
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError("at least one expected-cost budget is required")
        if bool(((values < 0.0) | (values > 1.0)).any()):
            raise ValueError("normalized-FLOPs budgets must be in [0,1]")
        self.budgets = values
        self.lambdas = torch.zeros_like(values)
        self.learning_rate = float(learning_rate)
        self.max_lambda = float(max_lambda)

    def values_for(self, budget_indices: torch.Tensor) -> torch.Tensor:
        return self.lambdas[budget_indices.to(self.lambdas.device)].detach()

    @torch.no_grad()
    def update(self, expected_costs: torch.Tensor) -> torch.Tensor:
        costs = expected_costs.detach().to(self.lambdas)
        if costs.shape != self.budgets.shape:
            raise ValueError(
                f"expected one cost per budget: {costs.shape} != {self.budgets.shape}"
            )
        self.lambdas.add_(self.learning_rate * (costs - self.budgets))
        self.lambdas.clamp_(0.0, self.max_lambda)
        return self.lambdas.clone()

    def state_dict(self) -> dict[str, Any]:
        return {
            "budgets": self.budgets.detach().cpu(),
            "lambdas": self.lambdas.detach().cpu(),
            "learning_rate": self.learning_rate,
            "max_lambda": self.max_lambda,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        budgets = torch.as_tensor(state["budgets"], dtype=torch.float32)
        if not torch.allclose(budgets.cpu(), self.budgets.detach().cpu()):
            raise ValueError("dual-state budgets do not match configured budgets")
        self.lambdas.copy_(torch.as_tensor(state["lambdas"]).to(self.lambdas))


@dataclass
class RouteAnalysisCache:
    states: torch.Tensor
    route_losses: torch.Tensor
    route_flops: torch.Tensor
    sample_ids: np.ndarray
    split: str
    metadata: dict[str, Any]

    @property
    def normalized_flops(self) -> torch.Tensor:
        return normalize_actual_flops(self.route_flops)


def _first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"cache is missing one of required keys: {tuple(names)}")


def _decode_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return dict(json.loads(value))
    raise TypeError(f"unsupported cache metadata type: {type(value)!r}")


def load_route_analysis_cache(
    path: str | Path,
    *,
    expected_split: str,
    route_flops: Sequence[float] | np.ndarray | torch.Tensor | None = None,
) -> RouteAnalysisCache:
    """Load the strict NPZ/JSON contract used by policy-only training.

    Canonical arrays are ``state_features [N,8,D]`` and ``route_losses [N,8]``
    (final per-sample MAE).  The route-analysis producer's native aliases
    ``route_features`` and ``mae`` are accepted. Measured route FLOPs may be
    inline or supplied from its sibling ``cost_profile.json``.
    """
    path = Path(path)
    expected_split = str(expected_split).lower()
    if expected_split not in {"train", "valid"}:
        raise ValueError("policy cache split must be TRAIN or VALID; TEST is forbidden")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as packed:
            values = {key: packed[key] for key in packed.files}
        metadata = _decode_metadata(values.get("metadata_json"))
        states = _first(
            values,
            ("state_features", "structured_states", "router_states", "route_features"),
        )
        losses = _first(
            values, ("route_losses", "final_route_mae", "route_final_mae", "mae")
        )
        flops = values.get(
            "route_flops", values.get("actual_route_flops", route_flops)
        )
        sample_ids = values.get(
            "sample_ids", values.get("indices", np.arange(len(states)))
        )
        split = str(values.get("split", metadata.get("split", expected_split)))
        if isinstance(values.get("split"), np.ndarray):
            split = str(values["split"].item())
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = _decode_metadata(payload.get("metadata", {}))
        split = str(payload.get("split", metadata.get("split", expected_split)))
        flops = payload.get(
            "route_flops", payload.get("actual_route_flops", route_flops)
        )
        if "records" in payload:
            records = payload["records"]
            states = [
                _first(
                    record,
                    (
                        "state_features",
                        "structured_states",
                        "router_states",
                        "route_features",
                    ),
                )
                for record in records
            ]
            losses = [
                _first(
                    record,
                    ("route_losses", "final_route_mae", "route_final_mae", "mae"),
                )
                for record in records
            ]
            sample_ids = np.asarray(
                [record.get("sample_id", index) for index, record in enumerate(records)]
            )
        else:
            states = _first(
                payload,
                ("state_features", "structured_states", "router_states", "route_features"),
            )
            losses = _first(
                payload,
                ("route_losses", "final_route_mae", "route_final_mae", "mae"),
            )
            sample_ids = np.asarray(
                payload.get("sample_ids", payload.get("indices", np.arange(len(states))))
            )
    else:
        raise ValueError("route analysis cache must be .npz or .json")

    split = split.lower()
    if split != expected_split:
        raise ValueError(f"cache split is {split!r}, expected {expected_split!r}")
    if split == "test":
        raise ValueError("TEST caches are forbidden during policy development")
    declared_routes = metadata.get("routes")
    if declared_routes is not None:
        actual = tuple(tuple(int(value) for value in route) for route in declared_routes)
        if actual != ROUTES:
            raise ValueError(f"cache route order mismatch: {actual} != {ROUTES}")
    feature_names = metadata.get("feature_names", [])
    validate_inference_safe_feature_names(feature_names)
    cost_unit = str(metadata.get("cost_unit", "FLOPs")).lower()
    if "flop" not in cost_unit:
        raise ValueError(f"primary policy cost must be actual FLOPs, got {cost_unit!r}")
    if bool(metadata.get("cost_is_proxy", False)):
        raise ValueError("proxy route costs are not accepted; profile actual route FLOPs")
    if bool(metadata.get("contains_target_features", False)):
        raise ValueError("cache declares target-derived router features")
    if bool(metadata.get("uses_target_in_features", False)):
        raise ValueError("cache declares target-derived router features")
    if flops is None:
        raise ValueError(
            "cache has no actual route FLOPs; pass the analysis cost_profile.json"
        )

    states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    losses_t = torch.as_tensor(np.asarray(losses), dtype=torch.float32)
    flops_t = torch.as_tensor(np.asarray(flops), dtype=torch.float32)
    if states_t.ndim != 3 or states_t.shape[1] != len(PREFIXES):
        raise ValueError(
            f"state_features must be [N,{len(PREFIXES)},D], got {tuple(states_t.shape)}"
        )
    if losses_t.shape != (states_t.shape[0], len(ROUTES)):
        raise ValueError(
            f"route_losses must be [N,{len(ROUTES)}], got {tuple(losses_t.shape)}"
        )
    if not bool(torch.isfinite(states_t).all()):
        raise ValueError("router state cache contains non-finite values")
    if not bool(torch.isfinite(losses_t).all()):
        raise ValueError("final per-sample MAE cache contains non-finite values")
    normalize_actual_flops(flops_t)
    if len(sample_ids) != states_t.shape[0]:
        raise ValueError("sample_ids length does not match state cache")
    return RouteAnalysisCache(
        states=states_t,
        route_losses=losses_t,
        route_flops=flops_t,
        sample_ids=np.asarray(sample_ids),
        split=split,
        metadata=metadata,
    )


def load_actual_route_flops(path: str | Path) -> torch.Tensor:
    """Load raw profiler FLOPs from the route-analysis cost profile."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    primary = str(report.get("primary_cost", "")).lower()
    if "flop" not in primary:
        raise ValueError("cost profile primary_cost must be profiler FLOPs")
    routes = report.get("routes")
    if not isinstance(routes, dict):
        raise ValueError("cost profile must contain a route-keyed routes mapping")
    values = []
    for name in ROUTE_NAMES:
        entry = routes.get(name)
        if entry is None:
            entry = routes.get(name.replace("-", "->"))
        if not isinstance(entry, dict) or "flops" not in entry:
            raise ValueError(f"cost profile is missing raw FLOPs for route {name}")
        values.append(float(entry["flops"]))
    flops = torch.tensor(values, dtype=torch.float32)
    normalize_actual_flops(flops)
    return flops


def require_passing_gate(path: str | Path) -> dict[str, Any]:
    """Require explicit oracle and observability approval before any training."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if bool(report.get("test_used", report.get("TEST_used", False))) or report.get(
        "test"
    ) not in (None, False):
        raise RuntimeError("gate report used TEST; policy training is forbidden")
    nested_gate = report.get("gate", {})
    nested_gates = nested_gate.get("gates", {}) if isinstance(nested_gate, dict) else {}
    oracle = report.get("oracle_gate", {})
    observable = report.get("observability_gate", report.get("observability_probe", {}))
    oracle_passed = bool(
        oracle.get("passed", report.get("oracle_gate_passed", False))
        if isinstance(oracle, dict)
        else oracle
    )
    if not oracle_passed:
        oracle_passed = bool(
            nested_gates.get("VALID_oracle_headroom_at_least_0.03", False)
        )
    observable_passed = bool(
        observable.get("passed", report.get("observability_gate_passed", False))
        if isinstance(observable, dict)
        else observable
    )
    if not observable_passed:
        observable_passed = bool(
            nested_gates.get("probe_recovers_at_least_25pct", False)
            and nested_gates.get("probe_discrimination_nontrivial", False)
        )
    overall = bool(report.get("passed", report.get("overall_passed", False)))
    if not overall:
        overall = bool(nested_gates.get("proceed_to_policy_learning", False))
    if not (overall and oracle_passed and observable_passed):
        raise RuntimeError(
            "ROUTE POLICY NOT TRAINED: route oracle/observability gate did not pass "
            f"(overall={overall}, oracle={oracle_passed}, observability={observable_passed})"
        )
    return report


def route_histogram_metrics(route_probs: torch.Tensor) -> dict[str, Any]:
    selected = route_probs.argmax(dim=-1)
    counts = torch.bincount(selected, minlength=len(ROUTES)).float()
    shares = counts / counts.sum().clamp_min(1.0)
    nonzero = shares[shares > 0]
    effective = torch.exp(-(nonzero * nonzero.log()).sum()) if nonzero.numel() else shares.new_zeros(())
    return {
        "counts": {name: int(counts[index].item()) for index, name in enumerate(ROUTE_NAMES)},
        "shares": {name: float(shares[index].item()) for index, name in enumerate(ROUTE_NAMES)},
        "effective_routes": float(effective.item()),
    }


def negative_control_states(
    states: torch.Tensor,
    mode: str,
    *,
    zr_feature_indices: Sequence[int] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Executable shuffle-state and no-Zr controls for policy evaluation."""
    mode = str(mode).lower()
    if mode == "full":
        return states
    if mode == "shuffle":
        order = torch.randperm(states.shape[0], generator=generator, device=states.device)
        return states[order]
    if mode == "no_zr":
        if not zr_feature_indices:
            raise ValueError("no_zr control requires declared Zr feature indices")
        controlled = states.clone()
        controlled[:, :, list(map(int, zr_feature_indices))] = 0.0
        return controlled
    raise ValueError(f"unknown negative control: {mode}")
