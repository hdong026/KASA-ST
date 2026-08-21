#!/usr/bin/env python3
"""Train/evaluate the exact eight-route ResolutionNative policy from caches.

The frozen forecaster is never imported.  This entry point accepts TRAIN and
VALID route-analysis caches only, refuses to train unless both pre-policy gates
passed, and never constructs or reads TEST.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.resolution_native_route_complete_policy import (
    PREFIX_INDEX,
    ROUTE_NAMES,
    ROUTES,
    PrimalDualBudgetController,
    ResolutionNativeExactRouter,
    RouteAnalysisCache,
    entropy_coefficient,
    exact_full_information_objective,
    load_actual_route_flops,
    load_route_analysis_cache,
    negative_control_states,
    require_passing_gate,
    robust_global_route_margin_scale,
    route_histogram_metrics,
    trajectory_kl,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_budgets(text: str) -> list[float]:
    budgets = sorted({float(value.strip()) for value in text.split(",") if value.strip()})
    if not budgets or any(value < 0.0 or value > 1.0 for value in budgets):
        raise ValueError("expected normalized-FLOPs budgets in [0,1]")
    return budgets


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _batch_distribution(
    model: ResolutionNativeExactRouter,
    states: torch.Tensor,
    budget: float,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[tuple[int, ...], torch.Tensor]]:
    route_probs = []
    action_probs: dict[tuple[int, ...], list[torch.Tensor]] = {
        prefix: [] for prefix in PREFIX_INDEX
    }
    for start in range(0, states.shape[0], batch_size):
        state_batch = states[start : start + batch_size].to(device)
        budget_batch = torch.full(
            (state_batch.shape[0],), float(budget), device=device
        )
        output = model.terminal_distribution(state_batch, budget_batch)
        route_probs.append(output["route_probs"].cpu())
        for prefix, log_probs in output["action_log_probs"].items():
            action_probs[prefix].append(log_probs.exp().cpu())
    return torch.cat(route_probs), {
        prefix: torch.cat(values) for prefix, values in action_probs.items()
    }


@torch.inference_mode()
def evaluate_policy(
    model: ResolutionNativeExactRouter,
    cache: RouteAnalysisCache,
    normalized_flops: torch.Tensor,
    budgets: Sequence[float],
    *,
    batch_size: int,
    device: torch.device,
    reference_model: ResolutionNativeExactRouter | None = None,
    states_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    if reference_model is not None:
        reference_model.eval()
    states = cache.states if states_override is None else states_override
    losses = cache.route_losses
    costs = normalized_flops.cpu()
    report: dict[str, Any] = {}
    for budget in budgets:
        probs, action_probs = _batch_distribution(
            model, states, budget, batch_size, device
        )
        expected_losses = (probs * losses).sum(dim=-1)
        expected_costs = (probs * costs.view(1, -1)).sum(dim=-1)
        selected = probs.argmax(dim=-1)
        rows = torch.arange(selected.numel())
        hard_losses = losses[rows, selected]
        hard_costs = costs[selected]
        if reference_model is None:
            kl = 0.0
        else:
            reference_probs, _ = _batch_distribution(
                reference_model, states, budget, batch_size, device
            )
            kl = float(trajectory_kl(reference_probs, probs))
        report[f"{budget:.6g}"] = {
            "budget": float(budget),
            "expected_MAE": float(expected_losses.mean()),
            "expected_normalized_FLOPs": float(expected_costs.mean()),
            "cost_violation": float(max(0.0, expected_costs.mean().item() - budget)),
            "hard_MAE": float(hard_losses.mean()),
            "hard_normalized_FLOPs": float(hard_costs.float().mean()),
            "trajectory_KL_from_initial_policy": kl,
            "route_histogram": route_histogram_metrics(probs),
            "mean_action_probabilities": {
                "/".join(map(str, prefix)) if prefix else "root": {
                    str(resolution): float(values[:, index].mean())
                    for index, resolution in enumerate((2, 3, 4, 6, 12))
                    if bool((values[:, index] > 0).any())
                }
                for prefix, values in action_probs.items()
            },
        }
    return report


def _max_entropy_cost_distribution(costs: torch.Tensor, target: float) -> torch.Tensor:
    """Route-independent randomized baseline exactly matched in expected cost."""
    costs = costs.double()
    target = min(max(float(target), float(costs.min())), float(costs.max()))
    if abs(target - float(costs.min())) < 1e-10:
        mask = costs == costs.min()
        return (mask.double() / mask.sum()).float()
    if abs(target - float(costs.max())) < 1e-10:
        mask = costs == costs.max()
        return (mask.double() / mask.sum()).float()
    low, high = -1e4, 1e4
    for _ in range(100):
        middle = 0.5 * (low + high)
        probs = torch.softmax(middle * costs, dim=0)
        mean_cost = float((probs * costs).sum())
        if mean_cost < target:
            low = middle
        else:
            high = middle
    return torch.softmax(0.5 * (low + high) * costs, dim=0).float()


def _best_budget_only_distribution(
    train_mean_losses: torch.Tensor,
    costs: torch.Tensor,
    budget: float,
) -> torch.Tensor:
    """Solve the one-constraint global route-mixture LP by vertex enumeration."""
    candidates: list[torch.Tensor] = []
    for index, cost in enumerate(costs):
        if float(cost) <= budget + 1e-9:
            value = torch.zeros_like(costs)
            value[index] = 1.0
            candidates.append(value)
    for left in range(len(costs)):
        for right in range(left + 1, len(costs)):
            c0, c1 = float(costs[left]), float(costs[right])
            if abs(c1 - c0) < 1e-12 or not min(c0, c1) <= budget <= max(c0, c1):
                continue
            weight_right = (budget - c0) / (c1 - c0)
            if 0.0 <= weight_right <= 1.0:
                value = torch.zeros_like(costs)
                value[left] = 1.0 - weight_right
                value[right] = weight_right
                candidates.append(value)
    if not candidates:
        value = torch.zeros_like(costs)
        value[int(costs.argmin())] = 1.0
        return value
    objectives = torch.stack(
        [(candidate * train_mean_losses).sum() for candidate in candidates]
    )
    return candidates[int(objectives.argmin())]


def evaluate_matched_baselines(
    train_cache: RouteAnalysisCache,
    valid_cache: RouteAnalysisCache,
    normalized_flops: torch.Tensor,
    policy_report: dict[str, Any],
) -> dict[str, Any]:
    train_mean = train_cache.route_losses.mean(dim=0)
    valid_losses = valid_cache.route_losses
    costs = normalized_flops.cpu()
    output: dict[str, Any] = {}
    for key, policy in policy_report.items():
        matched_cost = float(policy["expected_normalized_FLOPs"])
        feasible = costs <= matched_cost + 1e-7
        fixed_index = int(
            torch.where(feasible, train_mean, torch.full_like(train_mean, float("inf"))).argmin()
        )
        fixed_mae = float(valid_losses[:, fixed_index].mean())
        random_probs = _max_entropy_cost_distribution(costs, matched_cost)
        random_mae = float((valid_losses * random_probs.view(1, -1)).sum(dim=-1).mean())
        budget_only_probs = _best_budget_only_distribution(train_mean, costs, matched_cost)
        budget_only_mae = float(
            (valid_losses * budget_only_probs.view(1, -1)).sum(dim=-1).mean()
        )
        feasible_oracle = torch.where(
            feasible.view(1, -1),
            valid_losses,
            torch.full_like(valid_losses, float("inf")),
        )
        oracle_mae = float(feasible_oracle.min(dim=-1).values.mean())
        policy_mae = float(policy["expected_MAE"])
        denominator = fixed_mae - oracle_mae
        recovery = (
            (fixed_mae - policy_mae) / denominator
            if denominator > 1e-12
            else float("nan")
        )
        output[key] = {
            "matched_cost": matched_cost,
            "best_fixed": {
                "route": ROUTE_NAMES[fixed_index],
                "MAE": fixed_mae,
                "normalized_FLOPs": float(costs[fixed_index]),
                "selection_split": "TRAIN",
            },
            "random_matched_cost": {
                "MAE": random_mae,
                "normalized_FLOPs": float((random_probs * costs).sum()),
                "route_probabilities": {
                    name: float(random_probs[index])
                    for index, name in enumerate(ROUTE_NAMES)
                },
            },
            "budget_only": {
                "MAE": budget_only_mae,
                "normalized_FLOPs": float((budget_only_probs * costs).sum()),
                "route_probabilities": {
                    name: float(budget_only_probs[index])
                    for index, name in enumerate(ROUTE_NAMES)
                },
            },
            "feasible_oracle": {"MAE": oracle_mae},
            "oracle_headroom_recovered": float(recovery),
        }
    return output


class SupervisedRouteBaseline(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(ROUTES)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def train_supervised_baseline(
    train_cache: RouteAnalysisCache,
    valid_cache: RouteAnalysisCache,
    normalized_flops: torch.Tensor,
    budgets: Sequence[float],
    *,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    device: torch.device,
    report_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """TRAIN-only oracle-label classifier, evaluated on VALID under hard budgets."""
    root = PREFIX_INDEX[()]
    model = SupervisedRouteBaseline(train_cache.states.shape[-1] + 1, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    sample_indices = torch.arange(train_cache.states.shape[0]).repeat_interleave(len(budgets))
    budget_indices = torch.arange(len(budgets)).repeat(train_cache.states.shape[0])
    budget_tensor = torch.tensor(budgets)
    costs = normalized_flops.cpu()
    labels_by_budget = []
    for budget in budgets:
        feasible = costs <= float(budget) + 1e-8
        labels_by_budget.append(
            torch.where(
                feasible.view(1, -1),
                train_cache.route_losses,
                torch.full_like(train_cache.route_losses, float("inf")),
            ).argmin(dim=-1)
        )
    labels_by_budget_t = torch.stack(labels_by_budget, dim=1)
    for _ in range(epochs):
        order = torch.randperm(sample_indices.numel())
        for start in range(0, order.numel(), batch_size):
            chosen = order[start : start + batch_size]
            samples = sample_indices[chosen]
            budget_ids = budget_indices[chosen]
            state = train_cache.states[samples, root]
            budget_values = budget_tensor[budget_ids].view(-1, 1)
            features = torch.cat((state, budget_values), dim=-1).to(device)
            labels = labels_by_budget_t[samples, budget_ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(features), labels)
            loss.backward()
            optimizer.step()
    model.eval()
    if report_keys is None:
        report_keys = [f"{budget:.6g}" for budget in budgets]
    if len(report_keys) != len(budgets):
        raise ValueError("supervised baseline report keys must match budgets")
    report = {}
    with torch.inference_mode():
        for report_key, budget in zip(report_keys, budgets):
            values = []
            for start in range(0, valid_cache.states.shape[0], batch_size):
                state = valid_cache.states[start : start + batch_size, root]
                features = torch.cat(
                    (state, torch.full((state.shape[0], 1), float(budget))), dim=-1
                ).to(device)
                logits = model(features).cpu()
                feasible = costs <= float(budget) + 1e-8
                logits[:, ~feasible] = float("-inf")
                values.append(logits.argmax(dim=-1))
            selected = torch.cat(values)
            rows = torch.arange(selected.numel())
            report[str(report_key)] = {
                "MAE": float(valid_cache.route_losses[rows, selected].mean()),
                "normalized_FLOPs": float(costs[selected].mean()),
                "matched_cost_constraint": float(budget),
                "route_histogram": route_histogram_metrics(
                    nn.functional.one_hot(selected, len(ROUTES)).float()
                ),
                "fit_split": "TRAIN",
                "evaluation_split": "VALID",
            }
    return report


def _zr_indices(metadata: dict[str, Any]) -> list[int]:
    groups = metadata.get("feature_groups", {})
    indices: list[int] = []
    if isinstance(groups, dict):
        for name, values in groups.items():
            if str(name).lower().startswith(("zr", "z_r", "current_forecast")):
                indices.extend(int(value) for value in values)
    # Native route-analysis _step_vector contract (D=61):
    # history/evidence occupy [0:18], current Zr/projected/gap/corrections and
    # branch diagnostics occupy [18:58], explicit resolution code is [58:61].
    if not indices and metadata.get("route_feature_shape", [None, None, None])[-1] == 61:
        indices.extend(range(18, 58))
    return sorted(set(indices))


def _mean_train_costs(
    model: ResolutionNativeExactRouter,
    train_cache: RouteAnalysisCache,
    normalized_flops: torch.Tensor,
    budgets: Sequence[float],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    values = []
    with torch.inference_mode():
        for budget in budgets:
            probs, _ = _batch_distribution(
                model, train_cache.states, budget, batch_size, device
            )
            values.append(
                (probs * normalized_flops.cpu().view(1, -1)).sum(dim=-1).mean()
            )
    return torch.stack(values).to(device)


def train(args: argparse.Namespace) -> dict[str, Any]:
    # Gate is checked before output directories, models, optimizers, or caches.
    gate = require_passing_gate(args.gate_report)
    cost_profile = getattr(args, "cost_profile", None)
    external_flops = (
        load_actual_route_flops(cost_profile) if cost_profile is not None else None
    )
    train_cache = load_route_analysis_cache(
        args.train_cache, expected_split="train", route_flops=external_flops
    )
    valid_cache = load_route_analysis_cache(
        args.valid_cache, expected_split="valid", route_flops=external_flops
    )
    if train_cache.states.shape[-1] != valid_cache.states.shape[-1]:
        raise ValueError("TRAIN/VALID router state dimensions differ")
    if not torch.allclose(train_cache.route_flops, valid_cache.route_flops):
        raise ValueError("TRAIN/VALID caches must use one frozen measured FLOPs table")
    train_forecaster = train_cache.metadata.get("forecaster_checkpoint")
    valid_forecaster = valid_cache.metadata.get("forecaster_checkpoint")
    if train_forecaster and valid_forecaster and train_forecaster != valid_forecaster:
        raise ValueError("TRAIN/VALID states and losses came from different forecasters")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = parse_budgets(args.budgets)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    normalized_flops = train_cache.normalized_flops.to(device)
    global_scale = robust_global_route_margin_scale(
        train_cache.route_losses, split="train"
    )
    model = ResolutionNativeExactRouter(
        train_cache.states.shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    initial_reference = copy.deepcopy(model).eval()
    trust_reference = copy.deepcopy(model).eval()
    for parameter in initial_reference.parameters():
        parameter.requires_grad_(False)
    for parameter in trust_reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    dual = PrimalDualBudgetController(
        budgets,
        learning_rate=args.dual_learning_rate,
        max_lambda=args.max_lambda,
        device=device,
    )

    n_samples = train_cache.states.shape[0]
    sample_indices = torch.arange(n_samples).repeat_interleave(len(budgets))
    budget_indices = torch.arange(len(budgets)).repeat(n_samples)
    budget_values = torch.tensor(budgets, dtype=torch.float32)
    history: list[dict[str, Any]] = []
    best = {"selection_score": float("inf"), "epoch": 0}
    best_path = output_dir / "best_valid_policy.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(sample_indices.numel())
        epoch_values: dict[str, list[float]] = {
            key: [] for key in ("loss", "utility", "cost", "kl", "entropy")
        }
        entropy_value = entropy_coefficient(
            epoch - 1,
            args.epochs,
            args.entropy_coefficient,
            anneal_fraction=args.entropy_anneal_fraction,
        )
        for start in range(0, order.numel(), args.batch_size):
            chosen = order[start : start + args.batch_size]
            samples = sample_indices[chosen]
            budget_ids = budget_indices[chosen]
            states = train_cache.states[samples].to(device)
            route_losses = train_cache.route_losses[samples].to(device)
            batch_budgets = budget_values[budget_ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model.terminal_distribution(states, batch_budgets)
            with torch.inference_mode():
                reference_probs = trust_reference.terminal_distribution(
                    states, batch_budgets
                )["route_probs"]
            loss, details = exact_full_information_objective(
                output["route_probs"],
                route_losses,
                normalized_flops,
                batch_budgets,
                dual.values_for(budget_ids.to(device)),
                global_margin_scale=global_scale,
                reference_probs=reference_probs,
                kl_coefficient=args.kl_coefficient,
                entropy_coefficient_value=entropy_value,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite policy loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            epoch_values["loss"].append(float(loss.detach()))
            epoch_values["utility"].append(float(details["expected_utility"].detach()))
            epoch_values["cost"].append(float(details["expected_cost"].detach()))
            epoch_values["kl"].append(float(details["kl"].detach()))
            epoch_values["entropy"].append(float(details["entropy"].detach()))

        train_costs = _mean_train_costs(
            model,
            train_cache,
            normalized_flops,
            budgets,
            args.eval_batch_size,
            device,
        )
        dual.update(train_costs)
        valid = evaluate_policy(
            model,
            valid_cache,
            normalized_flops,
            budgets,
            batch_size=args.eval_batch_size,
            device=device,
            reference_model=initial_reference,
        )
        selection_score = float(
            np.mean(
                [
                    values["expected_MAE"]
                    + args.selection_violation_penalty * values["cost_violation"]
                    for values in valid.values()
                ]
            )
        )
        improved = selection_score < best["selection_score"]
        if improved:
            best = {"selection_score": selection_score, "epoch": epoch}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_args": {
                        "state_dim": train_cache.states.shape[-1],
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                    },
                    "budgets": budgets,
                    "global_train_route_margin_scale": global_scale,
                    "actual_route_flops": train_cache.route_flops,
                    "normalized_route_flops": normalized_flops.cpu(),
                    "dual_state": dual.state_dict(),
                    "selection": best,
                    "routes": ROUTES,
                    "gate_report": str(args.gate_report),
                    "TEST_loaded": False,
                },
                best_path,
            )
        row = {
            "epoch": epoch,
            **{
                key: float(np.mean(values)) for key, values in epoch_values.items()
            },
            "entropy_coefficient": entropy_value,
            "dual_lambdas": dual.lambdas.detach().cpu().tolist(),
            "train_expected_costs": train_costs.detach().cpu().tolist(),
            "valid_selection_score": selection_score,
            "improved": improved,
        }
        history.append(row)
        dump_json(output_dir / "training_history.json", history)
        print(
            f"[route-policy] epoch={epoch:03d} loss={row['loss']:.5f} "
            f"VALID_score={selection_score:.5f} "
            f"lambda={row['dual_lambdas']}",
            flush=True,
        )
        if epoch % args.reference_update_epochs == 0:
            trust_reference.load_state_dict(model.state_dict())

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    valid = evaluate_policy(
        model,
        valid_cache,
        normalized_flops,
        budgets,
        batch_size=args.eval_batch_size,
        device=device,
        reference_model=initial_reference,
    )
    baselines = evaluate_matched_baselines(
        train_cache, valid_cache, normalized_flops.cpu(), valid
    )
    valid_keys = list(valid)
    matched_supervised_budgets = [
        float(valid[key]["expected_normalized_FLOPs"]) for key in valid_keys
    ]
    supervised = train_supervised_baseline(
        train_cache,
        valid_cache,
        normalized_flops.cpu(),
        matched_supervised_budgets,
        epochs=args.supervised_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        device=device,
        report_keys=valid_keys,
    )
    generator = torch.Generator().manual_seed(args.seed + 991)
    shuffled = negative_control_states(
        valid_cache.states, "shuffle", generator=generator
    )
    controls: dict[str, Any] = {
        "shuffle_state": evaluate_policy(
            model,
            valid_cache,
            normalized_flops,
            budgets,
            batch_size=args.eval_batch_size,
            device=device,
            states_override=shuffled,
        )
    }
    zr_indices = _zr_indices(valid_cache.metadata)
    if zr_indices:
        controls["no_Zr"] = evaluate_policy(
            model,
            valid_cache,
            normalized_flops,
            budgets,
            batch_size=args.eval_batch_size,
            device=device,
            states_override=negative_control_states(
                valid_cache.states, "no_zr", zr_feature_indices=zr_indices
            ),
        )
    else:
        controls["no_Zr"] = {
            "executed": False,
            "reason": "cache metadata lacks feature_groups entries beginning with Zr",
        }
    report = {
        "method": "ResolutionNativeRouteCompleteExactFullInformationPolicy",
        "routes": [list(route) for route in ROUTES],
        "gate": gate,
        "cache_contract": {
            "train": str(args.train_cache),
            "valid": str(args.valid_cache),
            "state_shape_train": list(train_cache.states.shape),
            "state_shape_valid": list(valid_cache.states.shape),
            "forecaster_frozen": True,
            "target_features_allowed": False,
        },
        "costs": {
            "unit": "actual profiler FLOPs",
            "profile": None if cost_profile is None else str(cost_profile),
            "raw": train_cache.route_flops.tolist(),
            "normalized": normalized_flops.detach().cpu().tolist(),
            "budgets": budgets,
            "constraint": "expected normalized FLOPs",
            "dual_lambdas_at_selected_checkpoint": checkpoint["dual_state"]["lambdas"].tolist(),
        },
        "objective": {
            "group": "same sample and same budget, all eight routes",
            "global_TRAIN_route_margin_scale": global_scale,
            "group_centered": True,
            "group_std_division": False,
            "exact_expected_trajectory_utility": True,
            "trajectory_distribution_KL_coefficient": args.kl_coefficient,
            "early_entropy_annealing": True,
        },
        "selection": {
            **best,
            "split": "VALID",
            "checkpoint": str(best_path),
            "TEST_loaded": False,
        },
        "valid_policy": valid,
        "matched_cost_baselines": baselines,
        "supervised_route_baseline": supervised,
        "negative_controls": controls,
        "anti_collapse_fields": [
            "route_histogram",
            "effective_routes",
            "trajectory_KL_from_initial_policy",
            "cost_violation",
            "oracle_headroom_recovered",
        ],
    }
    dump_json(output_dir / "final_valid_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--valid-cache", required=True)
    parser.add_argument("--gate-report", required=True)
    parser.add_argument(
        "--cost-profile",
        help="route-analysis cost_profile.json; optional only when caches embed route_flops",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--supervised-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dual-learning-rate", type=float, default=0.1)
    parser.add_argument("--max-lambda", type=float, default=100.0)
    parser.add_argument("--kl-coefficient", type=float, default=0.02)
    parser.add_argument("--reference-update-epochs", type=int, default=2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--entropy-anneal-fraction", type=float, default=0.25)
    parser.add_argument("--selection-violation-penalty", type=float, default=10.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.supervised_epochs = min(args.supervised_epochs, 2)
        args.batch_size = min(args.batch_size, 32)
        args.eval_batch_size = min(args.eval_batch_size, 64)
    if args.reference_update_epochs <= 0:
        parser.error("--reference-update-epochs must be positive")
    return args


def main() -> int:
    args = parse_args()
    seed_all(args.seed)
    try:
        report = train(args)
    except RuntimeError as error:
        if "ROUTE POLICY NOT TRAINED" in str(error):
            print(str(error), file=sys.stderr, flush=True)
            return 2
        raise
    print(
        f"[route-policy] complete: {Path(args.output_dir) / 'final_valid_report.json'} "
        f"(best VALID epoch {report['selection']['epoch']}; TEST not loaded)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
