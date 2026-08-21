#!/usr/bin/env python3
"""TRAIN/VALID-only constrained GRPO pilot for ResolutionNative post-Z3 routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1_shared_prefix import (
    F2FCoTResolutionNativeV1SharedPrefixNet,
)
from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.post_z3_constrained_grpo import (
    LONG,
    SHORT,
    BudgetDualPanel,
    FrozenPostZ3Environment,
    PostZ3BudgetRouter,
    PostZ3Observation,
    bernoulli_log_prob,
    clipped_trajectory_grpo_loss,
    leave_one_out_advantages,
)
from scripts.f2f_cot_resolution_native_v1_experiment import model_args
from scripts.f2f_cot_runtime import load_rescale, make_loader, per_sample_mae, select_batch


EXPERIMENT = "f2f_cot_resolution_native_post_z3_grpo"
FROZEN_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "PEMS04"
    / "H12"
    / "f2f_cot_resolution_native_v1_shared_prefix"
    / "formal_v1_seed1"
    / "shared_prefix_best.pt"
)
SOURCE_COST_REPORT = (
    ROOT
    / "results"
    / "f2f_cot_resolution_native_v1_shared_prefix"
    / "formal_v1_seed1"
    / "shared_prefix_report.json"
)
DEFAULT_BUDGETS = (0.0, 0.25, 0.5, 0.75, 1.0)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_budgets(text: str) -> tuple[float, ...]:
    values = tuple(sorted({float(item.strip()) for item in text.split(",") if item.strip()}))
    if len(values) < 3 or values[0] != 0.0 or values[-1] != 1.0:
        raise ValueError("budget panel must include 0, 1, and at least one interior value")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("budgets are normalized extra-compute fractions in [0,1]")
    return values


def load_frozen_environment(device: torch.device):
    if not FROZEN_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing frozen shared-prefix checkpoint: {FROZEN_CHECKPOINT}")
    checkpoint = torch.load(FROZEN_CHECKPOINT, map_location="cpu", weights_only=False)
    model = F2FCoTResolutionNativeV1SharedPrefixNet(**model_args()).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    environment = FrozenPostZ3Environment(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("forecasting environment was not frozen")
    return model, environment, checkpoint


def group_per_sample_mae(prediction_raw: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    """Return physical-scale MAE for predictions ``[B,G,T,N,1]``."""
    target = target_raw[:, None]
    mask = ~torch.isclose(
        target,
        torch.zeros((), device=target.device, dtype=target.dtype),
        atol=5e-5,
        rtol=0.0,
    )
    error = (prediction_raw - target).abs() * mask
    return error.sum(dim=(2, 3, 4)) / mask.sum(dim=(2, 3, 4)).clamp_min(1)


def balanced_budget_indices(
    batch: int, panel_size: int, epoch: int, batch_index: int, device: torch.device
) -> torch.Tensor:
    offset = (epoch * 104729 + batch_index * max(batch, 1)) % panel_size
    values = (torch.arange(batch, device=device) + offset) % panel_size
    if batch > 1:
        values = values[torch.randperm(batch, device=device)]
    return values.long()


def behavior_log_prob(probability: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    expanded = probability[:, None].expand_as(actions).clamp(1e-8, 1.0 - 1e-8)
    return torch.where(actions.bool(), expanded.log(), (1.0 - expanded).log())


def _ablate_z3(observation: PostZ3Observation) -> PostZ3Observation:
    ablated = observation.detached()
    if ablated.state.latest_forecast is None:
        raise RuntimeError("Z3 ablation requires a reached forecast")
    zero = torch.zeros_like(ablated.state.latest_forecast)
    ablated.state.latest_forecast = zero
    ablated.state.forecasts = (zero,)
    diagnostics = dict(ablated.diagnostics)
    for key in ("raw_correction", "low_frequency_correction", "detail_correction"):
        value = diagnostics.get(key)
        if torch.is_tensor(value):
            diagnostics[key] = torch.zeros_like(value)
    ablated.diagnostics = diagnostics
    return ablated


@torch.inference_mode()
def collect_panel(
    environment: FrozenPostZ3Environment,
    policy: PostZ3BudgetRouter,
    loader,
    budgets: Sequence[float],
    device: torch.device,
    rescale,
    *,
    max_batches: int | None = None,
    control: str = "full",
    seed: int = 1,
) -> dict[str, np.ndarray]:
    """Execute both real branches and collect policy scores without target leakage."""
    environment.freeze_forecaster()
    policy.eval()
    chunks: dict[str, list[np.ndarray]] = {
        "indices": [],
        "short_mae": [],
        "long_mae": [],
        "logits": [],
        "probabilities": [],
    }
    batch_short, batch_long = [], []
    generator = torch.Generator(device=device).manual_seed(seed)
    budget_tensor_template = torch.tensor(budgets, device=device)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target, indices = select_batch(batch, device)
        observation = environment.begin(history)
        pair = environment.forced_pair(observation)
        target_raw = rescale(target)
        losses = group_per_sample_mae(rescale(pair), target_raw)
        score_observation = observation
        if control == "shuffle":
            order = torch.randperm(history.shape[0], generator=generator, device=device)
            score_observation = observation.index_select(order)
        elif control == "no_z3":
            score_observation = _ablate_z3(observation)
        elif control != "full":
            raise ValueError(f"unknown state control: {control}")
        logits = []
        for budget in budget_tensor_template:
            values = torch.full((history.shape[0],), float(budget), device=device)
            logits.append(policy(score_observation, values))
        logits_t = torch.stack(logits, dim=1)
        chunks["indices"].append(indices.numpy())
        chunks["short_mae"].append(losses[:, SHORT].cpu().numpy())
        chunks["long_mae"].append(losses[:, LONG].cpu().numpy())
        chunks["logits"].append(logits_t.cpu().numpy())
        chunks["probabilities"].append(torch.sigmoid(logits_t).cpu().numpy())
        batch_short.append(float(per_sample_mae(rescale(pair[:, SHORT]), target_raw).mean()))
        batch_long.append(float(per_sample_mae(rescale(pair[:, LONG]), target_raw).mean()))
    if not chunks["indices"]:
        raise RuntimeError("panel collection received no batches")
    arrays = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    order = np.argsort(arrays["indices"], kind="stable")
    arrays = {key: value[order] for key, value in arrays.items()}
    arrays["batch_mean_short_mae"] = np.asarray([np.mean(batch_short)])
    arrays["batch_mean_long_mae"] = np.asarray([np.mean(batch_long)])
    return arrays


def robust_margin_scale(short_mae: np.ndarray, long_mae: np.ndarray) -> float:
    margins = np.abs(short_mae.astype(np.float64) - long_mae.astype(np.float64))
    margins = margins[np.isfinite(margins) & (margins > 1e-8)]
    return max(float(np.median(margins)) if len(margins) else 1.0, 1e-6)


def expected_panel_report(
    arrays: Mapping[str, np.ndarray], budgets: Sequence[float], margin_scale: float
) -> dict[str, Any]:
    short = arrays["short_mae"].astype(np.float64)
    long = arrays["long_mae"].astype(np.float64)
    probabilities = arrays["probabilities"].astype(np.float64)
    mean_short, mean_long = float(short.mean()), float(long.mean())
    report = {}
    selection_terms = []
    violations = []
    for index, budget in enumerate(budgets):
        probability = probabilities[:, index]
        share = float(probability.mean())
        expected = float(((1.0 - probability) * short + probability * long).mean())
        fixed = (1.0 - share) * mean_short + share * mean_long
        improvement = fixed - expected
        violation = max(0.0, share - float(budget))
        if 0.0 < budget < 1.0:
            selection_terms.append(improvement / margin_scale)
            violations.append(violation)
        report[f"{budget:.6g}"] = {
            "budget_extra_compute_fraction": float(budget),
            "expected_long_share": share,
            "expected_per_sample_MAE": expected,
            "matched_fixed_mixture_MAE": fixed,
            "gain_over_matched_fixed_mixture": improvement,
            "constraint_violation": violation,
        }
    score = float(np.mean(selection_terms) - 10.0 * np.mean(violations))
    return {
        "budgets": report,
        "selection_score": score,
        "maximum_constraint_violation": float(max(violations, default=0.0)),
        "always_short_per_sample_MAE": mean_short,
        "always_long_per_sample_MAE": mean_long,
    }


def fit_train_thresholds(
    arrays: Mapping[str, np.ndarray], budgets: Sequence[float]
) -> dict[str, dict[str, float | int]]:
    """TRAIN-only threshold calibration under an upper expected-cost budget."""
    gain = arrays["short_mae"].astype(np.float64) - arrays["long_mae"].astype(np.float64)
    scores = arrays["logits"].astype(np.float64)
    count = len(gain)
    output = {}
    for index, budget in enumerate(budgets):
        order = np.argsort(-scores[:, index], kind="stable")
        maximum = min(count, int(math.floor(float(budget) * count + 1e-9)))
        cumulative = np.concatenate(([0.0], np.cumsum(gain[order[:maximum]])))
        chosen_count = int(np.argmax(cumulative))
        if chosen_count == 0:
            threshold = math.inf
        elif chosen_count == count:
            threshold = -math.inf
        else:
            upper = scores[order[chosen_count - 1], index]
            lower = scores[order[chosen_count], index]
            threshold = float(0.5 * (upper + lower))
        output[f"{budget:.6g}"] = {
            "budget": float(budget),
            "threshold": float(threshold),
            "selected_long_count": chosen_count,
            "selected_long_share": chosen_count / count,
            "TRAIN_cumulative_MAE_gain": float(cumulative[chosen_count] / count),
        }
    return output


def globally_shuffle_scores(
    arrays: Mapping[str, np.ndarray], seed: int
) -> dict[str, np.ndarray]:
    """Destroy sample/state alignment while retaining losses and score marginals."""
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(arrays["short_mae"]))
    output = {key: np.asarray(value).copy() for key, value in arrays.items()}
    output["logits"] = np.asarray(arrays["logits"])[order].copy()
    output["probabilities"] = np.asarray(arrays["probabilities"])[order].copy()
    return output


def _oracle_selection(gain: np.ndarray, maximum: int, *, exact: bool) -> np.ndarray:
    order = np.argsort(-gain, kind="stable")
    if exact:
        count = maximum
    else:
        cumulative = np.concatenate(([0.0], np.cumsum(gain[order[:maximum]])))
        count = int(np.argmax(cumulative))
    selected = np.zeros(len(gain), dtype=bool)
    selected[order[:count]] = True
    return selected


def hard_panel_report(
    arrays: Mapping[str, np.ndarray],
    budgets: Sequence[float],
    thresholds: Mapping[str, Mapping[str, float | int]],
    cost: Mapping[str, float],
) -> dict[str, Any]:
    short = arrays["short_mae"].astype(np.float64)
    long = arrays["long_mae"].astype(np.float64)
    gain = short - long
    scores = arrays["logits"].astype(np.float64)
    mean_short, mean_long = float(short.mean()), float(long.mean())
    rows = {}
    for index, budget in enumerate(budgets):
        threshold = float(thresholds[f"{budget:.6g}"]["threshold"])
        selected = scores[:, index] >= threshold
        maximum = int(math.floor(float(budget) * len(gain) + 1e-9))
        if int(selected.sum()) > maximum:
            # Average-budget deployment may rank a cohort by policy score.  The
            # cap is target-free and also makes tied scores obey the hard ceiling.
            selected.fill(False)
            selected[np.argsort(-scores[:, index], kind="stable")[:maximum]] = True
        share = float(selected.mean())
        chosen = np.where(selected, long, short)
        fixed = (1.0 - share) * mean_short + share * mean_long
        exact_oracle = _oracle_selection(gain, int(selected.sum()), exact=True)
        budget_oracle = _oracle_selection(
            gain, maximum, exact=False
        )
        exact_oracle_mae = float(np.where(exact_oracle, long, short).mean())
        budget_oracle_mae = float(np.where(budget_oracle, long, short).mean())
        bootstrap = []
        generator = np.random.default_rng(1709 + index)
        for _ in range(1000):
            sampled = generator.integers(0, len(gain), size=len(gain))
            sampled_selected = selected[sampled]
            sampled_share = float(sampled_selected.mean())
            sampled_short = short[sampled]
            sampled_long = long[sampled]
            sampled_policy = np.where(
                sampled_selected, sampled_long, sampled_short
            ).mean()
            sampled_fixed = (
                (1.0 - sampled_share) * sampled_short.mean()
                + sampled_share * sampled_long.mean()
            )
            bootstrap.append(float(sampled_fixed - sampled_policy))
        gain_ci = np.percentile(bootstrap, (2.5, 97.5)).tolist()
        selected_gain = gain[selected]
        rejected_gain = gain[~selected]
        normalized_flops = (1.0 - share) * cost["short_flops"] + share * cost["long_flops"]
        latency = (1.0 - share) * cost["short_ms"] + share * cost["long_ms"]
        rows[f"{budget:.6g}"] = {
            "budget_extra_compute_fraction": float(budget),
            "long_count": int(selected.sum()),
            "long_share": share,
            "per_sample_MAE": float(chosen.mean()),
            "gain_over_always_short": mean_short - float(chosen.mean()),
            "gain_over_always_long": mean_long - float(chosen.mean()),
            "matched_fixed_mixture_MAE": fixed,
            "gain_over_matched_fixed_mixture": fixed - float(chosen.mean()),
            "gain_over_matched_fixed_mixture_bootstrap_95pct_CI": gain_ci,
            "selected_long_mean_realized_gain": (
                float(selected_gain.mean()) if len(selected_gain) else None
            ),
            "selected_short_mean_realized_gain": (
                float(rejected_gain.mean()) if len(rejected_gain) else None
            ),
            "realized_gain_separation_long_minus_short": (
                float(selected_gain.mean() - rejected_gain.mean())
                if len(selected_gain) and len(rejected_gain)
                else None
            ),
            "matched_share_oracle_MAE": exact_oracle_mae,
            "matched_share_oracle_headroom": fixed - exact_oracle_mae,
            "fraction_of_matched_oracle_headroom_recovered": (
                (fixed - float(chosen.mean())) / (fixed - exact_oracle_mae)
                if fixed - exact_oracle_mae > 1e-12
                else 0.0
            ),
            "budget_ceiling_oracle_MAE": budget_oracle_mae,
            "budget_ceiling_oracle_long_share": float(budget_oracle.mean()),
            "forecast_route_FLOPs": normalized_flops,
            "forecast_route_latency_ms": latency,
        }
    return {
        "always_short": {
            "per_sample_MAE": mean_short,
            "FLOPs": cost["short_flops"],
            "latency_ms": cost["short_ms"],
        },
        "always_long": {
            "per_sample_MAE": mean_long,
            "FLOPs": cost["long_flops"],
            "latency_ms": cost["long_ms"],
        },
        "budgets": rows,
    }


def fixed_costs_from_source() -> dict[str, float]:
    report = json.loads(SOURCE_COST_REPORT.read_text(encoding="utf-8"))
    cost = report["cost"]
    return {
        "short_flops": float(cost["flops"]["short_total_X_Z3_Z12"]),
        "long_flops": float(cost["flops"]["long_total_X_Z3_Z6_Z12"]),
        "extra_flops": float(cost["flops"]["extra_z6_path_flops"]),
        "short_ms": float(cost["latency_ms"]["short_total_prefix_plus_continuation_median_ms"]),
        "long_ms": float(cost["latency_ms"]["long_total_prefix_plus_continuation_median_ms"]),
        "extra_ms": float(cost["latency_ms"]["extra_z6_path_median_ms"]),
        "source": str(SOURCE_COST_REPORT),
    }


def save_router_checkpoint(
    path: Path,
    policy: PostZ3BudgetRouter,
    optimizer: torch.optim.Optimizer,
    dual: BudgetDualPanel,
    epoch: int,
    best: Mapping[str, object],
    args,
    frozen_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dual_state_dict": dual.state_dict(),
            "epoch": int(epoch),
            "best": dict(best),
            "budgets": list(args.budget_values),
            "frozen_forecaster_checkpoint": str(FROZEN_CHECKPOINT),
            "frozen_forecaster_sha256": frozen_hash,
            "method": "PostZ3ConstrainedTrajectoryGRPO_RLOO",
            "uses_TEST": False,
        },
        path,
    )


def train_epoch(
    environment: FrozenPostZ3Environment,
    policy: PostZ3BudgetRouter,
    loader,
    optimizer,
    dual: BudgetDualPanel,
    budgets: torch.Tensor,
    margin_scale: float,
    device: torch.device,
    rescale,
    epoch: int,
    args,
) -> dict[str, float]:
    policy.train()
    metrics: dict[str, list[float]] = {
        key: []
        for key in (
            "policy_loss",
            "raw_sampled_MAE",
            "advantage_abs",
            "entropy",
            "kl",
            "clip_fraction",
            "long_share",
            "gradient_norm",
        )
    }
    for batch_index, batch in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        history, target, _ = select_batch(batch, device)
        observation = environment.begin(history)
        budget_indices = balanced_budget_indices(
            history.shape[0], len(budgets), epoch, batch_index, device
        )
        sample_budgets = budgets.index_select(0, budget_indices)
        with torch.no_grad():
            old_logits = policy(observation, sample_budgets)
            old_probability = torch.sigmoid(old_logits)
            behavior_probability = (
                (1.0 - args.exploration_mix) * old_probability
                + args.exploration_mix * 0.5
            )
            actions = torch.bernoulli(
                behavior_probability[:, None].expand(-1, args.group_size)
            ).long()
            old_behavior_log_prob = behavior_log_prob(behavior_probability, actions)
            prediction = environment.continue_actions(observation, actions)
            route_losses = group_per_sample_mae(rescale(prediction), rescale(target))
            lambdas = dual.values(budget_indices, device)
            returns = (
                -route_losses / margin_scale
                - lambdas[:, None]
                * (actions.to(route_losses.dtype) - sample_budgets[:, None])
            )
            advantages = leave_one_out_advantages(returns)

        for _ in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            current_logits = policy(observation, sample_budgets)
            loss, details = clipped_trajectory_grpo_loss(
                current_logits,
                actions,
                old_behavior_log_prob,
                advantages,
                old_probability,
                clip_ratio=args.clip_ratio,
                entropy_coefficient=args.entropy_coefficient,
                kl_coefficient=args.kl_coefficient,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite GRPO loss at epoch={epoch} batch={batch_index}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
        dual.update(budget_indices, old_probability)
        metrics["policy_loss"].append(float(loss.detach()))
        metrics["raw_sampled_MAE"].append(float(route_losses.mean()))
        metrics["advantage_abs"].append(float(advantages.abs().mean()))
        metrics["entropy"].append(float(details["entropy"].detach()))
        metrics["kl"].append(float(details["kl"].detach()))
        metrics["clip_fraction"].append(float(details["clip_fraction"].detach()))
        metrics["long_share"].append(float(actions.float().mean()))
        metrics["gradient_norm"].append(float(gradient_norm))
    if not metrics["policy_loss"]:
        raise RuntimeError("training epoch received no batches")
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.mean(values)),
        "p90_ms": float(np.percentile(values, 90)),
    }


@torch.inference_mode()
def profile_router_overhead(
    environment: FrozenPostZ3Environment,
    policy: PostZ3BudgetRouter,
    example: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if example.device.type != "cuda":
        return {"available": False, "reason": "CUDA required"}
    policy.eval()
    budget = torch.full((1,), 0.5, device=example.device)

    def timed(function):
        for _ in range(warmup):
            function()
        torch.cuda.synchronize(example.device)
        values = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
        return _latency_summary(values)

    observation = environment.begin(example)

    def actor_step():
        return policy(observation, budget)

    def adaptive(action: int):
        obs = environment.begin(example)
        policy(obs, budget)
        actions = torch.full((1,), action, device=example.device, dtype=torch.long)
        return environment.continue_actions(obs, actions)

    def fixed(action: int):
        route = (3, 12) if action == SHORT else (3, 6, 12)
        return environment.forecaster.rollout(example, route)

    def paired_timed(action: int):
        functions = {
            "fixed": lambda: fixed(action),
            "adaptive": lambda: adaptive(action),
        }
        for _ in range(warmup):
            functions["fixed"]()
            functions["adaptive"]()
        torch.cuda.synchronize(example.device)
        values = {"fixed": [], "adaptive": []}
        for repeat in range(repeats):
            order = ("fixed", "adaptive") if repeat % 2 == 0 else ("adaptive", "fixed")
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                functions[name]()
                end.record()
                end.synchronize()
                values[name].append(float(start.elapsed_time(end)))
        return {name: _latency_summary(samples) for name, samples in values.items()}

    def profiled_flops(function) -> int:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if example.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities, record_shapes=True, with_flops=True
        ) as profiler:
            function()
            torch.cuda.synchronize(example.device)
        return int(
            sum(
                event.flops
                for event in profiler.key_averages()
                if event.flops is not None
            )
        )

    paired_short = paired_timed(SHORT)
    paired_long = paired_timed(LONG)
    return {
        "available": True,
        "device": str(example.device),
        "batch_size": 1,
        "warmup": warmup,
        "repeats": repeats,
        "policy_step": timed(actor_step),
        "adaptive_forced_short_total": timed(lambda: adaptive(SHORT)),
        "adaptive_forced_long_total": timed(lambda: adaptive(LONG)),
        "paired_fixed_vs_adaptive": {
            "short": paired_short,
            "long": paired_long,
        },
        "profiler_supported_FLOPs": {
            "policy_step": profiled_flops(actor_step),
            "fixed_short_total": profiled_flops(lambda: fixed(SHORT)),
            "fixed_long_total": profiled_flops(lambda: fixed(LONG)),
            "adaptive_forced_short_total": profiled_flops(lambda: adaptive(SHORT)),
            "adaptive_forced_long_total": profiled_flops(lambda: adaptive(LONG)),
        },
        "includes_policy_overhead": True,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tag", default="formal_v1")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dual-learning-rate", type=float, default=0.08)
    parser.add_argument("--exploration-mix", type=float, default=0.10)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--entropy-coefficient", type=float, default=0.005)
    parser.add_argument("--kl-coefficient", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-valid-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.budget_values = parse_budgets(args.budgets)
    if args.smoke:
        args.tag = "smoke"
        args.epochs = 2
        args.batch_size = min(args.batch_size, 4)
        args.group_size = 4
        args.workers = 0
        args.max_train_batches = 2
        args.max_valid_batches = 2
        args.patience = 2
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    result_dir = ROOT / "results" / EXPERIMENT / f"{args.tag}_seed{args.seed}"
    checkpoint_dir = (
        ROOT / "checkpoints" / "PEMS04" / "H12" / EXPERIMENT / f"{args.tag}_seed{args.seed}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    frozen_hash_before = sha256(FROZEN_CHECKPOINT)
    forecaster, environment, source_checkpoint = load_frozen_environment(device)
    policy = PostZ3BudgetRouter(node_size=forecaster.node_size).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    budget_tensor = torch.tensor(args.budget_values, device=device)
    dual = BudgetDualPanel(
        budget_tensor.cpu(), learning_rate=args.dual_learning_rate, maximum=100.0
    )
    rescale = load_rescale()
    train_loader = make_loader("train", args.batch_size, True, args.workers)
    train_eval_loader = make_loader("train", args.batch_size, False, args.workers)
    valid_loader = make_loader("valid", args.batch_size, False, args.workers)

    protocol = {
        "method": "post-Z3 constrained trajectory-level GRPO/RLOO",
        "environment": "X->Z3->{Z12,Z6->Z12}",
        "frozen_forecaster": str(FROZEN_CHECKPOINT),
        "frozen_forecaster_sha256": frozen_hash_before,
        "frozen_forecaster_epoch": int(source_checkpoint.get("epoch", -1)),
        "forecaster_parameters_trainable": 0,
        "policy_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "budgets": list(args.budget_values),
        "cost_definition": "normalized extra compute: short=0, long=1; backed by actual profiler FLOPs",
        "reward": "-physical_per_sample_MAE/TRAIN_robust_margin_scale - dual*(cost-budget)",
        "advantage": "leave-one-out group relative; no group standard-deviation normalization",
        "importance_ratio": "complete one-decision trajectory probability / exploration-mixture behavior probability",
        "test_constructed": False,
    }
    dump_json(result_dir / "protocol.json", protocol)

    print("[calibration] executing real TRAIN/VALID branch pairs", flush=True)
    initial_train = collect_panel(
        environment,
        policy,
        train_eval_loader,
        args.budget_values,
        device,
        rescale,
        max_batches=args.max_train_batches if args.smoke else None,
    )
    margin_scale = robust_margin_scale(initial_train["short_mae"], initial_train["long_mae"])
    protocol["TRAIN_robust_pairwise_margin_scale_MAE"] = margin_scale
    dump_json(result_dir / "protocol.json", protocol)
    initial_valid = collect_panel(
        environment,
        policy,
        valid_loader,
        args.budget_values,
        device,
        rescale,
        max_batches=args.max_valid_batches,
    )
    initial_report = {
        "TRAIN": expected_panel_report(initial_train, args.budget_values, margin_scale),
        "VALID": expected_panel_report(initial_valid, args.budget_values, margin_scale),
    }
    dump_json(result_dir / "initial_policy_report.json", initial_report)

    best = {"selection_score": -math.inf, "epoch": 0, "maximum_constraint_violation": math.inf}
    best_path = checkpoint_dir / "router_best.pt"
    last_path = checkpoint_dir / "router_last.pt"
    history_rows = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = train_epoch(
            environment,
            policy,
            train_loader,
            optimizer,
            dual,
            budget_tensor,
            margin_scale,
            device,
            rescale,
            epoch,
            args,
        )
        valid_arrays = collect_panel(
            environment,
            policy,
            valid_loader,
            args.budget_values,
            device,
            rescale,
            max_batches=args.max_valid_batches,
            seed=args.seed + epoch,
        )
        valid_report = expected_panel_report(valid_arrays, args.budget_values, margin_scale)
        score = float(valid_report["selection_score"])
        constraint = float(valid_report["maximum_constraint_violation"])
        eligible = constraint <= (0.10 if args.smoke else 0.03)
        improved = eligible and score > float(best["selection_score"])
        if improved:
            best = {
                "selection_score": score,
                "epoch": epoch,
                "maximum_constraint_violation": constraint,
            }
            save_router_checkpoint(
                best_path, policy, optimizer, dual, epoch, best, args, frozen_hash_before
            )
            stale = 0
        else:
            stale += 1
        save_router_checkpoint(
            last_path, policy, optimizer, dual, epoch, best, args, frozen_hash_before
        )
        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "train": train_metrics,
            "valid": valid_report,
            "dual_lambdas": dual.lambdas.tolist(),
            "eligible": eligible,
            "improved": improved,
        }
        history_rows.append(row)
        dump_json(result_dir / "training_history.json", history_rows)
        print(
            f"[post-z3-grpo] epoch={epoch:02d} score={score:+.5f} "
            f"violation={constraint:.4f} entropy={train_metrics['entropy']:.4f} "
            f"sampled_long={train_metrics['long_share']:.3f} "
            f"dual={[round(value, 3) for value in dual.lambdas.tolist()]} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
        if epoch >= 5 and stale >= args.patience:
            print(f"[post-z3-grpo] early stop after {stale} stale epochs", flush=True)
            break
    if not best_path.is_file():
        raise RuntimeError("no GRPO checkpoint satisfied VALID compute constraints")

    selected = torch.load(best_path, map_location=device, weights_only=False)
    policy.load_state_dict(selected["policy_state_dict"], strict=True)
    dual.load_state_dict(selected["dual_state_dict"])
    policy.eval()
    print(f"[selected] epoch={selected['epoch']} score={selected['best']['selection_score']:+.5f}")

    train_arrays = collect_panel(
        environment,
        policy,
        train_eval_loader,
        args.budget_values,
        device,
        rescale,
        max_batches=args.max_train_batches if args.smoke else None,
        seed=args.seed + 701,
    )
    valid_arrays = collect_panel(
        environment,
        policy,
        valid_loader,
        args.budget_values,
        device,
        rescale,
        max_batches=args.max_valid_batches,
        seed=args.seed + 702,
    )
    thresholds = fit_train_thresholds(train_arrays, args.budget_values)
    fixed_cost = fixed_costs_from_source()
    full_hard = hard_panel_report(
        valid_arrays, args.budget_values, thresholds, fixed_cost
    )

    controls = {}
    for control_index, control in enumerate(("shuffle", "no_z3")):
        if control == "shuffle":
            controlled_train = globally_shuffle_scores(
                train_arrays, args.seed + 800 + control_index
            )
            controlled_valid = globally_shuffle_scores(
                valid_arrays, args.seed + 900 + control_index
            )
        else:
            controlled_train = collect_panel(
                environment,
                policy,
                train_eval_loader,
                args.budget_values,
                device,
                rescale,
                max_batches=args.max_train_batches if args.smoke else None,
                control=control,
                seed=args.seed + 800 + control_index,
            )
            controlled_valid = collect_panel(
                environment,
                policy,
                valid_loader,
                args.budget_values,
                device,
                rescale,
                max_batches=args.max_valid_batches,
                control=control,
                seed=args.seed + 900 + control_index,
            )
        controlled_thresholds = fit_train_thresholds(controlled_train, args.budget_values)
        controls[control] = hard_panel_report(
            controlled_valid, args.budget_values, controlled_thresholds, fixed_cost
        )

    example = select_batch(next(iter(valid_loader)), device)[0][:1]
    overhead = profile_router_overhead(
        environment,
        policy,
        example,
        warmup=3 if args.smoke else 10,
        repeats=5 if args.smoke else 50,
    )
    if overhead.get("available"):
        actor_ms = float(overhead["policy_step"]["median_ms"])
        for row in full_hard["budgets"].values():
            row["estimated_latency_including_policy_ms"] = row["forecast_route_latency_ms"] + actor_ms

    interior = [row for key, row in full_hard["budgets"].items() if key not in {"0", "1"}]
    nontrivial_shares = [row["long_share"] for row in interior]
    adaptive_gains = [row["gain_over_matched_fixed_mixture"] for row in interior]
    control_gains = {
        name: [
            report["budgets"][key]["gain_over_matched_fixed_mixture"]
            for key in report["budgets"]
            if key not in {"0", "1"}
        ]
        for name, report in controls.items()
    }
    verdict = {
        "collapsed_to_one_route_across_interior_budgets": bool(
            all(share <= 1e-6 for share in nontrivial_shares)
            or all(share >= 1.0 - 1e-6 for share in nontrivial_shares)
        ),
        "uses_both_routes_at_any_interior_budget": bool(
            any(1e-3 < share < 1.0 - 1e-3 for share in nontrivial_shares)
        ),
        "mean_gain_over_matched_fixed_mixture": float(np.mean(adaptive_gains)),
        "full_state_beats_shuffle_mean": bool(
            np.mean(adaptive_gains) > np.mean(control_gains["shuffle"]) + 1e-6
        ),
        "full_state_beats_no_z3_mean": bool(
            np.mean(adaptive_gains) > np.mean(control_gains["no_z3"]) + 1e-6
        ),
    }
    verdict["genuinely_sample_specific"] = bool(
        verdict["uses_both_routes_at_any_interior_budget"]
        and verdict["mean_gain_over_matched_fixed_mixture"] > 0.0
        and verdict["full_state_beats_shuffle_mean"]
    )
    verdict["extend_to_full_DAG"] = bool(
        verdict["genuinely_sample_specific"]
        and verdict["full_state_beats_no_z3_mean"]
    )

    frozen_hash_after = sha256(FROZEN_CHECKPOINT)
    if frozen_hash_after != frozen_hash_before:
        raise RuntimeError("protected forecasting checkpoint changed during policy training")
    final_report = {
        "method": "PostZ3ConstrainedTrajectoryGRPO_RLOO",
        "selected_checkpoint": str(best_path),
        "selected_epoch": int(selected["epoch"]),
        "frozen_forecaster_checkpoint": str(FROZEN_CHECKPOINT),
        "frozen_forecaster_sha256_before": frozen_hash_before,
        "frozen_forecaster_sha256_after": frozen_hash_after,
        "frozen_forecaster_unchanged": True,
        "policy_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "TRAIN_margin_scale_MAE": margin_scale,
        "budget_semantics": "upper bound on mean extra-Z6 fraction",
        "cost": fixed_cost,
        "policy_overhead": overhead,
        "TRAIN_thresholds": thresholds,
        "VALID_expected_policy": expected_panel_report(
            valid_arrays, args.budget_values, margin_scale
        ),
        "VALID_hard_policy": full_hard,
        "VALID_negative_controls": controls,
        "verdict": verdict,
        "test": None,
        "TEST_used": False,
    }
    dump_json(result_dir / "final_report.json", final_report)
    print(json.dumps({"VALID_hard_policy": full_hard, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
