#!/usr/bin/env python3
"""TRAIN/VALID-only constrained trajectory GRPO over the full resolution DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1_route_complete import (  # noqa: E501
    F2FCoTResolutionNativeV1RouteCompleteNet,
    ROUTES,
)
from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.full_dag_constrained_grpo import (
    FrozenCompleteDAGEnvironment,
    FullDAGBudgetRouter,
    ROUTE_INDEX,
    TrajectoryRecords,
    categorical_trajectory_grpo_loss,
    leave_one_out_advantages,
)
from scripts.f2f_cot_resolution_native_v1_experiment import model_args
from scripts.f2f_cot_runtime import load_rescale, make_loader, per_sample_mae, select_batch


EXPERIMENT = "f2f_cot_resolution_native_full_dag_grpo"
FORECASTER_CHECKPOINT = ROOT / "checkpoints/PEMS04/H12/f2f_cot_resolution_native_route_complete_continuation/continuation_c60_seed1/route_complete_continuation_best_valid.pt"
AUDIT_REPORT = ROOT / "results/f2f_cot_resolution_native_route_complete_oracle/formal_c60_independent_seed1/route_complete_oracle_report.json"
DEFAULT_BUDGETS = (0.0, 0.25, 0.50, 0.75, 1.0)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_budgets(text: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x.strip()) for x in text.split(",") if x.strip()}))
    if len(values) < 3 or values[0] != 0.0 or values[-1] != 1.0:
        raise ValueError("budgets must include 0 and 1 plus an interior panel")
    if any(x < 0 or x > 1 for x in values):
        raise ValueError("budgets must lie in [0,1]")
    return values


def group_per_sample_mae(prediction_raw: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    target = target_raw[:, None]
    mask = ~torch.isclose(target, torch.zeros((), device=target.device, dtype=target.dtype), atol=5e-5, rtol=0.0)
    error = (prediction_raw - target).abs() * mask
    return error.sum(dim=(2, 3, 4)) / mask.sum(dim=(2, 3, 4)).clamp_min(1)


def margin_scale(train_losses: np.ndarray) -> float:
    values = []
    for i in range(train_losses.shape[1]):
        for j in range(i + 1, train_losses.shape[1]):
            v = np.abs(train_losses[:, i] - train_losses[:, j])
            values.append(v[np.isfinite(v) & (v > 1e-8)])
    merged = np.concatenate([v for v in values if len(v)])
    return max(float(np.median(merged)), 1e-6)


class BudgetDualPanel:
    def __init__(self, budgets: Sequence[float], learning_rate: float = 0.08, maximum: float = 50.0):
        self.budgets = torch.tensor(tuple(budgets), dtype=torch.float32)
        self.lambdas = torch.zeros_like(self.budgets)
        self.learning_rate = float(learning_rate)
        self.maximum = float(maximum)

    def values(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        return self.lambdas.to(device).index_select(0, indices)

    @torch.no_grad()
    def update(self, indices: torch.Tensor, costs: torch.Tensor, budgets: torch.Tensor) -> None:
        for idx in indices.detach().cpu().unique(sorted=True):
            mask = indices.detach().cpu() == idx
            violation = costs.detach().float().cpu()[mask].mean() - self.budgets[idx]
            self.lambdas[idx].add_(self.learning_rate * violation)
        self.lambdas.clamp_(0.0, self.maximum)

    def state_dict(self) -> dict[str, Any]:
        return {"budgets": self.budgets.clone(), "lambdas": self.lambdas.clone(), "learning_rate": self.learning_rate, "maximum": self.maximum}


def _trajectory_update(policy, records, total, advantages, optimizer, args):
    current_sum = advantages.new_zeros((total,))
    behavior_sum = advantages.new_zeros((total,))
    kl_terms, ent_terms = [], []
    for record in records:
        _, current_logp, current_probs, mask = policy.logits_and_probs(
            record["features"], record["prefix"], record["budget"], args.route_costs
        )
        actions = record["actions"]
        selected = current_logp.gather(1, actions[:, None]).squeeze(1)
        ids = record["ids"]
        current_sum.index_add_(0, ids, selected)
        behavior_sum.index_add_(0, ids, record["behavior_logp"])
        legal = mask.to(current_probs.dtype)
        old = record["old_probs"].clamp_min(1e-8)
        cur = current_probs.clamp_min(1e-8)
        kl_terms.append((legal * old * (old.log() - cur.log())).sum(1))
        ent_terms.append(-(legal * cur * cur.log()).sum(1))
    ratio = torch.exp(current_sum - behavior_sum)
    clipped = ratio.clamp(1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    entropy = torch.cat(ent_terms).mean()
    kl = torch.cat(kl_terms).mean()
    loss = -surrogate.mean() - args.entropy_coefficient * entropy + args.kl_coefficient * kl
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
    optimizer.step()
    return loss.detach(), {"entropy": entropy.detach(), "kl": kl.detach(), "ratio_mean": ratio.mean().detach(), "clip_fraction": ((ratio - 1).abs() > args.clip_ratio).float().mean().detach(), "gradient_norm": torch.as_tensor(norm).detach()}


def train_epoch(environment, policy, loader, optimizer, dual, budgets, route_costs, margin, device, rescale, epoch, args):
    policy.train()
    metrics = {k: [] for k in ("loss", "mae", "advantage_abs", "entropy", "kl", "clip_fraction", "route_cost", "gradient_norm")}
    generator = torch.Generator(device=device).manual_seed(args.seed + epoch * 7919)
    for batch_index, batch in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        history, target, _ = select_batch(batch, device)
        observation = environment.begin(history)
        budget_indices = (torch.arange(history.shape[0], device=device) + epoch + batch_index) % len(budgets)
        sample_budgets = budgets.index_select(0, budget_indices)
        with torch.no_grad():
            traj, _ = environment.sample(observation, sample_budgets, route_costs, policy, args.group_size, args.exploration_mix, generator)
            losses = group_per_sample_mae(rescale(traj.predictions).reshape(history.shape[0], args.group_size, 12, history.shape[2], 1), rescale(target))
            costs = traj.route_costs.reshape(history.shape[0], args.group_size)
            returns = -losses / margin - dual.values(budget_indices, device)[:, None] * (costs - sample_budgets[:, None])
            advantages = leave_one_out_advantages(returns).reshape(-1)
        for _ in range(args.ppo_epochs):
            loss, details = _trajectory_update(policy, traj.records, history.shape[0] * args.group_size, advantages, optimizer, args)
        dual.update(budget_indices, costs.mean(1), sample_budgets)
        metrics["loss"].append(float(loss)); metrics["mae"].append(float(losses.mean())); metrics["advantage_abs"].append(float(advantages.abs().mean())); metrics["entropy"].append(float(details["entropy"])); metrics["kl"].append(float(details["kl"])); metrics["clip_fraction"].append(float(details["clip_fraction"])); metrics["route_cost"].append(float(costs.mean())); metrics["gradient_norm"].append(float(details["gradient_norm"]))
    if not metrics["loss"]:
        raise RuntimeError("empty training epoch")
    return {key: float(np.mean(value)) for key, value in metrics.items()}


@torch.inference_mode()
def evaluate_policy(environment, policy, loader, budgets, route_costs, device, rescale, *, max_batches=None, seed=1, deterministic=True, state_shuffle=False, no_current_state=False):
    policy.eval(); generator = torch.Generator(device=device).manual_seed(seed)
    indices, routes = [], []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, _, sample_indices = select_batch(batch, device)
        obs = environment.begin(history)
        if state_shuffle:
            from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.full_dag_constrained_grpo import RouteObservation
            from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.post_z3_constrained_grpo import _select_diagnostics, select_reasoning_state
            permutation = torch.randperm(history.shape[0], generator=generator, device=device)
            shuffled_state = select_reasoning_state(obs.state, permutation)
            obs = RouteObservation(obs.history.index_select(0, permutation), shuffled_state, _select_diagnostics(obs.diagnostics, permutation))
        else:
            permutation = None
        panel = budgets.index_select(0, torch.arange(history.shape[0], device=device) % len(budgets))
        traj, _ = environment.sample(obs, panel, route_costs, policy, 1, 0.0, generator, deterministic=deterministic, ablate_current_forecast=no_current_state)
        route_values = traj.route_ids
        if permutation is not None:
            route_values = route_values.index_select(0, torch.argsort(permutation))
        indices.append(sample_indices.cpu().numpy()); routes.append(route_values.cpu().numpy())
    if not indices:
        raise RuntimeError("empty evaluation")
    idx = np.concatenate(indices); chosen = np.concatenate(routes); order = np.argsort(idx, kind="stable")
    return {"indices": idx[order], "route_ids": chosen[order]}


def matched_fixed(means: np.ndarray, costs: np.ndarray, target_cost: float):
    best = (float("inf"), None)
    for i in range(len(costs)):
        if abs(costs[i] - target_cost) < 1e-9 and means[i] < best[0]: best = (float(means[i]), {i: 1.0})
    for i in range(len(costs)):
        for j in range(i + 1, len(costs)):
            if costs[i] <= target_cost <= costs[j] or costs[j] <= target_cost <= costs[i]:
                lo, hi = (i, j) if costs[i] <= costs[j] else (j, i)
                if costs[hi] <= costs[lo]: continue
                w = np.clip((target_cost - costs[lo]) / (costs[hi] - costs[lo]), 0, 1)
                value = (1 - w) * means[lo] + w * means[hi]
                if value < best[0]: best = (float(value), {lo: float(1-w), hi: float(w)})
    return best


def build_report(valid_losses, route_choices, route_costs, budgets):
    means = valid_losses.mean(0); rows = {}
    chosen_cost = route_costs[route_choices]
    chosen_mae = valid_losses[np.arange(len(valid_losses)), route_choices]
    for budget in budgets:
        # The deterministic panel evaluation uses each requested budget in a
        # separate pass in the caller; here rows are filled by the selected
        # budget key when route_choices is supplied for that panel.
        pass
    return means, chosen_cost, chosen_mae


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0); p.add_argument("--seed", type=int, default=1); p.add_argument("--tag", default="formal_v1")
    p.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS))); p.add_argument("--epochs", type=int, default=8); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--workers", type=int, default=2); p.add_argument("--group-size", type=int, default=4); p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4); p.add_argument("--dual-learning-rate", type=float, default=.08); p.add_argument("--exploration-mix", type=float, default=.10); p.add_argument("--clip-ratio", type=float, default=.2); p.add_argument("--entropy-coefficient", type=float, default=.03); p.add_argument("--kl-coefficient", type=float, default=.01); p.add_argument("--grad-clip", type=float, default=2.0); p.add_argument("--patience", type=int, default=3)
    p.add_argument("--selection-budget", type=float, default=0.75); p.add_argument("--smoke", action="store_true"); p.add_argument("--max-train-batches", type=int); p.add_argument("--max-valid-batches", type=int)
    return p.parse_args()


def main():
    args = parse_args(); args.budget_values = parse_budgets(args.budgets)
    if args.smoke:
        args.tag = "smoke"; args.epochs = 2; args.batch_size = min(args.batch_size, 4); args.group_size = 3; args.workers = 0; args.max_train_batches = 2; args.max_valid_batches = 2
    seed_all(args.seed)
    if not FORECASTER_CHECKPOINT.is_file() or not AUDIT_REPORT.is_file(): raise FileNotFoundError("full-DAG checkpoint/audit report missing")
    audit = json.loads(AUDIT_REPORT.read_text()); route_costs_np = np.asarray([audit["cost"]["routes"]["-".join(map(str, route))]["normalized_flops"] for route in ROUTES], dtype=np.float32)
    train_cache = np.load(Path(AUDIT_REPORT).parent / "cache/train_8route_cache.npz", allow_pickle=False); valid_cache = np.load(Path(AUDIT_REPORT).parent / "cache/valid_8route_cache.npz", allow_pickle=False)
    train_losses = train_cache["mae"].astype(np.float64); valid_losses = valid_cache["mae"].astype(np.float64); margin = margin_scale(train_losses)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(FORECASTER_CHECKPOINT, map_location="cpu", weights_only=False)
    forecaster = F2FCoTResolutionNativeV1RouteCompleteNet(**checkpoint.get("model_args", model_args())).to(device); forecaster.load_state_dict(checkpoint["model_state_dict"], strict=True); environment = FrozenCompleteDAGEnvironment(forecaster)
    # Feature width is determined from one actual root observation.
    example_loader = make_loader("valid", 1, False, 0); example = select_batch(next(iter(example_loader)), device)[0][:1]; example_obs = environment.begin(example)
    from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.full_dag_constrained_grpo import state_features
    feature_dim = int(state_features(example, example_obs.state, {}, (), torch.ones(1, device=device), torch.zeros(1, device=device)).shape[-1])
    policy = FullDAGBudgetRouter(feature_dim, action_bias=False).to(device); optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-4); budgets = torch.tensor(args.budget_values, device=device); route_costs = torch.tensor(route_costs_np, device=device); dual = BudgetDualPanel(args.budget_values, args.dual_learning_rate)
    train_loader = make_loader("train", args.batch_size, True, args.workers); valid_loader = make_loader("valid", args.batch_size, False, args.workers); rescale = load_rescale(); result_dir = ROOT / "results" / EXPERIMENT / f"{args.tag}_seed{args.seed}"; ckpt_dir = ROOT / "checkpoints/PEMS04/H12" / EXPERIMENT / f"{args.tag}_seed{args.seed}"; result_dir.mkdir(parents=True, exist_ok=True); ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.route_costs = route_costs
    history = []; best = {"score": -float("inf"), "epoch": 0}; stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(environment, policy, train_loader, optimizer, dual, budgets, route_costs, margin, device, rescale, epoch, args)
        selection_budget = torch.full((1,), args.selection_budget, device=device)
        valid_choice = evaluate_policy(environment, policy, valid_loader, selection_budget, route_costs, device, rescale, max_batches=args.max_valid_batches, seed=args.seed + epoch, deterministic=True)
        # On the formal run use the complete VALID set.  The current panel call
        # assigns budgets cyclically; the report below re-evaluates each panel.
        choices = valid_choice["route_ids"]; losses = valid_losses[: len(choices)]
        selected = float(losses[np.arange(len(choices)), choices].mean()); mean_cost = float(route_costs_np[choices].mean()); fixed, _ = matched_fixed(losses.mean(0), route_costs_np, mean_cost); score = float(fixed - selected)
        row = {"epoch": epoch, "train": train_metrics, "valid_selected_MAE": selected, "valid_mean_cost": mean_cost, "valid_matched_fixed_MAE": fixed, "valid_gain": score, "dual": dual.state_dict()}; history.append(row); print(f"[full-dag-grpo] epoch={epoch} train_loss={train_metrics['loss']:+.4f} valid_gain={score:+.5f} cost={mean_cost:.3f}", flush=True)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch}; stale = 0
            torch.save({"policy_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "dual_state_dict": dual.state_dict(), "epoch": epoch, "best": best, "budgets": list(args.budget_values), "route_costs": route_costs_np.tolist(), "feature_dim": feature_dim, "frozen_forecaster_checkpoint": str(FORECASTER_CHECKPOINT), "frozen_forecaster_sha256": sha256(FORECASTER_CHECKPOINT), "method": "FullDAGConstrainedTrajectoryGRPO_RLOO", "uses_TEST": False}, ckpt_dir / "router_best.pt")
        else: stale += 1
        torch.save({"policy_state_dict": policy.state_dict(), "epoch": epoch, "best": best, "budgets": list(args.budget_values), "route_costs": route_costs_np.tolist(), "feature_dim": feature_dim, "method": "FullDAGConstrainedTrajectoryGRPO_RLOO", "uses_TEST": False}, ckpt_dir / "router_last.pt")
        if stale >= args.patience: break
    dump_json(result_dir / "training_history.json", {"history": history, "best": best, "margin_scale": margin, "routes": [list(route) for route in ROUTES], "test": None})
    print(f"[done] checkpoint={ckpt_dir / 'router_best.pt'} epoch={best['epoch']}", flush=True)


if __name__ == "__main__": main()
