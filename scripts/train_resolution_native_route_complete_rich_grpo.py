#!/usr/bin/env python3
"""Counterfactual, node-aware GRPO for complete ResolutionNative routing.

Each batch executes the frozen route tree exactly once.  All budget-feasible
terminal routes form the same-sample GRPO group, so the update sees the true
leave-one-out route advantages instead of a small stochastic subset of paths.
"""

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

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1_route_complete import (
    F2FCoTResolutionNativeV1RouteCompleteNet,
    ROUTES,
)
from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.full_dag_constrained_grpo import (
    FrozenCompleteDAGEnvironment,
    PREFIXES,
    PREFIX_INDEX,
    ROUTE_INDEX,
    RichFullDAGBudgetRouter,
    prefix_cost_lower_bound,
)
from scripts.f2f_cot_resolution_native_v1_experiment import model_args
from scripts.f2f_cot_runtime import load_rescale, make_loader, per_sample_mae, select_batch
from scripts.train_resolution_native_route_complete_grpo import (
    AUDIT_REPORT,
    FORECASTER_CHECKPOINT,
    matched_fixed,
    margin_scale,
)


EXPERIMENT = "f2f_cot_resolution_native_full_dag_rich_grpo"
DEFAULT_BUDGETS = (0.0, 0.25, 0.50, 0.75, 1.0)
# At budget 1 all terminal routes are feasible, so accuracy is lexicographically
# prior to centered sample-wise gain. Interior budgets retain centered gain.
FULL_BUDGET_QUALITY_ANCHOR = True


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
        raise ValueError("budgets must include 0 and 1 and an interior panel")
    return values


class DualPanel:
    def __init__(self, budgets: Sequence[float], learning_rate: float = 0.08, maximum: float = 50.0):
        self.budgets = torch.tensor(tuple(budgets), dtype=torch.float32)
        self.lambdas = torch.zeros_like(self.budgets)
        self.learning_rate = float(learning_rate)
        self.maximum = float(maximum)

    def values(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        return self.lambdas.to(device).index_select(0, indices)

    @torch.no_grad()
    def update(self, indices: torch.Tensor, expected_cost: torch.Tensor) -> None:
        cpu_indices = indices.detach().cpu()
        cpu_cost = expected_cost.detach().float().cpu()
        for index in cpu_indices.unique(sorted=True):
            mask = cpu_indices == index
            violation = cpu_cost[mask].mean() - self.budgets[index]
            self.lambdas[index].add_(self.learning_rate * violation)
        self.lambdas.clamp_(0.0, self.maximum)

    def state_dict(self):
        return {
            "budgets": self.budgets.clone(),
            "lambdas": self.lambdas.clone(),
            "learning_rate": self.learning_rate,
            "maximum": self.maximum,
        }


def group_mae(route_outputs: Mapping[tuple[int, ...], Mapping[str, torch.Tensor]], target_raw: torch.Tensor, rescale) -> torch.Tensor:
    values = []
    for route in ROUTES:
        prediction = rescale(route_outputs[route]["pred"])
        values.append(per_sample_mae(prediction, target_raw))
    return torch.stack(values, dim=1)


def build_prefix_feature_map(model, policy, history, raw_tree, budgets, route_costs):
    feature_map = {}
    for prefix in PREFIXES:
        state = raw_tree["states"][prefix]
        diagnostics = raw_tree["edge_steps"].get(prefix, {})
        consumed = history.new_full(
            (history.shape[0],), prefix_cost_lower_bound(prefix, route_costs)
        )
        features = policy.build_features(
            history,
            state,
            diagnostics,
            prefix,
            budgets,
            consumed,
        )
        with torch.inference_mode(False):
            feature_map[prefix] = features.detach().clone()
    return feature_map


def route_distribution(policy, feature_map, budgets, route_costs):
    local_logp = {}
    local_probs = {}
    local_masks = {}
    for prefix in PREFIXES:
        _, logp, probs, mask = policy.logits_and_probs(
            feature_map[prefix], prefix, budgets, route_costs
        )
        local_logp[prefix] = logp
        local_probs[prefix] = probs
        local_masks[prefix] = mask
    route_logp = []
    for route in ROUTES:
        prefix = ()
        terms = []
        for action in route:
            terms.append(local_logp[prefix][:, int({2: 0, 3: 1, 4: 2, 6: 3, 12: 4}[action])])
            prefix = (*prefix, action)
            if action == 12:
                break
        route_logp.append(torch.stack(terms, dim=-1).sum(dim=-1))
    route_logp = torch.stack(route_logp, dim=-1)
    route_probs = route_logp.exp()
    route_probs = route_probs / route_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return {
        "route_logp": route_logp,
        "route_probs": route_probs,
        "local_logp": local_logp,
        "local_probs": local_probs,
        "local_masks": local_masks,
    }


def masked_loo_advantages(utility: torch.Tensor, route_costs: torch.Tensor, budgets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    feasible = route_costs.to(utility).view(1, -1) <= budgets.reshape(-1, 1) + 1e-6
    count = feasible.sum(dim=1, keepdim=True)
    safe_count = count.clamp_min(2)
    total = (utility * feasible).sum(dim=1, keepdim=True)
    others = (total - utility) / (safe_count - 1)
    advantages = torch.where(count > 1, utility - others, torch.zeros_like(utility))
    return advantages, feasible


def counterfactual_grpo_loss(
    current_probs: torch.Tensor,
    behavior_probs: torch.Tensor,
    advantages: torch.Tensor,
    feasible: torch.Tensor,
    old_probs: torch.Tensor,
    *,
    clip_ratio: float,
    entropy_coefficient: float,
    kl_coefficient: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = current_probs / behavior_probs.clamp_min(1e-8)
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    weights = behavior_probs * feasible.to(behavior_probs.dtype)
    loss = -(weights * surrogate).sum(dim=1).mean()
    current_safe = current_probs.clamp_min(1e-8)
    old_safe = old_probs.clamp_min(1e-8)
    legal = feasible.to(current_probs.dtype)
    entropy = -(legal * current_safe * current_safe.log()).sum(dim=1).mean()
    kl = (legal * old_safe * (old_safe.log() - current_safe.log())).sum(dim=1).mean()
    loss = loss - entropy_coefficient * entropy + kl_coefficient * kl
    return loss, {
        "surrogate": (weights * surrogate).sum(dim=1).mean(),
        "entropy": entropy,
        "kl": kl,
        "ratio_mean": ratio[feasible].mean() if bool(feasible.any()) else ratio.mean(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_ratio).float()[feasible].mean() if bool(feasible.any()) else ratio.new_zeros(()),
    }


def route_losses_from_tree(raw_tree, target_raw, rescale):
    return group_mae(raw_tree["routes"], target_raw, rescale)


def train_epoch(environment, model, policy, loader, optimizer, dual, budgets, route_costs, margin, route_baseline, device, rescale, epoch, args):
    policy.train()
    metrics = {key: [] for key in ("loss", "route_mae", "advantage_abs", "entropy", "kl", "clip_fraction", "expected_cost", "gradient_norm")}
    for batch_index, batch in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        history, target, _ = select_batch(batch, device)
        budget_indices = (torch.arange(history.shape[0], device=device) + epoch + batch_index) % len(budgets)
        sample_budgets = budgets.index_select(0, budget_indices)
        with torch.inference_mode():
            raw_tree = model.rollout_all_routes_shared(history)
            target_raw = rescale(target)
            losses = route_losses_from_tree(raw_tree, target_raw, rescale)
            feature_map = build_prefix_feature_map(model, policy, history, raw_tree, sample_budgets, route_costs)
        with torch.no_grad():
            old = route_distribution(policy, feature_map, sample_budgets, route_costs)
            feasible = route_costs.to(device).view(1, -1) <= sample_budgets[:, None] + 1e-6
            uniform = feasible.to(old["route_probs"].dtype) / feasible.sum(dim=1, keepdim=True).clamp_min(1)
            behavior = (1.0 - args.exploration_mix) * old["route_probs"] + args.exploration_mix * uniform
            lambdas = dual.values(budget_indices, device)
            # Remove the TRAIN-fitted mean of each route.  Absolute MAE would
            # reward the globally best path at every sample; centered utility
            # asks only whether this sample benefits from deviating from the
            # corresponding fixed-route baseline. Once the full DAG is
            # affordable, however, there is no remaining compute tradeoff:
            # restore the route-quality ordering as a lexicographic accuracy
            # anchor. The anchor uses the same TRAIN margin scale, so it does
            # not mix millisecond and MAE units.
            centered_utility = -(losses - route_baseline[None, :]) / margin
            quality_gap = (route_baseline - route_baseline.min()) / margin
            full_budget = (sample_budgets >= 1.0 - 1e-6).to(centered_utility) if FULL_BUDGET_QUALITY_ANCHOR else torch.zeros_like(sample_budgets, dtype=centered_utility.dtype)
            utility = centered_utility - full_budget[:, None] * quality_gap[None, :] - lambdas[:, None] * (route_costs[None, :] - sample_budgets[:, None])
            advantages, feasible = masked_loo_advantages(utility, route_costs, sample_budgets)
            advantages = advantages.detach()
        for _ in range(args.ppo_epochs):
            current = route_distribution(policy, feature_map, sample_budgets, route_costs)
            loss, details = counterfactual_grpo_loss(
                current["route_probs"],
                behavior,
                advantages,
                feasible,
                old["route_probs"],
                clip_ratio=args.clip_ratio,
                entropy_coefficient=args.entropy_coefficient,
                kl_coefficient=args.kl_coefficient,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
        expected_cost = (current["route_probs"].detach() * route_costs[None, :]).sum(dim=1)
        dual.update(budget_indices, expected_cost)
        metrics["loss"].append(float(loss.detach())); metrics["route_mae"].append(float(losses.mean())); metrics["advantage_abs"].append(float(advantages.abs().mean())); metrics["entropy"].append(float(details["entropy"].detach())); metrics["kl"].append(float(details["kl"].detach())); metrics["clip_fraction"].append(float(details["clip_fraction"].detach())); metrics["expected_cost"].append(float(expected_cost.mean())); metrics["gradient_norm"].append(float(norm))
    if not metrics["loss"]:
        raise RuntimeError("empty training epoch")
    return {key: float(np.mean(value)) for key, value in metrics.items()}


@torch.inference_mode()
def evaluate_policy(model, policy, loader, budget_value, route_costs, device, rescale, *, max_batches=None, state_shuffle=False, no_current_state=False, seed=1):
    policy.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    all_losses, all_choices = [], []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target, _ = select_batch(batch, device)
        raw_tree = model.rollout_all_routes_shared(history)
        target_raw = rescale(target)
        losses = route_losses_from_tree(raw_tree, target_raw, rescale)
        budgets = torch.full((history.shape[0],), float(budget_value), device=device)
        if state_shuffle:
            permutation = torch.randperm(history.shape[0], generator=generator, device=device)
            # The route tree is still evaluated on the original samples; only
            # the policy state is permuted, then choices are inverted.
            shuffled_history = history.index_select(0, permutation)
            shuffled_states = {key: value for key, value in raw_tree["states"].items()}
            for key, state in shuffled_states.items():
                from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.post_z3_constrained_grpo import select_reasoning_state
                shuffled_states[key] = select_reasoning_state(state, permutation)
            feature_map = {}
            for prefix in PREFIXES:
                consumed = budgets.new_full((history.shape[0],), prefix_cost_lower_bound(prefix, route_costs))
                diagnostics = raw_tree["edge_steps"].get(prefix, {})
                if diagnostics:
                    from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.post_z3_constrained_grpo import _select_diagnostics
                    diagnostics = _select_diagnostics(diagnostics, permutation)
                feature_map[prefix] = policy.build_features(
                    shuffled_history,
                    shuffled_states[prefix],
                    diagnostics,
                    prefix,
                    budgets,
                    consumed,
                    ablate_current_forecast=no_current_state,
                )
        else:
            feature_map = build_prefix_feature_map(model, policy, history, raw_tree, budgets, route_costs)
            if no_current_state:
                feature_map = build_prefix_feature_map(model, policy, history, raw_tree, budgets, route_costs)
                for prefix in PREFIXES:
                    state = raw_tree["states"][prefix]
                    consumed = budgets.new_full((history.shape[0],), prefix_cost_lower_bound(prefix, route_costs))
                    feature_map[prefix] = policy.build_features(history, state, raw_tree["edge_steps"].get(prefix, {}), prefix, budgets, consumed, ablate_current_forecast=True, ablate_current_hidden=True)
        distribution = route_distribution(policy, feature_map, budgets, route_costs)
        feasible = route_costs[None, :] <= budgets[:, None] + 1e-6
        masked_probs = distribution["route_probs"].masked_fill(~feasible, -1.0)
        choices = masked_probs.argmax(dim=1)
        all_losses.append(losses.cpu()); all_choices.append(choices.cpu())
    losses = torch.cat(all_losses).numpy(); choices = torch.cat(all_choices).numpy()
    return losses, choices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0); parser.add_argument("--seed", type=int, default=1); parser.add_argument("--tag", default="rich_v1")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS))); parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--workers", type=int, default=2); parser.add_argument("--ppo-epochs", type=int, default=2); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--dual-learning-rate", type=float, default=.08); parser.add_argument("--exploration-mix", type=float, default=.10); parser.add_argument("--clip-ratio", type=float, default=.2); parser.add_argument("--entropy-coefficient", type=float, default=.01); parser.add_argument("--kl-coefficient", type=float, default=.01); parser.add_argument("--grad-clip", type=float, default=2.0); parser.add_argument("--selection-budget", type=float, default=.75); parser.add_argument("--patience", type=int, default=3); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--max-train-batches", type=int); parser.add_argument("--max-valid-batches", type=int)
    return parser.parse_args()


def main():
    args = parse_args(); budget_values = parse_budgets(args.budgets)
    if args.smoke:
        if args.tag == "rich_v1":
            args.tag = "smoke"
        args.epochs = 2; args.batch_size = min(args.batch_size, 4); args.workers = 0; args.max_train_batches = 2; args.max_valid_batches = 2; args.ppo_epochs = 1
    seed_all(args.seed)
    audit = json.loads(AUDIT_REPORT.read_text())
    route_costs_np = np.asarray([audit["cost"]["routes"]["-".join(map(str, route))]["normalized_flops"] for route in ROUTES], dtype=np.float32)
    train_cache = np.load(Path(AUDIT_REPORT).parent / "cache/train_8route_cache.npz", allow_pickle=False)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_route_losses = train_cache["mae"].astype(np.float64)
    margin = margin_scale(train_route_losses)
    route_baseline = torch.tensor(train_route_losses.mean(0), dtype=torch.float32, device=device)
    checkpoint = torch.load(FORECASTER_CHECKPOINT, map_location="cpu", weights_only=False)
    model = F2FCoTResolutionNativeV1RouteCompleteNet(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    environment = FrozenCompleteDAGEnvironment(model)
    policy = RichFullDAGBudgetRouter(model.node_size).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    budgets = torch.tensor(budget_values, device=device)
    route_costs = torch.tensor(route_costs_np, device=device)
    dual = DualPanel(budget_values, args.dual_learning_rate)
    train_loader = make_loader("train", args.batch_size, True, args.workers)
    valid_loader = make_loader("valid", args.batch_size, False, args.workers)
    rescale = load_rescale()
    result_dir = ROOT / "results" / EXPERIMENT / f"{args.tag}_seed{args.seed}"
    checkpoint_dir = ROOT / "checkpoints/PEMS04/H12" / EXPERIMENT / f"{args.tag}_seed{args.seed}"
    result_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], {"score": -float("inf"), "epoch": 0}, 0
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(environment, model, policy, train_loader, optimizer, dual, budgets, route_costs, margin, route_baseline, device, rescale, epoch, args)
        valid_losses, valid_choices = evaluate_policy(model, policy, valid_loader, args.selection_budget, route_costs, device, rescale, max_batches=args.max_valid_batches, seed=args.seed + epoch)
        selected = float(valid_losses[np.arange(len(valid_losses)), valid_choices].mean())
        selected_cost = float(route_costs_np[valid_choices].mean())
        fixed, _ = matched_fixed(valid_losses.mean(0), route_costs_np, selected_cost)
        score = float(fixed - selected)
        row = {"epoch": epoch, "train": metrics, "valid_selected_MAE": selected, "valid_mean_normalized_FLOPs": selected_cost, "valid_matched_fixed_MAE": fixed, "valid_gain": score, "dual": dual.state_dict()}
        history.append(row)
        print(f"[rich-grpo] epoch={epoch} train_loss={metrics['loss']:+.4f} valid_gain={score:+.5f} cost={selected_cost:.3f}", flush=True)
        if score > best["score"] + 1e-10:
            best = {"score": score, "epoch": epoch}; stale = 0
            torch.save({"policy_state_dict": policy.state_dict(), "epoch": epoch, "best": best, "budgets": list(budget_values), "route_costs": route_costs_np.tolist(), "route_baseline_TRAIN_MAE": route_baseline.detach().cpu().tolist(), "node_size": model.node_size, "node_feature_dim": policy.node_feature_dim, "frozen_forecaster_checkpoint": str(FORECASTER_CHECKPOINT), "frozen_forecaster_sha256": sha256(FORECASTER_CHECKPOINT), "method": "FullDAGRichCounterfactualCenteredGainGRPO_RLOO", "full_budget_quality_anchor": FULL_BUDGET_QUALITY_ANCHOR, "uses_TEST": False}, checkpoint_dir / "router_best.pt")
        else:
            stale += 1
        torch.save({"policy_state_dict": policy.state_dict(), "epoch": epoch, "best": best, "budgets": list(budget_values), "route_costs": route_costs_np.tolist(), "route_baseline_TRAIN_MAE": route_baseline.detach().cpu().tolist(), "node_size": model.node_size, "node_feature_dim": policy.node_feature_dim, "method": "FullDAGRichCounterfactualCenteredGainGRPO_RLOO", "full_budget_quality_anchor": FULL_BUDGET_QUALITY_ANCHOR, "uses_TEST": False}, checkpoint_dir / "router_last.pt")
        if stale >= args.patience:
            break
    dump_json(result_dir / "training_history.json", {"history": history, "best": best, "margin_scale": margin, "route_baseline_TRAIN_MAE": route_baseline.detach().cpu().tolist(), "policy_parameters": sum(parameter.numel() for parameter in policy.parameters()), "state": "node-preserving post-Z3-style representation plus frozen active hidden summaries", "training": "counterfactual exact shared-tree centered-gain GRPO/RLOO with full-budget quality anchor", "full_budget_quality_anchor": FULL_BUDGET_QUALITY_ANCHOR, "test": None})
    print(f"[done] checkpoint={checkpoint_dir / 'router_best.pt'} epoch={best['epoch']}", flush=True)


if __name__ == "__main__":
    main()
