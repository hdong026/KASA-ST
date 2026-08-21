"""On-policy online controller for a protected corrective terminal transition.

The sampled action is taken at the actually reached Z6 state.  Skip executes
the untouched native 6->12 edge; refine executes the learned residual-
corrective 6->12 edge.  No unchosen action or cached route outcome is used in
policy training.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .diagnose_learned_consistency import HistoryForecastConsistency
from .run_corrective_terminal_transition import (
    CorrectiveTerminalTransition,
    corrective_prediction,
)
from .run_online_sequential_rl import _profile_paired_callables
from .run_pipeline import (
    DEFAULT_DATA_DIR,
    WindowDataset,
    canonical_audit,
    load_data,
    make_loader,
    per_sample_mae,
    prepare_batch,
    seed_everything,
    to_physical,
)
from .run_selector import DEFAULT_BRIDGE_CHECKPOINT, build_frozen_forecaster


class OnlineCorrectiveActorCritic(nn.Module):
    def __init__(self, node_count: int, width: int = 64):
        super().__init__()
        self.consistency = HistoryForecastConsistency(node_count, 2, width)

    def forward(self, history, z6, z3):
        output = self.consistency(
            history[..., 0], z6[..., 0], z3[..., 0]
        )
        return output[:, 0], output[:, 1]


@torch.no_grad()
def reach_z6(model, history):
    z3 = model.execute_transition(history, None, 3, None)
    z6 = model.execute_transition(history, 3, 6, z3)
    return z3, z6


@torch.no_grad()
def execute_selected(model, transition, history, z6, refine):
    """Execute each selected real terminal transition, grouped by action."""
    batch = len(history)
    predictions = torch.empty(
        batch, 12, history.shape[2], 1, device=history.device, dtype=history.dtype
    )
    canonical_predictions = torch.empty_like(predictions)
    for action in (False, True):
        indices = torch.nonzero(refine == action, as_tuple=False).flatten()
        if not len(indices):
            continue
        h = history.index_select(0, indices)
        reached = z6.index_select(0, indices)
        provisional = model.execute_transition(h, 6, 12, reached)
        canonical = model.finalize_forecast(provisional, h)
        if action:
            _, prediction, _ = corrective_prediction(
                model, transition, h, reached, provisional
            )
        else:
            prediction = canonical
        predictions.index_copy_(0, indices, prediction)
        canonical_predictions.index_copy_(0, indices, canonical)
    return predictions, canonical_predictions


def train_epoch(model, transition, policy, loader, optimizer, device, mean, std, args):
    policy.train()
    started = time.perf_counter()
    totals = Counter()
    samples = 0
    for batch_index, batch in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        history, target = prepare_batch(batch, device)
        with torch.no_grad():
            z3, z6 = reach_z6(model, history)
        logit, value = policy(history, z6, z3)
        distribution = torch.distributions.Bernoulli(logits=logit)
        refine = distribution.sample().bool()
        # The selected action is immediately followed by its real transition.
        prediction, canonical = execute_selected(
            model, transition, history, z6, refine
        )
        target_raw = to_physical(target, mean, std)
        loss_selected = per_sample_mae(
            to_physical(prediction, mean, std), target_raw, 0.0
        )
        loss_canonical = per_sample_mae(
            to_physical(canonical, mean, std), target_raw, 0.0
        )
        # Per-sample improvement removes absolute difficulty variance. Skip's
        # outcome is identically zero; refine observes only its executed gain.
        reward = (loss_canonical - loss_selected).detach()
        log_prob = distribution.log_prob(refine.float())
        advantage = reward - value.detach()
        actor = -(log_prob * advantage).mean()
        critic = F.smooth_l1_loss(value, reward)
        entropy = distribution.entropy().mean()
        objective = actor + args.critic_weight * critic - args.entropy_weight * entropy
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        count = len(history)
        samples += count
        totals["objective"] += float(objective.detach()) * count
        totals["reward"] += float(reward.mean()) * count
        totals["refine"] += int(refine.sum())
    return {
        "seconds": time.perf_counter() - started,
        "objective": totals["objective"] / samples,
        "mean_on_policy_gain": totals["reward"] / samples,
        "refine_fraction": totals["refine"] / samples,
    }


@torch.inference_mode()
def collect_losses(model, transition, policy, loader, device, mean, std, max_batches=None):
    policy.eval()
    canonical_parts, corrected_parts, chosen_parts, action_parts = [], [], [], []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target = prepare_batch(batch, device)
        z3, z6 = reach_z6(model, history)
        logit, _ = policy(history, z6, z3)
        refine = logit > 0
        chosen, canonical = execute_selected(model, transition, history, z6, refine)
        # Execute the corrective route for all samples only for evaluation and
        # oracle accounting; it is never used by policy training.
        provisional = model.execute_transition(history, 6, 12, z6)
        _, corrected, _ = corrective_prediction(
            model, transition, history, z6, provisional
        )
        target_raw = to_physical(target, mean, std)
        canonical_parts.append(per_sample_mae(to_physical(canonical, mean, std), target_raw, 0.0).cpu())
        corrected_parts.append(per_sample_mae(to_physical(corrected, mean, std), target_raw, 0.0).cpu())
        chosen_parts.append(per_sample_mae(to_physical(chosen, mean, std), target_raw, 0.0).cpu())
        action_parts.append(refine.cpu())
    return {
        "canonical": torch.cat(canonical_parts),
        "corrected": torch.cat(corrected_parts),
        "chosen": torch.cat(chosen_parts),
        "refine": torch.cat(action_parts),
    }


@torch.inference_mode()
def profile_costs(model, transition, policy, history, args):
    def fixed(refine):
        z3, z6 = reach_z6(model, history)
        provisional = model.execute_transition(history, 6, 12, z6)
        if refine:
            return corrective_prediction(model, transition, history, z6, provisional)[1]
        return model.finalize_forecast(provisional, history)

    def adaptive(refine):
        z3, z6 = reach_z6(model, history)
        policy(history, z6, z3)
        provisional = model.execute_transition(history, 6, 12, z6)
        if refine:
            return corrective_prediction(model, transition, history, z6, provisional)[1]
        return model.finalize_forecast(provisional, history)

    result = {}
    for name, refine in (("canonical", False), ("corrected", True)):
        fixed_row, adaptive_row = _profile_paired_callables(
            history.device, lambda r=refine: fixed(r), lambda r=refine: adaptive(r),
            args.latency_warmup, args.latency_repeats,
        )
        result[name] = {
            "fixed": fixed_row,
            "adaptive": adaptive_row,
            "policy_overhead_p90_ms": adaptive_row["p90_ms"] - fixed_row["p90_ms"],
        }
    return result


def budget_report(losses, costs, budgets):
    rows = []
    canonical = losses["canonical"]
    corrected = losses["corrected"]
    chosen_refine = losses["refine"]
    for budget in budgets:
        refine_feasible = costs["corrected"]["adaptive"]["p90_ms"] <= budget + 1e-9
        action = chosen_refine if refine_feasible else torch.zeros_like(chosen_refine)
        adaptive_loss = torch.where(action, corrected, canonical)
        adaptive_cost = torch.where(
            action,
            torch.full_like(canonical, costs["corrected"]["adaptive"]["p90_ms"]),
            torch.full_like(canonical, costs["canonical"]["adaptive"]["p90_ms"]),
        )
        fixed_candidates = [("canonical", canonical)]
        if costs["corrected"]["fixed"]["p90_ms"] <= budget + 1e-9:
            fixed_candidates.append(("corrected", corrected))
        best_fixed_name, best_fixed_loss = min(
            fixed_candidates, key=lambda item: float(item[1].mean())
        )
        oracle_candidates = [canonical]
        if refine_feasible:
            oracle_candidates.append(corrected)
        oracle = torch.stack(oracle_candidates).min(0).values
        oracle_headroom = float(canonical.mean() - oracle.mean())
        recovered = float(canonical.mean() - adaptive_loss.mean())
        rows.append({
            "budget_ms": float(budget),
            "adaptive_mae": float(adaptive_loss.mean()),
            "adaptive_mean_accounted_p90_cost_ms": float(adaptive_cost.mean()),
            "route_counts": {
                "3->6->12": int((~action).sum()),
                "3->6->corrective12": int(action.sum()),
            },
            "best_fixed_route": best_fixed_name,
            "best_fixed_mae": float(best_fixed_loss.mean()),
            "oracle_mae": float(oracle.mean()),
            "oracle_headroom_vs_canonical": oracle_headroom,
            "headroom_recovered": recovered,
            "headroom_recovered_fraction": recovered / oracle_headroom if oracle_headroom > 0 else 0.0,
        })
    return rows


def parse_args():
    from .run_online_sequential_rl import parse_args as online_parse_args
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--f2f-checkpoint", type=Path)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE_CHECKPOINT)
    parser.add_argument("--transition-checkpoint", type=Path, required=True)
    parser.add_argument("--correction-limit", type=float, default=0.1)
    parser.add_argument("--transition-width", type=int, default=64)
    parser.add_argument("--policy-width", type=int, default=64)
    parser.add_argument("--bridge-correction-limit", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--critic-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max_eval_batches", type=int)
    parser.add_argument("--latency-warmup", type=int, default=30)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    old = sys.argv
    try:
        sys.argv = [old[0]]
        defaults = online_parse_args()
    finally:
        sys.argv = old
    for key in ("config", "f2f_checkpoint"):
        if getattr(args, key) is None:
            setattr(args, key, getattr(defaults, key))
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data, indices, mean, std = load_data(args.data_dir)
    datasets = {split: WindowDataset(data, indices[split]) for split in ("train", "valid", "test")}

    def loader(split, shuffle=False, batch_size=None):
        return make_loader(datasets[split], batch_size=batch_size or args.batch_size,
                           shuffle=shuffle, workers=args.workers, device=device, seed=args.seed)

    model = build_frozen_forecaster(args, device)
    transition = CorrectiveTerminalTransition(
        data.shape[1], args.transition_width, args.correction_limit
    ).to(device)
    transition.load_state_dict(
        torch.load(args.transition_checkpoint, map_location=device)["transition_state_dict"], strict=True
    )
    transition.requires_grad_(False).eval()
    policy = OnlineCorrectiveActorCritic(data.shape[1], args.policy_width).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    canonical_audit(model, loader("valid"), device)
    best_path = args.output_dir / "online_corrective_policy_best.pt"
    best_mae = float("inf")
    stale = 0
    history_log = []
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(model, transition, policy, loader("train", True), optimizer, device, mean, std, args)
        valid = collect_losses(model, transition, policy, loader("valid"), device, mean, std, args.max_eval_batches)
        valid_mae = float(valid["chosen"].mean())
        improved = valid_mae < best_mae
        if improved:
            best_mae = valid_mae
            torch.save({"epoch": epoch, "policy_state_dict": policy.state_dict(), "valid_mae": valid_mae}, best_path)
            stale = 0
        else:
            stale += 1
        row = {"epoch": epoch, "train": train, "valid_mae": valid_mae,
               "valid_refine_fraction": float(valid["refine"].float().mean()), "best": improved}
        history_log.append(row)
        (args.output_dir / "history.json").write_text(json.dumps(history_log, indent=2), encoding="utf-8")
        print(f"[epoch {epoch:02d}] seconds={train['seconds']:.1f} valid={valid_mae:.5f} refine={row['valid_refine_fraction']:.3f} reward={train['mean_on_policy_gain']:.5f} best={improved}")
        if stale >= args.patience:
            break
    checkpoint = torch.load(best_path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    canonical_audit(model, loader("valid"), device)
    profile_batch = next(iter(loader("valid", batch_size=1)))
    profile_history, _ = prepare_batch(profile_batch, device)
    costs = profile_costs(model, transition, policy, profile_history, args)
    adaptive_costs = [costs[name]["adaptive"]["p90_ms"] for name in ("canonical", "corrected")]
    fixed_costs = [costs[name]["fixed"]["p90_ms"] for name in ("canonical", "corrected")]
    minimum_adaptive = costs["canonical"]["adaptive"]["p90_ms"]
    budgets = sorted(
        value for value in set(adaptive_costs + [max(adaptive_costs), max(fixed_costs)])
        if value + 1e-9 >= minimum_adaptive
    )
    validation = collect_losses(model, transition, policy, loader("valid"), device, mean, std, args.max_eval_batches)
    # TEST is touched only now, after the checkpoint and method were selected on VALID.
    test = collect_losses(model, transition, policy, loader("test"), device, mean, std, args.max_eval_batches)
    report = {
        "method": "genuine_online_on_policy_corrective_terminal_controller",
        "best_epoch": checkpoint["epoch"],
        "training_uses_cached_route_outcomes": False,
        "selected_transition_executed_online": True,
        "policy_parameters": sum(p.numel() for p in policy.parameters()),
        "epoch_seconds": [row["train"]["seconds"] for row in history_log],
        "physical_latency": costs,
        "validation": budget_report(validation, costs, budgets),
        "test": budget_report(test, costs, budgets),
        "canonical_exact": {"torch_equal": True, "max_abs": 0.0},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
