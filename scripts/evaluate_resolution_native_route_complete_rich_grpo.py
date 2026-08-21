#!/usr/bin/env python3
"""VALID-only evaluation for the node-aware counterfactual full-DAG router."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1_route_complete import F2FCoTResolutionNativeV1RouteCompleteNet, ROUTES
from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.full_dag_constrained_grpo import PREFIXES, RichFullDAGBudgetRouter, prefix_cost_lower_bound
from scripts.f2f_cot_runtime import load_rescale, make_loader, per_sample_mae, select_batch
from scripts.train_resolution_native_route_complete_grpo import AUDIT_REPORT, FORECASTER_CHECKPOINT, matched_fixed
from scripts.train_resolution_native_route_complete_rich_grpo import build_prefix_feature_map, route_distribution


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def bootstrap_gain(losses, choices, costs, repeats, seed):
    rng = np.random.default_rng(seed); values = []; n = len(losses)
    for _ in range(repeats):
        ix = rng.integers(0, n, size=n); sampled = losses[ix]; sampled_choices = choices[ix]
        adaptive = float(sampled[np.arange(n), sampled_choices].mean())
        fixed, _ = matched_fixed(sampled.mean(0), costs, float(costs[sampled_choices].mean()))
        values.append(float(fixed - adaptive))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def parse_args():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--gpu", type=int, default=0); parser.add_argument("--tag", default="formal_v1"); parser.add_argument("--seed", type=int, default=1); parser.add_argument("--split", choices=("valid", "test"), default="valid"); parser.add_argument("--output", type=Path); parser.add_argument("--budgets", default="0,0.25,0.5,0.75,1"); parser.add_argument("--bootstrap", type=int, default=500); parser.add_argument("--full-budget-fallback", choices=("none", "train-best"), default="none", help="at budget 1, abstain to the TRAIN-selected best fixed route") ; return parser.parse_args()


@torch.inference_mode()
def evaluate_panel(model, policy, loader, budgets, route_costs, device, rescale, *, full_budget_route_index=None):
    outputs = {float(b): {"losses": [], "choices": [], "shuffle_choices": [], "no_current_choices": []} for b in budgets}
    generator = torch.Generator(device=device).manual_seed(901)
    for batch in loader:
        history, target, _ = select_batch(batch, device)
        raw = model.rollout_all_routes_shared(history)
        target_raw = rescale(target)
        losses = torch.stack([per_sample_mae(rescale(raw["routes"][route]["pred"]), target_raw) for route in ROUTES], dim=1)
        for budget_value in budgets:
            budget = torch.full((history.shape[0],), float(budget_value), device=device)
            feature_map = build_prefix_feature_map(model, policy, history, raw, budget, route_costs)
            distribution = route_distribution(policy, feature_map, budget, route_costs)
            feasible = route_costs[None, :] <= budget[:, None] + 1e-6
            choices = distribution["route_probs"].masked_fill(~feasible, -1.0).argmax(dim=1)
            if full_budget_route_index is not None:
                full_budget = budget >= 1.0 - 1e-6
                choices = torch.where(full_budget, torch.full_like(choices, int(full_budget_route_index)), choices)

            permutation = torch.randperm(history.shape[0], generator=generator, device=device)
            from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.post_z3_constrained_grpo import _select_diagnostics, select_reasoning_state
            shuffled_history = history.index_select(0, permutation)
            shuffled_feature_map = {}
            no_current_feature_map = {}
            inverse = torch.argsort(permutation)
            for prefix in PREFIXES:
                consumed = budget.new_full((history.shape[0],), prefix_cost_lower_bound(prefix, route_costs))
                shuffled_state = select_reasoning_state(raw["states"][prefix], permutation)
                diagnostics = raw["edge_steps"].get(prefix, {})
                if diagnostics:
                    diagnostics = _select_diagnostics(diagnostics, permutation)
                # Keep the shuffled order while scoring.  Choices are mapped
                # back to original sample positions after the policy forward.
                shuffled_feature_map[prefix] = policy.build_features(shuffled_history, shuffled_state, diagnostics, prefix, budget, consumed)
                no_current_feature_map[prefix] = policy.build_features(history, raw["states"][prefix], raw["edge_steps"].get(prefix, {}), prefix, budget, consumed, ablate_current_forecast=True, ablate_current_hidden=True)
            shuffled_distribution = route_distribution(policy, shuffled_feature_map, budget, route_costs)
            no_current_distribution = route_distribution(policy, no_current_feature_map, budget, route_costs)
            shuffled_choices = shuffled_distribution["route_probs"].masked_fill(~feasible, -1.0).argmax(dim=1)
            no_current_choices = no_current_distribution["route_probs"].masked_fill(~feasible, -1.0).argmax(dim=1)
            if full_budget_route_index is not None:
                full_budget = budget >= 1.0 - 1e-6
                fixed = torch.full_like(shuffled_choices, int(full_budget_route_index))
                shuffled_choices = torch.where(full_budget, fixed, shuffled_choices)
                no_current_choices = torch.where(full_budget, fixed, no_current_choices)
            outputs[float(budget_value)]["losses"].append(losses.cpu().numpy())
            outputs[float(budget_value)]["choices"].append(choices.cpu().numpy())
            # Deliberately retain shuffled-position choices: they are evaluated
            # against the original target rows to destroy state/sample
            # correspondence.
            outputs[float(budget_value)]["shuffle_choices"].append(shuffled_choices.cpu().numpy())
            outputs[float(budget_value)]["no_current_choices"].append(no_current_choices.cpu().numpy())
    for budget in budgets:
        for key in outputs[float(budget)]:
            outputs[float(budget)][key] = np.concatenate(outputs[float(budget)][key], axis=0) if key == "losses" else np.concatenate(outputs[float(budget)][key], axis=0)
    return outputs


def main():
    args = parse_args(); budgets = tuple(float(x) for x in args.budgets.split(",")); audit = json.loads(AUDIT_REPORT.read_text())
    raw_flops = np.asarray([audit["cost"]["routes"]["-".join(map(str, r))]["flops"] for r in ROUTES], dtype=np.float64)
    costs = np.asarray([audit["cost"]["routes"]["-".join(map(str, r))]["normalized_flops"] for r in ROUTES], dtype=np.float64)
    latency = np.asarray([audit["cost"]["routes"]["-".join(map(str, r))]["latency"]["median_ms"] for r in ROUTES], dtype=np.float64)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    forecaster_checkpoint = torch.load(FORECASTER_CHECKPOINT, map_location="cpu", weights_only=False)
    model = F2FCoTResolutionNativeV1RouteCompleteNet(**forecaster_checkpoint["model_args"]).to(device); model.load_state_dict(forecaster_checkpoint["model_state_dict"], strict=True)
    model.eval()
    router_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    anchor_enabled = bool(router_checkpoint.get("full_budget_quality_anchor", "full_anchor" in str(args.checkpoint)))
    policy = RichFullDAGBudgetRouter(int(router_checkpoint["node_size"]), int(router_checkpoint["node_feature_dim"]), action_bias=False).to(device); policy.load_state_dict(router_checkpoint["policy_state_dict"], strict=True)
    train_cache_path = Path(AUDIT_REPORT).parent / "cache/train_8route_cache.npz"
    train_best_route = int(np.load(train_cache_path, allow_pickle=False)["mae"].mean(0).argmin())
    fallback_route_index = train_best_route if args.full_budget_fallback == "train-best" else None
    loader = make_loader(args.split, 16, False, 2); panel = evaluate_panel(model, policy, loader, budgets, torch.tensor(costs, device=device, dtype=torch.float32), device, load_rescale(), full_budget_route_index=fallback_route_index)
    rows = {}; route_means = None
    for budget in budgets:
        result = panel[float(budget)]; losses = result["losses"]; choices = result["choices"]; shuffled = result["shuffle_choices"]; no_current = result["no_current_choices"]
        means = losses.mean(0); route_means = means; share = np.bincount(choices, minlength=len(ROUTES)) / len(choices); mean_cost = float(costs[choices].mean()); fixed, mixture = matched_fixed(means, costs, mean_cost); feasible = np.flatnonzero(costs <= budget + 1e-8); best_fixed = int(feasible[np.argmin(means[feasible])]); oracle = losses[:, feasible].min(1); selected = losses[np.arange(len(losses)), choices]; global_permutation = np.random.default_rng(1300 + int(100 * budget)).permutation(len(choices)); global_shuffled_mae = float(losses[np.arange(len(losses)), choices[global_permutation]].mean()); shuffled_mae = float(losses[np.arange(len(losses)), shuffled].mean()); no_current_mae = float(losses[np.arange(len(losses)), no_current].mean())
        dominant = np.argsort(-share)[:2]
        pair_conditioning = None
        if share[dominant[0]] > 0 and share[dominant[1]] > 0:
            left, right = int(dominant[1]), int(dominant[0])
            margin = losses[:, left] - losses[:, right]
            left_selected = choices == left
            right_selected = choices == right
            pair_conditioning = {"route_left": list(ROUTES[left]), "route_right": list(ROUTES[right]), "margin_left_minus_right": float(margin.mean()), "margin_when_left_selected": float(margin[left_selected].mean()) if bool(left_selected.any()) else None, "margin_when_right_selected": float(margin[right_selected].mean()) if bool(right_selected.any()) else None, "selection_margin_separation": float(margin[right_selected].mean() - margin[left_selected].mean()) if bool(left_selected.any()) and bool(right_selected.any()) else None}
        rows[f"{budget:.6g}"] = {"budget_normalized_FLOPs": float(budget), "route_share": {"-".join(map(str, ROUTES[i])): float(share[i]) for i in range(len(ROUTES))}, "nonzero_route_count": int((share > 0).sum()), "route_share_entropy": float(-(share[share > 0] * np.log(share[share > 0])).sum()), "policy_MAE": float(selected.mean()), "mean_normalized_FLOPs": mean_cost, "mean_actual_FLOPs": float(raw_flops[choices].mean()), "mean_latency_ms": float(latency[choices].mean()), "matched_fixed_mixture_MAE": float(fixed), "matched_fixed_mixture": {"-".join(map(str, ROUTES[i])): float(w) for i, w in mixture.items()}, "gain_over_matched_fixed_mixture": float(fixed - selected.mean()), "gain_bootstrap_95pct_CI": bootstrap_gain(losses, choices, costs, args.bootstrap, 700 + int(100 * budget)), "pairwise_conditioning": pair_conditioning, "global_state_shuffle_MAE": global_shuffled_mae, "global_state_shuffle_delta": float(global_shuffled_mae - selected.mean()), "batch_state_shuffle_MAE": shuffled_mae, "batch_state_shuffle_delta": float(shuffled_mae - selected.mean()), "no_current_forecast_MAE": no_current_mae, "no_current_forecast_delta": float(no_current_mae - selected.mean()), "best_fixed_feasible_route": list(ROUTES[best_fixed]), "best_fixed_feasible_MAE": float(means[best_fixed]), "budget_oracle_MAE": float(oracle.mean()), "budget_oracle_headroom": float(means[best_fixed] - oracle.mean()), "budget_oracle_mean_cost": float(costs[feasible[np.argmin(losses[:, feasible], axis=1)]].mean())}
    report = {"method": "FullDAGRichCounterfactualQualityAnchoredGRPO_RLOO" if anchor_enabled else "FullDAGRichCounterfactualCenteredGainGRPO_RLOO", "evaluation_split": args.split.upper(), "evaluation_seed": int(args.seed), "TEST_used": args.split == "test", "state": "node-preserving post-Z3-style temporal/evidence/forecast/diagnostic representation plus frozen reasoner active-hidden summaries with attention pooling", "objective": {"quality": "negative physical masked MAE after subtracting the TRAIN-only mean of the selected route, divided by TRAIN-only robust route-margin scale", "adaptive_baseline": "per-route TRAIN MAE baseline removes global fixed-path preference without forcing route diversity", "full_budget_quality_anchor": "absolute route-quality ordering is restored only at budget 1, where every terminal route is affordable" if anchor_enabled else "disabled", "full_budget_fallback": "TRAIN-selected best fixed route at budget 1" if fallback_route_index is not None else "disabled", "group_relative_advantage": "exact feasible-route leave-one-out centering, no group standard-deviation normalization", "importance_ratio": "clipped route-trajectory ratio using exact counterfactual route groups", "compute_constraint": "actual normalized route FLOP upper budget with hard feasible-route masks", "targets_in_policy_state": False}, "router_checkpoint": str(args.checkpoint), "frozen_forecaster_checkpoint": str(FORECASTER_CHECKPOINT), "routes": [list(r) for r in ROUTES], "route_costs_actual_FLOPs": raw_flops.tolist(), "route_costs_normalized_FLOPs": costs.tolist(), "route_latency_median_ms": latency.tolist(), "always_short": {"route": list(ROUTES[0]), "MAE": float(route_means[0]), "FLOPs": float(raw_flops[0]), "latency_ms": float(latency[0])}, "always_long": {"route": list(ROUTES[5]), "MAE": float(route_means[5]), "FLOPs": float(raw_flops[5]), "latency_ms": float(latency[5])}, "route_mean_MAE": {"-".join(map(str, ROUTES[i])): float(route_means[i]) for i in range(len(ROUTES))}, "budgets": rows, "adaptivity_verdict": "sample_specific_route_use" if any(row["nonzero_route_count"] > 1 and row["gain_over_matched_fixed_mixture"] > 0 for row in rows.values()) else "collapsed_or_no_gain", "test": None}
    output = args.output if args.output is not None else ROOT / "results" / "f2f_cot_resolution_native_full_dag_rich_grpo" / f"{args.tag}_seed{args.seed}" / f"{args.split}_report.json"; dump(output, report); print(f"[done] report={output}")


if __name__ == "__main__":
    main()
