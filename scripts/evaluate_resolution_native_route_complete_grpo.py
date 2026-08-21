#!/usr/bin/env python3
"""Evaluate a full-DAG GRPO router on VALID only, with matched-cost controls."""

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
from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.full_dag_constrained_grpo import FullDAGBudgetRouter
from scripts.f2f_cot_resolution_native_v1_experiment import model_args
from scripts.f2f_cot_runtime import load_rescale, make_loader, select_batch
from scripts.train_resolution_native_route_complete_grpo import AUDIT_REPORT, FORECASTER_CHECKPOINT, evaluate_policy


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def matched_fixed(means, costs, target):
    best = (float("inf"), {})
    for i in range(len(costs)):
        if abs(costs[i] - target) < 1e-8 and means[i] < best[0]: best = (float(means[i]), {i: 1.0})
    for i in range(len(costs)):
        for j in range(i + 1, len(costs)):
            lo, hi = (i, j) if costs[i] <= costs[j] else (j, i)
            if costs[lo] <= target <= costs[hi] and costs[hi] > costs[lo]:
                weight = np.clip((target - costs[lo]) / (costs[hi] - costs[lo]), 0, 1)
                value = (1 - weight) * means[lo] + weight * means[hi]
                if value < best[0]: best = (float(value), {lo: float(1-weight), hi: float(weight)})
    return best


def bootstrap_gain(losses, chosen, costs, repeats, seed):
    rng = np.random.default_rng(seed); n = len(losses); values = []
    selected_cost = float(costs[chosen].mean())
    for _ in range(repeats):
        ix = rng.integers(0, n, size=n); sampled = losses[ix]; sampled_chosen = chosen[ix]
        adaptive = float(sampled[np.arange(n), sampled_chosen].mean()); fixed, _ = matched_fixed(sampled.mean(0), costs, float(costs[sampled_chosen].mean())); values.append(fixed - adaptive)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--gpu", type=int, default=0); p.add_argument("--budgets", default="0,0.25,0.5,0.75,1"); p.add_argument("--tag", default="formal_v1"); p.add_argument("--bootstrap", type=int, default=500); return p.parse_args()


def main():
    args = parse_args(); budgets = tuple(float(x) for x in args.budgets.split(",")); audit = json.loads(AUDIT_REPORT.read_text()); raw_flops = np.asarray([audit["cost"]["routes"]["-".join(map(str, route))]["flops"] for route in ROUTES], dtype=np.float64); costs = np.asarray([audit["cost"]["routes"]["-".join(map(str, route))]["normalized_flops"] for route in ROUTES], dtype=np.float64); latency = np.asarray([audit["cost"]["routes"]["-".join(map(str, route))]["latency"]["median_ms"] for route in ROUTES], dtype=np.float64)
    cache = np.load(Path(AUDIT_REPORT).parent / "cache/valid_8route_cache.npz", allow_pickle=False); cache_indices = cache["indices"].astype(int); order = np.argsort(cache_indices); cache_losses = cache["mae"].astype(np.float64)[order]; cache_indices = cache_indices[order]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"); forecaster_ckpt = torch.load(FORECASTER_CHECKPOINT, map_location="cpu", weights_only=False); forecaster = F2FCoTResolutionNativeV1RouteCompleteNet(**forecaster_ckpt.get("model_args", model_args())).to(device); forecaster.load_state_dict(forecaster_ckpt["model_state_dict"], strict=True)
    policy_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False); action_bias = any(str(key).endswith("actor.4.bias") for key in policy_ckpt["policy_state_dict"]); policy = FullDAGBudgetRouter(int(policy_ckpt["feature_dim"]), action_bias=action_bias).to(device); policy.load_state_dict(policy_ckpt["policy_state_dict"], strict=True)
    from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.full_dag_constrained_grpo import FrozenCompleteDAGEnvironment
    environment = FrozenCompleteDAGEnvironment(forecaster); loader = make_loader("valid", 16, False, 2); rescale = load_rescale(); route_choices = {}; shuffled_choices = {}; no_current_choices = {}
    for budget in budgets:
        panel = torch.full((1,), budget, device=device); result = evaluate_policy(environment, policy, loader, panel, torch.tensor(costs, device=device, dtype=torch.float32), device, rescale, seed=19 + int(100*budget), deterministic=True); route_choices[budget] = result["route_ids"].astype(int); shuffled = evaluate_policy(environment, policy, loader, panel, torch.tensor(costs, device=device, dtype=torch.float32), device, rescale, seed=191 + int(100*budget), deterministic=True, state_shuffle=True); shuffled_choices[budget] = shuffled["route_ids"].astype(int); ablated = evaluate_policy(environment, policy, loader, panel, torch.tensor(costs, device=device, dtype=torch.float32), device, rescale, seed=291 + int(100*budget), deterministic=True, no_current_state=True); no_current_choices[budget] = ablated["route_ids"].astype(int)
    means = cache_losses.mean(0); rows = {}
    for budget, choices in route_choices.items():
        if len(choices) != len(cache_losses): raise RuntimeError("VALID policy/cache sample count mismatch")
        selected = cache_losses[np.arange(len(cache_losses)), choices]; share = np.bincount(choices, minlength=len(ROUTES)) / len(choices); mean_cost = float(costs[choices].mean()); fixed, mixture = matched_fixed(means, costs, mean_cost); feasible = np.flatnonzero(costs <= budget + 1e-8); oracle = cache_losses[:, feasible].min(1); best_fixed_feasible = int(feasible[np.argmin(means[feasible])]); shuffled = shuffled_choices[budget]; shuffled_mae = float(cache_losses[np.arange(len(cache_losses)), shuffled].mean()); ablated = no_current_choices[budget]; ablated_mae = float(cache_losses[np.arange(len(cache_losses)), ablated].mean()); selected_route_mean_gain = float(np.mean(means[choices] - selected)); rows[f"{budget:.6g}"] = {"budget_normalized_FLOPs": float(budget), "route_share": {"-".join(map(str, ROUTES[i])): float(share[i]) for i in range(len(ROUTES))}, "nonzero_route_count": int((share > 0).sum()), "route_share_entropy": float(-(share[share > 0] * np.log(share[share > 0])).sum()), "policy_MAE": float(selected.mean()), "mean_normalized_FLOPs": mean_cost, "mean_actual_FLOPs": float(raw_flops[choices].mean()), "mean_latency_ms": float(latency[choices].mean()), "matched_fixed_mixture_MAE": float(fixed), "matched_fixed_mixture": {"-".join(map(str, ROUTES[i])): float(w) for i,w in mixture.items()}, "gain_over_matched_fixed_mixture": float(fixed - selected.mean()), "gain_bootstrap_95pct_CI": bootstrap_gain(cache_losses, choices, costs, args.bootstrap, 1000 + int(100*budget)), "selected_route_mean_gain": selected_route_mean_gain, "global_state_shuffle_MAE": shuffled_mae, "global_state_shuffle_delta": float(shuffled_mae - selected.mean()), "no_current_forecast_MAE": ablated_mae, "no_current_forecast_delta": float(ablated_mae - selected.mean()), "best_fixed_feasible_route": list(ROUTES[best_fixed_feasible]), "best_fixed_feasible_MAE": float(means[best_fixed_feasible]), "budget_oracle_MAE": float(oracle.mean()), "budget_oracle_headroom": float(means[best_fixed_feasible] - oracle.mean()), "budget_oracle_mean_cost": float(costs[feasible[np.argmin(cache_losses[:, feasible], axis=1)]].mean()), "selected_mean_realized_loss": float(cache_losses[np.arange(len(cache_losses)), choices].mean())}
        long_gain = means[0] - means[choices] if False else None
    report = {"method": "FullDAGConstrainedTrajectoryGRPO_RLOO", "objective": {"quality": "negative physical masked MAE divided by TRAIN-only robust route-margin scale", "group_relative_advantage": "leave-one-out centered, no group standard-deviation normalization", "importance_ratio": "clipped trajectory-level ratio over the sum of sequential action log-probabilities", "compute_constraint": "primal-dual upper expected normalized actual-FLOPs budget with hard legal/budget action masks", "regularization": "entropy plus behavior-policy KL", "targets_in_policy_state": False}, "frozen_forecaster_checkpoint": str(FORECASTER_CHECKPOINT), "router_checkpoint": str(args.checkpoint), "routes": [list(r) for r in ROUTES], "route_costs_actual_FLOPs": raw_flops.tolist(), "route_costs_normalized_FLOPs": costs.tolist(), "route_latency_median_ms": latency.tolist(), "always_short": {"route": list(ROUTES[0]), "MAE": float(means[0]), "FLOPs": float(raw_flops[0]), "latency_ms": float(latency[0])}, "always_long": {"route": list(ROUTES[5]), "MAE": float(means[5]), "FLOPs": float(raw_flops[5]), "latency_ms": float(latency[5])}, "route_mean_MAE": {"-".join(map(str, ROUTES[i])): float(means[i]) for i in range(len(ROUTES))}, "budgets": rows, "adaptivity_verdict": "collapsed_fixed_by_budget" if all(row["nonzero_route_count"] <= 1 for row in rows.values()) else "sample_specific_route_use", "test": None}
    out = ROOT / "results" / "f2f_cot_resolution_native_full_dag_grpo" / f"{args.tag}_seed1" / "final_report.json"; dump(out, report); print(f"[done] report={out}")


if __name__ == "__main__": main()
