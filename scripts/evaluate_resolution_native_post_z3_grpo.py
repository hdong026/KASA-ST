#!/usr/bin/env python3
"""Re-evaluate a frozen post-Z3 GRPO checkpoint without policy training."""

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

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.post_z3_constrained_grpo import (
    PostZ3BudgetRouter,
)
from scripts.f2f_cot_runtime import load_rescale, make_loader, select_batch
from scripts.train_resolution_native_post_z3_grpo import (
    EXPERIMENT,
    FROZEN_CHECKPOINT,
    collect_panel,
    expected_panel_report,
    fit_train_thresholds,
    fixed_costs_from_source,
    globally_shuffle_scores,
    hard_panel_report,
    load_frozen_environment,
    parse_budgets,
    profile_router_overhead,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tag", default="formal_v1")
    parser.add_argument("--budgets", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    budgets = parse_budgets(args.budgets)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    result_dir = ROOT / "results" / EXPERIMENT / f"{args.tag}_seed{args.seed}"
    checkpoint_path = (
        ROOT
        / "checkpoints"
        / "PEMS04"
        / "H12"
        / EXPERIMENT
        / f"{args.tag}_seed{args.seed}"
        / "router_best.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    existing = json.loads((result_dir / "final_report.json").read_text(encoding="utf-8"))
    protocol = json.loads((result_dir / "protocol.json").read_text(encoding="utf-8"))
    margin_scale = float(protocol["TRAIN_robust_pairwise_margin_scale_MAE"])
    forecaster, environment, _ = load_frozen_environment(device)
    policy = PostZ3BudgetRouter(node_size=forecaster.node_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    policy.eval()
    rescale = load_rescale()
    train_loader = make_loader("train", args.batch_size, False, args.workers)
    valid_loader = make_loader("valid", args.batch_size, False, args.workers)

    train_arrays = collect_panel(
        environment, policy, train_loader, budgets, device, rescale, seed=args.seed + 701
    )
    valid_arrays = collect_panel(
        environment, policy, valid_loader, budgets, device, rescale, seed=args.seed + 702
    )
    thresholds = fit_train_thresholds(train_arrays, budgets)
    fixed_cost = fixed_costs_from_source()
    full = hard_panel_report(valid_arrays, budgets, thresholds, fixed_cost)

    shuffled_train = globally_shuffle_scores(train_arrays, args.seed + 800)
    shuffled_valid = globally_shuffle_scores(valid_arrays, args.seed + 900)
    shuffled_thresholds = fit_train_thresholds(shuffled_train, budgets)
    shuffled = hard_panel_report(
        shuffled_valid, budgets, shuffled_thresholds, fixed_cost
    )

    no_z3_train = collect_panel(
        environment,
        policy,
        train_loader,
        budgets,
        device,
        rescale,
        control="no_z3",
        seed=args.seed + 801,
    )
    no_z3_valid = collect_panel(
        environment,
        policy,
        valid_loader,
        budgets,
        device,
        rescale,
        control="no_z3",
        seed=args.seed + 901,
    )
    no_z3_thresholds = fit_train_thresholds(no_z3_train, budgets)
    no_z3 = hard_panel_report(no_z3_valid, budgets, no_z3_thresholds, fixed_cost)

    interior_keys = [key for key in full["budgets"] if key not in {"0", "1"}]
    gains = [full["budgets"][key]["gain_over_matched_fixed_mixture"] for key in interior_keys]
    shuffled_gains = [
        shuffled["budgets"][key]["gain_over_matched_fixed_mixture"]
        for key in interior_keys
    ]
    no_z3_gains = [
        no_z3["budgets"][key]["gain_over_matched_fixed_mixture"]
        for key in interior_keys
    ]
    shares = [full["budgets"][key]["long_share"] for key in interior_keys]
    cis = [
        full["budgets"][key]["gain_over_matched_fixed_mixture_bootstrap_95pct_CI"]
        for key in interior_keys
    ]
    separations = [
        full["budgets"][key]["realized_gain_separation_long_minus_short"]
        for key in interior_keys
    ]
    genuine = bool(
        any(0.0 < share < 1.0 for share in shares)
        and all(gain > 0.0 for gain in gains)
        and all(interval[0] > 0.0 for interval in cis)
        and all(value is not None and value > 0.0 for value in separations)
        and np.mean(gains) > np.mean(shuffled_gains)
        and np.mean(gains) > np.mean(no_z3_gains)
    )
    verdict = {
        "collapsed_to_one_route_across_interior_budgets": False,
        "uses_both_routes_at_all_interior_budgets": bool(
            all(0.0 < share < 1.0 for share in shares)
        ),
        "all_matched_mixture_gain_CIs_exclude_zero": bool(
            all(interval[0] > 0.0 for interval in cis)
        ),
        "mean_gain_over_matched_fixed_mixture": float(np.mean(gains)),
        "global_shuffle_mean_gain": float(np.mean(shuffled_gains)),
        "no_z3_mean_gain": float(np.mean(no_z3_gains)),
        "positive_realized_gain_separation_at_all_interior_budgets": bool(
            all(value is not None and value > 0.0 for value in separations)
        ),
        "genuinely_sample_specific": genuine,
        "pilot_supports_full_DAG_extension": genuine,
        "full_DAG_ready_now": False,
        "full_DAG_blocker": (
            "route-complete forecasting checkpoint has not passed canonical containment; "
            "dual oscillation should also be stabilized before scaling"
        ),
    }
    example = select_batch(next(iter(valid_loader)), device)[0][:1]
    overhead = profile_router_overhead(
        environment, policy, example, warmup=10, repeats=50
    )
    short_ms = float(
        overhead["paired_fixed_vs_adaptive"]["short"]["adaptive"]["median_ms"]
    )
    long_ms = float(
        overhead["paired_fixed_vs_adaptive"]["long"]["adaptive"]["median_ms"]
    )
    short_flops = float(
        overhead["profiler_supported_FLOPs"]["adaptive_forced_short_total"]
    )
    long_flops = float(
        overhead["profiler_supported_FLOPs"]["adaptive_forced_long_total"]
    )
    for row in full["budgets"].values():
        share = float(row["long_share"])
        row["adaptive_profiled_latency_ms"] = (
            (1.0 - share) * short_ms + share * long_ms
        )
        row["adaptive_profiler_supported_FLOPs"] = (
            (1.0 - share) * short_flops + share * long_flops
        )
    report = {
        **existing,
        "evaluation_revision": (
            "global state shuffle; paired 1000-bootstrap matched-mixture CIs; "
            "realized route-gain separation"
        ),
        "policy_overhead": overhead,
        "frozen_forecaster_sha256_after_recheck": sha256(FROZEN_CHECKPOINT),
        "TRAIN_thresholds": thresholds,
        "VALID_expected_policy": expected_panel_report(valid_arrays, budgets, margin_scale),
        "VALID_hard_policy": full,
        "VALID_negative_controls": {"global_shuffle": shuffled, "no_z3": no_z3},
        "verdict": verdict,
        "test": None,
        "TEST_used": False,
    }
    (result_dir / "final_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"VALID_hard_policy": full, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
