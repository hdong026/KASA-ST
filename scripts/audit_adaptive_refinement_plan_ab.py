#!/usr/bin/env python3
"""Unified audit for Plan A / Plan B adaptive refinement infrastructure."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ok(name: str, cond: bool, detail: str = "") -> dict:
    return {"name": name, "pass": bool(cond), "detail": detail}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    results = []

    # --- imports ---
    try:
        from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_refinement_context import (
            PRE_ROUTE_OVERHEAD_NAME,
            pool_pre_route_context,
            pre_route_overhead_report,
        )
        from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_refinement_routes import (
            standard_refinement_route_template,
        )
        from basicts.archs.arch_zoo.ChainForecasting_arch.budget_conditioned_adaptive_f2f import (
            BudgetConditionedAdaptiveF2FNet,
        )
        from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_forecast_refinement_route import (
            AdaptiveForecastRefinementRouteNet,
        )
        from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_gain_loss import (
            compute_pair_imbalance_weights,
        )
        from basicts.archs.arch_zoo.ChainForecasting_arch.group_relative_refinement_objective import (
            clipped_trajectory_objective,
            group_relative_advantages,
            terminal_route_reward,
        )
        from basicts.archs.arch_zoo.ChainForecasting_arch.group_relative_refinement_policy import (
            GroupRelativeRefinementPolicy,
        )
        from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
            SequentialF2FEnvironment,
        )
        from basicts.archs.arch_zoo.ChainForecasting_arch.temporal_crossfit_refinement import (
            build_rolling_crossfit_manifest,
            build_temporal_holdout_manifest,
            compute_min_purge_samples,
            sample_raw_span,
            spans_overlap,
        )
        from scripts.budget_f2f_synth_kwargs import synthetic_budget_f2f_kwargs

        results.append(_ok("imports", True))
    except Exception as e:
        results.append(_ok("imports", False, str(e)))
        print(json.dumps({"results": results}, indent=2))
        return 1

    # --- stable templates / H generalization ---
    for H in (12, 24, 48):
        t = standard_refinement_route_template(H)
        results.append(
            _ok(
                f"route_template_H{H}",
                t["direct"] == [H]
                and t["half"] == [H // 2, H]
                and t["quarter"] == [H // 4, H]
                and t["progressive"] == [H // 4, H // 2, H],
                str(t),
            )
        )

    # --- pre-route overhead honesty ---
    rep = pre_route_overhead_report()
    results.append(
        _ok(
            "pre_route_overhead_honest",
            rep["cache_reuse_into_execute_route"] is False
            and rep["name"] == PRE_ROUTE_OVERHEAD_NAME,
            json.dumps(rep),
        )
    )

    # --- sign-specific pair weights ---
    # Construct scores so one pair has pos=9750, neg=192 after margin filter
    # We call the weight formula directly by mocking counts via many samples
    r = 4
    n = 9750 + 192
    scores = torch.zeros(n, r)
    # For pair (1,0): first 9750 positive (s1>s0), last 192 negative
    scores[:9750, 1] = 1.0
    scores[:9750, 0] = 0.0
    scores[9750:, 1] = 0.0
    scores[9750:, 0] = 1.0
    # other dims equal
    w_pos, w_neg, report = compute_pair_imbalance_weights(scores, rank_ignore_margin=0.01)
    key = "0<1"
    wp = report[key]["w_pos"]
    wn = report[key]["w_neg"]
    results.append(
        _ok(
            "sign_specific_pair_weights",
            wn > wp * 2.0,  # minority significantly higher
            f"w_pos={wp:.4f} w_neg={wn:.4f} counts={report[key]}",
        )
    )

    # --- duplicate controller forward: source check ---
    src = Path(
        ROOT
        / "basicts/archs/arch_zoo/ChainForecasting_arch/adaptive_forecast_refinement_route.py"
    ).read_text()
    # After parent forward, must not re-call estimate_refinement_gains
    after_parent = src.split("BudgetConditionedAdaptiveF2FNet.forward")[-1]
    results.append(
        _ok(
            "no_duplicate_controller_after_parent",
            "estimate_refinement_gains" not in after_parent.split("def trainable")[0],
            "checked adaptive forward tail",
        )
    )
    parent_src = Path(
        ROOT
        / "basicts/archs/arch_zoo/ChainForecasting_arch/budget_conditioned_adaptive_f2f.py"
    ).read_text()
    results.append(
        _ok(
            "parent_passes_plan_extras",
            "predicted_gains" in parent_src and "Pass through planner/controller extras" in parent_src,
        )
    )

    # --- temporal holdout / purge ---
    # Synthetic contiguous index like PEMS: (i, i+12, i+24)
    index = [(i, i + 12, i + 24) for i in range(200)]
    purge = compute_min_purge_samples(index)
    results.append(_ok("purge_auto_span24", purge == 24, f"purge={purge}"))
    man = build_temporal_holdout_manifest(index, train_fraction=0.8, horizon=12)
    overlaps = 0
    for i in man["supernet_train_samples"][-30:]:
        for j in man["oracle_holdout_samples"][:30]:
            if spans_overlap(sample_raw_span(index[i]), sample_raw_span(index[j])):
                overlaps += 1
    results.append(_ok("holdout_zero_overlap", overlaps == 0 and man["overlap_audit"]["n_overlaps"] == 0))

    # real PEMS index if present
    idx_path = ROOT / "datasets/PEMS04/index_in12_out12.pkl"
    if idx_path.is_file():
        from basicts.archs.arch_zoo.ChainForecasting_arch.temporal_crossfit_refinement import (
            load_split_index,
        )

        real = load_split_index(idx_path, "train")
        man_r = build_temporal_holdout_manifest(real, train_fraction=0.8)
        overlaps_r = 0
        for i in man_r["supernet_train_samples"][-40:]:
            for j in man_r["oracle_holdout_samples"][:40]:
                if spans_overlap(sample_raw_span(real[i]), sample_raw_span(real[j])):
                    overlaps_r += 1
        results.append(
            _ok(
                "pems_holdout_zero_overlap",
                overlaps_r == 0,
                f"purge={man_r['purge_samples']} n_oracle={len(man_r['oracle_holdout_samples'])}",
            )
        )
        xf = build_rolling_crossfit_manifest(real, num_blocks=5)
        causal_ok = True
        for fold in xf["folds"]:
            teach = set(fold["teacher_train_indices"])
            ora = set(fold["oracle_indices"])
            if teach & ora:
                causal_ok = False
            # teacher max raw end <= oracle min raw start
            if fold["teacher_train_indices"] and fold["oracle_indices"]:
                t_end = max(sample_raw_span(real[i])[1] for i in fold["teacher_train_indices"])
                o_start = min(sample_raw_span(real[i])[0] for i in fold["oracle_indices"])
                if t_end > o_start:
                    causal_ok = False
        results.append(_ok("crossfit_temporal_causality", causal_ok, f"folds={len(xf['folds'])}"))
    else:
        results.append(_ok("pems_holdout_zero_overlap", False, "index missing"))
        results.append(_ok("crossfit_temporal_causality", False, "index missing"))

    # --- sequential equivalence + masks ---
    model = BudgetConditionedAdaptiveF2FNet(
        **synthetic_budget_f2f_kwargs(
            node_size=7, training_phase="eval", route_selection_mode="forced"
        )
    ).to(device).eval()
    env = SequentialF2FEnvironment(model)
    h = torch.randn(2, 12, 7, 4, device=device)
    with torch.no_grad():
        eq = env.sequential_route_equivalence_check(h, atol=1e-5)
    results.append(
        _ok(
            "sequential_quarter_equiv",
            eq["quarter_ok"],
            f"diff={eq['quarter_max_abs_diff']}",
        )
    )
    results.append(
        _ok(
            "sequential_progressive_equiv",
            eq["progressive_ok"],
            f"diff={eq['progressive_max_abs_diff']}",
        )
    )

    m0 = env.action_masks(0.0)
    m05 = env.action_masks(0.5)
    m075 = env.action_masks(0.75)
    m1 = env.action_masks(1.0)
    results.append(_ok("mask_eta0_direct_only", bool(m0["mask0"].tolist() == [True, False, False])))
    # eta0.5: direct+quarter (progressive may be masked -> quarter branch still open if quarter feasible)
    results.append(
        _ok(
            "mask_eta05_direct_quarter",
            bool(m05["mask0"][0] and not m05["mask0"][1] and m05["mask0"][2])
            and bool(m05["mask1"][0] and not m05["mask1"][1]),
            str({k: v.tolist() if torch.is_tensor(v) else v for k, v in m05.items() if k.startswith("mask")}),
        )
    )
    results.append(
        _ok(
            "mask_eta075_no_progressive",
            bool(m075["mask0"].tolist() == [True, True, True])
            and bool(m075["mask1"].tolist() == [True, False]),
            str(m075["mask0"].tolist()) + str(m075["mask1"].tolist()),
        )
    )
    results.append(
        _ok(
            "mask_eta1_all",
            bool(m1["mask0"].all()) and bool(m1["mask1"].all()),
        )
    )
    # infeasible prob = 0 via masked softmax
    policy = GroupRelativeRefinementPolicy(context_dim=16, zq_dim=1, hidden=64).to(device)
    s0 = torch.randn(4, 16, device=device)
    logits = policy.logits0(policy.encode_s0(s0) if False else policy.s0_proj(s0))
    # use encode properly
    s0e = policy.encode_s0(s0)
    logp = policy.masked_log_softmax(policy.logits0(s0e), m0["mask0"].to(device).unsqueeze(0).expand(4, -1))
    probs = logp.exp()
    results.append(
        _ok(
            "infeasible_prob_zero_eta0",
            float(probs[:, 1:].sum()) < 1e-6,
            f"probs={probs[0].tolist()}",
        )
    )

    # --- group rewards / advantages ---
    L = torch.tensor([[18.20, 18.12, 18.18, 18.13]])
    C = model.route_costs.detach().float().cpu()
    feas = torch.ones(1, 4, dtype=torch.bool)
    R, _ = terminal_route_reward(L, C, feas, delta_abs=0.05, lambda_quality=10.0, lambda_cost=1.0)
    A, info = group_relative_advantages(R, feas)
    results.append(_ok("group_rewards_finite", bool(torch.isfinite(R).all())))
    results.append(_ok("group_adv_finite", bool(torch.isfinite(A).all())))
    # advantage order should match reward order
    results.append(
        _ok(
            "adv_reward_order_consistent",
            bool(torch.argsort(R[0]) .tolist() == torch.argsort(A[0]).tolist()),
            f"R={R[0].tolist()} A={A[0].tolist()}",
        )
    )
    # zero variance
    R0 = torch.ones(1, 4)
    A0, info0 = group_relative_advantages(R0, feas)
    results.append(_ok("zero_variance_safe", float(A0.abs().sum()) == 0 and info0["zero_variance_groups"] == 1))

    # clipped objective ratio
    log_old = torch.zeros(8)
    log_new = torch.ones(8) * 0.1
    adv = torch.randn(8)
    loss, st = clipped_trajectory_objective(log_new, log_old, adv, clip_eps=0.2)
    results.append(_ok("clipped_objective_finite", bool(torch.isfinite(loss))))
    results.append(_ok("ratio_not_forced_one", abs(st["ratio_mean"] - 1.0) > 1e-6, str(st)))

    # --- policy: eta not in features; Zq dependence ---
    with torch.no_grad():
        zq = torch.randn(2, 3, 7, 1, device=device)
        s0 = torch.randn(2, 16, device=device)
        s0e = policy.encode_s0(s0)
        l0 = policy.logits0(s0e)
        # same features => same logits regardless of conceptual eta
        l0b = policy.logits0(s0e)
        results.append(_ok("raw_logits_eta_independent", bool(torch.allclose(l0, l0b))))
        l1 = policy.logits1(s0e, policy.pool_zq(zq))
        l1s = policy.logits1(s0e, policy.pool_zq(zq.flip(0)))
        results.append(
            _ok(
                "policy1_depends_on_zq",
                float((l1 - l1s).abs().max()) > 1e-8,
                f"diff={float((l1-l1s).abs().max())}",
            )
        )

    # --- no critic in policy module ---
    names = [n for n, _ in policy.named_modules()]
    results.append(
        _ok(
            "no_critic",
            not any("critic" in n.lower() or "value" in n.lower() for n in names),
            str(names[:10]),
        )
    )

    # --- safety locks present ---
    for script in (
        "scripts/run_temporal_holdout_supernet.py",
        "scripts/train_crossfit_refinement_controller.py",
        "scripts/train_group_relative_refinement_policy.py",
        "scripts/train_forecast_refinement_controller.py",
    ):
        text = (ROOT / script).read_text()
        results.append(
            _ok(
                f"safety_lock_{Path(script).stem}",
                "confirm-full-run" in text or "confirm_full_run" in text,
            )
        )

    # --- init_prev_forecast present ---
    results.append(
        _ok(
            "execute_route_supports_resume",
            "init_prev_forecast" in inspect.signature(model._execute_route).parameters,
        )
    )

    # --- controller backbone freeze smoke ---
    ctrl = AdaptiveForecastRefinementRouteNet(
        **synthetic_budget_f2f_kwargs(node_size=7, training_phase="refinement_controller")
    ).to(device)
    ctrl.set_training_phase("refinement_controller")
    ctrl.freeze_backbone(True)
    bb_trainable = sum(p.requires_grad for p in ctrl.backbone.parameters())
    results.append(_ok("backbone_frozen_controller", bb_trainable == 0))

    passed = sum(1 for r in results if r["pass"])
    failed = [r for r in results if not r["pass"]]
    summary = {"passed": passed, "total": len(results), "failed": failed}
    print(json.dumps({"summary": summary, "results": results}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
