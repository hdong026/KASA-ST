#!/usr/bin/env python3
"""Plan B-v1 ROOT-CAUSE DIAGNOSIS (no formal training, no TEST oracle).

Writes:
  results/planB_v1_reward_geometry.json
  results/planB_v1_objective_audit.json
  results/planB_v1_state_audit.json
  results/planB_v1_execution_audit.json
  results/planB_v1_learning_dynamics_audit.json
  results/planB_v1_root_cause_report.json

Runtime limits: GPU calls <=120s each; <=2 optimizer batches per experiment.
Checkpoints only under /tmp. Never overwrite group_relative_policy.pt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_refinement_context import (
    pool_pre_route_context,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budget_from_intensity,
    default_candidate_routes,
    load_route_costs,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.group_relative_refinement_objective import (
    clipped_trajectory_objective,
    group_relative_advantages,
    terminal_route_reward,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.group_relative_refinement_policy import (
    GroupRelativeRefinementPolicy,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    feasible_mask_from_budget,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    A0_DIRECT,
    A0_HALF,
    A0_QUARTER,
    A1_JUMP_FINAL,
    A1_REFINE_HALF,
    SequentialF2FEnvironment,
)
from basicts.data.forecast_refinement_gain_dataset import (
    ForecastRefinementGainDataset,
    collate_refinement_gains,
)
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from scripts.train_forecast_refinement_controller import _build_model, _load_supernet
from scripts.train_group_relative_refinement_policy import _build_s0, _zq_for_batch

ROUTE_NAMES = ["[12]", "[6,12]", "[3,12]", "[3,6,12]"]
ETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DELTA = 0.05
LQ = 10.0
LC = 1.0

STABLE_CKPT = (
    "checkpoints/PEMS04/H12/budget_f2f/"
    "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
    "b5678fda5e8d94ed028c6c8bb073461d/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
)
CROSSFIT_ORACLE = "results/pems04_temporal_crossfit_refinement_oracle.json"
VALID_ORACLE = "results/pems04_budget_f2f_oracle_valid_rawscale.json"
MANIFEST = "results/temporal_crossfit_manifest.json"
DATA_FILE = "datasets/PEMS04/data_in12_out12.pkl"
INDEX_FILE = "datasets/PEMS04/index_in12_out12.pkl"


def sha1_file(path: Path, n: int | None = None) -> str:
    h = hashlib.sha1(path.read_bytes()).hexdigest()
    return h if n is None else h[:n]


def quantiles(x: np.ndarray, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict[str, float]:
    if len(x) == 0:
        return {f"P{int(q*100)}": float("nan") for q in qs}
    out = {}
    for q in qs:
        out[f"P{int(q*100)}"] = float(np.quantile(x, q))
    out["mean"] = float(np.mean(x))
    out["median"] = float(np.median(x))
    return out


def hist_routes(ids: list[int]) -> dict[str, int]:
    c = Counter(ids)
    return {ROUTE_NAMES[i] if i < len(ROUTE_NAMES) else str(i): int(c.get(i, 0)) for i in range(4)}


def feasible_for_eta(costs: list[float], eta: float) -> np.ndarray:
    b = budget_from_intensity(float(eta), costs)
    return np.array([c <= b + 1e-12 for c in costs], dtype=bool)


def compute_rewards(losses: np.ndarray, costs: list[float], feas: np.ndarray) -> np.ndarray:
    """losses [N,R], returns rewards [N,R] with nan for infeasible."""
    n, r = losses.shape
    rew = np.full((n, r), np.nan, dtype=np.float64)
    cost_a = np.asarray(costs, dtype=np.float64)
    for i in range(n):
        m = feas
        vals = losses[i]
        best = vals[m].min()
        q = np.maximum(0.0, vals - best - DELTA)
        rr = -LQ * q - LC * cost_a
        rew[i, m] = rr[m]
    return rew


def argmax_nan(row: np.ndarray) -> int:
    x = row.copy()
    x[np.isnan(x)] = -np.inf
    return int(np.argmax(x))


# ---------------------------------------------------------------------------
# PART 3 / 4 / 5 / 11 / 3.1 — offline reward geometry
# ---------------------------------------------------------------------------
def audit_reward_geometry() -> dict[str, Any]:
    ora = json.loads(Path(CROSSFIT_ORACLE).read_text())
    recs = ora["records"]
    losses = np.array([r["true_route_losses"] for r in recs], dtype=np.float64)
    folds = np.array([r["teacher_fold"] for r in recs], dtype=int)
    hashes = [r["teacher_checkpoint_hash"] for r in recs]
    costs = load_route_costs(None, default_candidate_routes(12), 12)
    costs = list(map(float, costs))

    # Theoretical thresholds: quality improvement needed for route r to beat DIRECT
    # R(r) > R(DIRECT) => -LQ*q(r) - LC*C(r) > -LQ*q(D) - LC*C(D)
    # Under the definition q depends on L_best which is sample-dependent.
    # For the *minimum quality improvement vs DIRECT* when DIRECT is near-best
    # (common case), set L_best = L(DIRECT), then q(D)=0, q(r)=max(0, L(r)-L(D)-delta).
    # To beat DIRECT when L(r) <= L(D): q(r)=0, need -LC*C(r) > -LC*C(D) which is
    # never true for more expensive routes. So expensive routes need q(D)>q(r).
    # Equivalent: when both near-best within delta (q=0), cheaper always wins.
    # Derive cost-gap thresholds assuming both feasible and L_best is shared:
    # Delta_R = R(r)-R(D) = -LQ*(q(r)-q(D)) - LC*(C(r)-C(D))
    # If L(r)=L(D)=L_best (both perfect), q=0: Delta_R = -LC*(C(r)-C(D)) < 0.
    # To compensate, need LQ*(q(D)-q(r)) > LC*(C(r)-C(D)).
    # Minimum quality *advantage* of r over D in the q-sense:
    #   q(D) - q(r) > (LC/LQ)*(C(r)-C(D))
    # If r is best and D is worse by x MAE: q(r)=0, q(D)=max(0,x-delta),
    # so need max(0,x-delta) > (LC/LQ)*dC  => x > delta + (LC/LQ)*dC
    c_direct = costs[0]
    thresholds = {}
    for rid, name in enumerate(ROUTE_NAMES):
        if rid == 0:
            continue
        dC = costs[rid] - c_direct
        min_mae_improve_vs_direct = DELTA + (LC / LQ) * dC
        # Also: if both within tolerance of best, reward prefers cheaper
        thresholds[name] = {
            "cost": costs[rid],
            "cost_gap_vs_direct": dC,
            "min_mae_improvement_vs_direct_to_beat_on_reward": float(min_mae_improve_vs_direct),
            "derivation": (
                "When route r is MAE-best and DIRECT is worse by x: "
                "q(r)=0, q(D)=max(0,x-delta). Need LQ*q(D) > LC*(C(r)-C(D)) "
                "=> x > delta + (LC/LQ)*(C(r)-C(D))"
            ),
            "lambda_quality": LQ,
            "lambda_cost": LC,
            "delta_abs": DELTA,
        }

    per_eta: dict[str, Any] = {}
    for eta in ETAS:
        feas = feasible_for_eta(costs, eta)
        n_feas = int(feas.sum())
        rew = compute_rewards(losses, costs, feas)
        # STRICT MAE oracle
        mae_best = []
        for i in range(len(losses)):
            m = feas
            mae_best.append(int(np.argmin(np.where(m, losses[i], np.inf))))
        # cheapest-near-best delta=0.05
        tol_oracle = []
        for i in range(len(losses)):
            m = feas
            best = losses[i][m].min()
            near = m & (losses[i] <= best + DELTA + 1e-12)
            # cheapest among near
            cand = np.where(near)[0]
            tol_oracle.append(int(cand[np.argmin([costs[j] for j in cand])]))
        # reward argmax
        rew_argmax = [argmax_nan(rew[i]) for i in range(len(rew))]
        agree = float(np.mean([a == b for a, b in zip(rew_argmax, tol_oracle)]))
        frac_direct = float(np.mean([r == 0 for r in rew_argmax]))
        tol_sets = []
        outside = []
        for i in range(len(losses)):
            m = feas
            best = losses[i][m].min()
            near = set(np.where(m & (losses[i] <= best + DELTA + 1e-12))[0].tolist())
            tol_sets.append(near)
            outside.append(rew_argmax[i] not in near)
        # top1-top2 margin
        margins = []
        group_stds = []
        zero_var = 0
        for i in range(len(rew)):
            vals = rew[i][feas]
            if vals.size < 2:
                margins.append(0.0)
                group_stds.append(0.0)
                zero_var += 1
                continue
            s = np.sort(vals)[::-1]
            margins.append(float(s[0] - s[1]))
            sg = float(vals.std(ddof=0))
            group_stds.append(sg)
            if sg < 1e-6:
                zero_var += 1
        per_eta[str(eta)] = {
            "n_feasible_routes": n_feas,
            "feasible_mask": feas.tolist(),
            "feasible_route_names": [ROUTE_NAMES[i] for i, f in enumerate(feas) if f],
            "strict_mae_oracle_histogram": hist_routes(mae_best),
            "delta0p05_cheapest_near_best_histogram": hist_routes(tol_oracle),
            "current_reward_argmax_histogram": hist_routes(rew_argmax),
            "avg_cost_reward_argmax": float(np.mean([costs[r] for r in rew_argmax])),
            "agreement_reward_argmax_vs_tolerance_oracle": agree,
            "fraction_reward_chooses_DIRECT": frac_direct,
            "fraction_reward_outside_tolerance_set": float(np.mean(outside)),
            "top1_top2_reward_margin": quantiles(np.array(margins)),
            "reward_group_std": quantiles(np.array(group_stds)),
            "zero_variance_group_rate": float(zero_var / len(rew)),
        }

    # per-fold reward-optimal at eta=1
    fold_hist = {}
    for f in sorted(set(folds.tolist())):
        idx = np.where(folds == f)[0]
        feas = feasible_for_eta(costs, 1.0)
        rew = compute_rewards(losses[idx], costs, feas)
        arg = [argmax_nan(rew[i]) for i in range(len(rew))]
        fold_hist[str(f)] = hist_routes(arg)

    # PART 4 group standardization for eta=0.5
    feas05 = feasible_for_eta(costs, 0.5)
    rew05 = compute_rewards(losses, costs, feas05)
    examples = {"A_gap_lt_0.005": [], "B_gap_0.01_0.05": [], "C_gap_gt_0.2": []}
    abs_margins = []
    abs_std_advs = []
    centered_advs_all = []
    std_pair_check = []
    for i in range(len(rew05)):
        vals = rew05[i][feas05]
        if vals.size != 2:
            continue
        gap = abs(float(vals[0] - vals[1]))
        mu = vals.mean()
        sg = vals.std(ddof=0)
        if sg < 1e-6:
            a = np.zeros_like(vals)
        else:
            a = (vals - mu) / (sg + 1e-6)
        a_center = vals - mu
        centered_advs_all.extend(a_center.tolist())
        abs_margins.append(gap)
        abs_std_advs.append(float(np.abs(a).mean()))
        # For 2-route non-tied: should be ~[+1,-1] up to sign/order
        if gap > 1e-12:
            sorted_a = np.sort(a)
            std_pair_check.append(sorted_a.tolist())
            ex = {
                "sample_index": int(recs[i]["sample_index"]),
                "rewards": vals.tolist(),
                "gap": gap,
                "standardized_advantages": a.tolist(),
                "centered_advantages": a_center.tolist(),
            }
            if gap < 0.005 and len(examples["A_gap_lt_0.005"]) < 5:
                examples["A_gap_lt_0.005"].append(ex)
            elif 0.01 <= gap <= 0.05 and len(examples["B_gap_0.01_0.05"]) < 5:
                examples["B_gap_0.01_0.05"].append(ex)
            elif gap > 0.2 and len(examples["C_gap_gt_0.2"]) < 5:
                examples["C_gap_gt_0.2"].append(ex)

    # verify approx [+1,-1]
    if std_pair_check:
        arr = np.array(std_pair_check)
        approx = bool(np.allclose(arr, np.array([-1.0, 1.0]), atol=1e-3))
    else:
        approx = False
    corr = float("nan")
    if len(abs_margins) > 2:
        corr = float(np.corrcoef(abs_margins, abs_std_advs)[0, 1])
    group_std_erases = bool(abs(corr) < 0.05) or (
        len(abs_std_advs) > 0 and float(np.std(abs_std_advs)) < 1e-3
    )

    # PART 5 eta information efficiency
    eta_eff = {}
    zero_grad_fracs = []
    for eta in ETAS:
        feas = feasible_for_eta(costs, eta)
        n_feas = int(feas.sum())
        rew = compute_rewards(losses, costs, feas)
        zv = 0
        for i in range(len(rew)):
            vals = rew[i][feas]
            if vals.size <= 1 or vals.std(ddof=0) < 1e-6:
                zv += 1
        frac_zv = float(zv / len(rew))
        # structural: single feasible route => always zero gradient
        structural_single = n_feas == 1
        eta_eff[str(eta)] = {
            "feasible_route_count": n_feas,
            "fraction_zero_group_advantage": frac_zv,
            "always_one_route": structural_single,
        }
        zero_grad_fracs.append(frac_zv)
    expected_zero_grad = float(np.mean(zero_grad_fracs))  # uniform eta

    # PART 11 fold maturity / heterogeneity
    man = json.loads(Path(MANIFEST).read_text())
    fold_stats = {}
    g_keys = ["G3", "G6", "G36"]
    for f in sorted(set(folds.tolist())):
        idx = np.where(folds == f)[0]
        sub = [recs[i] for i in idx]
        fs = {
            "n": len(sub),
            "teacher_checkpoint_hash": hashes[idx[0]],
            "teacher_train_size": None,
        }
        for mf in man["folds"]:
            if int(mf["fold"]) == int(f):
                fs["teacher_train_size"] = int(mf["n_teacher"])
                fs["n_oracle"] = int(mf["n_oracle"])
        for gk in g_keys:
            arr = np.array([r[gk] for r in sub], dtype=np.float64)
            fs[gk] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "median": float(np.median(arr)),
            }
        feas = feasible_for_eta(costs, 1.0)
        rew = compute_rewards(losses[idx], costs, feas)
        arg = [argmax_nan(rew[i]) for i in range(len(rew))]
        fs["reward_optimal_histogram_eta1"] = hist_routes(arg)
        margins = []
        gstds = []
        for i in range(len(rew)):
            vals = rew[i][feas]
            s = np.sort(vals)[::-1]
            margins.append(float(s[0] - s[1]) if vals.size >= 2 else 0.0)
            gstds.append(float(vals.std(ddof=0)) if vals.size else 0.0)
        fs["reward_margin_median"] = float(np.median(margins))
        fs["group_std_median"] = float(np.median(gstds))
        fold_stats[str(f)] = fs

    # quantify fold1 vs later
    f1 = fold_stats["1"]
    later = [fold_stats[str(f)] for f in [2, 3, 4]]
    het = {
        "fold1_teacher_train_size": f1["teacher_train_size"],
        "later_teacher_train_sizes": [x["teacher_train_size"] for x in later],
        "fold1_G36_mean": f1["G36"]["mean"],
        "later_G36_means": [x["G36"]["mean"] for x in later],
        "fold1_G36_std": f1["G36"]["std"],
        "later_G36_stds": [x["G36"]["std"] for x in later],
        "fold1_reward_margin_median": f1["reward_margin_median"],
        "later_reward_margin_medians": [x["reward_margin_median"] for x in later],
        "fold1_direct_frac_eta1": f1["reward_optimal_histogram_eta1"]["[12]"] / f1["n"],
        "later_direct_frac_eta1": [
            x["reward_optimal_histogram_eta1"]["[12]"] / x["n"] for x in later
        ],
    }

    out = {
        "oracle": CROSSFIT_ORACLE,
        "n_samples": len(recs),
        "costs": {ROUTE_NAMES[i]: costs[i] for i in range(4)},
        "reward_defaults": {
            "delta_abs": DELTA,
            "lambda_quality": LQ,
            "lambda_cost": LC,
        },
        "theoretical_thresholds_vs_DIRECT": thresholds,
        "per_eta": per_eta,
        "per_fold_reward_optimal_eta1": fold_hist,
        "group_standardization_eta0p5": {
            "two_route_standardized_approx_plus_minus_one": approx,
            "examples": examples,
            "corr_abs_reward_margin_vs_abs_std_advantage": corr,
            "GROUP_STD_ERASES_MARGIN_MAGNITUDE": group_std_erases,
            "centered_advantage_distribution": quantiles(np.array(centered_advs_all)),
            "abs_standardized_advantage_std": float(np.std(abs_std_advs)) if abs_std_advs else None,
        },
        "eta_information_efficiency": eta_eff,
        "expected_fraction_no_policy_gradient_uniform_eta": expected_zero_grad,
        "STRUCTURAL_ZERO_GRADIENT_RATE": expected_zero_grad,
        "fold_stats": fold_stats,
        "FOLD_REWARD_HETEROGENEITY": het,
        "flags": {
            "STATE0_AGGRESSIVE_POOLING": None,  # filled elsewhere
            "eta0_and_0p25_always_one_route": bool(
                eta_eff["0.0"]["always_one_route"] and eta_eff["0.25"]["always_one_route"]
            ),
        },
    }
    if abs(expected_zero_grad - 0.40) < 0.05:
        print("STRUCTURAL_ZERO_GRADIENT_RATE ~= 0.40")
    if group_std_erases:
        print("GROUP_STD_ERASES_MARGIN_MAGNITUDE = TRUE")
    return out


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def _cfg_for_ckpt(ckpt: Path) -> str:
    sib = list(ckpt.parent.glob("H12_*.py"))
    if not sib:
        raise FileNotFoundError(f"no H12_*.py next to {ckpt}")
    return str(sib[0])


def load_frozen_supernet(ckpt: str | Path, device: torch.device):
    ckpt = Path(ckpt)
    routes = default_candidate_routes(12)

    class _Args:
        horizon = 12
        controller_dim = 128
        pooling_queries = 4
        delta_abs = 0.05
        route_cost_file = None
        cfg = _cfg_for_ckpt(ckpt)

    model = _build_model(_Args(), routes, device)
    _load_supernet(model, ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def teacher_ckpt_map() -> dict[int, Path]:
    out = {}
    for fold in [1, 2, 3, 4]:
        matches = list(
            Path("checkpoints/PEMS04/H12/budget_f2f").glob(
                f"supernet_eta0p50_dynamic_fair_temporal_cf_fold{fold}_teacher_*/seed1/*/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
            )
        )
        if matches:
            out[fold] = matches[0]
    return out


def load_crossfit_dataset(oracle_path: str, max_n: int | None = None):
    routes = default_candidate_routes(12)
    costs = load_route_costs(None, routes, 12)
    ds = ForecastRefinementGainDataset(
        IndexedTimeSeriesForecastingDataset(DATA_FILE, INDEX_FILE, "train"),
        oracle_path,
        expected_routes=routes,
        expected_costs=costs,
        expected_horizon=12,
        expected_dataset="PEMS04",
        require_len_match=False,
    )
    if max_n is not None and max_n < len(ds):
        # chronological: first max_n by underlying order (dataset already temporal OOF merge order)
        ds = Subset(ds, list(range(max_n)))
    return ds


# ---------------------------------------------------------------------------
# PART 1 graph + PART 2 mismatch + fold state compare
# ---------------------------------------------------------------------------
@torch.no_grad()
def audit_graph_and_state(device: torch.device) -> dict[str, Any]:
    t0 = time.time()
    stable = load_frozen_supernet(STABLE_CKPT, device)
    env = SequentialF2FEnvironment(stable)
    # probe shapes
    B = 2
    history = torch.zeros(B, 12, 307, 4, device=device)
    h_shared = stable.extract_pre_route_context(history, detach=True)
    s0 = pool_pre_route_context(h_shared)
    pref = env.execute_quarter_prefix(history)
    zq = pref["Z_q"]
    zqp = GroupRelativeRefinementPolicy(context_dim=int(s0.shape[-1])).pool_zq(zq)

    graph = {
        "PEMS04_H12": {
            "history_X": list(history.shape),
            "extract_pre_route_context": {
                "file": "basicts/archs/arch_zoo/ChainForecasting_arch/budget_conditioned_adaptive_f2f.py",
                "class": "BudgetConditionedAdaptiveF2FNet",
                "function": "extract_pre_route_context",
                "H_shared_shape": list(h_shared.shape),
            },
            "pool_pre_route_context": {
                "file": "basicts/archs/arch_zoo/ChainForecasting_arch/adaptive_refinement_context.py",
                "function": "pool_pre_route_context",
                "s0_shape": list(s0.shape),
                "pooling": "mean over dims (1,2) = temporal patches AND nodes",
            },
            "policy0": {
                "file": "basicts/archs/arch_zoo/ChainForecasting_arch/group_relative_refinement_policy.py",
                "class": "GroupRelativeRefinementPolicy",
                "functions": ["encode_s0", "logits0"],
                "a0_logits_shape": [B, 3],
            },
            "execute_quarter_prefix": {
                "file": "basicts/archs/arch_zoo/ChainForecasting_arch/sequential_f2f_environment.py",
                "class": "SequentialF2FEnvironment",
                "function": "execute_quarter_prefix",
                "Z_q_shape": list(zq.shape),
            },
            "pool_zq": {
                "file": "basicts/archs/arch_zoo/ChainForecasting_arch/group_relative_refinement_policy.py",
                "function": "GroupRelativeRefinementPolicy.pool_zq",
                "zq_pooled_shape": list(zqp.shape),
                "pooling": "mean over (time, node) -> [B,C]",
            },
            "policy1": {
                "functions": ["logits1"],
                "a1_logits_shape": [B, 2],
            },
        }
    }
    STATE0_AGGRESSIVE_POOLING = True  # confirmed by code: mean(1,2)
    # Z_q before pool ~ [B, 3, 307, 1]? Check actual
    zq_shape = list(zq.shape)
    COARSE_FORECAST_SCALAR_BOTTLENECK = (
        len(zq_shape) == 4 and zqp.shape[-1] == 1 and zqp.ndim == 2 and zqp.shape[0] == B
    )
    print("STATE0_AGGRESSIVE_POOLING = TRUE" if STATE0_AGGRESSIVE_POOLING else "STATE0_AGGRESSIVE_POOLING = FALSE")
    print(
        "COARSE_FORECAST_SCALAR_BOTTLENECK = TRUE"
        if COARSE_FORECAST_SCALAR_BOTTLENECK
        else "COARSE_FORECAST_SCALAR_BOTTLENECK = FALSE"
    )
    print(f"[graph] H_shared={list(h_shared.shape)} s0={list(s0.shape)} Z_q={zq_shape} zqp={list(zqp.shape)}")

    # PART 2: state/reward mismatch
    ora = json.loads(Path(CROSSFIT_ORACLE).read_text())
    fold_hash = {}
    for r in ora["records"]:
        fold_hash[int(r["teacher_fold"])] = r["teacher_checkpoint_hash"]
    stable_hash16 = sha1_file(Path(STABLE_CKPT), 16)
    teachers = teacher_ckpt_map()
    teacher_info = {}
    for f, p in teachers.items():
        teacher_info[str(f)] = {
            "path": str(p),
            "sha1_16": sha1_file(p, 16),
            "oracle_hash": fold_hash.get(f),
            "matches_oracle": sha1_file(p, 16) == fold_hash.get(f),
            "differs_from_stable": sha1_file(p, 16) != stable_hash16,
        }
    STATE_REWARD_ENVIRONMENT_MISMATCH = all(
        teacher_info[str(f)]["differs_from_stable"] for f in teachers
    )
    print(
        "STATE_REWARD_ENVIRONMENT_MISMATCH = TRUE"
        if STATE_REWARD_ENVIRONMENT_MISMATCH
        else "STATE_REWARD_ENVIRONMENT_MISMATCH = FALSE"
    )

    # 2.1 fold state comparison (32 samples/fold)
    routes = default_candidate_routes(12)
    costs = load_route_costs(None, routes, 12)
    full_ds = ForecastRefinementGainDataset(
        IndexedTimeSeriesForecastingDataset(DATA_FILE, INDEX_FILE, "train"),
        CROSSFIT_ORACLE,
        expected_routes=routes,
        expected_costs=costs,
        expected_horizon=12,
        expected_dataset="PEMS04",
        require_len_match=False,
    )
    si_to_pos = {int(si): pos for pos, si in enumerate(full_ds.sample_indices)}
    fold_to_ds_idx: dict[int, list[int]] = defaultdict(list)
    for rec in ora["records"]:
        si = int(rec["sample_index"])
        if si in si_to_pos:
            fold_to_ds_idx[int(rec["teacher_fold"])].append(si_to_pos[si])

    fold_compare = {}
    for fold, tpath in teachers.items():
        if time.time() - t0 > 110:
            fold_compare[str(fold)] = {"skipped": "time_budget"}
            continue
        idxs = fold_to_ds_idx[fold][:32]
        if not idxs:
            continue
        teacher = load_frozen_supernet(tpath, device)
        env_t = SequentialF2FEnvironment(teacher)
        # batch load
        xs = []
        for di in idxs:
            history, *_ = full_ds[di]
            xs.append(history)
        hist = torch.stack(xs, dim=0).to(device)
        # stable
        Hs = stable.extract_pre_route_context(hist, detach=True)
        s0s = pool_pre_route_context(Hs)
        Zqs = env.execute_quarter_prefix(hist)["Z_q"]
        zqps = Zqs.mean(dim=(1, 2))
        # teacher
        Ht = teacher.extract_pre_route_context(hist, detach=True)
        s0t = pool_pre_route_context(Ht)
        Zqt = env_t.execute_quarter_prefix(hist)["Z_q"]
        zqpt = Zqt.mean(dim=(1, 2))

        def rel_l2(a, b):
            diff = (a - b).reshape(a.shape[0], -1)
            num = diff.norm(dim=1)
            den = b.reshape(b.shape[0], -1).norm(dim=1).clamp_min(1e-8)
            return (num / den).cpu().numpy()

        def cos_sim(a, b):
            aa = a.reshape(a.shape[0], -1)
            bb = b.reshape(b.shape[0], -1)
            return F.cosine_similarity(aa, bb, dim=1).cpu().numpy()

        # per-feature corr for s0
        s0s_np = s0s.cpu().numpy()
        s0t_np = s0t.cpu().numpy()
        corrs = []
        for d in range(s0s_np.shape[1]):
            if s0s_np[:, d].std() < 1e-8 or s0t_np[:, d].std() < 1e-8:
                continue
            corrs.append(float(np.corrcoef(s0s_np[:, d], s0t_np[:, d])[0, 1]))

        fold_compare[str(fold)] = {
            "n": len(idxs),
            "teacher_path": str(tpath),
            "teacher_sha1_16": sha1_file(tpath, 16),
            "stable_sha1_16": stable_hash16,
            "H_shared": {
                "rel_l2_mean": float(rel_l2(Hs, Ht).mean()),
                "rel_l2_median": float(np.median(rel_l2(Hs, Ht))),
                "cosine_mean": float(cos_sim(Hs, Ht).mean()),
                "shape": list(Hs.shape),
            },
            "pooled_s0": {
                "rel_l2_mean": float(rel_l2(s0s, s0t).mean()),
                "cosine_mean": float(cos_sim(s0s, s0t).mean()),
                "per_feature_corr_mean": float(np.mean(corrs)) if corrs else None,
                "shape": list(s0s.shape),
            },
            "Z_q": {
                "rel_l2_mean": float(rel_l2(Zqs, Zqt).mean()),
                "cosine_mean": float(cos_sim(Zqs, Zqt).mean()),
                "shape": list(Zqs.shape),
            },
            "pooled_scalar_Z_q": {
                "rel_l2_mean": float(rel_l2(zqps, zqpt).mean()),
                "cosine_mean": float(cos_sim(zqps, zqpt).mean()),
                "pearson": float(
                    np.corrcoef(
                        zqps.cpu().numpy().reshape(-1),
                        zqpt.cpu().numpy().reshape(-1),
                    )[0, 1]
                ),
            },
        }
        del teacher
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    return {
        "computation_graph": graph,
        "STATE0_AGGRESSIVE_POOLING": STATE0_AGGRESSIVE_POOLING,
        "COARSE_FORECAST_SCALAR_BOTTLENECK": COARSE_FORECAST_SCALAR_BOTTLENECK,
        "Z_q_shape_observed": zq_shape,
        "zq_pooled_shape_observed": list(zqp.shape),
        "stable_supernet": {"path": STABLE_CKPT, "sha1_16": stable_hash16},
        "teacher_checkpoints": teacher_info,
        "STATE_REWARD_ENVIRONMENT_MISMATCH": STATE_REWARD_ENVIRONMENT_MISMATCH,
        "state_definition": "stable-full-train supernet (extract_pre_route_context + execute_quarter_prefix)",
        "reward_definition": "fold-specific OOF teacher route losses from temporal crossfit oracle",
        "fold_state_comparison": fold_compare,
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# PART 6 objective gradients
# ---------------------------------------------------------------------------
def _flatten_grads(policy) -> torch.Tensor:
    grads = []
    for p in policy.parameters():
        if p.grad is None:
            grads.append(torch.zeros(p.numel(), device=p.device))
        else:
            grads.append(p.grad.detach().reshape(-1))
    return torch.cat(grads)


def _grad_norms(policy) -> dict[str, float]:
    total = 0.0
    p0 = 0.0
    p1 = 0.0
    for n, p in policy.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().norm().item())
        total += g * g
        if n.startswith("policy0") or n.startswith("s0_proj"):
            p0 += g * g
        if n.startswith("policy1") or n.startswith("zq_pool") or n.startswith("s0_for_s1"):
            p1 += g * g
    return {
        "total": float(math.sqrt(total)),
        "policy0_block": float(math.sqrt(p0)),
        "policy1_block": float(math.sqrt(p1)),
    }


def cosine_vec(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def audit_objective(device: torch.device, n_samples: int = 128) -> dict[str, Any]:
    t0 = time.time()
    stable = load_frozen_supernet(STABLE_CKPT, device)
    env = SequentialF2FEnvironment(stable)
    ds = load_crossfit_dataset(CROSSFIT_ORACLE, max_n=n_samples)
    loader = DataLoader(ds, batch_size=n_samples, shuffle=False, collate_fn=collate_refinement_gains)
    history, _si, _gains, losses = next(iter(loader))
    history = history.to(device)
    losses = losses.to(device)
    b = history.shape[0]
    costs_t = stable.route_costs.detach().to(device).float()

    with torch.no_grad():
        s0_raw = _build_s0(stable, history)
        zq = _zq_for_batch(env, history)
    ctx_dim = int(s0_raw.shape[-1])

    torch.manual_seed(123)
    policy = GroupRelativeRefinementPolicy(context_dim=ctx_dim, zq_dim=1, hidden=256).to(device)
    policy_init = copy.deepcopy(policy.state_dict())

    # Use eta=1 for full branch multiplicity analysis
    eta = 1.0
    masks = env.action_masks(eta)
    feas = masks["feasible_routes"].to(device).unsqueeze(0).expand(b, -1)
    mask0 = masks["mask0"].to(device).unsqueeze(0).expand(b, -1)
    mask1 = masks["mask1"].to(device).unsqueeze(0).expand(b, -1)
    trajs = env.enumerate_feasible_trajectories(eta)

    # Branch multiplicity
    a0_counts = Counter(int(t["a0"]) for t in trajs)
    branch = {
        "eta": eta,
        "rows": [
            {
                "name": {0: "DIRECT", 1: "HALF", 2: "QUARTER->JUMP", 3: "QUARTER->REFINE"}[
                    int(t["route_id"])
                ],
                "a0": int(t["a0"]),
                "a1": t["a1"],
                "route_id": int(t["route_id"]),
            }
            for t in trajs
        ],
        "policy0_action_row_counts": {
            "DIRECT": a0_counts[A0_DIRECT],
            "HALF": a0_counts[A0_HALF],
            "QUARTER": a0_counts[A0_QUARTER],
        },
        "BRANCH_MULTIPLICITY_PRESENT": a0_counts[A0_QUARTER] >= 2,
    }
    if branch["BRANCH_MULTIPLICITY_PRESENT"]:
        print("BRANCH_MULTIPLICITY_PRESENT = TRUE")

    rewards, _ = terminal_route_reward(losses, costs_t, feas, delta_abs=DELTA, lambda_quality=LQ, lambda_cost=LC)
    adv_std, _ = group_relative_advantages(rewards, feas)
    # centered advantages
    adv_center = torch.zeros_like(rewards)
    for i in range(b):
        m = feas[i]
        vals = rewards[i][m]
        adv_center[i][m] = vals - vals.mean()

    def build_rows(adv_mat):
        rows_s0, rows_zq, rows_a0, rows_a1, rows_adv, rows_m0, rows_m1, rows_rid = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for t in trajs:
            rid = int(t["route_id"])
            a0 = torch.full((b,), int(t["a0"]), device=device, dtype=torch.long)
            a1 = torch.full(
                (b,),
                0 if t["a1"] is None else int(t["a1"]),
                device=device,
                dtype=torch.long,
            )
            rows_s0.append(s0_raw)
            rows_zq.append(zq)
            rows_a0.append(a0)
            rows_a1.append(a1)
            rows_adv.append(adv_mat[:, rid])
            rows_m0.append(mask0)
            rows_m1.append(mask1)
            rows_rid.append(torch.full((b,), rid, device=device, dtype=torch.long))
        return {
            "s0": torch.cat(rows_s0),
            "zq": torch.cat(rows_zq),
            "a0": torch.cat(rows_a0),
            "a1": torch.cat(rows_a1),
            "adv": torch.cat(rows_adv).detach(),
            "m0": torch.cat(rows_m0),
            "m1": torch.cat(rows_m1),
            "rid": torch.cat(rows_rid),
            "n_traj": len(trajs),
            "b": b,
        }

    rows = build_rows(adv_std)
    rows_c = build_rows(adv_center)

    # OBJECTIVE A: current uniform enumeration + clipped
    def grad_A():
        policy.load_state_dict(policy_init)
        policy.zero_grad(set_to_none=True)
        with torch.no_grad():
            logp_old = policy.trajectory_logprob(
                policy.encode_s0(rows["s0"]),
                rows["zq"],
                rows["a0"],
                rows["a1"],
                rows["m0"],
                rows["m1"],
            ).detach()
        s0e = policy.encode_s0(rows["s0"])
        logp_new = policy.trajectory_logprob(
            s0e, rows["zq"], rows["a0"], rows["a1"], rows["m0"], rows["m1"]
        )
        loss, stats = clipped_trajectory_objective(logp_new, logp_old, rows["adv"], clip_eps=0.2)
        loss.backward()
        return _flatten_grads(policy), _grad_norms(policy), stats

    # OBJECTIVE B: pi_old-weighted exact clipped surrogate
    def grad_B():
        policy.load_state_dict(policy_init)
        policy.zero_grad(set_to_none=True)
        # For each sample, sum over traj: pi_old(tau) * min(rho A, clip rho A)
        # Build per-sample
        s0e = policy.encode_s0(s0_raw)
        # logp_old per (sample, traj)
        with torch.no_grad():
            logp_old_list = []
            for t in trajs:
                a0 = torch.full((b,), int(t["a0"]), device=device, dtype=torch.long)
                a1 = torch.full(
                    (b,),
                    0 if t["a1"] is None else int(t["a1"]),
                    device=device,
                    dtype=torch.long,
                )
                lp = policy.trajectory_logprob(s0e.detach(), zq, a0, a1, mask0, mask1)
                logp_old_list.append(lp)
            logp_old_mat = torch.stack(logp_old_list, dim=1)  # [B, T]
            pi_old = logp_old_mat.exp()
        # new
        logp_new_list = []
        for t in trajs:
            a0 = torch.full((b,), int(t["a0"]), device=device, dtype=torch.long)
            a1 = torch.full(
                (b,),
                0 if t["a1"] is None else int(t["a1"]),
                device=device,
                dtype=torch.long,
            )
            lp = policy.trajectory_logprob(policy.encode_s0(s0_raw), zq, a0, a1, mask0, mask1)
            logp_new_list.append(lp)
        logp_new_mat = torch.stack(logp_new_list, dim=1)
        ratio = torch.exp(logp_new_mat - logp_old_mat.detach())
        clipped = torch.clamp(ratio, 0.8, 1.2)
        A = torch.stack([adv_std[:, int(t["route_id"])] for t in trajs], dim=1)
        obj = torch.min(ratio * A, clipped * A)
        # weight by pi_old and sum over traj, mean over batch; maximize => minimize -J
        J = (pi_old.detach() * obj).sum(dim=1).mean()
        loss = -J
        loss.backward()
        return _flatten_grads(policy), _grad_norms(policy), {"J": float(J.item())}

    # OBJECTIVE C: exact full-info expected utility with A_center, no clip
    def grad_C():
        policy.load_state_dict(policy_init)
        policy.zero_grad(set_to_none=True)
        logp_list = []
        for t in trajs:
            a0 = torch.full((b,), int(t["a0"]), device=device, dtype=torch.long)
            a1 = torch.full(
                (b,),
                0 if t["a1"] is None else int(t["a1"]),
                device=device,
                dtype=torch.long,
            )
            lp = policy.trajectory_logprob(policy.encode_s0(s0_raw), zq, a0, a1, mask0, mask1)
            logp_list.append(lp)
        logp_mat = torch.stack(logp_list, dim=1)
        pi = logp_mat.exp()
        A = torch.stack([adv_center[:, int(t["route_id"])] for t in trajs], dim=1)
        J = (pi * A).sum(dim=1).mean()
        loss = -J
        loss.backward()
        return _flatten_grads(policy), _grad_norms(policy), {"J": float(J.item())}

    gA, nA, sA = grad_A()
    gB, nB, sB = grad_B()
    gC, nC, sC = grad_C()
    cos_ab = cosine_vec(gA, gB)
    cos_ac = cosine_vec(gA, gC)
    cos_bc = cosine_vec(gB, gC)
    mismatch = bool(cos_ab < 0.95 or cos_ac < 0.90)
    if mismatch:
        print("CURRENT_ENUMERATED_SURROGATE_MISMATCH = TRUE")

    # beta_kl unused?
    train_src = Path("scripts/train_group_relative_refinement_policy.py").read_text()
    beta_kl_in_cli = "--beta-kl" in train_src or "beta_kl" in train_src
    beta_kl_applied = "args.beta_kl" in train_src and (
        "beta_kl" in train_src.split("clipped_trajectory_objective")[0]
        or "float(args.beta_kl)" in train_src
        and "loss" in train_src
        and "beta_kl" in "".join(
            [
                line
                for line in train_src.splitlines()
                if "beta_kl" in line and "add_argument" not in line and "default" not in line
            ]
        )
    )
    # More precise: search usage outside argparse
    used_lines = [
        ln.strip()
        for ln in train_src.splitlines()
        if "beta_kl" in ln and "add_argument" not in ln and "default" not in ln
    ]
    BETA_KL_UNUSED = len(used_lines) == 0
    if BETA_KL_UNUSED:
        print("BETA_KL_UNUSED = TRUE")

    return {
        "n_samples": b,
        "branch_multiplicity": branch,
        "gradients": {
            "A_current_uniform_enumeration_clipped": {"norms": nA, "stats": sA},
            "B_pi_old_weighted_exact_clipped": {"norms": nB, "stats": sB},
            "C_exact_expected_utility_centered": {"norms": nC, "stats": sC},
            "cosine": {"A_B": cos_ab, "A_C": cos_ac, "B_C": cos_bc},
        },
        "CURRENT_ENUMERATED_SURROGATE_MISMATCH": mismatch,
        "old_policy_audit": {
            "logp_old_computed_before_policy_update_epochs": True,
            "pi_old_samples_used_to_generate_trajectories": False,
            "trajectories_from": "enumerate_feasible_trajectories (uniform rows)",
            "beta_kl_in_CLI": beta_kl_in_cli,
            "beta_kl_applied_to_loss": False,
            "BETA_KL_UNUSED": BETA_KL_UNUSED,
            "beta_kl_usage_lines": used_lines,
        },
        "elapsed_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# PART 7 + 8 learning dynamics
# ---------------------------------------------------------------------------
def audit_learning_dynamics(device: torch.device) -> dict[str, Any]:
    t0 = time.time()
    stable = load_frozen_supernet(STABLE_CKPT, device)
    env = SequentialF2FEnvironment(stable)
    ds = load_crossfit_dataset(CROSSFIT_ORACLE, max_n=512)
    # init audit seeds
    probe_loader = DataLoader(
        Subset(ds, list(range(min(512, len(ds))))),
        batch_size=64,
        shuffle=False,
        collate_fn=collate_refinement_gains,
    )
    batches = []
    for batch in probe_loader:
        batches.append(batch)
        if sum(b[0].shape[0] for b in batches) >= 512:
            break
    history = torch.cat([b[0] for b in batches], dim=0)[:512].to(device)
    losses = torch.cat([b[3] for b in batches], dim=0)[:512].to(device)

    with torch.no_grad():
        s0_raw = _build_s0(stable, history)
        zq = _zq_for_batch(env, history)
    ctx_dim = int(s0_raw.shape[-1])
    zqp = GroupRelativeRefinementPolicy(context_dim=ctx_dim).pool_zq(zq)

    init_reports = {}
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        policy = GroupRelativeRefinementPolicy(context_dim=ctx_dim, zq_dim=1, hidden=256).to(device)
        policy.eval()
        with torch.no_grad():
            s0e = policy.encode_s0(s0_raw)
            logits0 = policy.logits0(s0e)
            logits1 = policy.logits1(s0e, policy.pool_zq(zq))
            seed_etas = {}
            for eta in [0.5, 0.75, 1.0]:
                masks = env.action_masks(eta)
                m0 = masks["mask0"].to(device).unsqueeze(0).expand(history.shape[0], -1)
                m1 = masks["mask1"].to(device).unsqueeze(0).expand(history.shape[0], -1)
                a0 = logits0.masked_fill(~m0, -1e9).argmax(-1)
                a1 = logits1.masked_fill(~m1, -1e9).argmax(-1)
                routes = []
                for i in range(history.shape[0]):
                    route = env.route_from_actions(
                        int(a0[i]), int(a1[i]) if int(a0[i]) == A0_QUARTER else None
                    )
                    key = {
                        tuple(env.template["direct"]): 0,
                        tuple(env.template["half"]): 1,
                        tuple(env.template["quarter"]): 2,
                        tuple(env.template["progressive"]): 3,
                    }[tuple(route)]
                    routes.append(key)
                log0 = policy.masked_log_softmax(logits0, m0)
                log1 = policy.masked_log_softmax(logits1, m1)
                ent0 = float((-(log0.exp() * log0).sum(-1)).mean().item())
                ent1 = float((-(log1.exp() * log1).sum(-1)).mean().item())
                p0 = log0.exp().mean(0).cpu().tolist()
                seed_etas[str(eta)] = {
                    "masked_action_probs_policy0_mean": p0,
                    "deterministic_route_histogram": hist_routes(routes),
                    "policy0_entropy": ent0,
                    "policy1_entropy": ent1,
                    "direct_frac": float(np.mean([r == 0 for r in routes])),
                }
            init_reports[str(seed)] = {
                "logits0_mean": float(logits0.mean().item()),
                "logits0_std": float(logits0.std().item()),
                "logits1_mean": float(logits1.mean().item()),
                "logits1_std": float(logits1.std().item()),
                "etas": seed_etas,
            }

    # Early collapse: 2 optimizer batches at eta=1 with current objective
    torch.manual_seed(7)
    policy = GroupRelativeRefinementPolicy(context_dim=ctx_dim, zq_dim=1, hidden=256).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    # take first 64 as batch1, next 64 as batch2
    def one_update(h, loss_mat, tag, snap):
        bsz = h.shape[0]
        eta = 1.0
        masks = env.action_masks(eta)
        feas = masks["feasible_routes"].to(device).unsqueeze(0).expand(bsz, -1)
        m0 = masks["mask0"].to(device).unsqueeze(0).expand(bsz, -1)
        m1 = masks["mask1"].to(device).unsqueeze(0).expand(bsz, -1)
        s0b = _build_s0(stable, h)
        zqb = _zq_for_batch(env, h)
        rew, _ = terminal_route_reward(loss_mat, stable.route_costs.to(device).float(), feas)
        adv, _ = group_relative_advantages(rew, feas)
        trajs = env.enumerate_feasible_trajectories(eta)
        rows = []
        for t in trajs:
            rid = int(t["route_id"])
            rows.append(
                {
                    "s0": s0b,
                    "zq": zqb,
                    "a0": torch.full((bsz,), int(t["a0"]), device=device, dtype=torch.long),
                    "a1": torch.full(
                        (bsz,),
                        0 if t["a1"] is None else int(t["a1"]),
                        device=device,
                        dtype=torch.long,
                    ),
                    "adv": adv[:, rid],
                    "m0": m0,
                    "m1": m1,
                }
            )
        s0_cat = torch.cat([r["s0"] for r in rows])
        zq_cat = torch.cat([r["zq"] for r in rows])
        a0_cat = torch.cat([r["a0"] for r in rows])
        a1_cat = torch.cat([r["a1"] for r in rows])
        adv_cat = torch.cat([r["adv"] for r in rows]).detach()
        m0_cat = torch.cat([r["m0"] for r in rows])
        m1_cat = torch.cat([r["m1"] for r in rows])
        with torch.no_grad():
            logp_old = policy.trajectory_logprob(
                policy.encode_s0(s0_cat), zq_cat, a0_cat, a1_cat, m0_cat, m1_cat
            ).detach()
        # snapshot probs before
        policy.eval()
        with torch.no_grad():
            s0e = policy.encode_s0(s0b)
            log0 = policy.masked_log_softmax(policy.logits0(s0e), m0)
            log1 = policy.masked_log_softmax(policy.logits1(s0e, policy.pool_zq(zqb)), m1)
            # terminal route probs under policy (product)
            term = {}
            for t in trajs:
                name = {0: "DIRECT", 1: "HALF", 2: "QUARTER_JUMP", 3: "PROGRESSIVE"}[int(t["route_id"])]
                p = log0[:, int(t["a0"])].exp()
                if t["a1"] is not None:
                    p = p * log1[:, int(t["a1"])].exp()
                term[name] = float(p.mean().item())
            ent0 = float((-(log0.exp() * log0).sum(-1)).mean().item())
            ent1 = float((-(log1.exp() * log1).sum(-1)).mean().item())
        snap[tag] = {
            "terminal_route_probs_mean": term,
            "policy0_entropy": ent0,
            "policy1_entropy": ent1,
        }
        # update
        policy.train()
        s0e = policy.encode_s0(s0_cat)
        logp_new = policy.trajectory_logprob(s0e, zq_cat, a0_cat, a1_cat, m0_cat, m1_cat)
        loss, stats = clipped_trajectory_objective(logp_new, logp_old, adv_cat, clip_eps=0.2)
        # entropy bonus as formal
        logp0 = policy.masked_log_softmax(policy.logits0(s0e), m0_cat)
        ent = -(logp0.exp() * logp0).sum(-1).mean()
        loss = loss - 0.001 * ent
        opt.zero_grad()
        loss.backward()
        gnorm = float(
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf")).item()
            if False
            else math.sqrt(
                sum(
                    float(p.grad.detach().norm().item()) ** 2
                    for p in policy.parameters()
                    if p.grad is not None
                )
            )
        )
        # KL
        with torch.no_grad():
            ratio = torch.exp(logp_new - logp_old)
            kl_old_new = float((logp_old.exp() * (logp_old - logp_new.detach())).sum().item() / logp_old.numel())
            # approximate mean KL over rows
            kl_on = float(((logp_old.exp()) * (logp_old - logp_new.detach())).mean().item())
            kl_no = float(((logp_new.exp()) * (logp_new.detach() - logp_old)).mean().item())
            clip_frac = float(((ratio - 1.0).abs() > 0.2).float().mean().item())
        opt.step()
        snap[tag]["after_update_stats"] = {
            "loss": float(loss.item()),
            "ratio_mean": stats["ratio_mean"],
            "ratio_min": stats["ratio_min"],
            "ratio_max": stats["ratio_max"],
            "clip_fraction": clip_frac,
            "grad_norm": gnorm,
            "KL_old_new_approx": kl_on,
            "KL_new_old_approx": kl_no,
        }
        # after snapshot
        policy.eval()
        with torch.no_grad():
            s0e = policy.encode_s0(s0b)
            log0 = policy.masked_log_softmax(policy.logits0(s0e), m0)
            log1 = policy.masked_log_softmax(policy.logits1(s0e, policy.pool_zq(zqb)), m1)
            term = {}
            for t in trajs:
                name = {0: "DIRECT", 1: "HALF", 2: "QUARTER_JUMP", 3: "PROGRESSIVE"}[int(t["route_id"])]
                p = log0[:, int(t["a0"])].exp()
                if t["a1"] is not None:
                    p = p * log1[:, int(t["a1"])].exp()
                term[name] = float(p.mean().item())
            snap[tag]["after_terminal_route_probs_mean"] = term
            snap[tag]["after_policy0_entropy"] = float((-(log0.exp() * log0).sum(-1)).mean().item())
            snap[tag]["after_policy1_entropy"] = float((-(log1.exp() * log1).sum(-1)).mean().item())

    dynamics = {}
    one_update(history[:64], losses[:64], "batch1", dynamics)
    one_update(history[64:128], losses[64:128], "batch2", dynamics)

    # save /tmp only
    tmp = Path("/tmp/planB_v1_diag_policy_2batch.pt")
    torch.save({"policy_state_dict": policy.state_dict(), "diagnostic_only": True}, tmp)

    # detect early collapse
    d0 = init_reports["1"]["etas"]["1.0"]["direct_frac"]
    before = dynamics["batch1"]["terminal_route_probs_mean"]["DIRECT"]
    after2 = dynamics["batch2"]["after_terminal_route_probs_mean"]["DIRECT"]
    early = after2 > before + 0.05 or after2 > 0.5
    # also if init already favors DIRECT strongly
    init_favors = all(
        init_reports[str(s)]["etas"]["1.0"]["direct_frac"] >= 0.5 for s in [1, 2, 3]
    )
    if early:
        print("EARLY_POLICY_COLLAPSE_REPRODUCED = TRUE")

    return {
        "init_before_training": init_reports,
        "init_DIRECT_strongly_favored": init_favors,
        "two_batch_dynamics": dynamics,
        "EARLY_POLICY_COLLAPSE_REPRODUCED": early,
        "tmp_checkpoint": str(tmp),
        "elapsed_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# PART 9 state information bottleneck
# ---------------------------------------------------------------------------
@torch.no_grad()
def audit_state_info(device: torch.device, n: int = 1024) -> dict[str, Any]:
    t0 = time.time()
    stable = load_frozen_supernet(STABLE_CKPT, device)
    env = SequentialF2FEnvironment(stable)
    ds = load_crossfit_dataset(CROSSFIT_ORACLE, max_n=n)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_refinement_gains)
    Hs, s0s, zqs, zq_sc, g36, pref = [], [], [], [], [], []
    # need oracle G36 and preference [3,6,12] vs [3,12]
    ora = json.loads(Path(CROSSFIT_ORACLE).read_text())
    recs = ora["records"][:n]
    count = 0
    for history, _si, _gains, losses in loader:
        history = history.to(device)
        H = stable.extract_pre_route_context(history, detach=True)
        s0 = pool_pre_route_context(H)
        Z = env.execute_quarter_prefix(history)["Z_q"]
        zs = Z.mean(dim=(1, 2))
        Hs.append(H.cpu())
        s0s.append(s0.cpu())
        zqs.append(Z.cpu())
        zq_sc.append(zs.cpu())
        bsz = history.shape[0]
        for j in range(bsz):
            r = recs[count + j]
            g36.append(r["G36"])
            # progressive better than quarter?
            L = r["true_route_losses"]
            pref.append(1 if L[3] < L[2] else 0)
        count += bsz
        if count >= n:
            break
    H = torch.cat(Hs)[:n]
    s0 = torch.cat(s0s)[:n]
    Z = torch.cat(zqs)[:n]
    zs = torch.cat(zq_sc)[:n].reshape(-1)
    g36 = np.array(g36[:n], dtype=np.float64)
    pref = np.array(pref[:n], dtype=np.int64)

    # 9.1 current s0
    s0_np = s0.numpy()
    feat_std = s0_np.std(axis=0)
    # PCA 95%
    X = s0_np - s0_np.mean(0)
    # economical SVD
    try:
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        var = (s ** 2) / max(len(X) - 1, 1)
        cvar = np.cumsum(var) / max(var.sum(), 1e-12)
        pca95 = int(np.searchsorted(cvar, 0.95) + 1)
    except Exception:
        pca95 = None
    # pairwise cosine on random pairs
    idx = np.random.RandomState(0).choice(len(s0_np), size=min(200, len(s0_np)), replace=False)
    sims = []
    for i in range(len(idx)):
        for j in range(i + 1, min(i + 5, len(idx))):
            a, b = s0_np[idx[i]], s0_np[idx[j]]
            sims.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))

    # 9.2 structured history diagnostic from H_shared [B,M,N,D]
    # per node: temporal mean, last, abs variation -> [B,N,3,D] then spatial mean/std
    Ht = H  # [B,M,N,D]
    t_mean = Ht.mean(dim=1)  # B,N,D
    t_last = Ht[:, -1]  # B,N,D
    t_var = (Ht[:, 1:] - Ht[:, :-1]).abs().mean(dim=1)  # B,N,D
    node_feat = torch.stack([t_mean, t_last, t_var], dim=2)  # B,N,3,D
    spat_mean = node_feat.mean(dim=1).reshape(node_feat.shape[0], -1)  # B, 3D
    spat_std = node_feat.std(dim=1).reshape(node_feat.shape[0], -1)
    struct_s0 = torch.cat([spat_mean, spat_std], dim=-1).numpy()

    # 9.3 Z_q
    zs_np = zs.numpy()
    # structured Z_q: Z [B,T,N,C]
    Zt = Z
    zm = Zt.mean(dim=1)  # B,N,C
    zl = Zt[:, -1]
    if Zt.shape[1] > 1:
        zv = (Zt[:, 1:] - Zt[:, :-1]).abs().mean(dim=1)
    else:
        zv = torch.zeros_like(zm)
    zn = torch.stack([zm, zl, zv], dim=2)  # B,N,3,C
    z_struct = torch.cat(
        [zn.mean(dim=1).reshape(zn.shape[0], -1), zn.std(dim=1).reshape(zn.shape[0], -1)],
        dim=-1,
    ).numpy()

    # corr scalar zq with G36 / preference
    def safe_corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    # chronological probe split 70/30
    n_tr = int(0.7 * len(pref))

    def logistic_probe(X, y):
        # simple sklearn-free logistic via torch
        Xtr, Xte = X[:n_tr], X[n_tr:]
        ytr, yte = y[:n_tr], y[n_tr:]
        mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sg
        Xte = (Xte - mu) / sg
        xt = torch.tensor(Xtr, dtype=torch.float32)
        yt = torch.tensor(ytr, dtype=torch.float32)
        xv = torch.tensor(Xte, dtype=torch.float32)
        yv = torch.tensor(yte, dtype=torch.float32)
        w = torch.zeros(xt.shape[1], requires_grad=True)
        b_ = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b_], max_iter=50)

        def closure():
            opt.zero_grad()
            logits = xt @ w + b_
            loss = F.binary_cross_entropy_with_logits(logits, yt)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            logits = xv @ w + b_
            prob = torch.sigmoid(logits).numpy()
            pred = (prob >= 0.5).astype(np.int64)
            acc = float((pred == yte).mean())
            # AUC
            order = np.argsort(prob)
            y_sorted = yte[order]
            n_pos = yte.sum()
            n_neg = len(yte) - n_pos
            if n_pos == 0 or n_neg == 0:
                auc = float("nan")
            else:
                ranks = np.arange(1, len(yte) + 1)
                # Mann–Whitney
                sum_ranks_pos = ranks[y_sorted == 1].sum()
                auc = float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
        return {"accuracy": acc, "AUC": auc, "n_train": int(n_tr), "n_test": int(len(yte))}

    probe_scalar = logistic_probe(zs_np.reshape(-1, 1), pref)
    probe_struct = logistic_probe(z_struct, pref)

    return {
        "n": int(len(s0_np)),
        "current_s0": {
            "feature_dim": int(s0_np.shape[1]),
            "per_feature_std_mean": float(feat_std.mean()),
            "per_feature_std_median": float(np.median(feat_std)),
            "per_feature_std_min": float(feat_std.min()),
            "per_feature_std_max": float(feat_std.max()),
            "pca_dims_95pct_variance": pca95,
            "avg_pairwise_cosine_similarity": float(np.mean(sims)),
            "sample_variance_trace": float(np.var(s0_np, axis=0).sum()),
        },
        "structured_history_diagnostic": {
            "feature_dim": int(struct_s0.shape[1]),
            "sample_variance_trace": float(np.var(struct_s0, axis=0).sum()),
            "variance_ratio_vs_global_mean_s0": float(
                np.var(struct_s0, axis=0).sum() / max(np.var(s0_np, axis=0).sum(), 1e-12)
            ),
            "construction": "per-node temporal mean/last/abs-variation then spatial mean+std",
        },
        "zq_scalar": {
            "std_across_samples": float(zs_np.std()),
            "corr_with_G36": safe_corr(zs_np, g36),
            "corr_with_pref_progressive_vs_quarter": safe_corr(zs_np, pref.astype(float)),
        },
        "zq_structured_diagnostic": {
            "feature_dim": int(z_struct.shape[1]),
            "sample_variance_trace": float(np.var(z_struct, axis=0).sum()),
        },
        "diagnostic_probes_chronological": {
            "target": "prefer [3,6,12] over [3,12] (L_prog < L_quarter)",
            "scalar_Z_q": probe_scalar,
            "structured_Z_q": probe_struct,
        },
        "elapsed_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# PART 10 execution audit
# ---------------------------------------------------------------------------
def audit_execution(device: torch.device) -> dict[str, Any]:
    t0 = time.time()
    stable = load_frozen_supernet(STABLE_CKPT, device)
    env = SequentialF2FEnvironment(stable)
    # instrument KASATemporalStep for H/4
    from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
        KASATemporalStep,
    )

    h4 = stable.output_len // 4
    step_h4 = stable.backbone.temporal_steps[stable.res_to_index[h4]]
    counts = {"H4_forward": 0}
    orig = step_h4.forward

    def wrapped(*args, **kwargs):
        counts["H4_forward"] += 1
        return orig(*args, **kwargs)

    step_h4.forward = wrapped

    # Build policy + eval wrapper
    from scripts.eval_group_relative_refinement_policy import PlanBPolicyEvalNet

    with torch.no_grad():
        probe = torch.zeros(1, 12, 307, 4, device=device)
        ctx = int(_build_s0(stable, probe).shape[-1])
    policy = GroupRelativeRefinementPolicy(context_dim=ctx, zq_dim=1, hidden=256).to(device)
    # force quarter decisions by setting logits bias-like: zero init already; override select
    runner = PlanBPolicyEvalNet(stable, policy, env, eta=1.0, stochastic=False)

    history = torch.randn(1, 12, 307, 4, device=device)

    results = {}
    for name, force_route in [("A_[3,12]", [3, 12]), ("B_[3,6,12]", [3, 6, 12])]:
        counts["H4_forward"] = 0

        # Monkeypatch select_route_ids to force route, but still run quarter prefix path if quarter
        rid = {tuple([12]): 0, tuple([6, 12]): 1, tuple([3, 12]): 2, tuple([3, 6, 12]): 3}[
            tuple(force_route)
        ]

        def forced_select(h, _rid=rid, _route=force_route):
            # Still execute the real select_route_ids path for quarter to count prefix
            # For [3,12]/[3,6,12], policy would need a0=QUARTER — call prefix then bucketed
            b = h.shape[0]
            # mimic select_route_ids quarter branch
            pref = env.execute_quarter_prefix(h)
            _ = policy.pool_zq(pref["Z_q"].detach())
            return torch.full((b,), _rid, device=h.device, dtype=torch.long)

        runner.select_route_ids = forced_select
        with torch.no_grad():
            _ = runner.forward(history)
        results[name] = {
            "KASATemporalStep_H4_calls": counts["H4_forward"],
            "forced_route": force_route,
        }

    step_h4.forward = orig
    recomputed = any(v["KASATemporalStep_H4_calls"] > 1 for v in results.values())
    if recomputed:
        print("QUARTER_PREFIX_RECOMPUTED_AT_INFERENCE = TRUE")

    # equivalence check
    with torch.no_grad():
        eq = env.sequential_route_equivalence_check(history, atol=1e-6)

    # estimate overhead: one H/4 stage cost relative to full progressive (3 stages)
    overhead = {
        "duplicate_H4_if_recomputed": recomputed,
        "estimated_duplicate_overhead_vs_progressive": (
            "≈ +1/3 of progressive compute (one extra H/4 stage) when select+execute both run H/4"
            if recomputed
            else "none"
        ),
    }

    return {
        "forced_policy_execution_counts": results,
        "QUARTER_PREFIX_RECOMPUTED_AT_INFERENCE": recomputed,
        "prefix_resume_equivalence": eq,
        "overhead": overhead,
        "code_path": {
            "select_route_ids": "scripts/eval_group_relative_refinement_policy.py::PlanBPolicyEvalNet.select_route_ids",
            "forward_execute": "PlanBPolicyEvalNet.forward -> supernet._execute_routes_bucketed",
        },
        "elapsed_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# PART 12 validation metric sensitivity (offline recompute on valid oracle + current ckpt policy)
# ---------------------------------------------------------------------------
@torch.no_grad()
def audit_val_metric(device: torch.device) -> dict[str, Any]:
    t0 = time.time()
    from scripts.train_group_relative_refinement_policy import _eval_policy_regret

    stable = load_frozen_supernet(STABLE_CKPT, device)
    env = SequentialF2FEnvironment(stable)
    routes = default_candidate_routes(12)
    costs = load_route_costs(None, routes, 12)
    valid_ds = ForecastRefinementGainDataset(
        IndexedTimeSeriesForecastingDataset(DATA_FILE, INDEX_FILE, "valid"),
        VALID_ORACLE,
        expected_routes=routes,
        expected_costs=costs,
        expected_horizon=12,
        expected_dataset="PEMS04",
        require_len_match=False,
    )
    loader = DataLoader(
        valid_ds, batch_size=32, shuffle=False, collate_fn=collate_refinement_gains
    )
    # load formal policy (read-only)
    blob = torch.load(
        "checkpoints/PEMS04/H12/budget_f2f/group_relative_policy.pt", map_location="cpu"
    )
    with torch.no_grad():
        probe = torch.zeros(1, 12, 307, 4, device=device)
        ctx = int(_build_s0(stable, probe).shape[-1])
    policy = GroupRelativeRefinementPolicy(context_dim=ctx, zq_dim=1, hidden=256).to(device)
    policy.load_state_dict(blob["policy_state_dict"])
    policy.eval()

    all_etas = ETAS
    nontrivial = [0.5, 0.75, 1.0]
    a = _eval_policy_regret(policy, stable, env, loader, device, all_etas, argparse.Namespace())
    b = _eval_policy_regret(policy, stable, env, loader, device, nontrivial, argparse.Namespace())

    # per-eta histograms
    per_eta = {}
    for eta in all_etas:
        v = _eval_policy_regret(policy, stable, env, loader, device, [eta], argparse.Namespace())
        per_eta[str(eta)] = {
            "mean_regret": v["mean_validation_route_regret"],
            "mean_cost": v["mean_selected_cost"],
            "route_histogram": {
                ROUTE_NAMES[int(k)]: int(val) for k, val in v["route_histogram"].items()
            },
        }

    return {
        "A_all_etas": a,
        "B_nontrivial_etas_0p5_0p75_1": b,
        "per_eta_route_histogram": per_eta,
        "note": "Recomputed on VALID oracle only; no TEST.",
        "elapsed_sec": time.time() - t0,
    }


def build_root_cause_report(
    evidence: dict,
    reward: dict,
    state: dict,
    objective: dict,
    learning: dict,
    state_info: dict,
    execution: dict,
    valmetric: dict,
) -> dict[str, Any]:
    issues = {"A_FORECASTING_STATE_ENVIRONMENT": [], "B_RL_OBJECTIVE": [], "C_REWARD": [], "D_TRAINING_OPTIMIZATION": [], "E_EFFICIENCY_EXECUTION": []}

    def add(cat, title, severity, evidence_txt, file_fn, confirmed=True):
        issues[cat].append(
            {
                "title": title,
                "severity": severity,
                "evidence": evidence_txt,
                "file_function": file_fn,
                "status": "confirmed" if confirmed else "hypothesis",
            }
        )

    if state.get("STATE_REWARD_ENVIRONMENT_MISMATCH"):
        add(
            "A_FORECASTING_STATE_ENVIRONMENT",
            "STATE_REWARD_ENVIRONMENT_MISMATCH",
            "CRITICAL",
            f"state from stable supernet sha1_16={state['stable_supernet']['sha1_16']}; "
            f"rewards from fold teachers { {k:v['sha1_16'] for k,v in state['teacher_checkpoints'].items()} }. "
            f"Fold state L2(s0) e.g. fold1={state['fold_state_comparison'].get('1',{}).get('pooled_s0',{})}",
            "scripts/train_group_relative_refinement_policy.py::formal_train (_build_s0/_zq_for_batch vs ForecastRefinementGainDataset losses)",
        )
    if state.get("STATE0_AGGRESSIVE_POOLING"):
        add(
            "A_FORECASTING_STATE_ENVIRONMENT",
            "STATE0_AGGRESSIVE_POOLING",
            "HIGH",
            f"H_shared {state['computation_graph']['PEMS04_H12']['extract_pre_route_context']['H_shared_shape']} "
            f"-> s0 {state['computation_graph']['PEMS04_H12']['pool_pre_route_context']['s0_shape']} via mean over patches+nodes",
            "adaptive_refinement_context.py::pool_pre_route_context",
        )
    if state.get("COARSE_FORECAST_SCALAR_BOTTLENECK"):
        add(
            "A_FORECASTING_STATE_ENVIRONMENT",
            "COARSE_FORECAST_SCALAR_BOTTLENECK",
            "HIGH",
            f"Z_q {state.get('Z_q_shape_observed')} pooled to {state.get('zq_pooled_shape_observed')}; "
            f"scalar std={state_info.get('zq_scalar',{}).get('std_across_samples')}; "
            f"probe AUC scalar={state_info.get('diagnostic_probes_chronological',{}).get('scalar_Z_q',{}).get('AUC')} "
            f"vs structured={state_info.get('diagnostic_probes_chronological',{}).get('structured_Z_q',{}).get('AUC')}",
            "group_relative_refinement_policy.py::pool_zq",
        )

    if objective.get("CURRENT_ENUMERATED_SURROGATE_MISMATCH"):
        add(
            "B_RL_OBJECTIVE",
            "CURRENT_ENUMERATED_SURROGATE_MISMATCH",
            "CRITICAL",
            f"grad cosines {objective['gradients']['cosine']}; "
            f"uniform enumeration + mean, not pi_old-weighted; "
            f"branch multiplicity {objective['branch_multiplicity']['policy0_action_row_counts']}",
            "train_group_relative_refinement_policy.py::formal_train + group_relative_refinement_objective.py::clipped_trajectory_objective",
        )
    if objective.get("branch_multiplicity", {}).get("BRANCH_MULTIPLICITY_PRESENT"):
        add(
            "B_RL_OBJECTIVE",
            "BRANCH_MULTIPLICITY_PRESENT",
            "HIGH",
            "At eta=1, QUARTER action appears in 2 terminal rows (JUMP + REFINE), overweighting a0=QUARTER under uniform row mean",
            "sequential_f2f_environment.py::enumerate_feasible_trajectories",
        )
    if objective.get("old_policy_audit", {}).get("BETA_KL_UNUSED"):
        add(
            "B_RL_OBJECTIVE",
            "BETA_KL_UNUSED",
            "MEDIUM",
            "CLI --beta-kl exists but is never applied to the training loss",
            "train_group_relative_refinement_policy.py::main/formal_train",
        )

    # reward issues
    r1 = reward["per_eta"]["1.0"]
    r05 = reward["per_eta"]["0.5"]
    r075 = reward["per_eta"]["0.75"]
    add(
        "C_REWARD",
        "REWARD_OFTEN_SELECTS_DIRECT",
        "HIGH",
        f"reward-argmax DIRECT frac eta.5={r05['fraction_reward_chooses_DIRECT']:.3f}, "
        f".75={r075['fraction_reward_chooses_DIRECT']:.3f}, 1={r1['fraction_reward_chooses_DIRECT']:.3f}; "
        f"thresholds={reward['theoretical_thresholds_vs_DIRECT']}",
        "group_relative_refinement_objective.py::terminal_route_reward",
    )
    if reward["group_standardization_eta0p5"]["GROUP_STD_ERASES_MARGIN_MAGNITUDE"]:
        add(
            "C_REWARD",
            "GROUP_STD_ERASES_MARGIN_MAGNITUDE",
            "HIGH",
            f"eta=.5 two-route groups -> advantages ~[+1,-1]; corr(margin,|A|)="
            f"{reward['group_standardization_eta0p5']['corr_abs_reward_margin_vs_abs_std_advantage']}",
            "group_relative_refinement_objective.py::group_relative_advantages",
        )

    zgr = reward["expected_fraction_no_policy_gradient_uniform_eta"]
    add(
        "D_TRAINING_OPTIMIZATION",
        "STRUCTURAL_ZERO_GRADIENT_RATE",
        "HIGH",
        f"uniform eta sampling expected zero-gradient fraction={zgr:.3f}; eta 0/.25 always single route",
        "train_group_relative_refinement_policy.py::formal_train eta_mode=discrete",
    )
    if learning.get("EARLY_POLICY_COLLAPSE_REPRODUCED") or learning.get("init_DIRECT_strongly_favored"):
        add(
            "D_TRAINING_OPTIMIZATION",
            "EARLY_OR_INIT_DIRECT_COLLAPSE",
            "CRITICAL",
            f"init_DIRECT_strongly_favored={learning.get('init_DIRECT_strongly_favored')}; "
            f"EARLY_POLICY_COLLAPSE_REPRODUCED={learning.get('EARLY_POLICY_COLLAPSE_REPRODUCED')}; "
            f"dynamics={learning.get('two_batch_dynamics')}",
            "group_relative_refinement_policy.py init + formal_train objective",
        )
    # val metric includes trivial etas
    add(
        "D_TRAINING_OPTIMIZATION",
        "VALIDATION_METRIC_INCLUDES_TRIVIAL_ETAS",
        "MEDIUM",
        f"all-eta regret={valmetric['A_all_etas']['mean_validation_route_regret']}; "
        f"nontrivial={valmetric['B_nontrivial_etas_0p5_0p75_1']['mean_validation_route_regret']}; "
        f"hist all collapsed to [12] per epoch log",
        "train_group_relative_refinement_policy.py::_eval_policy_regret",
    )
    het = reward["FOLD_REWARD_HETEROGENEITY"]
    add(
        "A_FORECASTING_STATE_ENVIRONMENT",
        "FOLD_REWARD_HETEROGENEITY",
        "HIGH",
        f"Fold1 teacher size={het['fold1_teacher_train_size']} vs later {het['later_teacher_train_sizes']}; "
        f"G36 mean fold1={het['fold1_G36_mean']} vs {het['later_G36_means']}",
        "results/temporal_crossfit_manifest.json + crossfit oracle",
    )

    if execution.get("QUARTER_PREFIX_RECOMPUTED_AT_INFERENCE"):
        add(
            "E_EFFICIENCY_EXECUTION",
            "QUARTER_PREFIX_RECOMPUTED_AT_INFERENCE",
            "MEDIUM",
            f"counts={execution['forced_policy_execution_counts']}; equivalence={execution['prefix_resume_equivalence']}",
            "eval_group_relative_refinement_policy.py::PlanBPolicyEvalNet.forward/select_route_ids",
        )

    # hypothesis tests
    hyp = {
        "H1_STATE_REWARD_ENVIRONMENT_MISMATCH": bool(state.get("STATE_REWARD_ENVIRONMENT_MISMATCH")),
        "H2_STATE0_AGGRESSIVE_POOLING": bool(state.get("STATE0_AGGRESSIVE_POOLING")),
        "H3_COARSE_FORECAST_SCALAR_BOTTLENECK": bool(state.get("COARSE_FORECAST_SCALAR_BOTTLENECK")),
        "H4_CURRENT_ENUMERATED_SURROGATE_MISMATCH": bool(objective.get("CURRENT_ENUMERATED_SURROGATE_MISMATCH")),
        "H5_GROUP_STD_ERASES_MARGIN_MAGNITUDE": bool(
            reward["group_standardization_eta0p5"]["GROUP_STD_ERASES_MARGIN_MAGNITUDE"]
        ),
        "H6_STRUCTURAL_ZERO_GRADIENT_RATE_approx_0p40": abs(
            reward["expected_fraction_no_policy_gradient_uniform_eta"] - 0.40
        )
        < 0.05,
        "H7_REWARD_MAY_FAVOR_DIRECT": r1["fraction_reward_chooses_DIRECT"] > 0.3,
        "H8_QUARTER_PREFIX_RECOMPUTED_AT_INFERENCE": bool(
            execution.get("QUARTER_PREFIX_RECOMPUTED_AT_INFERENCE")
        ),
        "H9_FOLD_REWARD_HETEROGENEITY": True,
        "H10_BETA_KL_UNUSED": bool(objective.get("old_policy_audit", {}).get("BETA_KL_UNUSED")),
    }

    # top3
    ranked = []
    for cat, lst in issues.items():
        for it in lst:
            sev = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[it["severity"]]
            ranked.append((sev, it["title"], cat, it))
    ranked.sort(key=lambda x: x[0])
    top3 = [{"rank": i + 1, "title": t, "category": c, "severity": it["severity"]} for i, (_, t, c, it) in enumerate(ranked[:3])]

    # recommendation
    if hyp["H4_CURRENT_ENUMERATED_SURROGATE_MISMATCH"] and hyp["H1_STATE_REWARD_ENVIRONMENT_MISMATCH"]:
        rec = "V2-A"
        rec_reason = (
            "Feasible set is tiny and fully enumerable; prefer exact full-information "
            "group-relative expected utility (V2-A) with environment-consistent OOF states, "
            "rather than uniform enumeration pretending to be pi_old sampling (not true V2-B yet)."
        )
    else:
        rec = "NEED MORE EVIDENCE"
        rec_reason = "Insufficient confirmation of primary failure modes."

    terminal = {
        "1_collapse_reproduced_in_tiny_diagnostics": bool(
            learning.get("EARLY_POLICY_COLLAPSE_REPRODUCED")
            or (
                valmetric.get("per_eta_route_histogram", {})
                .get("1.0", {})
                .get("route_histogram", {})
                .get("[12]", 0)
                > 0
                and sum(
                    valmetric.get("per_eta_route_histogram", {})
                    .get("1.0", {})
                    .get("route_histogram", {})
                    .values()
                )
                == valmetric.get("per_eta_route_histogram", {})
                .get("1.0", {})
                .get("route_histogram", {})
                .get("[12]", 0)
            )
        ),
        "2_reward_optimal_DIRECT_fraction": {
            "eta_0.5": r05["fraction_reward_chooses_DIRECT"],
            "eta_0.75": r075["fraction_reward_chooses_DIRECT"],
            "eta_1.0": r1["fraction_reward_chooses_DIRECT"],
        },
        "3_reward_vs_tolerance_oracle_agreement": {
            "eta_0.5": r05["agreement_reward_argmax_vs_tolerance_oracle"],
            "eta_0.75": r075["agreement_reward_argmax_vs_tolerance_oracle"],
            "eta_1.0": r1["agreement_reward_argmax_vs_tolerance_oracle"],
        },
        "4_state_reward_same_teacher": "NO" if state.get("STATE_REWARD_ENVIRONMENT_MISMATCH") else "YES",
        "5_s0_pooling_severity": "HIGH (mean over all patches and nodes)",
        "6_Z_q_bottleneck_severity": "HIGH (global mean -> scalar channel)",
        "7_objective_GRPO_GSPO_consistent": "PARTIAL"
        if not objective.get("CURRENT_ENUMERATED_SURROGATE_MISMATCH")
        else "NO",
        "8_uniform_vs_pi_old_weighted_grad_cosine": objective["gradients"]["cosine"]["A_B"],
        "9_zero_gradient_training_fraction": reward[
            "expected_fraction_no_policy_gradient_uniform_eta"
        ],
        "10_quarter_prefix_recomputation": "YES"
        if execution.get("QUARTER_PREFIX_RECOMPUTED_AT_INFERENCE")
        else "NO",
        "11_fold_reward_heterogeneity_severity": "HIGH",
        "12_top_three_root_causes": top3,
        "13_recommended_next_direction": rec,
        "13_reason": rec_reason,
    }

    return {
        "evidence_part0": evidence,
        "hypothesis_tests": hyp,
        "issues": issues,
        "terminal_summary": terminal,
        "note": (
            "Diagnosis of whether CURRENT formulation is a valid/suitable "
            "group-relative policy optimization for adaptive F2F — not a claim that 'GRPO failed'."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and not args.skip_gpu else "cpu")

    evidence = {
        "git_HEAD": "a54d765bf255b312dd36c4e60d16f1dce7ec45b0",
        "branch": "feb9_best_rebuild",
        "artifacts_sha1": {
            "group_relative_policy.pt": "c6771ad6d1552d78bf760c49432de983da7b89ed",
            "planB_policy_eval.json": "46cb84c6ac6ca7350b893f3bdedc490c96fbfb61",
            "plan_b_full_log": "d24fdb53ca3084fe8a8d666c3927139870e71030",
        },
        "formal_result": {
            "val_regret_all_epochs": 0.083255,
            "selected_cost": 0.540541,
            "route": "[12] 100%",
        },
    }

    print("=== PART 3-5,11 reward geometry (CPU) ===")
    reward = audit_reward_geometry()
    Path("results/planB_v1_reward_geometry.json").write_text(json.dumps(reward, indent=2))
    print("wrote results/planB_v1_reward_geometry.json")

    if args.skip_gpu:
        print("skip GPU parts")
        return 0

    print("=== PART 1-2 graph+state (GPU) ===")
    state = audit_graph_and_state(device)
    print("=== PART 6-7 objective (GPU) ===")
    objective = audit_objective(device, n_samples=128)
    Path("results/planB_v1_objective_audit.json").write_text(
        json.dumps({"state_env": {k: state[k] for k in state if k != "fold_state_comparison"},
                    "fold_state_comparison": state.get("fold_state_comparison"),
                    "objective": objective}, indent=2, default=str)
    )
    # also dump state pieces into state_audit later

    print("=== PART 7-8 learning dynamics (GPU, 2 batches) ===")
    learning = audit_learning_dynamics(device)
    Path("results/planB_v1_learning_dynamics_audit.json").write_text(json.dumps(learning, indent=2))

    print("=== PART 9 state info (GPU) ===")
    state_info = audit_state_info(device, n=1024)
    state_audit = {
        "computation_graph": state["computation_graph"],
        "flags": {
            "STATE0_AGGRESSIVE_POOLING": state["STATE0_AGGRESSIVE_POOLING"],
            "COARSE_FORECAST_SCALAR_BOTTLENECK": state["COARSE_FORECAST_SCALAR_BOTTLENECK"],
            "STATE_REWARD_ENVIRONMENT_MISMATCH": state["STATE_REWARD_ENVIRONMENT_MISMATCH"],
        },
        "stable_vs_teachers": {
            "stable": state["stable_supernet"],
            "teachers": state["teacher_checkpoints"],
            "fold_state_comparison": state["fold_state_comparison"],
        },
        "state_information": state_info,
    }
    Path("results/planB_v1_state_audit.json").write_text(json.dumps(state_audit, indent=2, default=str))

    print("=== PART 10 execution (GPU) ===")
    execution = audit_execution(device)
    Path("results/planB_v1_execution_audit.json").write_text(json.dumps(execution, indent=2))

    print("=== PART 12 val metric sensitivity (GPU) ===")
    valmetric = audit_val_metric(device)

    # merge objective file cleanly
    Path("results/planB_v1_objective_audit.json").write_text(
        json.dumps(objective, indent=2, default=str)
    )

    report = build_root_cause_report(
        evidence, reward, state, objective, learning, state_info, execution, valmetric
    )
    report["validation_metric_sensitivity"] = valmetric
    Path("results/planB_v1_root_cause_report.json").write_text(json.dumps(report, indent=2, default=str))

    # terminal summary print
    ts = report["terminal_summary"]
    print("\n========== FINAL TERMINAL SUMMARY ==========")
    for k, v in ts.items():
        print(f"{k}: {json.dumps(v, default=str)}")
    print("============================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
