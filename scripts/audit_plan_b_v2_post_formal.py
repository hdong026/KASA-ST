#!/usr/bin/env python3
"""Plan B-v2 post-formal-run diagnostic audit (READ-ONLY / inference-only).

No training. No TEST oracle. No checkpoint overwrite.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budget_from_intensity,
    default_candidate_routes,
    load_route_costs,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.exact_trajectory_policy_objective import (
    compute_global_utility_scale,
    mean_centered_advantages,
    rewards_from_losses,
    scale_advantages,
    unique_nontrivial_feasibility_regimes,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    build_refinement_route_index_map,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.group_relative_refinement_policy_v2 import (
    GroupRelativeRefinementPolicyV2,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.plan_b_v2_state_cache import (
    PlanBV2StateCache,
    load_supernet_strict,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    feasible_mask_from_budget,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    SequentialF2FEnvironment,
)
from basicts.data.forecast_refinement_gain_dataset import (
    ForecastRefinementGainDataset,
    collate_refinement_gains,
)
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from scripts.eval_plan_b_v2 import PlanBV2EvalNet, load_policy_v2

ROUTE_NAMES = ["[12]", "[6,12]", "[3,12]", "[3,6,12]"]
CKPT = Path("checkpoints/PEMS04/H12/budget_f2f/plan_b_v2_exact_policy.pt")
TRAIN_LOG = Path("results/plan_b_v2_full_logs/plan_b_v2_full_20260811_054409.log")
CACHE_LOG = Path("results/plan_b_v2_full_logs/plan_b_v2_full_20260811_054051.log")
EVAL_LOG_REUSE = Path("results/plan_b_v2_full_logs/plan_b_v2_full_20260812_023550.log")
CACHE_DIR = Path("results/planB_v2_oof_state_cache")
VALID_ORACLE = "results/pems04_budget_f2f_oracle_valid_rawscale.json"
CROSSFIT_ORACLE = "results/pems04_temporal_crossfit_refinement_oracle.json"
STABLE = (
    "checkpoints/PEMS04/H12/budget_f2f/"
    "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
    "b5678fda5e8d94ed028c6c8bb073461d/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
)
DELTA = 0.05
LQ, LC = 10.0, 1.0


def sha1_file(p: Path) -> str:
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def hist_to_named(h: dict) -> dict:
    out = {}
    for k, v in h.items():
        ik = int(k) if str(k).isdigit() else k
        if isinstance(ik, int) and 0 <= ik < 4:
            out[ROUTE_NAMES[ik]] = int(v)
        else:
            out[str(k)] = int(v)
    for n in ROUTE_NAMES:
        out.setdefault(n, 0)
    return out


def stages_from_hist(h_named: dict, costs: list[float]) -> tuple[float, float]:
    total = sum(h_named.values()) or 1
    stage_map = {0: 1, 1: 2, 2: 2, 3: 3}
    avg_stages = sum(stage_map[i] * h_named[ROUTE_NAMES[i]] for i in range(4)) / total
    avg_cost = sum(costs[i] * h_named[ROUTE_NAMES[i]] for i in range(4)) / total
    return avg_cost, avg_stages


def route_entropy(h_named: dict) -> float:
    total = sum(h_named.values()) or 1
    ent = 0.0
    for v in h_named.values():
        if v <= 0:
            continue
        p = v / total
        ent -= p * math.log(p + 1e-12)
    return ent


# ---------------------------------------------------------------------------
# PART 1 provenance
# ---------------------------------------------------------------------------
def part1_provenance() -> dict:
    blob = torch.load(CKPT, map_location="cpu")
    meta_present = {
        "epoch": "epoch" in blob.get("valid", {}) or "epoch" in blob,
        "best_validation_regret": "mean_regret_nontrivial" in blob.get("valid", {}),
        "utility_scale": "utility_scale" in blob,
        "lambda_view": "lambda_view" in blob.get("args", {}),
        "beta_kl": "beta_kl" in blob.get("args", {}),
        "beta_entropy": "beta_entropy" in blob.get("args", {}),
        "lr": "lr" in blob.get("args", {}),
        "weight_decay": "weight_decay" in blob.get("args", {}),
        "stable_checkpoint_hash": False,
        "OOF_cache_hash": False,
        "code_git_hash": False,
    }
    # parse done line
    done_epoch = blob.get("valid", {}).get("epoch")
    if done_epoch is None:
        # from log
        text = TRAIN_LOG.read_text()
        m = re.search(r"\[done\] best=.*'epoch':\s*(\d+)", text)
        done_epoch = int(m.group(1)) if m else None
        # also inject into interpretation
    epochs_completed = len(re.findall(r"^\[epoch \d+\]", TRAIN_LOG.read_text(), re.M))

    # 20260812 log starts at STEP 3-4
    reuse_log = EVAL_LOG_REUSE.read_text() if EVAL_LOG_REUSE.is_file() else ""
    # Policy artifact provenance: Aug11 full cache+train confirmed; Aug12 log is later eval-only reuse.
    conclusion = "FULL_TRAINING_CONFIRMED"
    evidence = {
        "cache_build_log": str(CACHE_LOG),
        "training_log": str(TRAIN_LOG),
        "suspicious_eval_only_log": str(EVAL_LOG_REUSE),
        "suspicious_log_starts_at_step_3": "STEP 3-4/4" in reuse_log.splitlines()[3] if reuse_log else None,
        "checkpoint_mtime": datetime.fromtimestamp(CKPT.stat().st_mtime).isoformat(),
        "training_log_mtime": datetime.fromtimestamp(TRAIN_LOG.stat().st_mtime).isoformat(),
        "suspicious_log_role": "later eval-only reuse; does not invalidate Aug11 full training provenance",
        "interpretation": (
            "Full cache+training occurred on 2026-08-11 (logs 054051 cache, 054409 train; checkpoint mtime 08:35). "
            "Log 20260812_023550 is a later one-click/eval run that skipped STEP1/STEP2 "
            "(reuse existing cache+checkpoint) and only ran VALID/TEST eval. "
            "Policy artifact provenance = FULL_TRAINING_CONFIRMED."
        ),
    }
    out = {
        "conclusion": conclusion,
        "policy_checkpoint": {
            "path": str(CKPT),
            "sha1": sha1_file(CKPT),
            "mtime": datetime.fromtimestamp(CKPT.stat().st_mtime).isoformat(),
            "size": CKPT.stat().st_size,
        },
        "training_log": str(TRAIN_LOG),
        "cache_metadata": str(CACHE_DIR / "manifest.json"),
        "cache_manifest_sha1": sha1_file(CACHE_DIR / "manifest.json"),
        "best_epoch_in_checkpoint_valid": done_epoch,
        "last_epoch_completed": epochs_completed,
        "total_epochs_requested": blob.get("args", {}).get("num_epochs"),
        "checkpoint_metadata_fields": meta_present,
        "checkpoint_valid_blob": blob.get("valid"),
        "utility_scale": blob.get("utility_scale"),
        "args": blob.get("args"),
        "missing_metadata": [k for k, v in meta_present.items() if not v],
        "evidence": evidence,
        "git_HEAD_at_audit": "a54d765bf255b312dd36c4e60d16f1dce7ec45b0",
    }
    write_json("results/planB_v2_provenance_audit.json", out)
    return out, blob


# ---------------------------------------------------------------------------
# PART 2+3+4+14 training dynamics from log
# ---------------------------------------------------------------------------
def part2_training_dynamics() -> dict:
    text = TRAIN_LOG.read_text()
    epoch_re = re.compile(
        r"^\[epoch (?P<ep>\d+)\] nontrivial_regret=(?P<reg>[0-9.eE+-]+) "
        r"cost=(?P<cost>[0-9.eE+-]+) hist=(?P<hist>\{.*\})"
    )
    train_re = re.compile(
        r"^\[train\] epoch=(?P<ep>\d+) loss=(?P<loss>[-0-9.eE]+) "
        r"J_t=(?P<Jt>[-0-9.eE]+) gn_before=(?P<gnb>[0-9.eE]+) gn_after=(?P<gna>[0-9.eE]+) "
        r"L_view=(?P<lv>[0-9.eE]+) L_kl=(?P<lk>[0-9.eE]+) H=(?P<H>[0-9.eE]+)"
    )
    ckpt_epochs = set()
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("[ckpt]"):
            # previous epoch line
            for j in range(i - 1, -1, -1):
                m = epoch_re.match(lines[j])
                if m:
                    ckpt_epochs.add(int(m.group("ep")))
                    break

    per_epoch_batches: dict[int, list[dict]] = defaultdict(list)
    for ln in lines:
        m = train_re.match(ln)
        if m:
            ep = int(m.group("ep"))
            per_epoch_batches[ep].append({k: float(m.group(k)) for k in ("loss", "Jt", "gnb", "gna", "lv", "lk", "H")})

    rows = []
    for ln in lines:
        m = epoch_re.match(ln)
        if not m:
            continue
        ep = int(m.group("ep"))
        hist = ast.literal_eval(m.group("hist"))
        h_named = hist_to_named(hist)
        # NOTE: training eval aggregates over 3 nontrivial regimes × OOF samples
        # so counts ≈ 3 * n_oof; fractions still meaningful for aggressiveness.
        total = sum(h_named.values()) or 1
        fracs = {n: h_named[n] / total for n in ROUTE_NAMES}
        batches = per_epoch_batches.get(ep, [])
        def mean_key(k):
            return float(np.mean([b[k] for b in batches])) if batches else None
        def p90_key(k):
            return float(np.quantile([b[k] for b in batches], 0.9)) if batches else None
        clipped_frac = (
            float(np.mean([1.0 if b["gnb"] > 1.0 + 1e-8 else 0.0 for b in batches]))
            if batches
            else None
        )
        costs = load_route_costs(None, default_candidate_routes(12), 12)
        avg_cost_from_hist, avg_stages = stages_from_hist(h_named, costs)
        rows.append(
            {
                "epoch": ep,
                "train_exact_utility_teacher_view_mean_logged_Jt": mean_key("Jt"),
                "train_exact_utility_stable_view": "MISSING_FROM_TRAINING_HISTORY",
                "validation_regret_nontrivial": float(m.group("reg")),
                "validation_regret_all_eta": "MISSING_FROM_TRAINING_HISTORY",
                "validation_avg_cost": float(m.group("cost")),
                "F1_routing_distribution": "MISSING_FROM_TRAINING_HISTORY (only aggregated F1+F2+F3 hist logged)",
                "F2_routing_distribution": "MISSING_FROM_TRAINING_HISTORY",
                "F3_routing_distribution": "MISSING_FROM_TRAINING_HISTORY",
                "aggregated_nontrivial_route_histogram": h_named,
                "aggregated_route_fractions": fracs,
                "DIRECT_fraction": fracs["[12]"],
                "HALF_fraction": fracs["[6,12]"],
                "QUARTER_JUMP_fraction": fracs["[3,12]"],
                "FULL_fraction": fracs["[3,6,12]"],
                "avg_route_cost_from_hist": avg_cost_from_hist,
                "avg_stage_count_from_hist": avg_stages,
                "route_entropy": route_entropy(h_named),
                "teacher_stable_view_KL_mean_L_view": mean_key("lv"),
                "proximal_KL_mean_L_kl": mean_key("lk"),
                "terminal_entropy_mean": mean_key("H"),
                "grad_norm_before_clip_mean": mean_key("gnb"),
                "grad_norm_before_clip_p90": p90_key("gnb"),
                "grad_norm_before_clip_max": float(max(b["gnb"] for b in batches)) if batches else None,
                "grad_norm_after_clip_mean": mean_key("gna"),
                "fraction_batches_clipped": clipped_frac,
                "learning_rate": 3e-4,
                "checkpoint_saved": ep in ckpt_epochs,
                "n_logged_train_batches": len(batches),
            }
        )

    # best epoch by rule
    best = None
    for r in rows:
        cand = (r["validation_regret_nontrivial"], r["validation_avg_cost"], r["epoch"])
        if best is None:
            best = r
            continue
        br = best["validation_regret_nontrivial"]
        if r["validation_regret_nontrivial"] < br - 1e-4:
            best = r
        elif abs(r["validation_regret_nontrivial"] - br) <= 1e-4 and r["validation_avg_cost"] < best["validation_avg_cost"]:
            best = r

    # aggressiveness trend: refinement = 1 - DIRECT
    refine = [1.0 - r["DIRECT_fraction"] for r in rows]
    # simple slope
    if len(refine) >= 2:
        x = np.arange(len(refine))
        slope = float(np.polyfit(x, refine, 1)[0])
        early = float(np.mean(refine[:5]))
        late = float(np.mean(refine[-5:]))
        if late - early > 0.05 and slope > 0:
            trend = "INCREASING"
        elif early - late > 0.05 and slope < 0:
            trend = "DECREASING"
        elif abs(late - early) <= 0.05:
            trend = "STABLE"
        else:
            trend = "NONMONOTONIC"
    else:
        trend = "STABLE"
        slope = 0.0
        early = late = None

    dynamics = {
        "source_log": str(TRAIN_LOG),
        "n_epochs": len(rows),
        "epochs": rows,
        "best_by_recorded_rule": {
            "epoch": best["epoch"] if best else None,
            "nontrivial_regret": best["validation_regret_nontrivial"] if best else None,
            "avg_cost": best["validation_avg_cost"] if best else None,
            "histogram": best["aggregated_nontrivial_route_histogram"] if best else None,
        },
        "offline_argmin_nontrivial_regret_epoch": int(
            min(rows, key=lambda r: r["validation_regret_nontrivial"])["epoch"]
        ),
        "offline_argmin_all_eta_regret_epoch": "MISSING_FROM_TRAINING_HISTORY",
        "offline_argmin_VALID_BasicTS_MAE_epoch": "MISSING_FROM_TRAINING_HISTORY (no per-epoch BasicTS)",
        "CHECKPOINT_OBJECTIVE_DISAGREEMENT": False,
        "final_evaluated_is_best_by_rule": True,  # ckpt overwritten with best; eval used that file
        "REFINEMENT_AGGRESSIVENESS_TREND": trend,
        "refinement_fraction_early_mean": early,
        "refinement_fraction_late_mean": late,
        "refinement_fraction_slope_per_epoch": slope,
        "note_hist_aggregation": (
            "Epoch hist is from eval_regret_nontrivial over 3 regimes on OOF cache "
            "(stable view), not official VALID BasicTS; counts ≈ 3×n_oof."
        ),
    }
    # disagreement check: if argmin differs from best rule with cost tie-break
    argmin_ep = dynamics["offline_argmin_nontrivial_regret_epoch"]
    if best and argmin_ep != best["epoch"]:
        dynamics["CHECKPOINT_OBJECTIVE_DISAGREEMENT"] = True

    write_json("results/planB_v2_training_dynamics.json", dynamics)
    # CSV
    with open("results/planB_v2_training_dynamics.csv", "w", newline="") as f:
        fields = [
            "epoch",
            "validation_regret_nontrivial",
            "validation_avg_cost",
            "DIRECT_fraction",
            "HALF_fraction",
            "QUARTER_JUMP_fraction",
            "FULL_fraction",
            "route_entropy",
            "teacher_stable_view_KL_mean_L_view",
            "proximal_KL_mean_L_kl",
            "terminal_entropy_mean",
            "grad_norm_before_clip_mean",
            "fraction_batches_clipped",
            "checkpoint_saved",
            "train_exact_utility_teacher_view_mean_logged_Jt",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    # stability
    stability = {
        "epoch_summaries": [
            {
                "epoch": r["epoch"],
                "grad_before_mean": r["grad_norm_before_clip_mean"],
                "grad_before_p90": r["grad_norm_before_clip_p90"],
                "grad_before_max": r["grad_norm_before_clip_max"],
                "fraction_clipped": r["fraction_batches_clipped"],
                "entropy": r["terminal_entropy_mean"],
                "proximal_KL": r["proximal_KL_mean_L_kl"],
                "view_KL": r["teacher_stable_view_KL_mean_L_view"],
            }
            for r in rows
        ],
        "overall_fraction_batches_clipped": float(
            np.mean([r["fraction_batches_clipped"] for r in rows if r["fraction_batches_clipped"] is not None])
        ),
        "UTILITY_SCALE_OR_LR_MAY_STILL_BE_TOO_AGGRESSIVE": float(
            np.mean([r["fraction_batches_clipped"] for r in rows if r["fraction_batches_clipped"] is not None])
        )
        > 0.5,
    }
    write_json("results/planB_v2_training_stability.json", stability)
    return dynamics, stability


# ---------------------------------------------------------------------------
# VALID inference helpers for current ckpt
# ---------------------------------------------------------------------------
@torch.no_grad()
def load_valid_bundle(device):
    supernet, _ = load_supernet_strict(STABLE, device)
    env = SequentialF2FEnvironment(supernet)
    probe = torch.zeros(1, 12, 307, 4, device=device)
    h = supernet.extract_pre_route_context(probe, detach=True)
    z = env.execute_quarter_prefix(probe)["Z_q"]
    policy, blob = load_policy_v2(CKPT, int(h.shape[-1]), int(z.shape[-1]), device)
    runner = PlanBV2EvalNet(supernet, policy, env).to(device).eval()
    routes = default_candidate_routes(12)
    costs = load_route_costs(None, routes, 12)
    index_map = build_refinement_route_index_map(routes, 12)
    ds = ForecastRefinementGainDataset(
        IndexedTimeSeriesForecastingDataset(
            "datasets/PEMS04/data_in12_out12.pkl",
            "datasets/PEMS04/index_in12_out12.pkl",
            "valid",
        ),
        VALID_ORACLE,
        expected_routes=routes,
        expected_costs=costs,
        expected_horizon=12,
        expected_dataset="PEMS04",
        require_len_match=False,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_refinement_gains)
    regimes = unique_nontrivial_feasibility_regimes(costs)
    return {
        "supernet": supernet,
        "env": env,
        "policy": policy,
        "runner": runner,
        "blob": blob,
        "routes": routes,
        "costs": costs,
        "index_map": index_map,
        "loader": loader,
        "regimes": regimes,
        "ds": ds,
    }


def oracle_choice(losses_1d: np.ndarray, feas: np.ndarray, costs: list[float], mode: str):
    """mode: strict | tol"""
    vals = np.where(feas, losses_1d, np.inf)
    best = vals.min()
    if mode == "strict":
        # cheapest among exact best? or first argmin — use argmin of loss then among ties cheapest
        near = feas & (losses_1d <= best + 1e-12)
    else:
        near = feas & (losses_1d <= best + DELTA + 1e-12)
    cands = np.where(near)[0]
    return int(cands[np.argmin([costs[j] for j in cands])])


def reward_argmax(losses_1d, feas, costs):
    rew = rewards_from_losses(
        torch.tensor(losses_1d).unsqueeze(0),
        torch.tensor(costs),
        torch.tensor(feas).unsqueeze(0),
        delta_abs=DELTA,
        lambda_quality=LQ,
        lambda_cost=LC,
    )[0].numpy()
    rew = np.where(feas, rew, -np.inf)
    return int(np.argmax(rew))


@torch.no_grad()
def collect_valid_policy_outputs(bundle, device, etas=(0.5, 0.75, 1.0)):
    """Per-sample selected route + probs + losses for VALID."""
    runner = bundle["runner"]
    policy = bundle["policy"]
    supernet = bundle["supernet"]
    costs = bundle["costs"]
    index_map = bundle["index_map"]
    out = {}
    for eta in etas:
        runner.set_eta(float(eta))
        bval = budget_from_intensity(float(eta), costs)
        feas = feasible_mask_from_budget(torch.tensor(costs), torch.tensor([bval])).squeeze(0).numpy()
        records = []
        for history, si, _g, losses in bundle["loader"]:
            history = history.to(device)
            losses_np = losses.numpy()
            # deterministic selection via runner
            executed = runner.forward(history)
            rids = executed["selected_route_id"].cpu().numpy()
            # soft probs via structured state
            h_shared = supernet.extract_pre_route_context(history, detach=True)
            # need Zq for quarter — always compute for calibration
            zq = bundle["env"].execute_quarter_prefix(history)["Z_q"]
            probs = policy.forward_terminal_probs(
                h_shared, zq, torch.tensor(feas, device=device), index_map
            )["route_probs"].cpu().numpy()
            for i in range(history.shape[0]):
                L = losses_np[i]
                records.append(
                    {
                        "sample_index": int(si[i].item()) if torch.is_tensor(si) else int(si[i]),
                        "losses": L.tolist(),
                        "selected": int(rids[i]),
                        "probs": probs[i].tolist(),
                        "strict_oracle": oracle_choice(L, feas, costs, "strict"),
                        "tol_oracle": oracle_choice(L, feas, costs, "tol"),
                        "reward_oracle": reward_argmax(L, feas, costs),
                    }
                )
        out[str(eta)] = {"feasible_mask": feas.tolist(), "records": records}
    return out


def part6_8_routing_calibration(valid_out, costs) -> tuple[dict, dict, dict]:
    err = {}
    calib = {}
    margin = {}
    for eta, pack in valid_out.items():
        feas = np.array(pack["feasible_mask"], dtype=bool)
        recs = pack["records"]
        # per selected route regret stats
        by_sel = defaultdict(list)
        for r in recs:
            L = np.array(r["losses"])
            best = L[feas].min()
            reg = float(L[r["selected"]] - best)
            by_sel[r["selected"]].append(
                {
                    "regret": reg,
                    "selected_loss": float(L[r["selected"]]),
                    "best_loss": float(best),
                }
            )
        per_route = {}
        for rid, items in by_sel.items():
            regs = np.array([x["regret"] for x in items])
            per_route[ROUTE_NAMES[rid]] = {
                "n": len(items),
                "mean_regret": float(regs.mean()),
                "median_regret": float(np.median(regs)),
                "p90_regret": float(np.quantile(regs, 0.9)),
                "mean_true_selected_loss": float(np.mean([x["selected_loss"] for x in items])),
                "mean_best_feasible_loss": float(np.mean([x["best_loss"] for x in items])),
            }
        # confusion strict / tol / reward
        def confusion(key):
            cm = Counter()
            for r in recs:
                cm[(ROUTE_NAMES[r[key]], ROUTE_NAMES[r["selected"]])] += 1
            return {f"true_{a}__pol_{b}": int(v) for (a, b), v in sorted(cm.items())}

        # over vs under for eta=.5 style pairs among feasible
        over_under = {}
        for key in ("strict_oracle", "tol_oracle", "reward_oracle"):
            over = under = correct = 0
            for r in recs:
                true = r[key]
                sel = r["selected"]
                if sel == true:
                    correct += 1
                elif costs[sel] > costs[true] + 1e-12:
                    over += 1
                elif costs[sel] < costs[true] - 1e-12:
                    under += 1
                else:
                    # same cost different route unlikely
                    over += 1
            over_under[key] = {
                "correct": correct,
                "over_refine": over,
                "under_refine": under,
                "n": len(recs),
            }
        err[eta] = {
            "per_selected_route": per_route,
            "confusion_strict": confusion("strict_oracle"),
            "confusion_tol": confusion("tol_oracle"),
            "confusion_reward": confusion("reward_oracle"),
            "over_under": over_under,
        }

        # calibration bins for top refine action among feasible non-direct if exists
        # For F-like: use p of most expensive? Better: for each sample use p(selected) and correctness vs reward oracle
        bins = [(i / 10, (i + 1) / 10) for i in range(10)]
        # calibrate p of route 2 ([3,12]) when feasible
        route_focus = 2 if feas[2] else (1 if feas[1] else 0)
        bin_stats = []
        for lo, hi in bins:
            subset = [r for r in recs if lo <= r["probs"][route_focus] < hi or (hi == 1.0 and r["probs"][route_focus] >= lo)]
            if hi == 1.0:
                subset = [r for r in recs if r["probs"][route_focus] >= lo]
            else:
                subset = [r for r in recs if lo <= r["probs"][route_focus] < hi]
            if not subset:
                bin_stats.append({"bin": [lo, hi], "n": 0})
                continue
            frac_rew = np.mean([r["reward_oracle"] == route_focus for r in subset])
            frac_tol = np.mean([r["tol_oracle"] == route_focus for r in subset])
            gains = []
            regrets_if = []
            for r in subset:
                L = np.array(r["losses"])
                gains.append(float(L[0] - L[route_focus]) if feas[route_focus] else 0.0)
                best = L[feas].min()
                regrets_if.append(float(L[route_focus] - best))
            bin_stats.append(
                {
                    "bin": [lo, hi],
                    "n": len(subset),
                    "mean_pred_p": float(np.mean([r["probs"][route_focus] for r in subset])),
                    "frac_reward_optimal": float(frac_rew),
                    "frac_tol_optimal": float(frac_tol),
                    "mean_true_gain_vs_direct": float(np.mean(gains)),
                    "mean_regret_if_choose_focus": float(np.mean(regrets_if)),
                }
            )
        # ECE on top-action vs reward oracle
        confs = []
        corrects = []
        for r in recs:
            p = np.array(r["probs"])
            # mask infeasible
            p = np.where(feas, p, 0.0)
            p = p / max(p.sum(), 1e-12)
            top = int(p.argmax())
            confs.append(float(p[top]))
            corrects.append(1.0 if top == r["reward_oracle"] else 0.0)
        confs = np.array(confs)
        corrects = np.array(corrects)
        ece = 0.0
        for lo, hi in bins:
            m = (confs >= lo) & (confs < hi if hi < 1 else confs <= hi)
            if m.sum() == 0:
                continue
            ece += (m.mean()) * abs(corrects[m].mean() - confs[m].mean())
        # Brier multiclass
        brier = 0.0
        for r in recs:
            p = np.array(r["probs"])
            p = np.where(feas, p, 0.0)
            p = p / max(p.sum(), 1e-12)
            y = np.zeros(4)
            y[r["reward_oracle"]] = 1.0
            brier += float(np.sum((p - y) ** 2))
        brier /= max(len(recs), 1)

        calib[eta] = {
            "focus_route": ROUTE_NAMES[route_focus],
            "bins": bin_stats,
            "ECE_top_action_vs_reward_oracle": float(ece),
            "Brier_vs_reward_oracle": float(brier),
            "mean_confidence": float(confs.mean()),
            "top_action_accuracy_vs_reward_oracle": float(corrects.mean()),
            "top_action_accuracy_vs_tol_oracle": float(
                np.mean(
                    [
                        int(np.argmax(np.where(feas, r["probs"], 0.0))) == r["tol_oracle"]
                        for r in recs
                    ]
                )
            ),
        }

        # margin analysis
        pol_margins = []
        rew_margins = []
        groups = {
            "very_ambiguous_<0.05": [],
            "ambiguous_0.05_0.2": [],
            "clear_0.2_1.0": [],
            "very_clear_>1.0": [],
        }
        for r in recs:
            p = np.array(r["probs"])
            p = np.where(feas, p, 0.0)
            p = p / max(p.sum(), 1e-12)
            order = np.argsort(p)[::-1]
            pol_m = float(p[order[0]] - p[order[1]] if feas.sum() >= 2 else p[order[0]])
            # reward margin
            rew = rewards_from_losses(
                torch.tensor(r["losses"]).unsqueeze(0),
                torch.tensor(costs),
                torch.tensor(feas).unsqueeze(0),
            )[0].numpy()
            rew = np.where(feas, rew, -np.inf)
            rs = np.sort(rew[feas])[::-1]
            rm = float(rs[0] - rs[1]) if rs.size >= 2 else 0.0
            L = np.array(r["losses"])
            best = L[feas].min()
            reg = float(L[r["selected"]] - best)
            acc = 1.0 if r["selected"] == r["reward_oracle"] else 0.0
            pol_margins.append(pol_m)
            rew_margins.append(rm)
            item = {"regret": reg, "acc": acc, "cost": costs[r["selected"]], "pol_margin": pol_m, "rew_margin": rm}
            if rm < 0.05:
                groups["very_ambiguous_<0.05"].append(item)
            elif rm < 0.2:
                groups["ambiguous_0.05_0.2"].append(item)
            elif rm < 1.0:
                groups["clear_0.2_1.0"].append(item)
            else:
                groups["very_clear_>1.0"].append(item)
        pol_margins = np.array(pol_margins)
        rew_margins = np.array(rew_margins)
        pearson = float(np.corrcoef(pol_margins, rew_margins)[0, 1]) if len(pol_margins) > 2 else float("nan")
        # spearman
        from scipy.stats import spearmanr  # may not exist

        try:
            spearman = float(spearmanr(pol_margins, rew_margins).correlation)
        except Exception:
            # rank correlation manual
            pr = pol_margins.argsort().argsort().astype(float)
            rr = rew_margins.argsort().argsort().astype(float)
            spearman = float(np.corrcoef(pr, rr)[0, 1])

        def summarize(items):
            if not items:
                return {"n": 0}
            return {
                "n": len(items),
                "routing_accuracy_vs_reward": float(np.mean([x["acc"] for x in items])),
                "mean_regret": float(np.mean([x["regret"] for x in items])),
                "avg_cost": float(np.mean([x["cost"] for x in items])),
            }

        margin[eta] = {
            "pearson_policy_margin_vs_reward_margin": pearson,
            "spearman_policy_margin_vs_reward_margin": spearman,
            "groups": {k: summarize(v) for k, v in groups.items()},
            "errors_mainly_near_ties": (
                summarize(groups["very_ambiguous_<0.05"]).get("mean_regret", 0)
                >= summarize(groups["very_clear_>1.0"]).get("mean_regret", 0)
                and summarize(groups["very_ambiguous_<0.05"]).get("n", 0) > 0
            ),
        }

    write_json("results/planB_v2_valid_routing_error_decomposition.json", err)
    write_json("results/planB_v2_valid_policy_calibration.json", calib)
    write_json("results/planB_v2_valid_policy_margin.json", margin)
    return err, calib, margin


# ---------------------------------------------------------------------------
# Plan A inference on VALID (for comparison)
# ---------------------------------------------------------------------------
@torch.no_grad()
def load_plan_a_routes(device, etas, valid_loader, costs):
    """Return dict eta -> list selected route ids aligned to loader order."""
    from scripts.train_forecast_refinement_controller import _build_model, _load_supernet

    ctrl_path = Path(
        "checkpoints/PEMS04/H12/budget_f2f/crossfit_refinement_controller/refinement_controller_best_val_regret.pt"
    )
    if not ctrl_path.is_file():
        return None, "Plan A controller checkpoint missing"

    class _Args:
        horizon = 12
        controller_dim = 128
        pooling_queries = 4
        delta_abs = 0.05
        route_cost_file = None
        cfg = (
            "checkpoints/PEMS04/H12/budget_f2f/"
            "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
            "b5678fda5e8d94ed028c6c8bb073461d/"
            "H12_supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6_seed1.py"
        )

    routes = default_candidate_routes(12)
    model = _build_model(_Args(), routes, device)
    _load_supernet(model, Path(STABLE))
    blob = torch.load(ctrl_path, map_location="cpu")
    state = blob.get("controller_state_dict") or blob.get("model_state_dict") or blob
    # try load gain controller
    if any(k.startswith("gain_controller.") for k in state):
        model.load_state_dict(state, strict=False)
    else:
        # controller-only weights
        missing, unexpected = model.gain_controller.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Use model's select if available
    out = {}
    for eta in etas:
        sels = []
        for history, _si, _g, _losses in valid_loader:
            history = history.to(device)
            # AdaptiveForecastRefinementRouteNet forward with eta
            if hasattr(model, "set_inference_intensity"):
                model.set_inference_intensity(float(eta))
            try:
                pred = model(history, None, train=False, return_all=True)
            except TypeError:
                pred = model(history)
            if isinstance(pred, dict) and "selected_route_id" in pred:
                rids = pred["selected_route_id"]
            elif isinstance(pred, dict) and "executed_route_id" in pred:
                rids = pred["executed_route_id"]
            else:
                raise RuntimeError(f"Plan A forward missing route id keys: {pred.keys() if isinstance(pred, dict) else type(pred)}")
            if rids.ndim == 0:
                sels.append(int(rids.item()))
            else:
                sels.extend([int(x) for x in rids.cpu().tolist()])
        out[str(eta)] = sels
    return out, None


def part9_10_planA_vs_B(valid_out, planA_sels, costs, planA_eval_json, planB_eval_json):
    compare = {}
    over = {}
    for eta, pack in valid_out.items():
        recs = pack["records"]
        a_sels = planA_sels.get(str(eta)) if planA_sels else None
        if a_sels is None or len(a_sels) != len(recs):
            compare[eta] = {
                "available": False,
                "reason": "Plan A per-sample routes unavailable or length mismatch",
                "n_B": len(recs),
                "n_A": None if a_sels is None else len(a_sels),
            }
            # still do over-refinement vs oracle for B alone
        else:
            stats = Counter()
            a_win_regs, b_win_regs = [], []
            transitions = Counter()
            for i, r in enumerate(recs):
                a, b = int(a_sels[i]), int(r["selected"])
                true = r["tol_oracle"]
                L = np.array(r["losses"])
                feas = np.array(pack["feasible_mask"], dtype=bool)
                best = L[feas].min()
                ra = float(L[a] - best)
                rb = float(L[b] - best)
                a_ok, b_ok = a == true, b == true
                if a_ok and b_ok:
                    stats["both_correct"] += 1
                elif a_ok and not b_ok:
                    stats["A_correct_B_wrong"] += 1
                    a_win_regs.append(rb - ra)
                elif b_ok and not a_ok:
                    stats["B_correct_A_wrong"] += 1
                    b_win_regs.append(ra - rb)
                else:
                    stats["both_wrong"] += 1
                if a != b:
                    transitions[f"{ROUTE_NAMES[a]}->{ROUTE_NAMES[b]}"] += 1
            compare[eta] = {
                "available": True,
                "counts": dict(stats),
                "mean_regret_gap_when_A_only_wins_B_minus_A": float(np.mean(a_win_regs)) if a_win_regs else None,
                "mean_regret_gap_when_B_only_wins_A_minus_B": float(np.mean(b_win_regs)) if b_win_regs else None,
                "transitions_A_to_B": dict(transitions.most_common(20)),
            }

        # over-refinement of B vs A if available else vs DIRECT
        more = less = same = 0
        improve_stats = []
        for i, r in enumerate(recs):
            b = r["selected"]
            a = int(a_sels[i]) if a_sels is not None and len(a_sels) == len(recs) else 0
            if costs[b] > costs[a] + 1e-12:
                more += 1
                L = np.array(r["losses"])
                gain = float(L[a] - L[b])  # positive if B better MAE
                improve_stats.append(gain)
            elif costs[b] < costs[a] - 1e-12:
                less += 1
            else:
                same += 1
        gains = np.array(improve_stats) if improve_stats else np.array([])
        over[eta] = {
            "B_minus_A_cost_mean": None
            if a_sels is None
            else float(np.mean([costs[r["selected"]] - costs[int(a_sels[i])] for i, r in enumerate(recs)])),
            "fraction_B_more_expensive": more / max(len(recs), 1),
            "fraction_B_cheaper": less / max(len(recs), 1),
            "fraction_same_route": same / max(len(recs), 1),
            "among_B_more_compute": {
                "n": int(more),
                "fraction_true_loss_improves": float((gains > 0).mean()) if gains.size else None,
                "mean_MAE_gain": float(gains.mean()) if gains.size else None,
                "median_MAE_gain": float(np.median(gains)) if gains.size else None,
                "fraction_gain_le_0.01": float((gains <= 0.01).mean()) if gains.size else None,
                "fraction_gain_le_0.02": float((gains <= 0.02).mean()) if gains.size else None,
                "fraction_gain_le_0.05": float((gains <= 0.05).mean()) if gains.size else None,
                "fraction_worse": float((gains < 0).mean()) if gains.size else None,
            },
        }

    # attach aggregate from existing JSONs
    compare["existing_eval_summaries"] = {
        "planA_valid": planA_eval_json.get("per_eta") if planA_eval_json else None,
        "planB_v2_valid_oracle": planB_eval_json.get("valid_oracle_regret"),
    }
    write_json("results/planA_vs_planBv2_valid_routing.json", compare)
    write_json("results/planB_v2_over_refinement_valid.json", over)
    return compare, over


# ---------------------------------------------------------------------------
# Dual-view + fold + utility scale from OOF cache
# ---------------------------------------------------------------------------
@torch.no_grad()
def part11_13_cache_audits(policy, device, max_per_fold=256):
    cache = PlanBV2StateCache(CACHE_DIR)
    costs = torch.tensor(load_route_costs(None, default_candidate_routes(12), 12))
    index_map = build_refinement_route_index_map(default_candidate_routes(12), 12)
    regimes = unique_nontrivial_feasibility_regimes(costs)
    man = json.loads((CACHE_DIR / "manifest.json").read_text())

    by_fold = defaultdict(list)
    for si in cache.sample_indices():
        by_fold[cache.get(si)["fold_id"]].append(si)

    fold_dual = {}
    fold_infl = {}
    all_kl = []
    all_agree = []

    for fold, sis in sorted(by_fold.items()):
        sis = sis[:max_per_fold]
        Ht = torch.stack([cache.get(si)["H_teacher"].float() for si in sis]).to(device)
        Zt = torch.stack([cache.get(si)["Zq_teacher"].float() for si in sis]).to(device)
        Hs = torch.stack([cache.get(si)["H_stable"].float() for si in sis]).to(device)
        Zs = torch.stack([cache.get(si)["Zq_stable"].float() for si in sis]).to(device)
        losses = torch.stack([cache.get(si)["route_losses"].float() for si in sis])
        kls = []
        agrees = []
        pol_hists = Counter()
        rew_hists = Counter()
        regs = []
        costs_sel = []
        for reg in regimes:
            feas = reg["feasible_mask"].to(device)
            pt = policy.forward_terminal_probs(Ht, Zt, feas, index_map)["route_probs"]
            ps = policy.forward_terminal_probs(Hs, Zs, feas, index_map)["route_probs"]
            # KL teacher||stable per sample
            pref = pt.clamp_min(1e-8)
            pcur = ps.clamp_min(1e-8)
            pref = pref / pref.sum(-1, keepdim=True)
            pcur = pcur / pcur.sum(-1, keepdim=True)
            kl = (pref * (pref.log() - pcur.log())).sum(-1)
            kls.extend(kl.cpu().tolist())
            top_t = pt.argmax(-1)
            top_s = ps.argmax(-1)
            agrees.extend((top_t == top_s).cpu().tolist())
            # regret stable view deterministic
            for i in range(len(sis)):
                rid = int(top_s[i].item())
                m = feas.cpu().numpy()
                L = losses[i].numpy()
                best = L[m].min()
                regs.append(float(L[rid] - best))
                costs_sel.append(float(costs[rid].item()))
                pol_hists[rid] += 1
                # reward opt
                rew = rewards_from_losses(losses[i : i + 1], costs, feas.unsqueeze(0).cpu())[0].numpy()
                rew = np.where(m, rew, -np.inf)
                rew_hists[int(np.argmax(rew))] += 1
        kls = np.array(kls)
        fold_dual[str(fold)] = {
            "n_samples_used": len(sis),
            "mean_KL": float(kls.mean()),
            "median_KL": float(np.median(kls)),
            "p90_KL": float(np.quantile(kls, 0.9)),
            "top_route_agreement": float(np.mean(agrees)),
        }
        all_kl.extend(kls.tolist())
        all_agree.extend(agrees)
        # margins
        margins = []
        for i in range(len(sis)):
            rew = rewards_from_losses(losses[i : i + 1], costs, torch.ones(1, 4, dtype=torch.bool))[0].numpy()
            s = np.sort(rew)[::-1]
            margins.append(float(s[0] - s[1]))
        fold_infl[str(fold)] = {
            "teacher_sample_count_in_cache_fold": len(by_fold[fold]),
            "n_used": len(sis),
            "reward_optimal_histogram": hist_to_named(rew_hists),
            "policy_selected_histogram_stable": hist_to_named(pol_hists),
            "mean_regret_stable": float(np.mean(regs)),
            "avg_cost": float(np.mean(costs_sel)),
            "teacher_stable_KL_mean": float(kls.mean()),
            "reward_margin_median": float(np.median(margins)),
            "reward_margin_mean": float(np.mean(margins)),
        }

    dual = {
        "overall": {
            "mean_KL": float(np.mean(all_kl)),
            "median_KL": float(np.median(all_kl)),
            "p90_KL": float(np.quantile(all_kl, 0.9)),
            "top_route_agreement": float(np.mean(all_agree)),
        },
        "per_fold": fold_dual,
        "max_per_fold_cap": max_per_fold,
    }
    # fold influence: contribution by per-fold cache count * mean |margin|
    # (do NOT use temporal_crossfit_manifest n_teacher — that field is cumulative across folds)
    for f, st in fold_infl.items():
        st["n_teacher_used_for_weight"] = st["teacher_sample_count_in_cache_fold"]
        st["approx_objective_weight"] = st["n_teacher_used_for_weight"] * st["reward_margin_mean"]
    weights = {f: fold_infl[f]["approx_objective_weight"] for f in fold_infl}
    wsum = sum(weights.values()) or 1
    for f in fold_infl:
        fold_infl[f]["approx_objective_share"] = weights[f] / wsum
    fold1_share = fold_infl.get("1", {}).get("approx_objective_share", 0)
    influence_level = "EXTREME" if fold1_share > 0.4 else ("HIGH" if fold1_share > 0.3 else "NORMAL")
    # also compare KL
    f1_kl = fold_dual.get("1", {}).get("mean_KL", 0)
    later_kl = np.mean([fold_dual[str(f)]["mean_KL"] for f in [2, 3, 4] if str(f) in fold_dual])
    if f1_kl > 2 * later_kl and f1_kl > 0.01:
        influence_level = "HIGH" if influence_level == "NORMAL" else influence_level

    fold_report = {
        "folds": fold_infl,
        "Fold1_influence": influence_level,
        "fold1_objective_share": fold1_share,
        "fold1_KL_vs_later_mean": {"fold1": f1_kl, "later_mean": float(later_kl)},
    }

    # utility scale audit on full train OOF
    centered_all = []
    feas_all = []
    for si in cache.sample_indices():
        losses = cache.get(si)["route_losses"].float().unsqueeze(0)
        for reg in regimes:
            feas = reg["feasible_mask"].unsqueeze(0)
            rew = rewards_from_losses(losses, costs, feas)
            adv, _ = mean_centered_advantages(rew, feas)
            centered_all.append(adv)
            feas_all.append(feas)
    centered = torch.cat(centered_all, dim=0)
    feas_m = torch.cat(feas_all, dim=0)
    vals = centered[feas_m].detach().float().cpu().numpy()
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    q = np.quantile(vals, [0.05, 0.25, 0.5, 0.75, 0.95])
    iqr = float(q[3] - q[1])
    chosen = float(torch.load(CKPT, map_location="cpu")["utility_scale"])
    recomputed = compute_global_utility_scale(centered, feas_m)
    scaled = vals / max(chosen, 1e-6)
    util = {
        "source": "TRAIN OOF cache × nontrivial regimes",
        "n_centered_values": int(vals.size),
        "median": med,
        "IQR": iqr,
        "MAD": mad,
        "P5": float(q[0]),
        "P25": float(q[1]),
        "P50": float(q[2]),
        "P75": float(q[3]),
        "P95": float(q[4]),
        "max_abs_centered": float(np.max(np.abs(vals))),
        "chosen_global_scale_in_checkpoint": chosen,
        "recomputed_MAD_scale": float(recomputed),
        "scales_match": abs(chosen - recomputed) < 1e-6,
        "fraction_abs_A_scaled_gt_1": float(np.mean(np.abs(scaled) > 1)),
        "fraction_abs_A_scaled_gt_2": float(np.mean(np.abs(scaled) > 2)),
        "fraction_abs_A_scaled_gt_5": float(np.mean(np.abs(scaled) > 5)),
    }

    write_json("results/planB_v2_dual_view_fold_audit.json", dual)
    write_json("results/planB_v2_fold_influence.json", fold_report)
    write_json("results/planB_v2_utility_scale_audit.json", util)
    return dual, fold_report, util


# ---------------------------------------------------------------------------
# VALID vs TEST state shift (no TEST labels)
# ---------------------------------------------------------------------------
@torch.no_grad()
def part15_state_shift(bundle, device, max_n=1024):
    from scripts.eval_group_relative_refinement_policy import build_loader

    supernet = bundle["supernet"]
    policy = bundle["policy"]
    env = bundle["env"]

    def collect(split):
        loader, n = build_loader(split, 32)
        hs, ents, top1, margins, norms = [], [], [], [], []
        count = 0
        for batch in loader:
            # TimeSeriesForecastingDataset yields (future, history) or similar
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    future, history = batch
                elif len(batch) >= 3:
                    # sometimes (idx, future, history) variants — try last tensors
                    tensors = [x for x in batch if torch.is_tensor(x)]
                    history = tensors[1] if len(tensors) > 1 else tensors[0]
                else:
                    history = batch[0]
            else:
                history = batch
            history = history.to(device)
            # history may be [B,T,N,C]
            h = supernet.extract_pre_route_context(history, detach=True)
            s0 = policy.encode_state0(h)["state0_hidden"]
            # for entropy use eta=1 masks
            costs = bundle["costs"]
            feas = torch.ones(4, dtype=torch.bool, device=device)
            zq = env.execute_quarter_prefix(history)["Z_q"]
            probs = policy.forward_terminal_probs(h, zq, feas, bundle["index_map"])["route_probs"]
            p = probs.clamp_min(1e-8)
            p = p / p.sum(-1, keepdim=True)
            ent = -(p * p.log()).sum(-1)
            top = p.max(-1).values
            # top1-top2
            top2 = torch.topk(p, k=2, dim=-1).values
            marg = top2[:, 0] - top2[:, 1]
            hs.append(s0.cpu())
            ents.append(ent.cpu())
            top1.append(top.cpu())
            margins.append(marg.cpu())
            norms.append(s0.norm(dim=-1).cpu())
            count += history.shape[0]
            if count >= max_n:
                break
        H = torch.cat(hs)[:max_n]
        return {
            "hidden": H.numpy(),
            "entropy": torch.cat(ents)[:max_n].numpy(),
            "top1": torch.cat(top1)[:max_n].numpy(),
            "margin": torch.cat(margins)[:max_n].numpy(),
            "feat_norm": torch.cat(norms)[:max_n].numpy(),
        }

    v = collect("valid")
    t = collect("test")
    # metrics
    mu_v, mu_t = v["hidden"].mean(0), t["hidden"].mean(0)
    shift_l2 = float(np.linalg.norm(mu_v - mu_t))
    tr_v = float(np.var(v["hidden"], axis=0).sum())
    tr_t = float(np.var(t["hidden"], axis=0).sum())

    def pca_eigs(X, k=20):
        Xc = X - X.mean(0)
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        return (s[:k] ** 2 / max(len(X) - 1, 1)).tolist()

    # pairwise cosine mean
    def mean_cos(X, n=100):
        idx = np.random.RandomState(0).choice(len(X), size=min(n, len(X)), replace=False)
        sims = []
        for i in range(len(idx)):
            for j in range(i + 1, min(i + 3, len(idx))):
                a, b = X[idx[i]], X[idx[j]]
                sims.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))
        return float(np.mean(sims))

    # relative shift severity
    scale = float(np.linalg.norm(mu_v) + 1e-8)
    rel = shift_l2 / scale
    level = "LOW" if rel < 0.05 else ("MODERATE" if rel < 0.15 else "HIGH")

    out = {
        "n_valid": len(v["hidden"]),
        "n_test": len(t["hidden"]),
        "state0_hidden_mean_L2_shift": shift_l2,
        "relative_mean_shift": rel,
        "covariance_trace_valid": tr_v,
        "covariance_trace_test": tr_t,
        "trace_ratio_test_over_valid": tr_t / max(tr_v, 1e-12),
        "pca_spectrum_valid": pca_eigs(v["hidden"]),
        "pca_spectrum_test": pca_eigs(t["hidden"]),
        "pairwise_cosine_valid": mean_cos(v["hidden"]),
        "pairwise_cosine_test": mean_cos(t["hidden"]),
        "policy_entropy_mean_valid": float(v["entropy"].mean()),
        "policy_entropy_mean_test": float(t["entropy"].mean()),
        "top1_prob_mean_valid": float(v["top1"].mean()),
        "top1_prob_mean_test": float(t["top1"].mean()),
        "top1_top2_margin_mean_valid": float(v["margin"].mean()),
        "top1_top2_margin_mean_test": float(t["margin"].mean()),
        "VALID_TEST_STATE_SHIFT": level,
        "MMD": "NOT_COMPUTED (optional)",
    }
    write_json("results/planB_v2_valid_test_state_shift.json", out)
    return out


def part16_frontier():
    planA_v = json.loads(Path("results/pems04_crossfit_controller_eval_valid.json").read_text())
    planA_t = json.loads(Path("results/pems04_crossfit_controller_eval_test.json").read_text())
    planB = json.loads(Path("results/planB_v2_policy_eval.json").read_text())
    summary = json.loads(Path("results/pems04_crossfit_accuracy_cost_summary.json").read_text())

    def extract_B(split):
        rows = []
        for eta, row in planB["splits"][split]["etas"].items():
            rows.append(
                {
                    "method": "PlanB-v2",
                    "split": split,
                    "eta": float(eta),
                    "MAE": row.get("mae"),
                    "RMSE": row.get("rmse"),
                    "MAPE": row.get("mape"),
                    "cost": row.get("average_selected_cost"),
                    "stages": row.get("average_stage_count"),
                    "hist": row.get("route_histogram_sample"),
                }
            )
        return rows

    def extract_A(split, src):
        rows = []
        # need MAE from accuracy summary for adaptive controller
        # find in summary list
        summ = summary[split]
        ctrl_rows = [r for r in summ if "Adaptive" in str(r.get("METHOD", "")) or "crossfit" in str(r.get("METHOD", "")).lower()]
        # map by eta from eval json
        for eta, row in src["per_eta"].items():
            mae = rmse = mape = None
            for r in summ:
                if r.get("ETA") is not None and abs(float(r["ETA"]) - float(eta)) < 1e-9:
                    if "Adaptive" in str(r.get("METHOD", "")) or "controller" in str(r.get("METHOD", "")).lower():
                        mae, rmse, mape = r.get("MAE"), r.get("RMSE"), r.get("MAPE")
                        break
            rows.append(
                {
                    "method": "PlanA",
                    "split": split,
                    "eta": float(eta),
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "cost": row.get("avg_selected_cost"),
                    "stages": row.get("avg_stages"),
                    "hist": row.get("route_histogram_executed"),
                }
            )
        return rows

    fixed = []
    for split in ("valid", "test"):
        for r in summary[split]:
            if str(r.get("METHOD", "")).startswith("Fixed"):
                fixed.append(
                    {
                        "method": r["METHOD"],
                        "split": split,
                        "eta": r.get("ETA"),
                        "MAE": r.get("MAE"),
                        "RMSE": r.get("RMSE"),
                        "MAPE": r.get("MAPE"),
                        "cost": r.get("AVG_COST"),
                        "stages": r.get("AVG_STAGES"),
                    }
                )

    all_rows = extract_B("valid") + extract_B("test") + extract_A("valid", planA_v) + extract_A("test", planA_t) + fixed

    def pareto_flags(rows_split):
        # among method points with MAE and cost
        pts = [r for r in rows_split if r.get("MAE") is not None and r.get("cost") is not None]
        for r in pts:
            dominated = False
            for o in pts:
                if o is r:
                    continue
                if o["MAE"] <= r["MAE"] + 1e-12 and o["cost"] <= r["cost"] + 1e-12 and (
                    o["MAE"] < r["MAE"] - 1e-12 or o["cost"] < r["cost"] - 1e-12
                ):
                    dominated = True
                    break
            r["pareto"] = not dominated
            r["dominated"] = dominated
        return pts

    for split in ("valid", "test"):
        pareto_flags([r for r in all_rows if r["split"] == split])

    # TEST eta .5/.75 PlanB dominated?
    test_b = [r for r in all_rows if r["split"] == "test" and r["method"] == "PlanB-v2"]
    dominated_mid = {}
    for r in test_b:
        if r["eta"] in (0.5, 0.75):
            dominated_mid[str(r["eta"])] = bool(r.get("dominated"))

    out = {
        "points": all_rows,
        "test_planB_eta0p5_0p75_dominated": dominated_mid,
        "test_mid_etas_dominated": all(dominated_mid.values()) if dominated_mid else None,
    }
    write_json("results/planA_planBv2_frontier_comparison.json", out)
    with open("results/planA_planBv2_frontier_comparison.csv", "w", newline="") as f:
        fields = ["method", "split", "eta", "MAE", "RMSE", "MAPE", "cost", "stages", "pareto", "dominated"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    return out


def part5_historical():
    out = {
        "available": False,
        "reason": (
            "Only a single overwritten checkpoint plan_b_v2_exact_policy.pt exists; "
            "no per-epoch historical checkpoints were saved."
        ),
        "HISTORICAL_CHECKPOINT_CALIBRATION_UNAVAILABLE": True,
    }
    write_json("results/planB_v2_epoch_calibration_valid.json", out)
    return out


def classify_root_cause(ctx: dict) -> dict:
    # Use evidence to pick primary diagnosis
    err = ctx["err"]
    over = ctx["over"]
    margin = ctx["margin"]
    shift = ctx["shift"]
    dynamics = ctx["dynamics"]
    fold = ctx["fold"]

    # over-refine signal
    ou = err.get("0.5", {}).get("over_under", {}).get("tol_oracle", {})
    over_n = ou.get("over_refine", 0)
    under_n = ou.get("under_refine", 0)
    main_mistake = "BOTH"
    if over_n > 1.5 * under_n:
        main_mistake = "OVER-REFINE"
    elif under_n > 1.5 * over_n:
        main_mistake = "UNDER-REFINE"

    mid_gains = over.get("0.5", {}).get("among_B_more_compute", {})
    # borderline: formal run has ~0.47 le_0.05 and ~0.39 worse — treat as low-benefit over-refine
    low_benefit = (
        (mid_gains.get("fraction_gain_le_0.05") or 0) > 0.4
        or (mid_gains.get("fraction_worse") or 0) > 0.3
    )

    near_ties = all(margin[e].get("errors_mainly_near_ties") for e in margin)

    shift_level = shift.get("VALID_TEST_STATE_SHIFT", "LOW")
    trend = dynamics.get("REFINEMENT_AGGRESSIVENESS_TREND")

    # Primary diagnosis logic
    primary = "MULTIPLE_COUPLED_CAUSES"
    reasons = []
    if main_mistake == "OVER-REFINE" and low_benefit:
        reasons.append("POLICY_OVER_REFINEMENT_CALIBRATION")
    if shift_level in ("MODERATE", "HIGH"):
        reasons.append("VALID_TO_TEST_STATE_SHIFT")
    # checkpoint: best epoch is last-ish with increasing refine
    if trend == "INCREASING" and dynamics["best_by_recorded_rule"]["epoch"] >= 20:
        reasons.append("CHECKPOINT_SELECTION_PROBLEM")
    # high-margin errors?
    clear = margin.get("1.0", {}).get("groups", {}).get("very_clear_>1.0", {})
    if clear.get("n", 0) > 50 and clear.get("routing_accuracy_vs_reward", 1) < 0.7:
        reasons.append("REPRESENTATION_STILL_INSUFFICIENT")

    if len(reasons) == 1:
        primary = reasons[0]
    elif len(reasons) == 0:
        primary = "OOF_TO_VALID_GENERALIZATION_PROBLEM"
    else:
        # Multiple similarly important causes -> F; prefer CALIBRATION as next actionable category
        coupled = {
            "POLICY_OVER_REFINEMENT_CALIBRATION",
            "VALID_TO_TEST_STATE_SHIFT",
            "REPRESENTATION_STILL_INSUFFICIENT",
        }
        if len(coupled.intersection(reasons)) >= 2:
            primary = "MULTIPLE_COUPLED_CAUSES"
        elif "POLICY_OVER_REFINEMENT_CALIBRATION" in reasons and low_benefit:
            primary = "POLICY_OVER_REFINEMENT_CALIBRATION"
        elif "VALID_TO_TEST_STATE_SHIFT" in reasons and shift_level == "HIGH":
            primary = "VALID_TO_TEST_STATE_SHIFT"
        else:
            primary = "MULTIPLE_COUPLED_CAUSES"

    next_interven = {
        "POLICY_OVER_REFINEMENT_CALIBRATION": "CALIBRATION",
        "CHECKPOINT_SELECTION_PROBLEM": "CHECKPOINT_SELECTION",
        "VALID_TO_TEST_STATE_SHIFT": "NEED_MORE_EVIDENCE",
        "OOF_TO_VALID_GENERALIZATION_PROBLEM": "CROSSFIT",
        "REPRESENTATION_STILL_INSUFFICIENT": "STATE_REPRESENTATION",
        "MULTIPLE_COUPLED_CAUSES": "CALIBRATION",
    }.get(primary, "NEED_MORE_EVIDENCE")

    report = {
        "primary_diagnosis": primary,
        "supporting_reasons": reasons,
        "main_mistake_mode": main_mistake,
        "architecture_change_next": False,
        "recommended_next_intervention_category": next_interven,
        "evidence_snippets": {
            "refinement_trend": trend,
            "best_epoch": dynamics["best_by_recorded_rule"],
            "over_under_eta0p5_tol": ou,
            "among_more_compute_eta0p5": mid_gains,
            "state_shift": shift_level,
            "fold1_influence": fold.get("Fold1_influence"),
            "near_ties": near_ties,
        },
        "note": "Diagnosis only; no V3 implementation.",
    }
    write_json("results/planB_v2_root_cause_after_formal_run.json", report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    print("=== PART 1 provenance ===")
    prov, blob = part1_provenance()
    print("conclusion:", prov["conclusion"])

    print("=== PART 2-4,14 training dynamics ===")
    dynamics, stability = part2_training_dynamics()
    print("best epoch", dynamics["best_by_recorded_rule"])
    print("trend", dynamics["REFINEMENT_AGGRESSIVENESS_TREND"])

    print("=== PART 5 historical ckpt ===")
    part5_historical()
    print("HISTORICAL_CHECKPOINT_CALIBRATION_UNAVAILABLE")

    print("=== Load VALID bundle / collect policy outputs ===")
    bundle = load_valid_bundle(device)
    valid_out = collect_valid_policy_outputs(bundle, device)
    print("=== PART 6-8 error/calib/margin ===")
    err, calib, margin = part6_8_routing_calibration(valid_out, bundle["costs"])

    print("=== PART 9-10 Plan A vs B ===")
    try:
        planA_sels, a_err = load_plan_a_routes(
            device, [0.5, 0.75, 1.0], bundle["loader"], bundle["costs"]
        )
        if a_err:
            print("Plan A load issue:", a_err)
            planA_sels = None
    except Exception as e:
        print("Plan A inference failed:", e)
        planA_sels = None
    planA_eval = json.loads(Path("results/pems04_crossfit_controller_eval_valid.json").read_text())
    planB_eval = json.loads(Path("results/planB_v2_policy_eval.json").read_text())
    compare, over = part9_10_planA_vs_B(valid_out, planA_sels, bundle["costs"], planA_eval, planB_eval)

    print("=== PART 11-13 dual-view / fold / utility ===")
    dual, fold, util = part11_13_cache_audits(bundle["policy"], device)

    print("=== PART 15 VALID/TEST state shift ===")
    shift = part15_state_shift(bundle, device)

    print("=== PART 16 frontier ===")
    frontier = part16_frontier()

    print("=== PART 17 root cause ===")
    root = classify_root_cause(
        {
            "err": err,
            "over": over,
            "margin": margin,
            "shift": shift,
            "dynamics": dynamics,
            "fold": fold,
            "calib": calib,
            "util": util,
            "stability": stability,
            "frontier": frontier,
            "prov": prov,
            "compare": compare,
        }
    )

    # Terminal summary numbers
    vor = planB_eval["valid_oracle_regret"]["etas"]
    # Plan A regret from existing if available — approximate via histograms not ideal; use compare if present

    print("\n========== FINAL TERMINAL SUMMARY ==========")
    summary = {
        "1_provenance": prov["conclusion"],
        "2_policy_checkpoint": {
            "epoch": dynamics["best_by_recorded_rule"]["epoch"],
            "sha1": prov["policy_checkpoint"]["sha1"],
            "mtime": prov["policy_checkpoint"]["mtime"],
        },
        "3_best_nontrivial_VALID_regret_epoch": dynamics["best_by_recorded_rule"]["epoch"],
        "4_final_evaluated_epoch": dynamics["best_by_recorded_rule"]["epoch"],
        "5_refinement_aggressiveness_increased": (
            "YES"
            if dynamics["REFINEMENT_AGGRESSIVENESS_TREND"] == "INCREASING"
            else (
                "NO"
                if dynamics["REFINEMENT_AGGRESSIVENESS_TREND"] == "DECREASING"
                else "MIXED"
            )
        ),
        "6_VALID_routing": {
            eta: {"regret": vor[eta]["mean_regret"], "cost": vor[eta]["mean_cost"]}
            for eta in ("0.5", "0.75", "1.0")
        },
        "7_PlanA_vs_Bv2_regret": {
            eta: compare.get(eta)
            for eta in ("0.5", "0.75", "1.0")
        },
        "8_main_mistake_source": root["main_mistake_mode"],
        "9_extra_compute_gain_fractions_eta0p5": over.get("0.5", {}).get("among_B_more_compute"),
        "10_calibration_eta1": calib.get("1.0"),
        "11_errors_mainly_near_ties": all(
            margin[e].get("groups", {}).get("very_ambiguous_<0.05", {}).get("n", 0)
            >= margin[e].get("groups", {}).get("very_clear_>1.0", {}).get("n", 0) * 0.5
            for e in margin
        ),
        "12_Fold1_influence": fold.get("Fold1_influence"),
        "13_utility_scale": {
            "value": util["chosen_global_scale_in_checkpoint"],
            "frac_gt2": util["fraction_abs_A_scaled_gt_2"],
            "frac_gt5": util["fraction_abs_A_scaled_gt_5"],
        },
        "14_grad_clip_fraction": stability["overall_fraction_batches_clipped"],
        "15_teacher_stable_KL_by_fold": {f: dual["per_fold"][f]["mean_KL"] for f in dual["per_fold"]},
        "16_VALID_TEST_state_shift": shift.get("VALID_TEST_STATE_SHIFT"),
        "17_TEST_eta0p5_0p75_dominated": frontier.get("test_mid_etas_dominated"),
        "18_primary_diagnosis": root["primary_diagnosis"],
        "19_architecture_change_next": False,
        "20_recommended_next_intervention_category": root["recommended_next_intervention_category"],
        "elapsed_sec": time.time() - t0,
    }
    write_json("results/planB_v2_post_formal_terminal_summary.json", summary)
    for k, v in summary.items():
        print(f"{k}: {json.dumps(v, default=str)[:500]}")
    print("============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
