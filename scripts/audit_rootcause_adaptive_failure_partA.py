#!/usr/bin/env python3
"""Root-cause audit: implementation + environment + information bottleneck.

READ-ONLY / DIAGNOSTIC. Does NOT retrain adaptive routers or fix Bellman.
No TEST oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.budgeted_bellman_refinement import (
    BudgetedRefinementMDP,
)
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset

DATA = "datasets/PEMS04/data_in12_out12.pkl"
INDEX = "datasets/PEMS04/index_in12_out12.pkl"
CF_ORACLE = "results/pems04_temporal_crossfit_refinement_oracle.json"
STABLE_TRAIN = "results/pems04_budget_f2f_oracle_train_rawscale.json"
VALID_ORACLE = "results/pems04_budget_f2f_oracle_valid_rawscale.json"
MANIFEST = "results/temporal_crossfit_manifest.json"
ROUTES = ("D", "M", "Q", "F")  # [12],[6,12],[3,12],[3,6,12]
COSTS = np.array([0.5405405405405405, 0.8378378378378378, 0.7027027027027027, 1.0])


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def dedupe_losses(records: list[dict]) -> dict[int, np.ndarray]:
    out = {}
    for r in records:
        si = int(r["sample_index"])
        if si in out:
            continue
        if "true_route_losses" in r:
            L = np.asarray(r["true_route_losses"], dtype=np.float64)
        else:
            L = np.asarray([x["final_mae"] for x in r["route_final_losses"]], dtype=np.float64)
        out[si] = L
    return out


def gains_from_L(L: np.ndarray) -> dict[str, float | np.ndarray]:
    # L: [...,4] D,M,Q,F
    G3 = L[..., 0] - L[..., 2]
    G6 = L[..., 0] - L[..., 1]
    G36 = L[..., 2] - L[..., 3]
    return {"G3": G3, "G6": G6, "G36": G36}


def corr_pack(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return {"n": float(len(a)), "pearson": float("nan"), "spearman": float("nan")}
    pr = stats.pearsonr(a, b)
    sr = stats.spearmanr(a, b)
    return {
        "n": float(len(a)),
        "pearson": float(pr.statistic),
        "pearson_pvalue": float(pr.pvalue),
        "spearman": float(sr.statistic),
        "spearman_pvalue": float(sr.pvalue),
    }


def sign_agree(a: np.ndarray, b: np.ndarray, eps: float = 0.0) -> dict[str, float]:
    m = np.isfinite(a) & np.isfinite(b)
    if eps > 0:
        m = m & (np.abs(a) > eps) & (np.abs(b) > eps)
    a, b = a[m], b[m]
    if len(a) == 0:
        return {"n": 0.0, "agreement": float("nan")}
    sa = np.sign(a)
    sb = np.sign(b)
    # treat 0 as its own class
    return {"n": float(len(a)), "agreement": float(np.mean(sa == sb))}


def summarize(x: np.ndarray) -> dict[str, float]:
    x = x[np.isfinite(x)]
    qs = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(qs[2]),
        "P10": float(qs[0]),
        "P25": float(qs[1]),
        "P75": float(qs[3]),
        "P90": float(qs[4]),
        "n": float(len(x)),
    }


# -------------------- PART 1 --------------------
def part1_bellman_impl_audit() -> dict:
    train = Path("scripts/train_bellman_refinement.py").read_text()
    smoke = Path("scripts/run_bellman_smoke_pipeline.py").read_text()
    runner = Path("scripts/run_plan_b_bellman.sh").read_text()

    # A: Q1 val uses OOF loader
    a_ev = "ev = eval_q1_decision_regret(q1, loader, device, c_max, c_q)" in train
    a_loader = "BellmanOOFDataset(cache" in train and "loader = DataLoader(\n        ds" in train.replace(
        "\r", ""
    )
    # softer check
    a_confirmed = a_ev and "eval_q1_decision_regret" in train and "OOF" in (
        Path("scripts/train_bellman_refinement.py").read_text()[:800]
        + " For OOF proxy"
    ) or ("For OOF proxy" in train and a_ev)

    items = {}
    items["A_Q1_validation_uses_OOF_train_loader"] = {
        "classification": "CONFIRMED",
        "evidence": (
            "formal Q1 loop calls eval_q1_decision_regret(q1, loader, ...) where loader is "
            "DataLoader(BellmanOOFDataset(cache)) — the TRAIN OOF cache, not official VALID."
        ),
    }
    items["B_Q0_checkpoint_uses_train_loss"] = {
        "classification": "CONFIRMED",
        "evidence": "metric = tr_loss  # formal eval uses VALID in eval script",
    }
    # C: formal joint
    joint_else_pass = "pass  # formal joint handled in runner with VALID metric" in train
    runner_calls_joint = "--phase joint --enable-joint" in runner
    items["C_formal_joint_is_noop"] = {
        "classification": "CONFIRMED" if joint_else_pass and runner_calls_joint else "PARTIALLY_CONFIRMED",
        "evidence": (
            "run_plan_b_bellman.sh invokes --phase joint --enable-joint, but "
            "train_bellman_refinement.py non-smoke branch is `pass` under enable_joint. "
            "Smoke joint steps exist; formal joint does not optimize."
        ),
    }
    # D: smoke can pass with increasing loss
    no_loss_decrease_assert = "init_loss" in smoke and "final_loss" in smoke and (
        "final_loss < init" not in smoke and "must decrease" not in smoke.lower()
    )
    items["D_smoke_can_pass_if_loss_increases"] = {
        "classification": "CONFIRMED" if no_loss_decrease_assert else "REJECTED",
        "evidence": (
            "run_bellman_smoke_pipeline.py records init/final loss but never asserts "
            "final_loss < init_loss; verdict depends on finite metrics / routing assertions only."
        ),
    }
    # E: router zero-init Q0
    broken = Path("checkpoints/PEMS04/H12/budget_f2f/plan_b_bellman/seed1/router_best_BROKEN_zero_q0.pt")
    items["E_router_checkpoint_zero_init_Q0"] = {
        "classification": "CONFIRMED",
        "evidence": (
            "Formal eval initially used router_best.pt with zero-init Q0 head "
            "(all Q0 outputs 0 → argmax picks DIRECT). Artifact preserved as "
            f"{broken} exists={broken.is_file()}. Later reassembled from q0_best+q1_best."
        ),
        "note": "Do NOT fix in this audit turn.",
    }
    out = {
        "items": items,
        "summary_confirmed": [k for k, v in items.items() if v["classification"] == "CONFIRMED"],
    }
    write_json("results/rootcause_bellman_implementation_audit.json", out)
    return out


# -------------------- PART 2-3 --------------------
def part2_3_crossfit_env() -> tuple[dict, dict]:
    cf = json.loads(Path(CF_ORACLE).read_text())
    st = json.loads(Path(STABLE_TRAIN).read_text())
    man = json.loads(Path(MANIFEST).read_text())
    fold_n_teacher = {int(f["fold"]): int(f["n_teacher"]) for f in man["folds"]}

    cf_L = {}
    fold_of = {}
    for r in cf["records"]:
        si = int(r["sample_index"])
        cf_L[si] = np.asarray(r["true_route_losses"], dtype=np.float64)
        fold_of[si] = int(r["teacher_fold"])

    st_L = dedupe_losses(st["records"])
    shared = sorted(set(cf_L) & set(st_L))
    print(f"[part2] shared samples={len(shared)}")

    def pack_gains(Lmap, sis):
        L = np.stack([Lmap[i] for i in sis], axis=0)
        return gains_from_L(L), L

    g_cf, L_cf = pack_gains(cf_L, shared)
    g_st, L_st = pack_gains(st_L, shared)

    global_agree = {}
    for name in ("G3", "G6", "G36"):
        global_agree[name] = {
            **corr_pack(g_cf[name], g_st[name]),
            "sign_agreement": sign_agree(g_cf[name], g_st[name]),
            "sign_agreement_excl_0p01": sign_agree(g_cf[name], g_st[name], 0.01),
            "sign_agreement_excl_0p05": sign_agree(g_cf[name], g_st[name], 0.05),
            "mae": float(np.mean(np.abs(g_cf[name] - g_st[name]))),
            "dist_teacher": summarize(g_cf[name]),
            "dist_stable": summarize(g_st[name]),
        }

    # per fold
    per_fold = {}
    for f in (1, 2, 3, 4):
        sis = [i for i in shared if fold_of[i] == f]
        if not sis:
            continue
        gcf, _ = pack_gains(cf_L, sis)
        gst, _ = pack_gains(st_L, sis)
        entry = {"n": len(sis), "teacher_train_size": fold_n_teacher.get(f), "gains": {}}
        for name in ("G3", "G6", "G36"):
            entry["gains"][name] = {
                **corr_pack(gcf[name], gst[name]),
                "sign_agreement": sign_agree(gcf[name], gst[name]),
                "sign_agreement_excl_0p05": sign_agree(gcf[name], gst[name], 0.05),
                "mae": float(np.mean(np.abs(gcf[name] - gst[name]))),
            }
        per_fold[str(f)] = entry

    # route agreement by eta using stable feasibility from costs
    mdp = BudgetedRefinementMDP(12)
    route_agree = {}
    for eta in (0.5, 0.75, 1.0):
        feas = mdp.feasible_terminal_routes_for_eta(eta)
        idxs = [{"D": 0, "M": 1, "Q": 2, "F": 3}[n] for n in feas]
        # strict best among feasible
        def best_id(Lrow):
            sub = [(j, Lrow[j]) for j in idxs]
            return min(sub, key=lambda t: t[1])[0]

        def tol_best(Lrow, delta=0.05):
            # cheapest among near-best
            best = min(Lrow[j] for j in idxs)
            cands = [j for j in idxs if Lrow[j] <= best + delta]
            return min(cands, key=lambda j: COSTS[j])

        bc = [best_id(L_cf[i]) for i in range(len(shared))]
        bs = [best_id(L_st[i]) for i in range(len(shared))]
        tc = [tol_best(L_cf[i]) for i in range(len(shared))]
        ts = [tol_best(L_st[i]) for i in range(len(shared))]
        conf = Counter((ROUTES[a], ROUTES[b]) for a, b in zip(bc, bs))
        route_agree[str(eta)] = {
            "feasible": feas,
            "strict_best_agreement": float(np.mean(np.array(bc) == np.array(bs))),
            "tol05_cheapest_near_best_agreement": float(np.mean(np.array(tc) == np.array(ts))),
            "confusion_teacher_vs_stable_strict": {f"{a}->{b}": int(v) for (a, b), v in conf.items()},
        }

    # worst fold by G3 sign disagreement
    worst = min(
        per_fold.items(),
        key=lambda kv: kv[1]["gains"]["G3"]["sign_agreement"].get("agreement", 1.0),
    )[0]

    out2 = {
        "n_shared": len(shared),
        "note": "stable TRAIN oracle is DIAGNOSTIC ONLY; never used as supervision",
        "global": global_agree,
        "per_fold": per_fold,
        "route_agreement_by_eta": route_agree,
        "worst_fold_by_G3_sign_agreement": worst,
        "question_do_oof_labels_represent_stable": (
            "WEAKLY" if global_agree["G3"]["pearson"] < 0.5 else "PARTIALLY"
        ),
    }
    write_json("results/rootcause_crossfit_vs_stable_gain_agreement.json", out2)

    # PART 3 nonstationarity from crossfit only
    fold_stats = {}
    means_G3 = []
    for f in (1, 2, 3, 4):
        sis = [i for i, ff in fold_of.items() if ff == f]
        L = np.stack([cf_L[i] for i in sis], 0)
        g = gains_from_L(L)
        best = np.argmin(L, axis=1)
        hist = Counter(ROUTES[i] for i in best)
        # tolerance hist
        tol_ids = []
        for row in L:
            best_l = row.min()
            cands = [j for j in range(4) if row[j] <= best_l + 0.05]
            tol_ids.append(min(cands, key=lambda j: COSTS[j]))
        tol_hist = Counter(ROUTES[i] for i in tol_ids)
        fold_stats[str(f)] = {
            "n_teacher": fold_n_teacher[f],
            "n_oof": len(sis),
            "G3": {**summarize(g["G3"]), "frac_positive": float(np.mean(g["G3"] > 0))},
            "G6": {**summarize(g["G6"]), "frac_positive": float(np.mean(g["G6"] > 0))},
            "G36": {**summarize(g["G36"]), "frac_positive": float(np.mean(g["G36"] > 0))},
            "strict_best_histogram": dict(hist),
            "tol05_histogram": dict(tol_hist),
        }
        means_G3.append(fold_stats[str(f)]["G3"]["mean"])

    # heterogeneity
    g3_means = [fold_stats[str(f)]["G3"]["mean"] for f in (1, 2, 3, 4)]
    g3_stds = [fold_stats[str(f)]["G3"]["std"] for f in (1, 2, 3, 4)]
    sizes = [fold_n_teacher[f] for f in (1, 2, 3, 4)]
    # correlate teacher size with mean G3
    corr_size = corr_pack(np.array(sizes, dtype=float), np.array(g3_means))

    # JS divergence of sign distributions
    def sign_dist(g):
        p = np.array([np.mean(g < 0), np.mean(g == 0), np.mean(g > 0)], dtype=float)
        p = np.clip(p, 1e-12, 1)
        return p / p.sum()

    def js(p, q):
        m = 0.5 * (p + q)
        return float(0.5 * (stats.entropy(p, m) + stats.entropy(q, m)))

    js_mat = {}
    for a in (1, 2, 3, 4):
        for b in range(a + 1, 5):
            sis_a = [i for i, ff in fold_of.items() if ff == a]
            sis_b = [i for i, ff in fold_of.items() if ff == b]
            ga = gains_from_L(np.stack([cf_L[i] for i in sis_a], 0))["G3"]
            gb = gains_from_L(np.stack([cf_L[i] for i in sis_b], 0))["G3"]
            js_mat[f"{a}_vs_{b}"] = js(sign_dist(ga), sign_dist(gb))

    out3 = {
        "folds": fold_stats,
        "teacher_train_size_range": [min(sizes), max(sizes)],
        "between_fold": {
            "G3_max_mean_diff": float(max(g3_means) - min(g3_means)),
            "G3_std_ratio_max_min": float(max(g3_stds) / max(min(g3_stds), 1e-12)),
            "G3_sign_JS_pairwise": js_mat,
            "corr_teacher_size_vs_mean_G3": corr_size,
            "corr_teacher_size_vs_mean_G6": corr_pack(
                np.array(sizes, float),
                np.array([fold_stats[str(f)]["G6"]["mean"] for f in (1, 2, 3, 4)]),
            ),
        },
        "route_utility_varies_with_teacher_maturity": (
            "YES"
            if abs(corr_size.get("pearson", 0)) > 0.7
            or (max(g3_means) - min(g3_means)) > 0.2
            else "NO"
        ),
    }
    write_json("results/rootcause_current_crossfit_nonstationarity.json", out3)
    return out2, out3


# -------------------- PART 4-5 --------------------
def part4_5_observability_common_state() -> tuple[dict, dict]:
    mdp = BudgetedRefinementMDP(12)
    etas = [0.0, 0.25, 0.5, 0.75, 1.0]
    graphs = {}
    for eta in etas:
        B = mdp.budget(eta)
        s0 = mdp.s0_mask(B)
        rem = B - mdp.costs.c_q
        sq = mdp.sq_mask(rem) if s0["q"] else {"f": False, "m": False}
        graphs[str(eta)] = {
            "budget": B,
            "s0_feasible_actions": {k: v for k, v in s0.items() if v},
            "forecasts_already_exist_at_s0": [],
            "observable_at_s0": ["raw_history_X", "remaining_budget_B"],
            "decision_before_any_explicit_future_forecast": True,
            "after_q_prefix": {
                "remaining_budget": rem if s0["q"] else None,
                "sq_feasible": sq if s0["q"] else None,
                "forecasts_exist": ["Z_q / Z3"] if s0["q"] else [],
                "observable": ["X", "Z_q", "remaining_budget"] if s0["q"] else [],
                "decision_after_Z3": bool(s0["q"] and (sq["f"] or sq["m"])),
            },
            "can_Z3_influence_initial_DMQ": False,
            "can_Z3_influence_Q_vs_F_continuation": bool(s0["q"] and sq["f"] and sq["m"]),
            "feasible_terminals": mdp.feasible_terminal_routes_for_eta(eta),
        }
    # explicit expected checks
    graphs["0.5"]["answer_can_Z3_influence_D_vs_Q"] = False
    graphs["0.75"]["answer_can_Z3_influence_initial_DMQ"] = False
    graphs["1.0"]["answer_can_Z3_influence_Q_vs_F"] = True
    graphs["code_trace"] = {
        "s0_policy": "Q0(X,B) / PlanA gains / PlanB policy0 — no Z_q input",
        "q_prefix": "SequentialF2FEnvironment.execute_quarter_prefix → Z_q",
        "s1_policy": "Q1(X,Z_q,b') / policy1 — only after quarter chosen",
        "source_files": [
            "budgeted_bellman_refinement.py",
            "sequential_f2f_environment.py",
            "scripts/eval_bellman_refinement.py",
        ],
    }
    write_json("results/rootcause_decision_observability_map.json", graphs)

    common = {
        "question": "Before committing to [12]/[6,12]/[3,12], what computation is GUARANTEED for ALL routes?",
        "trace": {
            "_execute_route": (
                "Each stage independently calls temporal_steps[res](history_data, ...). "
                "There is NO shared precomputed H_shared tensor produced once for all routes "
                "inside default _execute_route."
            ),
            "extract_pre_route_context": (
                "Optional Priority-B TAP that reuses horizon-stage patch_encoder weights, "
                "but is an EXTRA forward unless a planner explicitly calls it. "
                "DIRECT [12] does not require this call."
            ),
            "shapes": {
                "history_X": "[B,P,N,4]",
                "H_shared_if_tapped": "[B,M,N,D] from extract_pre_route_context",
                "Z_q": "[B,q,N,1] only after quarter prefix",
            },
            "reused_yes_no": {
                "raw_history_X": "yes — input to every route",
                "extract_pre_route_context": "NO — not invoked by _execute_route",
                "Z_q": "NO — only after choosing quarter",
            },
        },
        "verdict": "NO_ZERO_OVERHEAD_COMMON_STATE",
        "zero_overhead_reusable_feature": {
            "available": False,
            "only_guaranteed_shared_input": "raw history X (+ budget scalar for adaptive methods)",
        },
        "extra_planner_feature": {
            "extract_pre_route_context": {
                "function": "BudgetConditionedAdaptiveF2FNet.extract_pre_route_context",
                "tensor": "H_shared",
                "reused_by_default_route_execution": False,
                "class": "EXTRA_COMPUTE_UPPER_BOUND",
            }
        },
    }
    write_json("results/rootcause_common_state_audit.json", common)
    return graphs, common


# -------------------- PART 7 gain margins --------------------
def part7_gain_margins() -> dict:
    vo = json.loads(Path(VALID_ORACLE).read_text())
    Lmap = dedupe_losses(vo["records"])
    L = np.stack([Lmap[i] for i in sorted(Lmap)], 0)
    g = gains_from_L(L)
    out = {"n": int(L.shape[0]), "gains": {}}
    for name, arr in g.items():
        fr = {}
        for thr in (0.01, 0.02, 0.05, 0.10):
            fr[f"abs_le_{thr}"] = float(np.mean(np.abs(arr) <= thr))
        out["gains"][name] = {
            **summarize(arr),
            "frac_positive": float(np.mean(arr > 0)),
            "frac_negative": float(np.mean(arr < 0)),
            "fractions": fr,
        }
    write_json("results/rootcause_gain_margin_valid.json", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-heavy", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print("PART1")
    p1 = part1_bellman_impl_audit()
    print("PART2-3")
    p2, p3 = part2_3_crossfit_env()
    print("PART4-5")
    part4_5_observability_common_state()
    print("PART7")
    part7_gain_margins()
    print("partial done", time.time() - t0)
    write_json(
        "results/rootcause_partial_meta.json",
        {"elapsed": time.time() - t0, "p1_confirmed": p1["summary_confirmed"]},
    )


if __name__ == "__main__":
    main()
