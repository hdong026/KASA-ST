#!/usr/bin/env python3
"""Rootcause PART6/8/9/10/11: same-hist random + learnability probes + verdict.

VALID only. No TEST oracle. Diagnostic probes only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.budgeted_bellman_refinement import (
    BudgetedRefinementMDP,
    greedy_masked_argmax,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.plan_b_v2_state_cache import load_supernet_strict
from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    SequentialF2FEnvironment,
)
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset

DATA = "datasets/PEMS04/data_in12_out12.pkl"
INDEX = "datasets/PEMS04/index_in12_out12.pkl"
VALID_ORACLE = "results/pems04_budget_f2f_oracle_valid_rawscale.json"
COSTS = np.array([0.5405405405405405, 0.8378378378378378, 0.7027027027027027, 1.0])
ROUTES = ("D", "M", "Q", "F")
DEFAULT_SUPERNET = (
    "checkpoints/PEMS04/H12/budget_f2f/"
    "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
    "b5678fda5e8d94ed028c6c8bb073461d/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def dedupe_L(path=VALID_ORACLE):
    recs = json.loads(Path(path).read_text())["records"]
    out = {}
    for r in recs:
        si = int(r["sample_index"])
        if si in out:
            continue
        out[si] = np.asarray([x["final_mae"] for x in r["route_final_losses"]], dtype=np.float64)
    return out


def gains(L):
    return L[:, 0] - L[:, 2], L[:, 0] - L[:, 1], L[:, 2] - L[:, 3]  # G3,G6,G36


# ---------- collect per-sample routes ----------
@torch.no_grad()
def collect_planA_routes(device, etas, n=None):
    from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_forecast_refinement_route import (
        AdaptiveForecastRefinementRouteNet,
    )
    from scripts.train_forecast_refinement_controller import _build_model

    ckpt = Path(
        "checkpoints/PEMS04/H12/budget_f2f/crossfit_refinement_controller/"
        "refinement_controller_best_val_regret.pt"
    )
    blob = torch.load(ckpt, map_location="cpu")
    # build via eval script pattern if needed
    from scripts.eval_forecast_refinement_controller import build_eval_model

    # fallback simple: use audit loader if available
    try:
        from scripts.audit_plan_b_v2_post_formal import load_plan_a_controller

        net = load_plan_a_controller(device)
    except Exception:
        # minimal rebuild
        supernet, _ = load_supernet_strict(DEFAULT_SUPERNET, device)
        # controller weights in blob
        state = blob.get("model_state_dict") or blob.get("state_dict") or blob
        # Prefer dedicated eval path
        from scripts.report_plan_a_crossfit_complete import load_controller_for_eval

        net = load_controller_for_eval(device)

    ds = IndexedTimeSeriesForecastingDataset(DATA, INDEX, "valid")
    n = len(ds) if n is None else min(n, len(ds))
    out = {str(e): np.zeros(n, dtype=np.int64) for e in etas}
    # Use controller forward if API known
    if hasattr(net, "set_eta"):
        for eta in etas:
            net.set_eta(eta)
            for i in range(n):
                fut, hist, si = ds[i]
                x = hist.unsqueeze(0).to(device)
                o = net(history_data=x, train=False, return_all=True)
                rid = o.get("executed_route_id", o.get("selected_route_id"))
                out[str(eta)][i] = int(rid.view(-1)[0].item())
    else:
        raise RuntimeError("Plan A net API unsupported")
    return out


@torch.no_grad()
def collect_routes_generic(device, etas, which: str, n=None):
    """Collect route ids for PlanA / Bv2 / Bellman on VALID."""
    from scripts.eval_bellman_refinement import BellmanEvalNet, load_router
    from scripts.eval_plan_b_v2 import PlanBV2EvalNet
    from basicts.archs.arch_zoo.ChainForecasting_arch.group_relative_refinement_policy_v2 import (
        GroupRelativeRefinementPolicyV2,
    )
    from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
        SequentialF2FEnvironment,
    )

    supernet, _ = load_supernet_strict(DEFAULT_SUPERNET, device)
    ds = IndexedTimeSeriesForecastingDataset(DATA, INDEX, "valid")
    n = len(ds) if n is None else min(n, len(ds))
    out = {str(e): np.zeros(n, dtype=np.int64) for e in etas}

    if which == "bellman":
        q0, q1, scale, c_max = load_router(
            Path("checkpoints/PEMS04/H12/budget_f2f/plan_b_bellman/seed1/router_best.pt"),
            device,
        )
        mdp = BudgetedRefinementMDP(12)
        net = BellmanEvalNet(supernet, q0, q1, mdp, c_max=c_max).to(device)
        for eta in etas:
            net.set_eta(eta)
            for i in range(n):
                _, hist, _ = ds[i]
                o = net(history_data=hist.unsqueeze(0).to(device), return_all=True)
                out[str(eta)][i] = int(o["executed_route_id"].view(-1)[0].item())
        return out

    if which == "bv2":
        pol_blob = torch.load(
            "checkpoints/PEMS04/H12/budget_f2f/plan_b_v2_exact_policy.pt", map_location="cpu"
        )
        policy = GroupRelativeRefinementPolicyV2()
        policy.load_state_dict(pol_blob["model"] if "model" in pol_blob else pol_blob.get("policy", pol_blob))
        policy.to(device).eval()
        env = SequentialF2FEnvironment(supernet)
        runner = PlanBV2EvalNet(supernet, policy, env, eta=1.0).to(device)
        for eta in etas:
            runner.set_eta(eta)
            for i in range(n):
                _, hist, _ = ds[i]
                o = runner(history_data=hist.unsqueeze(0).to(device), return_all=True)
                out[str(eta)][i] = int(o["executed_route_id"].view(-1)[0].item())
        return out

    if which == "plana":
        # reuse post-formal helper
        from scripts.audit_plan_b_v2_post_formal import load_plan_a_and_routes

        # fallback: manual using controller eval net
        from scripts.eval_forecast_refinement_controller import main as _unused

        # Implement via AdaptiveForecastRefinementRouteNet load in train script style
        import importlib

        mod = importlib.import_module("scripts.eval_forecast_refinement_controller")
        # Build using known checkpoint
        ckpt = (
            "checkpoints/PEMS04/H12/budget_f2f/crossfit_refinement_controller/"
            "refinement_controller_best_val_regret.pt"
        )
        # Use evaluate path internals: load model function if present
        if hasattr(mod, "load_model"):
            net = mod.load_model(ckpt, device)
        else:
            # Construct via report_plan_a
            from scripts.report_plan_a_crossfit_complete import build_adaptive_eval_net

            net = build_adaptive_eval_net(device)
        for eta in etas:
            if hasattr(net, "set_eta"):
                net.set_eta(float(eta))
            elif hasattr(net, "eta"):
                net.eta = float(eta)
            for i in range(n):
                _, hist, _ = ds[i]
                o = net(history_data=hist.unsqueeze(0).to(device), train=False, return_all=True)
                rid = o.get("executed_route_id", o.get("selected_route_id"))
                out[str(eta)][i] = int(torch.as_tensor(rid).view(-1)[0].item())
        return out

    raise ValueError(which)


def same_hist_random(routes: np.ndarray, L: np.ndarray, n_perm=1000, seed=0):
    """routes: [N] route ids; L: [N,4] losses."""
    N = len(routes)
    actual_loss = L[np.arange(N), routes]
    actual_mae = float(actual_loss.mean())
    best = L.min(axis=1)
    actual_regret = float((actual_loss - best).mean())
    rng = np.random.default_rng(seed)
    maes = []
    regs = []
    base = routes.copy()
    for _ in range(n_perm):
        perm = rng.permutation(base)
        loss = L[np.arange(N), perm]
        maes.append(loss.mean())
        regs.append((loss - best).mean())
    maes = np.asarray(maes)
    regs = np.asarray(regs)
    # percentile rank of actual (lower MAE better)
    rank = float(np.mean(maes >= actual_mae))  # fraction of random worse or equal
    pval = float(np.mean(maes <= actual_mae))  # one-sided: random as good or better
    return {
        "actual_selected_MAE": actual_mae,
        "actual_regret": actual_regret,
        "random_hist_mean_MAE": float(maes.mean()),
        "random_hist_std_MAE": float(maes.std()),
        "random_hist_MAE_p5": float(np.quantile(maes, 0.05)),
        "random_hist_MAE_p50": float(np.quantile(maes, 0.50)),
        "random_hist_MAE_p95": float(np.quantile(maes, 0.95)),
        "actual_minus_random_mean_MAE": float(actual_mae - maes.mean()),
        "percentile_rank_actual_better_than_random": rank,
        "permutation_pvalue_actual_le_random": pval,
        "avg_cost_actual": float(COSTS[routes].mean()),
        "avg_cost_random_same": float(COSTS[base].mean()),
        "n_perm": n_perm,
        "hist": {ROUTES[k]: int(v) for k, v in zip(*np.unique(routes, return_counts=True))},
    }


def p0_features_from_X(X: np.ndarray) -> np.ndarray:
    """X: [N,P,Nnodes,C] -> [N,F] lightweight global summaries."""
    traffic = X[..., 0]  # [N,P,nodes]
    mean = traffic.mean(axis=(1, 2))
    last = traffic[:, -1, :].mean(axis=1)
    std = traffic.std(axis=1).mean(axis=1)
    absv = np.abs(np.diff(traffic, axis=1)).mean(axis=(1, 2))
    slope = (traffic[:, -1, :] - traffic[:, 0, :]).mean(axis=1)
    # spatial diversity
    node_mean = traffic.mean(axis=1)  # [N,nodes]
    spat_std = node_mean.std(axis=1)
    spat_max = node_mean.max(axis=1)
    spat_min = node_mean.min(axis=1)
    return np.stack([mean, last, std, absv, slope, spat_std, spat_max, spat_min], axis=1)


def chrono_folds(n: int):
    # three forward-chaining blocks
    b = n // 4
    splits = []
    # A: train 0:b -> test b:2b
    splits.append(("A", np.arange(0, b), np.arange(b, 2 * b)))
    # B: train 0:2b -> test 2b:3b
    splits.append(("B", np.arange(0, 2 * b), np.arange(2 * b, 3 * b)))
    # C: train 0:3b -> test 3b:n
    splits.append(("C", np.arange(0, 3 * b), np.arange(3 * b, n)))
    return splits


def eval_regression_probe(Xtr, ytr, Xte, yte):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xtr_s, ytr)
    pred = ridge.predict(Xte_s)
    pr = stats.pearsonr(pred, yte)
    sr = stats.spearmanr(pred, yte)
    # sign
    y_bin = (yte > 0).astype(int)
    if y_bin.min() == y_bin.max():
        auc = float("nan")
        ap = float("nan")
        sign_acc = float("nan")
    else:
        clf = LogisticRegression(max_iter=500)
        clf.fit(Xtr_s, (ytr > 0).astype(int))
        proba = clf.predict_proba(Xte_s)[:, 1]
        auc = float(roc_auc_score(y_bin, proba))
        ap = float(average_precision_score(y_bin, proba))
        sign_acc = float(np.mean((proba >= 0.5) == y_bin))
    # mlp
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=0)
    mlp.fit(Xtr_s, ytr)
    pred_m = mlp.predict(Xte_s)
    sr_m = stats.spearmanr(pred_m, yte)
    return {
        "ridge_pearson": float(pr.statistic),
        "ridge_spearman": float(sr.statistic),
        "ridge_sign_acc_via_logistic": sign_acc,
        "logistic_ROC_AUC": auc,
        "logistic_PR_AUC": ap,
        "mlp_spearman": float(sr_m.statistic),
        "mlp_pearson": float(stats.pearsonr(pred_m, yte).statistic),
        "pred_ridge": pred,
        "pred_mlp": pred_m,
    }


def decision_from_gains(pred_g3, pred_g6, eta, Lte):
    mdp = BudgetedRefinementMDP(12)
    feas = mdp.feasible_terminal_routes_for_eta(eta)
    # map to ids
    name_to_i = {"D": 0, "M": 1, "Q": 2, "F": 3}
    # approximate Q values: D=0, M=g6, Q=g3 (centered return style)
    N = len(pred_g3)
    sel = np.zeros(N, dtype=np.int64)
    for i in range(N):
        scores = {"D": 0.0}
        if "M" in feas:
            scores["M"] = float(pred_g6[i])
        if "Q" in feas:
            scores["Q"] = float(pred_g3[i])
        # F not chosen at pre-route without Z3 continuation model
        best = max((n for n in feas if n in scores), key=lambda n: scores[n])
        sel[i] = name_to_i[best]
    loss = Lte[np.arange(N), sel]
    best = np.array([min(Lte[i, name_to_i[n]] for n in feas) for i in range(N)])
    direct = Lte[:, 0]
    oracle = best
    mae = float(loss.mean())
    regret = float((loss - best).mean())
    denom = float((direct.mean() - oracle.mean()))
    recovery = float((direct.mean() - mae) / denom) if denom > 1e-9 else float("nan")
    return {
        "selected_MAE": mae,
        "strict_regret": regret,
        "avg_cost": float(COSTS[sel].mean()),
        "ORACLE_GAIN_RECOVERY_FRACTION": recovery,
        "MAE_direct": float(direct.mean()),
        "MAE_oracle": float(oracle.mean()),
        "hist": {ROUTES[k]: int(v) for k, v in zip(*np.unique(sel, return_counts=True))},
    }


def signal_label(auc, spearman):
    if (not np.isfinite(auc) or auc < 0.60) and (not np.isfinite(spearman) or abs(spearman) < 0.15):
        return "LOW"
    if (np.isfinite(auc) and auc >= 0.70) or (np.isfinite(spearman) and abs(spearman) >= 0.30):
        return "USEFUL"
    return "WEAK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-valid", type=int, default=None, help="optional cap for speed")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--skip-route-collect", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    Lmap = dedupe_L()
    sis = sorted(Lmap.keys())
    if args.n_valid is not None:
        sis = sis[: args.n_valid]
    L = np.stack([Lmap[i] for i in sis], 0)
    G3, G6, G36 = gains(L)
    N = len(sis)
    etas = [0.5, 0.75, 1.0]

    # Load or collect routes
    cache_path = Path("results/rootcause_valid_route_cache.npz")
    if cache_path.is_file() and args.skip_route_collect:
        z = np.load(cache_path, allow_pickle=True)
        routes = {k: z[k] for k in z.files if k.startswith("plana") or k.startswith("bv2") or k.startswith("bellman")}
        # reshape keys
        route_maps = {
            "PlanA": {e: z[f"plana_{e}"][:N] for e in map(str, etas)},
            "PlanB-v2": {e: z[f"bv2_{e}"][:N] for e in map(str, etas)},
            "Bellman": {e: z[f"bellman_{e}"][:N] for e in map(str, etas)},
        }
    else:
        print("[routes] collecting PlanA/Bv2/Bellman VALID routes (inference-only)...")
        route_maps = {}
        # Prefer lighter collection via histograms-matched from existing eval if full too slow:
        # Still need per-sample — run methods.
        for name, which in [("PlanA", "plana"), ("PlanB-v2", "bv2"), ("Bellman", "bellman")]:
            try:
                print(" collecting", name)
                route_maps[name] = collect_routes_generic(device, etas, which, n=N)
            except Exception as e:
                print("FAILED", name, e)
                route_maps[name] = None
        # save
        save = {}
        for name, key in [("PlanA", "plana"), ("PlanB-v2", "bv2"), ("Bellman", "bellman")]:
            if route_maps[name] is None:
                continue
            for e, arr in route_maps[name].items():
                save[f"{key}_{e}"] = arr
        if save:
            np.savez(cache_path, **save)

    # PART 6 same-hist
    same = {}
    for name, rm in route_maps.items():
        if rm is None:
            same[name] = {"available": False}
            continue
        same[name] = {"available": True, "etas": {}}
        for eta in etas:
            r = rm[str(eta)][:N]
            same[name]["etas"][str(eta)] = same_hist_random(r, L, n_perm=args.n_perm, seed=1)
    write_json("results/rootcause_same_hist_random_selector.json", same)
    print("[part6] done", time.time() - t0)

    # Features P0
    print("[features] building P0 from raw X...")
    ds = IndexedTimeSeriesForecastingDataset(DATA, INDEX, "valid")
    Xs = []
    for i in range(N):
        _, hist, _ = ds[i]
        Xs.append(hist.numpy())
    X_all = np.stack(Xs, 0)  # [N,P,nodes,C]
    Fp0 = p0_features_from_X(X_all)

    # P2 H_shared pooled + Z3 for subset speed: use all if possible
    print("[features] P2 extract_pre_route_context + Z3 (GPU)...")
    supernet, _ = load_supernet_strict(DEFAULT_SUPERNET, device)
    env = SequentialF2FEnvironment(supernet)
    H_feats = []
    Z_feats = []
    bs = 16
    for start in range(0, N, bs):
        batch = []
        for i in range(start, min(start + bs, N)):
            batch.append(torch.from_numpy(X_all[i]))
        xb = torch.stack(batch, 0).to(device)
        with torch.no_grad():
            H = supernet.extract_pre_route_context(xb, detach=True)  # [B,M,N,D]
            # pool
            hp = torch.stack(
                [H.mean(dim=(1, 2)), H[:, -1].mean(dim=1), H.std(dim=(1, 2))], dim=-1
            )  # rough
            # better: mean over M,N then flatten last dims partially
            hp = H.mean(dim=(1, 2))  # [B,D]
            z = env.execute_quarter_prefix(xb)["Z_q"]  # [B,q,N,1]
            z0 = z[..., 0]
            zf = torch.stack(
                [
                    z0.mean(dim=(1, 2)),
                    z0[:, -1].mean(dim=1),
                    z0.std(dim=(1, 2)),
                    (z0[:, -1] - z0[:, 0]).mean(dim=1),
                    (z0[:, 1:] - z0[:, :-1]).abs().mean(dim=(1, 2)),
                ],
                dim=-1,
            )
        H_feats.append(hp.cpu().numpy())
        Z_feats.append(zf.cpu().numpy())
    Fp2 = np.concatenate(H_feats, 0)
    Fz = np.concatenate(Z_feats, 0)
    Fx_z = np.concatenate([Fp0, Fz], axis=1)

    # Learnability P0/P2 for G3/G6
    preroute = {"feature_sets": {}, "protocol": "forward_chaining_VALID_only"}
    for fs_name, F in [("P0_raw_history", Fp0), ("P2_EXTRA_pre_route_context", Fp2)]:
        preroute["feature_sets"][fs_name] = {"G3": [], "G6": [], "mark": "EXTRA_COMPUTE_UPPER_BOUND" if "P2" in fs_name else "ZERO_OVERHEAD_RAW"}
        for target_name, y in [("G3", G3), ("G6", G6)]:
            fold_rows = []
            for split_name, tr_idx, te_idx in chrono_folds(N):
                res = eval_regression_probe(F[tr_idx], y[tr_idx], F[te_idx], y[te_idx])
                # decision metrics on last split mainly
                dec05 = decision_from_gains(res["pred_ridge"] if target_name == "G3" else np.zeros_like(res["pred_ridge"]),
                                           res["pred_ridge"] if target_name == "G6" else np.zeros_like(res["pred_ridge"]),
                                           0.5, L[te_idx])
                # For joint D/Q need both gains — compute after both; here store target metrics
                fold_rows.append(
                    {
                        "split": split_name,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "pearson": res["ridge_pearson"],
                        "spearman": res["ridge_spearman"],
                        "ROC_AUC": res["logistic_ROC_AUC"],
                        "PR_AUC": res["logistic_PR_AUC"],
                        "sign_accuracy": res["ridge_sign_acc_via_logistic"],
                        "mlp_spearman": res["mlp_spearman"],
                    }
                )
            # aggregate mean over splits
            def avg_key(k):
                vals = [r[k] for r in fold_rows if np.isfinite(r[k])]
                return float(np.mean(vals)) if vals else float("nan")

            preroute["feature_sets"][fs_name][target_name] = {
                "folds": fold_rows,
                "mean_pearson": avg_key("pearson"),
                "mean_spearman": avg_key("spearman"),
                "mean_ROC_AUC": avg_key("ROC_AUC"),
                "mean_sign_accuracy": avg_key("sign_accuracy"),
            }

    # Joint decision probe using both G3 and G6 predictions on chrono split C
    for fs_name, F in [("P0_raw_history", Fp0), ("P2_EXTRA_pre_route_context", Fp2)]:
        _, tr, te = chrono_folds(N)[-1]
        r3 = eval_regression_probe(F[tr], G3[tr], F[te], G3[te])
        r6 = eval_regression_probe(F[tr], G6[tr], F[te], G6[te])
        preroute["feature_sets"][fs_name]["decision"] = {
            "eta0.5": decision_from_gains(r3["pred_ridge"], r6["pred_ridge"], 0.5, L[te]),
            "eta0.75": decision_from_gains(r3["pred_ridge"], r6["pred_ridge"], 0.75, L[te]),
        }
    preroute["P1_zero_overhead_common_state"] = {
        "available": False,
        "reason": "NO_ZERO_OVERHEAD_COMMON_STATE from Part5",
    }
    write_json("results/rootcause_preroute_learnability_valid.json", preroute)

    # Post Z3 G36
    post = {"feature_sets": {}}
    for fs_name, F in [("Z0_raw_X", Fp0), ("Z1_Z3_only", Fz), ("Z2_X_plus_Z3", Fx_z)]:
        folds = []
        for split_name, tr, te in chrono_folds(N):
            res = eval_regression_probe(F[tr], G36[tr], F[te], G36[te])
            folds.append(
                {
                    "split": split_name,
                    "pearson": res["ridge_pearson"],
                    "spearman": res["ridge_spearman"],
                    "ROC_AUC": res["logistic_ROC_AUC"],
                    "sign_accuracy": res["ridge_sign_acc_via_logistic"],
                    "mlp_spearman": res["mlp_spearman"],
                }
            )
        def avg(k):
            vals = [r[k] for r in folds if np.isfinite(r[k])]
            return float(np.mean(vals)) if vals else float("nan")
        post["feature_sets"][fs_name] = {
            "folds": folds,
            "mean_pearson": avg("pearson"),
            "mean_spearman": avg("spearman"),
            "mean_ROC_AUC": avg("ROC_AUC"),
            "mean_sign_accuracy": avg("sign_accuracy"),
        }
    # deltas
    post["delta_AUC_XplusZ3_minus_X"] = float(
        post["feature_sets"]["Z2_X_plus_Z3"]["mean_ROC_AUC"]
        - post["feature_sets"]["Z0_raw_X"]["mean_ROC_AUC"]
    )
    post["delta_Spearman_XplusZ3_minus_X"] = float(
        post["feature_sets"]["Z2_X_plus_Z3"]["mean_spearman"]
        - post["feature_sets"]["Z0_raw_X"]["mean_spearman"]
    )
    # regret via logistic score as G36 proxy on split C
    _, tr, te = chrono_folds(N)[-1]
    r = eval_regression_probe(Fx_z[tr], G36[tr], Fx_z[te], G36[te])
    # decide Q vs F given quarter already taken: choose F if pred>0 else Q
    pred = r["pred_ridge"]
    sel = np.where(pred > 0, 3, 2)  # F else Q
    loss = L[te, sel]
    best = np.minimum(L[te, 2], L[te, 3])
    post["decision_Q_vs_F_splitC"] = {
        "MAE": float(loss.mean()),
        "regret": float((loss - best).mean()),
        "ORACLE_GAIN_RECOVERY_FRACTION": float(
            (L[te, 2].mean() - loss.mean()) / max(L[te, 2].mean() - best.mean(), 1e-9)
        ),
    }
    write_json("results/rootcause_postforecast_learnability_valid.json", post)

    # signals
    best_g3 = max(
        [
            ("P0", preroute["feature_sets"]["P0_raw_history"]["G3"]),
            ("P2", preroute["feature_sets"]["P2_EXTRA_pre_route_context"]["G3"]),
        ],
        key=lambda t: (t[1]["mean_ROC_AUC"] if np.isfinite(t[1]["mean_ROC_AUC"]) else -1),
    )
    best_g6 = max(
        [
            ("P0", preroute["feature_sets"]["P0_raw_history"]["G6"]),
            ("P2", preroute["feature_sets"]["P2_EXTRA_pre_route_context"]["G6"]),
        ],
        key=lambda t: (t[1]["mean_ROC_AUC"] if np.isfinite(t[1]["mean_ROC_AUC"]) else -1),
    )
    best_g36 = max(
        post["feature_sets"].items(),
        key=lambda t: (t[1]["mean_ROC_AUC"] if np.isfinite(t[1]["mean_ROC_AUC"]) else -1),
    )

    sig = {
        "PRE_ROUTE_G3_SIGNAL": signal_label(best_g3[1]["mean_ROC_AUC"], best_g3[1]["mean_spearman"]),
        "PRE_ROUTE_G6_SIGNAL": signal_label(best_g6[1]["mean_ROC_AUC"], best_g6[1]["mean_spearman"]),
        "POST_Z3_G36_SIGNAL": signal_label(best_g36[1]["mean_ROC_AUC"], best_g36[1]["mean_spearman"]),
        "best_G3": {"feature": best_g3[0], **best_g3[1]},
        "best_G6": {"feature": best_g6[0], **best_g6[1]},
        "best_G36": {"feature": best_g36[0], **best_g36[1]},
        "delta_AUC_Z3": post["delta_AUC_XplusZ3_minus_X"],
        "delta_Spearman_Z3": post["delta_Spearman_XplusZ3_minus_X"],
        "elapsed_sec": time.time() - t0,
    }
    write_json("results/rootcause_information_stage_signals.json", sig)
    print("[done heavy]", time.time() - t0, sig)


if __name__ == "__main__":
    main()
