#!/usr/bin/env python3
"""Plan A complete inference-only / offline diagnostics report.

NO training. Uses existing checkpoints + oracles only.
Never builds test oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_refinement_context import (
    PRE_ROUTE_OVERHEAD_NAME,
    pre_route_overhead_report,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    default_candidate_routes,
    load_route_costs,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_decision import (
    select_routes_from_scores,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_gain_loss import (
    gain_diagnostics,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    build_refinement_route_index_map,
    gains_from_route_losses,
    route_scores_from_gains,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    feasible_mask_from_budget,
    oracle_best_feasible_route,
)
from basicts.data import SCALER_REGISTRY
from basicts.data.forecast_refinement_gain_dataset import ForecastRefinementGainDataset
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from basicts.data.route_quality_dataset import dedupe_route_loss_records, load_oracle_json
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.utils import load_pkl
from scripts.eval_budget_conditioned_f2f_intensity import evaluate_loader, load_checkpoint_strict
from scripts.train_forecast_refinement_controller import _build_model, _load_supernet


ETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
ROUTE_NAMES = {
    0: "[12]",
    1: "[6,12]",
    2: "[3,12]",
    3: "[3,6,12]",
}


def sha1_file(path: Path, n: int = 16) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:n]


def pct(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q)) if len(x) else float("nan")


def hist_entropy(hist: dict[str, int]) -> float:
    n = sum(hist.values()) or 1
    ent = 0.0
    for c in hist.values():
        p = c / n
        ent -= p * math.log(p + 1e-12)
    return float(ent)


def gain_channel_stats(pred: torch.Tensor, true: torch.Tensor, name: str) -> dict[str, float]:
    p = pred.detach().float().cpu()
    t = true.detach().float().cpu()
    p0 = p - p.mean()
    t0 = t - t.mean()
    pear = float(((p0 * t0).sum() / (p0.norm() * t0.norm() + 1e-6)).item())
    rp = p.argsort().argsort().float()
    rt = t.argsort().argsort().float()
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    spear = float(((rp * rt).sum() / (rp.norm() * rt.norm() + 1e-6)).item())
    sign_acc = float(((p > 0) == (t > 0)).float().mean().item())
    return {
        "predicted_mean": float(p.mean()),
        "predicted_std": float(p.std(unbiased=False)),
        "true_mean": float(t.mean()),
        "true_std": float(t.std(unbiased=False)),
        "MAE": float((p - t).abs().mean()),
        "centered_MAE": float((p0 - t0).abs().mean()),
        "Pearson": pear,
        "Spearman": spear,
        "sign_accuracy": sign_acc,
        "predicted_positive_rate": float((p > 0).float().mean()),
        "true_positive_rate": float((t > 0).float().mean()),
        "calibration_bias": float(p.mean() - t.mean()),
        "name": name,
    }


def pairwise_ranking(scores_hat: torch.Tensor, scores_true: torch.Tensor, routes: list) -> dict:
    r = scores_hat.shape[-1]
    pairs = {}
    accs = []
    for i in range(r):
        for j in range(i):
            d_hat = scores_hat[:, i] - scores_hat[:, j]
            d_true = scores_true[:, i] - scores_true[:, j]
            mask = d_true.abs() >= 1e-8
            if not bool(mask.any()):
                continue
            ok = ((d_hat[mask] > 0) == (d_true[mask] > 0)).float().mean()
            key = f"{ROUTE_NAMES.get(j, j)} vs {ROUTE_NAMES.get(i, i)}"
            pairs[key] = {
                "n": int(mask.sum()),
                "preference_accuracy": float(ok.item()),
                "routes": [list(routes[j]), list(routes[i])],
            }
            accs.append(float(ok.item()))
    return {
        "pairs": pairs,
        "macro_pairwise_ranking_accuracy": float(sum(accs) / max(len(accs), 1)),
    }


def regret_stats(regrets: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(regrets.mean()) if len(regrets) else float("nan"),
        "median": float(np.median(regrets)) if len(regrets) else float("nan"),
        "p75": pct(regrets, 0.75),
        "p90": pct(regrets, 0.90),
        "p95": pct(regrets, 0.95),
        "max": float(regrets.max()) if len(regrets) else float("nan"),
        "frac_le_0.01": float((regrets <= 0.01).mean()) if len(regrets) else float("nan"),
        "frac_le_0.02": float((regrets <= 0.02).mean()) if len(regrets) else float("nan"),
        "frac_le_0.05": float((regrets <= 0.05).mean()) if len(regrets) else float("nan"),
        "frac_gt_0.10": float((regrets > 0.10).mean()) if len(regrets) else float("nan"),
    }


def oracle_gain_summary(oracle_path: Path, horizon: int = 12) -> dict[str, Any]:
    oracle = load_oracle_json(oracle_path)
    # Support holdout-style records with G3 already present, or route-loss oracles.
    recs = oracle.get("records") or []
    if recs and "G3" in recs[0]:
        g3 = np.array([float(r["G3"]) for r in recs], dtype=np.float64)
        g6 = np.array([float(r["G6"]) for r in recs], dtype=np.float64)
        g36 = np.array([float(r["G36"]) for r in recs], dtype=np.float64)
        routes = oracle["metadata"].get("candidate_routes")
    else:
        packed = dedupe_route_loss_records(oracle)
        routes = packed["candidate_routes"]
        index_map = build_refinement_route_index_map(routes, horizon)
        g3, g6, g36 = [], [], []
        for si in packed["sample_indices"]:
            losses = packed["route_losses"][si]
            by = {
                "direct": losses[index_map["direct"]],
                "half": losses[index_map["half"]],
                "quarter": losses[index_map["quarter"]],
                "progressive": losses[index_map["progressive"]],
            }
            g = gains_from_route_losses(by)
            g3.append(g["g3"])
            g6.append(g["g6"])
            g36.append(g["g36"])
        g3, g6, g36 = map(np.asarray, (g3, g6, g36))

    def one(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "pos_rate": float((arr > 0).mean()),
            "neg_rate": float((arr < 0).mean()),
            "P1": pct(arr, 0.01),
            "P5": pct(arr, 0.05),
            "P50": pct(arr, 0.50),
            "P95": pct(arr, 0.95),
            "P99": pct(arr, 0.99),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    return {
        "n": int(len(g3)),
        "routes": routes,
        "G3": one(g3),
        "G6": one(g6),
        "G36": one(g36),
        "extreme_G3_neg_count_lt_m1": int((g3 < -1.0).sum()),
        "extreme_G3_neg_count_lt_m2": int((g3 < -2.0).sum()),
    }


def locate_paths() -> dict[str, Any]:
    stable = Path(
        "checkpoints/PEMS04/H12/budget_f2f/"
        "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
        "b5678fda5e8d94ed028c6c8bb073461d/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
    )
    ctrl = Path(
        "checkpoints/PEMS04/H12/budget_f2f/crossfit_refinement_controller/"
        "refinement_controller_best_val_regret.pt"
    )
    hist = Path(
        "checkpoints/PEMS04/H12/budget_f2f/crossfit_refinement_controller/train_history.json"
    )
    cfg = Path(
        "checkpoints/PEMS04/H12/budget_f2f/"
        "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
        "b5678fda5e8d94ed028c6c8bb073461d/"
        "H12_supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6_seed1.py"
    )
    paths = {
        "stable_supernet_checkpoint": str(stable),
        "stable_supernet_sha1_16": sha1_file(stable),
        "crossfit_controller_checkpoint": str(ctrl),
        "controller_sha1_16": sha1_file(ctrl),
        "controller_train_history": str(hist),
        "valid_oracle": "results/pems04_budget_f2f_oracle_valid_rawscale.json",
        "crossfit_merged_oracle": "results/pems04_temporal_crossfit_refinement_oracle.json",
        "train_insample_oracle": "results/pems04_budget_f2f_oracle_train_rawscale.json",
        "eval_valid_routing_only": "results/pems04_crossfit_controller_eval_valid.json",
        "eval_test_routing_only": "results/pems04_crossfit_controller_eval_test.json",
        "crossfit_manifest": "results/temporal_crossfit_manifest.json",
        "stable_supernet_cfg": str(cfg),
        "fold_oracles": {
            f"fold{k}": f"results/pems04_cf_fold{k}_oracle.json" for k in range(1, 5)
        },
    }
    for k, v in list(paths.items()):
        if k in {"fold_oracles"}:
            continue
        if k.endswith("_sha1_16"):
            continue
        if not Path(v).exists():
            raise FileNotFoundError(f"missing {k}: {v}")
    for k, v in paths["fold_oracles"].items():
        if not Path(v).exists():
            raise FileNotFoundError(f"missing {k}: {v}")
    return paths


def verify_consistency(paths: dict[str, Any]) -> dict[str, Any]:
    ckpt = torch.load(paths["crossfit_controller_checkpoint"], map_location="cpu")
    cfg = ckpt.get("controller_config") or {}
    routes = cfg.get("candidate_routes")
    costs = cfg.get("route_costs")
    horizon = int(cfg.get("horizon", -1))
    expected_routes = default_candidate_routes(12)
    expected_costs = load_route_costs(None, expected_routes, 12)
    if routes != expected_routes:
        raise RuntimeError(f"candidate_routes mismatch: {routes} vs {expected_routes}")
    if horizon != 12:
        raise RuntimeError(f"horizon mismatch: {horizon}")
    if abs(np.asarray(costs) - np.asarray(expected_costs)).max() > 1e-5:
        raise RuntimeError(f"route_costs mismatch: {costs} vs {expected_costs}")
    supernet_path = Path(ckpt.get("supernet_checkpoint", ""))
    if supernet_path.as_posix() != Path(paths["stable_supernet_checkpoint"]).as_posix() and str(
        supernet_path
    ) != paths["stable_supernet_checkpoint"]:
        # allow relative vs absolute
        if Path(supernet_path).resolve() != Path(paths["stable_supernet_checkpoint"]).resolve():
            raise RuntimeError(
                f"supernet path mismatch: ckpt={supernet_path} "
                f"expected={paths['stable_supernet_checkpoint']}"
            )
    if cfg.get("architecture_version") != "forecast_refinement_gain_v1":
        raise RuntimeError(f"architecture_version mismatch: {cfg.get('architecture_version')}")
    # Valid / crossfit oracle route order
    for key in ("valid_oracle", "crossfit_merged_oracle"):
        md = load_oracle_json(paths[key])["metadata"]
        if md.get("candidate_routes") != expected_routes:
            raise RuntimeError(f"{key} candidate_routes mismatch")
        if "route_costs" in md and abs(np.asarray(md["route_costs"]) - np.asarray(expected_costs)).max() > 1e-5:
            raise RuntimeError(f"{key} route_costs mismatch")
        if int(md.get("horizon", 12)) != 12:
            raise RuntimeError(f"{key} horizon mismatch")
        if str(md.get("dataset", "PEMS04")) != "PEMS04":
            raise RuntimeError(f"{key} dataset mismatch")
    return {
        "dataset": "PEMS04",
        "horizon": 12,
        "candidate_routes": expected_routes,
        "route_costs": list(map(float, expected_costs)),
        "delta_abs": float(cfg.get("delta_abs", 0.05)),
        "architecture_version": cfg.get("architecture_version"),
        "controller_epoch_best": int(ckpt.get("epoch", -1)),
        "status": "PASS",
    }


def load_adaptive_model(paths: dict, device: torch.device, routes, costs, delta_abs: float):
    class NS:
        pass

    ns = NS()
    ns.cfg = None
    ns.horizon = 12
    ns.controller_dim = 128
    ns.pooling_queries = 4
    ns.delta_abs = float(delta_abs)
    ns.route_cost_file = None
    model = _build_model(ns, routes, device)
    # Load supernet backbone strictly for forecasting weights; planner keys may differ.
    raw = torch.load(paths["stable_supernet_checkpoint"], map_location="cpu")
    sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_m = [k for k in missing if not k.startswith("gain_controller.") and not str(k).endswith("num_batches_tracked")]
    bad_u = [k for k in unexpected if not str(k).endswith("num_batches_tracked")]
    # Expected: controller missing from supernet ckpt
    unexpected_material = [k for k in bad_u if not k.startswith("gain_controller.")]
    if unexpected_material:
        raise RuntimeError(f"supernet unexpected material keys: {unexpected_material[:20]}")
    ctrl = torch.load(paths["crossfit_controller_checkpoint"], map_location="cpu")
    model.gain_controller.load_state_dict(ctrl["controller_state_dict"], strict=True)
    model.freeze_backbone(True)
    model.set_training_phase("eval")
    model.route_granularity = "sample"
    model.route_selection_mode = "sample"
    model.delta_abs = float(delta_abs)
    model.eval()
    return model, {
        "supernet_missing_non_controller": [k for k in bad_m if not k.startswith("gain_controller.")][:20],
        "controller_loaded_strict": True,
    }


@torch.no_grad()
def collect_valid_predictions(model, valid_ds, device, batch_size=64):
    loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch], 0),
            torch.tensor([b[1] for b in batch], dtype=torch.long),
            torch.stack([b[2] for b in batch], 0),
            torch.stack([b[3] for b in batch], 0),
        ),
    )
    preds, trues, losses, histories = [], [], [], []
    for hist, _si, gains, loss_vec in loader:
        hist = hist.to(device)
        out = model.estimate_refinement_gains(hist)
        preds.append(out["predicted_gains"].detach().cpu())
        trues.append(gains.cpu())
        losses.append(loss_vec.cpu())
        histories.append(hist.detach().cpu())
    return (
        torch.cat(preds, 0),
        torch.cat(trues, 0),
        torch.cat(losses, 0),
        torch.cat(histories, 0),
    )


def analyze_eta_oracle(
    pred_gains: torch.Tensor,
    true_losses: torch.Tensor,
    routes,
    costs: torch.Tensor,
    eta: float,
    delta_abs: float,
    index_map,
    prior_scores: torch.Tensor | None = None,
) -> dict[str, Any]:
    scores = route_scores_from_gains(
        pred_gains[:, 0], pred_gains[:, 1], pred_gains[:, 2],
        index_map=index_map, n_routes=len(routes),
    )
    dec = select_routes_from_scores(scores, costs, eta, delta_abs=delta_abs)
    feas = dec["feasible_mask"]
    sel = dec["selected_route_id"]
    sel_loss = true_losses.gather(1, sel.unsqueeze(1)).squeeze(1)
    # strict oracle
    inf = true_losses.new_tensor(float("inf"))
    masked = torch.where(feas, true_losses, inf)
    best_loss, best_id = masked.min(dim=-1)
    strict_regret = sel_loss - best_loss
    # tolerance oracle: cheapest among near-best
    near = feas & (true_losses <= (best_loss.unsqueeze(-1) + delta_abs + 1e-8))
    big = true_losses.new_tensor(1e9)
    cost_mat = costs.reshape(1, -1).expand_as(true_losses)
    tol_score = torch.where(near, cost_mat, big)
    tol_id = tol_score.argmin(dim=-1)
    tol_loss = true_losses.gather(1, tol_id.unsqueeze(1)).squeeze(1)
    tol_cost = costs.gather(0, tol_id)
    sel_cost = costs.gather(0, sel)

    hist = Counter(int(x) for x in sel.tolist())
    tol_hist = Counter(int(x) for x in tol_id.tolist())

    out = {
        "eta": float(eta),
        "controller": {
            "per_sample_mae_mean": float(sel_loss.mean()),
            "avg_cost": float(sel_cost.mean()),
            "avg_stages": float(np.mean([len(routes[i]) for i in sel.tolist()])),
            "histogram": {ROUTE_NAMES[k]: v for k, v in sorted(hist.items())},
            "entropy": hist_entropy({str(k): v for k, v in hist.items()}),
            "strict_regret": regret_stats(strict_regret.numpy()),
            "mae_minus_tolerance_oracle": float((sel_loss - tol_loss).mean()),
            "cost_minus_tolerance_oracle": float((sel_cost - tol_cost).mean()),
        },
        "strict_oracle": {
            "per_sample_mae_mean": float(best_loss.mean()),
            "avg_cost": float(costs.gather(0, best_id).mean()),
            "histogram": {
                ROUTE_NAMES[k]: v
                for k, v in sorted(Counter(int(x) for x in best_id.tolist()).items())
            },
        },
        "tolerance_oracle": {
            "delta_abs": float(delta_abs),
            "per_sample_mae_mean": float(tol_loss.mean()),
            "avg_cost": float(tol_cost.mean()),
            "histogram": {ROUTE_NAMES[k]: v for k, v in sorted(tol_hist.items())},
        },
    }

    # baselines
    baselines = {}
    # cheapest / most expensive / global-best mean-loss among feasible
    mean_loss = true_losses.mean(dim=0)  # not used for selection of D; D uses train prior means externally
    for name, mode in [
        ("cheapest_feasible", "cheap"),
        ("most_expensive_feasible", "expensive"),
    ]:
        if mode == "cheap":
            score = torch.where(feas, cost_mat, big)
            bid = score.argmin(-1)
        else:
            score = torch.where(feas, -cost_mat, big)
            bid = score.argmin(-1)
        bloss = true_losses.gather(1, bid.unsqueeze(1)).squeeze(1)
        breg = bloss - best_loss
        bhist = Counter(int(x) for x in bid.tolist())
        baselines[name] = {
            "histogram": {ROUTE_NAMES[k]: v for k, v in sorted(bhist.items())},
            "avg_cost": float(costs.gather(0, bid).mean()),
            "mean_regret": float(breg.mean()),
            "median_regret": float(breg.median()),
            "p90_regret": float(torch.quantile(breg, 0.9)),
            "per_sample_mae_mean": float(bloss.mean()),
        }

    if prior_scores is not None:
        pdec = select_routes_from_scores(prior_scores, costs, eta, delta_abs=delta_abs)
        pid = pdec["selected_route_id"]
        ploss = true_losses.gather(1, pid.unsqueeze(1)).squeeze(1)
        preg = ploss - best_loss
        phist = Counter(int(x) for x in pid.tolist())
        baselines["train_crossfit_route_prior"] = {
            "histogram": {ROUTE_NAMES[k]: v for k, v in sorted(phist.items())},
            "avg_cost": float(costs.gather(0, pid).mean()),
            "mean_regret": float(preg.mean()),
            "median_regret": float(preg.median()),
            "p90_regret": float(torch.quantile(preg, 0.9)),
            "per_sample_mae_mean": float(ploss.mean()),
        }
        # also global-best by training mean route loss among feasible
        # prior_scores higher=better; for mean-loss prior we pass negative losses as scores externally
    out["baselines"] = baselines

    # compare adaptive vs prior
    if "train_crossfit_route_prior" in baselines:
        prior_r = baselines["train_crossfit_route_prior"]["mean_regret"]
        ctrl_r = out["controller"]["strict_regret"]["mean"]
        imp = prior_r - ctrl_r
        rel = (imp / max(abs(prior_r), 1e-8)) * 100.0
        out["vs_route_prior"] = {
            "regret_improvement": float(imp),
            "relative_regret_improvement_pct": float(rel),
            "controller_mean_regret": float(ctrl_r),
            "prior_mean_regret": float(prior_r),
        }
    return out


def feature_dependence_audit(model, history: torch.Tensor, true_losses: torch.Tensor,
                             costs, index_map, routes, delta_abs, device, eta=0.75):
    history = history.to(device)
    n = history.shape[0]

    def run(h):
        g = model.estimate_refinement_gains(h)["predicted_gains"]
        scores = route_scores_from_gains(
            g[:, 0], g[:, 1], g[:, 2], index_map=index_map, n_routes=len(routes)
        )
        dec = select_routes_from_scores(scores, costs.to(device), eta, delta_abs=delta_abs)
        sel = dec["selected_route_id"]
        feas = dec["feasible_mask"]
        sel_loss = true_losses.to(device).gather(1, sel.unsqueeze(1)).squeeze(1)
        inf = true_losses.new_tensor(float("inf")).to(device)
        best = torch.where(feas, true_losses.to(device), inf).min(dim=-1).values
        regret = sel_loss - best
        hist = Counter(int(x) for x in sel.tolist())
        return g.detach().cpu(), {
            "gain_pred_mean": g.mean(0).detach().cpu().tolist(),
            "mean_regret": float(regret.mean()),
            "histogram": {ROUTE_NAMES[k]: v for k, v in sorted(hist.items())},
            "entropy": hist_entropy({str(k): v for k, v in hist.items()}),
            "avg_cost": float(costs.to(device).gather(0, sel).mean()),
        }

    g_n, s_n = run(history)
    perm = torch.randperm(n, device=device)
    g_p, s_p = run(history[perm])
    g_z, s_z = run(torch.zeros_like(history))
    g_r, s_r = run(torch.flip(history, dims=[1]))
    # node permute
    node_perm = torch.randperm(history.shape[2], device=device)
    g_np, s_np = run(history[:, :, node_perm, :])

    def corr(a, b):
        out = {}
        for i, name in enumerate(["g3", "g6", "g36"]):
            x, y = a[:, i], b[:, i]
            x0, y0 = x - x.mean(), y - y.mean()
            out[f"pearson_{name}_vs_true"] = float(
                ((x0 * y0).sum() / (x0.norm() * y0.norm() + 1e-6)).item()
            ) if False else None
        return out

    # correlations vs true gains not available here for variants easily — compute vs normal gains magnitude
    def pearson_to_true(g_hat, true_g):
        o = {}
        for i, name in enumerate(["g3", "g6", "g36"]):
            x, y = g_hat[:, i], true_g[:, i]
            x0, y0 = x - x.mean(), y - y.mean()
            o[f"Pearson_{name}"] = float(((x0 * y0).sum() / (x0.norm() * y0.norm() + 1e-6)).item())
            rx = x.argsort().argsort().float(); ry = y.argsort().argsort().float()
            rx, ry = rx - rx.mean(), ry - ry.mean()
            o[f"Spearman_{name}"] = float(((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-6)).item())
        return o

    # true gains from losses
    # L12 - L312 etc using index map
    L = true_losses.cpu()
    true_g = torch.stack([
        L[:, 0] - L[:, 2],
        L[:, 0] - L[:, 1],
        L[:, 2] - L[:, 3],
    ], dim=-1)

    variants = {
        "normal": (g_n, s_n),
        "sample_shuffled": (g_p, s_p),
        "zero": (g_z, s_z),
        "temporal_reversed": (g_r, s_r),
        "node_permuted": (g_np, s_np),
    }
    report = {}
    for name, (g, s) in variants.items():
        report[name] = {
            **s,
            **pearson_to_true(g, true_g),
            "mean_abs_gain_vs_normal": float((g - g_n).abs().mean()) if name != "normal" else 0.0,
        }
    # verdict signal
    if (
        report["normal"]["mean_regret"] + 1e-4
        >= min(report["sample_shuffled"]["mean_regret"], report["zero"]["mean_regret"])
    ):
        report["verdict_flag"] = "SAMPLE SIGNAL FAILURE"
    else:
        report["verdict_flag"] = "SAMPLE SIGNAL PRESENT"
    return report


def basicts_eval_adaptive(model, loader, device, scaler, null_val, forward_features, target_features, routes, eta):
    model.inference_intensity = float(eta)
    model.set_forced_route(None)
    model.route_selection_mode = "sample"
    model.route_granularity = "sample"
    return evaluate_loader(
        model,
        loader,
        device=device,
        forward_features=forward_features,
        target_features=target_features,
        scaler=scaler,
        null_val=null_val,
        candidates=routes,
    )


def basicts_eval_forced(model, loader, device, scaler, null_val, forward_features, target_features, routes, route):
    model.set_forced_route(list(route))
    model.route_selection_mode = "forced"
    out = evaluate_loader(
        model,
        loader,
        device=device,
        forward_features=forward_features,
        target_features=target_features,
        scaler=scaler,
        null_val=null_val,
        candidates=routes,
    )
    model.set_forced_route(None)
    return out


def build_loader_split(split: str, batch_size: int, data_dir="datasets/PEMS04", horizon=12):
    # TimeSeriesForecastingDataset returns (future, history) — matches evaluate_loader
    from basicts.data import TimeSeriesForecastingDataset

    mode = "valid" if split in {"val", "valid"} else split
    ds = TimeSeriesForecastingDataset(
        data_file_path=f"{data_dir}/data_in12_out{horizon}.pkl",
        index_file_path=f"{data_dir}/index_in12_out{horizon}.pkl",
        mode=mode,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2), len(ds)


def latency_profile(model, loader, device, routes, warmup=20, measure=50):
    model.eval()
    model.set_forced_route(None)
    model.route_selection_mode = "sample"
    model.route_granularity = "sample"
    model.inference_intensity = 1.0

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    batches = []
    for batch in loader:
        batches.append(batch[1].to(device) if len(batch) == 2 else batch[1].to(device))
        # TimeSeries: future, history
        if len(batches) >= warmup + measure:
            break
    if len(batches) < warmup + measure:
        # reuse
        while len(batches) < warmup + measure:
            batches.append(batches[len(batches) % max(len(batches), 1)])

    # warmup
    with torch.no_grad():
        for i in range(warmup):
            h = batches[i]
            _ = model(history_data=h, train=False, return_all=True)
            sync()

    def timed(fn, idxs):
        xs = []
        with torch.no_grad():
            for i in idxs:
                h = batches[i]
                sync()
                t0 = time.perf_counter()
                fn(h)
                sync()
                xs.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(xs, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "p90": float(np.quantile(arr, 0.9)),
            "n": int(len(arr)),
        }

    idxs = list(range(warmup, warmup + measure))

    def feat(h):
        return model.extract_pre_route_context(h, detach=True)

    def ctrl(h):
        hs = model.extract_pre_route_context(h, detach=True)
        return model.gain_controller(hs)

    def total(h):
        return model(history_data=h, train=False, return_all=True)

    def route_only(h):
        # forced [12] as baseline route execution without controller
        model.set_forced_route([12])
        model.route_selection_mode = "forced"
        out = model(history_data=h, train=False, return_all=True)
        model.set_forced_route(None)
        model.route_selection_mode = "sample"
        return out

    report = {
        "pre_route_feature_extraction_ms": timed(feat, idxs),
        "controller_plus_feature_ms": timed(ctrl, idxs),
        "total_adaptive_forward_ms": timed(total, idxs),
        "forced_direct_route_ms": timed(route_only, idxs),
    }
    # isolate controller-only approx
    report["controller_only_ms_approx"] = {
        k: float(report["controller_plus_feature_ms"][k] - report["pre_route_feature_extraction_ms"][k])
        if k in {"mean", "median", "p90"}
        else report["controller_plus_feature_ms"][k]
        for k in report["controller_plus_feature_ms"]
    }
    # fixed routes
    fixed = {}
    for route in routes:
        def make_fn(r):
            def fn(h):
                model.set_forced_route(list(r))
                model.route_selection_mode = "forced"
                out = model(history_data=h, train=False, return_all=True)
                model.set_forced_route(None)
                model.route_selection_mode = "sample"
                return out
            return fn
        fixed[str(route)] = timed(make_fn(route), idxs)
    report["fixed_routes_ms"] = fixed
    report["warmup"] = warmup
    report["measure_batches"] = measure
    report["feature_reused"] = False
    report["feature_overhead_name"] = PRE_ROUTE_OVERHEAD_NAME
    report["pre_route_overhead"] = pre_route_overhead_report()
    return report


def static_duplicate_trace() -> dict[str, Any]:
    from basicts.archs.arch_zoo.ChainForecasting_arch import budget_conditioned_adaptive_f2f as bf
    from basicts.archs.arch_zoo.ChainForecasting_arch import adaptive_forecast_refinement_route as ar

    extract_src = inspect.getsource(bf.BudgetConditionedAdaptiveF2FNet.extract_pre_route_context)
    exec_src = inspect.getsource(bf.BudgetConditionedAdaptiveF2FNet._execute_route)
    fwd_src = inspect.getsource(ar.AdaptiveForecastRefinementRouteNet.forward)
    return {
        "files": {
            "extract_pre_route_context": "basicts/archs/arch_zoo/ChainForecasting_arch/budget_conditioned_adaptive_f2f.py",
            "execute_route": "basicts/archs/arch_zoo/ChainForecasting_arch/budget_conditioned_adaptive_f2f.py",
            "adaptive_forward": "basicts/archs/arch_zoo/ChainForecasting_arch/adaptive_forecast_refinement_route.py",
        },
        "pre_route_calls": [
            "backbone.temporal_steps[H].patch_encoder embed path (_embed_serial_concat / data_encoder)",
            "backbone._spatial_codebook()",
        ],
        "execute_route_calls": [
            "KASATemporalStep.forward for each stage resolution (includes patch_encoder again)",
            "progressive spatial refine",
            "forecast_state_adapter at supernet idx==1",
        ],
        "cache_passed_into_executor": False,
        "feature_reused": False,
        "conclusion": "PRE-ROUTE FEATURE EXTRACTION IS ADDITIONAL OVERHEAD",
        "notes": (
            "extract_pre_route_context taps H-stage patch_encoder embedding only; "
            "_execute_route re-runs full KASATemporalStep per stage. No tensor cache is "
            "passed from pre-route context into _execute_route."
        ),
        "extract_mentions_patch_encoder": "patch_encoder" in extract_src,
        "execute_mentions_temporal_steps": "temporal_steps" in exec_src,
        "adaptive_forward_mentions_parent": "BudgetConditionedAdaptiveF2FNet.forward" in fwd_src,
    }


def pareto_analyze(points: list[dict]) -> list[dict]:
    # points: {eta, cost, mae, label}
    out = []
    for i, a in enumerate(points):
        dominated_by = []
        for j, b in enumerate(points):
            if i == j:
                continue
            if (b["cost"] <= a["cost"] and b["mae"] <= a["mae"]) and (
                b["cost"] < a["cost"] or b["mae"] < a["mae"]
            ):
                dominated_by.append(b["label"])
        out.append({
            **a,
            "pareto": len(dominated_by) == 0,
            "dominated_by": dominated_by,
            "flag": None if not dominated_by else "DOMINATED OPERATING POINT",
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--skip-basicts", action="store_true")
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--latency-warmup", type=int, default=20)
    p.add_argument("--latency-measure", type=int, default=50)
    args = p.parse_args()

    t_start = time.time()
    reused = []
    newly = []

    print("=== 0. Locate paths ===")
    paths = locate_paths()
    print(json.dumps(paths, indent=2))
    meta = verify_consistency(paths)
    print("[consistency]", meta["status"], meta)
    reused.extend([
        paths["stable_supernet_checkpoint"],
        paths["crossfit_controller_checkpoint"],
        paths["controller_train_history"],
        paths["valid_oracle"],
        paths["crossfit_merged_oracle"],
        paths["eval_valid_routing_only"],
        paths["eval_test_routing_only"],
    ])

    routes = meta["candidate_routes"]
    costs = torch.tensor(meta["route_costs"], dtype=torch.float32)
    delta_abs = float(meta["delta_abs"])
    index_map = build_refinement_route_index_map(routes, 12)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    # ---- checkpoint selection audit (offline) ----
    hist = json.loads(Path(paths["controller_train_history"]).read_text())
    best = hist["best"]
    best_ep = int(best["epoch"])
    best_row = next(h for h in hist["history"] if int(h["epoch"]) == best_ep)
    # confirm selection by scanning history
    min_regret = min(h["valid"]["mean_validation_route_regret"] for h in hist["history"])
    selection_ok = abs(float(best["mean_validation_route_regret"]) - float(min_regret)) <= 1e-8
    ckpt_audit = {
        "best_checkpoint_epoch": best_ep,
        "best_validation_mean_regret": float(best["mean_validation_route_regret"]),
        "best_validation_mean_cost": float(best["mean_selected_cost"]),
        "validation_diagnostics_at_best": best_row["valid"]["diagnostics"],
        "route_prior_regrets_at_best": {
            eta: best_row["valid"]["per_eta"][eta]["route_prior_mean_regret"]
            for eta in best_row["valid"]["per_eta"]
        },
        "feature_dependence_at_best": best_row["valid"].get("feature_dependence"),
        "selection_rule": "min validation mean route regret; tie-break lower cost",
        "CHECKPOINT_SELECTION_MISMATCH": (not selection_ok),
    }
    reused.append(paths["controller_train_history"])

    # ---- crossfit summaries (offline) ----
    crossfit_stats = {
        "folds": {},
        "merged": oracle_gain_summary(Path(paths["crossfit_merged_oracle"])),
        "official_valid": oracle_gain_summary(Path(paths["valid_oracle"])),
    }
    for k, path in paths["fold_oracles"].items():
        crossfit_stats["folds"][k] = oracle_gain_summary(Path(path))
        reused.append(path)
    newly.append("crossfit/valid offline gain summaries")

    # ---- load model + VALID gain predictions ----
    print("=== load adaptive model ===")
    model, load_rep = load_adaptive_model(paths, device, routes, costs, delta_abs)
    print(load_rep)

    valid_ds = ForecastRefinementGainDataset(
        IndexedTimeSeriesForecastingDataset(
            "datasets/PEMS04/data_in12_out12.pkl",
            "datasets/PEMS04/index_in12_out12.pkl",
            "valid",
        ),
        paths["valid_oracle"],
        expected_routes=routes,
        expected_costs=meta["route_costs"],
        expected_horizon=12,
        expected_dataset="PEMS04",
        require_len_match=True,
    )
    print("=== VALID controller predictions ===")
    pred_g, true_g, true_L, hist_x = collect_valid_predictions(
        model, valid_ds, device, batch_size=args.batch_size
    )
    newly.append("VALID controller gain inference")

    # 1. gain diagnostics
    gain_diag = {
        "G3": gain_channel_stats(pred_g[:, 0], true_g[:, 0], "G3"),
        "G6": gain_channel_stats(pred_g[:, 1], true_g[:, 1], "G6"),
        "G36": gain_channel_stats(pred_g[:, 2], true_g[:, 2], "G36"),
        "composite": gain_diagnostics(pred_g, true_g),
    }
    scores_hat = route_scores_from_gains(
        pred_g[:, 0], pred_g[:, 1], pred_g[:, 2], index_map=index_map, n_routes=4
    )
    scores_true = route_scores_from_gains(
        true_g[:, 0], true_g[:, 1], true_g[:, 2], index_map=index_map, n_routes=4
    )
    ranking = pairwise_ranking(scores_hat, scores_true, routes)

    # prior from CROSSFIT only (no VALID leakage)
    cf_ds = ForecastRefinementGainDataset(
        IndexedTimeSeriesForecastingDataset(
            "datasets/PEMS04/data_in12_out12.pkl",
            "datasets/PEMS04/index_in12_out12.pkl",
            "train",
        ),
        paths["crossfit_merged_oracle"],
        expected_routes=routes,
        expected_costs=meta["route_costs"],
        expected_horizon=12,
        expected_dataset="PEMS04",
        require_len_match=False,
    )
    all_cf_gains = torch.stack(
        [torch.tensor(cf_ds.gains[i]) for i in cf_ds.sample_indices], dim=0
    )
    mean_cf = all_cf_gains.mean(0)
    # also mean route losses for D
    all_cf_losses = torch.stack(
        [torch.tensor(cf_ds.route_losses[i]) for i in cf_ds.sample_indices], dim=0
    )
    mean_cf_loss = all_cf_losses.mean(0)
    prior_scores = route_scores_from_gains(
        mean_cf[0].expand(pred_g.shape[0]),
        mean_cf[1].expand(pred_g.shape[0]),
        mean_cf[2].expand(pred_g.shape[0]),
        index_map=index_map,
        n_routes=4,
    )
    # baseline D: scores = -mean train route loss (constant)
    global_best_scores = (-mean_cf_loss).unsqueeze(0).expand(pred_g.shape[0], -1)

    print("=== VALID oracle regret / baselines ===")
    per_eta = {}
    beat_prior = []
    for eta in ETAS:
        row = analyze_eta_oracle(
            pred_g, true_L, routes, costs, eta, delta_abs, index_map, prior_scores=prior_scores
        )
        # add global-best-by-mean-loss baseline
        gdec = select_routes_from_scores(global_best_scores, costs, eta, delta_abs=delta_abs)
        gid = gdec["selected_route_id"]
        feas = gdec["feasible_mask"]
        inf = true_L.new_tensor(float("inf"))
        best = torch.where(feas, true_L, inf).min(-1).values
        gloss = true_L.gather(1, gid.unsqueeze(1)).squeeze(1)
        greg = gloss - best
        ghist = Counter(int(x) for x in gid.tolist())
        row["baselines"]["global_best_feasible_by_crossfit_mean_loss"] = {
            "histogram": {ROUTE_NAMES[k]: v for k, v in sorted(ghist.items())},
            "avg_cost": float(costs.gather(0, gid).mean()),
            "mean_regret": float(greg.mean()),
            "median_regret": float(greg.median()),
            "p90_regret": float(torch.quantile(greg, 0.9)),
            "per_sample_mae_mean": float(gloss.mean()),
        }
        per_eta[str(eta)] = row
        if "vs_route_prior" in row:
            beat_prior.append(row["vs_route_prior"]["regret_improvement"] > 1e-6)
    if any(beat_prior[2:]):  # focus on etas with nontrivial feasible sets
        adaptivity_flag = "SAMPLE-ADAPTIVE ROUTING SUPPORTED"
    else:
        adaptivity_flag = "CROSS-FIT CONTROLLER DOES NOT BEAT ROUTE PRIOR"

    print("=== feature dependence ===")
    # use all valid if feasible
    feat_audit = feature_dependence_audit(
        model, hist_x, true_L, costs, index_map, routes, delta_abs, device, eta=0.75
    )
    newly.append("VALID feature dependence audit")

    diagnostics = {
        "paths": paths,
        "consistency": meta,
        "gain_prediction": gain_diag,
        "pairwise_ranking": ranking,
        "per_eta_oracle_regret": per_eta,
        "adaptivity_flag": adaptivity_flag,
        "feature_dependence": feat_audit,
        "checkpoint_selection_audit": ckpt_audit,
        "crossfit_oracle_summary": crossfit_stats,
        "note": "oracle MAE here is per-sample raw route-loss average; not BasicTS headline MAE",
    }
    diag_path = Path("results/pems04_crossfit_controller_diagnostics_valid.json")
    diag_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    newly.append(str(diag_path))
    print("Wrote", diag_path)

    # ---- BasicTS metrics ----
    basicts_valid = {}
    basicts_test = {}
    fixed_baselines = {"valid": {}, "test": {}}
    if not args.skip_basicts:
        print("=== BasicTS adaptive + fixed (VALID/TEST) ===")
        scaler = load_pkl("datasets/PEMS04/scaler_in12_out12.pkl")
        # match cfg features from generated config
        from easytorch.config import import_config

        cfg = import_config(
            str(Path(paths["stable_supernet_cfg"]).resolve().relative_to(ROOT.resolve())).replace("\\", "/")
        )
        forward_features = list(cfg.MODEL.FORWARD_FEATURES)
        target_features = list(cfg.MODEL.TARGET_FEATURES)
        null_val = float(getattr(cfg.TRAIN, "NULL_VAL", 0.0))

        for split in ("valid", "test"):
            loader, n = build_loader_split(split, args.batch_size)
            print(f"[{split}] n={n}")
            # fixed routes
            for route in routes:
                print(f"  fixed {route}")
                row = basicts_eval_forced(
                    model, loader, device, scaler, null_val, forward_features, target_features, routes, route
                )
                fixed_baselines[split][str(route)] = row
            # adaptive etas
            split_out = {}
            for eta in ETAS:
                print(f"  adaptive eta={eta}")
                row = basicts_eval_adaptive(
                    model, loader, device, scaler, null_val, forward_features, target_features, routes, eta
                )
                split_out[str(eta)] = row
            if split == "valid":
                basicts_valid = split_out
            else:
                basicts_test = split_out

        Path("results/pems04_crossfit_controller_basicts_valid.json").write_text(
            json.dumps({"split": "valid", "etas": basicts_valid, "metric_path": "basicts.metrics.masked_* + scaler inverse"}, indent=2) + "\n"
        )
        Path("results/pems04_crossfit_controller_basicts_test.json").write_text(
            json.dumps({"split": "test", "etas": basicts_test, "metric_path": "basicts.metrics.masked_* + scaler inverse"}, indent=2) + "\n"
        )
        Path("results/pems04_crossfit_fixed_route_baselines.json").write_text(
            json.dumps(fixed_baselines, indent=2) + "\n"
        )
        newly.extend([
            "results/pems04_crossfit_controller_basicts_valid.json",
            "results/pems04_crossfit_controller_basicts_test.json",
            "results/pems04_crossfit_fixed_route_baselines.json",
        ])
    else:
        print("skip BasicTS")

    # ---- accuracy-cost table + pareto ----
    def table_rows(split_name, adaptive, fixed):
        rows = []
        for route in routes:
            key = str(route)
            r = fixed[key]
            rows.append({
                "METHOD": f"Fixed {key}",
                "ETA": None,
                "MAE": r["mae"],
                "RMSE": r["rmse"],
                "MAPE": r["mape"],
                "AVG_COST": r["average_selected_cost"],
                "AVG_STAGES": r["average_stage_count"],
                "split": split_name,
            })
        for eta in ETAS:
            r = adaptive[str(eta)]
            rows.append({
                "METHOD": "Adaptive crossfit controller",
                "ETA": float(eta),
                "MAE": r["mae"],
                "RMSE": r["rmse"],
                "MAPE": r["mape"],
                "AVG_COST": r["average_selected_cost"],
                "AVG_STAGES": r["average_stage_count"],
                "histogram": r.get("route_histogram_sample"),
                "split": split_name,
            })
        return rows

    summary = {"valid": [], "test": []}
    if basicts_valid and basicts_test:
        summary["valid"] = table_rows("valid", basicts_valid, fixed_baselines["valid"])
        summary["test"] = table_rows("test", basicts_test, fixed_baselines["test"])

        def pareto_for(adaptive):
            pts = [
                {
                    "label": f"eta={eta}",
                    "eta": float(eta),
                    "cost": float(adaptive[str(eta)]["average_selected_cost"]),
                    "mae": float(adaptive[str(eta)]["mae"]),
                }
                for eta in ETAS
            ]
            return pareto_analyze(pts)

        summary["pareto_valid"] = pareto_for(basicts_valid)
        summary["pareto_test"] = pareto_for(basicts_test)

        def incremental(adaptive):
            out = []
            for a, b in zip(ETAS[:-1], ETAS[1:]):
                ra, rb = adaptive[str(a)], adaptive[str(b)]
                d_mae = rb["mae"] - ra["mae"]
                d_cost = rb["average_selected_cost"] - ra["average_selected_cost"]
                out.append({
                    "from_eta": a,
                    "to_eta": b,
                    "delta_MAE": d_mae,
                    "delta_cost": d_cost,
                    "MAE_improvement_per_0p1_cost": (
                        (-d_mae) / (d_cost / 0.1) if abs(d_cost) > 1e-12 else None
                    ),
                })
            return out

        summary["incremental_valid"] = incremental(basicts_valid)
        summary["incremental_test"] = incremental(basicts_test)

        # key comparisons
        def key_cmp(split, adaptive, fixed):
            out = {}
            # eta 0.5
            a = adaptive["0.5"]
            out["eta_0.5"] = {
                "adaptive": {"MAE": a["mae"], "cost": a["average_selected_cost"], "stages": a["average_stage_count"]},
                "fixed_[12]": {"MAE": fixed["[12]"]["mae"], "cost": fixed["[12]"]["average_selected_cost"]},
                "fixed_[3,12]": {"MAE": fixed["[3, 12]"]["mae"], "cost": fixed["[3, 12]"]["average_selected_cost"]},
            }
            a = adaptive["0.75"]
            out["eta_0.75"] = {
                "adaptive": {"MAE": a["mae"], "cost": a["average_selected_cost"], "stages": a["average_stage_count"]},
                "fixed_[3,12]": {"MAE": fixed["[3, 12]"]["mae"], "cost": fixed["[3, 12]"]["average_selected_cost"]},
                "fixed_[6,12]": {"MAE": fixed["[6, 12]"]["mae"], "cost": fixed["[6, 12]"]["average_selected_cost"]},
            }
            a = adaptive["1.0"]
            f = fixed["[3, 6, 12]"]
            out["eta_1.0_vs_fixed_full"] = {
                "adaptive_MAE": a["mae"],
                "fixed_full_MAE": f["mae"],
                "MAE_difference": a["mae"] - f["mae"],
                "RMSE_difference": a["rmse"] - f["rmse"],
                "MAPE_difference": a["mape"] - f["mape"],
                "adaptive_cost": a["average_selected_cost"],
                "fixed_full_cost": f["average_selected_cost"],
                "cost_reduction_pct": 100.0 * (1.0 - a["average_selected_cost"] / max(f["average_selected_cost"], 1e-8)),
                "adaptive_stages": a["average_stage_count"],
                "fixed_full_stages": f["average_stage_count"],
                "stage_reduction_pct": 100.0 * (1.0 - a["average_stage_count"] / max(f["average_stage_count"], 1e-8)),
                "adaptive_histogram": a.get("route_histogram_sample"),
                "note": "cost is normalized_static_cost proxy, NOT measured wall-clock speedup",
            }
            return out

        # fix keys - evaluate_loader hist uses "12" style via _route_key
        def remap_fixed(fd):
            # keys like "12", "6,12", "3,12", "3,6,12" from _hist but evaluate stores under str(route)= "[12]" wait
            # we used str(route) which is "[12]" with spaces? str([12]) = '[12]'; str([6,12])='[6, 12]'
            return fd

        summary["key_comparisons_valid"] = key_cmp("valid", basicts_valid, remap_fixed(fixed_baselines["valid"]))
        summary["key_comparisons_test"] = key_cmp("test", basicts_test, remap_fixed(fixed_baselines["test"]))

        Path("results/pems04_crossfit_accuracy_cost_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        # CSV
        import csv

        with open("results/pems04_crossfit_accuracy_cost_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["split", "METHOD", "ETA", "MAE", "RMSE", "MAPE", "AVG_COST", "AVG_STAGES"],
            )
            w.writeheader()
            for split in ("valid", "test"):
                for row in summary[split]:
                    w.writerow({
                        "split": split,
                        "METHOD": row["METHOD"],
                        "ETA": row["ETA"],
                        "MAE": row["MAE"],
                        "RMSE": row["RMSE"],
                        "MAPE": row["MAPE"],
                        "AVG_COST": row["AVG_COST"],
                        "AVG_STAGES": row["AVG_STAGES"],
                    })
        newly.extend([
            "results/pems04_crossfit_accuracy_cost_summary.json",
            "results/pems04_crossfit_accuracy_cost_summary.csv",
        ])

    # ---- latency ----
    latency = {}
    if not args.skip_latency:
        print("=== latency profile ===")
        loader, _ = build_loader_split("valid", args.batch_size)
        latency = latency_profile(
            model, loader, device, routes,
            warmup=args.latency_warmup, measure=args.latency_measure,
        )
        latency["static_duplicate_trace"] = static_duplicate_trace()
        Path("results/pems04_crossfit_latency_profile.json").write_text(
            json.dumps(latency, indent=2) + "\n"
        )
        newly.append("results/pems04_crossfit_latency_profile.json")

    # ---- verdict ----
    verdict = "VERDICT D: PLAN A NOT SUPPORTED"
    reasons = []
    if feat_audit.get("verdict_flag") == "SAMPLE SIGNAL PRESENT" and adaptivity_flag == "SAMPLE-ADAPTIVE ROUTING SUPPORTED":
        reasons.append("beats route prior on nontrivial etas + sample signal present")
    elif adaptivity_flag == "SAMPLE-ADAPTIVE ROUTING SUPPORTED":
        reasons.append("beats route prior on some etas")
    else:
        reasons.append(adaptivity_flag)

    if basicts_test:
        k = summary["key_comparisons_test"]["eta_1.0_vs_fixed_full"]
        mae_ok = k["MAE_difference"] <= 0.05  # within ~0.05 MAE of full
        cost_ok = k["cost_reduction_pct"] >= 10.0
        pareto_nonempty = any(x["pareto"] for x in summary["pareto_test"])
        dominated = [x for x in summary["pareto_test"] if x.get("flag")]
        if adaptivity_flag == "SAMPLE-ADAPTIVE ROUTING SUPPORTED" and mae_ok and cost_ok and pareto_nonempty:
            if latency and latency.get("feature_reused") is False:
                # check if overhead large vs savings
                verdict = "VERDICT B: PLAN A SUPPORTED BUT NEEDS EFFICIENCY FIX"
                reasons.append("accuracy-cost routing useful but pre-route overhead not reused")
            else:
                verdict = "VERDICT A: PLAN A STRONGLY SUPPORTED"
        elif adaptivity_flag == "SAMPLE-ADAPTIVE ROUTING SUPPORTED" and not (mae_ok and cost_ok):
            verdict = "VERDICT C: ROUTING WORKS BUT FORECASTING BENEFIT NOT ESTABLISHED"
            reasons.append(f"eta1 vs full MAE_diff={k['MAE_difference']:.4f}, cost_red%={k['cost_reduction_pct']:.2f}")
            if dominated:
                reasons.append(f"dominated points: {[d['label'] for d in dominated]}")
        else:
            verdict = "VERDICT D: PLAN A NOT SUPPORTED"
            reasons.append("failed route-prior and/or forecasting tradeoff checks")
        # soft upgrade/downgrade with pearson
        pears = [gain_diag["G3"]["Pearson"], gain_diag["G6"]["Pearson"], gain_diag["G36"]["Pearson"]]
        if max(pears) < 0.2 and verdict.startswith("VERDICT A"):
            verdict = "VERDICT C: ROUTING WORKS BUT FORECASTING BENEFIT NOT ESTABLISHED"
            reasons.append(f"low VALID gain Pearson {pears}")

    report = {
        "A_controller_quality": {
            "VALID_gain": gain_diag,
            "pairwise_ranking_macro": ranking["macro_pairwise_ranking_accuracy"],
            "pairwise_ranking": ranking["pairs"],
        },
        "B_sample_adaptivity": {
            "feature_dependence": feat_audit,
            "adaptivity_flag": adaptivity_flag,
            "vs_prior_per_eta": {e: per_eta[e].get("vs_route_prior") for e in per_eta},
        },
        "C_oracle_regret": per_eta,
        "D_forecasting_performance": {"valid": basicts_valid, "test": basicts_test},
        "E_fixed_routes": fixed_baselines,
        "F_pareto": {
            "valid": summary.get("pareto_valid"),
            "test": summary.get("pareto_test"),
            "incremental_valid": summary.get("incremental_valid"),
            "incremental_test": summary.get("incremental_test"),
        },
        "G_eta1_key_comparison": {
            "valid": summary.get("key_comparisons_valid", {}).get("eta_1.0_vs_fixed_full"),
            "test": summary.get("key_comparisons_test", {}).get("eta_1.0_vs_fixed_full"),
        },
        "H_runtime": latency,
        "I_crossfit_statistics": crossfit_stats,
        "checkpoint_selection_audit": ckpt_audit,
        "paths": paths,
        "consistency": meta,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "provenance": {
            "reused_existing_files": reused,
            "newly_computed": newly,
            "supernet_retrained": False,
            "controller_retrained": False,
            "crossfit_teachers_retrained": False,
            "test_oracle_built": False,
            "architecture_changed": False,
        },
        "wall_time_sec": time.time() - t_start,
    }
    out = Path("results/pems04_crossfit_planA_complete_report.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("\n========== PLAN A COMPLETE REPORT ==========")
    print("VERDICT:", verdict)
    for r in reasons:
        print(" -", r)
    print("Wrote", out)
    print("wall_time_sec", report["wall_time_sec"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
