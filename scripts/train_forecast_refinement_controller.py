#!/usr/bin/env python3
"""Train Forecast Refinement Gain Controller on a frozen F2F supernet.

Default workflow Phase D. Does not joint-finetune the backbone.
Does not use test data for training or model selection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_forecast_refinement_route import (
    AdaptiveForecastRefinementRouteNet,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    default_candidate_routes,
    load_route_costs,
    parse_candidate_routes,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_decision import (
    select_routes_from_scores,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_gain_loss import (
    compute_pair_imbalance_weights,
    gain_diagnostics,
    refinement_gain_total_loss,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    route_scores_from_gains,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    feasible_mask_from_budget,
    oracle_best_feasible_route,
)
from basicts.data.forecast_refinement_gain_dataset import (
    ForecastRefinementGainDataset,
    collate_refinement_gains,
)
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_model(args, routes, device):
    kwargs = {
        "node_size": 307,
        "input_len": 12,
        "output_len": int(args.horizon),
        "input_dim": 4,
        "output_dim": 1,
        "patch_len": 3,
        "stride": 4,
        "td_size": 288,
        "dw_size": 7,
        "d_td": 32,
        "d_dw": 32,
        "d_d": 32,
        "d_spa": 32,
        "num_layer": 2,
        "if_time_in_day": True,
        "if_day_in_week": True,
        "if_spatial": True,
        "spatial_scheme": "C",
        "adj_mx_path": "datasets/PEMS04/adj_mx.pkl",
        "use_gcn": True,
        "gcn_hidden_dim": 64,
        "use_dynamic_spatial": True,
        "dyn_hidden_dim": 64,
        "dyn_topk": 20,
        "use_adaptive_adj": True,
        "adp_hidden_dim": 32,
        "adp_topk": 20,
        "use_hybrid_graph": True,
        "hybrid_alpha": 0.2,
        "use_patch_branch": True,
        "use_downsample_branch": True,
        "use_linear_residual_branch": True,
        "patch_embedding_mode": "serial_concat",
        "patch_data_input_mode": "all",
        "post_spatial_mode": "adaptive_only",
        "spatial_placement": "interleaved_progressive",
        "use_prev_condition": True,
        "progressive_spatial_ratios": [0.25, 0.5, 1.0],
        "progressive_spatial_topks": [8, 16, 32],
        "progressive_spatial_alphas": [0.03, 0.06, 0.10],
        "use_forecast_state_adapter": True,
        "forecast_state_adapter_mode": "condition_only",
        "forecast_state_adapter_hidden_dim": 16,
        "forecast_state_adapter_epsilon": 0.02,
        "candidate_routes": routes,
        "route_selection_mode": "sample",
        "route_granularity": "sample",
        "route_cost_type": "normalized_static_cost",
        "route_cost_file": args.route_cost_file,
        "training_phase": "refinement_controller",
        "freeze_forecasting_backbone": True,
        "delta_abs": float(args.delta_abs),
        "controller_dim": int(args.controller_dim),
        "pooling_queries": int(args.pooling_queries),
        "dataset_name": "PEMS04",
    }
    if args.cfg:
        import importlib.util

        spec = importlib.util.spec_from_file_location("frc_cfg", args.cfg)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        kwargs.update(dict(mod.CFG.MODEL.PARAM))
        kwargs["candidate_routes"] = routes
        kwargs["training_phase"] = "refinement_controller"
        kwargs["freeze_forecasting_backbone"] = True
    model = AdaptiveForecastRefinementRouteNet(**kwargs)
    model.to(device)
    return model


def _load_supernet(model, ckpt: Path):
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.freeze_backbone(True)
    model.backbone.eval()


@torch.no_grad()
def validate(model, loader, intensities, device, args, pair_weights, mean_train_gains):
    model.eval()
    model.backbone.eval()
    all_pred, all_true, all_losses = [], [], []
    per_eta = {
        str(e): {
            "regrets": [],
            "costs": [],
            "hist": Counter(),
            "sel_true": [],
            "ora_true": [],
            "stages": [],
        }
        for e in intensities
    }
    prior_regrets = {str(e): [] for e in intensities}

    for history, _si, gains, losses in loader:
        history = history.to(device)
        gains = gains.to(device)
        losses = losses.to(device)
        pred = model.estimate_refinement_gains(history)["predicted_gains"]
        all_pred.append(pred.cpu())
        all_true.append(gains.cpu())
        all_losses.append(losses.cpu())
        scores = route_scores_from_gains(
            pred[:, 0], pred[:, 1], pred[:, 2],
            index_map=model.index_map, n_routes=len(model.candidate_routes),
        )
        prior_scores = route_scores_from_gains(
            mean_train_gains[0].expand(history.shape[0]),
            mean_train_gains[1].expand(history.shape[0]),
            mean_train_gains[2].expand(history.shape[0]),
            index_map=model.index_map,
            n_routes=len(model.candidate_routes),
        )
        for eta in intensities:
            dec = select_routes_from_scores(
                scores, model.route_costs, float(eta), delta_abs=float(args.delta_abs)
            )
            pdec = select_routes_from_scores(
                prior_scores, model.route_costs, float(eta), delta_abs=float(args.delta_abs)
            )
            feas = dec["feasible_mask"]
            strict = oracle_best_feasible_route(losses, model.route_costs, feas, delta_abs=0.0)
            sel = dec["selected_route_id"]
            sel_true = losses.gather(1, sel.unsqueeze(1)).squeeze(1)
            regret = sel_true - strict["oracle_best_feasible_loss"]
            prior_sel = pdec["selected_route_id"]
            prior_true = losses.gather(1, prior_sel.unsqueeze(1)).squeeze(1)
            prior_regrets[str(eta)].append(
                (prior_true - strict["oracle_best_feasible_loss"]).cpu()
            )
            bucket = per_eta[str(eta)]
            bucket["regrets"].append(regret.cpu())
            bucket["costs"].append(dec["selected_cost"].cpu())
            bucket["hist"].update(int(i) for i in sel.tolist())
            bucket["sel_true"].append(sel_true.cpu())
            bucket["ora_true"].append(strict["oracle_best_feasible_loss"].cpu())
            bucket["stages"].extend(
                len(model.candidate_routes[int(i)]) for i in sel.tolist()
            )

    pred_cat = torch.cat(all_pred)
    true_cat = torch.cat(all_true)
    diag = gain_diagnostics(pred_cat, true_cat)
    for name in ("g3", "g6", "g36"):
        if diag.get(f"GAIN_COLLAPSE_WARNING_{name}"):
            print(f"[GAIN COLLAPSE WARNING] {name}")

    eta_report = {}
    mean_regrets = []
    mean_costs = []
    for eta, bucket in per_eta.items():
        regrets = torch.cat(bucket["regrets"])
        costs_t = torch.cat(bucket["costs"])
        n = int(regrets.numel())
        hist = bucket["hist"]
        ent = 0.0
        for c in hist.values():
            p = c / max(n, 1)
            ent -= p * math.log(p + 1e-12)
        if float(eta) in {0.5, 0.75, 1.0} and len(hist) <= 1:
            print(f"[ROUTE COLLAPSE WARNING] eta={eta} hist={dict(hist)}")
        prior_r = torch.cat(prior_regrets[eta]).mean().item()
        mean_r = float(regrets.mean().item())
        mean_regrets.append(mean_r)
        mean_costs.append(float(costs_t.mean().item()))
        eta_report[eta] = {
            "route_regret_mean": mean_r,
            "route_regret_median": float(regrets.median().item()),
            "route_regret_p90": float(torch.quantile(regrets, 0.9).item()),
            "avg_selected_cost": float(costs_t.mean().item()),
            "avg_stage_count": float(sum(bucket["stages"]) / max(n, 1)),
            "route_histogram": {str(k): v for k, v in sorted(hist.items())},
            "entropy": ent,
            "unique_routes": len(hist),
            "controller_true_mae": float(torch.cat(bucket["sel_true"]).mean().item()),
            "strict_oracle_true_mae": float(torch.cat(bucket["ora_true"]).mean().item()),
            "route_prior_mean_regret": float(prior_r),
        }

    # Feature dependence on up to 256 samples
    feat_audit = None
    n_feat = min(256, pred_cat.shape[0])
    # rebuild one pass
    xs, ys = [], []
    for history, _si, gains, losses in loader:
        xs.append(history)
        ys.append(losses)
        if sum(x.shape[0] for x in xs) >= n_feat:
            break
    history = torch.cat(xs, dim=0)[:n_feat].to(device)
    losses = torch.cat(ys, dim=0)[:n_feat].to(device)
    gn = model.estimate_refinement_gains(history)["predicted_gains"]
    gp = model.estimate_refinement_gains(history[torch.randperm(n_feat, device=device)])[
        "predicted_gains"
    ]
    gz = model.estimate_refinement_gains(torch.zeros_like(history))["predicted_gains"]
    gr = model.estimate_refinement_gains(torch.flip(history, dims=[1]))["predicted_gains"]

    def _regret_of(gains_hat):
        sc = route_scores_from_gains(
            gains_hat[:, 0], gains_hat[:, 1], gains_hat[:, 2],
            index_map=model.index_map, n_routes=len(model.candidate_routes),
        )
        dec = select_routes_from_scores(sc, model.route_costs, 0.75, delta_abs=float(args.delta_abs))
        feas = dec["feasible_mask"]
        strict = oracle_best_feasible_route(losses, model.route_costs, feas, delta_abs=0.0)
        sel_true = losses.gather(1, dec["selected_route_id"].unsqueeze(1)).squeeze(1)
        return float((sel_true - strict["oracle_best_feasible_loss"]).mean().item())

    rn, rp, rz, rr = _regret_of(gn), _regret_of(gp), _regret_of(gz), _regret_of(gr)
    feat_audit = {
        "mean_abs_gain_normal_vs_permuted": float((gn - gp).abs().mean().item()),
        "mean_abs_gain_normal_vs_zero": float((gn - gz).abs().mean().item()),
        "mean_abs_gain_normal_vs_reverse": float((gn - gr).abs().mean().item()),
        "regret_normal": rn,
        "regret_permuted": rp,
        "regret_zero": rz,
        "regret_reverse": rr,
    }
    if rn >= min(rp, rz) - 1e-6:
        print("[SAMPLE SIGNAL FAILURE] normal history regret not better than permuted/zero")
        feat_audit["SAMPLE_SIGNAL_FAILURE"] = True

    return {
        "diagnostics": diag,
        "per_eta": eta_report,
        "mean_validation_route_regret": float(sum(mean_regrets) / max(len(mean_regrets), 1)),
        "mean_selected_cost": float(sum(mean_costs) / max(len(mean_costs), 1)),
        "feature_dependence": feat_audit,
        "target_scale": "raw_physical_mae_gain",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--supernet-checkpoint", required=True)
    parser.add_argument("--train-oracle", required=True)
    parser.add_argument("--valid-oracle", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--controller-dim", type=int, default=128)
    parser.add_argument("--pooling-queries", type=int, default=4)
    parser.add_argument("--lambda-abs", type=float, default=0.25)
    parser.add_argument("--lambda-center", type=float, default=1.0)
    parser.add_argument("--lambda-corr", type=float, default=0.5)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--lambda-full", type=float, default=0.5)
    parser.add_argument("--rank-ignore-margin", type=float, default=0.02)
    parser.add_argument("--rank-temperature", type=float, default=0.05)
    parser.add_argument("--delta-abs", type=float, default=0.05)
    parser.add_argument("--candidate-routes", nargs="+", default=None)
    parser.add_argument("--route-cost-file", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default="checkpoints/PEMS04/H12/budget_f2f/refinement_controller")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--data-dir", default="datasets/PEMS04")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--val-intensities", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    args = parser.parse_args()

    _set_seed(int(args.seed))
    device = torch.device(args.device)
    routes = (
        parse_candidate_routes(args.candidate_routes, args.horizon)
        if args.candidate_routes
        else default_candidate_routes(args.horizon)
    )
    costs = load_route_costs(args.route_cost_file, routes, args.horizon)

    data_file = str(Path(args.data_dir) / f"data_in12_out{args.horizon}.pkl")
    index_file = str(Path(args.data_dir) / f"index_in12_out{args.horizon}.pkl")
    train_ds = ForecastRefinementGainDataset(
        IndexedTimeSeriesForecastingDataset(data_file, index_file, "train"),
        args.train_oracle,
        expected_routes=routes,
        expected_costs=costs,
        expected_horizon=args.horizon,
        expected_dataset="PEMS04",
    )
    valid_ds = ForecastRefinementGainDataset(
        IndexedTimeSeriesForecastingDataset(data_file, index_file, "valid"),
        args.valid_oracle,
        expected_routes=routes,
        expected_costs=costs,
        expected_horizon=args.horizon,
        expected_dataset="PEMS04",
    )
    print(f"[data] train={len(train_ds)} valid={len(valid_ds)} target_scale={train_ds.target_scale}")

    train_loader = DataLoader(
        train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=2,
        collate_fn=collate_refinement_gains,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=2,
        collate_fn=collate_refinement_gains,
    )

    # Pair imbalance from full train gains
    all_gains = torch.stack(
        [torch.tensor(train_ds.gains[i]) for i in train_ds.sample_indices], dim=0
    )
    mean_train_gains = all_gains.mean(dim=0).to(device)
    train_scores = route_scores_from_gains(
        all_gains[:, 0], all_gains[:, 1], all_gains[:, 2],
        index_map=train_ds.index_map, n_routes=len(routes),
    )
    pair_weights, pair_report = compute_pair_imbalance_weights(
        train_scores, rank_ignore_margin=float(args.rank_ignore_margin)
    )
    print("[pair_weights]", json.dumps(pair_report, indent=2))

    model = _build_model(args, routes, device)
    _load_supernet(model, Path(args.supernet_checkpoint))
    rep = model.trainable_parameter_report()
    print("[params]", {k: rep[k] for k in rep if k != "trainable_names"})
    print("[trainable]", rep["trainable_names"])

    opt = torch.optim.AdamW(
        model.gain_controller.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"mean_validation_route_regret": float("inf"), "mean_selected_cost": float("inf")}
    bad_epochs = 0
    run_history = []

    for epoch in range(1, int(args.num_epochs) + 1):
        model.train()
        model.backbone.eval()
        running = Counter()
        n_batches = 0
        for batch_history, _si, gains, _losses in train_loader:
            batch_history = batch_history.to(device)
            gains = gains.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model.estimate_refinement_gains(batch_history)["predicted_gains"]
            loss, parts = refinement_gain_total_loss(
                pred,
                gains,
                index_map=model.index_map,
                n_routes=len(routes),
                lambda_abs=float(args.lambda_abs),
                lambda_center=float(args.lambda_center),
                lambda_corr=float(args.lambda_corr),
                lambda_rank=float(args.lambda_rank),
                lambda_full=float(args.lambda_full),
                rank_ignore_margin=float(args.rank_ignore_margin),
                rank_temperature=float(args.rank_temperature),
                pair_weights=pair_weights,
            )
            loss.backward()
            for p in model.backbone.parameters():
                if p.grad is not None and float(p.grad.abs().sum()) > 0:
                    raise RuntimeError("backbone received gradients")
            opt.step()
            for k, v in parts.items():
                running[k] += v
            n_batches += 1
        train_parts = {k: running[k] / max(n_batches, 1) for k in running}
        val = validate(
            model, valid_loader, list(args.val_intensities), device, args,
            pair_weights, mean_train_gains,
        )
        run_history.append({"epoch": epoch, "train": train_parts, "valid": val})
        print(
            f"[epoch {epoch}] train={train_parts.get('total', 0):.4f} "
            f"val_regret={val['mean_validation_route_regret']:.4f} "
            f"val_cost={val['mean_selected_cost']:.4f}"
        )
        improved = val["mean_validation_route_regret"] < best["mean_validation_route_regret"] - 1e-4
        tie = abs(val["mean_validation_route_regret"] - best["mean_validation_route_regret"]) <= 1e-4
        cheaper = val["mean_selected_cost"] < best["mean_selected_cost"]
        if improved or (tie and cheaper):
            best = {
                "mean_validation_route_regret": val["mean_validation_route_regret"],
                "mean_selected_cost": val["mean_selected_cost"],
                "epoch": epoch,
            }
            bad_epochs = 0
            ckpt = {
                "controller_state_dict": model.gain_controller.state_dict(),
                "model_state_dict": model.state_dict(),
                "controller_config": {
                    "controller_dim": int(args.controller_dim),
                    "pooling_queries": int(args.pooling_queries),
                    "delta_abs": float(args.delta_abs),
                    "candidate_routes": routes,
                    "route_costs": costs,
                    "horizon": int(args.horizon),
                    "architecture_version": "forecast_refinement_gain_v1",
                },
                "supernet_checkpoint": str(args.supernet_checkpoint),
                "valid": val,
                "pair_imbalance_weights": pair_report,
                "epoch": epoch,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "refinement_controller_best_val_regret.pt")
            print(f"[ckpt] saved best_regret @ epoch {epoch}")
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                print(f"[early-stop] patience={args.patience}")
                break

    (out_dir / "train_history.json").write_text(
        json.dumps({"history": run_history, "best": best}, indent=2) + "\n", encoding="utf-8"
    )
    print("[done] best=", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
