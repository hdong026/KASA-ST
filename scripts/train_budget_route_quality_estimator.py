#!/usr/bin/env python3
"""Train Route Quality Estimator on a frozen F2F supernet (default workflow Phase D).

Does not joint-finetune the forecasting backbone.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_conditioned_route_quality_f2f import (
    BudgetConditionedRouteQualityF2FNet,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budget_from_intensity,
    default_candidate_routes,
    load_route_costs,
    parse_candidate_routes,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_decision import (
    feasible_mask_from_budget,
    oracle_best_feasible_route,
    select_route_ids_from_quality,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.route_quality_loss import (
    route_quality_diagnostics,
    route_quality_total_loss,
    summarize_regrets,
)
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from basicts.data.route_quality_dataset import RouteQualityDataset, collate_route_quality


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_supernet_into_model(model: BudgetConditionedRouteQualityF2FNet, ckpt: Path) -> None:
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Allow loading from BudgetConditionedAdaptiveF2FNet checkpoints.
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.freeze_backbone(True)
    model.backbone.eval()


def _build_model(args, routes: list[list[int]], device: torch.device):
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
        "dyn_tau": 0.5,
        "dyn_static_weight": 0.2,
        "use_adaptive_adj": True,
        "adp_hidden_dim": 32,
        "adp_topk": 20,
        "adp_tau": 0.5,
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
        "route_cost_type": (
            "measured_latency"
            if args.route_cost_source == "measured_latency"
            else "normalized_static_cost"
        ),
        "route_cost_file": args.route_cost_file,
        "training_phase": "route_quality",
        "freeze_forecasting_backbone": True,
        "delta_abs": float(args.delta_abs),
        "delta_rel": float(args.delta_rel),
        "rq_d_model": int(args.d_model),
        "rq_temporal_layers": int(args.temporal_layers),
        "rq_spatial_query_count": int(args.spatial_query_count),
        "dataset_name": "PEMS04",
    }
    if args.cfg:
        # Optional: user-provided EasyTorch cfg module path for PARAM overrides.
        import importlib.util

        spec = importlib.util.spec_from_file_location("rq_cfg", args.cfg)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        cfg = mod.CFG
        kwargs.update(dict(cfg.MODEL.PARAM))
        kwargs["candidate_routes"] = routes
        kwargs["training_phase"] = "route_quality"
        kwargs["freeze_forecasting_backbone"] = True
    model = BudgetConditionedRouteQualityF2FNet(**kwargs)
    model.to(device)
    return model


@torch.no_grad()
def validate_route_regret(
    model: BudgetConditionedRouteQualityF2FNet,
    loader: DataLoader,
    intensities: list[float],
    device: torch.device,
    args,
) -> dict[str, Any]:
    model.eval()
    model.backbone.eval()
    costs = model.route_costs.detach().cpu()
    all_pred = []
    all_true = []
    per_eta: dict[str, Any] = {}
    for eta in intensities:
        per_eta[str(eta)] = {
            "regrets": [],
            "tol_regrets": [],
            "selected_costs": [],
            "stage_counts": [],
            "hist": Counter(),
            "selected_true": [],
            "oracle_true": [],
        }

    for history, _si, true_losses in loader:
        history = history.to(device)
        true_losses = true_losses.to(device)
        pred = model.estimate_route_quality(history)["predicted_route_losses"]
        all_pred.append(pred.cpu())
        all_true.append(true_losses.cpu())
        for eta in intensities:
            dec = select_route_ids_from_quality(
                pred,
                model.route_costs,
                float(eta),
                delta_abs=float(args.delta_abs),
                delta_rel=float(args.delta_rel),
            )
            feas = dec["feasible_mask"]
            strict = oracle_best_feasible_route(
                true_losses, model.route_costs, feas, delta_abs=0.0, delta_rel=0.0
            )
            tol_ora = oracle_best_feasible_route(
                true_losses,
                model.route_costs,
                feas,
                delta_abs=float(args.delta_abs),
                delta_rel=float(args.delta_rel),
            )
            sel = dec["selected_route_id"]
            sel_true = true_losses.gather(1, sel.unsqueeze(1)).squeeze(1)
            regret = sel_true - strict["oracle_best_feasible_loss"]
            tol_regret = sel_true - tol_ora["oracle_selected_loss"]
            bucket = per_eta[str(eta)]
            bucket["regrets"].append(regret.cpu())
            bucket["tol_regrets"].append(tol_regret.cpu())
            bucket["selected_costs"].append(dec["selected_cost"].cpu())
            bucket["stage_counts"].extend(
                len(model.candidate_routes[int(i)]) for i in sel.tolist()
            )
            bucket["hist"].update(int(i) for i in sel.tolist())
            bucket["selected_true"].append(sel_true.cpu())
            bucket["oracle_true"].append(strict["oracle_best_feasible_loss"].cpu())

    pred_cat = torch.cat(all_pred, dim=0)
    true_cat = torch.cat(all_true, dim=0)
    diag = route_quality_diagnostics(
        pred_cat, true_cat, rank_ignore_margin=float(args.rank_ignore_margin)
    )
    eta_report = {}
    mean_regret_all = []
    mean_cost_all = []
    for eta, bucket in per_eta.items():
        regrets = torch.cat(bucket["regrets"])
        stats = summarize_regrets(regrets)
        mean_regret_all.append(stats["mean_route_regret"])
        costs_t = torch.cat(bucket["selected_costs"])
        mean_cost_all.append(float(costs_t.mean().item()))
        hist = bucket["hist"]
        n = int(regrets.numel())
        ent = 0.0
        for c in hist.values():
            p = c / max(n, 1)
            ent -= p * math.log(p + 1e-12)
        oracle_hist_note = ""
        if float(eta) in {0.5, 0.75, 1.0} and len(hist) <= 1:
            oracle_hist_note = "ROUTE COLLAPSE WARNING"
            print(
                f"[ROUTE COLLAPSE WARNING] eta={eta}: unique_selected={len(hist)} hist={dict(hist)}"
            )
        eta_report[eta] = {
            **stats,
            "avg_selected_cost": float(costs_t.mean().item()),
            "avg_stage_count": float(sum(bucket["stage_counts"]) / max(n, 1)),
            "route_histogram": {str(k): v for k, v in sorted(hist.items())},
            "entropy": ent,
            "unique_selected_routes": len(hist),
            "selected_true_mae": float(torch.cat(bucket["selected_true"]).mean().item()),
            "oracle_true_mae": float(torch.cat(bucket["oracle_true"]).mean().item()),
            "warning": oracle_hist_note,
        }
    selection_score = float(sum(mean_regret_all) / max(len(mean_regret_all), 1))
    avg_cost = float(sum(mean_cost_all) / max(len(mean_cost_all), 1))
    return {
        "diagnostics": diag,
        "per_eta": eta_report,
        "mean_validation_route_regret": selection_score,
        "mean_selected_cost": avg_cost,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--supernet-checkpoint", required=True)
    parser.add_argument("--train-oracle", required=True)
    parser.add_argument("--valid-oracle", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--spatial-query-count", type=int, default=4)
    parser.add_argument("--lambda-abs", type=float, default=0.25)
    parser.add_argument("--lambda-center", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--lambda-list", type=float, default=0.25)
    parser.add_argument("--rank-ignore-margin", type=float, default=0.02)
    parser.add_argument("--rank-temperature", type=float, default=1.0)
    parser.add_argument("--list-temperature", type=float, default=1.0)
    parser.add_argument("--delta-abs", type=float, default=0.05)
    parser.add_argument("--delta-rel", type=float, default=0.0)
    parser.add_argument("--candidate-routes", nargs="+", default=None)
    parser.add_argument(
        "--route-cost-source", default="static", choices=["static", "measured_latency"]
    )
    parser.add_argument("--route-cost-file", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default="checkpoints/PEMS04/H12/budget_f2f/route_quality")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--data-dir", default="datasets/PEMS04")
    parser.add_argument(
        "--val-intensities",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    args = parser.parse_args()

    _set_seed(int(args.seed))
    device = torch.device(args.device)
    routes = (
        parse_candidate_routes(args.candidate_routes, args.horizon)
        if args.candidate_routes
        else default_candidate_routes(args.horizon)
    )
    costs = load_route_costs(
        args.route_cost_file,
        routes,
        args.horizon,
        cost_type=(
            "measured_latency"
            if args.route_cost_source == "measured_latency"
            else "normalized_static_cost"
        ),
    )

    data_file = str(Path(args.data_dir) / f"data_in12_out{args.horizon}.pkl")
    index_file = str(Path(args.data_dir) / f"index_in12_out{args.horizon}.pkl")
    train_base = IndexedTimeSeriesForecastingDataset(data_file, index_file, "train")
    valid_base = IndexedTimeSeriesForecastingDataset(data_file, index_file, "valid")
    train_ds = RouteQualityDataset(
        train_base,
        args.train_oracle,
        expected_routes=routes,
        expected_costs=costs,
        expected_horizon=args.horizon,
        expected_dataset="PEMS04",
    )
    valid_ds = RouteQualityDataset(
        valid_base,
        args.valid_oracle,
        expected_routes=routes,
        expected_costs=costs,
        expected_horizon=args.horizon,
        expected_dataset="PEMS04",
    )
    print(f"[data] train={len(train_ds)} valid={len(valid_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=2,
        collate_fn=collate_route_quality,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=2,
        collate_fn=collate_route_quality,
    )

    model = _build_model(args, routes, device)
    _load_supernet_into_model(model, Path(args.supernet_checkpoint))
    report = model.trainable_parameter_report()
    print("[params]", json.dumps({k: report[k] for k in report if k != "trainable_names"}))
    print("[trainable]", report["trainable_names"][:20], "...")

    opt = torch.optim.AdamW(
        [p for p in model.route_quality_estimator.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"mean_validation_route_regret": float("inf"), "mean_selected_cost": float("inf")}
    history = []

    for epoch in range(1, int(args.num_epochs) + 1):
        model.train()
        model.backbone.eval()
        running = Counter()
        n_batches = 0
        for history_x, _si, true_losses in train_loader:
            history_x = history_x.to(device)
            true_losses = true_losses.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model.estimate_route_quality(history_x)["predicted_route_losses"]
            loss, parts = route_quality_total_loss(
                pred,
                true_losses,
                lambda_abs=float(args.lambda_abs),
                lambda_center=float(args.lambda_center),
                lambda_rank=float(args.lambda_rank),
                lambda_list=float(args.lambda_list),
                rank_ignore_margin=float(args.rank_ignore_margin),
                rank_temperature=float(args.rank_temperature),
                list_temperature=float(args.list_temperature),
            )
            loss.backward()
            # Gradient audit: backbone must stay None/zero.
            for p in model.backbone.parameters():
                if p.grad is not None and float(p.grad.abs().sum()) > 0:
                    raise RuntimeError("backbone received non-zero gradients")
            opt.step()
            for k, v in parts.items():
                if k.startswith("L_") or k == "total":
                    running[k] += v
            n_batches += 1
        train_parts = {k: running[k] / max(n_batches, 1) for k in running}
        val = validate_route_regret(
            model, valid_loader, list(args.val_intensities), device, args
        )
        row = {"epoch": epoch, "train": train_parts, "valid": val}
        history.append(row)
        print(
            f"[epoch {epoch}] train_total={train_parts.get('total', 0):.4f} "
            f"val_regret={val['mean_validation_route_regret']:.4f} "
            f"val_cost={val['mean_selected_cost']:.4f}"
        )
        improved = val["mean_validation_route_regret"] < best["mean_validation_route_regret"] - 1e-6
        tie = abs(
            val["mean_validation_route_regret"] - best["mean_validation_route_regret"]
        ) <= 1e-6
        cheaper = val["mean_selected_cost"] < best["mean_selected_cost"]
        if improved or (tie and cheaper):
            best = {
                "mean_validation_route_regret": val["mean_validation_route_regret"],
                "mean_selected_cost": val["mean_selected_cost"],
                "epoch": epoch,
            }
            ckpt_path = out_dir / "BudgetConditionedRouteQualityF2FNet_best_val_regret.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "estimator_state_dict": model.route_quality_estimator.state_dict(),
                    "epoch": epoch,
                    "valid": val,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[ckpt] saved {ckpt_path}")

    (out_dir / "train_history.json").write_text(
        json.dumps({"history": history, "best": best}, indent=2) + "\n", encoding="utf-8"
    )
    print("[done] best=", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
