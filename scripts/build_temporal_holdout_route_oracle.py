#!/usr/bin/env python3
"""Build raw-scale route oracle on temporal unseen holdout samples (Plan A)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _file_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _extract_state_dict(obj):
    if not isinstance(obj, dict):
        raise TypeError(type(obj))
    if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        sd = obj["model_state_dict"]
    elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    else:
        sd = obj
    if any(k.startswith("module.") for k in sd):
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    return sd


def per_sample_masked_mae(pred, target, null_val: float = 0.0):
    import torch

    if null_val != null_val:
        mask = ~torch.isnan(target)
    else:
        mask = ~torch.isclose(
            target,
            torch.tensor(null_val, device=target.device, dtype=target.dtype),
            atol=5e-5,
            rtol=0.0,
        )
    err = (pred - target).abs() * mask.float()
    denom = mask.float().flatten(1).sum(dim=1).clamp_min(1.0)
    return err.flatten(1).sum(dim=1) / denom


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cfg", default=None, help="optional; if omitted use synth smoke model")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--out", default="results/pems04_temporal_holdout_route_oracle.json")
    p.add_argument(
        "--scaler",
        default=str(ROOT / "datasets/PEMS04/scaler_in12_out12.pkl"),
    )
    args = p.parse_args()

    import torch
    from torch.utils.data import DataLoader, Subset

    from basicts.archs.arch_zoo.ChainForecasting_arch.budget_conditioned_adaptive_f2f import (
        BudgetConditionedAdaptiveF2FNet,
    )
    from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
        load_route_costs,
        route_to_key,
    )
    from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
        gains_from_route_losses,
        build_refinement_route_index_map,
    )
    from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
    from basicts.utils import load_pkl
    from scripts.budget_f2f_synth_kwargs import synthetic_budget_f2f_kwargs

    manifest = json.loads(Path(args.manifest).read_text())
    idxs = list(manifest["oracle_holdout_samples"])
    if args.smoke_test:
        idxs = idxs[: max(8, min(16, len(idxs)))]
        args.max_samples = len(idxs)
        out = Path("/tmp/kasa_planA_holdout_oracle_smoke.json")
    else:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        if args.max_samples is not None:
            idxs = idxs[: int(args.max_samples)]

    device = torch.device(
        args.device
        if ("cuda" not in str(args.device) or torch.cuda.is_available())
        else "cpu"
    )

    if args.smoke_test or args.cfg is None:
        # Code-path smoke: synthetic forced-route MAE oracle (NOT scientific).
        model = BudgetConditionedAdaptiveF2FNet(
            **synthetic_budget_f2f_kwargs(
                node_size=7,
                training_phase="eval",
                route_selection_mode="forced",
            )
        ).to(device).eval()
        routes = [list(r) for r in model.candidate_routes]
        costs = [float(c) for c in model.route_costs.detach().cpu().tolist()]
        records = []
        with torch.no_grad():
            for si in idxs:
                h = torch.randn(1, 12, 7, 4, device=device)
                y = torch.randn(1, 12, 7, 1, device=device)
                route_losses = []
                for rid, route in enumerate(routes):
                    model.set_forced_route(route)
                    pred = model(history_data=h, train=False, return_all=False)
                    mae = float((pred - y).abs().mean().item())
                    route_losses.append(
                        {
                            "route_id": rid,
                            "route": list(route),
                            "final_mae": mae,
                            "cost": costs[rid],
                        }
                    )
                index_map = build_refinement_route_index_map(routes, 12)
                by_name = {
                    name: route_losses[index_map[name]]["final_mae"]
                    for name in ("direct", "half", "quarter", "progressive")
                }
                g = gains_from_route_losses(by_name)
                records.append(
                    {
                        "sample_index": int(si),
                        "split": "temporal_unseen_holdout",
                        "route_final_losses": route_losses,
                        "G3": g["g3"],
                        "G6": g["g6"],
                        "G36": g["g36"],
                    }
                )
        meta = {
            "dataset": manifest.get("dataset", "PEMS04"),
            "horizon": int(manifest.get("horizon", 12)),
            "split_type": "temporal_unseen_holdout",
            "loss_scale": "raw_physical_scale",
            "candidate_routes": routes,
            "candidate_routes_order": [route_to_key(r) for r in routes],
            "route_costs": costs,
            "sample_count": len(records),
            "teacher_checkpoint": str(args.checkpoint),
            "teacher_hash": "smoke",
            "manifest_hash": manifest.get("manifest_hash"),
            "smoke_test": True,
            "note": "SMOKE TEST ONLY - NOT A SCIENTIFIC RESULT",
        }
        out.write_text(json.dumps({"metadata": meta, "records": records}, indent=2))
        print("SMOKE TEST ONLY - NOT A SCIENTIFIC RESULT")
        print("wrote", out, "n=", len(records))
        return 0

    # Formal path: load real checkpoint + subset of TRAIN by holdout indices
    from easytorch.config import import_config
    from basicts.data import SCALER_REGISTRY

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = import_config(str(cfg_path.relative_to(ROOT.resolve())).replace("\\", "/"))
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = (ROOT / ckpt).resolve()

    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM)
    sd = _extract_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    model.set_forced_route(None)

    scaler = load_pkl(str(args.scaler))
    rescale = SCALER_REGISTRY.get(scaler["func"])
    null_val = float(getattr(cfg.TRAIN, "NULL_VAL", 0.0))
    forward_features = list(cfg.MODEL.FORWARD_FEATURES)
    target_features = list(cfg.MODEL.TARGET_FEATURES)

    data_dir = Path(cfg.TRAIN.DATA.DIR)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    h = int(cfg.DATASET_OUTPUT_LEN)
    pin = int(cfg.DATASET_INPUT_LEN)
    base = IndexedTimeSeriesForecastingDataset(
        str(data_dir / f"data_in{pin}_out{h}.pkl"),
        str(data_dir / f"index_in{pin}_out{h}.pkl"),
        "train",
    )
    ds = Subset(base, idxs)
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False)

    routes = [list(r) for r in model.candidate_routes]
    costs = load_route_costs(
        None,
        routes,
        h,
        cost_type=str(cfg.MODEL.PARAM.get("route_cost_type", "normalized_static_cost")),
    )
    index_map = build_refinement_route_index_map(routes, h)
    records = []
    with torch.no_grad():
        for future, history, sample_index in loader:
            history = history.to(device)[..., forward_features]
            future = future.to(device)
            target = future[..., target_features]
            sample_index = sample_index.view(-1).tolist()
            route_mae = []
            for route in routes:
                model.set_forced_route(route)
                model.route_selection_mode = "forced"
                pred = model(history_data=history, train=False, return_all=False)
                y = target[..., : pred.shape[-1]]
                mae_b = per_sample_masked_mae(
                    rescale(pred, **scaler["args"]),
                    rescale(y, **scaler["args"]),
                    null_val=null_val,
                )
                route_mae.append(mae_b.detach().cpu())
            route_mae_t = torch.stack(route_mae, dim=0)
            for bi, si in enumerate(sample_index):
                route_losses = [
                    {
                        "route_id": rid,
                        "route": list(route),
                        "final_mae": float(route_mae_t[rid, bi].item()),
                        "cost": float(costs[rid]),
                    }
                    for rid, route in enumerate(routes)
                ]
                by_name = {
                    name: route_losses[index_map[name]]["final_mae"]
                    for name in ("direct", "half", "quarter", "progressive")
                }
                g = gains_from_route_losses(by_name)
                records.append(
                    {
                        "sample_index": int(si),
                        "split": "temporal_unseen_holdout",
                        "route_final_losses": route_losses,
                        "G3": g["g3"],
                        "G6": g["g6"],
                        "G36": g["g36"],
                    }
                )

    meta = {
        "dataset": str(cfg.DATASET_NAME),
        "horizon": int(h),
        "split_type": "temporal_unseen_holdout",
        "loss_scale": "raw_physical_scale",
        "candidate_routes": routes,
        "candidate_routes_order": [route_to_key(r) for r in routes],
        "route_costs": list(map(float, costs)),
        "sample_count": len(records),
        "teacher_checkpoint": str(ckpt),
        "teacher_hash": _file_sha1(ckpt)[:16],
        "manifest_hash": manifest.get("manifest_hash"),
        "null_val": null_val,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metadata": meta, "records": records}, indent=2))
    print("wrote", out, "n=", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
