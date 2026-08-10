#!/usr/bin/env python3
"""Build offline oracle route labels for planner imitation (train/valid only).

Uses batched forced-route forwards and raw-physical-scale per-sample MAE.
Does not use the test split for training oracles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _normalize_split(split: str) -> str:
    s = str(split).lower()
    if s in {"val", "valid"}:
        return "valid"
    if s == "train":
        return "train"
    if s == "test":
        raise ValueError("test split is not allowed for training oracle generation")
    raise ValueError(f"unsupported split: {split!r}")


def _cfg_import_path(cfg: Path) -> str:
    cfg = cfg.resolve()
    return str(cfg.relative_to(ROOT.resolve())).replace("\\", "/")


def _extract_state_dict(obj):
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(obj)}")
    if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        sd = obj["model_state_dict"]
    elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    else:
        sd = obj
    if any(k.startswith("module.") for k in sd):
        sd = {
            k[len("module.") :] if k.startswith("module.") else k: v
            for k, v in sd.items()
        }
    return sd


def _allowed_key(name: str) -> bool:
    return str(name).endswith("num_batches_tracked")


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def _source_fingerprint() -> str:
    paths = [
        ROOT / "basicts/archs/arch_zoo/ChainForecasting_arch/budget_conditioned_adaptive_f2f.py",
        ROOT / "basicts/archs/arch_zoo/ChainForecasting_arch/budget_conditioned_f2f_loss.py",
        ROOT / "basicts/runners/runner_zoo/chain_forecasting_runner.py",
        ROOT / "basicts/archs/arch_zoo/ChainForecasting_arch/budget_route_utils.py",
        ROOT / "scripts/build_budget_route_oracle.py",
        ROOT / "scripts/run_budget_conditioned_f2f.py",
    ]
    sha = hashlib.sha1()
    for p in paths:
        sha.update(str(p.relative_to(ROOT)).encode())
        sha.update(b"\0")
        if p.is_file():
            sha.update(p.read_bytes())
        sha.update(b"\0")
    return sha.hexdigest()[:12]


def per_sample_masked_mae(pred, target, null_val: float = 0.0):
    """Raw-scale per-sample masked MAE → [B]."""
    import torch

    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch pred={tuple(pred.shape)} tgt={tuple(target.shape)}")
    if null_val != null_val:  # NaN
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--route-cost-file", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "valid"])
    parser.add_argument(
        "--intensities",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.0,
        help="raw-scale MAE tolerance for near-best cheapest oracle",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--scaler",
        default=str(ROOT / "datasets/PEMS04/scaler_in12_out12.pkl"),
    )
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Subset
    from easytorch.config import import_config

    from basicts.data import IndexedTimeSeriesForecastingDataset, SCALER_REGISTRY
    from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
        budget_from_intensity,
        load_route_costs,
        route_to_key,
    )
    from basicts.utils import load_pkl

    split = _normalize_split(args.split)
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (ROOT / ckpt_path).resolve()
    scaler_path = Path(args.scaler)
    if not scaler_path.is_absolute():
        scaler_path = (ROOT / scaler_path).resolve()

    cfg = import_config(_cfg_import_path(cfg_path))
    scaler = load_pkl(str(scaler_path))
    rescale = SCALER_REGISTRY.get(scaler["func"])
    null_val = float(getattr(cfg.TRAIN, "NULL_VAL", 0.0))
    forward_features = list(cfg.MODEL.FORWARD_FEATURES)
    target_features = list(cfg.MODEL.TARGET_FEATURES)

    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM)
    raw = torch.load(str(ckpt_path), map_location="cpu")
    sd = _extract_state_dict(raw)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing, unexpected = list(missing), list(unexpected)
    print(
        json.dumps(
            {"missing_keys": missing, "unexpected_keys": unexpected},
            indent=2,
        )
    )
    bad_m = [k for k in missing if not _allowed_key(k)]
    bad_u = [k for k in unexpected if not _allowed_key(k)]
    if bad_m or bad_u:
        raise RuntimeError(
            f"checkpoint mismatch: missing={bad_m} unexpected={bad_u}"
        )

    if "cuda" in str(args.device).lower() and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    model = model.to(device).eval()
    model.set_forced_route(None)

    data_dir = Path(
        cfg.TRAIN.DATA.DIR if split == "train" else cfg.VAL.DATA.DIR
    )
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    h = int(cfg.DATASET_OUTPUT_LEN)
    p = int(cfg.DATASET_INPUT_LEN)
    ds = IndexedTimeSeriesForecastingDataset(
        data_file_path=str(data_dir / f"data_in{p}_out{h}.pkl"),
        index_file_path=str(data_dir / f"index_in{p}_out{h}.pkl"),
        mode=split,
    )
    if args.max_samples is not None:
        ds = Subset(ds, list(range(min(int(args.max_samples), len(ds)))))
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False)

    routes = [list(r) for r in model.candidate_routes]
    cost_type = str(cfg.MODEL.PARAM.get("route_cost_type", "normalized_static_cost"))
    costs = load_route_costs(
        args.route_cost_file,
        routes,
        h,
        cost_type=cost_type,
    )

    records = []
    with torch.no_grad():
        for future, history, sample_index in loader:
            history = history.to(device)[..., forward_features]
            future = future.to(device)
            target = future[..., target_features]
            b = history.shape[0]
            sample_index = sample_index.view(-1).tolist()

            # [R, B] raw-scale per-sample MAE
            route_mae = []
            for rid, route in enumerate(routes):
                model.set_forced_route(route)
                model.route_selection_mode = "forced"
                pred = model(history_data=history, train=False, return_all=False)
                y = target[..., : pred.shape[-1]]
                pred_raw = rescale(pred, **scaler["args"])
                y_raw = rescale(y, **scaler["args"])
                mae_b = per_sample_masked_mae(pred_raw, y_raw, null_val=null_val)
                route_mae.append(mae_b.detach().cpu())
            route_mae_t = torch.stack(route_mae, dim=0)  # [R,B]

            for bi in range(b):
                route_losses = []
                for rid, route in enumerate(routes):
                    route_losses.append(
                        {
                            "route_id": rid,
                            "route": list(route),
                            "final_mae": float(route_mae_t[rid, bi].item()),
                            "cost": float(costs[rid]),
                        }
                    )
                for eta in args.intensities:
                    bud = budget_from_intensity(float(eta), costs)
                    feasible = [
                        e for e in route_losses if e["cost"] <= bud + 1e-8
                    ]
                    if not feasible:
                        feasible = [min(route_losses, key=lambda e: e["cost"])]
                    best = min(feasible, key=lambda e: e["final_mae"])
                    near = [
                        e
                        for e in feasible
                        if e["final_mae"] <= best["final_mae"] + float(args.delta)
                    ]
                    oracle = min(near, key=lambda e: e["cost"])
                    if oracle not in feasible:
                        raise RuntimeError("oracle not in feasible set")
                    # Assert oracle id is among feasible route ids
                    feas_ids = {e["route_id"] for e in feasible}
                    if oracle["route_id"] not in feas_ids:
                        raise RuntimeError(
                            f"oracle route_id {oracle['route_id']} not feasible "
                            f"for intensity={eta} budget={bud}"
                        )
                    records.append(
                        {
                            "sample_index": int(sample_index[bi]),
                            "split": split,
                            "intensity": float(eta),
                            "budget": float(bud),
                            "feasible_routes": [e["route"] for e in feasible],
                            "feasible_route_ids": sorted(feas_ids),
                            "route_final_losses": route_losses,
                            "oracle_route_id": int(oracle["route_id"]),
                            "oracle_route": list(oracle["route"]),
                            "oracle_route_cost": float(oracle["cost"]),
                            "best_route_loss": float(best["final_mae"]),
                        }
                    )

    metadata = {
        "dataset": str(cfg.DATASET_NAME),
        "horizon": int(h),
        "split": split,
        "candidate_routes": routes,
        "candidate_routes_order": [route_to_key(r) for r in routes],
        "route_costs": list(map(float, costs)),
        "route_cost_type": cost_type,
        "intensities": [float(x) for x in args.intensities],
        "scaler_path": str(scaler_path),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_hash": _file_sha1(ckpt_path)[:16],
        "source_fingerprint": _source_fingerprint(),
        "delta": float(args.delta),
        "loss_scale": "raw_physical_scale",
        "null_val": null_val,
        "batch_size": int(args.batch_size),
        "n_records": len(records),
        "n_samples": len({r["sample_index"] for r in records}),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"metadata": metadata, "records": records}, indent=2),
        encoding="utf-8",
    )
    print("wrote", out, "n_records=", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
