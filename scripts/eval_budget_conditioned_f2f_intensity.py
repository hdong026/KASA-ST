#!/usr/bin/env python3
"""Checkpoint-only evaluation for budget-conditioned adaptive F2F.

Loads a trained checkpoint and evaluates:
  1) forced routes (val + test)
  2) planner intensity sweeps (batch + sample granularity)

No training, no optimizer, no checkpoint write-back.
All MAE/RMSE/MAPE are computed on inverse-transformed (raw physical) scale.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keys that may appear as missing/unexpected without indicating structural drift.
# Only non-persistent / tracking buffers are allowed to be ignored.
_ALLOWED_KEY_SUFFIXES = (
    "num_batches_tracked",
)


def _route_key(route: list[int]) -> str:
    return ",".join(str(int(x)) for x in route)


def _is_allowed_key(name: str) -> bool:
    return any(str(name).endswith(suf) for suf in _ALLOWED_KEY_SUFFIXES)


def _material_keys(keys: list[str]) -> list[str]:
    return [k for k in keys if not _is_allowed_key(k)]


def _extract_state_dict(obj: Any) -> dict:
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


def _select_features(data, indices: list[int] | None):
    if indices is None:
        return data
    return data[..., indices]


def _hist_from_ids(ids: list[int], candidates: list[list[int]]) -> dict[str, int]:
    counter = Counter(int(i) for i in ids)
    out: dict[str, int] = {}
    for rid, cnt in sorted(counter.items()):
        if 0 <= rid < len(candidates):
            key = _route_key(candidates[rid])
        else:
            key = f"id:{rid}"
        out[key] = int(cnt)
    return out


def load_checkpoint_strict(model, checkpoint_path: Path) -> dict[str, Any]:
    import torch

    raw = torch.load(str(checkpoint_path), map_location="cpu")
    sd = _extract_state_dict(raw)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = list(missing)
    unexpected = list(unexpected)
    report = {
        "checkpoint": str(checkpoint_path),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "allowed_ignored_missing": [k for k in missing if _is_allowed_key(k)],
        "allowed_ignored_unexpected": [k for k in unexpected if _is_allowed_key(k)],
    }
    print("[checkpoint] load report:")
    print(json.dumps(report, indent=2))
    bad_missing = _material_keys(missing)
    bad_unexpected = _material_keys(unexpected)
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "checkpoint/model structural mismatch (not ignorable buffers):\n"
            f"  material missing_keys={bad_missing}\n"
            f"  material unexpected_keys={bad_unexpected}"
        )
    return report


def _normalize_split(split: str) -> str:
    """Map CLI aliases to TimeSeriesForecastingDataset modes."""
    s = str(split).lower()
    if s in {"val", "valid"}:
        return "valid"
    if s == "test":
        return "test"
    if s == "train":
        return "train"
    raise ValueError(f"unsupported split: {split!r} (expected val/valid/test)")


def build_loader(cfg, split: str, batch_size: int):
    from torch.utils.data import DataLoader
    from basicts.data import TimeSeriesForecastingDataset

    mode = _normalize_split(split)
    data_dir = Path(cfg.TEST.DATA.DIR if mode == "test" else cfg.VAL.DATA.DIR)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    h = int(cfg.DATASET_OUTPUT_LEN)
    p = int(cfg.DATASET_INPUT_LEN)
    ds = TimeSeriesForecastingDataset(
        data_file_path=str(data_dir / f"data_in{p}_out{h}.pkl"),
        index_file_path=str(data_dir / f"index_in{p}_out{h}.pkl"),
        mode=mode,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False), len(ds)


def evaluate_loader(
    model,
    loader,
    *,
    device,
    forward_features: list[int],
    target_features: list[int],
    scaler: dict,
    null_val: float,
    candidates: list[list[int]],
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run one evaluation pass. Metrics on raw physical scale after inverse transform."""
    import torch
    from basicts.data import SCALER_REGISTRY
    from basicts.metrics import masked_mae, masked_mape, masked_rmse

    preds: list[torch.Tensor] = []
    tgts: list[torch.Tensor] = []
    sample_route_ids: list[int] = []
    batch_route_ids: list[int] = []
    costs: list[torch.Tensor] = []
    budgets: list[torch.Tensor] = []
    sample_stage_counts: list[float] = []
    n_batches = 0
    t0 = time.perf_counter()

    rescale = SCALER_REGISTRY.get(scaler["func"])
    with torch.no_grad():
        for future, history in loader:
            if max_batches is not None and n_batches >= int(max_batches):
                break
            history = history.to(device)
            future = future.to(device)
            history = _select_features(history, forward_features)
            target = _select_features(future, target_features)

            out = model(history_data=history, train=False, return_all=True)
            pred = out["pred"]
            y = target[..., : pred.shape[-1]]

            preds.append(pred.detach().cpu())
            tgts.append(y.detach().cpu())

            rid = out["selected_route_id"].detach().cpu().view(-1)
            sample_route_ids.extend(int(x) for x in rid.tolist())
            batch_id = out.get("batch_route_id")
            if batch_id is None:
                counts = torch.bincount(rid, minlength=len(candidates))
                batch_id = int(counts.argmax().item())
            batch_route_ids.append(int(batch_id))

            cost = out["selected_cost"].detach().cpu().view(-1)
            bud = out["budget"].detach().cpu().view(-1)
            costs.append(cost)
            budgets.append(bud)

            for sid in rid.tolist():
                sid = int(sid)
                if 0 <= sid < len(candidates):
                    sample_stage_counts.append(float(len(candidates[sid])))
                else:
                    sample_stage_counts.append(float(len(out["chain_resolutions"])))

            n_batches += 1
            if device.type == "cuda":
                torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    pred = torch.cat(preds, dim=0)
    tgt = torch.cat(tgts, dim=0)
    pred_raw = rescale(pred, **scaler["args"])
    target_raw = rescale(tgt, **scaler["args"])

    cost = torch.cat(costs, dim=0).float()
    bud = torch.cat(budgets, dim=0).float()
    n_samples = int(pred_raw.shape[0])

    return {
        "mae": float(masked_mae(pred_raw, target_raw, null_val=null_val).item()),
        "rmse": float(masked_rmse(pred_raw, target_raw, null_val=null_val).item()),
        "mape": float(masked_mape(pred_raw, target_raw, null_val=null_val).item()),
        "route_histogram_batch": _hist_from_ids(batch_route_ids, candidates),
        "route_histogram_sample": _hist_from_ids(sample_route_ids, candidates),
        "average_stage_count": float(
            sum(sample_stage_counts) / max(len(sample_stage_counts), 1)
        ),
        "average_selected_cost": float(cost.mean().item()) if cost.numel() else 0.0,
        "budget": float(bud.mean().item()) if bud.numel() else 0.0,
        "violation_rate": float((cost > bud + 1e-8).float().mean().item())
        if cost.numel()
        else 0.0,
        "n_samples": n_samples,
        "n_batches": int(n_batches),
        "wall_time_sec": float(elapsed),
        "loss_scale": "raw_physical_scale",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Budget-conditioned F2F checkpoint evaluation (no training)."
    )
    parser.add_argument("--cfg", required=True, help="Generated EasyTorch config .py")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--scaler",
        default=str(ROOT / "datasets/PEMS04/scaler_in12_out12.pkl"),
        help="Scaler pkl path (default: datasets/PEMS04/scaler_in12_out12.pkl)",
    )
    parser.add_argument(
        "--forced-routes",
        nargs="+",
        default=["12", "6,12", "3,12", "3,6,12"],
        help="Forced routes to evaluate (comma-separated resolutions).",
    )
    parser.add_argument(
        "--intensities",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["valid", "test"],
        choices=["val", "valid", "test"],
        help="Dataset splits to evaluate (val is alias of valid; default: valid test).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on batches per eval pass (for smoke tests).",
    )
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument(
        "--skip-intensity",
        action="store_true",
        help="Only run forced-route evaluation.",
    )
    parser.add_argument(
        "--skip-forced",
        action="store_true",
        help="Only run intensity evaluation.",
    )
    args = parser.parse_args()

    import torch
    from easytorch.config import import_config

    from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
        parse_route,
        validate_route,
    )
    from basicts.utils import load_pkl

    # EasyTorch import_config expects a repo-relative path / dotted module path,
    # not an absolute filesystem path (absolute -> ModuleNotFoundError: '.home').
    cfg_path = Path(args.cfg)
    if cfg_path.is_absolute():
        try:
            cfg_rel = cfg_path.resolve().relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                f"cfg must live under repo root {ROOT}, got {cfg_path}"
            ) from exc
    else:
        cfg_rel = cfg_path
    cfg_path = (ROOT / cfg_rel).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"cfg not found: {cfg_path}")
    cfg_import = str(cfg_rel).replace("\\", "/")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (ROOT / ckpt_path).resolve()
    scaler_path = Path(args.scaler)
    if not scaler_path.is_absolute():
        scaler_path = (ROOT / scaler_path).resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"scaler not found: {scaler_path}")

    cfg = import_config(cfg_import)
    scaler = load_pkl(str(scaler_path))
    if not isinstance(scaler, dict) or "func" not in scaler or "args" not in scaler:
        raise ValueError(
            f"invalid scaler object from {scaler_path}: keys={list(scaler)}"
        )

    null_val = float(getattr(cfg.TRAIN, "NULL_VAL", 0.0))
    forward_features = list(cfg.MODEL.FORWARD_FEATURES)
    target_features = list(cfg.MODEL.TARGET_FEATURES)
    h = int(cfg.DATASET_OUTPUT_LEN)

    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM)
    load_report = load_checkpoint_strict(model, ckpt_path)

    if "cuda" in str(args.device).lower() and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        if "cuda" in str(args.device).lower() and not torch.cuda.is_available():
            print("[warn] CUDA unavailable; falling back to CPU")
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    if hasattr(model, "set_training_phase"):
        model.set_training_phase("eval")

    candidates = [list(r) for r in model.candidate_routes]
    results: list[dict[str, Any]] = []

    # ---- forced routes ----
    if not args.skip_forced:
        forced_routes = []
        for spec in args.forced_routes:
            route = parse_route(spec)
            validate_route(route, horizon=h)
            if route not in candidates:
                raise ValueError(
                    f"forced route {route} not in candidate pool {candidates}"
                )
            forced_routes.append(route)

        for split in args.splits:
            mode = _normalize_split(split)
            loader, n_ds = build_loader(cfg, mode, args.batch_size)
            for route in forced_routes:
                model.set_forced_route(route)
                model.route_selection_mode = "forced"
                row = evaluate_loader(
                    model,
                    loader,
                    device=device,
                    forward_features=forward_features,
                    target_features=target_features,
                    scaler=scaler,
                    null_val=null_val,
                    candidates=candidates,
                    max_batches=args.max_batches,
                )
                row.update(
                    {
                        "eval_mode": "forced",
                        "split": mode,
                        "route": _route_key(route),
                        "eta": None,
                        "route_selection_mode": "forced",
                        "route_granularity": None,
                        "dataset_size": int(n_ds),
                    }
                )
                results.append(row)
                print(
                    f"[forced] split={row['split']} route={row['route']} "
                    f"mae={row['mae']:.4f} rmse={row['rmse']:.4f} mape={row['mape']:.4f} "
                    f"n={row['n_samples']}/{row['n_batches']}"
                )

    # ---- intensity sweeps (batch + sample) ----
    if not args.skip_intensity:
        model.set_forced_route(None)
        for split in args.splits:
            mode = _normalize_split(split)
            loader, n_ds = build_loader(cfg, mode, args.batch_size)
            for eta in args.intensities:
                model.inference_intensity = float(eta)
                for gran in ("batch", "sample"):
                    model.route_selection_mode = gran
                    model.route_granularity = gran
                    row = evaluate_loader(
                        model,
                        loader,
                        device=device,
                        forward_features=forward_features,
                        target_features=target_features,
                        scaler=scaler,
                        null_val=null_val,
                        candidates=candidates,
                        max_batches=args.max_batches,
                    )
                    row.update(
                        {
                            "eval_mode": "intensity",
                            "split": mode,
                            "route": None,
                            "eta": float(eta),
                            "route_selection_mode": gran,
                            "route_granularity": gran,
                            "dataset_size": int(n_ds),
                        }
                    )
                    results.append(row)
                    print(
                        f"[intensity] split={row['split']} eta={eta} gran={gran} "
                        f"mae={row['mae']:.4f} budget={row['budget']:.4f} "
                        f"viol={row['violation_rate']:.4f} "
                        f"hist_sample={row['route_histogram_sample']}"
                    )

    payload = {
        "checkpoint_load": load_report,
        "scaler_path": str(scaler_path),
        "cfg_path": str(cfg_path),
        "null_val": null_val,
        "candidate_routes": [_route_key(r) for r in candidates],
        "results": results,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
