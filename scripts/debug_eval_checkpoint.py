#!/usr/bin/env python3
"""Direct checkpoint evaluation for val/test split debugging."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs import ChainForecasting
from basicts.data import SCALER_REGISTRY
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.utils import load_pkl


def load_cfg(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("debug_eval_cfg", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.CFG


def resolve_ckpt(cfg, ckpt_arg: str | None) -> Path:
    if ckpt_arg:
        path = Path(ckpt_arg)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    ckpt_dir = Path(cfg.TRAIN.CKPT_SAVE_DIR)
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    candidates = list(ckpt_dir.rglob("*best_val_MAE.pt"))
    if not candidates:
        raise FileNotFoundError(f"No best_val checkpoint under {ckpt_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_checkpoint(model: ChainForecasting, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[warn] ignored unexpected checkpoint keys ({len(unexpected)}): {unexpected[:5]}...")
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:5]}...")


def rescale(tensor: torch.Tensor, scaler: dict) -> torch.Tensor:
    return SCALER_REGISTRY.get(scaler["func"])(tensor, **scaler["args"])


def load_batches(cfg, split: str, batch_size: int, max_batches: int):
    data_dir = Path(cfg.TRAIN.DATA.DIR)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    in_len = int(cfg.DATASET_INPUT_LEN)
    out_len = int(cfg.DATASET_OUTPUT_LEN)
    data_obj = load_pkl(str(data_dir / f"data_in{in_len}_out{out_len}.pkl"))
    index_obj = load_pkl(str(data_dir / f"index_in{in_len}_out{out_len}.pkl"))
    data = torch.from_numpy(data_obj["processed_data"]).float()
    rows = index_obj[split]
    limit = len(rows) if max_batches < 0 else min(len(rows), max_batches * batch_size)
    batches = []
    for start in range(0, limit, batch_size):
        chunk = rows[start : start + batch_size]
        if not chunk:
            break
        hist_list, fut_list = [], []
        for r in chunk:
            s, m, e = int(r[0]), int(r[1]), int(r[2])
            hist_list.append(data[s:m])
            fut_list.append(data[m:e])
        batches.append((torch.stack(fut_list), torch.stack(hist_list)))
        if max_batches >= 0 and len(batches) >= max_batches:
            break
    return batches


def eval_split(model, batches, cfg, scaler, device, null_val, split: str, ckpt: Path):
    forward_features = list(cfg.MODEL.FORWARD_FEATURES)
    target_features = list(cfg.MODEL.TARGET_FEATURES)
    preds, targets = [], []
    output_types = set()
    model.eval()
    with torch.no_grad():
        for fut, hist in batches:
            hist_in = hist[..., forward_features].to(device)
            fut_t = fut[..., target_features].to(device)
            out = model(hist_in, return_all=False, train=False)
            output_types.add(type(out).__name__)
            if isinstance(out, dict):
                pred = out.get("pred") or out.get("prediction")
            else:
                pred = out
            preds.append(pred.cpu())
            targets.append(fut_t.cpu())
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    pred_r = rescale(pred_all, scaler)
    target_r = rescale(target_all, scaler)
    mae = float(masked_mae(pred_r, target_r, null_val=null_val).item())
    rmse = float(masked_rmse(pred_r, target_r, null_val=null_val).item())
    mape = float(masked_mape(pred_r, target_r, null_val=null_val).item())
    return {
        "split": split,
        "checkpoint": str(ckpt),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "num_samples": int(pred_all.shape[0]),
        "prediction_shape": list(pred_all.shape),
        "target_shape": list(target_all.shape),
        "output_types": sorted(output_types),
        "pred_norm_mean": float(pred_all.mean()),
        "pred_norm_std": float(pred_all.std(unbiased=False)),
        "target_norm_mean": float(target_all.mean()),
        "target_norm_std": float(target_all.std(unbiased=False)),
        "pred_rescaled_mean": float(pred_r.mean()),
        "pred_rescaled_std": float(pred_r.std(unbiased=False)),
        "target_rescaled_mean": float(target_r.mean()),
        "target_rescaled_std": float(target_r.std(unbiased=False)),
        "scaler_func": scaler["func"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug evaluate checkpoint on val/test.")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max_batches", type=int, default=-1)
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_cfg(cfg_path)
    ckpt = resolve_ckpt(cfg, args.ckpt)
    device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")

    data_dir = Path(cfg.TRAIN.DATA.DIR)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    in_len = int(cfg.DATASET_INPUT_LEN)
    out_len = int(cfg.DATASET_OUTPUT_LEN)
    scaler = load_pkl(str(data_dir / f"scaler_in{in_len}_out{out_len}.pkl"))
    null_val = float(cfg.TRAIN.NULL_VAL)

    split_key = {"train": "TRAIN", "valid": "VAL", "test": "TEST"}[args.split]
    batch_size = int(getattr(cfg, split_key).DATA.BATCH_SIZE)

    model = ChainForecasting(**dict(cfg.MODEL.PARAM))
    load_checkpoint(model, ckpt)
    model.to(device)

    batches = load_batches(cfg, args.split, batch_size, args.max_batches)
    result = eval_split(model, batches, cfg, scaler, device, null_val, args.split, ckpt)
    for k, v in result.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
