#!/usr/bin/env python3
"""Analyze interleaved progressive forecast states (T_k vs S_k) on PeMS04."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs import ChainForecasting
from basicts.data import SCALER_REGISTRY
from basicts.metrics import masked_mae
from basicts.utils import load_pkl

CKPT_ROOT = ROOT / "checkpoints" / "fixed_input_horizon_pems04"
TEMP_CFG_DIR = ROOT / "tmp_configs" / "fixed_input_horizon_pems04"
STAGE_LABELS = {
    12: ["T3", "S1/4", "T6", "S1/2", "T12", "Sall"],
    24: ["T6", "S1/4", "T12", "S1/2", "T24", "Sall"],
    48: ["T12", "S1/4", "T24", "S1/2", "T48", "Sall"],
}


def load_cfg(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("ist_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CFG


def find_state_dict(ckpt_obj) -> dict | None:
    if isinstance(ckpt_obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    return None


def resolve_assets(horizon: int, variant: str, seed: int) -> tuple[Path, Path]:
    ckpt_base = CKPT_ROOT / f"h{horizon}" / f"{variant}_seed{seed}"
    cfg_path = TEMP_CFG_DIR / f"h{horizon}_{variant}_seed{seed}.py"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_base.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {ckpt_base}")

    for run_dir in sorted(ckpt_base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        ckpt = run_dir / "ChainForecasting_best_val_MAE.pt"
        if ckpt.is_file():
            return cfg_path, ckpt
    raise FileNotFoundError(f"No best checkpoint under {ckpt_base}")


def build_dataloader(cfg, split: str = "test") -> DataLoader:
    data_cfg = cfg.TEST if split == "test" else cfg.VAL
    data_file = f"{data_cfg.DATA.DIR}/data_in{cfg.DATASET_INPUT_LEN}_out{cfg.DATASET_OUTPUT_LEN}.pkl"
    index_file = f"{data_cfg.DATA.DIR}/index_in{cfg.DATASET_INPUT_LEN}_out{cfg.DATASET_OUTPUT_LEN}.pkl"
    dataset = cfg.DATASET_CLS(data_file_path=data_file, index_file_path=index_file, mode=split)
    return DataLoader(dataset, batch_size=data_cfg.DATA.BATCH_SIZE, shuffle=False, num_workers=0)


def rescale(scaler: dict, x: torch.Tensor) -> torch.Tensor:
    return SCALER_REGISTRY.get(scaler["func"])(x, **scaler["args"])


def mae_on_pooled(pred: torch.Tensor, target: torch.Tensor, scaler: dict, null_val: float) -> float:
    pred_r = rescale(scaler, pred)
    target_r = rescale(scaler, target)
    return float(masked_mae(pred_r, target_r, null_val=null_val).item())


@torch.no_grad()
def evaluate_states(
    model: ChainForecasting,
    loader: DataLoader,
    scaler: dict,
    null_val: float,
    device: str,
) -> dict:
    chain_lengths = model.chain_lengths
    n = len(chain_lengths)
    temporal_mae = [0.0] * n
    spatial_mae = [0.0] * n
    delta_mae = [0.0] * n
    count = 0
    pred_sall_match = True
    max_pred_diff = 0.0

    for future_data, history_data in loader:
        history_data = history_data.to(device)
        future_data = future_data.to(device)
        target = future_data[..., :1]

        out = model(history_data=history_data, return_all=True)
        temporal_preds = out["temporal_preds"]
        spatial_preds = out["spatial_preds"]
        pred = out["pred"]

        if not torch.allclose(pred, spatial_preds[-1], atol=1e-5, rtol=1e-5):
            pred_sall_match = False
            max_pred_diff = max(max_pred_diff, float((pred - spatial_preds[-1]).abs().max().item()))

        for k, r in enumerate(chain_lengths):
            tgt_k = ChainForecasting.pool_target(target, r)
            temporal_mae[k] += mae_on_pooled(temporal_preds[k], tgt_k, scaler, null_val)
            spatial_mae[k] += mae_on_pooled(spatial_preds[k], tgt_k, scaler, null_val)
        count += 1

    if count == 0:
        raise RuntimeError("Empty dataloader.")

    temporal_mae = [v / count for v in temporal_mae]
    spatial_mae = [v / count for v in spatial_mae]
    delta_mae = [spatial_mae[k] - temporal_mae[k] for k in range(n)]

    return {
        "temporal_mae": temporal_mae,
        "spatial_mae": spatial_mae,
        "delta_mae": delta_mae,
        "pred_equals_sall": pred_sall_match,
        "max_pred_sall_diff": max_pred_diff,
        "chain_lengths": chain_lengths,
    }


def print_report(result: dict, horizon: int) -> None:
    labels = STAGE_LABELS.get(horizon, [])
    print(f"\n=== Interleaved Progressive State Analysis (F={horizon}) ===")
    for k, r in enumerate(result["chain_lengths"]):
        t_label = labels[2 * k] if 2 * k < len(labels) else f"T{r}"
        s_label = labels[2 * k + 1] if 2 * k + 1 < len(labels) else f"S{k}"
        print(
            f"  stage {k+1} (len={r}): "
            f"MAE({t_label})={result['temporal_mae'][k]:.4f}  "
            f"MAE({s_label})={result['spatial_mae'][k]:.4f}  "
            f"delta={result['delta_mae'][k]:+.4f}"
        )
    print(f"  pred == S_all: {result['pred_equals_sall']} "
          f"(max |pred-S_all|={result['max_pred_sall_diff']:.2e})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze IST-FSC interleaved progressive states.")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--variant", default="chain_interleaved_progressive_spatial")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="results/pems04_interleaved_progressive_state_analysis.csv")
    args = parser.parse_args()

    cfg_path, ckpt_path = resolve_assets(args.horizon, args.variant, args.seed)
    cfg = load_cfg(cfg_path)
    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM).to(args.device)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    state = find_state_dict(ckpt)
    if state is None:
        raise ValueError(f"No state dict in {ckpt_path}")
    model.load_state_dict(state, strict=True)
    model.eval()

    if str(cfg.MODEL.PARAM.get("spatial_placement", "")).lower() != "interleaved_progressive":
        print(f"Warning: spatial_placement={cfg.MODEL.PARAM.get('spatial_placement')}")

    scaler = load_pkl(
        f"{cfg.TRAIN.DATA.DIR}/scaler_in{cfg.DATASET_INPUT_LEN}_out{cfg.DATASET_OUTPUT_LEN}.pkl"
    )
    loader = build_dataloader(cfg, split="test")
    result = evaluate_states(model, loader, scaler, cfg.TRAIN.NULL_VAL, args.device)
    print_report(result, args.horizon)

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = STAGE_LABELS.get(args.horizon, [])
    for k, r in enumerate(result["chain_lengths"]):
        rows.append(
            {
                "horizon": args.horizon,
                "variant": args.variant,
                "seed": args.seed,
                "stage_idx": k,
                "chain_len": r,
                "temporal_label": labels[2 * k] if 2 * k < len(labels) else f"T{r}",
                "spatial_label": labels[2 * k + 1] if 2 * k + 1 < len(labels) else f"S{k}",
                "mae_temporal": result["temporal_mae"][k],
                "mae_spatial": result["spatial_mae"][k],
                "delta_mae": result["delta_mae"][k],
            }
        )
    rows.append(
        {
            "horizon": args.horizon,
            "variant": args.variant,
            "seed": args.seed,
            "stage_idx": "final",
            "chain_len": args.horizon,
            "temporal_label": "",
            "spatial_label": "pred==Sall",
            "mae_temporal": "",
            "mae_spatial": int(result["pred_equals_sall"]),
            "delta_mae": result["max_pred_sall_diff"],
        }
    )
    fields = [
        "horizon", "variant", "seed", "stage_idx", "chain_len",
        "temporal_label", "spatial_label", "mae_temporal", "mae_spatial", "delta_mae",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
