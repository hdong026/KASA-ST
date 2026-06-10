#!/usr/bin/env python3
"""Estimate mean absolute output norms for KASA temporal branches."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_cfg(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("kasa_cfg", cfg_path)
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


def load_model(cfg, ckpt_path: Path, device: str):
    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = find_state_dict(ckpt)
    if state is None:
        raise ValueError(f"Could not find model state dict in {ckpt_path}")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_one_batch(cfg, split: str = "val"):
    from basicts.data import TimeSeriesForecastingDataset

    data_cfg = getattr(cfg, split.upper())
    dataset = cfg.DATASET_CLS(
        data_cfg.DATA.DIR,
        data_name=cfg.DATASET_NAME,
        input_len=cfg.DATASET_INPUT_LEN,
        output_len=cfg.DATASET_OUTPUT_LEN,
        mode=split,
        logger=None,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_cfg.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    return next(iter(loader))


def branch_abs_mean(tensor: torch.Tensor) -> float:
    return float(tensor.detach().abs().mean().cpu())


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate KASA branch output norms.")
    parser.add_argument("--cfg", default="examples/KASAST_v2/KASAST_PEMS04.py")
    parser.add_argument("--ckpt", required=True, help="Path to trained checkpoint")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out", default="results/kasa_branch_output_norms.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg_path = ROOT / args.cfg
    ckpt_path = Path(args.ckpt)
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}")
        return 1
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    try:
        cfg = load_cfg(cfg_path)
        model = load_model(cfg, ckpt_path, args.device)
        batch = load_one_batch(cfg, args.split)
        history = batch["inputs"].to(args.device)
        future = batch.get("target")
        if future is not None:
            future = future.to(args.device)

        with torch.no_grad():
            # Recompute branch tensors with the same logic as KASA forward.
            if model.use_pre_temporal_spatial_enhancement:
                spatial_codebook_for_encoder = model.spatial_module.get_enhanced_spatial_embedding(
                    model.spa_codebook
                )
            else:
                spatial_codebook_for_encoder = model.spa_codebook

            main_input = history[..., :3]
            from math import ceil

            in_len_add = ceil(1.0 * model.input_len / model.stride) * model.stride - model.input_len
            if in_len_add:
                main_input_aug = torch.cat(
                    (main_input[:, -1:, :, :].expand(-1, in_len_add, -1, -1), main_input),
                    dim=1,
                )
            else:
                main_input_aug = main_input

            downsamp_input = torch.stack(
                [main_input_aug[:, i :: model.stride, :, :] for i in range(model.stride)],
                dim=1,
            )
            patch_input = (
                main_input_aug.unfold(dimension=1, size=model.patch_len, step=model.patch_len)
                .permute(0, 1, 4, 2, 3)
            )

            patch_predict = model.patch_encoder(patch_input, spatial_codebook=spatial_codebook_for_encoder)
            downsamp_predict = model.downsamp_encoder(downsamp_input, spatial_codebook=spatial_codebook_for_encoder)
            res_input = history[..., 0:1].permute(0, 1, 2, 3)
            res_out = model.residual(res_input)

        row = {
            "checkpoint": str(ckpt_path),
            "split": args.split,
            "patch_abs_mean": branch_abs_mean(patch_predict),
            "downsample_abs_mean": branch_abs_mean(downsamp_predict),
            "residual_abs_mean": branch_abs_mean(res_out),
        }
    except Exception as e:
        print(f"Failed to estimate branch norms: {e}")
        print("TODO: hook-based branch extraction not implemented; skipped.")
        return 1

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(row)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
