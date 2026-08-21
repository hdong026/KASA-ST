"""Train an optional residual correction at native Z6 without changing F2F.

Skip:   START->3->6->12, exactly the mature canonical model.
Refine: START->3->6->[bounded Z6 correction]->12, where the final 6->12
        transition is the same frozen native module.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .forecast_transition_bridge import ForecastTransitionBridge
from .run_pipeline import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_F2F_CHECKPOINT,
    WindowDataset,
    canonical_audit,
    load_data,
    make_loader,
    per_sample_mae,
    prepare_batch,
    seed_everything,
    to_physical,
)
from .run_selector import DEFAULT_BRIDGE_CHECKPOINT, build_frozen_forecaster


class OptionalZ6Refiner(nn.Module):
    def __init__(self, model, correction_limit: float = 0.5):
        super().__init__()
        f2f = model.f2f
        self.bridge = ForecastTransitionBridge(
            source_resolution=6,
            target_resolution=6,
            correction_limit=correction_limit,
            input_len=f2f.input_len,
            patch_len=f2f.patch_len,
            stride=f2f.stride,
            td_size=f2f.td_size,
            dw_size=f2f.dw_size,
            td_codebook=f2f.td_codebook,
            dw_codebook=f2f.dw_codebook,
            spa_codebook=f2f.spa_codebook,
            if_time_in_day=f2f.if_time_in_day,
            if_day_in_week=f2f.if_day_in_week,
            if_spatial=f2f.if_spatial,
            d_d=f2f.d_d,
            d_td=f2f.d_td,
            d_dw=f2f.d_dw,
            d_spa=f2f.d_spa,
            num_layer=f2f.num_layer,
            use_patch_branch=f2f.use_patch_branch,
            use_downsample_branch=f2f.use_downsample_branch,
            use_linear_residual_branch=f2f.use_linear_residual_branch,
            patch_data_input_mode=f2f.patch_data_input_mode,
            patch_embedding_mode=f2f.patch_embedding_mode,
            patch_feature_dim=f2f.patch_feature_dim,
        )

    def forward(self, history, z6, spatial_codebook):
        return self.bridge(
            history, previous_forecast=z6, spatial_codebook=spatial_codebook
        )


@torch.no_grad()
def canonical_states(model, history):
    z3 = model.execute_transition(history, None, 3, None)
    z6 = model.execute_transition(history, 3, 6, z3)
    z12 = model.execute_transition(history, 6, 12, z6)
    return z6, model.finalize_forecast(z12, history)


def refined_prediction(model, refiner, history, z6):
    refined_z6 = refiner(history, z6, model.f2f._spatial_codebook())
    z12 = model.execute_transition(history, 6, 12, refined_z6)
    return refined_z6, model.finalize_forecast(z12, history)


def train_epoch(model, refiner, loader, optimizer, device, mean, std, args):
    refiner.train()
    started = time.perf_counter()
    totals = {"loss": 0.0, "refined": 0.0, "harm": 0.0}
    samples = 0
    for batch_index, batch in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        history, target = prepare_batch(batch, device)
        target_raw = to_physical(target, mean, std)
        with torch.no_grad():
            z6, canonical = canonical_states(model, history)
            canonical_loss = per_sample_mae(
                to_physical(canonical, mean, std), target_raw, 0.0
            )
        _, refined = refined_prediction(model, refiner, history, z6)
        refined_loss = per_sample_mae(
            to_physical(refined, mean, std), target_raw, 0.0
        )
        harm = torch.relu(refined_loss - canonical_loss).mean()
        loss = refined_loss.mean() + args.harm_weight * harm
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in refiner.parameters() if p.requires_grad], 5.0
        )
        optimizer.step()
        count = len(history)
        samples += count
        totals["loss"] += float(loss.detach()) * count
        totals["refined"] += float(refined_loss.mean().detach()) * count
        totals["harm"] += float(harm.detach()) * count
    return {
        **{key: value / samples for key, value in totals.items()},
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate(model, refiner, loader, device, mean, std, args):
    refiner.eval()
    canonical_parts, refined_parts = [], []
    correction_parts = []
    for batch_index, batch in enumerate(loader):
        if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
            break
        history, target = prepare_batch(batch, device)
        target_raw = to_physical(target, mean, std)
        z6, canonical = canonical_states(model, history)
        refined_z6, refined = refined_prediction(model, refiner, history, z6)
        canonical_parts.append(
            per_sample_mae(to_physical(canonical, mean, std), target_raw, 0.0).cpu()
        )
        refined_parts.append(
            per_sample_mae(to_physical(refined, mean, std), target_raw, 0.0).cpu()
        )
        correction_parts.append((refined_z6 - z6).abs().mean((1, 2, 3)).cpu())
    canonical = torch.cat(canonical_parts)
    refined = torch.cat(refined_parts)
    correction = torch.cat(correction_parts)
    gain = canonical - refined
    oracle = torch.minimum(canonical, refined)
    harm = torch.relu(-gain)
    return {
        "canonical_mae": float(canonical.mean()),
        "refined_fixed_mae": float(refined.mean()),
        "two_action_oracle_mae": float(oracle.mean()),
        "oracle_gain_vs_canonical": float(canonical.mean() - oracle.mean()),
        "refinement_gain_mean": float(gain.mean()),
        "refinement_benefit_fraction": float((gain > 0).float().mean()),
        "mean_harm": float(harm.mean()),
        "mean_abs_z6_correction": float(correction.mean()),
        "selection_objective": float(
            refined.mean() + args.harm_weight * harm.mean()
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--f2f-checkpoint", type=Path, default=DEFAULT_F2F_CHECKPOINT)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--harm-weight", type=float, default=2.0)
    parser.add_argument("--correction-limit", type=float, default=0.1)
    parser.add_argument("--bridge-correction-limit", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.workers = 0
        args.max_train_batches = 2
        args.max_eval_batches = 2
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data, indices, mean, std = load_data(args.data_dir)
    datasets = {split: WindowDataset(data, indices[split]) for split in ("train", "valid")}

    def loader(split, shuffle=False):
        return make_loader(
            datasets[split], batch_size=args.batch_size, shuffle=shuffle,
            workers=args.workers, device=device, seed=args.seed,
        )

    model = build_frozen_forecaster(args, device)
    canonical_audit(model, loader("valid"), device)
    refiner = OptionalZ6Refiner(model, args.correction_limit).to(device)
    canonical_ids = {id(parameter) for parameter in model.parameters()}
    trainable = [
        parameter for parameter in refiner.parameters()
        if id(parameter) not in canonical_ids
    ]
    refiner.requires_grad_(False)
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    best_path = args.output_dir / "z6_refiner_best.pt"
    best_score = float("inf")
    stale = 0
    history_log = []
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(
            model, refiner, loader("train", True), optimizer,
            device, mean, std, args,
        )
        valid = evaluate(model, refiner, loader("valid"), device, mean, std, args)
        improved = valid["selection_objective"] < best_score
        if improved:
            best_score = valid["selection_objective"]
            torch.save(
                {
                    "epoch": epoch,
                    "refiner_state_dict": refiner.state_dict(),
                    "validation": valid,
                    "canonical_f2f_frozen": True,
                },
                best_path,
            )
            stale = 0
        else:
            stale += 1
        history_log.append({"epoch": epoch, "train": train, "valid": valid, "best": improved})
        (args.output_dir / "history.json").write_text(
            json.dumps(history_log, indent=2), encoding="utf-8"
        )
        print(
            f"[epoch {epoch:02d}] seconds={train['seconds']:.1f} "
            f"canonical={valid['canonical_mae']:.4f} refined={valid['refined_fixed_mae']:.4f} "
            f"oracle={valid['two_action_oracle_mae']:.4f} "
            f"benefit={valid['refinement_benefit_fraction']:.3f} best={improved}"
        )
        if stale >= args.patience:
            break
    checkpoint = torch.load(best_path, map_location=device)
    refiner.load_state_dict(checkpoint["refiner_state_dict"], strict=True)
    canonical_audit(model, loader("valid"), device)
    final = evaluate(model, refiner, loader("valid"), device, mean, std, args)
    report = {
        "method": "optional_bounded_z6_residual_then_frozen_native_6_to_12",
        "checkpoint": str(best_path),
        "test_evaluated": False,
        "validation": final,
        "canonical_exact": {"torch_equal": True, "max_abs": 0.0},
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "valid_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
