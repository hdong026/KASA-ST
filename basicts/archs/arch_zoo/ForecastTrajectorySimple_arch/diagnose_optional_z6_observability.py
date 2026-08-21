"""Diagnosis-only observability probe for the optional native-Z6 refiner.

The target is the signed loss change produced by executing the refiner and the
unchanged native 6->12 transition.  Targets are used only to test whether the
inference-safe state (X, Z3, Z6, and Z6-I(Z3)) contains selection information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .diagnose_learned_consistency import HistoryForecastConsistency
from .run_optional_z6_refiner import (
    OptionalZ6Refiner,
    canonical_states,
    refined_prediction,
)
from .run_corrective_terminal_transition import (
    CorrectiveTerminalTransition,
    corrective_prediction as terminal_corrective_prediction,
    reached_states as terminal_reached_states,
)
from .run_pipeline import (
    DEFAULT_DATA_DIR,
    WindowDataset,
    load_data,
    make_loader,
    per_sample_mae,
    prepare_batch,
    seed_everything,
    to_physical,
)
from .run_selector import DEFAULT_BRIDGE_CHECKPOINT, build_frozen_forecaster


@torch.inference_mode()
def build_split(model, refiner, loader, device, mean, std, max_batches=None, mechanism="z6"):
    histories, z3s, z6s, losses = [], [], [], []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target = prepare_batch(batch, device)
        target_raw = to_physical(target, mean, std)
        if mechanism == "terminal":
            z3, z6, z12, canonical = terminal_reached_states(model, history)
            _, refined, _ = terminal_corrective_prediction(
                model, refiner, history, z6, z12
            )
        else:
            z3 = model.execute_transition(history, None, 3, None)
            z6, canonical = canonical_states(model, history)
            _, refined = refined_prediction(model, refiner, history, z6)
        canonical_loss = per_sample_mae(
            to_physical(canonical, mean, std), target_raw, 0.0
        )
        refined_loss = per_sample_mae(
            to_physical(refined, mean, std), target_raw, 0.0
        )
        histories.append(history[..., 0].cpu())
        z3s.append(z3[..., 0].cpu())
        z6s.append(z6[..., 0].cpu())
        losses.append(torch.stack((canonical_loss, refined_loss), 1).cpu())
    return {
        "history": torch.cat(histories),
        "z3": torch.cat(z3s),
        "z6": torch.cat(z6s),
        "losses": torch.cat(losses),
    }


@torch.no_grad()
def predict(model, data, device):
    model.eval()
    output = []
    loader = DataLoader(
        TensorDataset(data["history"], data["z6"], data["z3"]), batch_size=256
    )
    for history, z6, z3 in loader:
        output.append(model(history.to(device), z6.to(device), z3.to(device)).cpu())
    return torch.cat(output).squeeze(1)


def metrics(predicted_delta, losses, threshold=0.0):
    true_delta = losses[:, 1] - losses[:, 0]
    true_gain = -true_delta
    predicted_gain = -predicted_delta
    beneficial = true_gain > 0
    selected = predicted_delta < threshold
    selected_loss = torch.where(selected, losses[:, 1], losses[:, 0])
    count = max(1, round(0.1 * len(losses)))
    true_top = set(torch.topk(true_gain, count).indices.tolist())
    predicted_top = set(torch.topk(predicted_gain, count).indices.tolist())
    auc = float(roc_auc_score(beneficial.numpy(), predicted_gain.numpy()))
    pearson = float(np.corrcoef(true_gain.numpy(), predicted_gain.numpy())[0, 1])
    spearman = float(spearmanr(true_gain.numpy(), predicted_gain.numpy()).statistic)
    tp = int((selected & beneficial).sum())
    return {
        "canonical_mae": float(losses[:, 0].mean()),
        "refined_fixed_mae": float(losses[:, 1].mean()),
        "oracle_mae": float(losses.min(1).values.mean()),
        "selected_mae": float(selected_loss.mean()),
        "gain_vs_canonical": float(losses[:, 0].mean() - selected_loss.mean()),
        "gain_pearson": pearson,
        "gain_spearman": spearman,
        "gain_auc": auc,
        "benefit_fraction": float(beneficial.float().mean()),
        "selected_fraction": float(selected.float().mean()),
        "selected_precision": tp / max(1, int(selected.sum())),
        "selected_recall": tp / max(1, int(beneficial.sum())),
        "top10_overlap": len(true_top.intersection(predicted_top)) / count,
        "threshold": float(threshold),
    }


def train_probe(train, valid, device, epochs):
    target = train["losses"][:, 1] - train["losses"][:, 0]
    target_mean = target.mean()
    target_std = target.std(unbiased=False).clamp_min(1e-4)
    model = HistoryForecastConsistency(train["history"].shape[-1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    dataset = TensorDataset(
        train["history"], train["z6"], train["z3"],
        ((target - target_mean) / target_std).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    best = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for history, z6, z3, standardized_target in loader:
            prediction = model(history.to(device), z6.to(device), z3.to(device))
            raw_prediction = prediction * target_std.to(device) + target_mean.to(device)
            sign_target = (standardized_target.to(device) < -target_mean.to(device) / target_std.to(device)).float()
            loss = F.smooth_l1_loss(prediction, standardized_target.to(device))
            loss = loss + 0.2 * F.binary_cross_entropy_with_logits(
                -raw_prediction / target_std.to(device), sign_target
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        standardized = predict(model, valid, device)
        predicted_delta = standardized * target_std + target_mean
        # Report both the scientifically fixed zero threshold and the best
        # validation threshold as an optimistic information upper bound.
        zero = metrics(predicted_delta, valid["losses"], 0.0)
        candidates = torch.quantile(
            predicted_delta, torch.linspace(0, 1, 101)
        ).unique().tolist()
        threshold_rows = [metrics(predicted_delta, valid["losses"], t) for t in candidates]
        calibrated = min(threshold_rows, key=lambda row: row["selected_mae"])
        improved = best is None or calibrated["selected_mae"] < best["calibrated"]["selected_mae"]
        if improved:
            best = {"epoch": epoch, "zero_threshold": zero, "calibrated": calibrated}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"[epoch {epoch:02d}] zero={zero['selected_mae']:.5f} calibrated={calibrated['selected_mae']:.5f} auc={zero['gain_auc']:.3f}")
        if stale >= 8:
            break
    return best


def parse_args():
    from .run_online_sequential_rl import parse_args as online_parse_args
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--f2f-checkpoint", type=Path)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE_CHECKPOINT)
    parser.add_argument("--mechanism", choices=("z6", "terminal"), default="z6")
    parser.add_argument("--refiner-checkpoint", type=Path)
    parser.add_argument("--terminal-checkpoint", type=Path)
    parser.add_argument("--correction-limit", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--bridge-correction-limit", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = sys.argv
    try:
        sys.argv = [old[0]]
        defaults = online_parse_args()
    finally:
        sys.argv = old
    for key in ("config", "f2f_checkpoint"):
        if getattr(args, key) is None:
            setattr(args, key, getattr(defaults, key))
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    data, indices, mean, std = load_data(args.data_dir)
    datasets = {split: WindowDataset(data, indices[split]) for split in ("train", "valid")}

    def loader(split):
        return make_loader(
            datasets[split], batch_size=args.batch_size, shuffle=False,
            workers=args.workers, device=device, seed=args.seed,
        )

    forecaster = build_frozen_forecaster(args, device)
    if args.mechanism == "terminal":
        if args.terminal_checkpoint is None:
            raise ValueError("--terminal-checkpoint is required for terminal mechanism")
        refiner = CorrectiveTerminalTransition(
            data.shape[1], args.width, args.correction_limit
        ).to(device)
        checkpoint = torch.load(args.terminal_checkpoint, map_location=device)
        refiner.load_state_dict(checkpoint["transition_state_dict"], strict=True)
    else:
        if args.refiner_checkpoint is None:
            raise ValueError("--refiner-checkpoint is required for z6 mechanism")
        refiner = OptionalZ6Refiner(forecaster, args.correction_limit).to(device)
        checkpoint = torch.load(args.refiner_checkpoint, map_location=device)
        refiner.load_state_dict(checkpoint["refiner_state_dict"], strict=True)
    refiner.requires_grad_(False).eval()
    train = build_split(forecaster, refiner, loader("train"), device, mean, std, args.max_train_batches, args.mechanism)
    valid = build_split(forecaster, refiner, loader("valid"), device, mean, std, args.max_eval_batches, args.mechanism)
    result = {
        "purpose": "diagnosis only; route outcomes never train the online controller",
        "inference_safe_inputs": ["history_X", "reached_Z3", "reached_Z6", "Z6_minus_interpolated_Z3"],
        "target_available_at_inference": False,
        "mechanism": args.mechanism,
        "validation": train_probe(train, valid, device, args.epochs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
