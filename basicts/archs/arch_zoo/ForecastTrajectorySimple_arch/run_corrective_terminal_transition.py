"""Train a protected residual-corrective alternative for the terminal 6->12 edge.

Skip executes the original native 6->12 transition exactly.  Refine first
executes that same frozen transition and then predicts a bounded correction
from structural relations among X, reached Z6, and the provisional Z12.  Only
the new correction module is trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

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


class CorrectiveTerminalTransition(nn.Module):
    """Node-shared structural residual with a small identity embedding."""

    def __init__(self, node_count: int, width: int = 64, correction_limit: float = 0.1):
        super().__init__()
        self.correction_limit = float(correction_limit)
        self.node_embedding = nn.Embedding(node_count, 8)
        # X, provisional Z12, interpolated Z6, X-Z12 mismatch, and two
        # first-difference sequences provide explicit continuity/shape signals.
        feature_dim = 12 * 6 + 8
        self.corrector = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 12),
        )
        nn.init.zeros_(self.corrector[-1].weight)
        nn.init.zeros_(self.corrector[-1].bias)

    def forward(self, history, z6, provisional_z12):
        x = history[..., 0].transpose(1, 2)
        z6n = z6[..., 0].transpose(1, 2)
        z6i = F.interpolate(z6n, size=12, mode="linear", align_corners=False)
        z12 = provisional_z12[..., 0].transpose(1, 2)
        dx = F.pad(x[..., 1:] - x[..., :-1], (1, 0))
        dz = F.pad(z12[..., 1:] - z12[..., :-1], (1, 0))
        embedding = self.node_embedding.weight.unsqueeze(0).expand(len(x), -1, -1)
        features = torch.cat((x, z12, z6i, z12 - x, dx, dz, embedding), dim=-1)
        correction = self.correction_limit * torch.tanh(self.corrector(features))
        return provisional_z12 + correction.transpose(1, 2).unsqueeze(-1), correction


@torch.no_grad()
def reached_states(model, history):
    z3 = model.execute_transition(history, None, 3, None)
    z6 = model.execute_transition(history, 3, 6, z3)
    z12 = model.execute_transition(history, 6, 12, z6)
    canonical = model.finalize_forecast(z12, history)
    return z3, z6, z12, canonical


def corrective_prediction(model, transition, history, z6, provisional_z12):
    corrected_z12, correction = transition(history, z6, provisional_z12)
    return corrected_z12, model.finalize_forecast(corrected_z12, history), correction


def train_epoch(model, transition, loader, optimizer, device, mean, std, args):
    transition.train()
    started = time.perf_counter()
    totals = {"objective": 0.0, "corrected": 0.0, "harm": 0.0}
    samples = 0
    for batch_index, batch in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        history, target = prepare_batch(batch, device)
        target_raw = to_physical(target, mean, std)
        with torch.no_grad():
            _, z6, z12, canonical = reached_states(model, history)
            canonical_loss = per_sample_mae(to_physical(canonical, mean, std), target_raw, 0.0)
        _, corrected, correction = corrective_prediction(model, transition, history, z6, z12)
        corrected_loss = per_sample_mae(to_physical(corrected, mean, std), target_raw, 0.0)
        harm = torch.relu(corrected_loss - canonical_loss).mean()
        objective = (
            corrected_loss.mean()
            + args.harm_weight * harm
            + args.correction_weight * correction.abs().mean() * std
        )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(transition.parameters(), 5.0)
        optimizer.step()
        count = len(history)
        samples += count
        totals["objective"] += float(objective.detach()) * count
        totals["corrected"] += float(corrected_loss.mean().detach()) * count
        totals["harm"] += float(harm.detach()) * count
    return {
        **{key: value / samples for key, value in totals.items()},
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate(model, transition, loader, device, mean, std, max_batches=None):
    transition.eval()
    canonical_parts, corrected_parts, correction_parts = [], [], []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target = prepare_batch(batch, device)
        target_raw = to_physical(target, mean, std)
        _, z6, z12, canonical = reached_states(model, history)
        _, corrected, correction = corrective_prediction(model, transition, history, z6, z12)
        canonical_parts.append(per_sample_mae(to_physical(canonical, mean, std), target_raw, 0.0).cpu())
        corrected_parts.append(per_sample_mae(to_physical(corrected, mean, std), target_raw, 0.0).cpu())
        correction_parts.append(correction.abs().mean((1, 2)).cpu())
    canonical = torch.cat(canonical_parts)
    corrected = torch.cat(corrected_parts)
    correction = torch.cat(correction_parts)
    gain = canonical - corrected
    return {
        "canonical_mae": float(canonical.mean()),
        "corrected_fixed_mae": float(corrected.mean()),
        "two_action_oracle_mae": float(torch.minimum(canonical, corrected).mean()),
        "oracle_gain_vs_canonical": float(canonical.mean() - torch.minimum(canonical, corrected).mean()),
        "correction_gain_mean": float(gain.mean()),
        "correction_benefit_fraction": float((gain > 0).float().mean()),
        "mean_harm": float(torch.relu(-gain).mean()),
        "mean_abs_normalized_correction": float(correction.mean()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--f2f-checkpoint", type=Path, default=DEFAULT_F2F_CHECKPOINT)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--harm-weight", type=float, default=0.25)
    parser.add_argument("--correction-weight", type=float, default=0.01)
    parser.add_argument("--correction-limit", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=64)
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
        args.epochs, args.workers = 1, 0
        args.max_train_batches = args.max_eval_batches = 2
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data, indices, mean, std = load_data(args.data_dir)
    datasets = {split: WindowDataset(data, indices[split]) for split in ("train", "valid")}

    def loader(split, shuffle=False):
        return make_loader(datasets[split], batch_size=args.batch_size, shuffle=shuffle,
                           workers=args.workers, device=device, seed=args.seed)

    model = build_frozen_forecaster(args, device)
    canonical_audit(model, loader("valid"), device)
    transition = CorrectiveTerminalTransition(
        node_count=data.shape[1], width=args.width, correction_limit=args.correction_limit
    ).to(device)
    optimizer = torch.optim.AdamW(transition.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_path = args.output_dir / "corrective_terminal_best.pt"
    initial = evaluate(model, transition, loader("valid"), device, mean, std, args.max_eval_batches)
    best_mae = initial["corrected_fixed_mae"]
    torch.save({"epoch": 0, "transition_state_dict": transition.state_dict(), "validation": initial}, best_path)
    stale = 0
    history_log = [{"epoch": 0, "valid": initial, "best": True}]
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(model, transition, loader("train", True), optimizer, device, mean, std, args)
        valid = evaluate(model, transition, loader("valid"), device, mean, std, args.max_eval_batches)
        improved = valid["corrected_fixed_mae"] < best_mae
        if improved:
            best_mae = valid["corrected_fixed_mae"]
            torch.save({"epoch": epoch, "transition_state_dict": transition.state_dict(), "validation": valid}, best_path)
            stale = 0
        else:
            stale += 1
        history_log.append({"epoch": epoch, "train": train, "valid": valid, "best": improved})
        (args.output_dir / "history.json").write_text(json.dumps(history_log, indent=2), encoding="utf-8")
        print(f"[epoch {epoch:02d}] seconds={train['seconds']:.1f} canonical={valid['canonical_mae']:.4f} corrected={valid['corrected_fixed_mae']:.4f} oracle={valid['two_action_oracle_mae']:.4f} benefit={valid['correction_benefit_fraction']:.3f} best={improved}")
        if stale >= args.patience:
            break
    checkpoint = torch.load(best_path, map_location=device)
    transition.load_state_dict(checkpoint["transition_state_dict"], strict=True)
    canonical_audit(model, loader("valid"), device)
    final = evaluate(model, transition, loader("valid"), device, mean, std, args.max_eval_batches)
    report = {
        "method": "protected_residual_corrective_terminal_6_to_12",
        "best_epoch": checkpoint["epoch"],
        "checkpoint": str(best_path),
        "test_evaluated": False,
        "validation": final,
        "canonical_exact": {"torch_equal": True, "max_abs": 0.0},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "valid_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
