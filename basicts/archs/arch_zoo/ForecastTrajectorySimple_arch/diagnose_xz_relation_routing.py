"""Diagnosis-only upper bound for the explicit X--Z relation encoder.

Route deltas are used only here to test held-out observability.  This probe is
never loaded by or used to initialize the genuine online controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .diagnose_learned_consistency import (
    R3_ROUTES,
    R4_ROUTES,
    build_split,
    report,
)
from .online_xz_relation_policy import LearnedHistoryForecastRelationEncoder
from .run_pipeline import (
    DEFAULT_DATA_DIR,
    WindowDataset,
    load_data,
    make_loader,
    seed_everything,
)
from .run_selector import DEFAULT_BRIDGE_CHECKPOINT, build_frozen_forecaster


class XZRelationBenefitProbe(nn.Module):
    def __init__(self, node_count, source, action_count, width=64):
        super().__init__()
        self.source = int(source)
        self.encoder = LearnedHistoryForecastRelationEncoder(
            node_count, width, dropout=0.1
        )
        self.head = nn.Linear(width, action_count)

    def forward(self, history, current, previous):
        del previous
        current = F.interpolate(
            current.transpose(1, 2), size=12, mode="linear", align_corners=False
        ).transpose(1, 2)
        calendar = torch.zeros(len(history), 24, device=history.device)
        state = torch.cat((history.flatten(1), current.flatten(1), calendar), dim=1)
        return self.head(self.encoder(self.source, state))


def train_probe(train, valid, state, routes, source, device, epochs):
    losses_key = "r3_losses" if state == "z3" else "r4_losses"
    target = train[losses_key][:, 1:] - train[losses_key][:, :1]
    target_mean = target.mean(0)
    target_std = target.std(0, unbiased=False).clamp_min(1e-4)
    model = XZRelationBenefitProbe(
        train["history"].shape[-1], source, len(routes) - 1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    previous = train["z3"] if state == "z4" else train["history"][:, -1:, :].expand(-1, 3, -1)
    dataset = TensorDataset(
        train["history"], train[state], previous,
        (target - target_mean) / target_std,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    best = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for history, current, prior, standardized_target in loader:
            prediction = model(
                history.to(device), current.to(device), prior.to(device)
            )
            raw = prediction * target_std.to(device) + target_mean.to(device)
            sign_target = (
                standardized_target.to(device) * target_std.to(device)
                + target_mean.to(device) < 0
            ).float()
            loss = F.smooth_l1_loss(prediction, standardized_target.to(device))
            loss = loss + 0.2 * F.binary_cross_entropy_with_logits(
                -raw / target_std.to(device), sign_target
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        row = report(model, valid, state, routes, device, target_mean, target_std)
        improved = best is None or row["selected_mae"] < best["selected_mae"]
        if improved:
            best = {**row, "epoch": epoch}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"[{state} epoch {epoch:02d}] selected={row['selected_mae']:.5f} auc={row['gain_auc']}")
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
    train = build_split(forecaster, loader("train"), device, mean, std, args.max_train_batches)
    valid = build_split(forecaster, loader("valid"), device, mean, std, args.max_eval_batches)
    result = {
        "purpose": "diagnosis only; never controller supervision or initialization",
        "forecast_graph_modified": False,
        "inference_safe_inputs": ["ordered_history_X", "actually_reached_Z_r", "explicit_learned_X_Z_contrast"],
        "z3": train_probe(train, valid, "z3", R3_ROUTES, 3, device, args.epochs),
        "z4": train_probe(train, valid, "z4", R4_ROUTES, 4, device, args.epochs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
