"""Learned X--Z/Delta-Z observability gate for sequential refinement.

Route losses are diagnosis-only labels.  The estimator input contains raw
history, the actually reached forecast, and its change from the previous state
(persistence at START->3).  It never sees Y or an unexecuted future forecast.
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

from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)

from .run_pipeline import (
    DEFAULT_DATA_DIR,
    WindowDataset,
    load_data,
    make_loader,
    per_sample_mae,
    prepare_batch,
    route_name,
    seed_everything,
    to_physical,
)
from .run_selector import DEFAULT_BRIDGE_CHECKPOINT, build_frozen_forecaster


R3_ROUTES = ((3, 6, 12), (3, 12), (3, 4, 6, 12), (3, 4, 12))
R4_ROUTES = ((3, 4, 6, 12), (3, 4, 12))


@torch.inference_mode()
def build_split(model, loader, device, mean, std, max_batches=None):
    histories, z3s, z4s = [], [], []
    r3_losses, r4_losses = [], []
    all_routes = tuple(dict.fromkeys(R3_ROUTES + R4_ROUTES))
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target = prepare_batch(batch, device)
        target_raw = to_physical(target, mean, std)
        z3 = model.execute_transition(history, None, 3, None)
        z4 = model.execute_transition(history, 3, 4, z3)
        losses = {}
        for route in all_routes:
            prediction = model.execute_trajectory(history, route)["pred"]
            losses[route] = per_sample_mae(
                to_physical(prediction, mean, std), target_raw, 0.0
            )
        histories.append(history[..., 0].cpu())
        z3s.append(z3[..., 0].cpu())
        z4s.append(z4[..., 0].cpu())
        r3_losses.append(torch.stack([losses[route] for route in R3_ROUTES], 1).cpu())
        r4_losses.append(torch.stack([losses[route] for route in R4_ROUTES], 1).cpu())
    return {
        "history": torch.cat(histories),
        "z3": torch.cat(z3s),
        "z4": torch.cat(z4s),
        "r3_losses": torch.cat(r3_losses),
        "r4_losses": torch.cat(r4_losses),
    }


class HistoryForecastConsistency(nn.Module):
    """Shared shape encoder plus learned low-rank spatial mismatch modes."""

    def __init__(self, node_count: int, action_count: int, width: int = 64):
        super().__init__()
        self.node_count = int(node_count)
        shape_dim = 24
        self.shape_encoder = nn.Sequential(
            nn.LayerNorm(12),
            nn.Linear(12, shape_dim),
            nn.SiLU(),
            nn.Linear(shape_dim, shape_dim),
            nn.SiLU(),
        )
        # Identity-sensitive spatial modes can represent changing dependency
        # patterns without materializing an N x N correlation matrix.
        self.spatial_projection = nn.Linear(node_count, 8, bias=False)
        node_input = shape_dim * 5
        self.node_encoder = nn.Sequential(
            nn.LayerNorm(node_input), nn.Linear(node_input, width), nn.SiLU()
        )
        global_input = width * 4 + 3 * 12 * 8
        self.head = nn.Sequential(
            nn.LayerNorm(global_input),
            nn.Linear(global_input, width),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(width, action_count),
        )

    def forward(
        self, history: torch.Tensor, current: torch.Tensor, previous: torch.Tensor
    ) -> torch.Tensor:
        # Align past and future shapes to twelve samples. Tensors arrive as
        # [B,T,N]; the shared encoder sees each sensor's ordered shape.
        current = F.interpolate(
            current.transpose(1, 2), size=12, mode="linear", align_corners=False
        ).transpose(1, 2)
        previous = F.interpolate(
            previous.transpose(1, 2), size=12, mode="linear", align_corners=False
        ).transpose(1, 2)
        delta = current - previous
        sequences = torch.stack(
            (history, current, previous, delta), dim=2
        ).permute(0, 3, 2, 1)
        encoded = self.shape_encoder(sequences)
        hist, forecast, prior, change = encoded.unbind(dim=2)
        mismatch = forecast - hist
        interaction = forecast * hist
        nodes = self.node_encoder(
            torch.cat((hist, forecast, mismatch, interaction, change), dim=-1)
        )
        pooled = torch.cat(
            (
                nodes.mean(1),
                nodes.std(1, unbiased=False),
                nodes.amax(1),
                nodes.amin(1),
            ),
            dim=1,
        )
        spatial = self.spatial_projection(
            torch.stack((history, current, delta), dim=1)
        ).flatten(1)
        return self.head(torch.cat((pooled, spatial), dim=1))


def previous_for_state(history, current_resolution, previous):
    if previous is not None:
        return previous
    return history[:, -1:, :].expand(-1, current_resolution, -1)


@torch.no_grad()
def report(model, data, state, routes, device, target_mean, target_std):
    model.eval()
    outputs = []
    loader = DataLoader(
        TensorDataset(data["history"], data[state], data.get("z3", data[state])),
        batch_size=256,
    )
    for history, current, stored_previous in loader:
        previous = (
            history[:, -1:, :].expand(-1, current.shape[1], -1)
            if state == "z3"
            else stored_previous
        )
        outputs.append(
            model(history.to(device), current.to(device), previous.to(device)).cpu()
        )
    standardized = torch.cat(outputs)
    predicted_delta = standardized * target_std + target_mean
    predicted = torch.cat((torch.zeros(len(predicted_delta), 1), predicted_delta), 1)
    losses = data["r3_losses" if state == "z3" else "r4_losses"]
    selected = predicted.argmin(1)
    rows = torch.arange(len(selected))
    selected_loss = losses[rows, selected]
    true_gain = losses[:, 0] - losses.min(1).values
    predicted_gain = -predicted.min(1).values
    positive = true_gain > 0
    auc = None
    if bool(positive.any()) and bool((~positive).any()):
        auc = float(roc_auc_score(positive.numpy(), predicted_gain.numpy()))
    count = max(1, round(0.1 * len(losses)))
    true_top = set(torch.topk(true_gain, count).indices.tolist())
    predicted_top = set(torch.topk(predicted_gain, count).indices.tolist())
    counts = torch.bincount(selected, minlength=len(routes))
    return {
        "canonical_mae": float(losses[:, 0].mean()),
        "oracle_mae": float(losses.min(1).values.mean()),
        "selected_mae": float(selected_loss.mean()),
        "gain_vs_canonical": float(losses[:, 0].mean() - selected_loss.mean()),
        "gain_pearson": float(np.corrcoef(true_gain.numpy(), predicted_gain.numpy())[0, 1]),
        "gain_spearman": float(spearmanr(true_gain.numpy(), predicted_gain.numpy()).statistic),
        "gain_auc": auc,
        "top10_overlap": len(true_top.intersection(predicted_top)) / count,
        "route_counts": {
            route_name(route): int(counts[index]) for index, route in enumerate(routes)
        },
    }


def train_probe(train, valid, state, routes, device, epochs=40):
    losses_key = "r3_losses" if state == "z3" else "r4_losses"
    target = train[losses_key][:, 1:] - train[losses_key][:, :1]
    target_mean = target.mean(0)
    target_std = target.std(0, unbiased=False).clamp_min(1e-4)
    model = HistoryForecastConsistency(
        train["history"].shape[-1], len(routes) - 1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    previous = (
        train["history"][:, -1:, :].expand(-1, train[state].shape[1], -1)
        if state == "z3"
        else train["z3"]
    )
    dataset = TensorDataset(
        train["history"], train[state], previous, (target - target_mean) / target_std
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
    best = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for history, current, prior, standardized_target in loader:
            prediction = model(
                history.to(device), current.to(device), prior.to(device)
            )
            regression = F.smooth_l1_loss(
                prediction, standardized_target.to(device)
            )
            raw_prediction = prediction * target_std.to(device) + target_mean.to(device)
            sign_target = (standardized_target.to(device) * target_std.to(device) + target_mean.to(device) < 0).float()
            classification = F.binary_cross_entropy_with_logits(
                -raw_prediction / target_std.to(device).clamp_min(1e-3), sign_target
            )
            loss = regression + 0.2 * classification
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        row = report(
            model, valid, state, routes, device, target_mean, target_std
        )
        improved = best is None or row["selected_mae"] < best["selected_mae"]
        if improved:
            best = dict(row)
            best["epoch"] = epoch
            best["state_dict"] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"[{state} epoch {epoch}] {json.dumps(row, sort_keys=True)}")
        if stale >= 8:
            break
    best.pop("state_dict")
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--epochs", type=int, default=40)
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
    args.bridge_correction_limit = defaults.bridge_correction_limit
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
    train = build_split(
        forecaster, loader("train"), device, mean, std, args.max_train_batches
    )
    valid = build_split(
        forecaster, loader("valid"), device, mean, std, args.max_eval_batches
    )
    result = {
        "purpose": "diagnosis only; labels never train the online policy",
        "target_available_at_inference": False,
        "z3": train_probe(train, valid, "z3", R3_ROUTES, device, args.epochs),
        "z4": train_probe(train, valid, "z4", R4_ROUTES, device, args.epochs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
