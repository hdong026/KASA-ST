"""Fine-tune only adaptive bridges to behave as safe residual refinements.

The mature F2F and its exact canonical 3->6->12 path are frozen.  In addition
to final route MAE, every non-native edge is trained under a common 12-step
stopping interpretation and penalized when its correction increases a
sample's MAE relative to the edge anchor.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import (
    interpolate_forecast,
)

from .run_online_sequential_rl import TRAJECTORIES
from .run_pipeline import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_F2F_CHECKPOINT,
    ROOT,
    WindowDataset,
    build_model,
    canonical_audit,
    load_bridge_checkpoint,
    load_data,
    make_loader,
    per_sample_mae,
    prepare_batch,
    route_name,
    seed_everything,
    to_physical,
)
from .run_selector import DEFAULT_BRIDGE_CHECKPOINT


CANONICAL = (3, 6, 12)
ADAPTIVE_ROUTES = tuple(route for route in TRAJECTORIES if route != CANONICAL)


def edge_refinement_terms(model, trace, history, target_raw, mean, std):
    local_losses = []
    harm_losses = []
    gains = []
    previous = None
    for (source, target_resolution), edge_type in zip(
        trace["trajectory_edges"], trace["edge_types"]
    ):
        current = trace["state_forecasts"][target_resolution]
        if edge_type == "bridge":
            anchor = (
                history[:, -1:, :, :1].expand(-1, target_resolution, -1, -1)
                if previous is None
                else interpolate_forecast(previous, target_resolution)
            )
            anchor_full = model.finalize_forecast(
                interpolate_forecast(anchor, model.output_len), history
            )
            refined_full = model.finalize_forecast(
                interpolate_forecast(current, model.output_len), history
            )
            anchor_mae = per_sample_mae(
                to_physical(anchor_full, mean, std), target_raw, 0.0
            )
            refined_mae = per_sample_mae(
                to_physical(refined_full, mean, std), target_raw, 0.0
            )
            local_losses.append(refined_mae.mean())
            harm_losses.append(torch.relu(refined_mae - anchor_mae).mean())
            gains.append((anchor_mae - refined_mae).detach())
        previous = current
    if not local_losses:
        raise RuntimeError("Adaptive training route contains no bridge edge.")
    return torch.stack(local_losses).mean(), torch.stack(harm_losses).mean(), gains


def route_objective(model, history, target, route, mean, std, local_weight, harm_weight):
    trace = model.execute_trajectory(history, route)
    target_raw = to_physical(target, mean, std)
    final_per_sample = per_sample_mae(
        to_physical(trace["pred"], mean, std), target_raw, 0.0
    )
    local, harm, gains = edge_refinement_terms(
        model, trace, history, target_raw, mean, std
    )
    loss = final_per_sample.mean() + local_weight * local + harm_weight * harm
    return loss, final_per_sample.detach(), local.detach(), harm.detach(), gains


def train_epoch(model, loader, optimizer, device, mean, std, args, epoch):
    model.train()
    started = time.perf_counter()
    totals = {"objective": 0.0, "final": 0.0, "local": 0.0, "harm": 0.0}
    samples = 0
    benefit = []
    for batch_index, batch in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        history, target = prepare_batch(batch, device)
        route = ADAPTIVE_ROUTES[
            ((epoch - 1) * len(loader) + batch_index) % len(ADAPTIVE_ROUTES)
        ]
        loss, final, local, harm, gains = route_objective(
            model, history, target, route, mean, std,
            args.local_weight, args.harm_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.bridge_parameters()), 5.0)
        optimizer.step()
        count = len(history)
        samples += count
        totals["objective"] += float(loss.detach()) * count
        totals["final"] += float(final.mean()) * count
        totals["local"] += float(local) * count
        totals["harm"] += float(harm) * count
        benefit.extend(gain.cpu() for gain in gains)
    all_gain = torch.cat(benefit)
    return {
        **{key: value / samples for key, value in totals.items()},
        "bridge_edge_benefit_fraction": float((all_gain > 0).float().mean()),
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate(model, loader, device, mean, std, args):
    model.eval()
    route_losses = {route: [] for route in TRAJECTORIES}
    local_values, harm_values, edge_gains = [], [], []
    for batch_index, batch in enumerate(loader):
        if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
            break
        history, target = prepare_batch(batch, device)
        target_raw = to_physical(target, mean, std)
        for route in TRAJECTORIES:
            trace = model.execute_trajectory(history, route)
            route_losses[route].append(
                per_sample_mae(
                    to_physical(trace["pred"], mean, std), target_raw, 0.0
                ).cpu()
            )
            if route != CANONICAL:
                local, harm, gains = edge_refinement_terms(
                    model, trace, history, target_raw, mean, std
                )
                local_values.append(local.cpu())
                harm_values.append(harm.cpu())
                edge_gains.extend(gain.cpu() for gain in gains)
    matrix = torch.stack(
        [torch.cat(route_losses[route]) for route in TRAJECTORIES], dim=1
    )
    fixed = matrix.mean(0)
    oracle = matrix.min(1).values.mean()
    gain = torch.cat(edge_gains)
    adaptive_mean = fixed[
        torch.tensor([route != CANONICAL for route in TRAJECTORIES])
    ].mean()
    local_mean = torch.stack(local_values).mean()
    harm_mean = torch.stack(harm_values).mean()
    selection = (
        adaptive_mean
        + args.local_weight * local_mean
        + args.harm_weight * harm_mean
    )
    return {
        "selection_objective": float(selection),
        "fixed_route_mae": {
            route_name(route): float(fixed[index])
            for index, route in enumerate(TRAJECTORIES)
        },
        "canonical_mae": float(fixed[list(TRAJECTORIES).index(CANONICAL)]),
        "best_fixed_mae": float(fixed.min()),
        "best_fixed_route": route_name(TRAJECTORIES[int(fixed.argmin())]),
        "oracle_mae": float(oracle),
        "mean_local_stopping_mae": float(local_mean),
        "mean_harm_penalty": float(harm_mean),
        "bridge_edge_benefit_fraction": float((gain > 0).float().mean()),
        "mean_bridge_edge_gain": float(gain.mean()),
    }


def save_checkpoint(path, model, optimizer, epoch, validation):
    torch.save(
        {
            "epoch": epoch,
            "bridge_state_dict": model.bridges.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation": validation,
            "routes": [list(route) for route in TRAJECTORIES],
            "canonical_f2f_frozen": True,
            "training_method": "edge_local_safe_residual_refinement",
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--f2f-checkpoint", type=Path, default=DEFAULT_F2F_CHECKPOINT)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--local-weight", type=float, default=0.25)
    parser.add_argument("--harm-weight", type=float, default=2.0)
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
        args.patience = 1
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

    model = build_model(args, device)
    starting = load_bridge_checkpoint(args.bridge_checkpoint, model, device)
    print(f"[init] fine-tune bridges from {args.bridge_checkpoint} epoch={starting.get('epoch')}")
    bridge_parameters = list(model.bridge_parameters())
    model.requires_grad_(False)
    for parameter in bridge_parameters:
        parameter.requires_grad_(True)
    canonical_audit(model, loader("valid"), device)
    optimizer = torch.optim.AdamW(
        bridge_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    best_path = args.output_dir / "progressive_bridges_best.pt"
    best_score = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(
            model, loader("train", True), optimizer, device, mean, std, args, epoch
        )
        valid = evaluate(model, loader("valid"), device, mean, std, args)
        improved = valid["selection_objective"] < best_score
        if improved:
            best_score = valid["selection_objective"]
            save_checkpoint(best_path, model, optimizer, epoch, valid)
            stale = 0
        else:
            stale += 1
        history.append({"epoch": epoch, "train": train, "valid": valid, "best": improved})
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"[epoch {epoch:02d}] seconds={train['seconds']:.1f} "
            f"valid_score={valid['selection_objective']:.4f} "
            f"benefit={valid['bridge_edge_benefit_fraction']:.3f} "
            f"best_fixed={valid['best_fixed_mae']:.4f} oracle={valid['oracle_mae']:.4f} "
            f"best={improved}"
        )
        if stale >= args.patience:
            break
    load_bridge_checkpoint(best_path, model, device)
    canonical_audit(model, loader("valid"), device)
    final = evaluate(model, loader("valid"), device, mean, std, args)
    report = {
        "method": "edge_local_safe_residual_refinement",
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
