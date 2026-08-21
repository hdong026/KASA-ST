"""Train and evaluate the small progressive trajectory selector on PEMS04."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .progressive_selector import (
    ROUTES,
    ProgressiveTrajectorySelector,
    decision_targets,
    forecast_state_features,
    history_state_features,
)
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


DEFAULT_BRIDGE_CHECKPOINT = ROOT / (
    "checkpoints/ForecastTrajectorySimple/seed1_20260815_080635/bridges_best.pt"
)
DEFAULT_CACHE_DIR = ROOT / "checkpoints/ForecastTrajectorySimpleSelector/cache_epoch45_node"


def build_frozen_forecaster(args, device: torch.device):
    model = build_model(args, device)
    checkpoint = load_bridge_checkpoint(args.bridge_checkpoint, model, device)
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("The F2F model and bridges must all be frozen.")
    print(
        f"[init] frozen bridges: {args.bridge_checkpoint} "
        f"(epoch {checkpoint.get('epoch', '?')})"
    )
    return model


@torch.inference_mode()
def build_selector_cache(
    model,
    loader,
    device: torch.device,
    mean: float,
    std: float,
    max_batches: int | None,
    split: str,
) -> dict[str, torch.Tensor]:
    """Cache online states and realized route losses for one split."""
    histories, z3_states, all_losses, all_node_losses = [], [], [], []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target = prepare_batch(batch, device)
        target_raw = to_physical(target, mean, std)
        canonical_trace = model(history, trajectory=ROUTES[0], return_all=True)
        predictions = [canonical_trace["pred"]]
        for route in ROUTES[1:]:
            predictions.append(model(history, trajectory=route))
        losses = torch.stack(
            [
                per_sample_mae(to_physical(prediction, mean, std), target_raw, 0.0)
                for prediction in predictions
            ],
            dim=1,
        )
        target_mask = target_raw.ne(0.0)
        node_losses = []
        for prediction in predictions:
            absolute = (to_physical(prediction, mean, std) - target_raw).abs()
            numerator = (absolute * target_mask).sum(dim=(1, 3))
            denominator = target_mask.sum(dim=(1, 3)).clamp_min(1)
            node_losses.append(numerator / denominator)
        histories.append(history_state_features(history).cpu())
        z3_states.append(
            forecast_state_features(
                history, canonical_trace["state_forecasts"][3]
            ).cpu()
        )
        all_losses.append(losses.cpu())
        all_node_losses.append(torch.stack(node_losses, dim=1).cpu())
    if not histories:
        raise RuntimeError(f"The {split} cache is empty.")
    cache = {
        "history": torch.cat(histories),
        "z3": torch.cat(z3_states),
        "losses": torch.cat(all_losses),
        "node_losses": torch.cat(all_node_losses),
    }
    print(
        f"[cache:{split}] samples={len(cache['losses'])} "
        f"history_dim={cache['history'].shape[1]} z3_dim={cache['z3'].shape[1]} "
        f"time={time.perf_counter() - started:.1f}s"
    )
    return cache


def load_or_build_cache(
    cache_dir: Path,
    split: str,
    model,
    loader,
    device,
    mean,
    std,
    max_batches,
):
    """Reuse deterministic frozen-forecast TRAIN/VALID caches when available."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "full" if max_batches is None else f"batches{max_batches}"
    path = cache_dir / f"{split}_{suffix}.pt"
    if path.is_file():
        cache = torch.load(path, map_location="cpu")
        print(f"[cache:{split}] loaded {path} ({len(cache['losses'])} samples)")
        return cache
    cache = build_selector_cache(
        model, loader, device, mean, std, max_batches, split
    )
    torch.save(cache, path)
    print(f"[cache:{split}] saved {path}")
    return cache


def selector_report(selector, cache: dict[str, torch.Tensor], device) -> dict:
    selector.eval()
    selected_parts = []
    batch_size = 1024
    with torch.inference_mode():
        for start in range(0, len(cache["losses"]), batch_size):
            selected_parts.append(
                selector.select_route_indices(
                    cache["history"][start : start + batch_size].to(device),
                    cache["z3"][start : start + batch_size].to(device),
                ).cpu()
            )
    selected_index = torch.cat(selected_parts)
    losses = cache["losses"]
    rows = torch.arange(len(losses))
    selected_loss = losses[rows, selected_index]
    oracle_loss, oracle_index = losses.min(dim=1)
    selected_counts = torch.bincount(selected_index, minlength=len(ROUTES))
    oracle_counts = torch.bincount(oracle_index, minlength=len(ROUTES))
    fixed = losses.mean(dim=0)
    return {
        "num_samples": len(losses),
        "canonical_mae": float(fixed[0]),
        "oracle_mae": float(oracle_loss.mean()),
        "learned_selection_mae": float(selected_loss.mean()),
        "selector_regret": float(selected_loss.mean() - oracle_loss.mean()),
        "gain_vs_canonical": float(fixed[0] - selected_loss.mean()),
        "selected_route_counts": {
            route_name(route): int(selected_counts[index])
            for index, route in enumerate(ROUTES)
        },
        "oracle_route_counts": {
            route_name(route): int(oracle_counts[index])
            for index, route in enumerate(ROUTES)
        },
        "fixed_route_mae": {
            route_name(route): float(fixed[index])
            for index, route in enumerate(ROUTES)
        },
    }


def train_selector(args, train_cache, valid_cache, device, output_dir):
    initial_target, z3_target = decision_targets(train_cache["losses"])
    node_initial_target, node_z3_target = decision_targets(
        train_cache["node_losses"]
    )
    node_initial_target = node_initial_target.transpose(1, 2)
    node_z3_target = node_z3_target.transpose(1, 2)
    selector = ProgressiveTrajectorySelector(
        train_cache["history"].shape[1],
        train_cache["z3"].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        safety_margin=args.safety_margin,
    ).to(device)
    selector.fit_normalizers(
        train_cache["history"].to(device),
        train_cache["z3"].to(device),
        initial_target.to(device),
        z3_target.to(device),
    )
    dataset = TensorDataset(
        train_cache["history"], train_cache["z3"], initial_target, z3_target,
        node_initial_target, node_z3_target,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.selector_batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=args.selector_lr, weight_decay=args.selector_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.selector_epochs, eta_min=args.selector_lr * 0.05
    )
    checkpoint_path = output_dir / "selector_best.pt"
    initial_valid = selector_report(selector, valid_cache, device)
    best_mae = initial_valid["learned_selection_mae"]
    best_epoch = 0
    stale = 0
    history_log = []
    torch.save(
        {
            "epoch": 0,
            "selector_state_dict": selector.state_dict(),
            "validation": initial_valid,
            "routes": [list(route) for route in ROUTES],
            "train_targets_only": True,
            "forecaster_frozen": True,
        },
        checkpoint_path,
    )
    print(
        f"[selector 000] valid={best_mae:.5f} "
        f"canonical={initial_valid['canonical_mae']:.5f} (safe initialization)"
    )
    for epoch in range(1, args.selector_epochs + 1):
        selector.train()
        total_loss = 0.0
        total_count = 0
        for (
            history_features, z3_features, initial, z3, node_initial, node_z3
        ) in loader:
            history_features = history_features.to(device)
            z3_features = z3_features.to(device)
            initial = initial.to(device)
            z3 = z3.to(device)
            node_initial = node_initial.to(device)
            node_z3 = node_z3.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted_initial, predicted_z3 = selector.normalized_predictions(
                history_features, z3_features
            )
            predicted_node_initial, predicted_node_z3 = (
                selector.normalized_sensor_predictions(history_features, z3_features)
            )
            # Magnitude-weighted pairwise classification is a bounded surrogate
            # for expected MAE gain: costly mistakes matter more, while a few
            # large regression outliers cannot dominate the fitted value.
            def pairwise_loss(logits, excess_cost, positive_weight):
                beneficial = (excess_cost < 0).to(logits.dtype)
                if args.pairwise_weighting == "magnitude":
                    weights = excess_cost.abs().clamp_min(1e-3)
                else:
                    weights = torch.ones_like(excess_cost)
                weights = weights * torch.where(
                    beneficial.bool(), positive_weight, 1.0
                )
                return F.binary_cross_entropy_with_logits(
                    logits, beneficial, weight=weights, reduction="sum"
                ) / weights.sum()

            loss = pairwise_loss(
                predicted_initial, initial, args.initial_positive_weight
            ) + pairwise_loss(
                predicted_z3, z3, args.z3_positive_weight
            ) + args.node_loss_weight * (
                pairwise_loss(
                    predicted_node_initial, node_initial,
                    args.initial_positive_weight,
                )
                + pairwise_loss(
                    predicted_node_z3, node_z3, args.z3_positive_weight,
                )
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(history_features)
            total_count += len(history_features)
        scheduler.step()
        valid = selector_report(selector, valid_cache, device)
        selection_mae = valid["learned_selection_mae"]
        improved = selection_mae < best_mae
        if improved:
            best_mae = selection_mae
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "selector_state_dict": selector.state_dict(),
                    "validation": valid,
                    "routes": [list(route) for route in ROUTES],
                    "train_targets_only": True,
                    "forecaster_frozen": True,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "train_regression_loss": total_loss / total_count,
            "validation": valid,
            "best": improved,
        }
        history_log.append(record)
        (output_dir / "selector_history.json").write_text(
            json.dumps(history_log, indent=2), encoding="utf-8"
        )
        print(
            f"[selector {epoch:03d}] train={total_loss / total_count:.5f} "
            f"valid={selection_mae:.5f} canonical={valid['canonical_mae']:.5f} "
            f"gain={valid['gain_vs_canonical']:+.5f} best={improved}"
        )
        if stale >= args.selector_patience:
            print(f"[early-stop] {args.selector_patience} selector epochs without improvement")
            break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    selector.load_state_dict(checkpoint["selector_state_dict"], strict=True)
    selector.eval()
    return selector, checkpoint_path, best_epoch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a TRAIN-supervised progressive trajectory selector."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--f2f-checkpoint", type=Path, default=DEFAULT_F2F_CHECKPOINT)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--forecast-batch-size", type=int, default=64)
    parser.add_argument("--selector-batch-size", type=int, default=256)
    parser.add_argument("--selector-epochs", type=int, default=100)
    parser.add_argument("--selector-patience", type=int, default=15)
    parser.add_argument("--selector-lr", type=float, default=1e-3)
    parser.add_argument("--selector-weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--safety-margin", type=float, default=-0.25,
        help="VALID-fixed benefit-logit threshold; lower values route more alternatives.",
    )
    parser.add_argument("--initial-positive-weight", type=float, default=1.0)
    parser.add_argument("--z3-positive-weight", type=float, default=1.0)
    parser.add_argument("--node-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--pairwise-weighting", choices=("magnitude", "uniform"),
        default="magnitude",
    )
    parser.add_argument("--bridge-correction-limit", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Development-only: stop after final VALID checkpoint selection.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.selector_epochs = 2
        args.selector_patience = 2
        args.workers = 0
        args.max_train_batches = 2
        args.max_eval_batches = 2
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if not args.bridge_checkpoint.is_file():
        raise FileNotFoundError(args.bridge_checkpoint)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or ROOT / (
        f"checkpoints/ForecastTrajectorySimpleSelector/seed{args.seed}_{run_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"[run] output: {output_dir}")

    data, indices, mean, std = load_data(args.data_dir)
    datasets = {
        split: WindowDataset(data, indices[split])
        for split in ("train", "valid")
    }
    def split_loader(split: str):
        return make_loader(
            datasets[split], batch_size=args.forecast_batch_size, shuffle=False,
            workers=args.workers, device=device, seed=args.seed,
        )

    model = build_frozen_forecaster(args, device)
    canonical_audit(model, split_loader("valid"), device)
    train_cache = load_or_build_cache(
        args.cache_dir, "train", model, split_loader("train"), device, mean, std,
        args.max_train_batches,
    )
    valid_cache = load_or_build_cache(
        args.cache_dir, "valid", model, split_loader("valid"), device, mean, std,
        args.max_eval_batches,
    )
    selector, checkpoint_path, best_epoch = train_selector(
        args, train_cache, valid_cache, device, output_dir
    )
    validation = selector_report(selector, valid_cache, device)
    print("\n=== FINAL VALID (checkpoint selected here) ===")
    print(json.dumps(validation, indent=2))

    if args.skip_test:
        report_path = output_dir / "selector_valid_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "method": "ProgressiveTrajectorySelector",
                    "best_valid_epoch": best_epoch,
                    "selector_checkpoint": str(checkpoint_path),
                    "bridge_checkpoint": str(args.bridge_checkpoint),
                    "validation": validation,
                    "test": None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("[done] --skip-test set; TEST was not evaluated")
        print(f"[done] validation report: {report_path}")
        return

    # TEST is deliberately touched only now, after the VALID checkpoint is fixed.
    datasets["test"] = WindowDataset(data, indices["test"])
    test_cache = build_selector_cache(
        model, split_loader("test"), device, mean, std,
        args.max_eval_batches, "test",
    )
    test = selector_report(selector, test_cache, device)
    report = {
        "method": "ProgressiveTrajectorySelector",
        "best_valid_epoch": best_epoch,
        "selector_checkpoint": str(checkpoint_path),
        "bridge_checkpoint": str(args.bridge_checkpoint),
        "canonical_exact": {"torch_equal": True, "max_abs": 0.0},
        "target_usage": {
            "train": "pairwise trajectory-loss supervision",
            "valid": "checkpoint selection and reporting only",
            "test": "one evaluation after checkpoint selection",
        },
        "validation": validation,
        "test": test,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    report_path = output_dir / "selector_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== ONE-TIME TEST ===")
    print(json.dumps(test, indent=2))
    print(f"[done] selector checkpoint: {checkpoint_path}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()
