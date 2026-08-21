"""One-command bridge training and trajectory-headroom evaluation on PEMS04."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import runpy
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .forecast_trajectory_simple import ForecastTrajectorySimple
from .objectives import per_sample_mae, trajectory_supervision_loss


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "examples/ChainForecasting/G1_final_adaptive_PEMS04_12to12.py"
DEFAULT_F2F_CHECKPOINT = ROOT / (
    "checkpoints/G1_final_adaptive_PEMS04_12to12/"
    "fe3d42a553f018278976a22677863ac5/ChainForecasting_best_val_MAE.pt"
)
DEFAULT_DATA_DIR = ROOT / "datasets/PEMS04"
ROUTES = (
    (3, 6, 12),
    (3, 12),
    (2, 4, 12),
    (4, 12),
    (3, 4, 6, 12),
)
CANONICAL_ROUTE = ROUTES[0]


class WindowDataset(Dataset):
    """Window view over one shared processed-data tensor."""

    def __init__(self, data: torch.Tensor, indices: Sequence[Sequence[int]]):
        self.data = data
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        start, split, end = self.indices[item]
        return self.data[split:end], self.data[start:split]


def route_name(route: Sequence[int]) -> str:
    return "-".join(str(value) for value in route)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_dir: Path):
    data_path = data_dir / "data_in12_out12.pkl"
    index_path = data_dir / "index_in12_out12.pkl"
    scaler_path = data_dir / "scaler_in12_out12.pkl"
    for path in (data_path, index_path, scaler_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with data_path.open("rb") as file:
        processed = pickle.load(file)["processed_data"]
    with index_path.open("rb") as file:
        indices = pickle.load(file)
    with scaler_path.open("rb") as file:
        scaler = pickle.load(file)
    data = torch.from_numpy(processed).float()
    mean = float(np.asarray(scaler["args"]["mean"]).reshape(-1)[0])
    std = float(np.asarray(scaler["args"]["std"]).reshape(-1)[0])
    return data, indices, mean, std


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def prepare_batch(batch, device: torch.device):
    future, history = batch
    history = history[..., :3].to(device, non_blocking=True)
    target = future[..., :1].to(device, non_blocking=True)
    return history, target


def to_physical(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return tensor * std + mean


def build_model(args, device: torch.device) -> ForecastTrajectorySimple:
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.f2f_checkpoint.is_file():
        raise FileNotFoundError(args.f2f_checkpoint)
    config = runpy.run_path(str(args.config))["CFG"]
    model_args = dict(config.MODEL.PARAM)
    model_args.update(
        trajectories=[list(route) for route in ROUTES],
        freeze_f2f=True,
        bridge_correction_limit=args.bridge_correction_limit,
    )
    model = ForecastTrajectorySimple(**model_args)
    load_result = model.load_pretrained_f2f(args.f2f_checkpoint, strict=True)
    print(f"[init] canonical checkpoint: {args.f2f_checkpoint}")
    print(f"[init] strict load: {load_result}")
    print(f"[init] bridge edges: {model.bridge_edges}")
    return model.to(device)


def canonical_audit(model, loader, device: torch.device) -> None:
    batch = next(iter(loader))
    history, _ = prepare_batch(batch, device)
    history = history[:1]
    model.eval()
    with torch.inference_mode():
        original = model.f2f(history)
        graph = model(history, trajectory=CANONICAL_ROUTE)
    if not torch.equal(original, graph):
        maximum = float((original - graph).abs().max().item())
        raise RuntimeError(f"Canonical equivalence failed; max_abs={maximum}")
    print("[audit] canonical [3,6,12]: torch.equal=True, max_abs=0.0")


def train_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    mean: float,
    std: float,
    epoch: int,
    max_batches: int | None,
    intermediate_bridge_weight: float,
) -> dict:
    model.train()
    bridge_routes = ROUTES[1:]
    route_loss_sum = {route: 0.0 for route in bridge_routes}
    route_steps = {route: 0 for route in bridge_routes}
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        route = bridge_routes[((epoch - 1) * len(loader) + batch_index) % len(bridge_routes)]
        history, target = prepare_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        trace = model(history, trajectory=route, return_all=True)
        target_raw = to_physical(target, mean, std)
        physical_trace = dict(trace)
        physical_trace["pred"] = to_physical(trace["pred"], mean, std)
        physical_trace["state_forecasts"] = {
            resolution: to_physical(state, mean, std)
            for resolution, state in trace["state_forecasts"].items()
        }
        loss = trajectory_supervision_loss(
            physical_trace,
            target_raw,
            null_val=0.0,
            intermediate_bridge_weight=intermediate_bridge_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.bridge_parameters()), max_norm=5.0)
        optimizer.step()
        route_loss_sum[route] += float(loss.detach().item())
        route_steps[route] += 1
    route_objective = {
        route_name(route): route_loss_sum[route] / max(route_steps[route], 1)
        for route in bridge_routes
    }
    values = [
        route_loss_sum[route] / route_steps[route]
        for route in bridge_routes
        if route_steps[route] > 0
    ]
    return {
        "mean_objective": float(np.mean(values)) if values else float("nan"),
        "route_objective": route_objective,
        "seconds": time.perf_counter() - started,
    }


def evaluate(
    model,
    loader,
    device: torch.device,
    mean: float,
    std: float,
    max_batches: int | None,
) -> dict:
    model.eval()
    route_sums = torch.zeros(len(ROUTES), dtype=torch.float64)
    oracle_sum = 0.0
    oracle_counts = torch.zeros(len(ROUTES), dtype=torch.long)
    sample_count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            history, target = prepare_batch(batch, device)
            target_raw = to_physical(target, mean, std)
            losses = []
            for route in ROUTES:
                prediction = model(history, trajectory=route)
                prediction_raw = to_physical(prediction, mean, std)
                losses.append(per_sample_mae(prediction_raw, target_raw, null_val=0.0))
            loss_matrix = torch.stack(losses, dim=1)
            route_sums += loss_matrix.sum(dim=0).double().cpu()
            oracle_loss, oracle_index = loss_matrix.min(dim=1)
            oracle_sum += float(oracle_loss.sum().item())
            oracle_counts += torch.bincount(
                oracle_index.cpu(), minlength=len(ROUTES)
            )
            sample_count += int(history.shape[0])
    if sample_count == 0:
        raise RuntimeError("Evaluation loader produced no samples.")
    fixed_mae = route_sums / sample_count
    best_fixed_index = int(fixed_mae.argmin().item())
    oracle_mae = oracle_sum / sample_count
    report = {
        "num_samples": sample_count,
        "routes": [list(route) for route in ROUTES],
        "fixed_route_mae": {
            route_name(route): float(fixed_mae[index].item())
            for index, route in enumerate(ROUTES)
        },
        "canonical_mae": float(fixed_mae[0].item()),
        "best_fixed_trajectory": list(ROUTES[best_fixed_index]),
        "best_fixed_mae": float(fixed_mae[best_fixed_index].item()),
        "oracle_mae": oracle_mae,
        "oracle_gain_vs_canonical": float(fixed_mae[0].item() - oracle_mae),
        "oracle_gain_vs_best_fixed": float(
            fixed_mae[best_fixed_index].item() - oracle_mae
        ),
        "oracle_route_counts": {
            route_name(route): int(oracle_counts[index].item())
            for index, route in enumerate(ROUTES)
        },
        # Checkpoint selection deliberately avoids the label oracle.
        "selection_mae": float(fixed_mae[1:].mean().item()),
    }
    return report


def save_bridge_checkpoint(path: Path, model, optimizer, epoch: int, report: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "bridge_state_dict": model.bridges.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation": report,
            "routes": [list(route) for route in ROUTES],
            "canonical_f2f_frozen": True,
        },
        path,
    )


def load_bridge_checkpoint(path: Path, model, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.bridges.load_state_dict(checkpoint["bridge_state_dict"], strict=True)
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ForecastTrajectorySimple bridges and measure oracle headroom."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--f2f-checkpoint", type=Path, default=DEFAULT_F2F_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--intermediate-bridge-weight",
        type=float,
        default=0.25,
        help="Weight on mean pooled-target MAE of intermediate bridge states.",
    )
    parser.add_argument(
        "--bridge-correction-limit",
        type=float,
        default=2.0,
        help="Maximum normalized correction magnitude around a forecast anchor.",
    )
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch and two batches per split; validates the pipeline only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.patience = 1
        args.workers = 0
        args.max_train_batches = 2
        args.max_eval_batches = 2
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    seed_everything(args.seed)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        ROOT / "checkpoints/ForecastTrajectorySimple" / f"seed{args.seed}_{run_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"[run] output: {output_dir}")
    print(f"[run] device: {device}")

    data, indices, mean, std = load_data(args.data_dir)
    datasets = {
        split: WindowDataset(data, indices[split])
        for split in ("train", "valid", "test")
    }
    loaders = {
        "train": make_loader(
            datasets["train"], batch_size=args.batch_size, shuffle=True,
            workers=args.workers, device=device, seed=args.seed,
        ),
        "valid": make_loader(
            datasets["valid"], batch_size=args.batch_size, shuffle=False,
            workers=args.workers, device=device, seed=args.seed,
        ),
        "test": make_loader(
            datasets["test"], batch_size=args.batch_size, shuffle=False,
            workers=args.workers, device=device, seed=args.seed,
        ),
    }
    print(f"[data] train={len(datasets['train'])} valid={len(datasets['valid'])} "
          f"test={len(datasets['test'])} mean={mean:.6f} std={std:.6f}")

    model = build_model(args, device)
    canonical_audit(model, loaders["valid"], device)
    bridge_parameters = list(model.bridge_parameters())
    optimizer = torch.optim.Adam(
        bridge_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, args.epochs // 2), max(2, (args.epochs * 5) // 6)],
        gamma=0.5,
    )

    best_path = output_dir / "bridges_best.pt"
    best_selection = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history_log = []
    for epoch in range(1, args.epochs + 1):
        train_report = train_epoch(
            model, loaders["train"], optimizer, device, mean, std, epoch,
            args.max_train_batches, args.intermediate_bridge_weight,
        )
        valid_report = evaluate(
            model, loaders["valid"], device, mean, std, args.max_eval_batches
        )
        scheduler.step()
        selection = valid_report["selection_mae"]
        improved = selection < best_selection
        if improved:
            best_selection = selection
            best_epoch = epoch
            stale_epochs = 0
            save_bridge_checkpoint(best_path, model, optimizer, epoch, valid_report)
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_report,
            "valid": valid_report,
            "best": improved,
        }
        history_log.append(record)
        print(
            f"[epoch {epoch:03d}] train_objective={train_report['mean_objective']:.4f} "
            f"valid_selection={selection:.4f} "
            f"valid_oracle={valid_report['oracle_mae']:.4f} "
            f"best={improved} time={train_report['seconds']:.1f}s"
        )
        (output_dir / "history.json").write_text(
            json.dumps(history_log, indent=2), encoding="utf-8"
        )
        if stale_epochs >= args.patience:
            print(f"[early-stop] no validation improvement for {args.patience} epochs")
            break

    load_bridge_checkpoint(best_path, model, device)
    canonical_audit(model, loaders["valid"], device)
    validation = evaluate(
        model, loaders["valid"], device, mean, std, args.max_eval_batches
    )
    test = evaluate(model, loaders["test"], device, mean, std, args.max_eval_batches)
    final_report = {
        "method": "ForecastTrajectorySimple",
        "best_epoch": best_epoch,
        "checkpoint": str(best_path),
        "canonical_f2f_checkpoint": str(args.f2f_checkpoint),
        "validation": validation,
        "test": test,
        "args": {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
    }
    report_path = output_dir / "headroom_report.json"
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print("\n=== TEST HEADROOM ===")
    print(json.dumps(test, indent=2))
    print(f"[done] bridge checkpoint: {best_path}")
    print(f"[done] headroom report: {report_path}")


if __name__ == "__main__":
    main()
