"""Formal 100-epoch PEMS04 training for the unchanged ResolutionNative V1.

The protocol mirrors the protected BasicTS/F2FNet run where appropriate.
TEST is not constructed until all training is complete and the best-VALID
checkpoint has been loaded.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch import (
    F2FCoTResolutionNativeV1Net,
)
from scripts.f2f_cot_resolution_native_v1_experiment import (
    FIXED_ROUTE,
    PROTECTED_F2F_COT_CHECKPOINT,
    evaluate,
    model_args,
    route_loss,
    structural_report,
)
from scripts.f2f_cot_runtime import load_rescale, make_loader, select_batch


EXPERIMENT = "f2f_cot_resolution_native_v1_formal"
MILESTONES = [1, 35, 60, 80, 95]
GAMMA = 0.5


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best: dict,
    history: list[dict],
    protocol: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "best": best,
            "history": history,
            "model_args": model_args(),
            "route": FIXED_ROUTE,
            "method": "F2FCoTResolutionNativeV1Net",
            "protocol": protocol,
        },
        path,
    )


def train_formal(
    model,
    train_loader,
    valid_loader,
    device,
    rescale,
    checkpoint_dir: Path,
    result_dir: Path,
    args,
    protocol: dict,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=MILESTONES,
        gamma=GAMMA,
    )
    best_path = checkpoint_dir / "resolution_native_v1_formal_best_val_MAE.pt"
    last_path = checkpoint_dir / "resolution_native_v1_formal_last.pt"
    best = {"MAE": float("inf"), "epoch": 0}
    history_rows: list[dict] = []
    start_epoch = 1

    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best = dict(checkpoint["best"])
        history_rows = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[formal-v1] resumed at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses = []
        epoch_start = time.perf_counter()
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        for batch_index, batch in enumerate(train_loader):
            if (
                args.max_train_batches is not None
                and batch_index >= args.max_train_batches
            ):
                break
            history, target, _ = select_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = route_loss(model, history, target, rescale)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch={epoch} batch={batch_index}"
                )
            loss.backward()
            # Deliberately no gradient clipping: matches protected BasicTS/F2FNet.
            optimizer.step()
            train_losses.append(float(loss.detach()))

        # Every epoch uses the complete VALID split. TEST does not exist yet.
        valid = evaluate(
            model,
            valid_loader,
            device,
            rescale,
            max_batches=args.max_valid_batches,
        )
        valid_mae = float(valid["MAE"])
        improved = valid_mae < float(best["MAE"])
        if improved:
            best = {"MAE": valid_mae, "epoch": int(epoch)}

        scheduler.step()
        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_losses)),
            "learning_rate_used": learning_rate_used,
            "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
            "valid": valid,
            "improved": bool(improved),
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history_rows.append(row)
        if improved:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best,
                history_rows,
                protocol,
            )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best,
            history_rows,
            protocol,
        )
        dump_json(result_dir / "training_history.json", history_rows)
        print(
            f"[formal-v1] epoch={epoch:03d} "
            f"lr={learning_rate_used:.8f} loss={row['train_loss']:.4f} "
            f"VALID_MAE={valid_mae:.4f} "
            f"best={best['MAE']:.4f}@{best['epoch']} "
            f"seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return best, history_rows, best_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tag", default="formal_basicts_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-valid-batches", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        args.workers = 0
        args.max_train_batches = 2
        args.max_valid_batches = 2
        args.tag = "smoke_formal"
    if args.epochs != 100 and not args.smoke:
        raise ValueError("formal protocol requires exactly 100 epochs")

    seed_all(args.seed)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    result_dir = ROOT / "results" / EXPERIMENT / f"{args.tag}_seed{args.seed}"
    checkpoint_dir = (
        ROOT
        / "checkpoints"
        / "PEMS04"
        / "H12"
        / EXPERIMENT
        / f"{args.tag}_seed{args.seed}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "optimizer": "Adam",
        "initial_learning_rate": args.learning_rate,
        "learning_rate_rationale": (
            "Half the scratch F2FNet LR because 1.52M parameters are warm-started "
            "while transition/query modules are newly initialized; milestone 1 "
            "then yields the probe-validated stable LR 5e-4."
        ),
        "weight_decay": args.weight_decay,
        "scheduler": "MultiStepLR",
        "milestones": MILESTONES,
        "gamma": GAMMA,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "validation_frequency": "every epoch",
        "selection": "minimum full-split VALID MAE",
        "loss": "raw-scale masked MAE, route weights [0.2,0.3,1.0]",
        "gradient_clipping": None,
        "early_stopping": False,
        "warm_start": str(PROTECTED_F2F_COT_CHECKPOINT),
        "optimizer_state_inherited_from_warm_start": False,
        "scheduler_state_inherited_from_warm_start": False,
        "probe_checkpoint_used": False,
        "TEST_constructed_during_training": False,
        "seed": args.seed,
        "route": list(FIXED_ROUTE),
    }

    model = F2FCoTResolutionNativeV1Net(**model_args()).to(device)
    protected = torch.load(
        PROTECTED_F2F_COT_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    warm_start = model.warm_start_from_f2f_cot(
        protected["model_state_dict"]
    )
    protocol["warm_start_details"] = warm_start

    rescale = load_rescale()
    train_loader = make_loader("train", args.batch_size, True, args.workers)
    valid_loader = make_loader("valid", args.batch_size, False, args.workers)
    example = select_batch(next(iter(valid_loader)), device)[0][:1]
    structure = structural_report(model, example)
    dump_json(result_dir / "protocol.json", protocol)
    dump_json(result_dir / "structural_report.json", structure)
    print(f"[formal-v1] protocol={protocol}", flush=True)
    print(f"[formal-v1] structure={structure}", flush=True)

    best, history_rows, best_path = train_formal(
        model,
        train_loader,
        valid_loader,
        device,
        rescale,
        checkpoint_dir,
        result_dir,
        args,
        protocol,
    )

    # Model selection is now immutable. Re-evaluate VALID, then construct TEST
    # for the first and only post-selection evaluation.
    selected_valid = evaluate(
        model,
        valid_loader,
        device,
        rescale,
        max_batches=args.max_valid_batches,
        include_ablations=True,
    )
    selection_fixed_before_test = True
    test_loader = make_loader("test", args.batch_size, False, args.workers)
    selected_test = evaluate(
        model,
        test_loader,
        device,
        rescale,
        max_batches=args.max_valid_batches,
    )

    report = {
        "method": "F2FCoTResolutionNativeV1Net",
        "architecture_unchanged": True,
        "protocol": protocol,
        "parameters": model.parameter_breakdown(),
        "protected_checkpoint_untouched": str(PROTECTED_F2F_COT_CHECKPOINT),
        "selected_checkpoint": str(best_path),
        "best": best,
        "epochs_completed": len(history_rows),
        "selected_valid": selected_valid,
        "selection_fixed_before_TEST": selection_fixed_before_test,
        "TEST_evaluations": 1,
        "selected_test": selected_test,
        "structural_verification": structure,
    }
    dump_json(result_dir / "final_report.json", report)
    print(
        f"[formal-v1] done best_VALID={best['MAE']:.4f}@{best['epoch']} "
        f"TEST_MAE={selected_test['MAE']:.4f} "
        f"report={result_dir / 'final_report.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
