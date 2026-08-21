#!/usr/bin/env python3
"""VALID-only continuation training for the route-complete RN forecaster.

Expected architecture contract (the architecture may be added independently):

* module:
  ``basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.``
  ``f2f_cot_resolution_native_v1_route_complete``
* exports ``F2FCoTResolutionNativeRouteCompleteNet``, ``ROUTES`` and
  ``CANONICAL_ROUTE``;
* ``rollout(history, route)`` returns the V1-style dictionary containing
  ``pred``, ``forecasts``, ``resolutions``, ``state`` and ``steps``;
* ``rollout_all_routes_shared(history, routes=ROUTES)`` returns those route
  dictionaries under ``route_outputs``/``outputs``/``by_route`` (or directly
  as a route-keyed mapping). Prefix forecasts and evidence in that call must
  be the exact same Python/Tensor objects, not independently recomputed values.

This script never constructs TEST and only writes a new route-complete
checkpoint/results namespace.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoTResolutionNative_arch.f2f_cot_resolution_native_v1 import (
    temporal_mean_pool,
)
from basicts.metrics import masked_mae
from scripts.f2f_cot_resolution_native_v1_experiment import model_args
from scripts.f2f_cot_runtime import NULL_VAL, load_rescale, make_loader, select_batch


ARCH_MODULE = (
    "basicts.archs.arch_zoo.F2FCoTResolutionNative_arch."
    "f2f_cot_resolution_native_v1_route_complete"
)
EXPERIMENT = "f2f_cot_resolution_native_route_complete_continuation"
FORMAL_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "PEMS04"
    / "H12"
    / "f2f_cot_resolution_native_v1_formal"
    / "formal_basicts_v1_seed1"
    / "resolution_native_v1_formal_best_val_MAE.pt"
)
EXPECTED_ROUTES = (
    (12,),
    (2, 12),
    (2, 4, 12),
    (2, 6, 12),
    (3, 12),
    (3, 6, 12),
    (4, 12),
    (6, 12),
)
EXPECTED_CANONICAL_ROUTE = (3, 6, 12)
OLD_CONDITIONER_INDEX = {0: 0, 3: 1, 6: 2, 12: 3}
CANONICAL_FRACTION = 0.60
CONTAINMENT_TOLERANCE = 0.10
CANONICAL_STATE_WEIGHTS = (0.2, 0.3, 1.0)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def route_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        cleaned = value.strip().replace("->", "-").replace("_", "-")
        return tuple(int(part) for part in cleaned.split("-") if part)
    return tuple(int(part) for part in value)


def route_key(route: Sequence[int]) -> str:
    return "-".join(str(int(value)) for value in route)


def load_architecture():
    try:
        module = importlib.import_module(ARCH_MODULE)
    except ModuleNotFoundError as error:
        if error.name == ARCH_MODULE:
            raise RuntimeError(
                f"route-complete architecture is not present yet: {ARCH_MODULE}. "
                "The training script was written against the contract in its module "
                "docstring; add that architecture before model smoke/training."
            ) from error
        raise
    model_class = getattr(module, "F2FCoTResolutionNativeRouteCompleteNet")
    routes = tuple(route_tuple(route) for route in getattr(module, "ROUTES"))
    canonical = route_tuple(getattr(module, "CANONICAL_ROUTE"))
    if routes != EXPECTED_ROUTES:
        raise RuntimeError(f"ROUTES must be exactly {EXPECTED_ROUTES}, got {routes}")
    if canonical != EXPECTED_CANONICAL_ROUTE:
        raise RuntimeError(
            f"CANONICAL_ROUTE must be {EXPECTED_CANONICAL_ROUTE}, got {canonical}"
        )
    return model_class, routes, canonical


def _extract_route_mapping(shared_output: Any) -> Mapping[Any, dict]:
    if not isinstance(shared_output, Mapping):
        raise TypeError("rollout_all_routes_shared must return a mapping")
    for name in ("route_outputs", "outputs", "by_route", "routes"):
        candidate = shared_output.get(name)
        if isinstance(candidate, Mapping):
            return candidate
    if all(isinstance(value, Mapping) and "pred" in value for value in shared_output.values()):
        return shared_output
    raise KeyError(
        "shared rollout must contain route_outputs/outputs/by_route, or itself "
        "be a route-keyed mapping of V1-style rollout dictionaries"
    )


def normalize_route_outputs(
    shared_output: Any, routes: Sequence[Sequence[int]]
) -> dict[tuple[int, ...], dict]:
    mapping = _extract_route_mapping(shared_output)
    normalized = {route_tuple(key): value for key, value in mapping.items()}
    wanted = tuple(route_tuple(route) for route in routes)
    missing = [route for route in wanted if route not in normalized]
    if missing:
        raise KeyError(f"shared rollout omitted routes: {missing}")
    outputs = {route: normalized[route] for route in wanted}
    for route, output in outputs.items():
        for key in ("pred", "forecasts", "resolutions", "state", "steps"):
            if key not in output:
                raise KeyError(f"route {route} output is missing {key!r}")
        if tuple(int(value) for value in output["resolutions"]) != route:
            raise RuntimeError(
                f"route {route} returned resolutions={output['resolutions']}"
            )
        if int(output["pred"].shape[1]) != 12:
            raise RuntimeError(f"route {route} did not finish at H=12")
    return outputs


def call_shared_rollout(model, history, routes=EXPECTED_ROUTES):
    method = getattr(model, "rollout_all_routes_shared", None)
    if method is None:
        raise AttributeError(
            "F2FCoTResolutionNativeRouteCompleteNet must provide "
            "rollout_all_routes_shared"
        )
    parameters = inspect.signature(method).parameters
    if "routes" in parameters:
        raw = method(history, routes=routes)
    elif "trajectories" in parameters:
        raw = method(history, trajectories=routes)
    else:
        raw = method(history)
    return raw, normalize_route_outputs(raw, routes)


def _state_evidence(output: dict):
    state = output["state"]
    return getattr(state, "evidence", None)


def assert_exact_shared_prefixes(
    model, raw_shared: Mapping[str, Any], outputs: Mapping[tuple[int, ...], dict]
) -> dict:
    """Hard-fail if paired paths use equal-but-recomputed prefixes."""
    checks: dict[str, Any] = {}
    evidence = [_state_evidence(output) for output in outputs.values()]
    evidence = [value for value in evidence if value is not None]
    if evidence:
        checks["root_evidence_same_object"] = all(value is evidence[0] for value in evidence)
        if not checks["root_evidence_same_object"]:
            raise RuntimeError("all-route shared rollout recomputed root evidence")

    for prefix in ((2,), (3,)):
        members = [
            output
            for route, output in outputs.items()
            if tuple(route[: len(prefix)]) == prefix
        ]
        tensors = [output["forecasts"][len(prefix) - 1] for output in members]
        name = route_key(prefix)
        checks[f"prefix_{name}_member_count"] = len(tensors)
        checks[f"prefix_{name}_same_object"] = all(
            tensor is tensors[0] for tensor in tensors
        )
        checks[f"prefix_{name}_torch_equal"] = all(
            torch.equal(tensor, tensors[0]) for tensor in tensors
        )
        if not checks[f"prefix_{name}_same_object"]:
            raise RuntimeError(
                f"routes sharing prefix {prefix} did not receive the exact prefix tensor"
            )

    encode_count = getattr(getattr(model, "evidence_encoder", None), "encode_count", None)
    if encode_count is None and isinstance(raw_shared, Mapping):
        encode_count = raw_shared.get("history_encode_count")
    checks["history_encode_count"] = None if encode_count is None else int(encode_count)
    if encode_count is not None and int(encode_count) != 1:
        raise RuntimeError(
            f"rollout_all_routes_shared encoded history {encode_count} times, expected 1"
        )
    return checks


def curriculum_assignments(num_batches: int, seed: int, epoch: int) -> list[str]:
    if num_batches <= 0:
        return []
    canonical_count = max(1, math.ceil(CANONICAL_FRACTION * num_batches))
    if num_batches >= 2:
        canonical_count = min(canonical_count, num_batches - 1)
    assignments = ["canonical"] * canonical_count + ["all_routes_shared"] * (
        num_batches - canonical_count
    )
    random.Random(seed * 100_003 + epoch).shuffle(assignments)
    return assignments


def route_edges(route: Sequence[int]) -> set[tuple[int, int]]:
    return set(zip((0, *tuple(route)[:-1]), route))


def curriculum_report(assignments: Sequence[str], routes=EXPECTED_ROUTES) -> dict:
    covered: set[tuple[int, int]] = set()
    if "canonical" in assignments:
        covered |= route_edges(EXPECTED_CANONICAL_ROUTE)
    if "all_routes_shared" in assignments:
        for route in routes:
            covered |= route_edges(route)
    required = set().union(*(route_edges(route) for route in routes))
    canonical_fraction = (
        assignments.count("canonical") / len(assignments) if assignments else 0.0
    )
    minimum_fraction = CANONICAL_FRACTION
    if len(assignments) >= 2:
        minimum_fraction = min(CANONICAL_FRACTION, (len(assignments) - 1) / len(assignments))
    return {
        "batches": len(assignments),
        "counts": dict(Counter(assignments)),
        "canonical_fraction": canonical_fraction,
        "canonical_fraction_meets_protocol": (
            canonical_fraction + 1e-12 >= minimum_fraction
        ),
        "covered_edges": sorted(covered),
        "required_edges": sorted(required),
        "missing_edges": sorted(required - covered),
    }


def _conditioner_index(model) -> dict[int, int]:
    candidates = [
        getattr(getattr(model, "reasoner", None), "conditioner", None),
        getattr(model, "conditioner", None),
    ]
    for candidate in candidates:
        index = getattr(candidate, "index", None)
        if isinstance(index, Mapping):
            return {int(key): int(value) for key, value in index.items()}
    raise AttributeError("route-complete model must expose conditioner.index")


def map_formal_v1_weights(
    model, source_state: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Map protected V1 weights while retaining initialized 2/4 embedding rows."""
    dedicated_loader = getattr(model, "load_v1_state_dict", None)
    if dedicated_loader is not None:
        report = dict(dedicated_loader(source_state))
        report.update(
            {
                "optimizer_state_loaded": False,
                "new_resolution_rows": [2, 4],
                "new_rows_initialized_by": (
                    "deterministic interpolation of formal conditioner rows"
                ),
            }
        )
        return report

    destination = model.state_dict()
    mapped = {name: tensor.clone() for name, tensor in destination.items()}
    copied: list[str] = []
    expanded: dict[str, dict[str, Any]] = {}
    new_index = _conditioner_index(model)

    for name, target in destination.items():
        source = source_state.get(name)
        if source is None:
            continue
        if source.shape == target.shape:
            mapped[name] = source.detach().to(dtype=target.dtype)
            copied.append(name)
            continue
        if name.endswith(("conditioner.src_embedding.weight", "conditioner.dst_embedding.weight")):
            if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
                raise RuntimeError(f"cannot map conditioner embedding {name}")
            value = target.clone()
            rows = {}
            for resolution, old_row in OLD_CONDITIONER_INDEX.items():
                if resolution not in new_index:
                    raise RuntimeError(f"new conditioner omits resolution {resolution}")
                new_row = new_index[resolution]
                value[new_row].copy_(source[old_row].to(dtype=value.dtype))
                rows[str(resolution)] = {"source_row": old_row, "destination_row": new_row}
            mapped[name] = value
            expanded[name] = {
                "source_shape": list(source.shape),
                "destination_shape": list(target.shape),
                "mapped_rows": rows,
                "new_rows_retained_from_initialization": [
                    new_index[value] for value in (2, 4)
                ],
            }
    incompatible = model.load_state_dict(mapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"mapped load was not strict: {incompatible}")
    source_parameter_count = sum(tensor.numel() for tensor in source_state.values())
    copied_parameter_count = sum(destination[name].numel() for name in copied)
    copied_parameter_count += sum(
        len(OLD_CONDITIONER_INDEX) * destination[name].shape[1] for name in expanded
    )
    return {
        "source_tensors": len(source_state),
        "destination_tensors": len(destination),
        "shape_compatible_tensors": len(copied),
        "expanded_conditioner_tensors": expanded,
        "copied_parameters": int(copied_parameter_count),
        "source_parameters": int(source_parameter_count),
        "new_resolution_rows": [2, 4],
        "optimizer_state_loaded": False,
    }


def _masked_state_mae(prediction, target, resolution, rescale):
    state_target = temporal_mean_pool(target, int(resolution))
    return masked_mae(rescale(prediction), rescale(state_target), NULL_VAL)


def canonical_loss(model, history, target, rescale):
    output = model.rollout(history, EXPECTED_CANONICAL_ROUTE)
    loss = history.new_zeros(())
    for resolution, prediction, weight in zip(
        EXPECTED_CANONICAL_ROUTE,
        output["forecasts"],
        CANONICAL_STATE_WEIGHTS,
    ):
        loss = loss + float(weight) * _masked_state_mae(
            prediction, target, resolution, rescale
        )
    return loss


def all_route_shared_loss(model, history, target, rescale, state_weight: float):
    raw, outputs = call_shared_rollout(model, history)
    identity = assert_exact_shared_prefixes(model, raw, outputs)
    final_losses = [
        masked_mae(rescale(output["pred"]), rescale(target), NULL_VAL)
        for output in outputs.values()
    ]
    final_loss = torch.stack(final_losses).mean()

    # Supervise every unique nonterminal state once. Shared prefixes therefore
    # receive one gradient contribution rather than being duplicated per route.
    seen: set[int] = set()
    state_losses = []
    for route, output in outputs.items():
        for resolution, prediction in zip(route[:-1], output["forecasts"][:-1]):
            if id(prediction) in seen:
                continue
            seen.add(id(prediction))
            state_losses.append(
                _masked_state_mae(prediction, target, resolution, rescale)
            )
    auxiliary = (
        torch.stack(state_losses).mean() if state_losses else history.new_zeros(())
    )
    return final_loss + float(state_weight) * auxiliary, identity


@torch.inference_mode()
def evaluate_all_routes(model, loader, device, rescale, max_batches=None):
    model.eval()
    batch_mae = {route: [] for route in EXPECTED_ROUTES}
    identity = None
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history, target, _ = select_batch(batch, device)
        raw, outputs = call_shared_rollout(model, history)
        current_identity = assert_exact_shared_prefixes(model, raw, outputs)
        identity = identity or current_identity
        target_raw = rescale(target)
        for route, output in outputs.items():
            batch_mae[route].append(
                float(masked_mae(rescale(output["pred"]), target_raw, NULL_VAL))
            )
        samples += int(history.shape[0])
    if samples == 0:
        raise RuntimeError("VALID evaluation received zero samples")
    means = {route: float(np.mean(values)) for route, values in batch_mae.items()}
    canonical_mae = means[EXPECTED_CANONICAL_ROUTE]
    mean_route_mae = float(np.mean(list(means.values())))
    return {
        "samples": samples,
        "route_batch_mean_MAE": {
            route_key(route): means[route] for route in EXPECTED_ROUTES
        },
        "canonical_MAE": canonical_mae,
        "mean_8route_MAE": mean_route_mae,
        "selection_composite": 0.5 * canonical_mae + 0.5 * mean_route_mae,
        "shared_prefix_identity": identity,
    }


def _formal_valid_mae(checkpoint: Mapping[str, Any], override: float | None) -> float:
    if override is not None:
        return float(override)
    best = checkpoint.get("best", {})
    for key in ("MAE", "mae", "valid_MAE"):
        if key in best:
            return float(best[key])
    raise KeyError(
        "formal checkpoint lacks best VALID MAE; pass --formal-valid-mae explicitly"
    )


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, history, protocol):
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
            "routes": EXPECTED_ROUTES,
            "canonical_route": EXPECTED_CANONICAL_ROUTE,
            "method": "F2FCoTResolutionNativeRouteCompleteNet",
            "protocol": protocol,
            "test": None,
        },
        path,
    )


def train(model, train_loader, valid_loader, device, rescale, checkpoint_dir, result_dir, args, protocol):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    milestones = sorted(
        {
            max(1, int(args.epochs * 0.50)),
            max(1, int(args.epochs * 0.75)),
            max(1, int(args.epochs * 0.90)),
        }
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5
    )
    best_path = checkpoint_dir / "route_complete_continuation_best_valid.pt"
    last_path = checkpoint_dir / "route_complete_continuation_last.pt"
    best = {
        "selection_composite": math.inf,
        "epoch": 0,
        "eligible": False,
        "canonical_MAE": math.inf,
    }
    history_rows = []
    start_epoch = 1
    if args.resume and last_path.is_file():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        best = dict(saved["best"])
        history_rows = list(saved.get("history", []))
        start_epoch = int(saved["epoch"]) + 1

    effective_batches = min(
        len(train_loader),
        args.max_train_batches if args.max_train_batches is not None else len(train_loader),
    )
    containment_tolerance = (
        10.0 if args.smoke else CONTAINMENT_TOLERANCE
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        assignments = curriculum_assignments(effective_batches, args.seed, epoch)
        curriculum = curriculum_report(assignments)
        if not curriculum["canonical_fraction_meets_protocol"]:
            raise RuntimeError("curriculum violated canonical >=60% (or max feasible)")
        if effective_batches >= 2 and curriculum["missing_edges"]:
            raise RuntimeError(f"curriculum omitted edges: {curriculum['missing_edges']}")
        losses = []
        used = Counter()
        prefix_identity = None
        epoch_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= effective_batches:
                break
            history, target, _ = select_batch(batch, device)
            assignment = assignments[batch_index]
            if assignment == "canonical":
                loss = canonical_loss(model, history, target, rescale)
            else:
                loss, prefix_identity = all_route_shared_loss(
                    model, history, target, rescale, args.state_weight
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch={epoch} batch={batch_index}"
                )
            (loss / float(args.gradient_accumulation)).backward()
            should_step = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == effective_batches
            )
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("non-finite gradient norm")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            used[assignment] += 1

        valid = evaluate_all_routes(
            model, valid_loader, device, rescale, args.max_valid_batches
        )
        eligible = (
            valid["canonical_MAE"]
            <= args.formal_valid_mae_resolved + containment_tolerance
        )
        score = float(valid["selection_composite"]) if eligible else math.inf
        improved = eligible and score < float(best["selection_composite"])
        if improved:
            best = {
                "selection_composite": score,
                "epoch": int(epoch),
                "eligible": True,
                "canonical_MAE": float(valid["canonical_MAE"]),
                "mean_8route_MAE": float(valid["mean_8route_MAE"]),
            }
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "route_batches": dict(used),
            "curriculum": curriculum,
            "prefix_identity_training": prefix_identity,
            "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
            "eligible": bool(eligible),
            "improved": bool(improved),
            "valid": valid,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history_rows.append(row)
        if improved:
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch, best, history_rows, protocol
            )
        save_checkpoint(
            last_path, model, optimizer, scheduler, epoch, best, history_rows, protocol
        )
        dump_json(result_dir / "training_history.json", history_rows)
        print(
            f"[route-complete] epoch={epoch:02d} loss={row['train_loss']:.4f} "
            f"canonical={valid['canonical_MAE']:.4f} "
            f"mean8={valid['mean_8route_MAE']:.4f} eligible={eligible} "
            f"best={best['selection_composite']:.4f}",
            flush=True,
        )
    if not best_path.is_file():
        raise RuntimeError("no checkpoint passed canonical containment")
    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    return best, history_rows, best_path


def self_test() -> None:
    for batches in (2, 3, 10, 101):
        assignments = curriculum_assignments(batches, seed=7, epoch=2)
        report = curriculum_report(assignments)
        assert report["canonical_fraction_meets_protocol"]
        assert not report["missing_edges"]
    assert route_tuple("2->4->12") == (2, 4, 12)
    print("route-complete training helper smoke passed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--state-weight", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--tag", default="continuation_v1")
    parser.add_argument("--formal-checkpoint", type=Path, default=FORMAL_CHECKPOINT)
    parser.add_argument("--formal-valid-mae", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-valid-batches", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 2)
        args.workers = 0
        args.max_train_batches = 2
        args.max_valid_batches = 2
        args.tag = "smoke"
    elif not 12 <= args.epochs <= 20:
        raise ValueError("continuation protocol is intentionally limited to 12-20 epochs")
    if args.gradient_accumulation < 1:
        raise ValueError("--gradient-accumulation must be positive")

    seed_all(args.seed)
    model_class, routes, canonical = load_architecture()
    if not args.formal_checkpoint.is_file():
        raise FileNotFoundError(f"missing protected formal checkpoint: {args.formal_checkpoint}")
    formal = torch.load(args.formal_checkpoint, map_location="cpu", weights_only=False)
    args.formal_valid_mae_resolved = _formal_valid_mae(formal, args.formal_valid_mae)
    args.containment_tolerance_resolved = (
        10.0 if args.smoke else CONTAINMENT_TOLERANCE
    )
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    model = model_class(**model_args()).to(device)
    mapping = map_formal_v1_weights(model, formal["model_state_dict"])

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
        "architecture_module": ARCH_MODULE,
        "routes": routes,
        "canonical_route": canonical,
        "formal_checkpoint_read_only": str(args.formal_checkpoint),
        "formal_VALID_MAE": args.formal_valid_mae_resolved,
        "weight_mapping": mapping,
        "new_optimizer": "Adam",
        "optimizer_state_from_formal_loaded": False,
        "micro_batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation
        ),
        "epochs": args.epochs,
        "canonical_batch_fraction_minimum": CANONICAL_FRACTION,
        "other_batches": "all 8 routes in one exact-shared-prefix DAG rollout",
        "loss": {
            "scale": "raw physical scale after inverse scaler",
            "metric": "masked MAE",
            "all_route_final_weight": 1.0,
            "unique_intermediate_state_weight": args.state_weight,
            "canonical_state_weights": CANONICAL_STATE_WEIGHTS,
        },
        "selection": (
            "VALID-only: minimize 0.5*canonical_MAE + 0.5*mean_8route_MAE "
            "among canonical-containment-eligible checkpoints"
        ),
        "containment_tolerance": args.containment_tolerance_resolved,
        "test_constructed": False,
    }
    dump_json(result_dir / "protocol.json", protocol)

    rescale = load_rescale()
    train_loader = make_loader("train", args.batch_size, True, args.workers)
    valid_loader = make_loader("valid", args.batch_size, False, args.workers)
    best, history, best_path = train(
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
    selected_valid = evaluate_all_routes(
        model, valid_loader, device, rescale, args.max_valid_batches
    )
    report = {
        "best": best,
        "selected_checkpoint": str(best_path),
        "selected_VALID": selected_valid,
        "epochs_completed": len(history),
        "canonical_containment": {
            "formal_VALID_MAE": args.formal_valid_mae_resolved,
            "tolerance": args.containment_tolerance_resolved,
            "selected_VALID_MAE": selected_valid["canonical_MAE"],
            "pass": selected_valid["canonical_MAE"]
            <= args.formal_valid_mae_resolved
            + args.containment_tolerance_resolved,
        },
        "test": None,
    }
    dump_json(result_dir / "continuation_report.json", report)
    print(f"[done] selected={best_path} report={result_dir / 'continuation_report.json'}")


if __name__ == "__main__":
    main()
