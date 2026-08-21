"""Post-selection audit. Safe only after F2FCoT methodology_fixed=true."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.F2FCoT_arch import F2FCoTNet
from scripts.f2f_cot_runtime import (
    EXTRA_ROUTE,
    FIXED_ROUTE,
    canonical_args,
    cot_args,
    evaluate_canonical,
    load_canonical,
    load_rescale,
    make_loader,
    profile_reasoning_steps,
    select_batch,
    weighted_route_loss,
)


REPORT = ROOT / "results" / "f2f_cot" / "formal_v1_seed1" / "final_report.json"
CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "PEMS04"
    / "H12"
    / "f2f_cot"
    / "formal_v1_seed1"
    / "extra_best.pt"
)
OUTPUT = REPORT.parent / "postfixed_audit.json"


def latency_summary(values):
    ordered = sorted(float(value) for value in values)
    return {
        "median_ms": float(statistics.median(ordered)),
        "mean_ms": float(statistics.mean(ordered)),
        "p90_ms": float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]),
    }


def profile_training(model, history, target, rescale, route, warmup=10, repeats=30):
    model.train()
    route = tuple(route)
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        loss, _ = weighted_route_loss(model, history, target, route, rescale)
        loss.backward()
    torch.cuda.synchronize(history.device)
    elapsed = []
    torch.cuda.reset_peak_memory_stats(history.device)
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss, _ = weighted_route_loss(model, history, target, route, rescale)
        loss.backward()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    summary = latency_summary(elapsed)
    summary.update(
        {
            "batch_size": int(history.shape[0]),
            "reasoning_calls": len(route),
            "mean_ms_per_call_amortized": summary["mean_ms"] / len(route),
            "peak_allocated_MiB": torch.cuda.max_memory_allocated(history.device)
            / (1024.0**2),
            "definition": "forward + physical-scale weighted loss + backward; no optimizer step",
        }
    )
    return summary


def main():
    selected = json.loads(REPORT.read_text(encoding="utf-8"))
    if not selected.get("methodology_fixed"):
        raise RuntimeError("methodology is not fixed; TEST audit is forbidden")
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    rescale = load_rescale()
    test_loader = make_loader("test", 32, False, 2)

    canonical, canonical_meta = load_canonical(device)
    canonical_test = evaluate_canonical(canonical, test_loader, device, rescale)
    del canonical

    model = F2FCoTNet(**cot_args()).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    batch = next(iter(test_loader))
    history, target, _ = select_batch(batch, device)
    inference = {
        "3-6-12": profile_reasoning_steps(model, history[:1], FIXED_ROUTE),
        "3-4-6-12": profile_reasoning_steps(model, history[:1], EXTRA_ROUTE),
    }
    training = {
        "3-6-12": profile_training(
            model, history, target, rescale, FIXED_ROUTE
        ),
        "3-4-6-12": profile_training(
            model, history, target, rescale, EXTRA_ROUTE
        ),
    }
    payload = {
        "guard": "executed only after final_report.methodology_fixed=true",
        "selected_checkpoint": str(CHECKPOINT),
        "selected_epoch": int(checkpoint["epoch"]),
        "canonical_model_args_match": canonical_args(),
        "canonical_meta": canonical_meta,
        "canonical_TEST_once": canonical_test,
        "new_TEST_once": selected["test"],
        "inference_latency": inference,
        "training_cost": training,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
