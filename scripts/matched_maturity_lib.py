#!/usr/bin/env python3
"""Shared constants / protocol checks for matched-maturity crossfit v2.

Future adaptive-router protocol (explicit):
  - adaptive router train/selection -> split ONLY within matched OOF TRAIN
  - official VALID -> forecasting / reporting protocol only (NOT router selection)
  - TEST -> final frozen evaluation only

This module does NOT train any adaptive router.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

STABLE_CFG = (
    ROOT
    / "checkpoints/PEMS04/H12/budget_f2f/"
    / "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
    / "b5678fda5e8d94ed028c6c8bb073461d/"
    / "H12_supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6_seed1.py"
)
STABLE_CKPT = (
    ROOT
    / "checkpoints/PEMS04/H12/budget_f2f/"
    / "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
    / "b5678fda5e8d94ed028c6c8bb073461d/"
    / "BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
)
MANIFEST_PATH = ROOT / "results/matched_maturity_crossfit_manifest.json"
STABLE_TRAIN_ORACLE = ROOT / "results/pems04_budget_f2f_oracle_train_rawscale.json"
OLD_CF_ORACLE = ROOT / "results/pems04_temporal_crossfit_refinement_oracle.json"
OLD_CF_AGREE = ROOT / "results/rootcause_crossfit_vs_stable_gain_agreement.json"

TEACHER_ROOT = ROOT / "checkpoints/PEMS04/H12/budget_f2f/matched_maturity_crossfit_v2"
DATA_FOLD_ROOT = ROOT / "datasets/PEMS04/matched_maturity_folds"
GEN_CFG_ROOT = ROOT / "generated/temp_configs_matched_maturity"

FUTURE_PROTOCOL = {
    "adaptive_router_train_selection": "split only within matched OOF TRAIN",
    "official_VALID": "reporting / forecasting protocol only; NOT adaptive-router selection",
    "TEST": "final frozen evaluation only",
    "stable_TRAIN_oracle": "DIAGNOSTIC ONLY; never teacher/router supervision",
}

# Material protocol expectations (stable CFG overrides if different).
EXPECTED = {
    "architecture": "BudgetConditionedAdaptiveF2FNet",
    "training_phase": "supernet",
    "route_sampling": "sandwich",
    "loss_mode": "dynamic_fair",
    "optim_type": "Adam",
    "lr": 0.002,
    "weight_decay": 1e-4,
    "scheduler_type": "MultiStepLR",
    "milestones": [1, 35, 60, 80, 95],
    "gamma": 0.5,
    "batch_size": 32,
    "epochs": 100,
    "seed": 1,
    "checkpoint_selection": "min_val_MAE",
}

ROUTES = [[12], [6, 12], [3, 12], [3, 6, 12]]


def sha1_bytes(data: bytes, n: int | None = 16) -> str:
    h = hashlib.sha1(data).hexdigest()
    return h if n is None else h[:n]


def sha1_file(path: Path, n: int | None = 16) -> str:
    return sha1_bytes(Path(path).read_bytes(), n)


def sha1_indices(indices: list[int], n: int | None = 40) -> str:
    payload = ",".join(str(i) for i in sorted(int(x) for x in indices)).encode()
    return sha1_bytes(payload, n)


def sha1_state_dict(state_dict: dict, n: int | None = 40) -> str:
    import torch

    h = hashlib.sha1()
    for k in sorted(state_dict.keys()):
        h.update(k.encode())
        t = state_dict[k]
        if hasattr(t, "detach"):
            arr = t.detach().cpu().contiguous().numpy()
            h.update(arr.tobytes())
        else:
            h.update(str(t).encode())
    digest = h.hexdigest()
    return digest if n is None else digest[:n]


def fold_dir(fold: int, smoke: bool = False) -> Path:
    base = TEACHER_ROOT / ("smoke" if smoke else "formal") / f"fold{int(fold)}"
    return base


def load_manifest(path: Path | None = None) -> dict:
    p = Path(path) if path else MANIFEST_PATH
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text())


def load_stable_cfg():
    """Load CFG from the stable supernet config file (source of truth)."""
    from easytorch.config import import_config

    cfg_path = STABLE_CFG
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    rel = str(cfg_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    return import_config(rel), cfg_path


def extract_protocol_fields(cfg) -> dict[str, Any]:
    param = cfg.MODEL.PARAM
    optim = cfg.TRAIN.OPTIM
    sched = cfg.TRAIN.LR_SCHEDULER
    arch_name = getattr(cfg.MODEL.ARCH, "__name__", str(cfg.MODEL.ARCH))
    return {
        "architecture": arch_name,
        "model_name": cfg.MODEL.NAME,
        "training_phase": param.get("training_phase"),
        "route_sampling": param.get("route_sampling"),
        "loss_mode": param.get("loss_mode") or param.get("chain_loss_mode"),
        "chain_loss_mode": param.get("chain_loss_mode"),
        "optim_type": optim.TYPE,
        "lr": float(optim.PARAM.get("lr")),
        "weight_decay": float(optim.PARAM.get("weight_decay")),
        "scheduler_type": sched.TYPE,
        "milestones": list(sched.PARAM.get("milestones")),
        "gamma": float(sched.PARAM.get("gamma")),
        "batch_size": int(cfg.TRAIN.DATA.BATCH_SIZE),
        "epochs": int(cfg.TRAIN.NUM_EPOCHS),
        "seed": int(cfg.ENV.SEED),
        "candidate_routes": [list(r) for r in param.get("candidate_routes", [])],
        "forecast_state_adapter_mode": param.get("forecast_state_adapter_mode"),
        "use_forecast_state_adapter": param.get("use_forecast_state_adapter"),
        "checkpoint_selection": "min_val_MAE",
    }


def compare_protocol(actual: dict) -> dict:
    """Field-by-field comparison vs EXPECTED; stable CFG wins on conflict."""
    rows = []
    material_fail = []
    for key, exp in EXPECTED.items():
        got = actual.get(key)
        if key == "milestones":
            ok = list(got) == list(exp)
        elif isinstance(exp, float):
            ok = got is not None and abs(float(got) - float(exp)) < 1e-12
        else:
            ok = got == exp
        rows.append({"field": key, "expected": exp, "actual": got, "match": bool(ok)})
        if not ok:
            material_fail.append(key)
    # Also verify routes
    routes_ok = actual.get("candidate_routes") == ROUTES
    rows.append(
        {
            "field": "candidate_routes",
            "expected": ROUTES,
            "actual": actual.get("candidate_routes"),
            "match": routes_ok,
        }
    )
    if not routes_ok:
        material_fail.append("candidate_routes")
    return {
        "rows": rows,
        "material_mismatches": material_fail,
        "MATCHED_PROTOCOL_MISMATCH": bool(material_fail),
        "future_protocol": FUTURE_PROTOCOL,
    }


def verify_manifest_integrity(man: dict) -> dict:
    folds = man["folds"]
    assert int(man["K"]) == 5
    hold = []
    for f in folds:
        hold.extend(f["heldout_sample_indices"])
    hold_sorted = sorted(hold)
    ok_union = hold_sorted == list(range(10181))
    counts = [int(f["n_train_after_purge"]) for f in folds]
    ok_counts = min(counts) >= 8098 and max(counts) <= 8122
    ok_overlap = all(int(f["raw_window_overlap_after_purge"]) == 0 for f in folds)
    return {
        "holdout_union_exact_0_10180": ok_union,
        "n_unique_holdout": len(set(hold)),
        "train_count_min": min(counts),
        "train_count_max": max(counts),
        "train_count_range_ok": ok_counts,
        "zero_raw_overlap": ok_overlap,
        "pass": ok_union and ok_counts and ok_overlap,
    }


def get_test_access_count() -> int:
    return int(os.environ.get("MATCHED_MATURITY_TEST_ACCESS_COUNT", "0"))


def bump_test_access() -> int:
    n = get_test_access_count() + 1
    os.environ["MATCHED_MATURITY_TEST_ACCESS_COUNT"] = str(n)
    return n


def reset_test_access() -> None:
    os.environ["MATCHED_MATURITY_TEST_ACCESS_COUNT"] = "0"


def install_test_access_guard() -> None:
    """Raise if any TimeSeriesForecastingDataset is built with mode=test."""
    from basicts.data.dataset import TimeSeriesForecastingDataset

    if getattr(TimeSeriesForecastingDataset, "_matched_maturity_guard", False):
        return
    _orig = TimeSeriesForecastingDataset.__init__

    def _guarded(self, data_file_path, index_file_path, mode, *args, **kwargs):
        if str(mode) == "test":
            bump_test_access()
            raise RuntimeError(
                "TEST_ACCESS_FORBIDDEN: matched-maturity teachers must not "
                f"instantiate TEST (count={get_test_access_count()})"
            )
        return _orig(self, data_file_path, index_file_path, mode, *args, **kwargs)

    TimeSeriesForecastingDataset.__init__ = _guarded  # type: ignore
    TimeSeriesForecastingDataset._matched_maturity_guard = True  # type: ignore


def dump_json(path: Path, obj: Any) -> None:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
