#!/usr/bin/env python3
"""Paper efficiency profiling on PEMS04 H=12.

Profiles trainable params / peak GPU memory / inference latency for:
  STID, AGCRN, HyperD, Single-stage, Two-stage, F2FNet.

Does not train models or invent MAE numbers.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import pickle
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

HORIZON_SCRIPT = ROOT / "scripts" / "run_chain_forecasting_horizon_pems04.py"
VARIANT = "chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss"
INPUT_LEN = 12
OUTPUT_LEN = 12
SEED = 1
NODE_SIZE = 307
TEMP_CFG_DIR = ROOT / "tmp_configs" / "paper_efficiency"
FULL_VARIANT = VARIANT
PRECISION = "fp32"
MEMORY_MODE = "training_step"  # peak allocated during full train step
LATENCY_MODE = "model_forward_with_h2d"  # forward + CPU->GPU transfer, synced

MODEL_SPECS = {
    "stid": {
        "display": "STID",
        "schedule": None,
        "num_stages": None,
        "cfg_name": "pems04_h12_stid.py",
        "source_cfg": ROOT / "examples" / "baselines" / "STID" / "STID_PEMS04.py",
        "expected_arch": "STID",
        "spatial": None,
        "family": "basicts_simple",
    },
    "agcrn": {
        "display": "AGCRN",
        "schedule": None,
        "num_stages": None,
        "cfg_name": "pems04_h12_agcrn.py",
        "source_cfg": ROOT / "examples" / "baselines" / "AGCRN" / "AGCRN_PEMS04.py",
        "expected_arch": "AGCRN",
        "spatial": None,
        "family": "basicts_simple",
    },
    "hyperd": {
        "display": "HyperD",
        "schedule": None,
        "num_stages": None,
        "cfg_name": "pems04_h12_hyperd.py",
        "source_cfg": None,
        "expected_arch": "HyperD",
        "spatial": None,
        "family": "hyperd",
    },
    "single_stage": {
        "display": "Single-stage",
        "schedule": [12],
        "num_stages": 1,
        "cfg_name": "pems04_h12_single_stage.py",
        "source_cfg": None,
        "expected_arch": None,
        "spatial": {
            "progressive_spatial_ratios": [1.0],
            "progressive_spatial_topks": [32],
            "progressive_spatial_alphas": [0.10],
        },
        "family": "f2fnet",
    },
    "two_stage": {
        "display": "Two-stage",
        "schedule": [6, 12],
        "num_stages": 2,
        "cfg_name": "pems04_h12_two_stage.py",
        "source_cfg": None,
        "expected_arch": None,
        "spatial": {
            "progressive_spatial_ratios": [0.5, 1.0],
            "progressive_spatial_topks": [16, 32],
            "progressive_spatial_alphas": [0.06, 0.10],
        },
        "family": "f2fnet",
    },
    "f2fnet": {
        "display": "F2FNet",
        "schedule": [3, 6, 12],
        "num_stages": 3,
        "cfg_name": "pems04_h12_f2fnet.py",
        "source_cfg": None,
        "expected_arch": None,
        "spatial": {
            "progressive_spatial_ratios": [0.25, 0.5, 1.0],
            "progressive_spatial_topks": [8, 16, 32],
            "progressive_spatial_alphas": [0.03, 0.06, 0.10],
        },
        "family": "f2fnet",
    },
}


def load_horizon_module():
    spec = importlib.util.spec_from_file_location("horizon_pems04", HORIZON_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def write_env_file(path: Path, args: argparse.Namespace, extra: dict) -> None:
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    lines = [
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"git_branch={branch}",
        f"git_commit={commit}",
        f"gpu={gpu_name}",
        f"cuda={torch.version.cuda}",
        f"torch={torch.__version__}",
        f"python={sys.version.split()[0]}",
        f"batch_size={args.batch_size}",
        f"precision={PRECISION}",
        f"warmup_steps={args.warmup_steps}",
        f"train_steps={args.train_steps}",
        f"infer_steps={args.infer_steps}",
        f"max_seconds={args.max_seconds}",
        f"memory_measurement_mode={MEMORY_MODE}",
        f"latency_measurement_mode={LATENCY_MODE}",
        f"models={' '.join(args.models)}",
    ]
    for k, v in extra.items():
        lines.append(f"{k}={v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_f2fnet_cfg(hz, model_key: str, seed: int = SEED, batch_size: int = 32) -> Path:
    TEMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    spec_info = MODEL_SPECS[model_key]
    base_spec = dict(hz.variant_spec(FULL_VARIANT, OUTPUT_LEN))
    base_spec["chain_lengths"] = list(spec_info["schedule"])
    for k, v in spec_info["spatial"].items():
        base_spec[k] = list(v)

    base_cfg = hz.base_cfg_for_variant(FULL_VARIANT)
    content = hz.strip_hardcoded_cuda_devices(base_cfg.read_text(encoding="utf-8"))
    ckpt_rel = os.path.join("checkpoints", "paper_efficiency", f"{model_key}_seed{seed}")
    lines = [
        "",
        f"# ===== paper_efficiency {model_key} overrides =====",
        f"CFG.ENV.SEED = {seed}",
        f"CFG.DATASET_INPUT_LEN = {INPUT_LEN}",
        f"CFG.DATASET_OUTPUT_LEN = {OUTPUT_LEN}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")',
        "CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]",
        "CFG.MODEL.TARGET_FEATURES = [0]",
        f'CFG.MODEL.PARAM["input_len"] = {INPUT_LEN}',
        f'CFG.MODEL.PARAM["output_len"] = {OUTPUT_LEN}',
        f'CFG.MODEL.PARAM["node_size"] = {NODE_SIZE}',
        f"CFG.TEST.EVALUATION_HORIZONS = list(range(1, {OUTPUT_LEN + 1}))",
        f"CFG.TRAIN.DATA.BATCH_SIZE = {batch_size}",
        f"CFG.VAL.DATA.BATCH_SIZE = {batch_size}",
        f"CFG.TEST.DATA.BATCH_SIZE = {batch_size}",
    ]
    model_name = base_spec.get("model_name")
    if model_name:
        lines.append(f"CFG.MODEL.NAME = {hz._py_literal(model_name)}")
    for key, val in base_spec.items():
        if key in hz._META_SPEC_KEYS:
            continue
        lines.append(f'CFG.MODEL.PARAM["{key}"] = {hz._py_literal(val)}')
    out = TEMP_CFG_DIR / spec_info["cfg_name"]
    out.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")
    return out


def generate_hyperd_cfg(batch_size: int) -> Path:
    TEMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    out = TEMP_CFG_DIR / MODEL_SPECS["hyperd"]["cfg_name"]
    text = f'''import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.HyperD.hyperd_config import build_cfg

CFG = build_cfg("PEMS04")
CFG.ENV.SEED = {SEED}
CFG.DATASET_INPUT_LEN = {INPUT_LEN}
CFG.DATASET_OUTPUT_LEN = {OUTPUT_LEN}
CFG.TRAIN.DATA.BATCH_SIZE = {batch_size}
CFG.VAL.DATA.BATCH_SIZE = {batch_size}
CFG.TEST.DATA.BATCH_SIZE = {batch_size}
CFG.MODEL.PARAM["seq_len"] = {INPUT_LEN}
CFG.MODEL.PARAM["pred_len"] = {OUTPUT_LEN}
'''
    out.write_text(text, encoding="utf-8")
    return out


def generate_basicts_cfg(model_key: str, batch_size: int) -> Path:
    """Copy official STID/AGCRN config; only override protocol fields."""
    TEMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    spec = MODEL_SPECS[model_key]
    src = Path(spec["source_cfg"])
    assert src.is_file(), src
    content = src.read_text(encoding="utf-8")
    lines = [
        "",
        f"# ===== paper_efficiency protocol overrides ({model_key}) =====",
        f"CFG.DATASET_INPUT_LEN = {INPUT_LEN}",
        f"CFG.DATASET_OUTPUT_LEN = {OUTPUT_LEN}",
        f"CFG.TRAIN.DATA.BATCH_SIZE = {batch_size}",
        f"CFG.VAL.DATA.BATCH_SIZE = {batch_size}",
        f"CFG.TEST.DATA.BATCH_SIZE = {batch_size}",
        # keep architecture PARAM untouched
    ]
    out = TEMP_CFG_DIR / spec["cfg_name"]
    out.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")
    return out


def validate_f2fnet_cfg(cfg, model_key: str) -> None:
    assert int(cfg.DATASET_INPUT_LEN) == INPUT_LEN, cfg.DATASET_INPUT_LEN
    assert int(cfg.DATASET_OUTPUT_LEN) == OUTPUT_LEN, cfg.DATASET_OUTPUT_LEN
    assert int(cfg.MODEL.PARAM["input_len"]) == INPUT_LEN
    assert int(cfg.MODEL.PARAM["output_len"]) == OUTPUT_LEN
    assert int(cfg.MODEL.PARAM["node_size"]) == NODE_SIZE
    assert list(cfg.MODEL.TARGET_FEATURES) == [0]
    assert list(cfg.MODEL.FORWARD_FEATURES) == [0, 1, 2, 3]
    expect = MODEL_SPECS[model_key]["schedule"]
    got = list(cfg.MODEL.PARAM["chain_lengths"])
    assert got == expect, f"{model_key} schedule {got} != {expect}"
    spat = MODEL_SPECS[model_key]["spatial"]
    for k, v in spat.items():
        assert list(cfg.MODEL.PARAM[k]) == list(v), f"{model_key}.{k} mismatch"


def validate_hyperd_cfg(cfg, batch_size: int = 32) -> None:
    assert int(cfg.DATASET_INPUT_LEN) == INPUT_LEN
    assert int(cfg.DATASET_OUTPUT_LEN) == OUTPUT_LEN
    assert int(cfg.MODEL.PARAM["num_nodes"]) == NODE_SIZE
    assert list(cfg.MODEL.TARGET_FEATURES) == [0]
    assert list(cfg.MODEL.FORWARD_FEATURES) == [0, 1, 2]
    assert int(cfg.MODEL.PARAM["seq_len"]) == INPUT_LEN
    assert int(cfg.MODEL.PARAM["pred_len"]) == OUTPUT_LEN
    assert int(cfg.TRAIN.DATA.BATCH_SIZE) == batch_size


def validate_basicts_cfg(cfg, model_key: str, batch_size: int) -> None:
    hz = load_horizon_module()
    src = hz.load_cfg(MODEL_SPECS[model_key]["source_cfg"])
    assert int(cfg.DATASET_INPUT_LEN) == INPUT_LEN
    assert int(cfg.DATASET_OUTPUT_LEN) == OUTPUT_LEN
    assert int(cfg.TRAIN.DATA.BATCH_SIZE) == batch_size
    assert cfg.MODEL.ARCH is src.MODEL.ARCH
    assert cfg.MODEL.ARCH.__name__ == MODEL_SPECS[model_key]["expected_arch"]
    # architecture hyperparameters must be unchanged vs official config
    for k, v in dict(src.MODEL.PARAM).items():
        assert cfg.MODEL.PARAM[k] == v, f"{model_key} PARAM[{k}] changed: {cfg.MODEL.PARAM[k]} vs {v}"
    assert list(cfg.MODEL.FORWARD_FEATURES) == list(src.MODEL.FORWARD_FEATURES)
    assert list(cfg.MODEL.TARGET_FEATURES) == list(src.MODEL.TARGET_FEATURES)
    assert cfg.TRAIN.LOSS is src.TRAIN.LOSS


def load_scaler(dataset_dir: Path, in_len: int, out_len: int) -> dict:
    path = dataset_dir / f"scaler_in{in_len}_out{out_len}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def build_holost_loaders(batch_size: int, forward_features: list[int] | None = None):
    """Shared PEMS04 holost loaders (4-channel). Feature selection happens in steps."""
    from basicts.data import TimeSeriesForecastingDataset

    data_dir = ROOT / "datasets" / "PEMS04"
    data_file = str(data_dir / f"data_in{INPUT_LEN}_out{OUTPUT_LEN}.pkl")
    index_file = str(data_dir / f"index_in{INPUT_LEN}_out{OUTPUT_LEN}.pkl")
    assert Path(data_file).is_file(), data_file
    assert Path(index_file).is_file(), index_file
    train_ds = TimeSeriesForecastingDataset(data_file_path=data_file, index_file_path=index_file, mode="train")
    test_ds = TimeSeriesForecastingDataset(data_file_path=data_file, index_file_path=index_file, mode="test")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    fut, hist = next(iter(train_loader))
    if tuple(hist.shape[1:]) != (INPUT_LEN, NODE_SIZE, 4):
        raise RuntimeError(f"unexpected hist shape {tuple(hist.shape)}")
    if tuple(fut.shape[1:]) != (OUTPUT_LEN, NODE_SIZE, 4):
        raise RuntimeError(f"unexpected fut shape {tuple(fut.shape)}")
    if forward_features is not None:
        # ensure selected features are available
        assert max(forward_features) < hist.shape[-1]
    return train_loader, test_loader, len(train_ds), len(test_ds)


def build_hyperd_loaders(cfg, batch_size: int):
    from baselines.HyperD.hyperd_runner import HyperDRunner

    train_ds = HyperDRunner._build_hyperd_dataset(cfg, "train")
    test_ds = HyperDRunner._build_hyperd_dataset(cfg, "test")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    fut, hist = next(iter(train_loader))
    if hist.shape[1] != INPUT_LEN or hist.shape[2] != NODE_SIZE:
        raise RuntimeError(f"unexpected HyperD hist shape {tuple(hist.shape)}")
    if fut.shape[1] != OUTPUT_LEN:
        raise RuntimeError(f"unexpected HyperD fut shape {tuple(fut.shape)}")
    return train_loader, test_loader, len(train_ds), len(test_ds)


def find_checkpoint(model_key: str, cfg) -> tuple[bool, str | None, str]:
    if model_key == "f2fnet":
        base = ROOT / "checkpoints" / "fixed_input_horizon_pems04" / "h12" / f"{FULL_VARIANT}_seed{SEED}"
        if base.is_dir():
            cands = sorted(base.rglob("*.pt")) + sorted(base.rglob("*.pth"))
            for p in cands:
                if "best" in p.name.lower() or p.suffix in {".pt", ".pth"}:
                    return True, str(p), "trained"
            if cands:
                return True, str(cands[0]), "trained"
    if model_key == "hyperd":
        base = ROOT / "checkpoints" / "baselines" / "HyperD_PEMS04_12to12"
        if base.is_dir():
            cands = sorted(base.rglob("*.pt")) + sorted(base.rglob("*.pth"))
            if cands:
                return True, str(cands[0]), "trained"
    if model_key in {"stid", "agcrn"}:
        name = MODEL_SPECS[model_key]["expected_arch"]
        base = ROOT / "checkpoints" / "baselines"
        if base.is_dir():
            for d in sorted(base.glob(f"{name}_PEMS04*")):
                cands = sorted(d.rglob("*.pt")) + sorted(d.rglob("*.pth"))
                if cands:
                    return True, str(cands[0]), "trained"
    return False, None, "initialized"


def try_load_weights(model: torch.nn.Module, ckpt_path: str | None) -> bool:
    if not ckpt_path:
        return False
    try:
        obj = torch.load(ckpt_path, map_location="cpu")
        state = obj
        if isinstance(obj, dict):
            for key in ("model_state_dict", "state_dict", "model", "network"):
                if key in obj and isinstance(obj[key], dict):
                    state = obj[key]
                    break
        if not isinstance(state, dict):
            return False
        cleaned = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=False)
        return True
    except Exception:
        return False


def select_features(x: torch.Tensor, feats: list[int]) -> torch.Tensor:
    return x[..., feats]


def f2fnet_train_step(model, batch, cfg, scaler, device, optimizer):
    from basicts.archs.arch_zoo.ChainForecasting_arch import ChainForecasting
    from basicts.data import SCALER_REGISTRY
    from basicts.losses.forecast_state_token_mae import forecast_state_token_mae

    future_data, history_data = batch
    history_data = history_data.to(device, non_blocking=True)
    future_data = future_data.to(device, non_blocking=True)
    hist = select_features(history_data, list(cfg.MODEL.FORWARD_FEATURES))
    fut = select_features(future_data, list(cfg.MODEL.FORWARD_FEATURES))
    target = select_features(future_data, list(cfg.MODEL.TARGET_FEATURES))

    optimizer.zero_grad(set_to_none=True)
    out = model(
        history_data=hist,
        future_data=fut,
        batch_seen=0,
        epoch=1,
        train=True,
        return_all=True,
    )
    chain_lengths = list(cfg.MODEL.PARAM["chain_lengths"])
    preds = list(out["chain_preds"])
    targets = [ChainForecasting.pool_target(target, k) for k in chain_lengths]

    def rescale_pair(pred, tgt):
        pred_r = SCALER_REGISTRY.get(scaler["func"])(pred, **scaler["args"])
        tgt_r = SCALER_REGISTRY.get(scaler["func"])(tgt, **scaler["args"])
        return pred_r, tgt_r

    loss = forecast_state_token_mae(
        preds,
        targets,
        null_val=float(getattr(cfg.TRAIN, "NULL_VAL", 0.0)),
        rescale_pair=rescale_pair,
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def f2fnet_infer_step(model, batch, cfg, device):
    future_data, history_data = batch
    history_data = history_data.to(device, non_blocking=True)
    future_data = future_data.to(device, non_blocking=True)
    hist = select_features(history_data, list(cfg.MODEL.FORWARD_FEATURES))
    fut = select_features(future_data, list(cfg.MODEL.FORWARD_FEATURES))
    _ = model(
        history_data=hist,
        future_data=fut,
        batch_seen=0,
        epoch=None,
        train=False,
        return_all=True,
    )


def hyperd_train_step(model, batch, cfg, scaler, device, optimizer, loss_fn):
    from basicts.data import SCALER_REGISTRY

    future_data, history_data = batch
    history_data = history_data.to(device, non_blocking=True)
    future_data = future_data.to(device, non_blocking=True)
    hist = select_features(history_data, list(cfg.MODEL.FORWARD_FEATURES))
    fut = select_features(future_data, list(cfg.MODEL.FORWARD_FEATURES))
    target = select_features(future_data, list(cfg.MODEL.TARGET_FEATURES))

    optimizer.zero_grad(set_to_none=True)
    model_return = model(
        history_data=hist,
        future_data=fut,
        batch_seen=0,
        epoch=1,
        train=True,
    )
    if isinstance(model_return, dict):
        prediction = model_return["prediction"]
        dual = model_return.get("dual_view_loss")
    elif isinstance(model_return, (tuple, list)):
        prediction = model_return[0]
        dual = model_return[1] if len(model_return) > 1 else None
    else:
        prediction = model_return
        dual = None
    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(-1)
    prediction = prediction[..., :1]
    pred_r = SCALER_REGISTRY.get(scaler["func"])(prediction, **scaler["args"])
    real_r = SCALER_REGISTRY.get(scaler["func"])(target, **scaler["args"])
    loss = loss_fn(pred_r, real_r)
    if dual is not None and torch.is_tensor(dual):
        loss = loss + dual
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def hyperd_infer_step(model, batch, cfg, device):
    future_data, history_data = batch
    history_data = history_data.to(device, non_blocking=True)
    future_data = future_data.to(device, non_blocking=True)
    hist = select_features(history_data, list(cfg.MODEL.FORWARD_FEATURES))
    fut = select_features(future_data, list(cfg.MODEL.FORWARD_FEATURES)).clone()
    fut[..., 0] = torch.empty_like(fut[..., 0])
    _ = model(history_data=hist, future_data=fut, batch_seen=0, epoch=None, train=False)


def basicts_simple_train_step(model, batch, cfg, scaler, device, optimizer):
    """Mirror SimpleTimeSeriesForecastingRunner.forward + BaseTimeSeriesForecastingRunner.train_iters loss."""
    from basicts.data import SCALER_REGISTRY

    future_data, history_data = batch
    history_data = history_data.to(device, non_blocking=True)
    future_data = future_data.to(device, non_blocking=True)
    hist = select_features(history_data, list(cfg.MODEL.FORWARD_FEATURES))
    fut_in = select_features(future_data, list(cfg.MODEL.FORWARD_FEATURES))
    target = select_features(future_data, list(cfg.MODEL.TARGET_FEATURES))

    optimizer.zero_grad(set_to_none=True)
    pred = model(
        history_data=hist,
        future_data=fut_in,
        batch_seen=0,
        epoch=1,
        train=True,
    )
    prediction = select_features(pred, list(cfg.MODEL.TARGET_FEATURES))
    pred_r = SCALER_REGISTRY.get(scaler["func"])(prediction, **scaler["args"])
    real_r = SCALER_REGISTRY.get(scaler["func"])(target, **scaler["args"])
    null_val = float(getattr(cfg.TRAIN, "NULL_VAL", getattr(cfg, "NULL_VAL", 0.0)))
    loss = cfg.TRAIN.LOSS(pred_r, real_r, null_val=null_val)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def basicts_simple_infer_step(model, batch, cfg, device):
    future_data, history_data = batch
    history_data = history_data.to(device, non_blocking=True)
    future_data = future_data.to(device, non_blocking=True)
    hist = select_features(history_data, list(cfg.MODEL.FORWARD_FEATURES))
    fut_in = select_features(future_data, list(cfg.MODEL.FORWARD_FEATURES))
    _ = model(
        history_data=hist,
        future_data=fut_in,
        batch_seen=0,
        epoch=None,
        train=False,
    )


def timed_loop(steps, iterator, step_fn, device):
    times = []
    it = iter(iterator)
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(iterator)
            batch = next(it)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        step_fn(batch)
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def empty_row(model_key: str) -> dict:
    return {
        "model": MODEL_SPECS[model_key]["display"],
        "model_key": model_key,
        "schedule": "" if MODEL_SPECS[model_key]["schedule"] is None else str(MODEL_SPECS[model_key]["schedule"]),
        "num_stages": MODEL_SPECS[model_key]["num_stages"] if MODEL_SPECS[model_key]["num_stages"] is not None else "",
        "status": "ok",
        "error": "",
        "checkpoint_loaded": False,
        "checkpoint_path": "",
        "weight_state": "initialized",
        "train_steps_used": 0,
        "infer_steps_used": 0,
        "train_epoch_time_mode": "estimated",
        "config_path": "",
        "model_class": "",
        "batch_size": "",
        "precision": PRECISION,
        "memory_measurement_mode": MEMORY_MODE,
        "latency_measurement_mode": LATENCY_MODE,
        "warmup_steps": "",
        "measured_steps": "",
        "protocol_valid": "",
    }


def profile_one(model_key: str, args: argparse.Namespace) -> dict:
    hz = load_horizon_module()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    row = empty_row(model_key)
    row["batch_size"] = int(args.batch_size)
    row["warmup_steps"] = int(args.warmup_steps)

    try:
        family = MODEL_SPECS[model_key]["family"]
        if family == "f2fnet":
            cfg_path = generate_f2fnet_cfg(hz, model_key, batch_size=args.batch_size)
            cfg = hz.load_cfg(cfg_path)
            validate_f2fnet_cfg(cfg, model_key)
            train_loader, test_loader, n_train, n_test = build_holost_loaders(args.batch_size)
            scaler = load_scaler(ROOT / "datasets" / "PEMS04", INPUT_LEN, OUTPUT_LEN)
            model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM).to(device)
            optim = torch.optim.Adam(model.parameters(), lr=float(cfg.TRAIN.OPTIM.PARAM.get("lr", 0.002)))

            def train_fn(batch):
                return f2fnet_train_step(model, batch, cfg, scaler, device, optim)

            def infer_fn(batch):
                return f2fnet_infer_step(model, batch, cfg, device)

        elif family == "hyperd":
            from baselines.HyperD.data_prepare import ensure_hyperd_data, init_npy_paths
            from baselines.HyperD.hyperd_runner import ensure_hyperd_scaler
            from baselines.HyperD.Initialization import run_initialization

            ensure_hyperd_data("PEMS04")
            daily, weekly = init_npy_paths("PEMS04")
            if not (daily.is_file() and weekly.is_file()):
                run_initialization("PEMS04")
            ensure_hyperd_scaler("PEMS04", INPUT_LEN, OUTPUT_LEN)
            cfg_path = generate_hyperd_cfg(args.batch_size)
            cfg = hz.load_cfg(cfg_path)
            validate_hyperd_cfg(cfg, args.batch_size)
            train_loader, test_loader, n_train, n_test = build_hyperd_loaders(cfg, args.batch_size)
            scaler = load_scaler(ROOT / "datasets" / "PEMS04", INPUT_LEN, OUTPUT_LEN)
            param = dict(cfg.MODEL.PARAM)
            if torch.is_tensor(param.get("adj")):
                param["adj"] = param["adj"].to(device)
            model = cfg.MODEL.ARCH(**param).to(device)
            optim = torch.optim.Adam(model.parameters(), lr=float(cfg.TRAIN.OPTIM.PARAM.get("lr", 0.005)))
            loss_fn = cfg.TRAIN.LOSS

            def train_fn(batch):
                return hyperd_train_step(model, batch, cfg, scaler, device, optim, loss_fn)

            def infer_fn(batch):
                return hyperd_infer_step(model, batch, cfg, device)

        elif family == "basicts_simple":
            cfg_path = generate_basicts_cfg(model_key, args.batch_size)
            cfg = hz.load_cfg(cfg_path)
            validate_basicts_cfg(cfg, model_key, args.batch_size)
            train_loader, test_loader, n_train, n_test = build_holost_loaders(
                args.batch_size, list(cfg.MODEL.FORWARD_FEATURES)
            )
            scaler = load_scaler(ROOT / "datasets" / "PEMS04", INPUT_LEN, OUTPUT_LEN)
            # instantiate with official PARAM only
            model = cfg.MODEL.ARCH(**dict(cfg.MODEL.PARAM)).to(device)
            if model.__class__.__name__ != MODEL_SPECS[model_key]["expected_arch"]:
                raise RuntimeError(
                    f"model class {model.__class__.__name__} != {MODEL_SPECS[model_key]['expected_arch']}"
                )
            optim = torch.optim.Adam(model.parameters(), lr=float(cfg.TRAIN.OPTIM.PARAM.get("lr", 0.002)))

            def train_fn(batch):
                return basicts_simple_train_step(model, batch, cfg, scaler, device, optim)

            def infer_fn(batch):
                return basicts_simple_infer_step(model, batch, cfg, device)

        else:
            raise ValueError(f"unknown family {family}")

        row["config_path"] = str(cfg_path)
        row["model_class"] = model.__class__.__name__

        found, ckpt, _ = find_checkpoint(model_key, cfg)
        loaded = try_load_weights(model, ckpt) if (args.checkpoints != "none" and found) else False
        row["checkpoint_loaded"] = bool(loaded)
        row["checkpoint_path"] = ckpt or ""
        row["weight_state"] = "trained" if loaded else "initialized"

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if params <= 0:
            raise RuntimeError("non-positive trainable parameter count")
        row["params"] = int(params)
        row["params_m"] = float(f"{params / 1e6:.3f}")
        row["num_train_batches"] = math.ceil(n_train / args.batch_size)
        row["num_test_batches"] = math.ceil(n_test / args.batch_size)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        model.train()
        it = iter(train_loader)
        for _ in range(args.warmup_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            train_fn(batch)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        train_times = timed_loop(args.train_steps, train_loader, train_fn, device)
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        row["peak_memory_allocated_mib"] = round(peak_alloc, 1)
        row["peak_memory_reserved_mib"] = round(peak_reserved, 1)
        row["train_ms_per_batch_mean"] = round(statistics.mean(train_times), 3)
        row["train_ms_per_batch_std"] = round(statistics.pstdev(train_times) if len(train_times) > 1 else 0.0, 3)
        row["train_ms_per_batch_median"] = round(statistics.median(train_times), 3)
        row["train_steps_used"] = len(train_times)
        row["estimated_train_seconds_per_epoch"] = round(
            row["train_ms_per_batch_median"] * row["num_train_batches"] / 1000.0, 3
        )

        model.eval()
        infer_warmup = max(20, int(args.warmup_steps))
        it = iter(test_loader)
        for _ in range(infer_warmup):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(test_loader)
                batch = next(it)
            infer_fn(batch)
        infer_times = timed_loop(args.infer_steps, test_loader, infer_fn, device)
        row["infer_ms_per_batch_mean"] = round(statistics.mean(infer_times), 3)
        row["infer_ms_per_batch_std"] = round(statistics.pstdev(infer_times) if len(infer_times) > 1 else 0.0, 3)
        row["infer_ms_per_batch_median"] = round(statistics.median(infer_times), 3)
        row["infer_steps_used"] = len(infer_times)
        row["measured_steps"] = len(infer_times)
        row["estimated_full_test_seconds"] = round(
            row["infer_ms_per_batch_median"] * row["num_test_batches"] / 1000.0, 3
        )
        row["memory_measurement_mode"] = MEMORY_MODE
        row["latency_measurement_mode"] = LATENCY_MODE
        row["precision"] = PRECISION
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def _parse_schedule_text(text: str) -> list[int] | None:
    import re

    m = re.search(r"\[([0-9,\s]+)\]", text)
    if not m:
        return None
    try:
        return [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
    except ValueError:
        return None


def _mae_row(model_key: str, schedule: list[int], mae: float | str, source: str, status: str) -> dict:
    return {
        "model": MODEL_SPECS[model_key]["display"],
        "schedule": str(schedule),
        "seed": 1,
        "dataset": "PEMS04",
        "input_len": 12,
        "output_len": 12,
        "test_mae": mae,
        "source": source,
        "match_status": status,
    }


def audit_existing_mae() -> list[dict]:
    """Search existing results for exact seed-1 MAE matches; never invent values."""
    import re

    wanted = {
        "single_stage": [12],
        "two_stage": [6, 12],
        "f2fnet": [3, 6, 12],
    }
    found: dict[str, dict] = {}
    candidates: list[Path] = []
    for base in (ROOT / "results", ROOT / "checkpoints", ROOT / "logs"):
        if not base.exists():
            continue
        candidates.extend(base.rglob("*.csv"))
        candidates.extend(base.rglob("*.md"))
        candidates.extend(base.rglob("*.json"))

    for path in candidates:
        name = path.name.lower()
        text_hint = str(path).lower()
        if "pems04" not in name and "pems04" not in text_hint:
            if VARIANT not in path.name and "state_adapter_fixed_token_loss" not in path.name:
                continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "PEMS04" not in raw and "pems04" not in raw.lower():
            continue
        if VARIANT not in raw and "state_adapter_fixed_token_loss" not in raw:
            continue
        if path.suffix == ".csv":
            try:
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        variant = (r.get("variant") or r.get("model") or "")
                        if VARIANT not in variant and "state_adapter_fixed_token_loss" not in variant:
                            continue
                        if str(r.get("seed", "")) not in {"1", "1.0"}:
                            continue
                        if str(r.get("horizon", r.get("output_len", "12"))) not in {"12", "12.0"}:
                            continue
                        mae = r.get("mae") or r.get("test_mae")
                        if mae in (None, "", "None"):
                            continue
                        cl = r.get("chain_lengths") or r.get("schedule") or ""
                        sched = _parse_schedule_text(cl) if cl else None
                        if sched is None and VARIANT in variant:
                            sched = [3, 6, 12]
                        if sched is None:
                            continue
                        if "token_loss" not in variant and "token" not in (r.get("loss") or "token").lower():
                            continue
                        for key, expect in wanted.items():
                            if sched == expect and key not in found:
                                found[key] = _mae_row(key, expect, float(mae), str(path), "exact_match")
            except Exception:
                continue
            continue
        if VARIANT not in raw:
            continue
        if not re.search(r"\bseed[_\s:=-]*1\b", raw, flags=re.I):
            continue
        for key, expect in wanted.items():
            if key in found:
                continue
            compact = str(expect).replace(" ", "")
            if compact not in raw.replace(" ", ""):
                continue
            spat = MODEL_SPECS[key]["spatial"]
            if spat is not None and key != "f2fnet":
                topk_mark = str(spat["progressive_spatial_topks"]).replace(" ", "")
                if topk_mark not in raw.replace(" ", ""):
                    continue
            m = re.search(r"(?:test[_\s-]*mae|MAE)\s*[:=]\s*([0-9]+\.[0-9]+)", raw, flags=re.I)
            if not m:
                continue
            found[key] = _mae_row(key, expect, float(m.group(1)), str(path), "exact_match")

    rows = []
    for key, sched in wanted.items():
        rows.append(found.get(key) or _mae_row(key, sched, "", "", "missing_exact_checkpoint"))
    return rows


def validate_protocol(rows: list[dict], args: argparse.Namespace) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if len(rows) != len(args.models):
        issues.append(f"row count {len(rows)} != models {len(args.models)}")
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if len(ok_rows) != len(args.models):
        issues.append("not all models status=ok")
        for r in rows:
            if r.get("status") != "ok":
                issues.append(f"{r.get('model_key')}: {r.get('status')} {r.get('error')}")
        return False, issues

    modes_mem = {r.get("memory_measurement_mode") for r in ok_rows}
    modes_lat = {r.get("latency_measurement_mode") for r in ok_rows}
    if modes_mem != {MEMORY_MODE}:
        issues.append(f"memory mode mismatch: {modes_mem}")
    if modes_lat != {LATENCY_MODE}:
        issues.append(f"latency mode mismatch: {modes_lat}")

    bs = {int(r.get("batch_size", -1)) for r in ok_rows}
    if bs != {int(args.batch_size)}:
        issues.append(f"batch_size mismatch: {bs}")
    prec = {r.get("precision") for r in ok_rows}
    if prec != {PRECISION}:
        issues.append(f"precision mismatch: {prec}")
    warm = {int(r.get("warmup_steps", -1)) for r in ok_rows}
    if warm != {int(args.warmup_steps)}:
        issues.append(f"warmup mismatch: {warm}")
    meas = {int(r.get("infer_steps_used", -1)) for r in ok_rows}
    if meas != {int(args.infer_steps)} and not all(int(r.get("infer_steps_used", 0)) >= 30 for r in ok_rows):
        issues.append(f"measured infer steps mismatch: {meas}")

    for r in ok_rows:
        key = r.get("model_key")
        exp = MODEL_SPECS[key].get("expected_arch")
        if exp and r.get("model_class") != exp:
            issues.append(f"{key} model_class {r.get('model_class')} != {exp}")
        if int(r.get("params", 0)) <= 0:
            issues.append(f"{key} non-positive params")
        if r.get("checkpoint_loaded") and False:
            pass  # loading ckpt is allowed; training is not

    # no full training launched: weight_state may be initialized or trained-from-ckpt only
    for r in ok_rows:
        if r.get("weight_state") not in {"initialized", "trained"}:
            issues.append(f"{r.get('model_key')} unexpected weight_state")

    return (len(issues) == 0), issues


def write_outputs(rows: list[dict], args: argparse.Namespace, mae_rows: list[dict], protocol_ok: bool, issues: list[str]) -> None:
    for r in rows:
        r["protocol_valid"] = bool(protocol_ok) and r.get("status") == "ok"

    fields = [
        "model", "schedule", "num_stages", "params", "params_m",
        "peak_memory_allocated_mib", "peak_memory_reserved_mib",
        "train_ms_per_batch_mean", "train_ms_per_batch_std", "train_ms_per_batch_median",
        "infer_ms_per_batch_mean", "infer_ms_per_batch_std", "infer_ms_per_batch_median",
        "num_train_batches", "num_test_batches",
        "estimated_train_seconds_per_epoch", "estimated_full_test_seconds",
        "train_epoch_time_mode",
        "checkpoint_loaded", "checkpoint_path", "weight_state",
        "train_steps_used", "infer_steps_used",
        "config_path", "model_class", "batch_size", "precision",
        "memory_measurement_mode", "latency_measurement_mode",
        "warmup_steps", "measured_steps", "protocol_valid",
        "status", "error",
    ]
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # markdown follows CLI order
    md = ["# Paper efficiency profiling (PEMS04, H=12)\n\n"]
    md.append(f"memory_measurement_mode = `{MEMORY_MODE}`  \n")
    md.append(f"latency_measurement_mode = `{LATENCY_MODE}`  \n")
    md.append(f"protocol_valid = `{protocol_ok}`  \n\n")
    if issues:
        md.append("Protocol issues:\n\n")
        for iss in issues:
            md.append(f"- {iss}\n")
        md.append("\n")
    md.append(
        "| Model | Schedule | Stages | Params (M) | Peak Mem. (MiB) | Infer. (ms/batch) | class | weight |\n"
    )
    md.append("|---|---|---:|---:|---:|---:|---|---|\n")
    for r in rows:
        sched = r.get("schedule") or "--"
        stages = r.get("num_stages") if r.get("num_stages") != "" else "--"
        pm = r.get("params_m", "")
        pm_s = f"{float(pm):.3f}" if pm not in ("", None) else ""
        md.append(
            f"| {r.get('model')} | `{sched}` | {stages} | {pm_s} | "
            f"{r.get('peak_memory_allocated_mib','')} | {r.get('infer_ms_per_batch_median','')} | "
            f"{r.get('model_class','')} | {r.get('weight_state','')} |\n"
        )
    Path(args.markdown).write_text("".join(md), encoding="utf-8")

    latex_path = Path(args.latex)
    if not protocol_ok:
        latex_path.write_text(
            "% invalid_protocol — LaTeX table not written.\n"
            + "\n".join(f"% {iss}" for iss in issues)
            + "\n",
            encoding="utf-8",
        )
    else:
        def cell(r, key, default="--"):
            v = r.get(key, default)
            return default if v in ("", None) else v

        use_compact = len(rows) >= 6
        colsep = "3pt" if use_compact else "3.4pt"
        conf_header = r"\textbf{Model}" if use_compact else r"\textbf{Configuration}"
        mem_header = r"\makecell{\textbf{Memory}\\\textbf{(MiB)}}" if use_compact else r"\makecell{\textbf{Peak Mem.}\\\textbf{(MiB)}}"

        note = (
            r"\vspace{-2pt}" "\n"
            r"{\fontsize{8pt}{9.5pt}\selectfont\noindent "
            r"All models are profiled on the same GPU with a batch size of 32. "
            r"Peak memory denotes the maximum allocated GPU memory during a training step. "
            r"Inference latency denotes the median model-forward time per batch after warm-up.}"
            "\n"
        )

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Computational profiles on PEMS04 with a forecasting horizon of 12.}",
            r"\label{tab:efficiency}",
            r"\fontsize{8.5pt}{10pt}\selectfont",
            rf"\setlength{{\tabcolsep}}{{{colsep}}}",
            r"\renewcommand{\arraystretch}{1.12}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            conf_header,
            r"& \textbf{Schedule}",
            r"& \textbf{Stages}",
            r"& \makecell{\textbf{Params}\\\textbf{(M)}}",
            f"& {mem_header}",
            r"& \makecell{\textbf{Infer.}\\\textbf{(ms/batch)}} \\",
            r"\midrule",
        ]
        for i, r in enumerate(rows):
            key = r.get("model_key")
            name = r["model"]
            if key == "f2fnet":
                name = r"\textbf{F2FNet}"
            sched = r.get("schedule") or "--"
            if sched and sched != "--":
                sched_tex = "$" + sched.replace(" ", "") + "$"
                sched_tex = sched_tex.replace(",", ",\\,")
            else:
                sched_tex = "--"
            stages = cell(r, "num_stages")
            pm = cell(r, "params_m")
            pm_s = f"{float(pm):.3f}" if pm not in ("--", "") else "--"
            infer = cell(r, "infer_ms_per_batch_median")
            infer_s = f"{float(infer):.3f}" if infer not in ("--", "") else "--"
            mem = cell(r, "peak_memory_allocated_mib")
            mem_s = f"{float(mem):.1f}" if mem not in ("--", "") else "--"
            lines.append(f"{name} & {sched_tex} & {stages} & {pm_s} & {mem_s} & {infer_s} \\\\")
            # midrule after HyperD (baseline group) when present
            if key == "hyperd" and i < len(rows) - 1:
                lines.append(r"\midrule")
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            note,
            r"\end{table}",
            "",
        ])
        latex_path.write_text("\n".join(lines), encoding="utf-8")

    mae_csv = ROOT / "results" / "paper_stage_schedule_existing_mae.csv"
    mae_fields = ["model", "schedule", "seed", "dataset", "input_len", "output_len", "test_mae", "source", "match_status"]
    with open(mae_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mae_fields)
        w.writeheader()
        for r in mae_rows:
            w.writerow(r)


def run_worker(model_key: str, args: argparse.Namespace) -> int:
    row = profile_one(model_key, args)
    out = Path(args.worker_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row), encoding="utf-8")
    print(json.dumps(row, indent=2))
    return 0 if row.get("status") == "ok" else 1


def prepare_cfg(hz, key: str, batch_size: int) -> None:
    family = MODEL_SPECS[key]["family"]
    if family == "f2fnet":
        cfg_path = generate_f2fnet_cfg(hz, key, batch_size=batch_size)
        cfg = hz.load_cfg(cfg_path)
        validate_f2fnet_cfg(cfg, key)
        print(f"[ok] cfg {key}: chain={cfg.MODEL.PARAM['chain_lengths']}")
    elif family == "hyperd":
        from baselines.HyperD.data_prepare import ensure_hyperd_data, init_npy_paths
        from baselines.HyperD.Initialization import run_initialization
        from baselines.HyperD.hyperd_runner import ensure_hyperd_scaler

        ensure_hyperd_data("PEMS04")
        daily, weekly = init_npy_paths("PEMS04")
        if not (daily.is_file() and weekly.is_file()):
            run_initialization("PEMS04")
        ensure_hyperd_scaler("PEMS04", INPUT_LEN, OUTPUT_LEN)
        cfg_path = generate_hyperd_cfg(batch_size)
        cfg = hz.load_cfg(cfg_path)
        validate_hyperd_cfg(cfg, batch_size)
        print(f"[ok] cfg hyperd: nodes={cfg.MODEL.PARAM['num_nodes']} arch={cfg.MODEL.ARCH.__name__}")
    elif family == "basicts_simple":
        cfg_path = generate_basicts_cfg(key, batch_size)
        cfg = hz.load_cfg(cfg_path)
        validate_basicts_cfg(cfg, key, batch_size)
        print(
            f"[ok] cfg {key}: arch={cfg.MODEL.ARCH.__name__} "
            f"fwd={list(cfg.MODEL.FORWARD_FEATURES)} tgt={list(cfg.MODEL.TARGET_FEATURES)}"
        )
    else:
        raise SystemExit(f"unknown family for {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper efficiency profiling on PEMS04 H=12")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--train_steps", type=int, default=40)
    parser.add_argument("--infer_steps", type=int, default=100)
    parser.add_argument("--max_seconds", type=int, default=540)
    parser.add_argument("--out", default="results/paper_efficiency.csv")
    parser.add_argument("--markdown", default="results/paper_efficiency.md")
    parser.add_argument("--latex", default="results/paper_efficiency_table.tex")
    parser.add_argument("--checkpoints", default="auto", choices=["auto", "none"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["stid", "agcrn", "hyperd", "single_stage", "two_stage", "f2fnet"],
    )
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        assert args.worker_out
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        args.gpu = "0"
        return run_worker(args.worker, args)

    t0 = time.perf_counter()
    write_env_file(ROOT / "results" / "paper_efficiency_environment.txt", args, {"mode": "orchestrator"})

    hz = load_horizon_module()
    for key in args.models:
        if key not in MODEL_SPECS:
            raise SystemExit(f"unknown model {key}")
        prepare_cfg(hz, key, args.batch_size)

    rows: list[dict] = []
    train_steps = args.train_steps
    infer_steps = args.infer_steps

    for i, key in enumerate(args.models):
        elapsed = time.perf_counter() - t0
        remaining = args.max_seconds - elapsed
        models_left = len(args.models) - i
        if remaining < 90 and models_left > 0:
            infer_steps = max(30, min(infer_steps, 40))
            train_steps = max(20, min(train_steps, 25))
        if remaining < 45:
            print(f"[stop] remaining {remaining:.1f}s insufficient for {key}")
            break

        worker_out = TEMP_CFG_DIR / f"profile_{key}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            key,
            "--worker_out",
            str(worker_out),
            "--gpu",
            str(args.gpu),
            "--batch_size",
            str(args.batch_size),
            "--warmup_steps",
            str(args.warmup_steps),
            "--train_steps",
            str(train_steps),
            "--infer_steps",
            str(infer_steps),
            "--checkpoints",
            args.checkpoints,
            "--max_seconds",
            str(args.max_seconds),
        ]
        print(f"[profile] {key} train_steps={train_steps} infer_steps={infer_steps} remaining={remaining:.1f}s")
        timeout = max(30, remaining - 5)
        try:
            subprocess.run(cmd, cwd=str(ROOT), timeout=timeout, check=False)
            if worker_out.is_file():
                row = json.loads(worker_out.read_text(encoding="utf-8"))
            else:
                row = empty_row(key)
                row["status"] = "error"
                row["error"] = "worker produced no output"
        except subprocess.TimeoutExpired:
            row = empty_row(key)
            row["status"] = "timeout"
            row["error"] = f"worker timeout after {timeout}s"
        rows.append(row)
        print(
            f"[done] {key} status={row.get('status')} class={row.get('model_class')} "
            f"params_m={row.get('params_m')} mem={row.get('peak_memory_allocated_mib')} "
            f"infer_med={row.get('infer_ms_per_batch_median')}"
        )

    # mark invalid_protocol on rows if needed
    protocol_ok, issues = validate_protocol(rows, args)
    if not protocol_ok:
        for r in rows:
            if r.get("status") == "ok":
                r["status"] = "invalid_protocol"
                r["error"] = "; ".join(issues[:3])

    mae_rows = audit_existing_mae()
    write_outputs(rows, args, mae_rows, protocol_ok, issues)
    elapsed = time.perf_counter() - t0
    write_env_file(
        ROOT / "results" / "paper_efficiency_environment.txt",
        args,
        {
            "elapsed_seconds": round(elapsed, 2),
            "models_completed": len(rows),
            "protocol_valid": protocol_ok,
            "protocol_issues": " | ".join(issues) if issues else "",
        },
    )
    print(f"[finished] elapsed={elapsed:.1f}s protocol_ok={protocol_ok} wrote {args.out}")
    if issues:
        for iss in issues:
            print(f"[protocol] {iss}")
    return 0 if protocol_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
