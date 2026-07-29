#!/usr/bin/env python3
"""Protocol A: fixed input H=12, variable forecast F∈{12,24,48} on traffic datasets."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "examples" / "run.py"

DATASET_SPECS = {
    "PEMS04": {
        "slug": "pems04",
        "chain_cfg": ROOT / "examples" / "ChainForecasting" / "ChainForecasting_PEMS04.py",
        "kasa_cfg": ROOT / "examples" / "KASAST_v2" / "KASAST_PEMS04.py",
    },
    "PEMS07": {
        "slug": "pems07",
        "chain_cfg": ROOT / "examples" / "ChainForecasting" / "ChainForecasting_PEMS07.py",
        "kasa_cfg": ROOT / "examples" / "KASAST_v2" / "KASAST_PEMS07.py",
    },
    "PEMS08": {
        "slug": "pems08",
        "chain_cfg": ROOT / "examples" / "ChainForecasting" / "ChainForecasting_PEMS08.py",
        "kasa_cfg": ROOT / "examples" / "KASAST_v2" / "KASAST_PEMS08.py",
    },
    "PEMS-BAY": {
        "slug": "pems_bay",
        "chain_cfg": ROOT / "examples" / "ChainForecasting" / "ChainForecasting_PEMS-BAY.py",
        "kasa_cfg": ROOT / "examples" / "KASAST_v2" / "KASAST_PEMS-BAY.py",
    },
    "PEMS03": {
        "slug": "pems03",
        "chain_cfg": ROOT / "examples" / "ChainForecasting" / "ChainForecasting_PEMS03.py",
        "kasa_cfg": ROOT / "examples" / "KASAST_v2" / "KASAST_PEMS03.py",
    },
    "KnowAir": {
        "slug": "knowair",
        "chain_cfg": ROOT / "examples" / "ChainForecasting" / "ChainForecasting_KnowAir.py",
        "kasa_cfg": ROOT / "examples" / "KASAST_v2" / "KASAST_KnowAir.py",
    },
}

# Filled in main() after --dataset is parsed
ACTIVE: dict | None = None
CHAIN_BASE_CFG: Path | None = None
KASA_BASE_CFG: Path | None = None
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_fixed_input_horizons.py"
TEMP_CFG_DIR: Path | None = None
LOG_DIR: Path | None = None
CKPT_ROOT: Path | None = None
DATASET_NAME: str | None = None
DATASET_SLUG: str | None = None


def activate_dataset(dataset: str) -> None:
    global ACTIVE, CHAIN_BASE_CFG, KASA_BASE_CFG, TEMP_CFG_DIR, LOG_DIR, CKPT_ROOT
    global DATASET_NAME, DATASET_SLUG
    if dataset not in DATASET_SPECS:
        raise SystemExit(f"Unsupported dataset: {dataset}")
    ACTIVE = DATASET_SPECS[dataset]
    DATASET_NAME = dataset
    DATASET_SLUG = ACTIVE["slug"]
    CHAIN_BASE_CFG = ACTIVE["chain_cfg"]
    KASA_BASE_CFG = ACTIVE["kasa_cfg"]
    TEMP_CFG_DIR = ROOT / "tmp_configs" / f"fixed_input_horizon_{DATASET_SLUG}"
    LOG_DIR = ROOT / "results" / f"fixed_input_horizon_{DATASET_SLUG}_logs"
    CKPT_ROOT = ROOT / "checkpoints" / f"fixed_input_horizon_{DATASET_SLUG}"

INPUT_LEN = 12
DEFAULT_HORIZONS = [12, 24, 48]
DEFAULT_VARIANTS = ["kasa_baseline", "chain_no_spatial", "chain_final_spatial"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]

HORIZON_CHAIN_LENGTHS: dict[int, list[int]] = {
    12: [3, 6, 12],
    24: [6, 12, 24],
    48: [12, 24, 48],
}
HORIZON_EVAL_STEPS: dict[int, list[int]] = {
    12: [3, 6, 12],
    24: [6, 12, 24],
    48: [12, 24, 48],
}
CHAIN_LOSS_WEIGHTS = [0.2, 0.3, 1.0]

BEST_SETTINGS = {
    "use_patch_branch": True,
    "use_downsample_branch": True,
    "use_linear_residual_branch": True,
    "post_spatial_mode": "adaptive_only",
    "use_pre_temporal_spatial_enhancement": False,
    "keep_output_prior_residual": False,
    "use_input_prior_enhancement": False,
    "use_graph_spectral_calibration": False,
    "use_extra_prior_input": False,
    "main_input_dim": 3,
    "patch_embedding_mode": "serial_concat",
    "patch_data_input_mode": "all",
}

VARIANT_SPECS: dict[str, dict] = {
    "kasa_baseline": {"is_chain": False},
    "chain_no_spatial": {
        "is_chain": True,
        "chain_lengths": None,
        "chain_loss_weights": CHAIN_LOSS_WEIGHTS,
        "use_prev_condition": True,
        "spatial_placement": "none",
    },
    "chain_final_spatial": {
        "is_chain": True,
        "chain_lengths": None,
        "chain_loss_weights": CHAIN_LOSS_WEIGHTS,
        "use_prev_condition": True,
        "spatial_placement": "final",
    },
    "chain_each_level_spatial": {
        "is_chain": True,
        "chain_lengths": None,
        "chain_loss_weights": CHAIN_LOSS_WEIGHTS,
        "use_prev_condition": True,
        "spatial_placement": "each_level",
    },
    "chain_interleaved_progressive_spatial": {
        "is_chain": True,
        "chain_lengths": None,
        "chain_loss_weights": CHAIN_LOSS_WEIGHTS,
        "use_prev_condition": True,
        "spatial_placement": "interleaved_progressive",
        "progressive_spatial_ratios": [0.25, 0.5, 1.0],
        "progressive_spatial_topks": [8, 16, 32],
        "progressive_spatial_alphas": [0.03, 0.06, 0.10],
        "post_spatial_mode": "adaptive_only",
        "use_adaptive_adj": True,
    },
    "chain_interleaved_progressive_spatial_light": {
        "is_chain": True,
        "chain_lengths": None,
        "chain_loss_weights": CHAIN_LOSS_WEIGHTS,
        "use_prev_condition": True,
        "spatial_placement": "interleaved_progressive",
        "progressive_spatial_ratios": [0.25, 0.5, 1.0],
        "progressive_spatial_topks": [4, 8, 16],
        "progressive_spatial_alphas": [0.02, 0.04, 0.08],
        "post_spatial_mode": "adaptive_only",
        "use_adaptive_adj": True,
    },
    "chain_interleaved_progressive_spatial_strong": {
        "is_chain": True,
        "chain_lengths": None,
        "chain_loss_weights": CHAIN_LOSS_WEIGHTS,
        "use_prev_condition": True,
        "spatial_placement": "interleaved_progressive",
        "progressive_spatial_ratios": [0.5, 0.75, 1.0],
        "progressive_spatial_topks": [16, 24, 40],
        "progressive_spatial_alphas": [0.05, 0.08, 0.12],
        "post_spatial_mode": "adaptive_only",
        "use_adaptive_adj": True,
    },
    # Experiment A: Spectral Stage Router (same chain/spatial; only temporal branch fusion)
    "chain_interleaved_progressive_spatial_router": {
        "is_chain": True,
        "model_name": "ChainForecasting_SpectralRouter",
        "chain_lengths": None,
        "chain_loss_weights": CHAIN_LOSS_WEIGHTS,
        "use_prev_condition": True,
        "spatial_placement": "interleaved_progressive",
        "progressive_spatial_ratios": [0.25, 0.5, 1.0],
        "progressive_spatial_topks": [8, 16, 32],
        "progressive_spatial_alphas": [0.03, 0.06, 0.10],
        "post_spatial_mode": "adaptive_only",
        "use_adaptive_adj": True,
        "use_spectral_stage_router": True,
    },
    # Experiment B: Forecast-State Token MAE (same model; only chain loss aggregation)
    "chain_interleaved_progressive_spatial_token_loss": {
        "is_chain": True,
        "model_name": "ChainForecasting_TokenMAE",
        "chain_lengths": None,
        "chain_loss_weights": None,
        "chain_loss_mode": "token_mae",
        "use_prev_condition": True,
        "spatial_placement": "interleaved_progressive",
        "progressive_spatial_ratios": [0.25, 0.5, 1.0],
        "progressive_spatial_topks": [8, 16, 32],
        "progressive_spatial_alphas": [0.03, 0.06, 0.10],
        "post_spatial_mode": "adaptive_only",
        "use_adaptive_adj": True,
        "use_spectral_stage_router": False,
    },
    # Light sample-level spectral router (bounded coefs; original weighted chain loss)
    "chain_interleaved_progressive_spatial_light_router": {
        "is_chain": True,
        "model_name": "ChainForecasting_LightSpectralRouter",
        "chain_lengths": None,
        "chain_loss_weights": CHAIN_LOSS_WEIGHTS,
        "use_prev_condition": True,
        "spatial_placement": "interleaved_progressive",
        "progressive_spatial_ratios": [0.25, 0.5, 1.0],
        "progressive_spatial_topks": [8, 16, 32],
        "progressive_spatial_alphas": [0.03, 0.06, 0.10],
        "post_spatial_mode": "adaptive_only",
        "use_adaptive_adj": True,
        "use_light_spectral_router": True,
        "router_hidden_dim": 8,
        "router_max_deviation": 0.05,
        "router_shared_across_stages": True,
    },
    # Forecast-State Dynamics Adapter after Spatial6/Spatial12 (shared, zero-init)
    "chain_interleaved_progressive_spatial_state_adapter": {
        "is_chain": True,
        "model_name": "ChainForecasting_StateAdapter",
        "chain_lengths": None,
        "chain_loss_weights": CHAIN_LOSS_WEIGHTS,
        "use_prev_condition": True,
        "spatial_placement": "interleaved_progressive",
        "progressive_spatial_ratios": [0.25, 0.5, 1.0],
        "progressive_spatial_topks": [8, 16, 32],
        "progressive_spatial_alphas": [0.03, 0.06, 0.10],
        "post_spatial_mode": "adaptive_only",
        "use_adaptive_adj": True,
        "use_forecast_state_adapter": True,
        "forecast_state_adapter_mode": "state_replace",
        "forecast_state_adapter_hidden_dim": 16,
        "forecast_state_adapter_epsilon": 0.05,
    },
    # Condition-only Adapter: fair init + Z6_condition for T12 only; L6/L12 on raw
    "chain_interleaved_progressive_spatial_state_adapter_fixed": {
        "is_chain": True,
        "model_name": "ChainForecasting_StateAdapterFixed",
        "chain_lengths": None,
        "chain_loss_weights": CHAIN_LOSS_WEIGHTS,
        "use_prev_condition": True,
        "spatial_placement": "interleaved_progressive",
        "progressive_spatial_ratios": [0.25, 0.5, 1.0],
        "progressive_spatial_topks": [8, 16, 32],
        "progressive_spatial_alphas": [0.03, 0.06, 0.10],
        "post_spatial_mode": "adaptive_only",
        "use_adaptive_adj": True,
        "use_forecast_state_adapter": True,
        "forecast_state_adapter_mode": "condition_only",
        "forecast_state_adapter_hidden_dim": 16,
        "forecast_state_adapter_epsilon": 0.02,
    },
    # Fixed Adapter model + Token-Normalized MAE (no artificial stage weights)
    "chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss": {
        "is_chain": True,
        "model_name": "ChainForecasting_StateAdapterFixed_TokenMAE",
        "chain_lengths": None,
        "chain_loss_weights": None,
        "chain_loss_mode": "token_normalized",
        "use_prev_condition": True,
        "spatial_placement": "interleaved_progressive",
        "progressive_spatial_ratios": [0.25, 0.5, 1.0],
        "progressive_spatial_topks": [8, 16, 32],
        "progressive_spatial_alphas": [0.03, 0.06, 0.10],
        "post_spatial_mode": "adaptive_only",
        "use_adaptive_adj": True,
        "use_forecast_state_adapter": True,
        "forecast_state_adapter_mode": "condition_only",
        "forecast_state_adapter_hidden_dim": 16,
        "forecast_state_adapter_epsilon": 0.02,
    },
}

METRIC_PATTERNS = {
    "mae": re.compile(r"test_MAE:\s*([0-9.eE+-]+)", re.I),
    "rmse": re.compile(r"test_RMSE:\s*([0-9.eE+-]+)", re.I),
    "mape": re.compile(r"test_MAPE:\s*([0-9.eE+-]+)", re.I),
    "inference_time": re.compile(r"test_time:\s*([0-9.eE+-]+)", re.I),
}
HORIZON_MAE_PATTERN = re.compile(
    r"Evaluate best model on test data for horizon\s+(\d+),\s*Test MAE:\s*([0-9.eE+-]+)",
    re.I,
)
RESULT_TEST_BLOCK = re.compile(
    r"Result\s*(?:<[^>]+>)?\s*:\s*\[(.*?)\]",
    re.I | re.S,
)
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
ERROR_TAIL_LINES = 120
SHOW_ERROR_LINES = 80


def load_cfg(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("horizon_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CFG


def base_cfg_for_variant(variant: str) -> Path:
    if variant == "kasa_baseline":
        return KASA_BASE_CFG
    return CHAIN_BASE_CFG


def chain_lengths_for(horizon: int) -> list[int]:
    if horizon not in HORIZON_CHAIN_LENGTHS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    return HORIZON_CHAIN_LENGTHS[horizon]


def variant_spec(variant: str, horizon: int) -> dict:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant: {variant}")
    spec = {**BEST_SETTINGS, **VARIANT_SPECS[variant]}
    if spec.get("is_chain"):
        spec = dict(spec)
        spec["chain_lengths"] = chain_lengths_for(horizon)
    return spec


def job_name(horizon: int, variant: str, seed: int) -> str:
    return f"h{horizon}_{variant}_seed{seed}"


def ckpt_dir_for(horizon: int, variant: str, seed: int) -> Path:
    return CKPT_ROOT / f"h{horizon}" / f"{variant}_seed{seed}"


def temp_cfg_path(horizon: int, variant: str, seed: int) -> Path:
    TEMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_CFG_DIR / f"h{horizon}_{variant}_seed{seed}.py"


def strip_hardcoded_cuda_devices(content: str) -> str:
    kept: list[str] = []
    for line in content.splitlines():
        if "CUDA_VISIBLE_DEVICES" in line and "os.environ" in line:
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def _py_literal(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, list):
        return repr(v)
    return str(v)


_META_SPEC_KEYS = {"is_chain", "model_name"}


def generate_temp_config(horizon: int, variant: str, seed: int) -> Path:
    spec = variant_spec(variant, horizon)
    base_cfg = base_cfg_for_variant(variant)
    content = strip_hardcoded_cuda_devices(base_cfg.read_text(encoding="utf-8"))
    ckpt_rel = os.path.join("checkpoints", f"fixed_input_horizon_{DATASET_SLUG}", f"h{horizon}", f"{variant}_seed{seed}")
    lines = [
        "",
        "# ===== fixed_input_horizon overrides (auto-generated) =====",
        f"CFG.ENV.SEED = {seed}",
        "if hasattr(CFG, 'SEED'):",
        f"    CFG.SEED = {seed}",
        "if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):",
        f"    CFG.TRAIN.SEED = {seed}",
        f'CFG.DATASET_NAME = "{DATASET_NAME}"',
        f'CFG.TRAIN.DATA.DIR = "datasets/{DATASET_NAME}"',
        f'CFG.VAL.DATA.DIR = "datasets/{DATASET_NAME}"',
        f'CFG.TEST.DATA.DIR = "datasets/{DATASET_NAME}"',
        f'CFG.MODEL.PARAM["adj_mx_path"] = "datasets/{DATASET_NAME}/adj_mx.pkl"',
        f"CFG.DATASET_INPUT_LEN = {INPUT_LEN}",
        f"CFG.DATASET_OUTPUT_LEN = {horizon}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")',
        "CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]",
        "CFG.MODEL.TARGET_FEATURES = [0]",
        f'CFG.MODEL.PARAM["input_len"] = {INPUT_LEN}',
        f'CFG.MODEL.PARAM["output_len"] = {horizon}',
        f"CFG.TEST.EVALUATION_HORIZONS = list(range(1, {horizon + 1}))",
    ]
    model_name = spec.get("model_name")
    if model_name:
        lines.append(f"CFG.MODEL.NAME = {_py_literal(model_name)}")
    for key, val in spec.items():
        if key in _META_SPEC_KEYS:
            continue
        # Keep explicit None for keys like chain_loss_weights under token_mae.
        lines.append(f'CFG.MODEL.PARAM["{key}"] = {_py_literal(val)}')
    out = temp_cfg_path(horizon, variant, seed)
    out.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")
    return out


def cfg_for_easytorch(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def count_params(cfg_path: Path) -> int | None:
    try:
        cfg = load_cfg(cfg_path)
        model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception:
        return None


def estimate_flops(cfg_path: Path) -> float | None:
    try:
        cfg = load_cfg(cfg_path)
        model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM).eval()
        b, n, c = 1, cfg.MODEL.PARAM["node_size"], cfg.MODEL.PARAM["input_dim"]
        x = torch.randn(b, INPUT_LEN, n, c)
        try:
            from thop import profile

            macs, _ = profile(model, inputs=(x,), verbose=False)
            return float(macs) * 2
        except Exception:
            pass
        with torch.no_grad():
            model(history_data=x, future_data=x, train=False)
        return None
    except Exception:
        return None


def _parse_block(block: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {k: None for k in METRIC_PATTERNS}
    for k, pat in METRIC_PATTERNS.items():
        m = pat.search(block)
        if m:
            out[k] = float(m.group(1))
    return out


def parse_metrics(log_text: str) -> dict[str, float | None]:
    best: dict[str, float | None] | None = None
    best_mae: float | None = None
    for m in RESULT_TEST_BLOCK.finditer(log_text):
        parsed = _parse_block(m.group(1))
        if parsed["mae"] is None:
            continue
        if best_mae is None or parsed["mae"] < best_mae:
            best_mae = parsed["mae"]
            best = parsed
    if best is not None:
        return best
    fallback = _parse_block(log_text)
    if fallback["mae"] is not None:
        return fallback
    return {k: None for k in METRIC_PATTERNS}


def parse_horizon_mae(log_text: str, horizons: list[int]) -> dict[int, float | None]:
    found: dict[int, float] = {}
    for m in HORIZON_MAE_PATTERN.finditer(log_text):
        h = int(m.group(1))
        found[h] = float(m.group(2))
    return {h: found.get(h) for h in horizons}


def collect_log_text(horizon: int, variant: str, seed: int, wrapper_log: Path | None) -> str:
    parts: list[str] = []
    if wrapper_log and wrapper_log.is_file():
        parts.append(wrapper_log.read_text(errors="replace"))
    ckpt_base = ckpt_dir_for(horizon, variant, seed)
    if ckpt_base.is_dir():
        tlogs = sorted(
            ckpt_base.glob("*/training_log_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for tlog in tlogs:
            parts.append(tlog.read_text(errors="replace"))
    return "\n".join(parts)


def wrapper_log_path(horizon: int, variant: str, seed: int) -> Path | None:
    matches = sorted(LOG_DIR.glob(f"h{horizon}_{variant}_seed{seed}_gpu*.log"))
    return matches[-1] if matches else None


def print_log_tail(horizon: int, variant: str, seed: int, lines: int = SHOW_ERROR_LINES) -> None:
    log_path = wrapper_log_path(horizon, variant, seed)
    if not log_path or not log_path.is_file():
        return
    text_lines = log_path.read_text(errors="replace").splitlines()
    tail = text_lines[-lines:]
    print(f"\n--- error tail: h={horizon} {variant} seed={seed} (last {len(tail)} lines) ---")
    for line in tail:
        print(line)
    print("--- end error tail ---\n")


def show_failed_errors(rows: list[dict], lines: int = SHOW_ERROR_LINES) -> None:
    failed = [r for r in rows if str(r.get("status", "")).startswith("exit_")]
    if not failed:
        return
    print("\nFailed run diagnostics:\n")
    for row in sorted(failed, key=lambda r: (int(r["horizon"]), r["variant"], int(r["seed"]))):
        print(f"  h={row['horizon']} {row['variant']} seed={row['seed']} status={row['status']}")
        if row.get("error_tail_file"):
            print(f"    error_tail_file: {row['error_tail_file']}")
        print_log_tail(int(row["horizon"]), row["variant"], int(row["seed"]), lines=lines)


def save_error_tail(wrapper_log: Path, horizon: int, variant: str, seed: int) -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tail_path = LOG_DIR / f"h{horizon}_{variant}_seed{seed}_error_tail.txt"
    if wrapper_log.is_file():
        lines = wrapper_log.read_text(errors="replace").splitlines()
        tail_path.write_text("\n".join(lines[-ERROR_TAIL_LINES:]) + "\n", encoding="utf-8")
    else:
        tail_path.write_text("", encoding="utf-8")
    return str(tail_path)


class GPUQueue:
    def __init__(self, gpus: list[str]):
        self.gpus = gpus
        self.lock = threading.Lock()
        self.available = list(gpus)

    def acquire(self) -> str:
        while True:
            with self.lock:
                if self.available:
                    return self.available.pop(0)
            time.sleep(2)

    def release(self, gpu: str):
        with self.lock:
            self.available.append(gpu)


def base_row(horizon: int, variant: str, seed: int, cfg_path: Path) -> dict:
    spec = variant_spec(variant, horizon)
    params = count_params(cfg_path)
    flops = estimate_flops(cfg_path)
    return {
        "horizon": horizon,
        "variant": variant,
        "seed": seed,
        "input_len": INPUT_LEN,
        "output_len": horizon,
        "chain_lengths": str(spec.get("chain_lengths", [])),
        "mae": None,
        "rmse": None,
        "mape": None,
        "params": params,
        "FLOPs": flops,
        "inference_time": None,
        "failed": 1,
        "status": "pending",
        "ckpt_dir": str(ckpt_dir_for(horizon, variant, seed)),
        "cfg_path": str(cfg_path),
        "log_file": "",
        "error_tail_file": "",
    }


def summarize_row(horizon: int, variant: str, seed: int, cfg_path: Path) -> dict:
    row = base_row(horizon, variant, seed, cfg_path)
    wrapper_log = wrapper_log_path(horizon, variant, seed)
    metrics = parse_metrics(collect_log_text(horizon, variant, seed, wrapper_log))
    row.update(
        {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "inference_time": metrics["inference_time"],
        }
    )
    eval_steps = HORIZON_EVAL_STEPS[horizon]
    hz = parse_horizon_mae(collect_log_text(horizon, variant, seed, wrapper_log), eval_steps)
    for step in eval_steps:
        row[f"horizon_mae_{step}"] = hz.get(step)
    error_tail = LOG_DIR / f"h{horizon}_{variant}_seed{seed}_error_tail.txt"
    if metrics["mae"] is not None:
        row["status"] = "ok"
        row["failed"] = 0
    else:
        row["status"] = "failed_no_metrics"
        row["failed"] = 1
    row["log_file"] = str(wrapper_log) if wrapper_log else ""
    row["error_tail_file"] = str(error_tail) if error_tail.is_file() else ""
    return row


def is_completed(horizon: int, variant: str, seed: int, cfg_path: Path) -> bool:
    row = summarize_row(horizon, variant, seed, cfg_path)
    return row.get("status") == "ok" and row.get("mae") is not None


def run_one(horizon: int, variant: str, cfg_path: Path, gpu: str, seed: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"h{horizon}_{variant}_seed{seed}_gpu{gpu}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, str(RUN_PY), "--cfg", cfg_for_easytorch(cfg_path), "--gpus", str(gpu)]
    row = base_row(horizon, variant, seed, cfg_path)
    row["log_file"] = str(log_file)
    row["status"] = "running"
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        metrics = parse_metrics(collect_log_text(horizon, variant, seed, log_file))
        row.update(metrics)
        eval_steps = HORIZON_EVAL_STEPS[horizon]
        hz = parse_horizon_mae(collect_log_text(horizon, variant, seed, log_file), eval_steps)
        for step in eval_steps:
            row[f"horizon_mae_{step}"] = hz.get(step)
        if proc.returncode != 0:
            row["status"] = f"exit_{proc.returncode}"
            row["failed"] = 1
            row["error_tail_file"] = save_error_tail(log_file, horizon, variant, seed)
        elif metrics["mae"] is None:
            row["status"] = "failed_no_metrics"
            row["failed"] = 1
        else:
            row["status"] = "ok"
            row["failed"] = 0
    except Exception as e:
        row["status"] = f"error:{e}"
        row["failed"] = 1
    return row


def t_crit(n: int) -> float:
    if n in T_CRIT:
        return T_CRIT[n]
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, n - 1))
    except Exception:
        return 1.96


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, float("nan")
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    return mean, std, t_crit(n) * std / math.sqrt(n)


def build_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((int(r["horizon"]), r["variant"]), []).append(r)

    summary = []
    for (horizon, variant), runs in sorted(groups.items()):
        ok = [r for r in runs if r.get("mae") is not None and r.get("failed", 1) == 0]
        failed = len(runs) - len(ok)
        for metric in ("mae", "rmse", "mape"):
            vals = [float(r[metric]) for r in ok if r.get(metric) is not None]
            m, s, c = mean_std_ci(vals)
            summary.append({
                "horizon": horizon,
                "variant": variant,
                "metric": metric,
                "n": len(vals),
                "mean": m,
                "std": s,
                "ci95": c,
                "failed": failed,
            })
    return summary


def fmt_val(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_outputs(rows: list[dict], out_csv: Path, out_md: Path, summary_csv: Path) -> None:
    rows = sorted(rows, key=lambda r: (int(r["horizon"]), r["variant"], int(r["seed"])))
    eval_cols = sorted({k for r in rows for k in r if k.startswith("horizon_mae_")})
    fields = [
        "horizon", "variant", "seed", "input_len", "output_len", "chain_lengths",
        "mae", "rmse", "mape", "params", "FLOPs", "inference_time", "failed",
        "status", "ckpt_dir", "cfg_path", "log_file", "error_tail_file",
    ] + eval_cols

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = build_summary(rows)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["horizon", "variant", "metric", "n", "mean", "std", "ci95", "failed"],
        )
        w.writeheader()
        w.writerows(summary)

    md = [
        "# Protocol A: Fixed Input H=12, Variable Forecast on PeMS04\n\n",
        "Input length fixed at 12; output length F∈{12,24,48}.\n",
        "Chain lengths follow ¼F → ½F → F.\n",
        "Official 6:2:2 split, TARGET=[0], same optimizer/LR/epochs as chain ablation.\n\n",
        "## Per-run results\n\n",
        "| horizon | variant | seed | input | output | chain_lengths | MAE | RMSE | MAPE | "
        "params | FLOPs | inference_time | failed |\n",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for r in rows:
        md.append(
            f"| {r['horizon']} | {r['variant']} | {r['seed']} | {r['input_len']} | {r['output_len']} | "
            f"{r['chain_lengths']} | {fmt_val(r['mae'])} | {fmt_val(r['rmse'])} | {fmt_val(r['mape'])} | "
            f"{fmt_val(r.get('params'))} | {fmt_val(r.get('FLOPs'))} | "
            f"{fmt_val(r.get('inference_time'))} | {r.get('failed', 1)} |\n"
        )

    md.append("\n## Summary (mean ± std, 95% CI)\n\n")
    md.append("| horizon | variant | metric | n | mean | std | 95% CI | failed |\n")
    md.append("|---|---|---|---:|---:|---:|---:|---:|\n")
    for s in summary:
        md.append(
            f"| {s['horizon']} | {s['variant']} | {s['metric']} | {s['n']} | "
            f"{fmt_val(s['mean'])} | {fmt_val(s['std'])} | {fmt_val(s['ci95'])} | {s['failed']} |\n"
        )

    md.append("\n## Horizon-wise MAE (key steps)\n\n")
    for horizon in sorted({int(r["horizon"]) for r in rows}):
        steps = HORIZON_EVAL_STEPS[horizon]
        md.append(f"### F={horizon}\n\n")
        md.append("| variant | seed | " + " | ".join(f"h={s}" for s in steps) + " |\n")
        md.append("|---|---|" + "|".join(["---:"] * len(steps)) + "|\n")
        for r in rows:
            if int(r["horizon"]) != horizon:
                continue
            vals = [fmt_val(r.get(f"horizon_mae_{s}")) for s in steps]
            md.append(f"| {r['variant']} | {r['seed']} | " + " | ".join(vals) + " |\n")
        md.append("\n")

    out_md.write_text("".join(md), encoding="utf-8")


def ensure_data(horizons: list[int]) -> int:
    cmd = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--dataset",
        DATASET_NAME,
        "--input-len",
        str(INPUT_LEN),
        "--horizons",
        *[str(h) for h in horizons],
    ]
    print(f"[prepare_data] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def verify_batch_shape(horizon: int) -> tuple[bool, str]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from basicts.data import TimeSeriesForecastingDataset
    from torch.utils.data import DataLoader

    data_dir = ROOT / "datasets" / DATASET_NAME
    data_file = data_dir / f"data_in{INPUT_LEN}_out{horizon}.pkl"
    index_file = data_dir / f"index_in{INPUT_LEN}_out{horizon}.pkl"
    if not data_file.is_file() or not index_file.is_file():
        return False, f"missing data for F={horizon}"
    ds = TimeSeriesForecastingDataset(
        data_file_path=str(data_file),
        index_file_path=str(index_file),
        mode="test",
    )
    future_data, history_data = next(iter(DataLoader(ds, batch_size=32, shuffle=False)))
    if tuple(history_data.shape[1:3]) != (INPUT_LEN, 307):
        return False, f"history shape {tuple(history_data.shape)}"
    if future_data.shape[1] != horizon:
        return False, f"future len {future_data.shape[1]} != {horizon}"
    return True, f"history={tuple(history_data.shape)} future={tuple(future_data.shape)}"


def dry_run_info(horizon: int, variant: str, seed: int, cfg_path: Path) -> None:
    spec = variant_spec(variant, horizon)
    cfg = load_cfg(cfg_path)
    cmd = f"{sys.executable} {RUN_PY} --cfg {cfg_for_easytorch(cfg_path)} --gpus <GPU>"
    print(f"  [h={horizon} {variant} seed={seed}]")
    print(f"    cfg: {cfg_path}")
    print(f"    ckpt: {ckpt_dir_for(horizon, variant, seed)}")
    print(f"    input_len: {cfg.DATASET_INPUT_LEN} (model {cfg.MODEL.PARAM['input_len']})")
    print(f"    output_len: {cfg.DATASET_OUTPUT_LEN} (model {cfg.MODEL.PARAM['output_len']})")
    print(f"    chain_lengths: {spec.get('chain_lengths', 'n/a')}")
    print(f"    spatial_placement: {spec.get('spatial_placement', 'n/a')}")
    print(f"    params: {count_params(cfg_path)}")
    print(f"    FLOPs: {estimate_flops(cfg_path)}")
    ok, msg = verify_batch_shape(horizon)
    print(f"    batch_check: {'OK' if ok else 'FAIL'} {msg}")
    print(f"    cmd: {cmd}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Protocol A fixed-input horizon ablation.")
    parser.add_argument("--dataset", default="PEMS04", choices=list(DATASET_SPECS.keys()))
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANT_SPECS.keys()))
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--prepare_data", action="store_true", help="Generate missing horizon data first.")
    parser.add_argument("--skip_completed", action="store_true", default=True)
    parser.add_argument("--no_skip_completed", action="store_false", dest="skip_completed")
    parser.add_argument("--summary-only", "--summary_only", action="store_true", dest="summary_only")
    parser.add_argument("--show_errors", action="store_true")
    args = parser.parse_args()
    activate_dataset(args.dataset)
    TEMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        args.out = f"results/{DATASET_SLUG}_fixed_input_horizon.csv"
    if args.markdown is None:
        args.markdown = f"results/{DATASET_SLUG}_fixed_input_horizon.md"

    for h in args.horizons:
        if h not in HORIZON_CHAIN_LENGTHS:
            print(f"Unsupported horizon: {h}")
            return 1

    if not CHAIN_BASE_CFG.is_file() or not KASA_BASE_CFG.is_file():
        print("Missing base configs.")
        return 1

    if args.prepare_data or args.dry_run:
        rc = ensure_data(args.horizons)
        if rc != 0:
            return rc

    out_csv = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_md = ROOT / args.markdown if not Path(args.markdown).is_absolute() else Path(args.markdown)
    summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")

    jobs = [
        (h, v, generate_temp_config(h, v, s), s)
        for h in args.horizons
        for v in args.variants
        for s in args.seeds
    ]
    expected_jobs = len(args.horizons) * len(args.variants) * len(args.seeds)

    if args.dry_run:
        print("Dry run — temp configs and commands:\n")
        ckpt_dirs = set()
        all_input_ok = True
        for horizon, variant, cfg_path, seed in jobs:
            dry_run_info(horizon, variant, seed, cfg_path)
            ckpt_dirs.add(str(ckpt_dir_for(horizon, variant, seed)))
            cfg = load_cfg(cfg_path)
            if cfg.DATASET_INPUT_LEN != INPUT_LEN or cfg.MODEL.PARAM["input_len"] != INPUT_LEN:
                all_input_ok = False
            if cfg.DATASET_OUTPUT_LEN != horizon or cfg.MODEL.PARAM["output_len"] != horizon:
                print(f"    ERROR: output_len mismatch for h={horizon}")
            expected_chain = chain_lengths_for(horizon)
            if variant != "kasa_baseline":
                if cfg.MODEL.PARAM.get("chain_lengths") != expected_chain:
                    print(f"    ERROR: chain_lengths {cfg.MODEL.PARAM.get('chain_lengths')} != {expected_chain}")
        print(f"\n{len(jobs)} jobs (expected {expected_jobs}), unique ckpt dirs: {len(ckpt_dirs)}")
        print(f"all input_len==12: {all_input_ok}")
        for h in args.horizons:
            ok, msg = verify_batch_shape(h)
            print(f"F={h} shape check: {'OK' if ok else 'FAIL'} — {msg}")
        return 0 if all_input_ok else 2

    if args.summary_only:
        rows = [summarize_row(h, v, s, p) for h, v, p, s in jobs]
        write_outputs(rows, out_csv, out_md, summary_csv)
        print(f"Wrote {out_csv}, {out_md}, {summary_csv}")
        if args.show_errors:
            show_failed_errors(rows)
        return 0

    queue = GPUQueue(args.gpus)
    rows: list[dict] = []
    lock = threading.Lock()

    def worker(horizon: int, variant: str, cfg_path: Path, seed: int):
        if args.skip_completed and is_completed(horizon, variant, seed, cfg_path):
            row = summarize_row(horizon, variant, seed, cfg_path)
            with lock:
                rows.append(row)
            print(f"[skip] h={horizon} {variant} seed={seed} (already ok, mae={row['mae']})")
            return
        gpu = queue.acquire()
        try:
            print(f"[start] h={horizon} {variant} seed={seed} gpu={gpu}")
            row = run_one(horizon, variant, cfg_path, gpu, seed)
            with lock:
                rows.append(row)
            print(f"[done]  h={horizon} {variant} seed={seed} status={row['status']} mae={row['mae']}")
        finally:
            queue.release(gpu)

    threads = []
    for horizon, variant, cfg_path, seed in jobs:
        t = threading.Thread(target=worker, args=(horizon, variant, cfg_path, seed))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    write_outputs(rows, out_csv, out_md, summary_csv)
    print(f"Wrote {out_csv}, {out_md}, {summary_csv} ({len(rows)}/{expected_jobs} jobs)")
    if args.show_errors:
        show_failed_errors(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
