#!/usr/bin/env python3
"""PeMS04 long-horizon Graph Resolution ablation (16→16 / 16→32 / 16→64)."""
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN_PY = ROOT / "examples" / "run.py"
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_pems04_fixed_input_horizons.py"
DEFAULT_WORK_DIR = ROOT / "experiments" / "pems04_long_horizon_graph"
DEFAULT_LOG_ROOT = ROOT / "logs" / "pems04_long_horizon_graph"
DEFAULT_CKPT_ROOT = ROOT / "checkpoints" / "pems04_long_horizon_graph"

DEFAULT_HORIZONS = [16, 32, 64]
DEFAULT_VARIANTS = [
    "G0_no_spatial",
    "G1_final_adaptive",
    "GR7_sparse_topk",
    "GR14_two_level_sparse",
]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_GPUS = ["0", "1"]

INPUT_LEN = 16

HORIZON_CONFIGS: dict[int, dict[str, Any]] = {
    16: {
        "base_cfg": ROOT / "examples" / "ChainForecasting" / "ChainForecasting_PEMS04_16to16.py",
        "input_len": 16,
        "output_len": 16,
        "chain_lengths": [4, 8, 16],
    },
    32: {
        "base_cfg": ROOT / "examples" / "ChainForecasting" / "ChainForecasting_PEMS04_16to32.py",
        "input_len": 16,
        "output_len": 32,
        "chain_lengths": [8, 16, 32],
    },
    64: {
        "base_cfg": ROOT / "examples" / "ChainForecasting" / "ChainForecasting_PEMS04_16to64.py",
        "input_len": 16,
        "output_len": 64,
        "chain_lengths": [16, 32, 64],
    },
}

GRAPH_RES_KEYS = (
    "graph_resolution_ratios",
    "graph_resolution_alphas",
    "graph_resolution_topks",
    "graph_resolution_betas",
    "graph_resolution_rhos",
)

GR7_SPARSE_BASE: dict[str, Any] = {
    "spatial_placement": "temporal_first_graph_resolution",
    "post_spatial_mode": "adaptive_only",
    "graph_resolution_ratios": [0.25, 0.50, 1.00],
    "graph_resolution_alphas": [0.03, 0.06, 0.10],
    "graph_resolution_topks": [4, 8, 16],
    "graph_resolution_betas": [1.0, 1.0, 1.0],
    "graph_resolution_rhos": [0.25, 0.50, 1.00],
    "clustering_seed": 0,
    "dataset_name": "PEMS04",
    "graph_cluster_method": "current",
}

COMMON_FIXED: dict[str, Any] = {
    "chain_loss_weights": [0.2, 0.3, 1.0],
    "use_prev_condition": True,
    "use_patch_branch": True,
    "use_downsample_branch": True,
    "use_linear_residual_branch": True,
    "patch_embedding_mode": "serial_concat",
    "patch_data_input_mode": "all",
    "use_pre_temporal_spatial_enhancement": False,
    "keep_output_prior_residual": False,
    "use_input_prior_enhancement": False,
    "use_graph_spectral_calibration": False,
    "use_extra_prior_input": False,
    "main_input_dim": 3,
}

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "G0_no_spatial": {
        "spatial_placement": "none",
        "post_spatial_mode": "none",
    },
    "G1_final_adaptive": {
        "spatial_placement": "final",
        "post_spatial_mode": "adaptive_only",
    },
    "GR7_sparse_topk": {
        **GR7_SPARSE_BASE,
    },
    "GR14_two_level_sparse": {
        **GR7_SPARSE_BASE,
        "graph_resolution_ratios": [0.50, 1.00],
        "graph_resolution_alphas": [0.06, 0.10],
        "graph_resolution_topks": [8, 16],
        "graph_resolution_betas": [1.0, 1.0],
        "graph_resolution_rhos": [0.50, 1.00],
    },
    "GR15_four_level_sparse": {
        **GR7_SPARSE_BASE,
        "graph_resolution_ratios": [0.125, 0.25, 0.50, 1.00],
        "graph_resolution_alphas": [0.02, 0.04, 0.06, 0.10],
        "graph_resolution_topks": [2, 4, 8, 16],
        "graph_resolution_betas": [1.0, 1.0, 1.0, 1.0],
        "graph_resolution_rhos": [0.125, 0.25, 0.50, 1.00],
    },
}

EPOCH_LINE = re.compile(r"Epoch\s+(\d+)\s*/", re.I)
VAL_LINE = re.compile(r"Result\s*<val>.*?val_MAE:\s*([0-9.eE+-]+)", re.I | re.S)
TRAIN_LINE = re.compile(r"Result\s*<train>.*?train_MAE:\s*([0-9.eE+-]+)", re.I | re.S)
TEST_BLOCK = re.compile(
    r"Result\s*<test>.*?test_MAE:\s*([0-9.eE+-]+).*?"
    r"test_RMSE:\s*([0-9.eE+-]+).*?"
    r"test_MAPE:\s*([0-9.eE+-]+)",
    re.I | re.S,
)
BEST_CKPT = re.compile(r"best_val_MAE\.pt saved", re.I)
TRAIN_TIME = re.compile(r"train_time:\s*([0-9.]+)", re.I)
VAL_TIME = re.compile(r"val_time:\s*([0-9.]+)", re.I)
TEST_TIME = re.compile(r"test_time:\s*([0-9.]+)", re.I)
ERROR_TAIL_LINES = 120
SHOW_ERROR_LINES = 80


def horizon_spec(horizon: int) -> dict[str, Any]:
    if horizon not in HORIZON_CONFIGS:
        raise ValueError(f"Unknown horizon: {horizon}. Choose from {list(HORIZON_CONFIGS)}")
    return HORIZON_CONFIGS[horizon]


def variant_spec(variant: str) -> dict[str, Any]:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant: {variant}")
    return {**COMMON_FIXED, **VARIANT_SPECS[variant]}


def cfg_dir(work_dir: Path) -> Path:
    return work_dir / "configs"


def ckpt_dir_for(horizon: int, variant: str, seed: int, ckpt_root: Path) -> Path:
    return ckpt_root / f"h{horizon}" / variant / f"seed{seed}"


def log_dir_for(horizon: int, variant: str, seed: int, log_root: Path) -> Path:
    return log_root / f"h{horizon}" / variant / f"seed{seed}"


def temp_cfg_path(horizon: int, variant: str, seed: int, work_dir: Path) -> Path:
    out = cfg_dir(work_dir) / f"h{horizon}_{variant}_seed{seed}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def strip_hardcoded_cuda_devices(content: str) -> str:
    kept: list[str] = []
    for line in content.splitlines():
        if "CUDA_VISIBLE_DEVICES" in line and "os.environ" in line:
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def _py_literal(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, list):
        return repr(v)
    return str(v)


def generate_temp_config(
    horizon: int,
    variant: str,
    seed: int,
    work_dir: Path,
    ckpt_root: Path,
) -> Path:
    hspec = horizon_spec(horizon)
    spec = variant_spec(variant)
    base_cfg = hspec["base_cfg"]
    content = strip_hardcoded_cuda_devices(base_cfg.read_text(encoding="utf-8"))
    ckpt_rel = os.path.join(
        "checkpoints", "pems04_long_horizon_graph", f"h{horizon}", variant, f"seed{seed}"
    )
    lines = [
        "",
        "# ===== pems04_long_horizon_graph overrides (auto-generated) =====",
        f"CFG.ENV.SEED = {seed}",
        "if hasattr(CFG, 'SEED'):",
        f"    CFG.SEED = {seed}",
        "if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):",
        f"    CFG.TRAIN.SEED = {seed}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")',
        "CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]",
        "CFG.MODEL.TARGET_FEATURES = [0]",
        (
            f'CFG.DESCRIPTION = "PeMS04 long horizon h{horizon}: {variant} seed{seed}"'
        ),
    ]
    for key, val in spec.items():
        if val is None:
            continue
        lines.append(f'CFG.MODEL.PARAM["{key}"] = {_py_literal(val)}')
    out = temp_cfg_path(horizon, variant, seed, work_dir)
    out.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")
    return out


def load_cfg(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("long_horizon_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CFG


def validate_generated_config(cfg_path: Path, horizon: int, variant: str) -> None:
    hspec = horizon_spec(horizon)
    spec = variant_spec(variant)
    cfg = load_cfg(cfg_path)
    if int(cfg.DATASET_INPUT_LEN) != int(hspec["input_len"]):
        raise ValueError(
            f"{cfg_path}: DATASET_INPUT_LEN expected {hspec['input_len']}, "
            f"got {cfg.DATASET_INPUT_LEN}"
        )
    if int(cfg.DATASET_OUTPUT_LEN) != int(hspec["output_len"]):
        raise ValueError(
            f"{cfg_path}: DATASET_OUTPUT_LEN expected {hspec['output_len']}, "
            f"got {cfg.DATASET_OUTPUT_LEN}"
        )
    param = cfg.MODEL.PARAM
    if list(param.get("chain_lengths", [])) != list(hspec["chain_lengths"]):
        raise ValueError(
            f"{cfg_path}: chain_lengths expected {hspec['chain_lengths']}, "
            f"got {param.get('chain_lengths')}"
        )
    placement = str(param.get("spatial_placement", "")).lower()
    post_mode = str(param.get("post_spatial_mode", "")).lower()
    if placement != str(spec["spatial_placement"]).lower():
        raise ValueError(
            f"{cfg_path}: spatial_placement expected {spec['spatial_placement']}, got {placement}"
        )
    if post_mode != str(spec["post_spatial_mode"]).lower():
        raise ValueError(
            f"{cfg_path}: post_spatial_mode expected {spec['post_spatial_mode']}, got {post_mode}"
        )
    if placement == "temporal_first_graph_resolution":
        lengths = []
        for key in GRAPH_RES_KEYS:
            if key not in param:
                raise ValueError(f"{cfg_path}: missing {key} for graph resolution variant")
            lengths.append(len(list(param[key])))
        if len(set(lengths)) != 1:
            raise ValueError(
                f"{cfg_path}: graph_resolution_* lengths mismatch: "
                f"{dict(zip(GRAPH_RES_KEYS, lengths))}"
            )
        if "clustering_seed" not in param:
            raise ValueError(f"{cfg_path}: missing clustering_seed")
        if param.get("spatial_graph_loss_weights") not in (None, "", []):
            raise ValueError(f"{cfg_path}: spatial_graph_loss_weights must not be enabled")
    if int(cfg.ENV.SEED) != int(cfg_path.stem.split("seed")[-1]):
        raise ValueError(f"{cfg_path}: seed mismatch in filename vs CFG.ENV.SEED")


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


def pick_canonical_training_log(ckpt_base: Path) -> Path | None:
    if not ckpt_base.is_dir():
        return None
    run_dirs = sorted(
        [d for d in ckpt_base.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    candidates: list[Path] = []
    for run_dir in run_dirs:
        candidates.extend(run_dir.glob("training_log_*.log"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def collect_log_text(
    horizon: int, variant: str, seed: int, log_root: Path, ckpt_root: Path
) -> str:
    parts: list[str] = []
    log_dir = log_dir_for(horizon, variant, seed, log_root)
    wrapper = log_dir / "train.log"
    if wrapper.is_file():
        parts.append(wrapper.read_text(errors="replace"))
    ckpt_base = ckpt_dir_for(horizon, variant, seed, ckpt_root)
    tlog = pick_canonical_training_log(ckpt_base)
    if tlog is not None:
        parts.append(tlog.read_text(errors="replace"))
    return "\n".join(parts)


def collect_log_sources(
    horizon: int, variant: str, seed: int, log_root: Path, ckpt_root: Path
) -> dict[str, str]:
    log_dir = log_dir_for(horizon, variant, seed, log_root)
    ckpt_base = ckpt_dir_for(horizon, variant, seed, ckpt_root)
    tlog = pick_canonical_training_log(ckpt_base)
    return {
        "parsed_wrapper_log": str(log_dir / "train.log") if (log_dir / "train.log").is_file() else "",
        "parsed_training_log": str(tlog) if tlog is not None else "",
        "parsed_checkpoint_dir": str(ckpt_base),
    }


def parse_training_log(log_text: str, source: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {
        "best_val_mae": None,
        "test_mae_at_best_val": None,
        "test_rmse_at_best_val": None,
        "test_mape_at_best_val": None,
        "final_test_mae": None,
        "final_test_rmse": None,
        "final_test_mape": None,
        "best_epoch": None,
        "train_mae_at_best_val": None,
        "epoch_time": None,
        "total_time": None,
        "parsed_val_source": source,
        "parsed_test_source": source,
    }
    if not log_text.strip():
        return out

    current_epoch: int | None = None
    last_train_mae: float | None = None
    last_val_mae: float | None = None
    last_train_time: float | None = None
    last_val_time: float | None = None
    total_time = 0.0
    best_val_so_far = float("inf")
    lines = log_text.splitlines()

    for i, line in enumerate(lines):
        m = EPOCH_LINE.search(line)
        if m:
            current_epoch = int(m.group(1))

        m = TRAIN_LINE.search(line)
        if m:
            last_train_mae = float(m.group(1))
            tm = TRAIN_TIME.search(line)
            if tm:
                last_train_time = float(tm.group(1))
                total_time += last_train_time

        m = VAL_LINE.search(line)
        if m:
            last_val_mae = float(m.group(1))
            vm = VAL_TIME.search(line)
            if vm:
                last_val_time = float(vm.group(1))
                total_time += last_val_time

        if BEST_CKPT.search(line):
            if last_val_mae is None:
                continue
            if last_val_mae <= best_val_so_far:
                best_val_so_far = last_val_mae
                out["best_val_mae"] = last_val_mae
                out["best_epoch"] = current_epoch
                out["train_mae_at_best_val"] = last_train_mae
                epoch_parts = [t for t in (last_train_time, last_val_time) if t is not None]
                for j in range(i + 1, min(i + 40, len(lines))):
                    tm = TEST_BLOCK.search(lines[j])
                    if tm:
                        out["test_mae_at_best_val"] = float(tm.group(1))
                        out["test_rmse_at_best_val"] = float(tm.group(2))
                        out["test_mape_at_best_val"] = float(tm.group(3))
                        out["parsed_test_source"] = source
                        ttm = TEST_TIME.search(lines[j])
                        if ttm:
                            epoch_parts.append(float(ttm.group(1)))
                        break
                if epoch_parts:
                    out["epoch_time"] = sum(epoch_parts)

        m = TEST_BLOCK.search(line)
        if m:
            out["final_test_mae"] = float(m.group(1))
            out["final_test_rmse"] = float(m.group(2))
            out["final_test_mape"] = float(m.group(3))

    if total_time > 0:
        out["total_time"] = total_time
    return out


def fmt_list(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return repr(v)
    return str(v)


def base_row(
    horizon: int, variant: str, seed: int, cfg_path: Path, ckpt_root: Path
) -> dict[str, Any]:
    hspec = horizon_spec(horizon)
    spec = variant_spec(variant)
    return {
        "horizon": horizon,
        "input_len": hspec["input_len"],
        "output_len": hspec["output_len"],
        "chain_lengths": fmt_list(hspec["chain_lengths"]),
        "variant": variant,
        "seed": seed,
        "spatial_placement": spec.get("spatial_placement"),
        "post_spatial_mode": spec.get("post_spatial_mode"),
        "graph_resolution_ratios": fmt_list(spec.get("graph_resolution_ratios")),
        "graph_resolution_alphas": fmt_list(spec.get("graph_resolution_alphas")),
        "graph_resolution_topks": fmt_list(spec.get("graph_resolution_topks")),
        "graph_resolution_betas": fmt_list(spec.get("graph_resolution_betas")),
        "graph_resolution_rhos": fmt_list(spec.get("graph_resolution_rhos")),
        "clustering_seed": spec.get("clustering_seed", ""),
        "graph_cluster_method": spec.get("graph_cluster_method", "current"),
        "best_epoch": None,
        "best_val_mae": None,
        "test_mae_at_best_val": None,
        "test_rmse_at_best_val": None,
        "test_mape_at_best_val": None,
        "final_test_mae": None,
        "train_mae_at_best_val": None,
        "params": count_params(cfg_path),
        "epoch_time": None,
        "total_time": None,
        "status": "pending",
        "error_message": "",
        "config_path": str(cfg_path),
        "checkpoint_dir": str(ckpt_dir_for(horizon, variant, seed, ckpt_root)),
        "log_file": "",
    }


def summarize_row(
    horizon: int,
    variant: str,
    seed: int,
    cfg_path: Path,
    log_root: Path,
    ckpt_root: Path,
) -> dict[str, Any]:
    row = base_row(horizon, variant, seed, cfg_path, ckpt_root)
    log_dir = log_dir_for(horizon, variant, seed, log_root)
    train_log = log_dir / "train.log"
    row["log_file"] = str(train_log) if train_log.is_file() else ""
    sources = collect_log_sources(horizon, variant, seed, log_root, ckpt_root)
    tlog = sources.get("parsed_training_log") or ""
    parsed = parse_training_log(
        collect_log_text(horizon, variant, seed, log_root, ckpt_root), source=tlog
    )
    row.update(parsed)
    row.update(sources)
    if parsed.get("test_mae_at_best_val") is not None:
        row["status"] = "ok"
    elif parsed.get("final_test_mae") is not None:
        row["status"] = "ok_final_only"
    else:
        row["status"] = "failed_no_metrics"
    return row


def is_completed(
    horizon: int,
    variant: str,
    seed: int,
    cfg_path: Path,
    log_root: Path,
    ckpt_root: Path,
) -> bool:
    row = summarize_row(horizon, variant, seed, cfg_path, log_root, ckpt_root)
    return row.get("test_mae_at_best_val") is not None


class GPUScheduler:
    def __init__(self, gpus: list[str], max_workers_per_gpu: int = 1):
        self.gpus = gpus
        self.max_workers = max(1, max_workers_per_gpu)
        self.lock = threading.Lock()
        self.in_use = {g: 0 for g in gpus}

    def acquire(self) -> str:
        while True:
            with self.lock:
                for gpu in self.gpus:
                    if self.in_use[gpu] < self.max_workers:
                        self.in_use[gpu] += 1
                        return gpu
            time.sleep(2)

    def release(self, gpu: str) -> None:
        with self.lock:
            self.in_use[gpu] = max(0, self.in_use[gpu] - 1)


def save_error_tail(
    log_file: Path, horizon: int, variant: str, seed: int, log_root: Path
) -> str:
    log_dir = log_dir_for(horizon, variant, seed, log_root)
    tail_path = log_dir / "error_tail.txt"
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_file.is_file():
        lines = log_file.read_text(errors="replace").splitlines()
        tail_path.write_text("\n".join(lines[-ERROR_TAIL_LINES:]) + "\n", encoding="utf-8")
    else:
        tail_path.write_text("", encoding="utf-8")
    return str(tail_path)


def print_log_tail(
    log_file: Path, horizon: int, variant: str, seed: int, lines: int = SHOW_ERROR_LINES
) -> None:
    if not log_file.is_file():
        return
    text_lines = log_file.read_text(errors="replace").splitlines()
    tail = text_lines[-lines:]
    print(
        f"\n--- error tail: h{horizon} {variant} seed={seed} "
        f"(last {len(tail)} lines) ---"
    )
    for line in tail:
        print(line)
    print("--- end error tail ---\n")


def show_failed_errors(rows: list[dict], lines: int = SHOW_ERROR_LINES) -> None:
    failed = [r for r in rows if not str(r.get("status", "")).startswith("ok")]
    if not failed:
        return
    print("\nFailed run diagnostics:\n")
    for row in sorted(
        failed, key=lambda r: (int(r["horizon"]), r["variant"], int(r["seed"]))
    ):
        log_path = Path(row.get("log_file", ""))
        print(
            f"  h{row['horizon']} {row['variant']} seed={row['seed']} "
            f"status={row['status']}"
        )
        print_log_tail(
            log_path, int(row["horizon"]), row["variant"], int(row["seed"]), lines=lines
        )


def run_one(
    horizon: int,
    variant: str,
    cfg_path: Path,
    gpu: str,
    seed: int,
    log_root: Path,
    ckpt_root: Path,
    show_errors: bool,
) -> dict[str, Any]:
    log_dir = log_dir_for(horizon, variant, seed, log_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, str(RUN_PY), "--cfg", cfg_for_easytorch(cfg_path), "--gpus", "0"]
    row = base_row(horizon, variant, seed, cfg_path, ckpt_root)
    row["log_file"] = str(log_file)
    row["status"] = "running"
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        sources = collect_log_sources(horizon, variant, seed, log_root, ckpt_root)
        tlog = sources.get("parsed_training_log") or ""
        parsed = parse_training_log(
            collect_log_text(horizon, variant, seed, log_root, ckpt_root), source=tlog
        )
        row.update(parsed)
        row.update(sources)
        if proc.returncode != 0:
            row["status"] = f"exit_{proc.returncode}"
            row["error_message"] = save_error_tail(log_file, horizon, variant, seed, log_root)
            if show_errors:
                print_log_tail(log_file, horizon, variant, seed)
        elif parsed.get("test_mae_at_best_val") is None and parsed.get("final_test_mae") is None:
            row["status"] = "failed_no_metrics"
            row["error_message"] = save_error_tail(log_file, horizon, variant, seed, log_root)
        elif parsed.get("test_mae_at_best_val") is not None:
            row["status"] = "ok"
        else:
            row["status"] = "ok_final_only"
    except Exception as e:
        row["status"] = f"error:{e}"
        row["error_message"] = str(e)
    return row


def build_jobs(
    horizons: list[int],
    variants: list[str],
    seeds: list[int],
    schedule_order: str,
) -> list[tuple[int, str, int]]:
    jobs: list[tuple[int, str, int]] = []
    if schedule_order == "variant_first":
        for horizon in horizons:
            for variant in variants:
                for seed in seeds:
                    jobs.append((horizon, variant, seed))
    elif schedule_order == "horizon_first":
        for seed in seeds:
            for horizon in horizons:
                for variant in variants:
                    jobs.append((horizon, variant, seed))
    else:
        for seed in seeds:
            for horizon in horizons:
                for variant in variants:
                    jobs.append((horizon, variant, seed))
    return jobs


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def fmt_val(v: Any, digits: int = 4) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def build_variant_summary(rows: list[dict], group_key: str = "variant") -> list[dict[str, Any]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        key = str(row[group_key])
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, runs in grouped.items():
        ok = [r for r in runs if r.get("test_mae_at_best_val") is not None]
        mae_vals = [float(r["test_mae_at_best_val"]) for r in ok]
        rmse_vals = [
            float(r["test_rmse_at_best_val"]) for r in ok if r.get("test_rmse_at_best_val") is not None
        ]
        mape_vals = [
            float(r["test_mape_at_best_val"]) for r in ok if r.get("test_mape_at_best_val") is not None
        ]
        m_mae, s_mae = mean_std(mae_vals)
        m_rmse, s_rmse = mean_std(rmse_vals)
        m_mape, s_mape = mean_std(mape_vals)
        summary.append({
            group_key: key,
            "n": len(ok),
            "failed": len(runs) - len(ok),
            "mae_mean": m_mae,
            "mae_std": s_mae,
            "rmse_mean": m_rmse,
            "rmse_std": s_rmse,
            "mape_mean": m_mape,
            "mape_std": s_mape,
        })
    summary.sort(key=lambda s: (math.inf if math.isnan(s["mae_mean"]) else s["mae_mean"]))
    return summary


def build_horizon_variant_summary(rows: list[dict]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["horizon"]), row["variant"]), []).append(row)
    summary = []
    for (horizon, variant), runs in grouped.items():
        ok = [r for r in runs if r.get("test_mae_at_best_val") is not None]
        mae_vals = [float(r["test_mae_at_best_val"]) for r in ok]
        m_mae, s_mae = mean_std(mae_vals)
        summary.append({
            "horizon": horizon,
            "variant": variant,
            "n": len(ok),
            "failed": len(runs) - len(ok),
            "mae_mean": m_mae,
            "mae_std": s_mae,
        })
    summary.sort(key=lambda s: (s["horizon"], math.inf if math.isnan(s["mae_mean"]) else s["mae_mean"]))
    return summary


def auto_conclusions(rows: list[dict], hv_summary: list[dict]) -> list[str]:
    lines: list[str] = []
    ok_summary = [s for s in hv_summary if s["n"] > 0 and not math.isnan(s["mae_mean"])]
    if not ok_summary:
        lines.append("- 暂无可汇总结果。")
        return lines

    horizons = sorted({int(s["horizon"]) for s in ok_summary})
    for horizon in horizons:
        h_rows = [s for s in ok_summary if int(s["horizon"]) == horizon]
        if not h_rows:
            continue
        best = min(h_rows, key=lambda s: s["mae_mean"])
        lines.append(
            f"- h{horizon} (16→{horizon}) 最优: **{best['variant']}** "
            f"(MAE = {fmt_val(best['mae_mean'])}, std = {fmt_val(best['mae_std'])})."
        )
        g1 = next((s for s in h_rows if s["variant"] == "G1_final_adaptive"), None)
        gr7 = next((s for s in h_rows if s["variant"] == "GR7_sparse_topk"), None)
        gr14 = next((s for s in h_rows if s["variant"] == "GR14_two_level_sparse"), None)
        g0 = next((s for s in h_rows if s["variant"] == "G0_no_spatial"), None)
        if g1 is not None and gr7 is not None:
            diff = gr7["mae_mean"] - g1["mae_mean"]
            lines.append(
                f"  - GR7 vs G1: {'更好' if diff < 0 else '更差' if diff > 0 else '持平'} "
                f"(ΔMAE = {fmt_val(diff)})."
            )
        if g1 is not None and gr14 is not None:
            diff = gr14["mae_mean"] - g1["mae_mean"]
            lines.append(
                f"  - GR14 vs G1: {'更好' if diff < 0 else '更差' if diff > 0 else '持平'} "
                f"(ΔMAE = {fmt_val(diff)})."
            )
        if g0 is not None and g1 is not None:
            diff = g1["mae_mean"] - g0["mae_mean"]
            lines.append(
                f"  - G1 vs G0: spatial {'有效' if diff < 0 else '无效' if diff > 0 else '持平'} "
                f"(ΔMAE = {fmt_val(diff)})."
            )

    gr7_by_h = {
        int(s["horizon"]): s["mae_mean"]
        for s in ok_summary
        if s["variant"] == "GR7_sparse_topk"
    }
    g1_by_h = {
        int(s["horizon"]): s["mae_mean"]
        for s in ok_summary
        if s["variant"] == "G1_final_adaptive"
    }
    if len(gr7_by_h) >= 2 and len(g1_by_h) >= 2:
        gr7_trend = [gr7_by_h.get(h) for h in horizons if gr7_by_h.get(h) is not None]
        g1_trend = [g1_by_h.get(h) for h in horizons if g1_by_h.get(h) is not None]
        if len(gr7_trend) >= 2 and len(g1_trend) >= 2:
            gr7_gap = gr7_trend[-1] - gr7_trend[0]
            g1_gap = g1_trend[-1] - g1_trend[0]
            if gr7_gap < g1_gap:
                lines.append(
                    "- 随 horizon 增大，GR7 相对 G1 的 MAE 增幅更小，"
                    "Graph Resolution 在长 horizon 上可能更有优势。"
                )
            elif gr7_gap > g1_gap:
                lines.append(
                    "- 随 horizon 增大，GR7 相对 G1 的 MAE 增幅更大，"
                    "Graph Resolution 在长 horizon 上优势不明显。"
                )
    return lines


def ensure_data(horizons: list[int]) -> int:
    if not PREPARE_SCRIPT.is_file():
        print(f"Missing data preparation script: {PREPARE_SCRIPT}")
        return 1
    cmd = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--input-len",
        str(INPUT_LEN),
        "--horizons",
        *[str(h) for h in horizons],
    ]
    print(f"[prepare_data] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def write_outputs(rows: list[dict], out_csv: Path, out_md: Path) -> None:
    rows = sorted(rows, key=lambda r: (int(r["horizon"]), r["variant"], int(r["seed"])))
    fields = [
        "horizon", "input_len", "output_len", "chain_lengths",
        "variant", "seed", "spatial_placement", "post_spatial_mode",
        "graph_resolution_ratios", "graph_resolution_alphas", "graph_resolution_topks",
        "graph_resolution_betas", "graph_resolution_rhos", "clustering_seed",
        "graph_cluster_method",
        "best_epoch", "best_val_mae", "test_mae_at_best_val", "test_rmse_at_best_val",
        "test_mape_at_best_val", "final_test_mae", "train_mae_at_best_val",
        "params", "epoch_time", "total_time", "status", "error_message",
        "config_path", "checkpoint_dir", "log_file",
        "parsed_training_log", "parsed_wrapper_log", "parsed_checkpoint_dir",
        "parsed_val_source", "parsed_test_source",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    hv_summary = build_horizon_variant_summary(rows)
    summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
    sum_fields = [
        "horizon", "variant", "n", "failed", "mae_mean", "mae_std",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(hv_summary)

    md = [
        "# PeMS04 Long-Horizon Graph Resolution Ablation\n\n",
        "Protocol: INPUT_LEN=16, horizons 16/32/64, KASA TemporalStep, "
        "use_prev_condition=True, no prior / no PTSE / no MTSR-P, "
        "no graph stage loss.\n\n",
        "## Per-run results\n\n",
        "| horizon | variant | seed | chain_lengths | test MAE@best | test RMSE@best | "
        "test MAPE@best | best val MAE | best epoch | status |\n",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---|\n",
    ]
    for r in rows:
        md.append(
            f"| {r['horizon']} | {r['variant']} | {r['seed']} | {r.get('chain_lengths','')} | "
            f"{fmt_val(r.get('test_mae_at_best_val'))} | {fmt_val(r.get('test_rmse_at_best_val'))} | "
            f"{fmt_val(r.get('test_mape_at_best_val'))} | {fmt_val(r.get('best_val_mae'))} | "
            f"{r.get('best_epoch','')} | {r.get('status','')} |\n"
        )

    md.append("\n## Per-horizon variant summary\n\n")
    md.append("| horizon | variant | n | MAE mean | MAE std | failed |\n")
    md.append("|---:|---|---:|---:|---:|---:|\n")
    for s in hv_summary:
        md.append(
            f"| {s['horizon']} | {s['variant']} | {s['n']} | "
            f"{fmt_val(s['mae_mean'])} | {fmt_val(s['mae_std'])} | {s['failed']} |\n"
        )

    md.append("\n## Auto conclusions\n\n")
    for line in auto_conclusions(rows, hv_summary):
        md.append(line + "\n")

    out_md.write_text("".join(md), encoding="utf-8")


def dry_run_info(horizon: int, variant: str, seed: int, cfg_path: Path, ckpt_root: Path) -> None:
    hspec = horizon_spec(horizon)
    spec = variant_spec(variant)
    cmd = f"{sys.executable} {RUN_PY} --cfg {cfg_for_easytorch(cfg_path)} --gpus <GPU>"
    print(f"  [h{horizon} {variant} seed={seed}]")
    print(f"    config_path: {cfg_path}")
    print(f"    checkpoint_dir: {ckpt_dir_for(horizon, variant, seed, ckpt_root)}")
    print(f"    input_len: {hspec['input_len']}, output_len: {hspec['output_len']}")
    print(f"    chain_lengths: {hspec['chain_lengths']}")
    print(f"    spatial_placement: {spec.get('spatial_placement')}")
    print(f"    post_spatial_mode: {spec.get('post_spatial_mode')}")
    if spec.get("spatial_placement") == "temporal_first_graph_resolution":
        for key in GRAPH_RES_KEYS:
            print(f"    {key}: {spec.get(key)}")
    print(f"    params: {count_params(cfg_path)}")
    print(f"    cmd: {cmd}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PeMS04 long-horizon Graph Resolution ablation (16→16/32/64)."
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=list(VARIANT_SPECS.keys()),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS)
    parser.add_argument("--work_dir", default=str(DEFAULT_WORK_DIR.relative_to(ROOT)))
    parser.add_argument("--out", default="results/pems04_16input_long_horizon_graph.csv")
    parser.add_argument("--markdown", default="results/pems04_16input_long_horizon_graph.md")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--prepare_data", action="store_true", help="Generate missing in16 horizon data first.")
    parser.add_argument("--summary-only", "--summary_only", action="store_true", dest="summary_only")
    parser.add_argument("--show_errors", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_workers_per_gpu", type=int, default=1)
    parser.add_argument(
        "--schedule_order",
        choices=["seed_first", "variant_first", "horizon_first"],
        default="seed_first",
    )
    args = parser.parse_args()

    for horizon in args.horizons:
        if horizon not in HORIZON_CONFIGS:
            print(f"Unknown horizon: {horizon}. Choose from {list(HORIZON_CONFIGS)}")
            return 1
        base_cfg = horizon_spec(horizon)["base_cfg"]
        if not base_cfg.is_file():
            print(f"Missing base config for h{horizon}: {base_cfg}")
            return 1

    if not RUN_PY.is_file():
        print(f"Missing run entry: {RUN_PY}")
        return 1

    if args.prepare_data or args.dry_run or not args.summary_only:
        rc = ensure_data(args.horizons)
        if rc != 0:
            return rc

    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = ROOT / work_dir
    log_root = DEFAULT_LOG_ROOT
    ckpt_root = DEFAULT_CKPT_ROOT
    out_csv = Path(args.out)
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    out_md = Path(args.markdown)
    if not out_md.is_absolute():
        out_md = ROOT / out_md

    job_keys = build_jobs(args.horizons, args.variants, args.seeds, args.schedule_order)
    configs: list[tuple[int, str, int, Path]] = []
    for horizon, variant, seed in job_keys:
        cfg_path = generate_temp_config(horizon, variant, seed, work_dir, ckpt_root)
        try:
            validate_generated_config(cfg_path, horizon, variant)
        except Exception as e:
            print(f"Config validation failed for h{horizon} {variant} seed={seed}: {e}")
            return 1
        configs.append((horizon, variant, seed, cfg_path))

    if args.dry_run:
        print("Dry run — horizons / variants / seeds / GPU queue:\n")
        print(f"  horizons: {args.horizons}")
        print(f"  variants: {args.variants}")
        print(f"  seeds: {args.seeds}")
        print(f"  gpus: {args.gpus}")
        print(f"  schedule_order: {args.schedule_order}")
        print()
        for horizon, variant, seed, cfg_path in configs:
            dry_run_info(horizon, variant, seed, cfg_path, ckpt_root)
        print(f"\n{len(configs)} jobs queued.")
        return 0

    if args.summary_only:
        rows = [
            summarize_row(h, v, s, p, log_root, ckpt_root)
            for h, v, s, p in configs
        ]
        write_outputs(rows, out_csv, out_md)
        print(f"Wrote {out_csv}, {out_md}, {out_csv.with_name(out_csv.stem + '_summary.csv')}")
        if args.show_errors:
            show_failed_errors(rows)
        return 0

    scheduler = GPUScheduler(args.gpus, max_workers_per_gpu=args.max_workers_per_gpu)
    rows: list[dict] = []
    lock = threading.Lock()

    def worker(horizon: int, variant: str, seed: int, cfg_path: Path):
        if args.skip_existing and is_completed(horizon, variant, seed, cfg_path, log_root, ckpt_root):
            row = summarize_row(horizon, variant, seed, cfg_path, log_root, ckpt_root)
            with lock:
                rows.append(row)
            print(
                f"[skip] h{horizon} {variant} seed={seed} "
                f"mae={row.get('test_mae_at_best_val')}"
            )
            return
        gpu = scheduler.acquire()
        try:
            print(f"[start] h{horizon} {variant} seed={seed} gpu={gpu}")
            row = run_one(
                horizon, variant, cfg_path, gpu, seed, log_root, ckpt_root, args.show_errors
            )
            with lock:
                rows.append(row)
            print(
                f"[done]  h{horizon} {variant} seed={seed} status={row['status']} "
                f"mae={row.get('test_mae_at_best_val')}"
            )
        finally:
            scheduler.release(gpu)

    threads = [
        threading.Thread(target=worker, args=(horizon, variant, seed, cfg_path))
        for horizon, variant, seed, cfg_path in configs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    write_outputs(rows, out_csv, out_md)
    print(f"Wrote {out_csv}, {out_md}, {out_csv.with_name(out_csv.stem + '_summary.csv')}")
    if args.show_errors:
        show_failed_errors(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
