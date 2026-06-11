#!/usr/bin/env python3
"""Unified KASA graph spectral prior / calibration ablation on PeMS04."""
from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = ROOT / "examples" / "KASAST_graph_spectral" / "KASAST_graph_spectral_PEMS04.py"
RUN_PY = ROOT / "examples" / "run.py"
TEMP_CFG_DIR = ROOT / "tmp_configs" / "kasa_unified_graph_spectral"
LOG_DIR = ROOT / "results" / "kasa_unified_graph_spectral_logs"
CKPT_ROOT = ROOT / "checkpoints" / "kasa_unified_graph_spectral"
BASE_DATASET_DIR = ROOT / "datasets" / "PEMS04"

DEFAULT_VARIANTS = [
    "base_adaptive",
    "offline_low_k16",
    "offline_low_k32",
    "offline_low_k64",
    "offline_low_high_k32",
    "online_calib",
    "offline_low_k32_online_calib",
    "offline_low_high_k32_online_calib",
]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]

BEST_SETTINGS = {
    "use_patch_branch": True,
    "use_downsample_branch": True,
    "use_linear_residual_branch": True,
    "post_spatial_mode": "adaptive_only",
    "use_pre_temporal_spatial_enhancement": False,
    "keep_output_prior_residual": False,
    "use_input_prior_enhancement": False,
}

VARIANT_SPECS: dict[str, dict] = {
    "base_adaptive": {
        "data_npz": "datasets/PEMS04/data.npz",
        "data_dir": "datasets/PEMS04",
        "forward_features": [0, 1, 2, 3],
        "target_features": [0],
        "input_dim": 4,
        "use_extra_prior_input": False,
        "main_input_dim": 3,
        "use_graph_spectral_calibration": False,
        "graph_spectral_calibration_mode": "none",
        "graph_spectral_k": 32,
        "graph_spectral_gamma_init": 0.0,
    },
    "offline_low_k16": {
        "data_npz": "datasets/PEMS04/data_graph_spectral_k16.npz",
        "data_dir": "datasets/PEMS04_graph_spectral_k16",
        "forward_features": [0, 1, 2, 3],
        "target_features": [0],
        "input_dim": 4,
        "use_extra_prior_input": True,
        "main_input_dim": 4,
        "use_graph_spectral_calibration": False,
        "graph_spectral_calibration_mode": "none",
        "graph_spectral_k": 16,
        "graph_spectral_gamma_init": 0.0,
    },
    "offline_low_k32": {
        "data_npz": "datasets/PEMS04/data_graph_spectral_k32.npz",
        "data_dir": "datasets/PEMS04_graph_spectral_k32",
        "forward_features": [0, 1, 2, 3],
        "target_features": [0],
        "input_dim": 4,
        "use_extra_prior_input": True,
        "main_input_dim": 4,
        "use_graph_spectral_calibration": False,
        "graph_spectral_calibration_mode": "none",
        "graph_spectral_k": 32,
        "graph_spectral_gamma_init": 0.0,
    },
    "offline_low_k64": {
        "data_npz": "datasets/PEMS04/data_graph_spectral_k64.npz",
        "data_dir": "datasets/PEMS04_graph_spectral_k64",
        "forward_features": [0, 1, 2, 3],
        "target_features": [0],
        "input_dim": 4,
        "use_extra_prior_input": True,
        "main_input_dim": 4,
        "use_graph_spectral_calibration": False,
        "graph_spectral_calibration_mode": "none",
        "graph_spectral_k": 64,
        "graph_spectral_gamma_init": 0.0,
    },
    "offline_low_high_k32": {
        "data_npz": "datasets/PEMS04/data_graph_spectral_k32_lowhigh.npz",
        "data_dir": "datasets/PEMS04_graph_spectral_k32_lowhigh",
        "forward_features": [0, 1, 2, 3, 4],
        "target_features": [0],
        "input_dim": 5,
        "use_extra_prior_input": True,
        "main_input_dim": 5,
        "use_graph_spectral_calibration": False,
        "graph_spectral_calibration_mode": "none",
        "graph_spectral_k": 32,
        "graph_spectral_gamma_init": 0.0,
    },
    "online_calib": {
        "data_npz": "datasets/PEMS04/data.npz",
        "data_dir": "datasets/PEMS04",
        "forward_features": [0, 1, 2, 3],
        "target_features": [0],
        "input_dim": 4,
        "use_extra_prior_input": False,
        "main_input_dim": 3,
        "use_graph_spectral_calibration": True,
        "graph_spectral_calibration_mode": "residual_safe_low_high",
        "graph_spectral_k": 32,
        "graph_spectral_gamma_init": 0.0,
    },
    "offline_low_k32_online_calib": {
        "data_npz": "datasets/PEMS04/data_graph_spectral_k32.npz",
        "data_dir": "datasets/PEMS04_graph_spectral_k32",
        "forward_features": [0, 1, 2, 3],
        "target_features": [0],
        "input_dim": 4,
        "use_extra_prior_input": True,
        "main_input_dim": 4,
        "use_graph_spectral_calibration": True,
        "graph_spectral_calibration_mode": "residual_safe_low_high",
        "graph_spectral_k": 32,
        "graph_spectral_gamma_init": 0.0,
    },
    "offline_low_high_k32_online_calib": {
        "data_npz": "datasets/PEMS04/data_graph_spectral_k32_lowhigh.npz",
        "data_dir": "datasets/PEMS04_graph_spectral_k32_lowhigh",
        "forward_features": [0, 1, 2, 3, 4],
        "target_features": [0],
        "input_dim": 5,
        "use_extra_prior_input": True,
        "main_input_dim": 5,
        "use_graph_spectral_calibration": True,
        "graph_spectral_calibration_mode": "residual_safe_low_high",
        "graph_spectral_k": 32,
        "graph_spectral_gamma_init": 0.0,
    },
}

PRIOR_GENERATION_COMMANDS = {
    "datasets/PEMS04/data_graph_spectral_k16.npz": (
        "python scripts/generate_pems04_graph_spectral_prior.py "
        "--input_npz datasets/PEMS04/data_in12_out12.npz "
        "--adj_path datasets/PEMS04/adj_mx.pkl "
        "--output_npz datasets/PEMS04/data_graph_spectral_k16.npz "
        "--k 16 --write_dataset_dir"
    ),
    "datasets/PEMS04/data_graph_spectral_k32.npz": (
        "python scripts/generate_pems04_graph_spectral_prior.py "
        "--input_npz datasets/PEMS04/data_in12_out12.npz "
        "--adj_path datasets/PEMS04/adj_mx.pkl "
        "--output_npz datasets/PEMS04/data_graph_spectral_k32.npz "
        "--k 32 --write_dataset_dir"
    ),
    "datasets/PEMS04/data_graph_spectral_k64.npz": (
        "python scripts/generate_pems04_graph_spectral_prior.py "
        "--input_npz datasets/PEMS04/data_in12_out12.npz "
        "--adj_path datasets/PEMS04/adj_mx.pkl "
        "--output_npz datasets/PEMS04/data_graph_spectral_k64.npz "
        "--k 64 --write_dataset_dir"
    ),
    "datasets/PEMS04/data_graph_spectral_k32_lowhigh.npz": (
        "python scripts/generate_pems04_graph_spectral_prior.py "
        "--input_npz datasets/PEMS04/data_in12_out12.npz "
        "--adj_path datasets/PEMS04/adj_mx.pkl "
        "--output_npz datasets/PEMS04/data_graph_spectral_k32_lowhigh.npz "
        "--k 32 --include_high --write_dataset_dir"
    ),
}

PAIRED_DIFFS = [
    ("offline_low_k32", "base_adaptive"),
    ("offline_low_high_k32", "base_adaptive"),
    ("online_calib", "base_adaptive"),
    ("offline_low_k32_online_calib", "base_adaptive"),
    ("offline_low_high_k32_online_calib", "base_adaptive"),
    ("offline_low_k32_online_calib", "offline_low_k32"),
    ("offline_low_high_k32_online_calib", "offline_low_high_k32"),
]

METRIC_PATTERNS = {
    "mae": re.compile(r"test_MAE:\s*([0-9.eE+-]+)", re.I),
    "rmse": re.compile(r"test_RMSE:\s*([0-9.eE+-]+)", re.I),
    "mape": re.compile(r"test_MAPE:\s*([0-9.eE+-]+)", re.I),
}
RESULT_TEST_BLOCK = re.compile(
    r"Result\s*(?:<[^>]+>)?\s*:\s*\[(.*?)\]",
    re.I | re.S,
)
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def variant_spec(variant: str) -> dict:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant: {variant}")
    return {**BEST_SETTINGS, **VARIANT_SPECS[variant]}


def ckpt_dir_for(variant: str, seed: int) -> Path:
    return CKPT_ROOT / f"{variant}_seed{seed}"


def temp_cfg_path(variant: str, seed: int) -> Path:
    TEMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_CFG_DIR / f"{variant}_seed{seed}.py"


def strip_hardcoded_cuda_devices(content: str) -> str:
    kept: list[str] = []
    for line in content.splitlines():
        if "CUDA_VISIBLE_DEVICES" in line and "os.environ" in line:
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def _py_literal(v) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, list):
        return repr(v)
    return str(v)


def ensure_dataset_pkl(npz_path: Path, dataset_dir: Path, input_len: int = 12, output_len: int = 12) -> None:
    """Build training pkl from graph spectral npz without overwriting PEMS04."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = dataset_dir / f"data_in{input_len}_out{output_len}.pkl"
    index_path = dataset_dir / f"index_in{input_len}_out{output_len}.pkl"
    if pkl_path.is_file() and index_path.is_file():
        return

    archive = np.load(npz_path)
    key = "processed_data" if "processed_data" in archive else "data"
    if key not in archive:
        raise KeyError(f"{npz_path} must contain key 'processed_data' or 'data'")
    processed_data = np.array(archive[key], dtype=np.float32)

    base_index = BASE_DATASET_DIR / f"index_in{input_len}_out{output_len}.pkl"
    if not base_index.is_file():
        raise FileNotFoundError(
            f"Missing base index file: {base_index}. "
            "Run PeMS04 data preparation first."
        )
    shutil.copy2(base_index, index_path)

    with open(pkl_path, "wb") as f:
        pickle.dump({"processed_data": processed_data}, f)

    adj_src = BASE_DATASET_DIR / "adj_mx.pkl"
    adj_dst = dataset_dir / "adj_mx.pkl"
    if adj_src.is_file() and not adj_dst.is_file():
        shutil.copy2(adj_src, adj_dst)

    scaler_src = BASE_DATASET_DIR / f"scaler_in{input_len}_out{output_len}.pkl"
    scaler_dst = dataset_dir / scaler_src.name
    if scaler_src.is_file() and not scaler_dst.is_file():
        shutil.copy2(scaler_src, scaler_dst)
    elif not scaler_dst.is_file():
        raise FileNotFoundError(
            f"Missing scaler file: {scaler_src}. "
            "Cannot build graph spectral dataset without base PeMS04 scaler."
        )


def check_required_prior_files(variants: list[str]) -> int:
    missing: dict[str, str] = {}
    for variant in variants:
        spec = variant_spec(variant)
        npz_rel = spec["data_npz"]
        if npz_rel == "datasets/PEMS04/data.npz":
            continue
        npz_path = ROOT / npz_rel
        if not npz_path.is_file() and npz_rel in PRIOR_GENERATION_COMMANDS:
            missing[npz_rel] = PRIOR_GENERATION_COMMANDS[npz_rel]

    if not missing:
        return 0

    print("Missing required graph spectral prior files:\n")
    for npz_rel, cmd in missing.items():
        print(f"  - {npz_rel}")
        print(f"    {cmd}\n")
    return 1


def prepare_variant_dataset(variant: str) -> None:
    spec = variant_spec(variant)
    npz_path = ROOT / spec["data_npz"]
    dataset_dir = ROOT / spec["data_dir"]
    if spec["data_npz"] != "datasets/PEMS04/data.npz":
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing prior npz: {npz_path}")
        ensure_dataset_pkl(npz_path, dataset_dir)


def generate_temp_config(variant: str, seed: int, prepare_data: bool = True) -> Path:
    if prepare_data:
        prepare_variant_dataset(variant)
    spec = variant_spec(variant)
    content = strip_hardcoded_cuda_devices(BASE_CFG.read_text(encoding="utf-8"))
    lines = [
        "",
        "# ===== kasa_unified_graph_spectral overrides (auto-generated) =====",
        f"CFG.ENV.SEED = {seed}",
        "if hasattr(CFG, 'SEED'):",
        f"    CFG.SEED = {seed}",
        "if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):",
        f"    CFG.TRAIN.SEED = {seed}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "kasa_unified_graph_spectral", "{variant}_seed{seed}")',
        f"CFG.TRAIN.DATA.DIR = {_py_literal(spec['data_dir'])}",
        f"CFG.VAL.DATA.DIR = {_py_literal(spec['data_dir'])}",
        f"CFG.TEST.DATA.DIR = {_py_literal(spec['data_dir'])}",
        f"CFG.MODEL.FORWARD_FEATURES = {_py_literal(spec['forward_features'])}",
        f"CFG.MODEL.TARGET_FEATURES = {_py_literal(spec['target_features'])}",
        f'CFG.MODEL.PARAM["input_dim"] = {spec["input_dim"]}',
        f'CFG.MODEL.PARAM["adj_mx_path"] = os.path.join({_py_literal(spec["data_dir"])}, "adj_mx.pkl")',
    ]
    for key in [
        "use_patch_branch",
        "use_downsample_branch",
        "use_linear_residual_branch",
        "post_spatial_mode",
        "use_pre_temporal_spatial_enhancement",
        "keep_output_prior_residual",
        "use_input_prior_enhancement",
        "use_extra_prior_input",
        "main_input_dim",
        "use_graph_spectral_calibration",
        "graph_spectral_calibration_mode",
        "graph_spectral_k",
        "graph_spectral_gamma_init",
    ]:
        lines.append(f'CFG.MODEL.PARAM["{key}"] = {_py_literal(spec[key])}')
    out = temp_cfg_path(variant, seed)
    out.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")
    return out


def cfg_for_easytorch(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


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


def collect_log_text(variant: str, seed: int, wrapper_log: Path | None) -> str:
    """Collect wrapper log plus all checkpoint training logs (newest first).

    Reading every training_log avoids false ``failed_no_metrics`` when EasyTorch
    resumes from an epoch-100 checkpoint and the newest log has no test block.
    """
    parts: list[str] = []
    if wrapper_log and wrapper_log.is_file():
        parts.append(wrapper_log.read_text(errors="replace"))
    ckpt_base = ckpt_dir_for(variant, seed)
    if ckpt_base.is_dir():
        tlogs = sorted(
            ckpt_base.glob("*/training_log_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for tlog in tlogs:
            parts.append(tlog.read_text(errors="replace"))
    return "\n".join(parts)


ERROR_TAIL_LINES = 120
SHOW_ERROR_LINES = 80


def save_error_tail(
    wrapper_log: Path,
    variant: str,
    seed: int,
    tail_lines: int = ERROR_TAIL_LINES,
) -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tail_path = LOG_DIR / f"{variant}_seed{seed}_error_tail.txt"
    if wrapper_log.is_file():
        lines = wrapper_log.read_text(errors="replace").splitlines()
        tail_path.write_text("\n".join(lines[-tail_lines:]) + "\n", encoding="utf-8")
    else:
        tail_path.write_text("", encoding="utf-8")
    return str(tail_path)


def print_log_tail(
    wrapper_log: Path,
    variant: str,
    seed: int,
    lines: int = SHOW_ERROR_LINES,
) -> None:
    if not wrapper_log.is_file():
        return
    text_lines = wrapper_log.read_text(errors="replace").splitlines()
    tail = text_lines[-lines:]
    print(f"\n--- error tail: {variant} seed={seed} (last {len(tail)} lines) ---")
    for line in tail:
        print(line)
    print("--- end error tail ---\n")


def show_failed_errors(rows: list[dict], lines: int = SHOW_ERROR_LINES) -> None:
    failed = [r for r in rows if str(r.get("status", "")).startswith("exit_")]
    if not failed:
        return
    print("\nFailed run diagnostics:\n")
    for row in sorted(failed, key=lambda r: (r["variant"], int(r["seed"]))):
        log_path = Path(row.get("log_file", ""))
        print(f"  {row['variant']} seed={row['seed']} status={row['status']}")
        if row.get("error_tail_file"):
            print(f"    error_tail_file: {row['error_tail_file']}")
        print_log_tail(log_path, row["variant"], int(row["seed"]), lines=lines)


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


def run_one(variant: str, cfg_path: Path, gpu: str, seed: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{variant}_seed{seed}_gpu{gpu}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, str(RUN_PY), "--cfg", cfg_for_easytorch(cfg_path), "--gpus", str(gpu)]
    row = {
        "variant": variant,
        "seed": seed,
        "mae": None,
        "rmse": None,
        "mape": None,
        "ckpt_dir": str(ckpt_dir_for(variant, seed)),
        "log_file": str(log_file),
        "status": "running",
        "error_tail_file": "",
    }
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        metrics = parse_metrics(collect_log_text(variant, seed, log_file))
        row.update(metrics)
        if proc.returncode != 0:
            row["status"] = f"exit_{proc.returncode}"
            row["error_tail_file"] = save_error_tail(log_file, variant, seed)
            print_log_tail(log_file, variant, seed)
        elif metrics["mae"] is None:
            row["status"] = "failed_no_metrics"
        else:
            row["status"] = "ok"
    except Exception as e:
        row["status"] = f"error:{e}"
    return row


def summarize_row(variant: str, seed: int) -> dict:
    wrapper_log = None
    if LOG_DIR.is_dir():
        matches = sorted(LOG_DIR.glob(f"{variant}_seed{seed}_gpu*.log"))
        if matches:
            wrapper_log = matches[-1]
    error_tail = LOG_DIR / f"{variant}_seed{seed}_error_tail.txt"
    metrics = parse_metrics(collect_log_text(variant, seed, wrapper_log))
    status = "ok" if metrics["mae"] is not None else "failed_no_metrics"
    return {
        "variant": variant,
        "seed": seed,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "ckpt_dir": str(ckpt_dir_for(variant, seed)),
        "log_file": str(wrapper_log) if wrapper_log else "",
        "status": status,
        "error_tail_file": str(error_tail) if error_tail.is_file() else "",
    }


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
    by_variant: dict[str, list[dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)

    summary = []
    for variant, runs in by_variant.items():
        ok = [
            r for r in runs
            if r.get("mae") is not None and r.get("rmse") is not None and r.get("mape") is not None
        ]
        failed = len(runs) - len(ok)
        mae_vals = [float(r["mae"]) for r in ok]
        rmse_vals = [float(r["rmse"]) for r in ok]
        mape_vals = [float(r["mape"]) for r in ok]
        m_mae, s_mae, c_mae = mean_std_ci(mae_vals)
        m_rmse, s_rmse, c_rmse = mean_std_ci(rmse_vals)
        m_mape, s_mape, c_mape = mean_std_ci(mape_vals)
        summary.append({
            "variant": variant,
            "n": len(ok),
            "mae_mean": m_mae,
            "mae_std": s_mae,
            "mae_ci95": c_mae,
            "rmse_mean": m_rmse,
            "rmse_std": s_rmse,
            "rmse_ci95": c_rmse,
            "mape_mean": m_mape,
            "mape_std": s_mape,
            "mape_ci95": c_mape,
            "failed": failed,
        })
    summary.sort(key=lambda s: (math.inf if math.isnan(s["mae_mean"]) else s["mae_mean"]))
    return summary


def paired_diff_table(rows: list[dict]) -> list[dict]:
    by_variant_seed: dict[tuple[str, int], dict] = {}
    for r in rows:
        if r.get("mae") is None:
            continue
        by_variant_seed[(r["variant"], int(r["seed"]))] = r

    out = []
    for a, b in PAIRED_DIFFS:
        diffs = []
        for (variant, seed), row in list(by_variant_seed.items()):
            if variant != a:
                continue
            other = by_variant_seed.get((b, seed))
            if other is None or other.get("mae") is None:
                continue
            diffs.append(float(row["mae"]) - float(other["mae"]))
        m, s, c = mean_std_ci(diffs)
        out.append({
            "pair": f"{a} - {b}",
            "n": len(diffs),
            "mae_diff_mean": m,
            "mae_diff_std": s,
            "mae_diff_ci95": c,
        })
    return out


def fmt_val(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_outputs(rows: list[dict], out_csv: Path, out_md: Path, summary_csv: Path):
    rows = sorted(rows, key=lambda r: (r["variant"], int(r["seed"])))
    fields = [
        "variant", "seed", "mae", "rmse", "mape",
        "ckpt_dir", "log_file", "status", "error_tail_file",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = build_summary(rows)
    sum_fields = [
        "variant", "n", "mae_mean", "mae_std", "mae_ci95",
        "rmse_mean", "rmse_std", "rmse_ci95",
        "mape_mean", "mape_std", "mape_ci95", "failed",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary)

    paired = paired_diff_table(rows)
    md = [
        "# KASA Unified Graph Spectral Prior / Calibration on PeMS04 12→12\n\n",
        "Protocol:\n",
        "official 6:2:2, TARGET=[0],\n",
        "full temporal branches,\n",
        "post_spatial_mode=adaptive_only,\n",
        "no temporal Fourier prior,\n",
        "no output prior residual,\n",
        "no pre-temporal spatial enhancement.\n\n",
        "## Per-run results\n\n",
        "| variant | seed | MAE | RMSE | MAPE | status |\n",
        "|---|---:|---:|---:|---:|---|\n",
    ]
    for r in rows:
        status = r["status"]
        if str(status).startswith("exit_") and r.get("error_tail_file"):
            status = f"{status} ({r['error_tail_file']})"
        md.append(
            f"| {r['variant']} | {r['seed']} | {fmt_val(r['mae'])} | {fmt_val(r['rmse'])} | "
            f"{fmt_val(r['mape'])} | {status} |\n"
        )
    md.append("\n## Summary\n\n")
    md.append(
        "| variant | n | MAE mean | MAE std | MAE 95% CI | RMSE mean | RMSE std | RMSE 95% CI | "
        "MAPE mean | MAPE std | MAPE 95% CI | failed |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in summary:
        md.append(
            f"| {s['variant']} | {s['n']} | {fmt_val(s['mae_mean'])} | {fmt_val(s['mae_std'])} | "
            f"{fmt_val(s['mae_ci95'])} | {fmt_val(s['rmse_mean'])} | {fmt_val(s['rmse_std'])} | "
            f"{fmt_val(s['rmse_ci95'])} | {fmt_val(s['mape_mean'])} | {fmt_val(s['mape_std'])} | "
            f"{fmt_val(s['mape_ci95'])} | {s['failed']} |\n"
        )
    md.append("\n## Paired differences\n\n")
    md.append("Positive value means the first variant is worse.\n\n")
    md.append("| pair | n | MAE diff mean | MAE diff std | MAE diff 95% CI |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for p in paired:
        md.append(
            f"| {p['pair']} | {p['n']} | {fmt_val(p['mae_diff_mean'])} | {fmt_val(p['mae_diff_std'])} | "
            f"{fmt_val(p['mae_diff_ci95'])} |\n"
        )
    out_md.write_text("".join(md), encoding="utf-8")


def dry_run_info(variant: str, seed: int, cfg_path: Path) -> None:
    spec = variant_spec(variant)
    cmd = f"{sys.executable} {RUN_PY} --cfg {cfg_for_easytorch(cfg_path)} --gpus <GPU>"
    print(f"  [{variant} seed={seed}]")
    print(f"    cfg: {cfg_path}")
    print(f"    ckpt: {ckpt_dir_for(variant, seed)}")
    print(f"    data path: {spec['data_npz']}")
    print(f"    data dir: {spec['data_dir']}")
    print(f"    FORWARD_FEATURES: {spec['forward_features']}")
    print(f"    TARGET_FEATURES: {spec['target_features']}")
    print(f"    use_extra_prior_input: {spec['use_extra_prior_input']}")
    print(f"    main_input_dim: {spec['main_input_dim']}")
    print(f"    use_graph_spectral_calibration: {spec['use_graph_spectral_calibration']}")
    print(f"    graph_spectral_calibration_mode: {spec['graph_spectral_calibration_mode']}")
    print(f"    graph_spectral_k: {spec['graph_spectral_k']}")
    print(f"    graph_spectral_gamma_init: {spec['graph_spectral_gamma_init']}")
    print(f"    post_spatial_mode: {spec['post_spatial_mode']}")
    print(f"    use_pre_temporal_spatial_enhancement: {spec['use_pre_temporal_spatial_enhancement']}")
    print(f"    keep_output_prior_residual: {spec['keep_output_prior_residual']}")
    print(f"    use_patch_branch: {spec['use_patch_branch']}")
    print(f"    use_downsample_branch: {spec['use_downsample_branch']}")
    print(f"    use_linear_residual_branch: {spec['use_linear_residual_branch']}")
    print(f"    cmd: {cmd}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified KASA graph spectral prior / calibration ablation on PeMS04."
    )
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANT_SPECS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--out", default="results/pems04_kasa_unified_graph_spectral.csv")
    parser.add_argument("--markdown", default="results/pems04_kasa_unified_graph_spectral.md")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary-only", "--summary_only", action="store_true", dest="summary_only")
    parser.add_argument(
        "--show_errors",
        action="store_true",
        help="Print last 80 log lines for failed variants after run/summary.",
    )
    args = parser.parse_args()

    if not BASE_CFG.is_file():
        print(f"Missing base config: {BASE_CFG}")
        return 1

    out_csv = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_md = ROOT / args.markdown if not Path(args.markdown).is_absolute() else Path(args.markdown)
    summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")

    if args.dry_run:
        jobs = [(v, generate_temp_config(v, s, prepare_data=False), s) for v in args.variants for s in args.seeds]
        print("Dry run — temp configs and commands:\n")
        for variant, cfg_path, seed in jobs:
            dry_run_info(variant, seed, cfg_path)
        print(f"\n{len(jobs)} jobs, GPUs: {args.gpus}")
        return 0

    if check_required_prior_files(args.variants):
        return 1

    jobs = [(v, generate_temp_config(v, s), s) for v in args.variants for s in args.seeds]

    if args.summary_only:
        rows = [summarize_row(v, s) for v, _, s in jobs]
        write_outputs(rows, out_csv, out_md, summary_csv)
        print(f"Wrote {out_csv}, {out_md}, {summary_csv}")
        if args.show_errors:
            show_failed_errors(rows)
        return 0

    queue = GPUQueue(args.gpus)
    rows: list[dict] = []
    lock = threading.Lock()

    def worker(variant: str, cfg_path: Path, seed: int):
        gpu = queue.acquire()
        try:
            print(f"[start] {variant} seed={seed} gpu={gpu}")
            row = run_one(variant, cfg_path, gpu, seed)
            with lock:
                rows.append(row)
            print(f"[done]  {variant} seed={seed} status={row['status']} mae={row['mae']}")
        finally:
            queue.release(gpu)

    threads = []
    for variant, cfg_path, seed in jobs:
        t = threading.Thread(target=worker, args=(variant, cfg_path, seed))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    write_outputs(rows, out_csv, out_md, summary_csv)
    print(f"Wrote {out_csv}, {out_md}, {summary_csv}")
    if args.show_errors:
        show_failed_errors(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
