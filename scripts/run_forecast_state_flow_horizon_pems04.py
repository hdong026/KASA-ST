#!/usr/bin/env python3
"""Forecast-State Flow Chain (FSF) horizon ablation on PeMS04 (H=12, F∈{12,24,48})."""
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
FSF_BASE_CFG = ROOT / "examples" / "ForecastStateFlow" / "ForecastStateFlow_PEMS04.py"
RUN_PY = ROOT / "examples" / "run.py"
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_pems04_fixed_input_horizons.py"
TEMP_CFG_DIR = ROOT / "tmp_configs" / "forecast_state_flow_horizon_pems04"
LOG_DIR = ROOT / "results" / "forecast_state_flow_horizon_pems04_logs"
CKPT_ROOT = ROOT / "checkpoints" / "forecast_state_flow_horizon_pems04"

INPUT_LEN = 12
DEFAULT_HORIZONS = [12, 24, 48]
DEFAULT_VARIANTS = ["fsf_flow", "fsf_no_fm", "fsf_no_native", "fsf_direct"]
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

VARIANT_SPECS: dict[str, dict] = {
    "fsf_flow": {
        "chain_lengths": None,
        "num_flow_steps": 1,
        "final_loss_weight": 1.0,
        "state_loss_weight": 0.3,
        "native_loss_weight": 0.1,
        "fm_loss_weight": 0.2,
    },
    "fsf_no_fm": {
        "chain_lengths": None,
        "num_flow_steps": 1,
        "final_loss_weight": 1.0,
        "state_loss_weight": 0.3,
        "native_loss_weight": 0.1,
        "fm_loss_weight": 0.0,
    },
    "fsf_no_native": {
        "chain_lengths": None,
        "num_flow_steps": 1,
        "final_loss_weight": 1.0,
        "state_loss_weight": 0.3,
        "native_loss_weight": 0.0,
        "fm_loss_weight": 0.2,
    },
    "fsf_direct": {
        "chain_lengths": "direct",
        "num_flow_steps": 1,
        "final_loss_weight": 1.0,
        "state_loss_weight": 0.3,
        "native_loss_weight": 0.1,
        "fm_loss_weight": 0.2,
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
    spec = importlib.util.spec_from_file_location("fsf_horizon_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CFG


def chain_lengths_for(horizon: int, variant: str) -> list[int]:
    if variant == "fsf_direct":
        return [horizon]
    if horizon not in HORIZON_CHAIN_LENGTHS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    return HORIZON_CHAIN_LENGTHS[horizon]


def variant_spec(variant: str, horizon: int) -> dict:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant: {variant}")
    spec = dict(VARIANT_SPECS[variant])
    if spec.get("chain_lengths") != "direct":
        spec["chain_lengths"] = chain_lengths_for(horizon, variant)
    else:
        spec["chain_lengths"] = [horizon]
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
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, list):
        return repr(v)
    return str(v)


def generate_temp_config(horizon: int, variant: str, seed: int) -> Path:
    spec = variant_spec(variant, horizon)
    content = strip_hardcoded_cuda_devices(FSF_BASE_CFG.read_text(encoding="utf-8"))
    ckpt_rel = os.path.join(
        "checkpoints", "forecast_state_flow_horizon_pems04", f"h{horizon}", f"{variant}_seed{seed}"
    )
    lines = [
        "",
        "# ===== forecast_state_flow_horizon_pems04 overrides (auto-generated) =====",
        f"CFG.ENV.SEED = {seed}",
        "if hasattr(CFG, 'SEED'):",
        f"    CFG.SEED = {seed}",
        "if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):",
        f"    CFG.TRAIN.SEED = {seed}",
        f"CFG.DATASET_INPUT_LEN = {INPUT_LEN}",
        f"CFG.DATASET_OUTPUT_LEN = {horizon}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")',
        "CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]",
        "CFG.MODEL.TARGET_FEATURES = [0]",
        f'CFG.MODEL.PARAM["input_len"] = {INPUT_LEN}',
        f'CFG.MODEL.PARAM["output_len"] = {horizon}',
        f"CFG.TEST.EVALUATION_HORIZONS = list(range(1, {horizon + 1}))",
    ]
    for key, val in spec.items():
        if val is None:
            continue
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


def verify_model_shapes(cfg_path: Path) -> tuple[bool, str]:
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from basicts.archs import ForecastStateFlow

        cfg = load_cfg(cfg_path)
        param = cfg.MODEL.PARAM
        model = ForecastStateFlow(**param)
        b, n = 2, param["node_size"]
        h, f_len = param["input_len"], param["output_len"]
        x = torch.randn(b, h, n, param["input_dim"])
        y = torch.randn(b, f_len, n, param["input_dim"])

        pred = model(x, return_all=False)
        if tuple(pred.shape) != (b, f_len, n, 1):
            return False, f"pred shape {tuple(pred.shape)}"

        out = model(x, future_data=y, return_all=True, train=True)
        r_list = param["chain_lengths"]
        for i, s in enumerate(out["states"]):
            if tuple(s.shape) != (b, f_len, n, 1):
                return False, f"state[{i}] shape {tuple(s.shape)}"
        for i, (ns, r) in enumerate(zip(out["native_states"], r_list)):
            if tuple(ns.shape) != (b, r, n, 1):
                return False, f"native_state[{i}] shape {tuple(ns.shape)}"
        fm_count = len(out.get("fm_items", []))
        expected_fm = max(len(r_list) - 1, 0)
        if fm_count != expected_fm:
            return False, f"fm_items={fm_count}, expected={expected_fm}"
        return True, f"pred={tuple(pred.shape)} states={len(out['states'])} fm={fm_count}"
    except Exception as e:
        return False, str(e)


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
    failed = [r for r in rows if r.get("failed", 1) == 1 or str(r.get("status", "")).startswith("exit_")]
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
        "num_flow_steps": spec.get("num_flow_steps"),
        "fm_loss_weight": spec.get("fm_loss_weight"),
        "native_loss_weight": spec.get("native_loss_weight"),
        "state_loss_weight": spec.get("state_loss_weight"),
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
    if not ckpt_dir_for(horizon, variant, seed).is_dir():
        return False
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
            summary.append(
                {
                    "horizon": horizon,
                    "variant": variant,
                    "metric": metric,
                    "n": len(vals),
                    "mean": m,
                    "std": s,
                    "ci95": c,
                    "failed": failed,
                }
            )
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
        "horizon",
        "variant",
        "seed",
        "input_len",
        "output_len",
        "chain_lengths",
        "num_flow_steps",
        "fm_loss_weight",
        "native_loss_weight",
        "state_loss_weight",
        "mae",
        "rmse",
        "mape",
        "params",
        "FLOPs",
        "inference_time",
        "failed",
        "status",
        "ckpt_dir",
        "cfg_path",
        "log_file",
        "error_tail_file",
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
        "# Forecast-State Flow Chain (FSF) Horizon Ablation on PeMS04\n\n",
        "Fixed input H=12; output F∈{12,24,48}. Chain: ¼F → ½F → F.\n\n",
        "## Per-run results\n\n",
        "| horizon | variant | seed | input | output | chain_lengths | flow_steps | "
        "fm_w | native_w | state_w | MAE | RMSE | MAPE | params | FLOPs | inference_time | failed |\n",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for r in rows:
        md.append(
            f"| {r['horizon']} | {r['variant']} | {r['seed']} | {r['input_len']} | {r['output_len']} | "
            f"{r['chain_lengths']} | {fmt_val(r.get('num_flow_steps'))} | "
            f"{fmt_val(r.get('fm_loss_weight'))} | {fmt_val(r.get('native_loss_weight'))} | "
            f"{fmt_val(r.get('state_loss_weight'))} | {fmt_val(r['mae'])} | {fmt_val(r['rmse'])} | "
            f"{fmt_val(r['mape'])} | {fmt_val(r.get('params'))} | {fmt_val(r.get('FLOPs'))} | "
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

    data_dir = ROOT / "datasets" / "PEMS04"
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
    shape_ok, shape_msg = verify_model_shapes(cfg_path)
    print(f"  [h={horizon} {variant} seed={seed}]")
    print(f"    cfg: {cfg_path}")
    print(f"    ckpt: {ckpt_dir_for(horizon, variant, seed)}")
    print(f"    input_len: {cfg.DATASET_INPUT_LEN} (model {cfg.MODEL.PARAM['input_len']})")
    print(f"    output_len: {cfg.DATASET_OUTPUT_LEN} (model {cfg.MODEL.PARAM['output_len']})")
    print(f"    chain_lengths: {spec.get('chain_lengths')}")
    print(f"    num_flow_steps: {spec.get('num_flow_steps')}")
    print(f"    fm_loss_weight: {spec.get('fm_loss_weight')}")
    print(f"    native_loss_weight: {spec.get('native_loss_weight')}")
    print(f"    state_loss_weight: {spec.get('state_loss_weight')}")
    print(f"    params: {count_params(cfg_path)}")
    print(f"    FLOPs: {estimate_flops(cfg_path)}")
    ok, msg = verify_batch_shape(horizon)
    print(f"    batch_check: {'OK' if ok else 'FAIL'} {msg}")
    print(f"    shape_check: {'OK' if shape_ok else 'FAIL'} {shape_msg}")
    print(f"    cmd: {cmd}")


def main() -> int:
    parser = argparse.ArgumentParser(description="FSF horizon ablation on PeMS04.")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANT_SPECS.keys()))
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--out", default="results/pems04_fsf_horizon.csv")
    parser.add_argument("--markdown", default="results/pems04_fsf_horizon.md")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip completed runs (default).")
    parser.add_argument("--force", action="store_true", help="Re-run even if completed.")
    parser.add_argument("--prepare_data", action="store_true")
    parser.add_argument("--summary-only", "--summary_only", action="store_true", dest="summary_only")
    parser.add_argument("--show_errors", action="store_true")
    args = parser.parse_args()

    skip_completed = not args.force
    if args.resume:
        skip_completed = True

    for h in args.horizons:
        if h not in HORIZON_CHAIN_LENGTHS and h not in {12, 24, 48}:
            print(f"Unsupported horizon: {h}")
            return 1

    if not FSF_BASE_CFG.is_file():
        print(f"Missing base config: {FSF_BASE_CFG}")
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
        all_ok = True
        for horizon, variant, cfg_path, seed in jobs:
            dry_run_info(horizon, variant, seed, cfg_path)
            ckpt_dirs.add(str(ckpt_dir_for(horizon, variant, seed)))
            cfg = load_cfg(cfg_path)
            if cfg.DATASET_INPUT_LEN != INPUT_LEN:
                all_ok = False
            expected_chain = chain_lengths_for(horizon, variant)
            if cfg.MODEL.PARAM.get("chain_lengths") != expected_chain:
                print(f"    ERROR: chain_lengths {cfg.MODEL.PARAM.get('chain_lengths')} != {expected_chain}")
                all_ok = False
            shape_ok, _ = verify_model_shapes(cfg_path)
            if not shape_ok:
                all_ok = False
        print(f"\n{len(jobs)} jobs (expected {expected_jobs}), unique ckpt dirs: {len(ckpt_dirs)}")
        chain_ckpt = ROOT / "checkpoints" / "fixed_input_horizon_pems04"
        fsf_ckpt = CKPT_ROOT
        print(f"FSF ckpt root: {fsf_ckpt} (separate from chain: {chain_ckpt})")
        for h in args.horizons:
            ok, msg = verify_batch_shape(h)
            print(f"F={h} batch shape: {'OK' if ok else 'FAIL'} — {msg}")
        return 0 if all_ok else 2

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
        if skip_completed and is_completed(horizon, variant, seed, cfg_path):
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
