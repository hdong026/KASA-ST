#!/usr/bin/env python3
"""Run PeMS04 baselines under Protocol A: fixed H=12, variable F∈{12,24,48}."""
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
KASA_CFG = ROOT / "examples" / "KASAST_v2" / "KASAST_PEMS04.py"
BASELINE_DIR = ROOT / "examples" / "baselines"
RUN_PY = ROOT / "examples" / "run.py"
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_pems04_fixed_input_horizons.py"
TEMP_CFG_DIR = ROOT / "tmp_configs" / "all_baselines_horizon_pems04"
LOG_DIR = ROOT / "results" / "all_baselines_horizon_pems04_logs"
CKPT_ROOT = ROOT / "checkpoints" / "all_baselines_horizon_pems04"
FSC_CSV = ROOT / "results" / "pems04_fixed_input_horizon.csv"

INPUT_LEN = 12
DEFAULT_HORIZONS = [12, 24, 48]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
REQUESTED_BASELINES = [
    "kasa_baseline",
    "STID",
    "GWNet",
    "MTGNN",
    "AGCRN",
    "D2STGNN",
    "STAEformer",
    "DLinear",
    "TimeMixer",
]
FSC_VARIANTS = ["chain_final_spatial", "chain_no_spatial"]
HORIZON_EVAL_STEPS = {12: [3, 6, 12], 24: [6, 12, 24], 48: [12, 24, 48]}

KASA_BEST_SETTINGS = {
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


def _py_literal(v) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, list):
        return repr(v)
    return str(v)


def baseline_cfg_path(name: str) -> Path | None:
    if name == "kasa_baseline":
        return KASA_CFG if KASA_CFG.is_file() else None
    p = BASELINE_DIR / name / f"{name}_PEMS04.py"
    return p if p.is_file() else None


def model_override_lines(baseline: str, horizon: int) -> list[str]:
    lines: list[str] = []
    if baseline == "kasa_baseline":
        lines += [
            f'CFG.MODEL.PARAM["input_len"] = {INPUT_LEN}',
            f'CFG.MODEL.PARAM["output_len"] = {horizon}',
            "CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]",
            "CFG.MODEL.TARGET_FEATURES = [0]",
        ]
        for key, val in KASA_BEST_SETTINGS.items():
            lines.append(f'CFG.MODEL.PARAM["{key}"] = {_py_literal(val)}')
    elif baseline == "STID":
        lines += [
            f'CFG.MODEL.PARAM["input_len"] = {INPUT_LEN}',
            f'CFG.MODEL.PARAM["output_len"] = {horizon}',
        ]
    elif baseline == "GWNet":
        lines.append(f'CFG.MODEL.PARAM["out_dim"] = {horizon}')
    elif baseline == "MTGNN":
        lines += [
            f'CFG.MODEL.PARAM["out_dim"] = {horizon}',
            f'CFG.MODEL.PARAM["seq_length"] = {INPUT_LEN}',
            f"CFG.TRAIN.CL.PREDICTION_LENGTH = {horizon}",
        ]
    elif baseline == "AGCRN":
        lines.append(f'CFG.MODEL.PARAM["horizon"] = {horizon}')
    elif baseline == "D2STGNN":
        lines += [
            f'CFG.MODEL.PARAM["input_seq_len"] = {INPUT_LEN}',
            f'CFG.MODEL.PARAM["output_seq_len"] = {horizon}',
            f'CFG.MODEL.PARAM["seq_length"] = {horizon}',
            f"CFG.TRAIN.CL.PREDICTION_LENGTH = {horizon}",
        ]
    elif baseline == "STAEformer":
        lines += [
            f'CFG.MODEL.PARAM["in_steps"] = {INPUT_LEN}',
            f'CFG.MODEL.PARAM["out_steps"] = {horizon}',
        ]
    return lines


def discover_baselines(requested: list[str]) -> tuple[list[str], list[str]]:
    available, missing = [], []
    for name in requested:
        if baseline_cfg_path(name) is not None:
            available.append(name)
        else:
            missing.append(name)
    return available, missing


def strip_hardcoded_cuda_devices(content: str) -> str:
    kept: list[str] = []
    for line in content.splitlines():
        if "CUDA_VISIBLE_DEVICES" in line and "os.environ" in line:
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def temp_cfg_path(baseline: str, horizon: int, seed: int) -> Path:
    TEMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_CFG_DIR / f"h{horizon}_{baseline}_seed{seed}.py"


def ckpt_dir_for(baseline: str, horizon: int, seed: int) -> Path:
    return CKPT_ROOT / f"h{horizon}" / f"{baseline}_seed{seed}"


def data_paths(horizon: int) -> tuple[Path, Path]:
    data_dir = ROOT / "datasets" / "PEMS04"
    stem = f"in{INPUT_LEN}_out{horizon}"
    return data_dir / f"data_{stem}.pkl", data_dir / f"index_{stem}.pkl"


def generate_temp_config(baseline: str, horizon: int, seed: int) -> Path:
    base = baseline_cfg_path(baseline)
    if base is None:
        raise FileNotFoundError(f"No config for baseline: {baseline}")
    content = strip_hardcoded_cuda_devices(base.read_text(encoding="utf-8"))
    ckpt_rel = f"checkpoints/all_baselines_horizon_pems04/h{horizon}/{baseline}_seed{seed}"
    lines = [
        "",
        "# ===== all_baselines_horizon_pems04 overrides (auto-generated) =====",
        f"CFG.ENV.SEED = {seed}",
        "if hasattr(CFG, 'SEED'):",
        f"    CFG.SEED = {seed}",
        "if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):",
        f"    CFG.TRAIN.SEED = {seed}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")',
        f"CFG.DATASET_INPUT_LEN = {INPUT_LEN}",
        f"CFG.DATASET_OUTPUT_LEN = {horizon}",
        f"CFG.TEST.EVALUATION_HORIZONS = list(range(1, {horizon + 1}))",
    ]
    lines.extend(model_override_lines(baseline, horizon))
    out = temp_cfg_path(baseline, horizon, seed)
    out.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")
    return out


def load_cfg(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("baseline_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CFG


def cfg_for_easytorch(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def find_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.is_dir():
        return None
    matches = sorted(ckpt_dir.rglob("*best_val_MAE.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


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
        b, n = 1, cfg.MODEL.PARAM.get("node_size") or cfg.MODEL.PARAM.get("num_nodes", 307)
        c = cfg.MODEL.PARAM.get("input_dim") or cfg.MODEL.PARAM.get("in_dim") or 1
        if isinstance(c, int) and c <= 4:
            pass
        else:
            c = 3
        x = torch.randn(b, INPUT_LEN, n, max(c, 1))
        try:
            from thop import profile

            macs, _ = profile(
                model,
                inputs=(x,),
                verbose=False,
            )
            return float(macs) * 2
        except Exception:
            with torch.no_grad():
                try:
                    model(history_data=x, future_data=x, train=False)
                except TypeError:
                    model(x, None, 0, 0, False)
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


def parse_horizon_mae(log_text: str, steps: list[int]) -> dict[int, float | None]:
    found: dict[int, float] = {}
    for m in HORIZON_MAE_PATTERN.finditer(log_text):
        found[int(m.group(1))] = float(m.group(2))
    return {h: found.get(h) for h in steps}


def wrapper_log_path(baseline: str, horizon: int, seed: int) -> Path | None:
    matches = sorted(LOG_DIR.glob(f"h{horizon}_{baseline}_seed{seed}_gpu*.log"))
    return matches[-1] if matches else None


def collect_log_text(baseline: str, horizon: int, seed: int, wrapper_log: Path | None) -> str:
    parts: list[str] = []
    if wrapper_log and wrapper_log.is_file():
        parts.append(wrapper_log.read_text(errors="replace"))
    ckpt_base = ckpt_dir_for(baseline, horizon, seed)
    if ckpt_base.is_dir():
        for tlog in sorted(ckpt_base.glob("*/training_log_*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
            parts.append(tlog.read_text(errors="replace"))
    return "\n".join(parts)


def verify_batch_shape(horizon: int) -> tuple[bool, str]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from basicts.data import TimeSeriesForecastingDataset
    from torch.utils.data import DataLoader

    data_file, index_file = data_paths(horizon)
    if not data_file.is_file() or not index_file.is_file():
        return False, f"missing {data_file.name} / {index_file.name}"
    ds = TimeSeriesForecastingDataset(
        data_file_path=str(data_file),
        index_file_path=str(index_file),
        mode="test",
    )
    future_data, history_data = next(iter(DataLoader(ds, batch_size=32, shuffle=False)))
    if history_data.shape[1] != INPUT_LEN:
        return False, f"history len {history_data.shape[1]} != {INPUT_LEN}"
    if future_data.shape[1] != horizon:
        return False, f"future len {future_data.shape[1]} != {horizon}"
    return True, f"history={tuple(history_data.shape)} future={tuple(future_data.shape)}"


def verify_cfg_output_len(cfg_path: Path, horizon: int) -> tuple[bool, str]:
    cfg = load_cfg(cfg_path)
    if cfg.DATASET_INPUT_LEN != INPUT_LEN:
        return False, f"DATASET_INPUT_LEN={cfg.DATASET_INPUT_LEN}"
    if cfg.DATASET_OUTPUT_LEN != horizon:
        return False, f"DATASET_OUTPUT_LEN={cfg.DATASET_OUTPUT_LEN}"
    return True, "ok"


def base_row(baseline: str, horizon: int, seed: int, cfg_path: Path) -> dict:
    ckpt = find_checkpoint(ckpt_dir_for(baseline, horizon, seed))
    return {
        "baseline": baseline,
        "horizon": horizon,
        "seed": seed,
        "input_len": INPUT_LEN,
        "output_len": horizon,
        "mae": None,
        "rmse": None,
        "mape": None,
        "params": count_params(cfg_path),
        "FLOPs": estimate_flops(cfg_path),
        "inference_time": None,
        "failed": 1,
        "status": "pending",
        "ckpt_dir": str(ckpt_dir_for(baseline, horizon, seed)),
        "checkpoint": str(ckpt) if ckpt else "",
        "cfg_path": str(cfg_path),
        "data_path": str(data_paths(horizon)[0]),
        "index_path": str(data_paths(horizon)[1]),
        "log_file": "",
        "error_tail_file": "",
    }


def summarize_row(baseline: str, horizon: int, seed: int, cfg_path: Path) -> dict:
    row = base_row(baseline, horizon, seed, cfg_path)
    wrapper_log = wrapper_log_path(baseline, horizon, seed)
    log_text = collect_log_text(baseline, horizon, seed, wrapper_log)
    metrics = parse_metrics(log_text)
    row.update(metrics)
    for step, val in parse_horizon_mae(log_text, HORIZON_EVAL_STEPS[horizon]).items():
        row[f"horizon_mae_{step}"] = val
    ckpt = find_checkpoint(ckpt_dir_for(baseline, horizon, seed))
    row["checkpoint"] = str(ckpt) if ckpt else ""
    row["log_file"] = str(wrapper_log) if wrapper_log else ""
    error_tail = LOG_DIR / f"h{horizon}_{baseline}_seed{seed}_error_tail.txt"
    row["error_tail_file"] = str(error_tail) if error_tail.is_file() else ""

    if metrics["mae"] is not None and ckpt is not None:
        row["status"] = "ok"
        row["failed"] = 0
    elif metrics["mae"] is not None and ckpt is None:
        row["status"] = "missing_checkpoint"
        row["failed"] = 1
    else:
        row["status"] = "failed_no_metrics"
        row["failed"] = 1
    return row


def is_completed(baseline: str, horizon: int, seed: int, cfg_path: Path) -> bool:
    row = summarize_row(baseline, horizon, seed, cfg_path)
    return row.get("status") == "ok"


def save_error_tail(wrapper_log: Path, baseline: str, horizon: int, seed: int) -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tail_path = LOG_DIR / f"h{horizon}_{baseline}_seed{seed}_error_tail.txt"
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


def run_one(baseline: str, horizon: int, cfg_path: Path, gpu: str, seed: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"h{horizon}_{baseline}_seed{seed}_gpu{gpu}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, str(RUN_PY), "--cfg", cfg_for_easytorch(cfg_path), "--gpus", str(gpu)]
    row = base_row(baseline, horizon, seed, cfg_path)
    row["log_file"] = str(log_file)
    row["status"] = "running"
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        log_text = collect_log_text(baseline, horizon, seed, log_file)
        metrics = parse_metrics(log_text)
        row.update(metrics)
        for step, val in parse_horizon_mae(log_text, HORIZON_EVAL_STEPS[horizon]).items():
            row[f"horizon_mae_{step}"] = val
        ckpt = find_checkpoint(ckpt_dir_for(baseline, horizon, seed))
        row["checkpoint"] = str(ckpt) if ckpt else ""
        if proc.returncode != 0:
            row["status"] = f"exit_{proc.returncode}"
            row["failed"] = 1
            row["error_tail_file"] = save_error_tail(log_file, baseline, horizon, seed)
        elif metrics["mae"] is None:
            row["status"] = "failed_no_metrics"
            row["failed"] = 1
            row["error_tail_file"] = save_error_tail(log_file, baseline, horizon, seed)
        elif ckpt is None:
            row["status"] = "missing_checkpoint"
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


def load_fsc_reference() -> dict[tuple[int, str, int], float]:
    ref: dict[tuple[int, str, int], float] = {}
    if not FSC_CSV.is_file():
        return ref
    with open(FSC_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variant = row.get("variant", "")
            if variant not in FSC_VARIANTS:
                continue
            mae = row.get("mae")
            if mae in (None, ""):
                continue
            ref[(int(row["horizon"]), variant, int(row["seed"]))] = float(mae)
    return ref


def build_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((int(r["horizon"]), r["baseline"]), []).append(r)
    summary: list[dict] = []
    for (horizon, baseline), runs in sorted(groups.items()):
        ok = [r for r in runs if r.get("failed", 1) == 0 and r.get("mae") is not None]
        failed = len(runs) - len(ok)
        for metric in ("mae", "rmse", "mape", "inference_time"):
            vals = [float(r[metric]) for r in ok if r.get(metric) is not None]
            m, s, c = mean_std_ci(vals)
            summary.append({
                "horizon": horizon,
                "baseline": baseline,
                "metric": metric,
                "n": len(vals),
                "mean": m,
                "std": s,
                "ci95": c,
                "failed": failed,
            })
        pvals = [float(r["params"]) for r in ok if r.get("params") is not None]
        if pvals:
            m, s, c = mean_std_ci(pvals)
            summary.append({
                "horizon": horizon,
                "baseline": baseline,
                "metric": "params",
                "n": len(pvals),
                "mean": m,
                "std": s,
                "ci95": c,
                "failed": failed,
            })
        fvals = [float(r["FLOPs"]) for r in ok if r.get("FLOPs") is not None]
        if fvals:
            m, s, c = mean_std_ci(fvals)
            summary.append({
                "horizon": horizon,
                "baseline": baseline,
                "metric": "FLOPs",
                "n": len(fvals),
                "mean": m,
                "std": s,
                "ci95": c,
                "failed": failed,
            })
    return summary


def build_fsc_comparison(rows: list[dict], fsc_ref: dict[tuple[int, str, int], float]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        if r.get("mae") is None:
            continue
        base_mae = float(r["mae"])
        for variant in FSC_VARIANTS:
            key = (int(r["horizon"]), variant, int(r["seed"]))
            if key not in fsc_ref:
                continue
            fsc_mae = fsc_ref[key]
            diff = base_mae - fsc_mae
            rel = (diff / base_mae * 100.0) if base_mae else float("nan")
            out.append({
                "horizon": r["horizon"],
                "baseline": r["baseline"],
                "seed": r["seed"],
                "reference": variant,
                "baseline_mae": base_mae,
                "reference_mae": fsc_mae,
                "mae_diff": diff,
                "relative_improvement_pct": rel,
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


def write_outputs(
    rows: list[dict],
    out_csv: Path,
    out_md: Path,
    summary_csv: Path,
    failures_log: Path,
    fsc_ref: dict[tuple[int, str, int], float],
) -> None:
    rows = sorted(rows, key=lambda r: (int(r["horizon"]), r["baseline"], int(r["seed"])))
    eval_cols = sorted({k for r in rows for k in r if k.startswith("horizon_mae_")})
    fields = [
        "baseline", "horizon", "seed", "input_len", "output_len",
        "mae", "rmse", "mape", "params", "FLOPs", "inference_time", "failed",
        "status", "ckpt_dir", "checkpoint", "cfg_path", "data_path", "index_path",
        "log_file", "error_tail_file",
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
            fieldnames=["horizon", "baseline", "metric", "n", "mean", "std", "ci95", "failed"],
        )
        w.writeheader()
        w.writerows(summary)

    comparison = build_fsc_comparison(rows, fsc_ref)
    comp_csv = out_csv.with_name(out_csv.stem + "_vs_fsc.csv")
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "horizon", "baseline", "seed", "reference",
                "baseline_mae", "reference_mae", "mae_diff", "relative_improvement_pct",
            ],
        )
        w.writeheader()
        w.writerows(comparison)

    failed_rows = [r for r in rows if r.get("failed", 1) == 1]
    failures_log.parent.mkdir(parents=True, exist_ok=True)
    with open(failures_log, "w", encoding="utf-8") as f:
        for r in failed_rows:
            f.write(
                f"h={r['horizon']} {r['baseline']} seed={r['seed']} status={r['status']} "
                f"log={r.get('log_file', '')}\n"
            )

    md = [
        "# PeMS04 All Baselines — Protocol A (H=12, F∈{12,24,48})\n\n",
        "Same data/split/scaler/mask/metrics as FSC horizon experiments.\n\n",
        "## Per-run results\n\n",
        "| horizon | baseline | seed | input | output | MAE | RMSE | MAPE | params | FLOPs | "
        "inference_time | failed | status |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for r in rows:
        md.append(
            f"| {r['horizon']} | {r['baseline']} | {r['seed']} | {r['input_len']} | {r['output_len']} | "
            f"{fmt_val(r['mae'])} | {fmt_val(r['rmse'])} | {fmt_val(r['mape'])} | "
            f"{fmt_val(r.get('params'))} | {fmt_val(r.get('FLOPs'))} | "
            f"{fmt_val(r.get('inference_time'))} | {r.get('failed', 1)} | {r.get('status', '')} |\n"
        )

    md.append("\n## Summary (mean ± std, 95% CI)\n\n")
    md.append("| horizon | baseline | metric | n | mean | std | 95% CI | failed |\n")
    md.append("|---|---|---|---:|---:|---:|---:|---:|\n")
    for s in summary:
        md.append(
            f"| {s['horizon']} | {s['baseline']} | {s['metric']} | {s['n']} | "
            f"{fmt_val(s['mean'])} | {fmt_val(s['std'])} | {fmt_val(s['ci95'])} | {s['failed']} |\n"
        )

    md.append("\n## vs FSC (baseline MAE − reference MAE; positive ⇒ reference better)\n\n")
    md.append("| horizon | baseline | seed | reference | baseline MAE | reference MAE | diff | rel % |\n")
    md.append("|---:|---|---:|---|---:|---:|---:|---:|\n")
    for c in comparison:
        md.append(
            f"| {c['horizon']} | {c['baseline']} | {c['seed']} | {c['reference']} | "
            f"{fmt_val(c['baseline_mae'])} | {fmt_val(c['reference_mae'])} | "
            f"{fmt_val(c['mae_diff'])} | {fmt_val(c['relative_improvement_pct'])} |\n"
        )

    md.append("\n## Horizon-wise MAE (key steps)\n\n")
    for horizon in sorted({int(r["horizon"]) for r in rows}):
        steps = HORIZON_EVAL_STEPS[horizon]
        md.append(f"### F={horizon}\n\n")
        md.append("| baseline | seed | " + " | ".join(f"h={s}" for s in steps) + " |\n")
        md.append("|---|---|" + "|".join(["---:"] * len(steps)) + "|\n")
        for r in rows:
            if int(r["horizon"]) != horizon:
                continue
            vals = [fmt_val(r.get(f"horizon_mae_{s}")) for s in steps]
            md.append(f"| {r['baseline']} | {r['seed']} | " + " | ".join(vals) + " |\n")
        md.append("\n")

    out_md.write_text("".join(md), encoding="utf-8")


def print_log_tail(baseline: str, horizon: int, seed: int, lines: int = SHOW_ERROR_LINES) -> None:
    log_path = wrapper_log_path(baseline, horizon, seed)
    if not log_path or not log_path.is_file():
        return
    tail = log_path.read_text(errors="replace").splitlines()[-lines:]
    print(f"\n--- error tail: h={horizon} {baseline} seed={seed} ---")
    for line in tail:
        print(line)
    print("--- end ---\n")


def show_failed_errors(rows: list[dict]) -> None:
    failed = [r for r in rows if r.get("failed", 1) == 1]
    if not failed:
        return
    print("\nFailed tasks:\n")
    for r in sorted(failed, key=lambda x: (int(x["horizon"]), x["baseline"], int(x["seed"]))):
        print(f"  h={r['horizon']} {r['baseline']} seed={r['seed']} status={r['status']}")
        print_log_tail(r["baseline"], int(r["horizon"]), int(r["seed"]))


def ensure_data(horizons: list[int]) -> int:
    cmd = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--input-len",
        str(INPUT_LEN),
        "--horizons",
        *[str(h) for h in horizons],
    ]
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def print_inventory(available: list[str], missing: list[str], invalid_horizons: list[int]) -> None:
    print("available baselines:", ", ".join(available) if available else "(none)")
    print("missing baselines:", ", ".join(missing) if missing else "(none)")
    print("unsupported horizons:", ", ".join(map(str, invalid_horizons)) if invalid_horizons else "(none)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all PeMS04 baselines under Protocol A.")
    parser.add_argument("--baselines", nargs="+", default=REQUESTED_BASELINES)
    parser.add_argument("--all_available", action="store_true", help="Run every baseline with a ready config.")
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--out", default="results/pems04_all_baselines_fixed_input.csv")
    parser.add_argument("--markdown", default="results/pems04_all_baselines_fixed_input.md")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip completed tasks with valid checkpoint+metrics.")
    parser.add_argument("--force", action="store_true", help="Re-run even if already completed.")
    parser.add_argument("--prepare_data", action="store_true")
    parser.add_argument("--show_errors", action="store_true")
    args = parser.parse_args()

    unsupported_horizons = [h for h in args.horizons if h not in HORIZON_EVAL_STEPS]
    requested = REQUESTED_BASELINES if args.all_available else args.baselines
    available, missing = discover_baselines(requested)

    print_inventory(available, missing, unsupported_horizons)
    if unsupported_horizons:
        return 1
    if not available:
        print("ERROR: no runnable baselines.")
        return 1

    if args.prepare_data or args.dry_run:
        rc = ensure_data(args.horizons)
        if rc != 0:
            return rc

    out_csv = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_md = ROOT / args.markdown if not Path(args.markdown).is_absolute() else Path(args.markdown)
    summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
    failures_log = out_csv.with_name(out_csv.stem + "_failures.log")

    jobs = [
        (b, h, generate_temp_config(b, h, s), s)
        for b in available
        for h in args.horizons
        for s in args.seeds
    ]
    expected = len(jobs)

    if args.dry_run:
        print(f"\nDry run — physical GPUs: {args.gpus}")
        print(f"Jobs: {len(jobs)} (expected {expected})\n")
        ckpt_dirs = set()
        for baseline, horizon, cfg_path, seed in jobs:
            cfg_ok, cfg_msg = verify_cfg_output_len(cfg_path, horizon)
            batch_ok, batch_msg = verify_batch_shape(horizon)
            ckpt = ckpt_dir_for(baseline, horizon, seed)
            ckpt_dirs.add(str(ckpt))
            cmd = (
                f"CUDA_VISIBLE_DEVICES=<gpu> {sys.executable} {RUN_PY} "
                f"--cfg {cfg_for_easytorch(cfg_path)} --gpus <gpu>"
            )
            print(f"[h={horizon} {baseline} seed={seed}]")
            print(f"  cfg      : {cfg_path}")
            print(f"  ckpt     : {ckpt}")
            print(f"  data     : {data_paths(horizon)[0]}")
            print(f"  index    : {data_paths(horizon)[1]}")
            print(f"  cfg_check: {'OK' if cfg_ok else 'FAIL'} ({cfg_msg})")
            print(f"  batch    : {'OK' if batch_ok else 'FAIL'} ({batch_msg})")
            print(f"  params   : {count_params(cfg_path)}")
            print(f"  cmd      : {cmd}")
        print(f"\nunique checkpoint dirs: {len(ckpt_dirs)}")
        for h in args.horizons:
            ok, msg = verify_batch_shape(h)
            print(f"F={h} data check: {'OK' if ok else 'FAIL'} — {msg}")
        return 0

    fsc_ref = load_fsc_reference()
    queue = GPUQueue(args.gpus)
    rows: list[dict] = []
    lock = threading.Lock()

    def worker(baseline: str, horizon: int, cfg_path: Path, seed: int):
        if args.resume and not args.force and is_completed(baseline, horizon, seed, cfg_path):
            row = summarize_row(baseline, horizon, seed, cfg_path)
            with lock:
                rows.append(row)
            print(f"[skip] h={horizon} {baseline} seed={seed} mae={row['mae']}")
            return
        gpu = queue.acquire()
        try:
            print(f"[start] h={horizon} {baseline} seed={seed} gpu={gpu}")
            row = run_one(baseline, horizon, cfg_path, gpu, seed)
            with lock:
                rows.append(row)
            print(f"[done]  h={horizon} {baseline} seed={seed} status={row['status']} mae={row['mae']}")
        finally:
            queue.release(gpu)

    threads = []
    for baseline, horizon, cfg_path, seed in jobs:
        t = threading.Thread(target=worker, args=(baseline, horizon, cfg_path, seed))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    write_outputs(rows, out_csv, out_md, summary_csv, failures_log, fsc_ref)
    print(f"Wrote {out_csv}, {out_md}, {summary_csv}, {failures_log} ({len(rows)}/{expected})")
    if args.show_errors:
        show_failed_errors(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
