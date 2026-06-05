#!/usr/bin/env python3
"""Run KASA-ST and PeMS04 baselines in one batch with GPU queue and summary."""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KASA_CFG = ROOT / "examples" / "KASAST_v2" / "KASAST_PEMS04.py"
BASELINE_DIR = ROOT / "examples" / "baselines"
RUN_PY = ROOT / "examples" / "run.py"
TEMP_CFG_DIR = ROOT / "tmp_configs" / "pems04_all_methods"
LOG_DIR = ROOT / "results" / "pems04_all_method_logs"
CKPT_ROOT = ROOT / "checkpoints" / "pems04_all_methods"

DEFAULT_METHODS = ["KASA", "STID", "D2STGNN", "AGCRN", "STGCN", "MTGNN", "StemGNN"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]

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


def config_path(method: str) -> Path | None:
    if method.upper() == "KASA":
        return KASA_CFG if KASA_CFG.is_file() else None
    p = BASELINE_DIR / method / f"{method}_PEMS04.py"
    return p if p.is_file() else None


def ckpt_dir_for(method: str, seed: int) -> Path:
    return CKPT_ROOT / f"{method}_seed{seed}"


def temp_cfg_path(method: str, seed: int) -> Path:
    TEMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_CFG_DIR / f"{method}_seed{seed}.py"


def generate_temp_config(base_cfg: Path, method: str, seed: int) -> Path:
    content = base_cfg.read_text(encoding="utf-8")
    ckpt_rel = f'os.path.join("checkpoints", "pems04_all_methods", "{method}_seed{seed}")'
    override_lines = [
        "",
        "# ===== pems04_all_methods overrides (auto-generated) =====",
        f"CFG.ENV.SEED = {seed}",
        "if hasattr(CFG, 'SEED'):",
        f"    CFG.SEED = {seed}",
        "if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):",
        f"    CFG.TRAIN.SEED = {seed}",
        f"CFG.TRAIN.CKPT_SAVE_DIR = {ckpt_rel}",
    ]
    if method.upper() == "KASA":
        override_lines += [
            'CFG.MODEL.PARAM["prior_mapper_type"] = "mlp"',
            'CFG.MODEL.PARAM["use_pre_temporal_spatial_enhancement"] = False',
            "CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]",
            "CFG.MODEL.TARGET_FEATURES = [0]",
        ]
    out = temp_cfg_path(method, seed)
    out.write_text(content + "\n".join(override_lines) + "\n", encoding="utf-8")
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


def collect_log_text(method: str, seed: int, wrapper_log: Path | None) -> str:
    parts: list[str] = []
    if wrapper_log and wrapper_log.is_file():
        parts.append(wrapper_log.read_text(errors="replace"))
    ckpt_base = ckpt_dir_for(method, seed)
    if ckpt_base.is_dir():
        for tlog in sorted(ckpt_base.glob("*/training_log_*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
            parts.append(tlog.read_text(errors="replace"))
            break
    return "\n".join(parts)


def save_error_tail(wrapper_log: Path, method: str, seed: int) -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tail_path = LOG_DIR / f"{method}_seed{seed}_error_tail.txt"
    if wrapper_log.is_file():
        lines = wrapper_log.read_text(errors="replace").splitlines()
        tail_path.write_text("\n".join(lines[-100:]) + "\n", encoding="utf-8")
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


def run_one(method: str, cfg_path: Path, gpu: str, seed: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{method}_seed{seed}_gpu{gpu}.log"
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    cfg_arg = cfg_for_easytorch(cfg_path)
    cmd = [sys.executable, str(RUN_PY), "--cfg", cfg_arg, "--gpus", str(gpu)]
    row = {
        "method": method,
        "seed": seed,
        "mae": None,
        "rmse": None,
        "mape": None,
        "ckpt_dir": str(ckpt_dir_for(method, seed)),
        "log_file": str(log_file),
        "status": "running",
        "error_tail_file": "",
    }
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        log_text = collect_log_text(method, seed, log_file)
        metrics = parse_metrics(log_text)
        row.update(metrics)
        if proc.returncode != 0:
            row["status"] = f"exit_{proc.returncode}"
            row["error_tail_file"] = save_error_tail(log_file, method, seed)
        elif metrics["mae"] is None:
            row["status"] = "failed_no_metrics"
        else:
            row["status"] = "ok"
    except Exception as e:
        row["status"] = f"error:{e}"
    return row


def summarize_row(method: str, seed: int) -> dict:
    wrapper_log = None
    if LOG_DIR.is_dir():
        matches = sorted(LOG_DIR.glob(f"{method}_seed{seed}_gpu*.log"))
        if matches:
            wrapper_log = matches[-1]
    error_tail = LOG_DIR / f"{method}_seed{seed}_error_tail.txt"
    log_text = collect_log_text(method, seed, wrapper_log) if wrapper_log else collect_log_text(method, seed, None)
    metrics = parse_metrics(log_text)
    status = "ok" if metrics["mae"] is not None else "failed_no_metrics"
    return {
        "method": method,
        "seed": seed,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "ckpt_dir": str(ckpt_dir_for(method, seed)),
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
        return 1.96  # normal approximation when scipy unavailable


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, float("nan")
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    ci = t_crit(n) * std / math.sqrt(n)
    return mean, std, ci


def build_summary(rows: list[dict]) -> list[dict]:
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    summary = []
    for method, runs in by_method.items():
        ok = [r for r in runs if r.get("mae") is not None and r.get("rmse") is not None and r.get("mape") is not None]
        failed = len(runs) - len(ok)
        mae_vals = [float(r["mae"]) for r in ok]
        rmse_vals = [float(r["rmse"]) for r in ok]
        mape_vals = [float(r["mape"]) for r in ok]
        m_mae, s_mae, c_mae = mean_std_ci(mae_vals)
        m_rmse, s_rmse, c_rmse = mean_std_ci(rmse_vals)
        m_mape, s_mape, c_mape = mean_std_ci(mape_vals)
        summary.append({
            "method": method,
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


def fmt_val(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_outputs(rows: list[dict], out_csv: Path, out_md: Path, summary_csv: Path):
    rows = sorted(rows, key=lambda r: (r["method"], int(r["seed"])))
    fields = ["method", "seed", "mae", "rmse", "mape", "ckpt_dir", "log_file", "status", "error_tail_file"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = build_summary(rows)
    sum_fields = [
        "method", "n", "mae_mean", "mae_std", "mae_ci95",
        "rmse_mean", "rmse_std", "rmse_ci95",
        "mape_mean", "mape_std", "mape_ci95", "failed",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary)

    md = ["# PeMS04 All Methods Results (12→12)\n\n", "## Per-run results\n\n",
          "| method | seed | MAE | RMSE | MAPE | status |\n", "|---|---:|---:|---:|---:|---|\n"]
    for r in rows:
        md.append(
            f"| {r['method']} | {r['seed']} | {fmt_val(r['mae'])} | {fmt_val(r['rmse'])} | "
            f"{fmt_val(r['mape'])} | {r['status']} |\n"
        )
    md.append("\n## Summary\n\n")
    md.append(
        "| method | n | MAE mean | MAE std | MAE 95% CI | RMSE mean | RMSE std | RMSE 95% CI | "
        "MAPE mean | MAPE std | MAPE 95% CI | failed |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in summary:
        md.append(
            f"| {s['method']} | {s['n']} | {fmt_val(s['mae_mean'])} | {fmt_val(s['mae_std'])} | {fmt_val(s['mae_ci95'])} | "
            f"{fmt_val(s['rmse_mean'])} | {fmt_val(s['rmse_std'])} | {fmt_val(s['rmse_ci95'])} | "
            f"{fmt_val(s['mape_mean'])} | {fmt_val(s['mape_std'])} | {fmt_val(s['mape_ci95'])} | {s['failed']} |\n"
        )
    out_md.write_text("".join(md), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run KASA-ST and PeMS04 baselines in one batch.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        help=f"Methods to run (default: {' '.join(DEFAULT_METHODS)})")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Seed list (default: 1 2 3 4 5)")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--out", default="results/pems04_all_methods.csv")
    parser.add_argument("--markdown", default="results/pems04_all_methods.md")
    parser.add_argument("--dry_run", action="store_true", help="Print configs/commands only")
    parser.add_argument("--summary-only", "--summary_only", action="store_true", dest="summary_only")
    args = parser.parse_args()

    summary_csv = ROOT / "results" / "pems04_all_methods_summary.csv"
    ready, skipped = [], []
    for method in args.methods:
        p = config_path(method)
        if p is None:
            skipped.append(method)
        else:
            ready.append((method, p))

    if skipped:
        print("Skipped (no config):", ", ".join(skipped))
    if not ready:
        print("No methods to run.")
        return 1

    jobs: list[tuple[str, Path, int, str]] = []
    for method, base_cfg in ready:
        for seed in args.seeds:
            tmp = generate_temp_config(base_cfg, method, seed)
            cmd = f"{sys.executable} {RUN_PY} --cfg {cfg_for_easytorch(tmp)} --gpus <GPU>"
            jobs.append((method, tmp, seed, cmd))

    if args.dry_run:
        print("Dry run — temp configs and commands:\n")
        for method, tmp, seed, cmd in jobs:
            print(f"  [{method} seed={seed}]")
            print(f"    cfg: {tmp}")
            print(f"    ckpt: {ckpt_dir_for(method, seed)}")
            print(f"    cmd: {cmd}")
        print(f"\n{len(jobs)} jobs, GPUs: {args.gpus}")
        return 0

    if args.summary_only:
        rows = [summarize_row(method, seed) for method, _ in ready for seed in args.seeds]
        write_outputs(rows, ROOT / args.out, ROOT / args.markdown, summary_csv)
        print(f"Wrote {args.out}, {args.markdown}, {summary_csv}")
        return 0

    queue = GPUQueue(args.gpus)
    rows: list[dict] = []
    lock = threading.Lock()

    def worker(method: str, cfg_path: Path, seed: int):
        gpu = queue.acquire()
        try:
            print(f"[start] {method} seed={seed} gpu={gpu}")
            row = run_one(method, cfg_path, gpu, seed)
            with lock:
                rows.append(row)
            print(f"[done]  {method} seed={seed} status={row['status']} mae={row['mae']}")
        finally:
            queue.release(gpu)

    threads = []
    for method, base_cfg in ready:
        for seed in args.seeds:
            tmp = generate_temp_config(base_cfg, method, seed)
            t = threading.Thread(target=worker, args=(method, tmp, seed))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()

    write_outputs(rows, ROOT / args.out, ROOT / args.markdown, summary_csv)
    print(f"Wrote {args.out}, {args.markdown}, {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
