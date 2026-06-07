#!/usr/bin/env python3
"""Run TF-STGN ablation experiments on PeMS04."""
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
RUN_PY = ROOT / "examples" / "run.py"
TEMP_CFG_DIR = ROOT / "tmp_configs" / "tfstgn_ablation"
LOG_DIR = ROOT / "results" / "tfstgn_ablation_logs"
CKPT_ROOT = ROOT / "checkpoints" / "tfstgn_ablation"

DEFAULT_VARIANTS = [
    "full",
    "wo_tf_guided_spatial",
    "wo_spectral_gate",
    "wo_temporal_gate",
    "wo_spatial_gate",
    "single_band",
    "wo_freq_loss",
]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]

VARIANT_CONFIG = {
    "full": ROOT / "examples" / "TFSTGN" / "TFSTGN_PEMS04_full.py",
    "wo_tf_guided_spatial": ROOT / "examples" / "TFSTGN" / "TFSTGN_PEMS04_wo_tf_guided_spatial.py",
    "wo_spectral_gate": ROOT / "examples" / "TFSTGN" / "TFSTGN_PEMS04_wo_spectral_gate.py",
    "wo_temporal_gate": ROOT / "examples" / "TFSTGN" / "TFSTGN_PEMS04_wo_temporal_gate.py",
    "wo_spatial_gate": ROOT / "examples" / "TFSTGN" / "TFSTGN_PEMS04_wo_spatial_gate.py",
    "single_band": ROOT / "examples" / "TFSTGN" / "TFSTGN_PEMS04_single_band.py",
    "wo_freq_loss": ROOT / "examples" / "TFSTGN" / "TFSTGN_PEMS04_wo_freq_loss.py",
}

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


def generate_temp_config(variant: str, seed: int) -> Path:
    base_cfg = VARIANT_CONFIG[variant]
    content = strip_hardcoded_cuda_devices(base_cfg.read_text(encoding="utf-8"))
    lines = [
        "",
        "# ===== tfstgn_ablation overrides (auto-generated) =====",
        f"CFG.ENV.SEED = {seed}",
        "if hasattr(CFG, 'SEED'):",
        f"    CFG.SEED = {seed}",
        "if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):",
        f"    CFG.TRAIN.SEED = {seed}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "tfstgn_ablation", "{variant}_seed{seed}")',
    ]
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
    return fallback if fallback["mae"] is not None else {k: None for k in METRIC_PATTERNS}


def collect_log_text(variant: str, seed: int, wrapper_log: Path | None) -> str:
    parts: list[str] = []
    if wrapper_log and wrapper_log.is_file():
        parts.append(wrapper_log.read_text(errors="replace"))
    ckpt_base = ckpt_dir_for(variant, seed)
    if ckpt_base.is_dir():
        for tlog in sorted(ckpt_base.glob("*/training_log_*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
            parts.append(tlog.read_text(errors="replace"))
            break
    return "\n".join(parts)


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
    cmd = [sys.executable, str(RUN_PY), "--cfg", cfg_for_easytorch(cfg_path), "--gpus", "0"]
    row = {
        "variant": variant,
        "seed": seed,
        "mae": None,
        "rmse": None,
        "mape": None,
        "ckpt_dir": str(ckpt_dir_for(variant, seed)),
        "log_file": str(log_file),
        "status": "running",
    }
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write(f"command: {' '.join(cmd)}\n")
            lf.write(f"CUDA_VISIBLE_DEVICES={gpu}\n\n")
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        metrics = parse_metrics(collect_log_text(variant, seed, log_file))
        row.update(metrics)
        row["status"] = "ok" if proc.returncode == 0 and metrics["mae"] is not None else f"exit_{proc.returncode}"
    except Exception as e:
        row["status"] = f"error:{e}"
    return row


def t_crit(n: int) -> float:
    return T_CRIT.get(n, 1.96)


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
        ok = [r for r in runs if r.get("mae") is not None]
        mae_vals = [float(r["mae"]) for r in ok]
        rmse_vals = [float(r["rmse"]) for r in ok if r.get("rmse") is not None]
        mape_vals = [float(r["mape"]) for r in ok if r.get("mape") is not None]
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
            "failed": len(runs) - len(ok),
        })
    summary.sort(key=lambda s: (math.inf if math.isnan(s["mae_mean"]) else s["mae_mean"]))
    return summary


def fmt_val(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_outputs(rows: list[dict], out_csv: Path, out_md: Path, summary_csv: Path):
    rows = sorted(rows, key=lambda r: (r["variant"], int(r["seed"])))
    fields = ["variant", "seed", "mae", "rmse", "mape", "ckpt_dir", "log_file", "status"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = build_summary(rows)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["variant"])
        w.writeheader()
        w.writerows(summary)

    md = ["# TF-STGN Ablation on PeMS04 12→12\n\n", "## Per-run results\n\n",
          "| variant | seed | MAE | RMSE | MAPE | status |\n", "|---|---:|---:|---:|---:|---|\n"]
    for r in rows:
        md.append(
            f"| {r['variant']} | {r['seed']} | {fmt_val(r['mae'])} | {fmt_val(r['rmse'])} | "
            f"{fmt_val(r['mape'])} | {r['status']} |\n"
        )
    md.append("\n## Summary\n\n")
    md.append("| variant | n | MAE mean | MAE std | MAE 95% CI | failed |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for s in summary:
        md.append(
            f"| {s['variant']} | {s['n']} | {fmt_val(s['mae_mean'])} | {fmt_val(s['mae_std'])} | "
            f"{fmt_val(s['mae_ci95'])} | {s['failed']} |\n"
        )
    out_md.write_text("".join(md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="TF-STGN ablation on PeMS04.")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANT_CONFIG.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--out", default="results/pems04_tfstgn_ablation.csv")
    parser.add_argument("--markdown", default="results/pems04_tfstgn_ablation.md")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary-only", "--summary_only", action="store_true", dest="summary_only")
    args = parser.parse_args()

    missing = [v for v in args.variants if not VARIANT_CONFIG[v].is_file()]
    if missing:
        print(f"Missing config files for variants: {missing}")
        return 1

    summary_csv = ROOT / "results" / "pems04_tfstgn_ablation_summary.csv"
    jobs = [(v, generate_temp_config(v, s), s) for v in args.variants for s in args.seeds]

    if args.dry_run:
        for variant, cfg_path, seed in jobs:
            print(f"[{variant} seed={seed}] cfg={cfg_path} ckpt={ckpt_dir_for(variant, seed)}")
            print(f"  cmd: CUDA_VISIBLE_DEVICES=<gpu> {sys.executable} {RUN_PY} --cfg {cfg_for_easytorch(cfg_path)} --gpus 0")
        print(f"\n{len(jobs)} jobs")
        return 0

    if args.summary_only:
        rows = []
        for variant, cfg_path, seed in jobs:
            log = LOG_DIR / f"{variant}_seed{seed}_gpu0.log"
            metrics = parse_metrics(collect_log_text(variant, seed, log if log.is_file() else None))
            rows.append({"variant": variant, "seed": seed, **metrics, "status": "ok" if metrics["mae"] else "failed", "ckpt_dir": str(ckpt_dir_for(variant, seed)), "log_file": str(log)})
        write_outputs(rows, ROOT / args.out, ROOT / args.markdown, summary_csv)
        return 0

    queue = GPUQueue(args.gpus)
    rows: list[dict] = []
    lock = threading.Lock()

    def worker(variant: str, cfg_path: Path, seed: int):
        gpu = queue.acquire()
        try:
            row = run_one(variant, cfg_path, gpu, seed)
            with lock:
                rows.append(row)
        finally:
            queue.release(gpu)

    threads = [threading.Thread(target=worker, args=(v, c, s)) for v, c, s in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    write_outputs(rows, ROOT / args.out, ROOT / args.markdown, summary_csv)
    print(f"Wrote {args.out}, {args.markdown}, {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
