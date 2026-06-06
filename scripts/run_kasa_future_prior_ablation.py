#!/usr/bin/env python3
"""Run KASA history-vs-future output prior-residual ablation on PeMS04."""
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
BASE_CFG = ROOT / "examples" / "KASAST_v2" / "KASAST_PEMS04.py"
RUN_PY = ROOT / "examples" / "run.py"
TEMP_CFG_DIR = ROOT / "tmp_configs" / "kasa_future_prior_ablation"
LOG_DIR = ROOT / "results" / "kasa_future_prior_ablation_logs"
CKPT_ROOT = ROOT / "checkpoints" / "kasa_future_prior_ablation"

DEFAULT_VARIANTS = ["woprior", "history_prior", "future_prior"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
FORWARD_FEATURES = [0, 1, 2, 3]
TARGET_FEATURES = [0]

REPORT_KEYS = [
    "prior_mapper_type",
    "use_pre_temporal_spatial_enhancement",
    "keep_output_prior_residual",
    "prior_source",
    "FORWARD_FEATURES",
    "TARGET_FEATURES",
]

VARIANT_OVERRIDES: dict[str, dict] = {
    "woprior": {
        "prior_mapper_type": "mlp",
        "use_pre_temporal_spatial_enhancement": False,
        "keep_output_prior_residual": False,
        "prior_source": "history",
    },
    "history_prior": {
        "prior_mapper_type": "mlp",
        "use_pre_temporal_spatial_enhancement": False,
        "keep_output_prior_residual": True,
        "prior_source": "history",
    },
    "future_prior": {
        "prior_mapper_type": "mlp",
        "use_pre_temporal_spatial_enhancement": False,
        "keep_output_prior_residual": True,
        "prior_source": "future",
    },
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

PAIRED_SECTIONS = [
    ("history_prior", "woprior", "history_prior - woprior", "history prior is worse than no prior"),
    ("future_prior", "woprior", "future_prior - woprior", "future prior is worse than no prior"),
    ("future_prior", "history_prior", "future_prior - history_prior", "future prior is worse than history prior"),
]


def ckpt_dir_for(variant: str, seed: int) -> Path:
    return CKPT_ROOT / f"{variant}_seed{seed}"


def temp_cfg_path(variant: str, seed: int) -> Path:
    TEMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_CFG_DIR / f"{variant}_seed{seed}.py"


def _py_literal(v) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return repr(v)
    return str(v)


def strip_hardcoded_cuda_devices(content: str) -> str:
    """Remove KASAST_PEMS04.py hardcoded CUDA_VISIBLE_DEVICES so --gpus can work."""
    kept: list[str] = []
    for line in content.splitlines():
        if 'CUDA_VISIBLE_DEVICES' in line and 'os.environ' in line:
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def variant_report(variant: str) -> dict:
    overrides = VARIANT_OVERRIDES[variant]
    return {
        "prior_mapper_type": overrides.get("prior_mapper_type", "mlp"),
        "use_pre_temporal_spatial_enhancement": overrides.get(
            "use_pre_temporal_spatial_enhancement", False
        ),
        "keep_output_prior_residual": overrides.get("keep_output_prior_residual", True),
        "prior_source": overrides.get("prior_source", "history"),
        "FORWARD_FEATURES": FORWARD_FEATURES,
        "TARGET_FEATURES": TARGET_FEATURES,
    }


def generate_temp_config(variant: str, seed: int) -> Path:
    if variant not in VARIANT_OVERRIDES:
        raise ValueError(f"Unknown variant: {variant}")
    content = strip_hardcoded_cuda_devices(BASE_CFG.read_text(encoding="utf-8"))
    overrides = VARIANT_OVERRIDES[variant]
    lines = [
        "",
        "# ===== kasa_future_prior_ablation overrides (auto-generated) =====",
        f"CFG.ENV.SEED = {seed}",
        "if hasattr(CFG, 'SEED'):",
        f"    CFG.SEED = {seed}",
        "if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):",
        f"    CFG.TRAIN.SEED = {seed}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "kasa_future_prior_ablation", "{variant}_seed{seed}")',
        f"CFG.MODEL.FORWARD_FEATURES = {FORWARD_FEATURES}",
        f"CFG.MODEL.TARGET_FEATURES = {TARGET_FEATURES}",
    ]
    for key, val in overrides.items():
        lines.append(f'CFG.MODEL.PARAM["{key}"] = {_py_literal(val)}')
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
    parts: list[str] = []
    if wrapper_log and wrapper_log.is_file():
        parts.append(wrapper_log.read_text(errors="replace"))
    ckpt_base = ckpt_dir_for(variant, seed)
    if ckpt_base.is_dir():
        for tlog in sorted(ckpt_base.glob("*/training_log_*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
            parts.append(tlog.read_text(errors="replace"))
            break
    return "\n".join(parts)


def save_error_tail(wrapper_log: Path, variant: str, seed: int) -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tail_path = LOG_DIR / f"{variant}_seed{seed}_error_tail.txt"
    if wrapper_log.is_file():
        lines = wrapper_log.read_text(errors="replace").splitlines()
        tail_path.write_text("\n".join(lines[-100:]) + "\n", encoding="utf-8")
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


def run_one(variant: str, cfg_path: Path, gpu: str, seed: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{variant}_seed{seed}_gpu{gpu}.log"
    env = os.environ.copy()
    # Pin each subprocess to one physical GPU; EasyTorch then uses logical device 0.
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
        "error_tail_file": "",
    }
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write(f"command: {' '.join(cmd)}\n")
            lf.write(f"CUDA_VISIBLE_DEVICES={gpu}\n\n")
            lf.flush()
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        metrics = parse_metrics(collect_log_text(variant, seed, log_file))
        row.update(metrics)
        if proc.returncode != 0:
            row["status"] = f"exit_{proc.returncode}"
            row["error_tail_file"] = save_error_tail(log_file, variant, seed)
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
    log_text = collect_log_text(variant, seed, wrapper_log)
    metrics = parse_metrics(log_text)
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
        ok = [r for r in runs if r.get("mae") is not None and r.get("rmse") is not None and r.get("mape") is not None]
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


def build_paired_diff(rows: list[dict], left: str, right: str) -> list[dict]:
    """Paired difference left - right by seed. Positive = left worse than right."""
    by_key: dict[tuple[str, int], dict] = {}
    for r in rows:
        if r.get("mae") is None:
            continue
        by_key[(r["variant"], int(r["seed"]))] = r

    paired: list[dict] = []
    for metric in ("mae", "rmse", "mape"):
        diffs: list[float] = []
        for seed in sorted({int(r["seed"]) for r in rows}):
            left_row = by_key.get((left, seed))
            right_row = by_key.get((right, seed))
            if not left_row or not right_row:
                continue
            lv = left_row.get(metric)
            rv = right_row.get(metric)
            if lv is None or rv is None:
                continue
            diffs.append(float(lv) - float(rv))
        m, s, c = mean_std_ci(diffs)
        paired.append({"metric": metric, "mean_diff": m, "std_diff": s, "ci95": c, "n_pairs": len(diffs)})
    return paired


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
    fields = ["variant", "seed", "mae", "rmse", "mape", "ckpt_dir", "log_file", "status", "error_tail_file"]
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

    md = [
        "# KASA Future Prior Ablation on PeMS04 12→12\n\n",
        "## Per-run results\n\n",
        "| variant | seed | MAE | RMSE | MAPE | status |\n",
        "|---|---:|---:|---:|---:|---|\n",
    ]
    for r in rows:
        md.append(
            f"| {r['variant']} | {r['seed']} | {fmt_val(r['mae'])} | {fmt_val(r['rmse'])} | "
            f"{fmt_val(r['mape'])} | {r['status']} |\n"
        )
    md.append("\n## Summary\n\n")
    md.append(
        "| variant | n | MAE mean | MAE std | MAE 95% CI | RMSE mean | RMSE std | RMSE 95% CI | "
        "MAPE mean | MAPE std | MAPE 95% CI | failed |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in summary:
        md.append(
            f"| {s['variant']} | {s['n']} | {fmt_val(s['mae_mean'])} | {fmt_val(s['mae_std'])} | {fmt_val(s['mae_ci95'])} | "
            f"{fmt_val(s['rmse_mean'])} | {fmt_val(s['rmse_std'])} | {fmt_val(s['rmse_ci95'])} | "
            f"{fmt_val(s['mape_mean'])} | {fmt_val(s['mape_std'])} | {fmt_val(s['mape_ci95'])} | {s['failed']} |\n"
        )

    for left, right, title, note in PAIRED_SECTIONS:
        paired = build_paired_diff(rows, left, right)
        md.append(f"\n## Paired difference: {title}\n\n")
        md.append(f"Positive value means {note}.\n\n")
        md.append("| metric | mean diff | std diff | 95% CI |\n")
        md.append("|---|---:|---:|---:|\n")
        for p in paired:
            md.append(
                f"| {p['metric']} | {fmt_val(p['mean_diff'])} | {fmt_val(p['std_diff'])} | {fmt_val(p['ci95'])} |\n"
            )

    out_md.write_text("".join(md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="KASA history vs future prior ablation on PeMS04.")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        choices=list(VARIANT_OVERRIDES.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--out", default="results/pems04_kasa_future_prior_ablation.csv")
    parser.add_argument("--markdown", default="results/pems04_kasa_future_prior_ablation.md")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary-only", "--summary_only", action="store_true", dest="summary_only")
    args = parser.parse_args()

    if not BASE_CFG.is_file():
        print(f"Missing base config: {BASE_CFG}")
        return 1

    summary_csv = ROOT / "results" / "pems04_kasa_future_prior_ablation_summary.csv"
    jobs = [(v, generate_temp_config(v, s), s) for v in args.variants for s in args.seeds]

    if args.dry_run:
        print("Dry run — temp configs and commands:\n")
        for variant, cfg_path, seed in jobs:
            report = variant_report(variant)
            print(f"  [{variant} seed={seed}]")
            print(f"    cfg: {cfg_path}")
            print(f"    ckpt: {ckpt_dir_for(variant, seed)}")
            for key in REPORT_KEYS:
                print(f"    {key}: {report[key]}")
            print(f"    CUDA_VISIBLE_DEVICES: <physical_gpu>")
            print(
                f"    cmd: CUDA_VISIBLE_DEVICES=<physical_gpu> {sys.executable} {RUN_PY} "
                f"--cfg {cfg_for_easytorch(cfg_path)} --gpus 0"
            )
        print(f"\n{len(jobs)} jobs, GPUs: {args.gpus}")
        return 0

    if args.summary_only:
        rows = [summarize_row(v, s) for v, _, s in jobs]
        write_outputs(rows, ROOT / args.out, ROOT / args.markdown, summary_csv)
        print(f"Wrote {args.out}, {args.markdown}, {summary_csv}")
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

    write_outputs(rows, ROOT / args.out, ROOT / args.markdown, summary_csv)
    print(f"Wrote {args.out}, {args.markdown}, {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
