#!/usr/bin/env python3
"""Run ready PeMS04 12->12 baseline configs with a simple GPU queue."""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "examples" / "baselines"
RUN_PY = ROOT / "examples" / "run.py"
LOG_DIR = ROOT / "results" / "baseline_logs"

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


def config_path(model: str) -> Path | None:
    p = BASELINE_DIR / model / f"{model}_PEMS04.py"
    return p if p.is_file() else None


def cfg_for_easytorch(path: Path) -> str:
    """EasyTorch expects a repo-relative module path, not an absolute filesystem path."""
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
    """Parse test metrics; prefer lowest-MAE result block, else fallback to full log."""
    out: dict[str, float | None] = {k: None for k in METRIC_PATTERNS}
    blocks = list(RESULT_TEST_BLOCK.finditer(log_text))

    best: dict[str, float | None] | None = None
    best_mae: float | None = None
    for m in blocks:
        parsed = _parse_block(m.group(1))
        if parsed["mae"] is None:
            continue
        if best_mae is None or parsed["mae"] < best_mae:
            best_mae = parsed["mae"]
            best = parsed

    if best is not None:
        return best

    # Fallback: scan entire log for metric triples.
    fallback = _parse_block(log_text)
    if fallback["mae"] is not None:
        return fallback
    return out


def collect_log_text(wrapper_log: Path, model: str, seed: int) -> str:
    """Merge launcher stdout with checkpoint training logs (metrics live there)."""
    parts = []
    if wrapper_log.is_file():
        parts.append(wrapper_log.read_text(errors="replace"))
    ckpt_root = ROOT / "checkpoints" / "baselines"
    if ckpt_root.is_dir():
        for d in sorted(ckpt_root.glob(f"{model}_PEMS04_*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if f"seed{seed}" not in d.name and seed != 1:
                continue
            for tlog in sorted(d.glob("*/training_log_*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
                parts.append(tlog.read_text(errors="replace"))
                break
            if len(parts) > 1:
                break
    return "\n".join(parts)


_CKPT_SAVE_DIR_CONCAT = re.compile(
    r'(CFG\.TRAIN\.CKPT_SAVE_DIR\s*=\s*os\.path\.join\("checkpoints",\s*"baselines",\s*")'
    r'([^"]*)(" \+ str\(CFG\.TRAIN\.NUM_EPOCHS\)\))'
)


def _seed_ckpt_prefix(prefix: str, seed: int) -> str:
    """Insert _seed<N>_ into baselines CKPT dir prefix, e.g. DCRNN_PEMS04_ -> DCRNN_PEMS04_seed3_."""
    base = re.sub(r"_seed\d+", "", prefix).strip("_")
    return f"{base}_seed{seed}_"


def make_seed_config(base_cfg: Path, model: str, seed: int) -> Path:
    text = base_cfg.read_text()
    if "CFG.ENV.SEED" in text:
        text = re.sub(r"CFG\.ENV\.SEED\s*=\s*\d+", f"CFG.ENV.SEED = {seed}", text)
    else:
        text = text.replace("CFG.ENV = EasyDict()", f"CFG.ENV = EasyDict()\nCFG.ENV.SEED = {seed}")

    def _ckpt_repl(m: re.Match) -> str:
        return f'{m.group(1)}{_seed_ckpt_prefix(m.group(2), seed)}{m.group(3)}'

    if _CKPT_SAVE_DIR_CONCAT.search(text):
        text = _CKPT_SAVE_DIR_CONCAT.sub(_ckpt_repl, text, count=1)
    else:
        # Fallback: literal third argument inside path.join (no "+ str(...)" suffix).
        text = re.sub(
            r'(CFG\.TRAIN\.CKPT_SAVE_DIR\s*=\s*os\.path\.join\("checkpoints",\s*"baselines",\s*")([^"]+)(")',
            lambda m: f'{m.group(1)}{_seed_ckpt_prefix(m.group(2), seed).rstrip("_")}{m.group(3)}',
            text,
            count=1,
        )
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{base_cfg.stem}_seed{seed}.py",
        delete=False,
        dir=base_cfg.parent,
        prefix="",
    )
    tmp.write(text)
    tmp.close()
    return Path(tmp.name)


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


def run_one(model: str, cfg: Path, gpu: str, seed: int, multi_seed: bool) -> dict:
    if multi_seed or seed != 1:
        cfg_to_run = make_seed_config(cfg, model, seed)
        cleanup_cfg = True
    else:
        cfg_to_run = cfg
        cleanup_cfg = False

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{model}_seed{seed}_gpu{gpu}.log"
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    cfg_arg = cfg_for_easytorch(cfg_to_run)
    cmd = [sys.executable, str(RUN_PY), "--cfg", cfg_arg, "--gpus", str(gpu)]
    row = {
        "model": model,
        "seed": seed,
        "mae": None,
        "rmse": None,
        "mape": None,
        "ckpt_dir": "",
        "log_file": str(log_file),
        "status": "running",
    }
    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        log_text = collect_log_text(log_file, model, seed)
        metrics = parse_metrics(log_text)
        row.update(metrics)
        if proc.returncode != 0:
            row["status"] = f"exit_{proc.returncode}"
        elif metrics["mae"] is None:
            row["status"] = "failed_no_metrics"
        else:
            row["status"] = "ok"
    except Exception as e:
        row["status"] = f"error:{e}"
    finally:
        if cleanup_cfg and cfg_to_run.exists():
            cfg_to_run.unlink()
    return row


def summarize_row(model: str, seed: int) -> dict:
    wrapper_log = None
    if LOG_DIR.is_dir():
        matches = sorted(LOG_DIR.glob(f"{model}_seed{seed}_gpu*.log"))
        if matches:
            wrapper_log = matches[-1]
    log_file = str(wrapper_log) if wrapper_log else ""
    log_text = collect_log_text(wrapper_log, model, seed) if wrapper_log else ""
    metrics = parse_metrics(log_text)
    status = "ok" if metrics["mae"] is not None else "failed_no_metrics"
    return {
        "model": model,
        "seed": seed,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "ckpt_dir": "",
        "log_file": log_file,
        "status": status,
    }


def t_crit(n: int) -> float:
    if n in T_CRIT:
        return T_CRIT[n]
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, n - 1))
    except Exception:
        # Normal approximation when scipy is unavailable.
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
    ci = t_crit(n) * std / math.sqrt(n)
    return mean, std, ci


def build_summary(rows: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    summary = []
    for model in sorted(by_model):
        runs = by_model[model]
        ok = [r for r in runs if r.get("mae") is not None and r.get("rmse") is not None and r.get("mape") is not None]
        failed = len(runs) - len(ok)
        mae_vals = [float(r["mae"]) for r in ok]
        rmse_vals = [float(r["rmse"]) for r in ok]
        mape_vals = [float(r["mape"]) for r in ok]
        m_mae, s_mae, c_mae = mean_std_ci(mae_vals)
        m_rmse, s_rmse, c_rmse = mean_std_ci(rmse_vals)
        m_mape, s_mape, c_mape = mean_std_ci(mape_vals)
        summary.append({
            "model": model,
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
    rows = sorted(rows, key=lambda r: (r["model"], int(r["seed"])))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "seed", "mae", "rmse", "mape", "ckpt_dir", "log_file", "status"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = build_summary(rows)
    sum_fields = [
        "model", "n", "mae_mean", "mae_std", "mae_ci95",
        "rmse_mean", "rmse_std", "rmse_ci95",
        "mape_mean", "mape_std", "mape_ci95", "failed",
    ]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary)

    md = ["# PeMS04 baseline results (12→12)\n\n", "## Per-run results\n\n",
          "| model | seed | MAE | RMSE | MAPE | status |\n", "|---|---:|---:|---:|---:|---|\n"]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['seed']} | {fmt_val(r['mae'])} | {fmt_val(r['rmse'])} | "
            f"{fmt_val(r['mape'])} | {r['status']} |\n"
        )
    md.append("\n## Summary\n\n")
    md.append(
        "| model | n | MAE mean | MAE std | MAE 95% CI | RMSE mean | RMSE std | RMSE 95% CI | "
        "MAPE mean | MAPE std | MAPE 95% CI | failed |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in summary:
        md.append(
            f"| {s['model']} | {s['n']} | {fmt_val(s['mae_mean'])} | {fmt_val(s['mae_std'])} | {fmt_val(s['mae_ci95'])} | "
            f"{fmt_val(s['rmse_mean'])} | {fmt_val(s['rmse_std'])} | {fmt_val(s['rmse_ci95'])} | "
            f"{fmt_val(s['mape_mean'])} | {fmt_val(s['mape_std'])} | {fmt_val(s['mape_ci95'])} | {s['failed']} |\n"
        )
    out_md.write_text("".join(md))


def resolve_seeds(args) -> list[int]:
    if args.seed_list:
        return list(args.seed_list)
    return list(range(1, args.seeds + 1))


def main():
    parser = argparse.ArgumentParser(
        description="Run PeMS04 baseline configs. "
        "Use --seeds N to run seeds 1..N, or --seed-list 1 2 3 for explicit seeds."
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Run seeds 1, 2, ..., N (inclusive). Example: --seeds 5 runs five seeds.",
    )
    parser.add_argument(
        "--seed-list",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seed list, e.g. --seed-list 1 2 3 4 5. Overrides --seeds.",
    )
    parser.add_argument("--out", default="results/pems04_baselines.csv")
    parser.add_argument("--markdown", default="results/pems04_baselines.md")
    parser.add_argument("--summary-only", "--summary_only", action="store_true", dest="summary_only")
    args = parser.parse_args()

    seed_list = resolve_seeds(args)
    multi_seed = len(seed_list) > 1
    summary_csv = ROOT / "results" / "pems04_baselines_summary.csv"

    ready, skipped = [], []
    for m in args.models:
        p = config_path(m)
        if p is None:
            skipped.append(m)
        else:
            ready.append((m, p))

    if skipped:
        print("Skipped (no ready config):", ", ".join(skipped))

    if args.summary_only:
        rows = [summarize_row(model, seed) for model, _ in ready for seed in seed_list]
        write_outputs(rows, ROOT / args.out, ROOT / args.markdown, summary_csv)
        print(f"Wrote {args.out}, {args.markdown}, {summary_csv}")
        return

    queue = GPUQueue(args.gpus)
    rows: list[dict] = []
    lock = threading.Lock()

    def worker(model: str, cfg: Path, seed: int):
        gpu = queue.acquire()
        try:
            print(f"[start] {model} seed={seed} gpu={gpu}")
            row = run_one(model, cfg, gpu, seed, multi_seed)
            with lock:
                rows.append(row)
            print(f"[done]  {model} seed={seed} status={row['status']} mae={row['mae']}")
        finally:
            queue.release(gpu)

    threads = []
    for model, cfg in ready:
        for seed in seed_list:
            t = threading.Thread(target=worker, args=(model, cfg, seed))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()

    write_outputs(rows, ROOT / args.out, ROOT / args.markdown, summary_csv)
    print(f"Wrote {args.out}, {args.markdown}, {summary_csv}")


if __name__ == "__main__":
    main()
