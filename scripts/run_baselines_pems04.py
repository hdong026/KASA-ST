#!/usr/bin/env python3
"""Run ready PeMS04 12->12 baseline configs with a simple GPU queue."""
from __future__ import annotations

import argparse
import csv
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

METRIC_PATTERNS = {
    "mae": re.compile(r"test_MAE:\s*([0-9.eE+-]+)", re.I),
    "rmse": re.compile(r"test_RMSE:\s*([0-9.eE+-]+)", re.I),
    "mape": re.compile(r"test_MAPE:\s*([0-9.eE+-]+)", re.I),
}
RESULT_TEST_BLOCK = re.compile(r"Result <test>:\s*\[(.*?)\]", re.I | re.S)


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


def parse_metrics(log_text: str) -> dict[str, float | None]:
    """Parse the last test result block (matches easytorch training logs)."""
    out: dict[str, float | None] = {k: None for k in METRIC_PATTERNS}
    blocks = list(RESULT_TEST_BLOCK.finditer(log_text))
    if not blocks:
        return out
    block = blocks[-1].group(1)
    for k, pat in METRIC_PATTERNS.items():
        m = pat.search(block)
        if m:
            out[k] = float(m.group(1))
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
            if parts:
                break
    return "\n".join(parts)


def make_seed_config(base_cfg: Path, model: str, seed: int) -> Path:
    text = base_cfg.read_text()
    if "CFG.ENV.SEED" in text:
        text = re.sub(r"CFG\.ENV\.SEED\s*=\s*\d+", f"CFG.ENV.SEED = {seed}", text)
    else:
        text = text.replace("CFG.ENV = EasyDict()", f"CFG.ENV = EasyDict()\nCFG.ENV.SEED = {seed}")
    # Separate checkpoint dirs per seed to avoid overwrites.
    text = re.sub(
        r'(CFG\.TRAIN\.CKPT_SAVE_DIR = os\.path\.join\("checkpoints", "baselines", ")([^"]+)(")',
        rf'\1\2_seed{seed}\3',
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

    log_dir = ROOT / "results" / "baseline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{model}_seed{seed}_gpu{gpu}.log"
    # EasyTorch launch_training calls set_visible_devices(--gpus) and overwrites
    # CUDA_VISIBLE_DEVICES. Always pass the physical GPU id here (not "0" for all jobs).
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


def write_outputs(rows: list[dict], out_csv: Path, out_md: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "seed", "mae", "rmse", "mape", "ckpt_dir", "log_file", "status"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# PeMS04 baseline results (12→12)\n",
        "| model | seed | MAE | RMSE | MAPE | status |\n",
        "|---|---:|---:|---:|---:|---|\n",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['seed']} | {r['mae']} | {r['rmse']} | {r['mape']} | {r['status']} |\n"
        )
    out_md.write_text("".join(lines))


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
    args = parser.parse_args()

    seed_list = resolve_seeds(args)
    multi_seed = len(seed_list) > 1

    ready, skipped = [], []
    for m in args.models:
        p = config_path(m)
        if p is None:
            skipped.append(m)
        else:
            ready.append((m, p))

    if skipped:
        print("Skipped (no ready config):", ", ".join(skipped))

    queue = GPUQueue(args.gpus)
    rows: list[dict] = []

    def worker(model: str, cfg: Path, seed: int):
        gpu = queue.acquire()
        try:
            print(f"[start] {model} seed={seed} gpu={gpu}")
            row = run_one(model, cfg, gpu, seed, multi_seed)
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

    write_outputs(rows, ROOT / args.out, ROOT / args.markdown)
    print(f"Wrote {args.out} and {args.markdown}")


if __name__ == "__main__":
    main()
