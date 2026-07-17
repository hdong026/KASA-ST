#!/usr/bin/env python3
"""Run CapDistRefine on PeMS04 horizons 16/32/64."""
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
MODEL_NAME = "CapDistRefine"
DEFAULT_WORK_DIR = ROOT / "experiments" / "capdist_refine_pems04"
DEFAULT_LOG_ROOT = ROOT / "logs" / "capdist_refine_pems04"
DEFAULT_CKPT_ROOT = ROOT / "checkpoints" / "capdist_refine_pems04"
DEFAULT_HORIZONS = [16, 32, 64]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_GPUS = ["0", "1"]

HORIZON_CONFIGS: dict[int, Path] = {
    16: ROOT / "examples" / "CapDistRefine" / "CapDistRefine_PEMS04_16to16.py",
    32: ROOT / "examples" / "CapDistRefine" / "CapDistRefine_PEMS04_16to32.py",
    64: ROOT / "examples" / "CapDistRefine" / "CapDistRefine_PEMS04_16to64.py",
}

HORIZON_CHAIN: dict[int, list[int]] = {
    16: [4, 8, 16],
    32: [8, 16, 32],
    64: [16, 32, 64],
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
SHOW_ERROR_LINES = 80


def horizon_spec(horizon: int) -> dict[str, Any]:
    if horizon not in HORIZON_CONFIGS:
        raise ValueError(f"Unknown horizon: {horizon}. Choose from {list(HORIZON_CONFIGS)}")
    return {
        "base_cfg": HORIZON_CONFIGS[horizon],
        "input_len": 16,
        "output_len": horizon,
        "chain_lengths": HORIZON_CHAIN[horizon],
    }


def cfg_dir(work_dir: Path) -> Path:
    return work_dir / "configs"


def ckpt_dir_for(horizon: int, seed: int, ckpt_root: Path) -> Path:
    return ckpt_root / f"h{horizon}" / MODEL_NAME / f"seed{seed}"


def log_dir_for(horizon: int, seed: int, log_root: Path) -> Path:
    return log_root / f"h{horizon}" / MODEL_NAME / f"seed{seed}"


def temp_cfg_path(horizon: int, seed: int, work_dir: Path) -> Path:
    out = cfg_dir(work_dir) / f"h{horizon}_{MODEL_NAME}_seed{seed}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def strip_hardcoded_cuda_devices(content: str) -> str:
    kept: list[str] = []
    for line in content.splitlines():
        if "CUDA_VISIBLE_DEVICES" in line and "os.environ" in line:
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def generate_temp_config(horizon: int, seed: int, work_dir: Path, ckpt_root: Path) -> Path:
    hspec = horizon_spec(horizon)
    base_cfg = hspec["base_cfg"]
    content = strip_hardcoded_cuda_devices(base_cfg.read_text(encoding="utf-8"))
    ckpt_rel = os.path.join(
        "checkpoints", "capdist_refine_pems04", f"h{horizon}", MODEL_NAME, f"seed{seed}"
    )
    lines = [
        "",
        "# ===== CapDistRefine runner overrides (auto-generated) =====",
        f"CFG.ENV.SEED = {seed}",
        "if hasattr(CFG, 'SEED'):",
        f"    CFG.SEED = {seed}",
        "if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):",
        f"    CFG.TRAIN.SEED = {seed}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")',
        "CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]",
        "CFG.MODEL.TARGET_FEATURES = [0]",
        f'CFG.DESCRIPTION = "PeMS04 CapDistRefine h{horizon} seed{seed}"',
    ]
    out = temp_cfg_path(horizon, seed, work_dir)
    out.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")
    return out


def load_cfg(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("capdist_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CFG


def validate_generated_config(cfg_path: Path, horizon: int, seed: int) -> None:
    hspec = horizon_spec(horizon)
    cfg = load_cfg(cfg_path)
    if int(cfg.DATASET_INPUT_LEN) != 16:
        raise ValueError(f"{cfg_path}: DATASET_INPUT_LEN must be 16")
    if int(cfg.DATASET_OUTPUT_LEN) != horizon:
        raise ValueError(f"{cfg_path}: DATASET_OUTPUT_LEN expected {horizon}")
    param = cfg.MODEL.PARAM
    if list(param.get("chain_lengths", [])) != hspec["chain_lengths"]:
        raise ValueError(f"{cfg_path}: chain_lengths mismatch")
    if str(param.get("spatial_placement", "")).lower() != "temporal_first_capdist_refine":
        raise ValueError(f"{cfg_path}: spatial_placement must be temporal_first_capdist_refine")
    if str(param.get("post_spatial_mode", "")).lower() != "adaptive_cluster_mix":
        raise ValueError(f"{cfg_path}: post_spatial_mode must be adaptive_cluster_mix")
    if str(param.get("unified_aux_loss_mode", "none")).lower() != "none":
        raise ValueError(f"{cfg_path}: unified_aux_loss_mode must be none")
    if str(param.get("capdist_cluster_method", "")) != "capdist_spectral_pair":
        raise ValueError(f"{cfg_path}: capdist_cluster_method must be capdist_spectral_pair")
    if int(cfg.ENV.SEED) != seed:
        raise ValueError(f"{cfg_path}: seed mismatch")


def cfg_for_easytorch(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


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


def collect_log_text(horizon: int, seed: int, log_root: Path, ckpt_root: Path) -> str:
    parts: list[str] = []
    log_dir = log_dir_for(horizon, seed, log_root)
    wrapper = log_dir / "train.log"
    if wrapper.is_file():
        parts.append(wrapper.read_text(errors="replace"))
    ckpt_base = ckpt_dir_for(horizon, seed, ckpt_root)
    tlog = pick_canonical_training_log(ckpt_base)
    if tlog is not None:
        parts.append(tlog.read_text(errors="replace"))
    return "\n".join(parts)


def parse_training_log(log_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "best_val_mae": None,
        "test_mae_at_best_val": None,
        "test_rmse_at_best_val": None,
        "best_epoch": None,
        "final_test_mae": None,
        "final_test_rmse": None,
    }
    if not log_text.strip():
        return out
    current_epoch: int | None = None
    last_val_mae: float | None = None
    best_val_so_far = float("inf")
    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        m = EPOCH_LINE.search(line)
        if m:
            current_epoch = int(m.group(1))
        m = VAL_LINE.search(line)
        if m:
            last_val_mae = float(m.group(1))
        if BEST_CKPT.search(line) and last_val_mae is not None and last_val_mae <= best_val_so_far:
            best_val_so_far = last_val_mae
            out["best_val_mae"] = last_val_mae
            out["best_epoch"] = current_epoch
            for j in range(i + 1, min(i + 40, len(lines))):
                tm = TEST_BLOCK.search(lines[j])
                if tm:
                    out["test_mae_at_best_val"] = float(tm.group(1))
                    out["test_rmse_at_best_val"] = float(tm.group(2))
                    break
        m = TEST_BLOCK.search(line)
        if m:
            out["final_test_mae"] = float(m.group(1))
            out["final_test_rmse"] = float(m.group(2))
    return out


def is_completed(horizon: int, seed: int, log_root: Path, ckpt_root: Path) -> bool:
    parsed = parse_training_log(collect_log_text(horizon, seed, log_root, ckpt_root))
    return parsed.get("test_mae_at_best_val") is not None


class GPUScheduler:
    def __init__(self, gpus: list[str]):
        self.gpus = gpus
        self.lock = threading.Lock()
        self.in_use = {g: 0 for g in gpus}

    def acquire(self) -> str:
        while True:
            with self.lock:
                for gpu in self.gpus:
                    if self.in_use[gpu] < 1:
                        self.in_use[gpu] += 1
                        return gpu
            time.sleep(2)

    def release(self, gpu: str) -> None:
        with self.lock:
            self.in_use[gpu] = max(0, self.in_use[gpu] - 1)


def print_log_tail(log_file: Path, horizon: int, seed: int, lines: int = SHOW_ERROR_LINES) -> None:
    if not log_file.is_file():
        return
    tail = log_file.read_text(errors="replace").splitlines()[-lines:]
    print(f"\n--- error tail: h{horizon} {MODEL_NAME} seed={seed} ---")
    for line in tail:
        print(line)
    print("--- end error tail ---\n")


def run_one(
    horizon: int,
    cfg_path: Path,
    gpu: str,
    seed: int,
    log_root: Path,
    ckpt_root: Path,
    show_errors: bool,
) -> dict[str, Any]:
    log_dir = log_dir_for(horizon, seed, log_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, str(RUN_PY), "--cfg", cfg_for_easytorch(cfg_path), "--gpus", "0"]
    row = {
        "horizon": horizon,
        "model": MODEL_NAME,
        "seed": seed,
        "status": "running",
        "config_path": str(cfg_path),
        "log_file": str(log_file),
    }
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        parsed = parse_training_log(collect_log_text(horizon, seed, log_root, ckpt_root))
        row.update(parsed)
        if proc.returncode != 0:
            row["status"] = f"exit_{proc.returncode}"
            if show_errors:
                print_log_tail(log_file, horizon, seed)
        elif parsed.get("test_mae_at_best_val") is None:
            row["status"] = "failed_no_metrics"
            if show_errors:
                print_log_tail(log_file, horizon, seed)
        else:
            row["status"] = "ok"
    except Exception as exc:
        row["status"] = f"error:{exc}"
    return row


def dry_run_info(horizons: list[int], seeds: list[int], work_dir: Path, ckpt_root: Path) -> list[dict]:
    rows = []
    for horizon in horizons:
        for seed in seeds:
            cfg_path = generate_temp_config(horizon, seed, work_dir, ckpt_root)
            validate_generated_config(cfg_path, horizon, seed)
            cfg = load_cfg(cfg_path)
            param = cfg.MODEL.PARAM
            rows.append({
                "horizon": horizon,
                "seed": seed,
                "config": str(cfg_path),
                "input_len": cfg.DATASET_INPUT_LEN,
                "output_len": cfg.DATASET_OUTPUT_LEN,
                "chain_lengths": param.get("chain_lengths"),
                "spatial_placement": param.get("spatial_placement"),
                "post_spatial_mode": param.get("post_spatial_mode"),
                "capdist_cluster_method": param.get("capdist_cluster_method"),
                "capdist_lambda_mix": param.get("capdist_lambda_mix"),
                "capdist_alphas": param.get("capdist_alphas"),
                "capdist_topks": param.get("capdist_topks"),
            })
    return rows


def write_markdown(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CapDistRefine PeMS04 Results",
        "",
        "| horizon | seed | status | best_val_MAE | test_MAE@best | test_RMSE@best | best_epoch |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (int(r["horizon"]), int(r["seed"]))):
        lines.append(
            f"| {row['horizon']} | {row['seed']} | {row.get('status', '')} | "
            f"{row.get('best_val_mae', '') or ''} | {row.get('test_mae_at_best_val', '') or ''} | "
            f"{row.get('test_rmse_at_best_val', '') or ''} | {row.get('best_epoch', '') or ''} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "horizon", "model", "seed", "status", "best_val_mae", "test_mae_at_best_val",
        "test_rmse_at_best_val", "best_epoch", "config_path", "log_file",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (int(r["horizon"]), int(r["seed"]))):
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CapDistRefine on PeMS04")
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", type=str, nargs="+", default=DEFAULT_GPUS)
    parser.add_argument("--work_dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--log_root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--ckpt_root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "capdist_refine_pems04.csv")
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--show_errors", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    horizons = sorted(set(args.horizons))
    seeds = sorted(set(args.seeds))
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        rows = dry_run_info(horizons, seeds, args.work_dir, args.ckpt_root)
        print(f"CapDistRefine dry-run: {len(rows)} jobs")
        for row in rows:
            print(
                f"  h{row['horizon']} seed{row['seed']}: "
                f"chain={row['chain_lengths']} placement={row['spatial_placement']} "
                f"cluster={row['capdist_cluster_method']}"
            )
        return

    scheduler = GPUScheduler(args.gpus)
    all_rows: list[dict] = []
    jobs = [(h, s) for h in horizons for s in seeds]

    for horizon, seed in jobs:
        cfg_path = generate_temp_config(horizon, seed, args.work_dir, args.ckpt_root)
        validate_generated_config(cfg_path, horizon, seed)
        if args.skip_existing and is_completed(horizon, seed, args.log_root, args.ckpt_root):
            parsed = parse_training_log(collect_log_text(horizon, seed, args.log_root, args.ckpt_root))
            row = {
                "horizon": horizon,
                "model": MODEL_NAME,
                "seed": seed,
                "status": "skipped_ok",
                "config_path": str(cfg_path),
                **parsed,
            }
            all_rows.append(row)
            print(f"[skip] h{horizon} seed{seed} already completed")
            continue

        gpu = scheduler.acquire()
        print(f"[run] h{horizon} seed{seed} gpu={gpu}")
        try:
            row = run_one(
                horizon, cfg_path, gpu, seed,
                args.log_root, args.ckpt_root, args.show_errors,
            )
        finally:
            scheduler.release(gpu)
        all_rows.append(row)
        print(
            f"[done] h{horizon} seed{seed} status={row['status']} "
            f"test_MAE={row.get('test_mae_at_best_val')}"
        )

    write_csv(all_rows, args.out)
    if args.markdown:
        write_markdown(all_rows, args.markdown)
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
