#!/usr/bin/env python3
"""Run D2STGNN on PeMS04 with 16-input horizons 16/32/64 and configurable gap."""
from __future__ import annotations

import argparse
import csv
import importlib.util
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
BASE_CFG = ROOT / "examples" / "baselines" / "D2STGNN" / "D2STGNN_PEMS04.py"
DEFAULT_WORK_DIR = ROOT / "experiments" / "d2stgnn_h16_pems04"
DEFAULT_LOG_ROOT = ROOT / "logs" / "d2stgnn_h16_pems04"
DEFAULT_CKPT_ROOT = ROOT / "checkpoints" / "d2stgnn_h16_pems04"

INPUT_LEN = 16
DEFAULT_HORIZONS = [16, 32, 64]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_GPUS = ["0", "1"]
ALLOWED_GAPS = (2, 4)
DEFAULT_GAP = 4

EPOCH_LINE = re.compile(r"Epoch\s+(\d+)\s*/", re.I)
VAL_LINE = re.compile(r"Result\s*<val>.*?val_MAE:\s*([0-9.eE+-]+)", re.I | re.S)
TEST_BLOCK = re.compile(
    r"Result\s*<test>.*?test_MAE:\s*([0-9.eE+-]+).*?"
    r"test_RMSE:\s*([0-9.eE+-]+)",
    re.I | re.S,
)
BEST_CKPT = re.compile(r"best_val_MAE\.pt saved", re.I)


def cfg_dir(work_dir: Path) -> Path:
    return work_dir / "configs"


def ckpt_dir_for(gap: int, horizon: int, seed: int, ckpt_root: Path) -> Path:
    return ckpt_root / f"gap{gap}" / f"h{horizon}" / f"seed{seed}"


def log_dir_for(gap: int, horizon: int, seed: int, log_root: Path) -> Path:
    return log_root / f"gap{gap}" / f"h{horizon}" / f"seed{seed}"


def temp_cfg_path(gap: int, horizon: int, seed: int, work_dir: Path) -> Path:
    out = cfg_dir(work_dir) / f"gap{gap}_h{horizon}_seed{seed}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def strip_hardcoded_cuda_devices(content: str) -> str:
    kept: list[str] = []
    for line in content.splitlines():
        if "CUDA_VISIBLE_DEVICES" in line and "os.environ" in line:
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def validate_gap_horizon(gap: int, horizon: int) -> None:
    if gap not in ALLOWED_GAPS:
        raise ValueError(f"gap must be one of {ALLOWED_GAPS}, got {gap}")
    if horizon % gap != 0:
        raise ValueError(
            f"horizon {horizon} is not divisible by gap {gap}; "
            f"pred length would be {horizon // gap * gap}"
        )


def generate_temp_config(
    gap: int,
    horizon: int,
    seed: int,
    work_dir: Path,
    ckpt_root: Path,
) -> Path:
    validate_gap_horizon(gap, horizon)
    content = strip_hardcoded_cuda_devices(BASE_CFG.read_text(encoding="utf-8"))
    ckpt_rel = os.path.join(
        "checkpoints",
        "d2stgnn_h16_pems04",
        f"gap{gap}",
        f"h{horizon}",
        f"seed{seed}",
    )
    lines = [
        "",
        "# ===== D2STGNN 16-input runner overrides (auto-generated) =====",
        f"CFG.ENV.SEED = {seed}",
        "if hasattr(CFG, 'SEED'):",
        f"    CFG.SEED = {seed}",
        "if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):",
        f"    CFG.TRAIN.SEED = {seed}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")',
        f'CFG.DESCRIPTION = "D2STGNN PeMS04 16→{horizon} gap={gap} seed{seed}"',
        f"CFG.DATASET_INPUT_LEN = {INPUT_LEN}",
        f"CFG.DATASET_OUTPUT_LEN = {horizon}",
        f"CFG.TEST.EVALUATION_HORIZONS = list(range(1, {horizon + 1}))",
        f'CFG.MODEL.PARAM["gap"] = {gap}',
        f'CFG.MODEL.PARAM["input_seq_len"] = {INPUT_LEN}',
        f'CFG.MODEL.PARAM["output_seq_len"] = {horizon}',
        f'CFG.MODEL.PARAM["seq_length"] = {horizon}',
        f"CFG.TRAIN.CL.PREDICTION_LENGTH = {horizon}",
    ]
    out = temp_cfg_path(gap, horizon, seed, work_dir)
    out.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")
    return out


def load_cfg(cfg_path: Path):
    spec = importlib.util.spec_from_file_location("d2stgnn_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CFG


def validate_generated_config(cfg_path: Path, gap: int, horizon: int, seed: int) -> None:
    validate_gap_horizon(gap, horizon)
    cfg = load_cfg(cfg_path)
    if int(cfg.DATASET_INPUT_LEN) != INPUT_LEN:
        raise ValueError(f"{cfg_path}: DATASET_INPUT_LEN must be {INPUT_LEN}")
    if int(cfg.DATASET_OUTPUT_LEN) != horizon:
        raise ValueError(f"{cfg_path}: DATASET_OUTPUT_LEN must be {horizon}")
    param = cfg.MODEL.PARAM
    if int(param.get("gap")) != gap:
        raise ValueError(f"{cfg_path}: gap expected {gap}, got {param.get('gap')}")
    if int(param.get("input_seq_len")) != INPUT_LEN:
        raise ValueError(f"{cfg_path}: input_seq_len must be {INPUT_LEN}")
    if int(param.get("output_seq_len")) != horizon:
        raise ValueError(f"{cfg_path}: output_seq_len must be {horizon}")
    if int(cfg.ENV.SEED) != seed:
        raise ValueError(f"{cfg_path}: seed mismatch")


def verify_forward_shape(cfg_path: Path, horizon: int) -> None:
    import torch

    cfg = load_cfg(cfg_path)
    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM).eval()
    x = torch.randn(1, INPUT_LEN, 307, 3)
    x[..., 1] = torch.rand(1, INPUT_LEN, 307)
    x[..., 2] = torch.randint(0, 7, (1, INPUT_LEN, 307)).float() / 7.0
    with torch.no_grad():
        y = model(x, None, 0, 0, False)
    if int(y.shape[1]) != horizon:
        raise ValueError(
            f"{cfg_path}: forward pred_T={y.shape[1]} expected {horizon}"
        )


def cfg_for_easytorch(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


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


def pick_training_log(ckpt_base: Path) -> Path | None:
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
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def collect_log_text(gap: int, horizon: int, seed: int, log_root: Path, ckpt_root: Path) -> str:
    parts: list[str] = []
    log_file = log_dir_for(gap, horizon, seed, log_root) / "train.log"
    if log_file.is_file():
        parts.append(log_file.read_text(errors="replace"))
    tlog = pick_training_log(ckpt_dir_for(gap, horizon, seed, ckpt_root))
    if tlog is not None:
        parts.append(tlog.read_text(errors="replace"))
    return "\n".join(parts)


def parse_training_log(log_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "best_val_mae": None,
        "test_mae_at_best_val": None,
        "test_rmse_at_best_val": None,
        "best_epoch": None,
    }
    if not log_text.strip():
        return out
    current_epoch = None
    last_val_mae = None
    best_val = float("inf")
    for i, line in enumerate(log_text.splitlines()):
        m = EPOCH_LINE.search(line)
        if m:
            current_epoch = int(m.group(1))
        m = VAL_LINE.search(line)
        if m:
            last_val_mae = float(m.group(1))
        if BEST_CKPT.search(line) and last_val_mae is not None and last_val_mae <= best_val:
            best_val = last_val_mae
            out["best_val_mae"] = last_val_mae
            out["best_epoch"] = current_epoch
            for j in range(i + 1, min(i + 40, len(log_text.splitlines()))):
                tm = TEST_BLOCK.search(log_text.splitlines()[j])
                if tm:
                    out["test_mae_at_best_val"] = float(tm.group(1))
                    out["test_rmse_at_best_val"] = float(tm.group(2))
                    break
    return out


def is_completed(gap: int, horizon: int, seed: int, log_root: Path, ckpt_root: Path) -> bool:
    return parse_training_log(collect_log_text(gap, horizon, seed, log_root, ckpt_root)).get(
        "test_mae_at_best_val"
    ) is not None


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


def run_one(
    gap: int,
    horizon: int,
    cfg_path: Path,
    gpu: str,
    seed: int,
    log_root: Path,
    ckpt_root: Path,
) -> dict[str, Any]:
    log_dir = log_dir_for(gap, horizon, seed, log_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, str(RUN_PY), "--cfg", cfg_for_easytorch(cfg_path), "--gpus", "0"]
    row = {
        "gap": gap,
        "horizon": horizon,
        "seed": seed,
        "input_len": INPUT_LEN,
        "output_len": horizon,
        "status": "running",
        "config_path": str(cfg_path),
        "log_file": str(log_file),
    }
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        parsed = parse_training_log(collect_log_text(gap, horizon, seed, log_root, ckpt_root))
        row.update(parsed)
        row["status"] = "ok" if parsed.get("test_mae_at_best_val") is not None else f"exit_{proc.returncode}"
    except Exception as exc:
        row["status"] = f"error:{exc}"
    return row


def dry_run_info(
    gap: int,
    horizons: list[int],
    seeds: list[int],
    work_dir: Path,
    ckpt_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for horizon in horizons:
        for seed in seeds:
            cfg_path = generate_temp_config(gap, horizon, seed, work_dir, ckpt_root)
            validate_generated_config(cfg_path, gap, horizon, seed)
            verify_forward_shape(cfg_path, horizon)
            cfg = load_cfg(cfg_path)
            rows.append(
                {
                    "gap": gap,
                    "horizon": horizon,
                    "seed": seed,
                    "config": str(cfg_path),
                    "input_len": INPUT_LEN,
                    "output_len": horizon,
                    "ar_steps": horizon // gap,
                    "gap_param": cfg.MODEL.PARAM["gap"],
                }
            )
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "gap", "horizon", "seed", "input_len", "output_len", "status",
        "best_val_mae", "test_mae_at_best_val", "test_rmse_at_best_val",
        "best_epoch", "config_path", "log_file",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (int(r["gap"]), int(r["horizon"]), int(r["seed"]))):
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="D2STGNN PeMS04 16-input with gap 2 or 4")
    parser.add_argument("--gap", type=int, default=DEFAULT_GAP, choices=ALLOWED_GAPS)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", type=str, nargs="+", default=DEFAULT_GPUS)
    parser.add_argument("--work_dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--log_root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--ckpt_root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "d2stgnn_h16_pems04.csv")
    parser.add_argument("--prepare_data", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    horizons = sorted(set(args.horizons))
    seeds = sorted(set(args.seeds))
    for h in horizons:
        validate_gap_horizon(args.gap, h)

    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare_data or args.dry_run:
        rc = ensure_data(horizons)
        if rc != 0:
            raise SystemExit(rc)

    if args.dry_run:
        rows = dry_run_info(args.gap, horizons, seeds, args.work_dir, args.ckpt_root)
        print(f"D2STGNN h16 dry-run gap={args.gap}: {len(rows)} jobs")
        for row in rows:
            print(
                f"  gap={row['gap']} h={row['horizon']} seed={row['seed']} "
                f"AR_steps={row['ar_steps']} cfg={row['config']}"
            )
        return

    scheduler = GPUScheduler(args.gpus)
    all_rows: list[dict] = []
    jobs = [(args.gap, h, s) for h in horizons for s in seeds]

    for gap, horizon, seed in jobs:
        cfg_path = generate_temp_config(gap, horizon, seed, args.work_dir, args.ckpt_root)
        validate_generated_config(cfg_path, gap, horizon, seed)
        if args.skip_existing and is_completed(gap, horizon, seed, args.log_root, args.ckpt_root):
            parsed = parse_training_log(
                collect_log_text(gap, horizon, seed, args.log_root, args.ckpt_root)
            )
            all_rows.append(
                {
                    "gap": gap,
                    "horizon": horizon,
                    "seed": seed,
                    "status": "skipped_ok",
                    **parsed,
                    "config_path": str(cfg_path),
                }
            )
            print(f"[skip] gap={gap} h={horizon} seed={seed}")
            continue

        gpu = scheduler.acquire()
        print(f"[run] gap={gap} h={horizon} seed={seed} gpu={gpu}")
        try:
            row = run_one(gap, horizon, cfg_path, gpu, seed, args.log_root, args.ckpt_root)
        finally:
            scheduler.release(gpu)
        all_rows.append(row)
        print(
            f"[done] gap={gap} h={horizon} seed={seed} status={row['status']} "
            f"test_MAE={row.get('test_mae_at_best_val')}"
        )

    write_csv(all_rows, args.out)
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
