#!/usr/bin/env python3
"""Unified launcher for HyperD baseline on PEMS03/04/07/08."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.HyperD.Initialization import run_initialization
from baselines.HyperD.data_prepare import (
    ensure_hyperd_data,
    init_npy_paths,
    missing_requirements,
)
from baselines.HyperD.hyperd_settings import HYPERD_DATASETS, INPUT_LEN, OUTPUT_LEN

LOG_DIR = ROOT / "logs" / "baselines" / "hyperd"
RESULT_DIR = ROOT / "results" / "baselines" / "hyperd"
TRAIN_PY = ROOT / "baselines" / "HyperD" / "train.py"

VAL_MAE_RE = re.compile(r"Result\s*<val>:\s*\[.*?val_MAE:\s*([0-9.]+)", re.I)
BEST_VAL_RE = re.compile(r"ChainForecasting_best_val_MAE|best_val", re.I)
TEST_BLOCK_RE = re.compile(r"Result\s*<test>:\s*\[(.*?)\]", re.I | re.S)
METRIC_RES = {
    "mae": re.compile(r"test_MAE:\s*([0-9.]+)", re.I),
    "rmse": re.compile(r"test_RMSE:\s*([0-9.]+)", re.I),
    "mape": re.compile(r"test_MAPE:\s*([0-9.]+)", re.I),
}


def cfg_path(dataset: str) -> Path:
    return ROOT / "baselines" / "HyperD" / f"{dataset.upper()}.py"


def cfg_for_launch(path: Path) -> str:
    return str(path.relative_to(ROOT))


def ensure_inits(dataset: str) -> tuple[Path, Path]:
    daily, weekly = init_npy_paths(dataset)
    if daily.is_file() and weekly.is_file():
        return daily, weekly
    return run_initialization(dataset)


def build_train_command(dataset: str, gpus: str, cfg_override: Path | None = None) -> list[str]:
    cfg = cfg_override or cfg_path(dataset)
    return [
        sys.executable,
        str(TRAIN_PY),
        "-c",
        cfg_for_launch(cfg),
        "-g",
        str(gpus),
    ]


def parse_log_metrics(log_text: str) -> dict:
    val_maes = [float(x) for x in VAL_MAE_RE.findall(log_text)]
    best_val = min(val_maes) if val_maes else float("nan")

    best_test = {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    best_mae = None
    for block in TEST_BLOCK_RE.finditer(log_text):
        chunk = block.group(1)
        parsed = {}
        for key, pat in METRIC_RES.items():
            m = pat.search(chunk)
            parsed[key] = float(m.group(1)) if m else float("nan")
        if math.isnan(parsed["mae"]):
            continue
        if best_mae is None or parsed["mae"] < best_mae:
            best_mae = parsed["mae"]
            best_test = parsed

    epoch = float("nan")
    epoch_matches = re.findall(r"Epoch\s+(\d+)\s+/\s+\d+", log_text)
    if val_maes and epoch_matches:
        # approximate epoch of best val
        best_idx = val_maes.index(min(val_maes))
        if best_idx < len(epoch_matches):
            epoch = float(epoch_matches[best_idx])

    return {
        "val_mae_best": best_val,
        "test_mae_at_best_val": best_test["mae"],
        "test_rmse": best_test["rmse"],
        "test_mape": best_test["mape"],
        "epoch": epoch,
    }


def write_dataset_result(dataset: str, row: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = RESULT_DIR / f"hyperd_{dataset.lower()}.md"
    csv_path = RESULT_DIR / f"hyperd_{dataset.lower()}.csv"

    headers = [
        "dataset",
        "model",
        "input_len",
        "output_len",
        "val_mae_best",
        "test_mae_at_best_val",
        "test_rmse",
        "test_mape",
        "epoch",
        "config",
        "log_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# HyperD {dataset} (12->12)\n\n")
        f.write("| field | value |\n|---|---|\n")
        for key in headers:
            f.write(f"| {key} | {row.get(key, '')} |\n")


def write_summary(rows: list[dict]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else []
    summary_csv = RESULT_DIR / "summary.csv"
    summary_md = RESULT_DIR / "summary.md"

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# HyperD baseline summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")


def run_dataset(dataset: str, gpus: str, dry_run: bool, skip_train: bool) -> dict:
    dataset = dataset.upper()
    missing = missing_requirements(dataset)
    if missing:
        raise FileNotFoundError("\n".join(missing))

    ensure_hyperd_data(dataset)
    daily, weekly = ensure_inits(dataset)
    cmd = build_train_command(dataset, gpus)

    row = {
        "dataset": dataset,
        "model": "HyperD",
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "val_mae_best": "",
        "test_mae_at_best_val": "",
        "test_rmse": "",
        "test_mape": "",
        "epoch": "",
        "config": cfg_for_launch(cfg_path(dataset)),
        "log_path": "",
        "daily_init": str(daily),
        "weekly_init": str(weekly),
        "command": " ".join(cmd),
    }

    if dry_run or skip_train:
        print(f"[HyperD][{dataset}] daily_init={daily}")
        print(f"[HyperD][{dataset}] weekly_init={weekly}")
        print(f"[HyperD][{dataset}] command: {' '.join(cmd)}")
        return row

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"hyperd_{dataset.lower()}_train.log"
    row["log_path"] = str(log_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpus)
    print(f"[HyperD][{dataset}] training -> {log_path}")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"HyperD training failed for {dataset}, see {log_path}")

    ckpt_logs = list((ROOT / "checkpoints" / "baselines" / f"HyperD_{dataset}_12to12").glob("*/training_log_*.log"))
    log_text = log_path.read_text(errors="replace")
    if ckpt_logs:
        log_text += "\n" + ckpt_logs[-1].read_text(errors="replace")

    metrics = parse_log_metrics(log_text)
    row.update(metrics)
    write_dataset_result(dataset, row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HyperD baseline experiments")
    parser.add_argument("--datasets", nargs="+", default=["PEMS04"], help="PEMS03 PEMS04 PEMS07 PEMS08")
    parser.add_argument("--gpus", default="0", help="CUDA device id")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    parser.add_argument("--skip_train", action="store_true", help="Prepare data/init only")
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        name = dataset.upper()
        if name not in HYPERD_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        try:
            row = run_dataset(name, args.gpus, args.dry_run, args.skip_train)
            rows.append({k: v for k, v in row.items() if k not in {"command", "daily_init", "weekly_init"}})
        except FileNotFoundError as exc:
            print(f"[HyperD][{name}] SKIP: {exc}")

    if rows and not args.dry_run and not args.skip_train:
        write_summary(rows)
        print(f"[HyperD] summary -> {RESULT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
