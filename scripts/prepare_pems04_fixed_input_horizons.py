#!/usr/bin/env python3
"""Prepare PeMS04 fixed-input (H=12) multi-horizon datasets for Protocol A."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "PEMS04"
GENERATE_SCRIPT = ROOT / "scripts" / "data_preparation" / "PEMS04" / "generate_holost_data.py"


def data_paths(input_len: int, output_len: int) -> dict[str, Path]:
    stem = f"in{input_len}_out{output_len}"
    return {
        "data": DATASET_DIR / f"data_{stem}.pkl",
        "index": DATASET_DIR / f"index_{stem}.pkl",
        "scaler": DATASET_DIR / f"scaler_{stem}.pkl",
        "npz": DATASET_DIR / f"data_{stem}.npz",
    }


def load_index(index_path: Path) -> dict:
    with open(index_path, "rb") as f:
        return pickle.load(f)


def split_stats(index: dict) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for split in ("train", "valid", "test"):
        entries = index[split]
        out[split] = (len(entries), entries[0], entries[-1])
    return out


def verify_horizon_files(input_len: int, output_len: int) -> tuple[bool, str]:
    paths = data_paths(input_len, output_len)
    missing = [name for name, path in paths.items() if name != "npz" and not path.is_file()]
    if missing:
        return False, f"missing {missing}"
    index = load_index(paths["index"])
    for split in ("train", "valid", "test"):
        if split not in index or len(index[split]) == 0:
            return False, f"empty split {split}"
    with open(paths["data"], "rb") as f:
        data = pickle.load(f)
    if data["processed_data"].shape[0] <= 0:
        return False, "empty processed_data"
    return True, "ok"


def generate_horizon(input_len: int, output_len: int, force: bool) -> None:
    if not force:
        ok, _ = verify_horizon_files(input_len, output_len)
        if ok:
            print(f"[skip] H={input_len} F={output_len}: existing files OK, not overwriting.")
            return
    cmd = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "--history_seq_len",
        str(input_len),
        "--future_seq_len",
        str(output_len),
        "--output_dir",
        str(DATASET_DIR),
    ]
    print(f"[gen] H={input_len} F={output_len}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def verify_batch_shapes(input_len: int, output_len: int, batch_size: int = 32) -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from basicts.data import TimeSeriesForecastingDataset

    paths = data_paths(input_len, output_len)
    dataset = TimeSeriesForecastingDataset(
        data_file_path=str(paths["data"]),
        index_file_path=str(paths["index"]),
        mode="test",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    future_data, history_data = next(iter(loader))
    return {
        "history_shape": tuple(history_data.shape),
        "future_shape": tuple(future_data.shape),
        "num_test_windows": len(dataset),
    }


def build_inspection_row(input_len: int, output_len: int) -> dict:
    paths = data_paths(input_len, output_len)
    index = load_index(paths["index"])
    stats = split_stats(index)
    train_n, first_idx, last_idx = stats["train"]
    _, _, test_last = stats["test"]
    batch = verify_batch_shapes(input_len, output_len)
    return {
        "horizon": output_len,
        "input_len": input_len,
        "output_len": output_len,
        "train_windows": train_n,
        "val_windows": stats["valid"][0],
        "test_windows": stats["test"][0],
        "data_path": str(paths["data"]),
        "index_path": str(paths["index"]),
        "first_index": str(first_idx),
        "last_index": str(test_last),
        "history_shape": batch["history_shape"],
        "future_shape": batch["future_shape"],
    }


def _fmt_shape(v):
    return str(v) if not isinstance(v, str) else v


def write_inspection_table(rows: list[dict], out_csv: Path) -> None:
    fields = [
        "horizon", "input_len", "output_len",
        "train_windows", "val_windows", "test_windows",
        "data_path", "index_path", "first_index", "last_index",
        "history_shape", "future_shape",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row["history_shape"] = _fmt_shape(out_row["history_shape"])
            out_row["future_shape"] = _fmt_shape(out_row["future_shape"])
            writer.writerow(out_row)


def print_inspection_table(rows: list[dict]) -> None:
    print("\n=== Protocol A data inspection ===")
    for row in rows:
        print(
            f"F={row['output_len']:>2}: train/val/test={row['train_windows']}/"
            f"{row['val_windows']}/{row['test_windows']} | "
            f"history={_fmt_shape(row['history_shape'])} future={_fmt_shape(row['future_shape'])}"
        )
        print(f"     data : {row['data_path']}")
        print(f"     index: {row['index_path']}")
        print(f"     first={row['first_index']} last={row['last_index']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare PeMS04 H=12, F∈{12,24,48} datasets.")
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--horizons", type=int, nargs="+", default=[12, 24, 48])
    parser.add_argument("--force", action="store_true", help="Regenerate even if files exist.")
    parser.add_argument(
        "--out",
        default="results/pems04_fixed_input_horizon_data_check.csv",
        help="Inspection table CSV path.",
    )
    args = parser.parse_args()

    if not GENERATE_SCRIPT.is_file():
        print(f"Missing generator: {GENERATE_SCRIPT}")
        return 1

    for horizon in args.horizons:
        generate_horizon(args.input_len, horizon, args.force)

    rows: list[dict] = []
    for horizon in args.horizons:
        ok, msg = verify_horizon_files(args.input_len, horizon)
        if not ok:
            print(f"ERROR: F={horizon} verification failed: {msg}")
            return 2
        row = build_inspection_row(args.input_len, horizon)
        if row["history_shape"][1] != args.input_len:
            print(f"ERROR: F={horizon} history len {row['history_shape'][1]} != {args.input_len}")
            return 2
        if row["future_shape"][1] != horizon:
            print(f"ERROR: F={horizon} future len {row['future_shape'][1]} != {horizon}")
            return 2
        rows.append(row)

    out_csv = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    write_inspection_table(rows, out_csv)
    print_inspection_table(rows)
    print(f"\nWrote inspection table: {out_csv}")
    print(
        "Note: windows are pre-generated index slices (BasicTS official 6:2:2 on window count). "
        "Existing in12_out12 is preserved; in12_out24/out48 are added alongside."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
