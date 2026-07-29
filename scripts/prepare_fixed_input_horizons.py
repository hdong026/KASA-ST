#!/usr/bin/env python3
"""Prepare fixed-input (H=12) multi-horizon datasets for Protocol A (any supported dataset)."""
from __future__ import annotations

import argparse
import csv
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET_SPECS = {
    "PEMS04": {
        "slug": "pems04",
        "generate_script": ROOT / "scripts" / "data_preparation" / "PEMS04" / "generate_holost_data.py",
        "train_ratio": 0.6,
        "valid_ratio": 0.2,
    },
    "PEMS07": {
        "slug": "pems07",
        "generate_script": ROOT / "scripts" / "data_preparation" / "PEMS07" / "generate_holost_data.py",
        "train_ratio": 0.6,
        "valid_ratio": 0.2,
    },
    "PEMS08": {
        "slug": "pems08",
        "generate_script": ROOT / "scripts" / "data_preparation" / "PEMS08" / "generate_holost_data.py",
        "train_ratio": 0.6,
        "valid_ratio": 0.2,
    },
    "PEMS-BAY": {
        "slug": "pems_bay",
        "generate_script": ROOT / "scripts" / "data_preparation" / "PEMS-BAY" / "generate_holost_data.py",
        "train_ratio": 0.6,
        "valid_ratio": 0.2,
    },
    "PEMS03": {
        "slug": "pems03",
        "generate_script": ROOT / "scripts" / "data_preparation" / "PEMS03" / "generate_holost_data.py",
        "train_ratio": 0.6,
        "valid_ratio": 0.2,
    },
    "KnowAir": {
        "slug": "knowair",
        "generate_script": ROOT / "scripts" / "data_preparation" / "KnowAir" / "generate_holost_data.py",
        "train_ratio": 0.6,
        "valid_ratio": 0.2,
    },
}


def dataset_dir(dataset: str) -> Path:
    return ROOT / "datasets" / dataset


def data_paths(dataset: str, input_len: int, output_len: int) -> dict[str, Path]:
    stem = f"in{input_len}_out{output_len}"
    d = dataset_dir(dataset)
    return {
        "data": d / f"data_{stem}.pkl",
        "index": d / f"index_{stem}.pkl",
        "scaler": d / f"scaler_{stem}.pkl",
        "npz": d / f"data_{stem}.npz",
    }


def load_index(index_path: Path) -> dict:
    with open(index_path, "rb") as f:
        return pickle.load(f)


def split_stats(index: dict) -> dict[str, tuple]:
    out = {}
    for split in ("train", "valid", "test"):
        entries = index[split]
        out[split] = (len(entries), entries[0], entries[-1])
    return out


def verify_horizon_files(dataset: str, input_len: int, output_len: int) -> tuple[bool, str]:
    paths = data_paths(dataset, input_len, output_len)
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


def generate_horizon(dataset: str, input_len: int, output_len: int, force: bool) -> None:
    spec = DATASET_SPECS[dataset]
    if not force:
        ok, _ = verify_horizon_files(dataset, input_len, output_len)
        if ok:
            print(f"[skip] {dataset} H={input_len} F={output_len}: existing files OK, not overwriting.")
            return
    cmd = [
        sys.executable,
        str(spec["generate_script"]),
        "--history_seq_len",
        str(input_len),
        "--future_seq_len",
        str(output_len),
        "--output_dir",
        str(dataset_dir(dataset)),
        "--train_ratio",
        str(spec["train_ratio"]),
        "--valid_ratio",
        str(spec["valid_ratio"]),
    ]
    print(f"[gen] {dataset} H={input_len} F={output_len}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def verify_batch_shapes(dataset: str, input_len: int, output_len: int, batch_size: int = 32) -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from torch.utils.data import DataLoader
    from basicts.data import TimeSeriesForecastingDataset

    paths = data_paths(dataset, input_len, output_len)
    ds = TimeSeriesForecastingDataset(
        data_file_path=str(paths["data"]),
        index_file_path=str(paths["index"]),
        mode="test",
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    future_data, history_data = next(iter(loader))
    return {
        "history_shape": tuple(history_data.shape),
        "future_shape": tuple(future_data.shape),
        "num_test_windows": len(ds),
    }


def build_inspection_row(dataset: str, input_len: int, output_len: int) -> dict:
    paths = data_paths(dataset, input_len, output_len)
    index = load_index(paths["index"])
    stats = split_stats(index)
    train_n, first_idx, _ = stats["train"]
    _, _, test_last = stats["test"]
    batch = verify_batch_shapes(dataset, input_len, output_len)
    return {
        "dataset": dataset,
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
    return str(v)


def write_inspection_table(rows: list[dict], out_csv: Path) -> None:
    fields = [
        "dataset", "horizon", "input_len", "output_len",
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
            f"{row['dataset']} F={row['output_len']:>2}: train/val/test="
            f"{row['train_windows']}/{row['val_windows']}/{row['test_windows']} | "
            f"history={_fmt_shape(row['history_shape'])} future={_fmt_shape(row['future_shape'])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare H=12 multi-horizon datasets (6:2:2).")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_SPECS.keys()))
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--horizons", type=int, nargs="+", default=[12, 24, 48])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]
    if not spec["generate_script"].is_file():
        print(f"Missing generator: {spec['generate_script']}")
        return 1

    for horizon in args.horizons:
        generate_horizon(args.dataset, args.input_len, horizon, args.force)

    rows: list[dict] = []
    for horizon in args.horizons:
        ok, msg = verify_horizon_files(args.dataset, args.input_len, horizon)
        if not ok:
            print(f"ERROR: {args.dataset} F={horizon} verification failed: {msg}")
            return 2
        row = build_inspection_row(args.dataset, args.input_len, horizon)
        if row["history_shape"][1] != args.input_len:
            print(f"ERROR: history len mismatch for F={horizon}")
            return 2
        if row["future_shape"][1] != horizon:
            print(f"ERROR: future len mismatch for F={horizon}")
            return 2
        rows.append(row)

    out_default = f"results/{spec['slug']}_fixed_input_horizon_data_check.csv"
    out_csv = ROOT / (args.out or out_default)
    if Path(args.out or out_default).is_absolute():
        out_csv = Path(args.out or out_default)
    write_inspection_table(rows, out_csv)
    print_inspection_table(rows)
    print(f"\nWrote inspection table: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
