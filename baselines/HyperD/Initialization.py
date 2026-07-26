"""Statistical prior initialization for HyperD daily/weekly embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.HyperD.data_prepare import (
    data_dat_path,
    desc_json_path,
    ensure_hyperd_data,
    init_npy_paths,
    missing_requirements,
)
from baselines.HyperD.hyperd_settings import DATASET_META


def statistical_prior_initialization(
    dataset_name: str,
    num_nodes: int,
    shape: tuple[int, int, int],
    daily_len: int,
    weekly_len: int,
    train_val_test_ratio: list[float],
) -> tuple[Path, Path]:
    data_file_path = data_dat_path(dataset_name)
    data = np.memmap(data_file_path, dtype="float32", mode="r", shape=shape)

    total_len = len(data)
    valid_len = int(total_len * train_val_test_ratio[1])
    test_len = int(total_len * train_val_test_ratio[2])
    train_len = total_len - valid_len - test_len

    train_data = np.array(data[:train_len])
    flow_data = train_data[:, :, 0]
    train_mean = np.mean(flow_data)
    train_std = np.std(flow_data)
    if train_std < 1e-6:
        train_std = 1.0
    flow_data = (flow_data - train_mean) / train_std

    time_of_day = train_data[:, 0, 1] * daily_len
    time_of_week = train_data[:, 0, 1] * daily_len + train_data[:, 0, 2] * weekly_len

    daily_init = np.zeros((daily_len, num_nodes), dtype=np.float32)
    for t in range(daily_len):
        idx = time_of_day == t
        if np.any(idx):
            daily_init[t] = flow_data[idx].mean(axis=0)

    weekly_init = np.zeros((weekly_len, num_nodes), dtype=np.float32)
    for t in range(weekly_len):
        idx = time_of_week == t
        if np.any(idx):
            weekly_init[t] = flow_data[idx].mean(axis=0)

    daily_path, weekly_path = init_npy_paths(dataset_name)
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(daily_path, daily_init)
    np.save(weekly_path, weekly_init)
    return daily_path, weekly_path


def run_initialization(dataset_name: str, daily_len: int = 288, weekly_len: int = 288 * 7) -> tuple[Path, Path]:
    dataset_name = dataset_name.upper()
    missing = missing_requirements(dataset_name)
    if missing:
        raise FileNotFoundError("\n".join(missing))

    ensure_hyperd_data(dataset_name)
    with open(desc_json_path(dataset_name), "r", encoding="utf-8") as f:
        description = json.load(f)

    ratio = description["regular_settings"]["TRAIN_VAL_TEST_RATIO"]
    return statistical_prior_initialization(
        dataset_name,
        description["num_nodes"],
        tuple(description["shape"]),
        daily_len,
        weekly_len,
        ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical prior initialization for HyperD learnable periodic embeddings",
    )
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="PEMS03/04/07/08")
    parser.add_argument("--daily_len", type=int, default=288)
    parser.add_argument("--weekly_len", type=int, default=288 * 7)
    args = parser.parse_args()

    daily_path, weekly_path = run_initialization(args.dataset_name, args.daily_len, args.weekly_len)
    print(f"[HyperD init] saved {daily_path}")
    print(f"[HyperD init] saved {weekly_path}")


if __name__ == "__main__":
    main()
