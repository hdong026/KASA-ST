"""HyperD official time-step split dataset (adapted from ll121202/HyperD)."""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from torch.utils.data import Dataset

from baselines.HyperD.data_prepare import data_dat_path, desc_json_path


@dataclass
class HyperDTimeSeriesDataset(Dataset):
    dataset_name: str
    train_val_test_ratio: List[float]
    mode: str
    input_len: int
    output_len: int
    overlap: bool = False
    logger: logging.Logger | None = None

    def __post_init__(self) -> None:
        assert self.mode in {"train", "valid", "test"}, f"Invalid mode: {self.mode}"
        self.data_file_path = str(data_dat_path(self.dataset_name))
        self.description_file_path = str(desc_json_path(self.dataset_name))
        self.description = self._load_description()
        self.data = self._load_data()

    def _load_description(self) -> dict:
        with open(self.description_file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_data(self) -> np.ndarray:
        data = np.memmap(
            self.data_file_path,
            dtype="float32",
            mode="r",
            shape=tuple(self.description["shape"]),
        )

        total_len = len(data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len

        minimal_len = self.input_len + self.output_len
        split_len = {"train": train_len, "valid": valid_len, "test": test_len}[self.mode]
        if minimal_len > split_len:
            self.overlap = True
            current_frame = inspect.currentframe()
            file_name = inspect.getfile(current_frame)
            line_number = current_frame.f_lineno - 6
            dataset = {"train": "Training", "valid": "Validation", "test": "Test"}[self.mode]
            msg = (
                f"{dataset} dataset is too short, enabling overlap. "
                f"See {file_name}:{line_number}."
            )
            if self.logger is not None:
                self.logger.info(msg)
            else:
                print(msg)

        if self.mode == "train":
            offset = self.output_len if self.overlap else 0
            return np.array(data[: train_len + offset])
        if self.mode == "valid":
            offset_left = self.input_len - 1 if self.overlap else 0
            offset_right = self.output_len if self.overlap else 0
            return np.array(data[train_len - offset_left : train_len + valid_len + offset_right])
        offset = self.input_len - 1 if self.overlap else 0
        return np.array(data[train_len + valid_len - offset :])

    def __getitem__(self, index: int) -> tuple:
        history_data = self.data[index : index + self.input_len]
        future_data = self.data[index + self.input_len : index + self.input_len + self.output_len]
        return future_data, history_data

    def __len__(self) -> int:
        return len(self.data) - self.input_len - self.output_len + 1
