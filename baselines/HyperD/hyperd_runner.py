"""HyperD runner: DVA loss + official eval masking, isolated from KASA main runners."""

from __future__ import annotations

import math
import pickle
from typing import Tuple, Union

import numpy as np
import torch

from basicts.data import SCALER_REGISTRY
from basicts.runners.runner_zoo.simple_tsf_runner import SimpleTimeSeriesForecastingRunner

from baselines.HyperD.data_prepare import data_dat_path, desc_json_path, ensure_hyperd_data
from baselines.HyperD.dataset import HyperDTimeSeriesDataset
from baselines.HyperD.hyperd_settings import TRAIN_VAL_TEST_RATIO


class HyperDRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for official HyperD baseline (dual-view auxiliary loss)."""

    def __init__(self, cfg: dict):
        ensure_hyperd_scaler(cfg["DATASET_NAME"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        super().__init__(cfg)
        self._last_dual_view_loss: torch.Tensor | None = None

    @staticmethod
    def _build_hyperd_dataset(cfg: dict, mode: str) -> HyperDTimeSeriesDataset:
        ensure_hyperd_data(cfg["DATASET_NAME"])
        return HyperDTimeSeriesDataset(
            dataset_name=cfg["DATASET_NAME"],
            train_val_test_ratio=TRAIN_VAL_TEST_RATIO,
            mode=mode,
            input_len=int(cfg["DATASET_INPUT_LEN"]),
            output_len=int(cfg["DATASET_OUTPUT_LEN"]),
        )

    def build_train_dataset(self, cfg: dict):
        dataset = self._build_hyperd_dataset(cfg, "train")
        print("train len: {0}".format(len(dataset)))
        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        self.iter_per_epoch = math.ceil(len(dataset) / batch_size)
        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        dataset = HyperDRunner._build_hyperd_dataset(cfg, "valid")
        print("val len: {0}".format(len(dataset)))
        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        dataset = HyperDRunner._build_hyperd_dataset(cfg, "test")
        print("test len: {0}".format(len(dataset)))
        return dataset

    def forward(
        self,
        data: tuple,
        epoch: int = None,
        iter_num: int = None,
        train: bool = True,
        **kwargs,
    ) -> tuple:
        future_data, history_data = data
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec = future_data_4_dec.clone()
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        model_return = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train,
        )

        if isinstance(model_return, dict):
            self._last_dual_view_loss = model_return.get("dual_view_loss")
            prediction_data = model_return["prediction"]
        else:
            self._last_dual_view_loss = None
            prediction_data = model_return

        assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes], (
            "HyperD output must be [B, L, N, C]"
        )
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value

    def _rescaled_pair(self, forward_return: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pred = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
        real = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
        return pred, real

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
        pred, real = self._rescaled_pair(forward_return)

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            pred = pred[:, :cl_length, :, :]
            real = real[:, :cl_length, :, :]

        loss = self.metric_forward(self.loss, [pred, real])
        if self._last_dual_view_loss is not None:
            loss = loss + self._last_dual_view_loss

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [pred, real])
            self.update_epoch_meter("train_" + metric_name, metric_item.item())
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
        pred, real = self._rescaled_pair(forward_return)
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [pred, real])
            self.update_epoch_meter("val_" + metric_name, metric_item.item())


def ensure_hyperd_scaler(dataset_name: str, input_len: int, output_len: int) -> str:
    """Write scaler pkl matching HyperD data.dat normalization (train-split z-score)."""
    import json

    data_dir = f"datasets/{dataset_name}"
    scaler_path = f"{data_dir}/scaler_in{input_len}_out{output_len}.pkl"
    if __import__("os").path.isfile(scaler_path):
        return scaler_path

    ensure_hyperd_data(dataset_name)
    with open(desc_json_path(dataset_name), "r", encoding="utf-8") as f:
        desc = json.load(f)

    shape = tuple(desc["shape"])
    data = np.memmap(data_dat_path(dataset_name), dtype="float32", mode="r", shape=shape)
    ratio = desc["regular_settings"]["TRAIN_VAL_TEST_RATIO"]
    total_len = len(data)
    valid_len = int(total_len * ratio[1])
    test_len = int(total_len * ratio[2])
    train_len = total_len - valid_len - test_len

    raw = np.load(f"datasets/raw_data/{dataset_name}/{dataset_name}.npz")["data"][..., 0]
    train_raw = raw[:train_len]
    mean = float(train_raw.mean())
    std = float(train_raw.std())
    if std < 1e-6:
        std = 1.0

    scaler = {
        "func": "re_standard_transform",
        "args": {"mean": mean, "std": std},
    }
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    return scaler_path
