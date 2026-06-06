import os
import random

import numpy as np
import torch

from ...utils.temporal_features import adapt_dow_as_class_index
from ..base_tsf_runner import BaseTimeSeriesForecastingRunner


class HimNetRunner(BaseTimeSeriesForecastingRunner):
    """Runner for HimNet (ported from BasicTS eb65f4b to v0.2 tuple forward API)."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)
        self.seed_everything(0)

    @staticmethod
    def seed_everything(seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        future_data, history_data = data
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = adapt_dow_as_class_index(self.select_input_features(history_data))
        future_data_4_dec = adapt_dow_as_class_index(self.select_input_features(future_data))

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        x = history_data
        y_true = future_data_4_dec[..., 0:1]
        y_cov = future_data_4_dec[..., 1:]

        if train:
            prediction_data = self.model(x, y_cov, y_true, iter_num + 1)
        else:
            prediction_data = self.model(x, y_cov)

        assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes]

        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value
