"""Dedicated runner for ChainForecasting multi-level loss."""

from __future__ import annotations

from typing import Tuple, Union

import torch

from basicts.archs.arch_zoo.ChainForecasting_arch import ChainForecasting
from ...data import SCALER_REGISTRY
from ..runner_zoo.simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class ChainForecastingRunner(SimpleTimeSeriesForecastingRunner):
    """Runner with external multi-level chain loss during training."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        param = cfg["MODEL"]["PARAM"]
        self.chain_lengths = list(param.get("chain_lengths", [3, 6, 12]))
        self.chain_loss_weights = list(param.get("chain_loss_weights", [0.2, 0.3, 1.0]))
        if len(self.chain_loss_weights) != len(self.chain_lengths):
            raise ValueError(
                "chain_loss_weights length must match chain_lengths: "
                f"{len(self.chain_loss_weights)} vs {len(self.chain_lengths)}"
            )

    def _model_forward_train(self, history_data, future_data_dec, iter_num, epoch):
        return self.model(
            history_data=history_data,
            future_data=future_data_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=True,
            return_all=True,
        )

    def _model_forward_eval(self, history_data, future_data_dec, iter_num, epoch, train: bool):
        return self.model(
            history_data=history_data,
            future_data=future_data_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train,
            return_all=False,
        )

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
        future_data_dec = self.select_input_features(future_data)

        if train:
            out = self._model_forward_train(history_data, future_data_dec, iter_num, epoch)
            prediction_data = out["pred"]
            self._last_chain_out = out
        else:
            prediction_data = self._model_forward_eval(
                history_data, future_data_dec, iter_num, epoch, train=train
            )
            self._last_chain_out = None

        assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes]
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value

    def _rescale(self, tensor: torch.Tensor) -> torch.Tensor:
        return SCALER_REGISTRY.get(self.scaler["func"])(tensor, **self.scaler["args"])

    def train_iters(
        self,
        epoch: int,
        iter_index: int,
        data: Union[torch.Tensor, Tuple],
    ) -> torch.Tensor:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        future_data, history_data = data
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)

        history_data = self.select_input_features(history_data)
        future_data_dec = self.select_input_features(future_data)

        out = self._model_forward_train(history_data, future_data_dec, iter_num, epoch)
        real_value = self.select_target_features(future_data)
        targets = ChainForecasting.build_chain_targets(real_value, self.chain_lengths)

        total_loss = torch.tensor(0.0, device=history_data.device)
        forward_pairs = []
        num_levels = len(self.chain_lengths)
        for level_idx, (weight, target) in enumerate(zip(self.chain_loss_weights, targets)):
            if level_idx == num_levels - 1:
                pred_scaled = self._rescale(out["pred"])
            else:
                pred_scaled = self._rescale(out["chain_preds"][level_idx])
            tgt_scaled = self._rescale(target)
            level_loss = self.metric_forward(self.loss, [pred_scaled, tgt_scaled])
            total_loss = total_loss + float(weight) * level_loss
            forward_pairs.append((pred_scaled, tgt_scaled))

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_pairs[-1])
            self.update_epoch_meter("train_" + metric_name, metric_item.item())

        return total_loss
