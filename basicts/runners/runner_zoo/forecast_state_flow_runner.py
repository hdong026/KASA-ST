from typing import Tuple, Union

import torch

from basicts.archs.arch_zoo.ForecastStateFlow_arch import ForecastStateFlow
from basicts.runners.base_tsf_runner import SCALER_REGISTRY
from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class ForecastStateFlowRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for Forecast-State Flow Chain with multi-component training loss."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        param = cfg["MODEL"]["PARAM"]
        self.chain_lengths = list(param.get("chain_lengths", [3, 6, 12]))
        self.final_loss_weight = float(param.get("final_loss_weight", 1.0))
        self.state_loss_weight = float(param.get("state_loss_weight", 0.3))
        self.native_loss_weight = float(param.get("native_loss_weight", 0.1))
        self.fm_loss_weight = float(param.get("fm_loss_weight", 0.2))

    def _rescale(self, tensor: torch.Tensor) -> torch.Tensor:
        return SCALER_REGISTRY.get(self.scaler["func"])(tensor, **self.scaler["args"])

    def _rescale_velocity(self, vel: torch.Tensor) -> torch.Tensor:
        """Rescale a difference quantity (Sb - Sa) under standard normalization."""
        std = self.scaler["args"].get("std", 1.0)
        if isinstance(std, torch.Tensor):
            return vel * std
        return vel * float(std)

    def _metric_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_r = self._rescale(pred)
        target_r = self._rescale(target)
        return self.metric_forward(self.loss, [pred_r, target_r])

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

        if train:
            out = self.model(
                history_data=history_data,
                future_data=future_data_4_dec,
                batch_seen=iter_num,
                epoch=epoch,
                train=train,
                return_all=True,
            )
            self._last_fsf_out = out
            prediction_data = out["pred"]
        else:
            prediction_data = self.model(
                history_data=history_data,
                future_data=future_data_4_dec,
                batch_seen=iter_num,
                epoch=epoch,
                train=train,
                return_all=False,
            )

        assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes], (
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        )
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value

    def train_iters(
        self,
        epoch: int,
        iter_index: int,
        data: Union[torch.Tensor, Tuple],
    ) -> torch.Tensor:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = list(self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True))

        future_data, _ = data
        future_data = self.to_running_device(future_data)
        real_value = self.select_target_features(future_data)

        out = self._last_fsf_out
        f_len = real_value.shape[1]
        loss = torch.tensor(0.0, device=real_value.device)

        final_pred = out["pred"]
        loss = loss + self.final_loss_weight * self._metric_loss(final_pred, real_value)

        for k, r in enumerate(self.chain_lengths):
            state_target = ForecastStateFlow.state_target(real_value, r, f_len)
            loss = loss + self.state_loss_weight * self._metric_loss(out["states"][k], state_target)

        if self.native_loss_weight > 0.0:
            for k, r in enumerate(self.chain_lengths):
                native_target = ForecastStateFlow.pool_target(real_value, r)
                loss = loss + self.native_loss_weight * self._metric_loss(
                    out["native_states"][k], native_target
                )

        if self.fm_loss_weight > 0.0 and out.get("fm_items"):
            for item in out["fm_items"]:
                vel_pred_r = self._rescale_velocity(item["vel_pred"])
                vel_target_r = self._rescale_velocity(item["vel_target"])
                loss = loss + self.fm_loss_weight * self.metric_forward(
                    self.loss, [vel_pred_r, vel_target_r]
                )

        pred_final_rescaled = self._rescale(forward_return[0])
        real_rescaled = self._rescale(forward_return[1])
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [pred_final_rescaled, real_rescaled])
            self.update_epoch_meter("train_" + metric_name, metric_item.item())

        return loss
