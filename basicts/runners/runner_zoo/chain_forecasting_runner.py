from typing import Tuple, Union

import torch

from basicts.archs.arch_zoo.ChainForecasting_arch import ChainForecasting
from basicts.runners.base_tsf_runner import SCALER_REGISTRY
from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class ChainForecastingRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for ChainForecasting with multi-scale chain training loss."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        param = cfg["MODEL"]["PARAM"]
        self.chain_lengths = list(param.get("chain_lengths", [3, 6, 12]))
        self.chain_loss_weights = list(param.get("chain_loss_weights", [0.2, 0.3, 1.0]))
        if len(self.chain_lengths) != len(self.chain_loss_weights):
            raise ValueError("chain_lengths and chain_loss_weights must have the same length.")

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
            self._last_chain_out = out
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

        future_data, history_data = data
        future_data = self.to_running_device(future_data)
        real_value = self.select_target_features(future_data)

        out = self._last_chain_out
        preds = out["chain_preds"]
        targets = [ChainForecasting.pool_target(real_value, k) for k in self.chain_lengths]

        loss = torch.tensor(0.0, device=real_value.device)
        if len(self.chain_loss_weights) > 1:
            for weight, pred, target in zip(
                self.chain_loss_weights[:-1], preds[:-1], targets[:-1]
            ):
                pred_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(pred, **self.scaler["args"])
                target_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(target, **self.scaler["args"])
                loss = loss + float(weight) * self.metric_forward(
                    self.loss, [pred_rescaled, target_rescaled]
                )

        final_weight = float(self.chain_loss_weights[-1])
        final_pred = out["pred"]
        final_target = targets[-1]
        final_pred_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(final_pred, **self.scaler["args"])
        final_target_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(final_target, **self.scaler["args"])
        loss = loss + final_weight * self.metric_forward(
            self.loss, [final_pred_rescaled, final_target_rescaled]
        )

        pred_final_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(
            forward_return[0], **self.scaler["args"]
        )
        real_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(
            forward_return[1], **self.scaler["args"]
        )
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [pred_final_rescaled, real_rescaled])
            self.update_epoch_meter("train_" + metric_name, metric_item.item())

        return loss
