from typing import Tuple, Union

import torch

from basicts.archs.arch_zoo.STForecastStateFlow_arch import STForecastStateFlow
from basicts.runners.base_tsf_runner import SCALER_REGISTRY
from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class STForecastStateFlowRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for ST-FSF with multi-component training loss."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        param = cfg["MODEL"]["PARAM"]
        self.stage_specs = list(param.get("stage_specs", []))
        if not self.stage_specs:
            self.stage_specs = STForecastStateFlow.build_stage_specs(
                output_len=param["output_len"],
                node_size=param["node_size"],
                q_ratio_1=param.get("q_ratio_1", 0.25),
                q_ratio_2=param.get("q_ratio_2", 0.50),
                q_list_override=param.get("q_list_override"),
                direct=param.get("direct_forecast", False),
            )
        self.final_loss_weight = float(param.get("final_loss_weight", 1.0))
        self.state_loss_weight = float(param.get("state_loss_weight", 0.3))
        self.native_loss_weight = float(param.get("native_loss_weight", 0.1))
        self.fm_loss_weight = float(param.get("fm_loss_weight", 0.2))

    def _rescale(self, tensor: torch.Tensor) -> torch.Tensor:
        return SCALER_REGISTRY.get(self.scaler["func"])(tensor, **self.scaler["args"])

    def _rescale_velocity(self, vel: torch.Tensor) -> torch.Tensor:
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
            self._last_st_fsf_out = out
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

        out = self._last_st_fsf_out
        f_len = real_value.shape[1]
        v = self.model.graph_basis
        stage_specs = out.get("stage_specs", self.stage_specs)
        loss = torch.tensor(0.0, device=real_value.device)

        loss = loss + self.final_loss_weight * self._metric_loss(out["pred"], real_value)

        for k, (r, q) in enumerate(stage_specs):
            state_target = STForecastStateFlow.st_target(real_value, v, r, q, f_len)
            loss = loss + self.state_loss_weight * self._metric_loss(out["states"][k], state_target)

        if self.native_loss_weight > 0.0:
            for k, (r, q) in enumerate(stage_specs):
                native_target = STForecastStateFlow.native_target(real_value, v, r, q)
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
