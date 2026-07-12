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
        self.spatial_stage_loss_weights = list(
            param.get("spatial_stage_loss_weights", [0.0, 0.0, 1.0])
        )
        self.spatial_graph_loss_weights = list(
            param.get("spatial_graph_loss_weights", [0.0, 0.0, 0.0])
        )
        self.post_spatial_mode = str(param.get("post_spatial_mode", "")).lower()

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

    def _weighted_loss(self, pred: torch.Tensor, target: torch.Tensor, weight: float) -> torch.Tensor:
        if float(weight) == 0.0:
            return torch.tensor(0.0, device=target.device)
        pred_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(pred, **self.scaler["args"])
        target_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(target, **self.scaler["args"])
        return float(weight) * self.metric_forward(self.loss, [pred_rescaled, target_rescaled])

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
        final_target = targets[-1]

        loss = torch.tensor(0.0, device=real_value.device)
        if len(self.chain_loss_weights) > 1:
            for weight, pred, target in zip(
                self.chain_loss_weights[:-1], preds[:-1], targets[:-1]
            ):
                loss = loss + self._weighted_loss(pred, target, weight)

        loss = loss + self._weighted_loss(out["pred"], final_target, self.chain_loss_weights[-1])

        spatial_stage_preds = out.get("spatial_stage_preds") or []
        final_pred = out["pred"]
        if spatial_stage_preds and any(float(w) != 0.0 for w in self.spatial_stage_loss_weights):
            weights = list(self.spatial_stage_loss_weights)
            if len(weights) < len(spatial_stage_preds):
                weights = weights + [weights[-1]] * (len(spatial_stage_preds) - len(weights))
            weights = weights[: len(spatial_stage_preds)]
            for pred, weight in zip(spatial_stage_preds, weights):
                if pred is final_pred:
                    continue
                loss = loss + self._weighted_loss(pred, final_target, weight)

        graph_stage_preds = []
        graph_diag = out.get("graph_resolution_diagnostics") or {}
        if graph_diag:
            graph_stage_preds = graph_diag.get("node_stage_preds") or []
        if graph_stage_preds and any(float(w) != 0.0 for w in self.spatial_graph_loss_weights):
            g_weights = list(self.spatial_graph_loss_weights)
            if len(g_weights) < len(graph_stage_preds):
                g_weights = g_weights + [g_weights[-1]] * (len(graph_stage_preds) - len(g_weights))
            g_weights = g_weights[: len(graph_stage_preds)]
            for pred, weight in zip(graph_stage_preds, g_weights):
                if len(graph_stage_preds) > 1 and pred is final_pred:
                    continue
                loss = loss + self._weighted_loss(pred, final_target, weight)

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

    def on_validating_end(self, train_epoch=None):
        super().on_validating_end(train_epoch)
        post_mode = self.post_spatial_mode
        if post_mode != "adaptive_multiscale_only":
            return
        model = self.model
        if not hasattr(model, "spatial_module") and not hasattr(model, "graph_resolution_stack"):
            return
        weights = None
        topks = None
        alpha = None
        entropy = None
        if hasattr(model, "spatial_module") and model.spatial_module is not None:
            diag = model.spatial_module.get_adaptive_ms_diagnostics()
            weights = diag.get("adaptive_ms_weights")
            topks = diag.get("adaptive_ms_topks")
            alpha = diag.get("adaptive_ms_alpha")
            entropy = diag.get("adaptive_ms_entropy")
        elif (
            hasattr(model, "graph_resolution_stack")
            and model.graph_resolution_stack is not None
            and model.graph_resolution_stack.spatial_modules
        ):
            diag = model.graph_resolution_stack.spatial_modules[-1].get_adaptive_ms_diagnostics()
            weights = diag.get("adaptive_ms_weights")
            topks = diag.get("adaptive_ms_topks")
            alpha = diag.get("adaptive_ms_alpha")
            entropy = diag.get("adaptive_ms_entropy")
        if weights is None:
            return
        w_list = weights.detach().cpu().tolist()
        if isinstance(w_list, float):
            w_list = [w_list]
        topk_str = ",".join(str(k) for k in (topks or []))
        w_str = ",".join(f"{w:.4f}" for w in w_list)
        self.logger.info(
            f"[AdaptiveMS epoch={train_epoch}] topks=[{topk_str}] "
            f"weights=[{w_str}] alpha={alpha} entropy={entropy}"
        )
