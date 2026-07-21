from __future__ import annotations

from typing import Tuple, Union

import torch
from easytorch.utils.dist import master_only

from basicts.runners.grad_surgery import compute_grad_surgery, trainable_parameters
from basicts.runners.runner_zoo.chain_forecasting_runner import ChainForecastingRunner
from basicts.runners.stagewise_training import temporal_downsample_target
from basicts.runners.base_tsf_runner import SCALER_REGISTRY


class G1FinalPrimaryGradSurgeryRunner(ChainForecastingRunner):
    """End-to-end G1 training with final-primary gradient surgery on T1/T2 aux losses."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        param = cfg["MODEL"]["PARAM"]
        self.final_primary_grad_surgery = bool(param.get("final_primary_grad_surgery", False))
        self.aux_grad_max_ratio = float(param.get("aux_grad_max_ratio", 0.2))
        if not self.final_primary_grad_surgery:
            raise ValueError(
                "G1FinalPrimaryGradSurgeryRunner requires final_primary_grad_surgery=True"
            )
        if self.stagewise_enabled:
            raise ValueError("G1_final_primary_grad_surgery does not support stagewise training")
        if self.unified_aux_loss_mode != "none":
            raise ValueError("G1_final_primary_grad_surgery requires unified_aux_loss_mode='none'")
        placement = str(param.get("spatial_placement", "")).lower()
        if placement != "final":
            raise ValueError(
                f"G1_final_primary_grad_surgery requires spatial_placement='final', got {placement!r}"
            )
        self._surgery_epoch_stats: dict[str, list[float | bool]] = {}
        self._best_test_mae: float | None = None
        self._best_test_rmse: float | None = None

    def init_training(self, cfg: dict):
        super().init_training(cfg)
        self.logger.info(
            "[G1_final_primary_grad_surgery] enabled=True aux_grad_max_ratio=%.2f "
            "L_final=primary objective T1/T2=auxiliary gradient surgery (no weighted sum loss)",
            self.aux_grad_max_ratio,
        )

    def _reset_surgery_epoch_stats(self) -> None:
        self._surgery_epoch_stats = {
            "L_final": [],
            "L_T1": [],
            "L_T2": [],
            "cos_g_T1_g_final": [],
            "cos_g_T2_g_final": [],
            "T1_projected": [],
            "T2_projected": [],
            "final_grad_norm": [],
            "aux_grad_norm": [],
        }

    def _accumulate_surgery_stats(self, stats: dict) -> None:
        for key in self._surgery_epoch_stats:
            if key in stats:
                self._surgery_epoch_stats[key].append(stats[key])

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(sum(values) / max(len(values), 1))

    @staticmethod
    def _bool_rate(values: list[bool]) -> float:
        if not values:
            return 0.0
        return float(sum(1 for v in values if v) / len(values))

    def on_epoch_start(self, epoch: int):
        self._reset_surgery_epoch_stats()

    def on_epoch_end(self, epoch: int):
        super().on_epoch_end(epoch)
        stats = self._surgery_epoch_stats
        if not stats["L_final"]:
            return
        val_mae = self.meter_pool.get_avg("val_MAE")
        self.logger.info(
            "[G1_final_primary_grad_surgery epoch=%s] L_final=%.4f L_T1=%.4f L_T2=%.4f "
            "cos(g_T1,g_final)=%.4f cos(g_T2,g_final)=%.4f "
            "T1_projected_rate=%.3f T2_projected_rate=%.3f "
            "aux_grad_norm/final_grad_norm=%.4f val_MAE=%.4f",
            epoch,
            self._mean(stats["L_final"]),
            self._mean(stats["L_T1"]),
            self._mean(stats["L_T2"]),
            self._mean(stats["cos_g_T1_g_final"]),
            self._mean(stats["cos_g_T2_g_final"]),
            self._bool_rate(stats["T1_projected"]),
            self._bool_rate(stats["T2_projected"]),
            self._mean(stats["aux_grad_norm"]) / max(self._mean(stats["final_grad_norm"]), 1e-8),
            float(val_mae) if val_mae is not None else float("nan"),
        )

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
        out = self._last_chain_out

        pred_final = out.get("pred")
        pred_t_low = out.get("pred_T_low")
        pred_t_mid = out.get("pred_T_mid")
        if pred_final is None or pred_t_low is None or pred_t_mid is None:
            raise RuntimeError(
                "G1_final_primary_grad_surgery requires pred, pred_T_low, pred_T_mid from forward"
            )

        target_full = real_value
        target_t1 = temporal_downsample_target(real_value, self.chain_lengths[0])
        target_t2 = temporal_downsample_target(real_value, self.chain_lengths[1])

        l_final = self._raw_loss(pred_final, target_full)
        l_t1 = self._raw_loss(pred_t_low, target_t1)
        l_t2 = self._raw_loss(pred_t_mid, target_t2)

        self._pending_grad_surgery = {
            "losses": {"L_final": l_final, "L_T1": l_t1, "L_T2": l_t2},
        }

        pred_final_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(
            forward_return[0], **self.scaler["args"]
        )
        real_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(
            forward_return[1], **self.scaler["args"]
        )
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [pred_final_rescaled, real_rescaled])
            self.update_epoch_meter("train_" + metric_name, metric_item.item())

        return l_final

    def backward(self, loss: torch.Tensor):
        del loss
        self.optim.zero_grad()
        model = self.model.module if hasattr(self.model, "module") else self.model
        params = trainable_parameters(model)
        pending = getattr(self, "_pending_grad_surgery", None)
        if pending is None:
            raise RuntimeError("Missing pending grad surgery state in backward")

        combined_grads, stats = compute_grad_surgery(
            pending["losses"],
            params,
            aux_grad_max_ratio=self.aux_grad_max_ratio,
        )
        self._accumulate_surgery_stats(stats)
        self._pending_grad_surgery = None

        grad_idx = 0
        for param in params:
            param.grad = combined_grads[grad_idx]
            grad_idx += 1

        if self.clip_grad_param is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self.clip_grad_param)
        self.optim.step()

    @master_only
    def save_best_model(self, epoch: int, metric_name: str, greater_best: bool = True):
        super().save_best_model(epoch, metric_name, greater_best)

    def on_test_end(self):
        super().on_test_end()
        self._best_test_mae = float(self.meter_pool.get_avg("test_MAE"))
        self._best_test_rmse = float(self.meter_pool.get_avg("test_RMSE"))

    def on_training_end(self):
        super().on_training_end()
        best_val = self.best_metrics.get("val_MAE")
        self.logger.info(
            "[G1_final_primary_grad_surgery FINAL] best_val_MAE=%.4f "
            "test_MAE@best-val=%.4f test_RMSE@best-val=%.4f",
            float(best_val if best_val is not None else float("nan")),
            float(self._best_test_mae or float("nan")),
            float(self._best_test_rmse or float("nan")),
        )
