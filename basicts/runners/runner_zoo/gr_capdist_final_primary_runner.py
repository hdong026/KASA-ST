from __future__ import annotations

from typing import Tuple, Union

import torch
from easytorch.utils.dist import master_only

from basicts.runners.grad_surgery import compute_grad_surgery, trainable_parameters
from basicts.runners.runner_zoo.chain_forecasting_runner import ChainForecastingRunner
from basicts.runners.stagewise_training import temporal_downsample_target
from basicts.runners.base_tsf_runner import SCALER_REGISTRY


GRAPH_AUX_LOSS_NAMES = ("L_G14", "L_G12")


class GRCapDistFinalPrimaryRunner(ChainForecastingRunner):
    """GR7_capdist_mix model with final-primary gradient surgery (temporal + graph aux)."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        param = cfg["MODEL"]["PARAM"]
        self.final_primary_grad_surgery = bool(param.get("final_primary_grad_surgery", False))
        self.aux_grad_max_ratio = float(param.get("aux_grad_max_ratio", 0.2))
        self.variant_name = str(param.get("variant_name", "GR_capdist_final_primary"))
        self.base_variant = str(param.get("base_variant", "GR7_capdist_mix"))
        if not self.final_primary_grad_surgery:
            raise ValueError(
                "GRCapDistFinalPrimaryRunner requires final_primary_grad_surgery=True"
            )
        if self.stagewise_enabled:
            raise ValueError("GR_capdist_final_primary does not support stagewise training")
        if self.unified_aux_loss_mode != "none":
            raise ValueError("GR_capdist_final_primary requires unified_aux_loss_mode='none'")
        placement = str(param.get("spatial_placement", "")).lower()
        if placement != "temporal_first_graph_resolution":
            raise ValueError(
                "GR_capdist_final_primary requires spatial_placement="
                "'temporal_first_graph_resolution'"
            )
        post_mode = str(param.get("post_spatial_mode", "")).lower()
        if post_mode != "adaptive_cluster_mix":
            raise ValueError(
                "GR_capdist_final_primary requires post_spatial_mode='adaptive_cluster_mix'"
            )
        self._surgery_epoch_stats: dict[str, list[float | bool]] = {}
        self._best_test_mae: float | None = None
        self._best_test_rmse: float | None = None
        self._aux_loss_names = ["L_T1", "L_T2", *list(GRAPH_AUX_LOSS_NAMES)]

    def init_training(self, cfg: dict):
        super().init_training(cfg)
        self.logger.info(
            "variant=%s base_variant=%s stagewise=disabled weighted_sum_loss=disabled "
            "final_primary_grad_projection=enabled graph_aux=enabled temporal_aux=enabled "
            "aux_grad_max_ratio=%.2f",
            self.variant_name,
            self.base_variant,
            self.aux_grad_max_ratio,
        )

    def _reset_surgery_epoch_stats(self) -> None:
        self._surgery_epoch_stats = {
            "L_final": [],
            "L_T1": [],
            "L_T2": [],
            "L_G14": [],
            "L_G12": [],
            "cos_g_T1_g_final": [],
            "cos_g_T2_g_final": [],
            "cos_G14_final": [],
            "cos_G12_final": [],
            "projected_rate": [],
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

    def _project_nodes(self, node_x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.einsum("mn,btnc->btmc", p, node_x)

    def _graph_stage_projected_loss(
        self,
        pred_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        projection: torch.Tensor,
    ) -> torch.Tensor:
        pred_proj = self._project_nodes(pred_nodes, projection)
        target_proj = self._project_nodes(target_nodes, projection)
        return self._raw_loss(pred_proj, target_proj)

    def _resolve_graph_aux(
        self,
        out: dict,
        target_full: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        graph_stage_preds = (
            out.get("graph_stage_preds")
            or out.get("graph_node_stage_preds")
            or (out.get("graph_resolution_diagnostics") or {}).get("node_stage_preds")
            or []
        )
        projection_matrices = (
            out.get("graph_projection_matrices")
            or (out.get("graph_resolution_diagnostics") or {}).get("graph_projection_matrices")
            or []
        )
        if len(graph_stage_preds) < 2:
            raise RuntimeError(
                "GR_capdist_final_primary requires >=2 graph stage preds for L_G14/L_G12"
            )
        if len(projection_matrices) < 2:
            raise RuntimeError(
                "GR_capdist_final_primary requires >=2 graph projection matrices"
            )

        graph_losses: dict[str, torch.Tensor] = {}
        aux_names = list(GRAPH_AUX_LOSS_NAMES)
        for idx, loss_name in enumerate(aux_names):
            if idx >= len(graph_stage_preds) - 1:
                break
            graph_losses[loss_name] = self._graph_stage_projected_loss(
                graph_stage_preds[idx],
                target_full,
                projection_matrices[idx],
            )
        return graph_losses, graph_stage_preds, projection_matrices

    def on_epoch_start(self, epoch: int):
        self._reset_surgery_epoch_stats()

    def on_epoch_end(self, epoch: int):
        val_mae = self.meter_pool.get_avg("val_MAE")
        super().on_epoch_end(epoch)
        stats = self._surgery_epoch_stats
        if not stats["L_final"]:
            return
        self.logger.info(
            "[%s epoch=%s] L_final=%.4f L_T1=%.4f L_T2=%.4f L_G14=%.4f L_G12=%.4f "
            "cos_T1_final=%.4f cos_T2_final=%.4f cos_G14_final=%.4f cos_G12_final=%.4f "
            "projected_ratio=%.3f aux_grad_norm/final_grad_norm=%.4f "
            "val_MAE=%.4f test_MAE@best-val=%.4f",
            self.variant_name,
            epoch,
            self._mean(stats["L_final"]),
            self._mean(stats["L_T1"]),
            self._mean(stats["L_T2"]),
            self._mean(stats["L_G14"]),
            self._mean(stats["L_G12"]),
            self._mean(stats["cos_g_T1_g_final"]),
            self._mean(stats["cos_g_T2_g_final"]),
            self._mean(stats["cos_G14_final"]),
            self._mean(stats["cos_G12_final"]),
            self._mean(stats["projected_rate"]),
            self._mean(stats["aux_grad_norm"])
            / max(self._mean(stats["final_grad_norm"]), 1e-8),
            float(val_mae) if val_mae is not None else float("nan"),
            float(self._best_test_mae or float("nan")),
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

        pred_final = out.get("pred_final") if out.get("pred_final") is not None else out.get("pred")
        pred_t_low = out.get("pred_T_low")
        pred_t_mid = out.get("pred_T_mid")
        if pred_final is None or pred_t_low is None or pred_t_mid is None:
            raise RuntimeError(
                "GR_capdist_final_primary requires pred, pred_T_low, pred_T_mid from forward"
            )

        target_full = real_value
        target_t1 = temporal_downsample_target(real_value, self.chain_lengths[0])
        target_t2 = temporal_downsample_target(real_value, self.chain_lengths[1])

        l_final = self._raw_loss(pred_final, target_full)
        l_t1 = self._raw_loss(pred_t_low, target_t1)
        l_t2 = self._raw_loss(pred_t_mid, target_t2)
        graph_losses, _, _ = self._resolve_graph_aux(out, target_full)

        losses = {
            "L_final": l_final,
            "L_T1": l_t1,
            "L_T2": l_t2,
            **graph_losses,
        }
        self._pending_grad_surgery = {"losses": losses}

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
            aux_loss_names=self._aux_loss_names,
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
            "[%s FINAL] best_val_MAE=%.4f test_MAE@best-val=%.4f test_RMSE@best-val=%.4f",
            self.variant_name,
            float(best_val if best_val is not None else float("nan")),
            float(self._best_test_mae or float("nan")),
            float(self._best_test_rmse or float("nan")),
        )
