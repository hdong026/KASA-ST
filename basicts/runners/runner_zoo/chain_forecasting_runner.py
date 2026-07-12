from __future__ import annotations

from typing import Tuple, Union

import torch

from basicts.archs.arch_zoo.ChainForecasting_arch import ChainForecasting
from basicts.runners.base_tsf_runner import SCALER_REGISTRY
from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class ChainForecastingRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for ChainForecasting with chain / unified auxiliary training loss."""

    UNIFIED_MODES = {
        "none",
        "unified_direct_small",
        "unified_residual_detach",
        "unified_mono",
    }

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
        self.dataset_input_len = int(cfg["DATASET_INPUT_LEN"])
        self.dataset_output_len = int(cfg["DATASET_OUTPUT_LEN"])

        self.unified_aux_loss_mode = str(param.get("unified_aux_loss_mode", "none")).lower()
        if self.unified_aux_loss_mode not in self.UNIFIED_MODES:
            raise ValueError(
                f"Unsupported unified_aux_loss_mode: {self.unified_aux_loss_mode}. "
                f"Expected one of {sorted(self.UNIFIED_MODES)}."
            )
        self.aux_eta_temporal = float(param.get("aux_eta_temporal", 0.05))
        self.aux_eta_spatial = float(param.get("aux_eta_spatial", 0.03))
        self.aux_temporal_anchor_weight = float(param.get("aux_temporal_anchor_weight", 0.02))
        self.aux_temporal_power = float(param.get("aux_temporal_power", 2.0))
        self.aux_spatial_power = float(param.get("aux_spatial_power", 2.0))
        self.aux_include_spatial_final = bool(param.get("aux_include_spatial_final", False))
        self.aux_mono_margin = float(param.get("aux_mono_margin", 0.0))
        self._aux_residual_target_detach = True

        if self.unified_aux_loss_mode != "none":
            if any(float(w) != 0.0 for w in self.spatial_graph_loss_weights):
                raise ValueError(
                    "spatial_graph_loss_weights must be zero when unified_aux_loss_mode is enabled."
                )
            total_aux = self.aux_eta_temporal + self.aux_temporal_anchor_weight + self.aux_eta_spatial
            if total_aux > 0.15 + 1e-9:
                raise ValueError(
                    f"Total unified auxiliary weight {total_aux:.4f} exceeds 0.15."
                )
            self._temporal_aux_weights = self._compute_temporal_aux_weights()
            self._spatial_aux_weights = None
            self._spatial_aux_stage_indices: list[int] = []
        else:
            self._temporal_aux_weights = []
            self._spatial_aux_weights = []

        self._last_chain_out = None
        self._last_unified_loss_parts: dict[str, float] = {}
        self._reset_val_unified_diag()

    def _reset_val_unified_diag(self) -> None:
        self._val_unified_diag = {
            "count": 0,
            "mae_temporal_final": 0.0,
            "mae_graph_stages": [],
            "mae_final": 0.0,
            "residual_energy_cluster": [],
            "residual_energy_lifted": [],
        }

    def _compute_temporal_aux_weights(self) -> list[float]:
        h_full = float(self.chain_lengths[-1])
        stage_lens = self.chain_lengths[:-1]
        if len(stage_lens) < 2:
            return [0.0] * max(len(stage_lens), 0)
        rhos = [float(h) / h_full for h in stage_lens[:2]]
        q = self.aux_temporal_power
        denom = sum(r ** q for r in rhos)
        if denom <= 0:
            return [0.0, 0.0]
        return [self.aux_eta_temporal * (r ** q) / denom for r in rhos]

    def _compute_spatial_aux_weights(self, graph_ratios: list[float], num_stages: int) -> tuple[list[int], list[float]]:
        if num_stages <= 0:
            return [], []
        if self.aux_include_spatial_final:
            stage_indices = list(range(num_stages))
        else:
            stage_indices = list(range(max(num_stages - 1, 0)))
        if not stage_indices:
            return [], []
        ratios = list(graph_ratios[:num_stages])
        aux_ratios = [float(ratios[i]) for i in stage_indices]
        p = self.aux_spatial_power
        denom = sum(r ** p for r in aux_ratios)
        if denom <= 0:
            return stage_indices, [0.0] * len(stage_indices)
        weights = [self.aux_eta_spatial * (r ** p) / denom for r in aux_ratios]
        return stage_indices, weights

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

        assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes], (
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        )
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value

    def _rescale_pair(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(pred, **self.scaler["args"])
        target_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(target, **self.scaler["args"])
        return pred_rescaled, target_rescaled

    def _raw_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_rescaled, target_rescaled = self._rescale_pair(pred, target)
        return self.metric_forward(self.loss, [pred_rescaled, target_rescaled])

    def _weighted_loss(self, pred: torch.Tensor, target: torch.Tensor, weight: float) -> torch.Tensor:
        if float(weight) == 0.0:
            return torch.tensor(0.0, device=target.device)
        return float(weight) * self._raw_loss(pred, target)

    def _loss_per_sample(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_rescaled, target_rescaled = self._rescale_pair(pred, target)
        if torch.isnan(torch.tensor(self.null_val)):
            mask = ~torch.isnan(target_rescaled)
        else:
            eps = 5e-5
            mask = ~torch.isclose(
                target_rescaled,
                torch.tensor(self.null_val, device=target_rescaled.device),
                atol=eps,
                rtol=0.0,
            )
        err = (pred_rescaled - target_rescaled).abs()
        err = err * mask.float()
        denom = mask.float().sum(dim=(1, 2, 3)).clamp_min(1.0)
        return err.sum(dim=(1, 2, 3)) / denom

    def _project_nodes(self, node_x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.einsum("mn,btnc->btmc", p, node_x)

    def _compute_unified_spatial_aux(
        self,
        out: dict,
        final_target: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        graph_diag = out.get("graph_resolution_diagnostics") or {}
        if not graph_diag:
            return torch.tensor(0.0, device=device)

        node_stage_preds = graph_diag.get("node_stage_preds") or []
        node_before_preds = graph_diag.get("node_before_preds") or []
        cluster_residuals = graph_diag.get("cluster_residuals") or []
        projection_matrices = graph_diag.get("graph_projection_matrices") or []
        graph_ratios = graph_diag.get("graph_ratios") or graph_diag.get("graph_resolution_ratios") or []

        num_stages = len(cluster_residuals)
        if num_stages == 0:
            return torch.tensor(0.0, device=device)

        stage_indices, nu_weights = self._compute_spatial_aux_weights(graph_ratios, num_stages)
        self._spatial_aux_stage_indices = stage_indices
        self._spatial_aux_weights = nu_weights

        loss = torch.tensor(0.0, device=device)
        mode = self.unified_aux_loss_mode

        for weight, stage_idx in zip(nu_weights, stage_indices):
            if float(weight) == 0.0:
                continue
            if mode == "unified_direct_small":
                if stage_idx >= len(node_stage_preds):
                    continue
                pred = node_stage_preds[stage_idx]
                loss = loss + self._weighted_loss(pred, final_target, weight)
            elif mode == "unified_residual_detach":
                if stage_idx >= len(cluster_residuals):
                    continue
                if stage_idx >= len(node_before_preds):
                    continue
                if stage_idx >= len(projection_matrices):
                    continue
                y_before = node_before_preds[stage_idx]
                delta = (final_target - y_before).detach()
                delta_c = self._project_nodes(delta, projection_matrices[stage_idx])
                r_c = cluster_residuals[stage_idx]
                if r_c.shape != delta_c.shape:
                    raise ValueError(
                        f"Graph residual shape mismatch at stage {stage_idx}: "
                        f"R_c={tuple(r_c.shape)} vs Delta_c={tuple(delta_c.shape)}"
                    )
                loss = loss + self._weighted_loss(r_c, delta_c, weight)
            elif mode == "unified_mono":
                if stage_idx >= len(node_stage_preds):
                    continue
                if stage_idx >= len(node_before_preds):
                    continue
                y_prev = node_before_preds[stage_idx]
                y_curr = node_stage_preds[stage_idx]
                e_prev = self._loss_per_sample(y_prev, final_target)
                e_curr = self._loss_per_sample(y_curr, final_target)
                mono = torch.relu(e_curr - e_prev.detach() + self.aux_mono_margin).mean()
                loss = loss + float(weight) * mono
        return loss

    def _compute_unified_loss(self, out: dict, real_value: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        device = real_value.device
        final_target = real_value
        h_full = int(self.chain_lengths[-1])
        targets = [ChainForecasting.pool_target(real_value, k) for k in self.chain_lengths]

        l_final = self._raw_loss(out["pred"], final_target)
        loss = l_final

        temporal_preds = out.get("temporal_preds") or out.get("temporal_stage_preds") or []
        l_temporal_aux = torch.tensor(0.0, device=device)
        if len(temporal_preds) >= 2:
            for weight, pred, target in zip(
                self._temporal_aux_weights,
                temporal_preds[:2],
                targets[:2],
            ):
                l_temporal_aux = l_temporal_aux + self._weighted_loss(pred, target, weight)
        loss = loss + l_temporal_aux

        l_temporal_anchor = torch.tensor(0.0, device=device)
        if self.aux_temporal_anchor_weight > 0.0 and temporal_preds:
            l_temporal_anchor = self._weighted_loss(
                temporal_preds[-1],
                final_target,
                self.aux_temporal_anchor_weight,
            )
            loss = loss + l_temporal_anchor

        l_spatial_aux = self._compute_unified_spatial_aux(out, final_target, device)
        loss = loss + l_spatial_aux

        parts = {
            "L_final": float(l_final.detach().item()),
            "L_temporal_aux": float(l_temporal_aux.detach().item()),
            "L_temporal_anchor": float(l_temporal_anchor.detach().item()),
            "L_spatial_aux": float(l_spatial_aux.detach().item()),
            "L_total": float(loss.detach().item()),
        }
        return loss, parts

    def _legacy_loss(self, out: dict, real_value: torch.Tensor) -> torch.Tensor:
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
        return loss

    def _log_unified_loss_parts(self, epoch: int, iter_index: int, parts: dict[str, float]) -> None:
        if iter_index != 0:
            return
        mu_str = ",".join(f"{w:.4f}" for w in self._temporal_aux_weights)
        nu_str = ",".join(f"{w:.4f}" for w in (self._spatial_aux_weights or []))
        self.logger.info(
            "[UnifiedAux epoch=%s] mode=%s input_len=%s output_len=%s chain_lengths=%s "
            "graph_ratios=%s aux_eta_temporal=%.4f aux_eta_spatial=%.4f "
            "aux_temporal_anchor_weight=%.4f aux_temporal_power=%.1f aux_spatial_power=%.1f "
            "residual_target_detach=%s | mu=[%s] nu=[%s] | "
            "L_final=%.4f L_temporal_aux=%.4f L_temporal_anchor=%.4f "
            "L_spatial_aux=%.4f L_total=%.4f",
            epoch,
            self.unified_aux_loss_mode,
            self.dataset_input_len,
            self.dataset_output_len,
            self.chain_lengths,
            (self._last_chain_out or {}).get("graph_ratios")
            or ((self._last_chain_out or {}).get("graph_resolution_diagnostics") or {}).get("graph_ratios"),
            self.aux_eta_temporal,
            self.aux_eta_spatial,
            self.aux_temporal_anchor_weight,
            self.aux_temporal_power,
            self.aux_spatial_power,
            self._aux_residual_target_detach,
            mu_str,
            nu_str,
            parts.get("L_final", 0.0),
            parts.get("L_temporal_aux", 0.0),
            parts.get("L_temporal_anchor", 0.0),
            parts.get("L_spatial_aux", 0.0),
            parts.get("L_total", 0.0),
        )

    def _update_val_unified_diag(self, out: dict, real_value: torch.Tensor) -> None:
        final_target = real_value
        temporal_preds = out.get("temporal_preds") or []
        graph_diag = out.get("graph_resolution_diagnostics") or {}
        node_stage_preds = graph_diag.get("node_stage_preds") or []

        def _batch_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
            pred_r, target_r = self._rescale_pair(pred, target)
            return float(self.metric_forward(self.loss, [pred_r, target_r]).item())

        self._val_unified_diag["count"] += 1
        if temporal_preds:
            self._val_unified_diag["mae_temporal_final"] += _batch_mae(temporal_preds[-1], final_target)
        self._val_unified_diag["mae_final"] += _batch_mae(out["pred"], final_target)

        if not self._val_unified_diag["mae_graph_stages"]:
            self._val_unified_diag["mae_graph_stages"] = [0.0] * len(node_stage_preds)
        for i, pred in enumerate(node_stage_preds):
            if i >= len(self._val_unified_diag["mae_graph_stages"]):
                self._val_unified_diag["mae_graph_stages"].append(0.0)
            self._val_unified_diag["mae_graph_stages"][i] += _batch_mae(pred, final_target)

        rec = graph_diag.get("residual_energy_cluster") or []
        lif = graph_diag.get("residual_energy_lifted") or []
        if not self._val_unified_diag["residual_energy_cluster"]:
            self._val_unified_diag["residual_energy_cluster"] = [0.0] * len(rec)
            self._val_unified_diag["residual_energy_lifted"] = [0.0] * len(lif)
        for i, val in enumerate(rec):
            if i < len(self._val_unified_diag["residual_energy_cluster"]):
                self._val_unified_diag["residual_energy_cluster"][i] += float(val)
        for i, val in enumerate(lif):
            if i < len(self._val_unified_diag["residual_energy_lifted"]):
                self._val_unified_diag["residual_energy_lifted"][i] += float(val)

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

        if self.unified_aux_loss_mode != "none":
            loss, parts = self._compute_unified_loss(out, real_value)
            self._last_unified_loss_parts = parts
            self._log_unified_loss_parts(epoch, iter_index, parts)
        else:
            loss = self._legacy_loss(out, real_value)

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

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
        out = self._last_chain_out
        if self.unified_aux_loss_mode != "none" or out.get("graph_resolution_diagnostics"):
            _, real_value = forward_return
            self._update_val_unified_diag(out, real_value)

        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(
            forward_return[0], **self.scaler["args"]
        )
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(
            forward_return[1], **self.scaler["args"]
        )
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
            self.update_epoch_meter("val_" + metric_name, metric_item.item())

    def on_validating_start(self, train_epoch: int = None):
        self._reset_val_unified_diag()

    def on_validating_end(self, train_epoch=None):
        super().on_validating_end(train_epoch)
        diag = self._val_unified_diag
        if diag["count"] > 0 and (
            self.unified_aux_loss_mode != "none"
            or (self._last_chain_out or {}).get("graph_resolution_diagnostics")
        ):
            n = diag["count"]
            graph_mae = [v / n for v in diag["mae_graph_stages"]]
            rec = [v / n for v in diag["residual_energy_cluster"]]
            lif = [v / n for v in diag["residual_energy_lifted"]]
            self.logger.info(
                "[UnifiedValDiag epoch=%s] MAE_temporal_final=%.4f graph_stage_MAE=%s "
                "MAE_final=%.4f mean_abs_cluster_residual=%s mean_abs_lifted_residual=%s",
                train_epoch,
                diag["mae_temporal_final"] / n,
                [round(v, 4) for v in graph_mae],
                diag["mae_final"] / n,
                [round(v, 6) for v in rec],
                [round(v, 6) for v in lif],
            )
        self._reset_val_unified_diag()

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
            diag_ms = model.spatial_module.get_adaptive_ms_diagnostics()
            weights = diag_ms.get("adaptive_ms_weights")
            topks = diag_ms.get("adaptive_ms_topks")
            alpha = diag_ms.get("adaptive_ms_alpha")
            entropy = diag_ms.get("adaptive_ms_entropy")
        elif (
            hasattr(model, "graph_resolution_stack")
            and model.graph_resolution_stack is not None
            and model.graph_resolution_stack.spatial_modules
        ):
            diag_ms = model.graph_resolution_stack.spatial_modules[-1].get_adaptive_ms_diagnostics()
            weights = diag_ms.get("adaptive_ms_weights")
            topks = diag_ms.get("adaptive_ms_topks")
            alpha = diag_ms.get("adaptive_ms_alpha")
            entropy = diag_ms.get("adaptive_ms_entropy")
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
