from __future__ import annotations

from typing import Tuple, Union

import torch
from easytorch.utils.dist import master_only

from basicts.archs.arch_zoo.ChainForecasting_arch import ChainForecasting
from basicts.losses.forecast_state_token_mae import forecast_state_token_mae
from basicts.runners.base_tsf_runner import SCALER_REGISTRY
from basicts.runners.stagewise_training import (
    STAGE_LOSS_NAMES,
    compute_stagewise_loss,
    set_trainable_by_stage,
    temporal_downsample_target,
)
from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class ChainForecastingRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for ChainForecasting with chain / unified auxiliary training loss."""

    UNIFIED_MODES = {
        "none",
        "unified_direct_small",
        "unified_residual_detach",
        "unified_mono",
    }
    CHAIN_LOSS_MODES = {"weighted", "token_mae"}

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        param = cfg["MODEL"]["PARAM"]
        self.chain_lengths = list(param.get("chain_lengths", [3, 6, 12]))
        self.chain_loss_mode = str(param.get("chain_loss_mode", "weighted")).lower()
        if self.chain_loss_mode not in self.CHAIN_LOSS_MODES:
            raise ValueError(
                f"Unsupported chain_loss_mode: {self.chain_loss_mode}. "
                f"Expected one of {sorted(self.CHAIN_LOSS_MODES)}."
            )
        raw_weights = param.get("chain_loss_weights", [0.2, 0.3, 1.0])
        if self.chain_loss_mode == "token_mae":
            # Artificial stage weights are unused; allow None / omit.
            self.chain_loss_weights = list(raw_weights) if raw_weights is not None else []
        else:
            if raw_weights is None:
                raise ValueError("chain_loss_weights is required when chain_loss_mode='weighted'.")
            self.chain_loss_weights = list(raw_weights)
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
        self._last_stagewise_loss_parts: dict[str, float] = {}
        self._last_stagewise_val_loss_parts: dict[str, float] = {}
        self._stagewise_alpha_logged: set[str] = set()
        self._stagewise_best_val_mae: float | None = None
        self._stagewise_best_test_mae: float | None = None
        self._stagewise_best_test_rmse: float | None = None
        self._stagewise_saved_best_this_epoch = False
        self._reset_val_unified_diag()

        train_cfg = cfg.get("TRAIN", {}) if hasattr(cfg, "get") else getattr(cfg, "TRAIN", {})
        sw = getattr(train_cfg, "STAGEWISE", None) if train_cfg is not None else None
        if sw is None and train_cfg is not None:
            sw = train_cfg.get("STAGEWISE") if hasattr(train_cfg, "get") else {}
        sw = sw or {}
        self.train_num_epochs = int(train_cfg.get("NUM_EPOCHS", 100) if hasattr(train_cfg, "get") else 100)
        self.stagewise_enabled = bool(sw.get("enabled", False))
        self.stagewise_stage = str(sw.get("stage", "T1")).upper() if self.stagewise_enabled else None
        self.stagewise_freeze_previous = bool(sw.get("freeze_previous", True))
        self.stagewise_detach_previous = bool(sw.get("detach_previous", True))
        self.stagewise_load_checkpoint = sw.get("load_checkpoint")
        self.stagewise_save_checkpoint = sw.get("save_checkpoint")
        self.stagewise_ckpt_root = sw.get("ckpt_root", "checkpoints/gr7_stagewise")
        self.stagewise_variant = str(sw.get("variant_name", param.get("variant_name", "")))
        self.stagewise_sequence = str(sw.get("stage_sequence", "full")).lower()
        self.stagewise_train_shared_temporal = bool(sw.get("train_shared_temporal", True))
        self._graph_resolution_alphas = list(
            param.get("graph_resolution_alphas", [0.03, 0.06, 0.10])
        )
        self._graph_resolution_ratios = list(
            param.get("graph_resolution_ratios", [0.25, 0.50, 1.00])
        )
        self._graph_resolution_topks = list(
            param.get("graph_resolution_topks", [4, 8, 16])
        )
        self._stagewise_trainable_info: dict = {}

    def _reset_val_unified_diag(self) -> None:
        self._val_unified_diag = {
            "count": 0,
            "mae_temporal_final": 0.0,
            "mae_before_graph": 0.0,
            "mae_graph_stages": [],
            "mae_final": 0.0,
            "residual_energy_cluster": [],
            "residual_energy_lifted": [],
            "a_cluster_density": 0.0,
            "a_adp_density": 0.0,
            "a_cluster_adp_mean_abs_diff": 0.0,
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
            return_intermediates=self.stagewise_enabled,
            stagewise_stage=self.stagewise_stage if self.stagewise_enabled else None,
            detach_previous=self.stagewise_detach_previous if self.stagewise_enabled else True,
            stagewise_sequence=self.stagewise_sequence if self.stagewise_enabled else "full",
        )
        self._last_chain_out = out
        prediction_data = out["pred"]
        real_for_metric = future_data_4_dec

        if self.stagewise_enabled:
            if self.stagewise_stage == "T1":
                prediction_data = out["pred_T_low"]
                real_for_metric = temporal_downsample_target(future_data_4_dec, self.chain_lengths[0])
            elif self.stagewise_stage == "T2":
                prediction_data = out["pred_T_mid"]
                real_for_metric = temporal_downsample_target(future_data_4_dec, self.chain_lengths[1])
            elif self.stagewise_stage == "T3":
                prediction_data = out["pred_T_full"]
            elif self.stagewise_stage in {"S14", "S12", "S1"}:
                prediction_data = out["pred"]

        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(real_for_metric)
        if (
            not self.stagewise_enabled
            or self.stagewise_stage in {"T3", "S14", "S12", "S1", "FT"}
        ):
            assert list(prediction.shape)[:3] == [batch_size, prediction.shape[1], num_nodes], (
                "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
            )
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

    def _token_mae_loss(self, out: dict, real_value: torch.Tensor) -> torch.Tensor:
        """Token-normalized MAE on post-spatial forecast states (no stage reweighting)."""
        preds = list(out["chain_preds"])
        targets = [ChainForecasting.pool_target(real_value, k) for k in self.chain_lengths]
        if len(preds) != len(targets):
            raise ValueError(
                f"chain_preds ({len(preds)}) and chain_lengths ({len(targets)}) mismatch."
            )
        # Do not add out["pred"] again: for interleaved spatial, chain_preds[-1] is the
        # final forecast state already (avoids double-counting T12).
        return forecast_state_token_mae(
            preds,
            targets,
            null_val=self.null_val,
            rescale_pair=self._rescale_pair,
        )

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
        before = graph_diag.get("temporal_input")
        if before is not None:
            self._val_unified_diag["mae_before_graph"] += _batch_mae(before, final_target)
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

        if graph_diag.get("a_cluster_density") is not None:
            self._val_unified_diag["a_cluster_density"] += float(graph_diag["a_cluster_density"])
        if graph_diag.get("a_adp_density") is not None:
            self._val_unified_diag["a_adp_density"] += float(graph_diag["a_adp_density"])
        if graph_diag.get("a_cluster_adp_mean_abs_diff") is not None:
            self._val_unified_diag["a_cluster_adp_mean_abs_diff"] += float(
                graph_diag["a_cluster_adp_mean_abs_diff"]
            )

    def init_training(self, cfg: dict):
        super().init_training(cfg)
        if not self.stagewise_enabled:
            return
        self._setup_stagewise_training(cfg)

    def _setup_stagewise_training(self, cfg: dict) -> None:
        import os
        from pathlib import Path

        import torch

        stage = self.stagewise_stage
        load_path = self.stagewise_load_checkpoint
        if load_path and Path(load_path).is_file():
            state = torch.load(load_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            self.logger.info(
                "[GR7_stagewise] loaded checkpoint=%s missing=%s unexpected=%s",
                load_path,
                len(missing),
                len(unexpected),
            )
        elif load_path:
            self.logger.warning("[GR7_stagewise] load_checkpoint not found: %s", load_path)

        self._stagewise_trainable_info = set_trainable_by_stage(
            self.model,
            stage,
            freeze_previous=self.stagewise_freeze_previous,
            train_shared_temporal=self.stagewise_train_shared_temporal,
            sequence=self.stagewise_sequence,
        )

        train_cfg = cfg.get("TRAIN", {}) if hasattr(cfg, "get") else getattr(cfg, "TRAIN", {})
        optim_param = train_cfg.get("OPTIM", {}).get("PARAM", {}) if hasattr(train_cfg, "get") else {}
        lr = float(optim_param.get("lr", 0.002))
        milestones = train_cfg.get("LR_SCHEDULER", {}).get("PARAM", {}).get("milestones", [])
        info = self._stagewise_trainable_info
        self.logger.info(
            "[GR7_stagewise] variant=%s stage=%s horizon=%s seed=%s chain_lengths=%s "
            "epochs=%s lr=%s milestones=%s freeze_previous=%s detach_previous=%s "
            "train_shared_temporal=%s stage_sequence=%s trainable_count=%s frozen_count=%s "
            "trainable_modules=%s loss_name=%s load_checkpoint=%s save_best_checkpoint=%s",
            self.stagewise_variant or "GR7_stagewise",
            stage,
            self.dataset_output_len,
            cfg.get("ENV", {}).get("SEED"),
            self.chain_lengths,
            self.train_num_epochs,
            lr,
            milestones,
            self.stagewise_freeze_previous,
            self.stagewise_detach_previous,
            self.stagewise_train_shared_temporal,
            self.stagewise_sequence,
            info.get("trainable_count"),
            info.get("frozen_count"),
            info.get("trainable_names"),
            STAGE_LOSS_NAMES.get(stage, "L_unknown"),
            load_path,
            self.stagewise_save_checkpoint,
        )
        if self.stagewise_sequence == "final_spatial_only":
            self.logger.info(
                "[GR7_stagewise] graph_resolution_ratios=%s graph_resolution_topks=%s "
                "graph_resolution_alphas=%s",
                self._graph_resolution_ratios,
                self._graph_resolution_topks,
                self._graph_resolution_alphas,
            )
        if stage in {"S14", "S12", "S1"}:
            from basicts.runners.stagewise_training import resolve_spatial_stage_idx

            num_spatial = 1
            if hasattr(self.model, "graph_resolution_stack") and self.model.graph_resolution_stack:
                num_spatial = len(self.model.graph_resolution_stack.spatial_modules)
            alpha_idx = resolve_spatial_stage_idx(stage, self.stagewise_sequence, num_spatial)
            alpha_r = (
                self._graph_resolution_alphas[alpha_idx]
                if alpha_idx < len(self._graph_resolution_alphas)
                else self._graph_resolution_alphas[-1]
            )
            self.logger.info(
                "[GR7_stagewise] stage=%s alpha_r=%.4f residual_scaled_by_alpha=True",
                stage,
                float(alpha_r),
            )
        if self.stagewise_sequence == "final_spatial_only" and stage == "S1":
            alpha_1 = float(self._graph_resolution_alphas[0] if self._graph_resolution_alphas else 0.10)
            self.logger.info(
                "[GR7_stagewise] previous_prediction=T3_output_Y_T spatial_stages_active=S1_only "
                "S14_disabled=True S12_disabled=True alpha_1=%.2f "
                "S1_loss=residual_if_R_n_available_else_prediction",
                alpha_1,
            )

    def _log_stagewise_loss_parts(self, epoch: int, iter_index: int, parts: dict[str, float]) -> None:
        if iter_index != 0:
            return
        self.logger.info(
            "[GR7_stagewise epoch=%s stage=%s] %s",
            epoch,
            self.stagewise_stage,
            ", ".join(f"{k}={v:.4f}" for k, v in parts.items()),
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

        if self.stagewise_enabled:
            loss, parts = compute_stagewise_loss(
                self.stagewise_stage,
                out,
                real_value,
                self.chain_lengths,
                self._raw_loss,
                sequence=self.stagewise_sequence,
                logger=self.logger,
                graph_resolution_alphas=self._graph_resolution_alphas,
                log_alpha_once=True,
                _alpha_logged=self._stagewise_alpha_logged,
            )
            self._last_stagewise_loss_parts = parts
            self._log_stagewise_loss_parts(epoch, iter_index, parts)
            if self.stagewise_stage == "FT":
                forward_return[0] = out["pred"]
        elif self.unified_aux_loss_mode != "none":
            loss, parts = self._compute_unified_loss(out, real_value)
            self._last_unified_loss_parts = parts
            self._log_unified_loss_parts(epoch, iter_index, parts)
        elif self.chain_loss_mode == "token_mae":
            loss = self._token_mae_loss(out, real_value)
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
        if self.stagewise_enabled:
            _, real_value = forward_return
            _, val_parts = compute_stagewise_loss(
                self.stagewise_stage,
                out,
                real_value,
                self.chain_lengths,
                self._raw_loss,
                sequence=self.stagewise_sequence,
                graph_resolution_alphas=self._graph_resolution_alphas,
            )
            self._last_stagewise_val_loss_parts = val_parts
        if self.unified_aux_loss_mode != "none" or out.get("graph_resolution_diagnostics") or self.stagewise_enabled:
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

    def on_epoch_end(self, epoch: int):
        super().on_epoch_end(epoch)

    @master_only
    def save_best_model(self, epoch: int, metric_name: str, greater_best: bool = True):
        metric = self.meter_pool.get_avg(metric_name)
        best_metric = self.best_metrics.get(metric_name)
        improved = best_metric is None or (
            metric > best_metric if greater_best else metric < best_metric
        )
        super().save_best_model(epoch, metric_name, greater_best)
        if not (improved and self.stagewise_enabled and self.stagewise_save_checkpoint):
            return
        from pathlib import Path

        import torch

        save_path = Path(self.stagewise_save_checkpoint)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model.state_dict(), save_path)
        self._stagewise_saved_best_this_epoch = True
        self._stagewise_best_val_mae = float(metric)
        self.logger.info("[GR7_stagewise] saved best-val checkpoint: %s (val_%s=%.4f)", save_path, metric_name, metric)

    def on_test_end(self):
        super().on_test_end()
        if not self.stagewise_enabled or not self._stagewise_saved_best_this_epoch:
            return
        self._stagewise_best_test_mae = float(self.meter_pool.get_avg("test_MAE"))
        self._stagewise_best_test_rmse = float(self.meter_pool.get_avg("test_RMSE"))
        self._stagewise_saved_best_this_epoch = False

    def on_training_end(self):
        super().on_training_end()
        if self.stagewise_enabled and self.stagewise_stage == "S1":
            variant = self.stagewise_variant or "GR7_stagewise"
            self.logger.info(
                "[GR7_stagewise FINAL S1] variant=%s stage_sequence=%s best_val_MAE=%.4f "
                "test_MAE@best-val=%.4f test_RMSE@best-val=%.4f saved_checkpoint=%s",
                variant,
                self.stagewise_sequence,
                float(self._stagewise_best_val_mae or self.best_metrics.get("val_MAE", float("nan"))),
                float(self._stagewise_best_test_mae or float("nan")),
                float(self._stagewise_best_test_rmse or float("nan")),
                self.stagewise_save_checkpoint,
            )

    def on_validating_end(self, train_epoch=None):
        super().on_validating_end(train_epoch)
        diag = self._val_unified_diag
        n = diag["count"]
        if n > 0 and (
            self.unified_aux_loss_mode != "none"
            or (self._last_chain_out or {}).get("graph_resolution_diagnostics")
        ):
            graph_mae = [v / n for v in diag["mae_graph_stages"]]
            rec = [v / n for v in diag["residual_energy_cluster"]]
            lif = [v / n for v in diag["residual_energy_lifted"]]
            self.logger.info(
                "[UnifiedValDiag epoch=%s] MAE_temporal_final=%.4f MAE_before_graph=%.4f "
                "graph_stage_MAE=%s MAE_final=%.4f mean_abs_cluster_residual=%s "
                "mean_abs_lifted_residual=%s A_cluster_density=%.6f A_adp_density=%.6f "
                "mean_abs_diff_cluster_vs_adp=%.6f",
                train_epoch,
                diag["mae_temporal_final"] / n,
                diag["mae_before_graph"] / n,
                [round(v, 4) for v in graph_mae],
                diag["mae_final"] / n,
                [round(v, 6) for v in rec],
                [round(v, 6) for v in lif],
                diag["a_cluster_density"] / n,
                diag["a_adp_density"] / n,
                diag["a_cluster_adp_mean_abs_diff"] / n,
            )
            graph_diag = (self._last_chain_out or {}).get("graph_resolution_diagnostics") or {}
            if graph_diag.get("model_name") == "CapDistRefine" and len(graph_mae) >= 2:
                self.logger.info(
                    "[CapDistRefineVal epoch=%s] MAE_before_spatial=%.4f "
                    "MAE_after_S12=%.4f MAE_after_S1=%.4f MAE_final=%.4f",
                    train_epoch,
                    diag["mae_before_graph"] / n,
                    graph_mae[0],
                    graph_mae[1],
                    diag["mae_final"] / n,
                )
        if self.stagewise_enabled and n > 0:
            train_parts = self._last_stagewise_loss_parts
            val_parts = self._last_stagewise_val_loss_parts
            self.logger.info(
                "[GR7_stagewise val epoch=%s stage=%s] train_stage_loss=%s val_stage_loss=%s "
                "val_final_MAE=%.4f",
                train_epoch,
                self.stagewise_stage,
                {k: round(v, 4) for k, v in (train_parts or {}).items()},
                {k: round(v, 4) for k, v in (val_parts or {}).items()},
                diag["mae_final"] / n,
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

    @torch.no_grad()
    @master_only
    def test(self):
        """Test with horizon filtering for stagewise partial outputs (e.g. T1 h=8 on h32)."""
        prediction = []
        real_value = []
        for _, data in enumerate(self.test_data_loader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            prediction.append(forward_return[0])
            real_value.append(forward_return[1])
        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        prediction = SCALER_REGISTRY.get(self.scaler["func"])(prediction, **self.scaler["args"])
        real_value = SCALER_REGISTRY.get(self.scaler["func"])(real_value, **self.scaler["args"])

        pred_len = int(prediction.shape[1])
        real_len = int(real_value.shape[1])
        if pred_len != real_len:
            min_len = min(pred_len, real_len)
            prediction = prediction[:, :min_len]
            real_value = real_value[:, :min_len]
            pred_len = min_len

        valid_horizons = [i for i in self.evaluation_horizons if i < pred_len]
        if self.stagewise_enabled:
            self.logger.info(
                "[GR7_stagewise test] stage=%s pred_len=%s valid_horizons=%s",
                self.stagewise_stage,
                pred_len,
                [h + 1 for h in valid_horizons],
            )
        for i in valid_horizons:
            pred = prediction[:, i, :, :]
            real = real_value[:, i, :, :]
            metric_repr = ""
            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, [pred, real])
                metric_repr += ", Test {0}: {1:.4f}".format(metric_name, metric_item.item())
            self.logger.info(
                "Evaluate best model on test data for horizon {:d}{}".format(i + 1, metric_repr)
            )
        for metric_name, metric_func in self.metrics.items():
            if self.evaluate_on_gpu:
                metric_item = self.metric_forward(metric_func, [prediction, real_value])
            else:
                metric_item = self.metric_forward(
                    metric_func, [prediction.detach().cpu(), real_value.detach().cpu()]
                )
            self.update_epoch_meter("test_" + metric_name, metric_item.item())
