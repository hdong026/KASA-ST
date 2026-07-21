from __future__ import annotations

from typing import Tuple, Union

import torch
from easytorch.utils.dist import master_only

from basicts.runners.g1_stagewise_training import (
    G1_STAGE_LOSS_NAMES,
    compute_stagewise_loss,
    set_trainable_by_stage,
    temporal_downsample_target,
)
from basicts.runners.runner_zoo.chain_forecasting_runner import ChainForecastingRunner
from basicts.runners.base_tsf_runner import SCALER_REGISTRY


class G1StagewiseRunner(ChainForecastingRunner):
    """Stagewise runner for G1_final_adaptive: T1 -> T2 -> T3 -> S1 (no S14/S12/FT)."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        param = cfg["MODEL"]["PARAM"]
        self.stagewise_backend = "g1"
        self.stagewise_base_variant = str(
            param.get("base_variant", "G1_final_adaptive")
        )
        if self.unified_aux_loss_mode != "none":
            raise ValueError("G1_stagewise requires unified_aux_loss_mode='none'")
        placement = str(param.get("spatial_placement", "")).lower()
        if placement != "final":
            raise ValueError(
                f"G1_stagewise requires spatial_placement='final', got {placement!r}"
            )

    def init_training(self, cfg: dict):
        if self.stagewise_enabled and self.stagewise_stage == "S1":
            self._enable_g1_s1_spatial_gate()
        super().init_training(cfg)

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
            stagewise_backend="g1",
        )
        self._last_chain_out = out
        prediction_data = out["pred"]
        real_for_metric = future_data_4_dec

        if self.stagewise_enabled:
            if self.stagewise_stage == "T1":
                prediction_data = out["pred_T_low"]
                real_for_metric = temporal_downsample_target(
                    future_data_4_dec, self.chain_lengths[0]
                )
            elif self.stagewise_stage == "T2":
                prediction_data = out["pred_T_mid"]
                real_for_metric = temporal_downsample_target(
                    future_data_4_dec, self.chain_lengths[1]
                )
            elif self.stagewise_stage == "T3":
                prediction_data = out["pred_T_full"]
            elif self.stagewise_stage == "S1":
                prediction_data = out.get("pred_final")
                if prediction_data is None:
                    prediction_data = out["pred"]

        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(real_for_metric)
        if (
            not self.stagewise_enabled
            or self.stagewise_stage in {"T3", "S1"}
        ):
            assert list(prediction.shape)[:3] == [batch_size, prediction.shape[1], num_nodes], (
                "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
            )
        return prediction, real_value

    def _enable_g1_s1_spatial_gate(self) -> None:
        model = self.model.module if hasattr(self.model, "module") else self.model
        spatial = getattr(model, "spatial_module", None)
        if spatial is None:
            raise RuntimeError("G1_stagewise S1 requires model.spatial_module")
        spatial.enable_g1_stagewise_s1_gate()
        self.logger.info(
            "[G1_stagewise S1] enabled zero-init residual gate on spatial_module "
            "(pred_final = pred_T_full + alpha * gate * spatial_residual)"
        )

    def _rebuild_optimizer(self, cfg: dict) -> None:
        from easytorch.core.optimizer_builder import build_lr_scheduler, build_optim

        train_cfg = cfg.get("TRAIN", {}) if hasattr(cfg, "get") else getattr(cfg, "TRAIN", {})
        self.optim = build_optim(train_cfg["OPTIM"], self.model)
        lr_cfg = train_cfg.get("LR_SCHEDULER")
        if lr_cfg is not None:
            self.scheduler = build_lr_scheduler(lr_cfg, self.optim)
        self.logger.info("[G1_stagewise S1] rebuilt optimizer/lr_scheduler for spatial-only params")

    def _masked_mean_abs(self, values: torch.Tensor, mask_reference: torch.Tensor) -> float:
        if torch.isnan(torch.tensor(self.null_val)):
            mask = ~torch.isnan(mask_reference)
        else:
            eps = 5e-5
            mask = ~torch.isclose(
                mask_reference,
                torch.tensor(self.null_val, device=mask_reference.device),
                atol=eps,
                rtol=0.0,
            )
        err = values.abs() * mask.float()
        denom = mask.float().sum().clamp_min(1.0)
        return float(err.sum().item() / denom.item())

    @torch.no_grad()
    def _log_g1_s1_initial_mae_diagnostic(self) -> None:
        if not self.stagewise_enabled or self.stagewise_stage != "S1":
            return
        data = next(iter(self.train_data_loader))
        self.model.eval()
        forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
        out = self._last_chain_out
        real_value = self.select_target_features(forward_return[1])
        pred_t3 = out.get("pred_T_full")
        pred_s1 = out.get("pred_final")
        if pred_s1 is None:
            pred_s1 = out.get("pred")
        if pred_t3 is None or pred_s1 is None:
            self.logger.warning("[G1_stagewise S1 init] missing pred_T_full or pred_final for diagnostic")
            self.model.train()
            return
        mae_t3 = float(self._raw_loss(pred_t3, real_value).detach().item())
        mae_s1 = float(self._raw_loss(pred_s1, real_value).detach().item())
        pred_s1_r, real_r = self._rescale_pair(pred_s1, real_value)
        pred_t3_r, _ = self._rescale_pair(pred_t3, real_value)
        mae_delta = self._masked_mean_abs(pred_s1_r - pred_t3_r, real_r)
        model = self.model.module if hasattr(self.model, "module") else self.model
        spatial = getattr(model, "spatial_module", None)
        gate_val = None
        alpha_val = None
        if spatial is not None and spatial.g1_stagewise_s1_gate is not None:
            gate_val = float(spatial.g1_stagewise_s1_gate.detach().item())
            alpha_val = float(spatial.hybrid_alpha)
        self.logger.info(
            "[G1_stagewise S1 init] mae_T3=%.4f mae_S1=%.4f mae_delta=%.4f "
            "residual_add=True alpha=%s gate=%s",
            mae_t3,
            mae_s1,
            mae_delta,
            f"{alpha_val:.4f}" if alpha_val is not None else "n/a",
            f"{gate_val:.4f}" if gate_val is not None else "n/a",
        )
        self.model.train()

    def _setup_stagewise_training(self, cfg: dict) -> None:
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
                "[G1_stagewise] loaded checkpoint=%s missing=%s unexpected=%s",
                load_path,
                len(missing),
                len(unexpected),
            )
        elif load_path:
            self.logger.warning("[G1_stagewise] load_checkpoint not found: %s", load_path)

        self._stagewise_trainable_info = set_trainable_by_stage(
            self.model,
            stage,
            freeze_previous=self.stagewise_freeze_previous,
            train_shared_temporal=self.stagewise_train_shared_temporal,
        )

        if stage == "S1":
            self._rebuild_optimizer(cfg)
            self._log_g1_s1_initial_mae_diagnostic()

        train_cfg = cfg.get("TRAIN", {}) if hasattr(cfg, "get") else getattr(cfg, "TRAIN", {})
        optim_param = train_cfg.get("OPTIM", {}).get("PARAM", {}) if hasattr(train_cfg, "get") else {}
        lr = float(optim_param.get("lr", 0.002))
        milestones = train_cfg.get("LR_SCHEDULER", {}).get("PARAM", {}).get("milestones", [])
        info = self._stagewise_trainable_info
        self.logger.info(
            "[G1_stagewise] variant=%s base_variant=%s stage=%s horizon=%s seed=%s "
            "chain_lengths=%s spatial=final_adaptive_only S14_disabled=True S12_disabled=True "
            "FT_disabled=True epochs=%s lr=%s milestones=%s freeze_previous=%s "
            "detach_previous=%s train_shared_temporal=%s trainable_count=%s frozen_count=%s "
            "trainable_modules=%s stage_loss_name=%s load_checkpoint=%s save_best_checkpoint=%s",
            self.stagewise_variant or "G1_stagewise",
            self.stagewise_base_variant,
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
            info.get("trainable_count"),
            info.get("frozen_count"),
            info.get("trainable_names"),
            G1_STAGE_LOSS_NAMES.get(stage, "L_unknown"),
            load_path,
            self.stagewise_save_checkpoint,
        )

    def _log_stagewise_loss_parts(self, epoch: int, iter_index: int, parts: dict[str, float]) -> None:
        if iter_index != 0:
            return
        self.logger.info(
            "[G1_stagewise epoch=%s stage=%s] %s",
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
            )
            self._last_stagewise_loss_parts = parts
            self._log_stagewise_loss_parts(epoch, iter_index, parts)
        elif self.unified_aux_loss_mode != "none":
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

    def backward(self, loss: torch.Tensor):
        super().backward(loss)
        if (
            self.stagewise_enabled
            and self.stagewise_stage == "S1"
            and not getattr(self, "_s1_grad_norm_logged", False)
        ):
            model = self.model.module if hasattr(self.model, "module") else self.model
            spatial = getattr(model, "spatial_module", None)
            if spatial is not None:
                sq_sum = 0.0
                tensor_count = 0
                for param in spatial.parameters():
                    if param.requires_grad and param.grad is not None:
                        sq_sum += float(param.grad.data.norm(2).item() ** 2)
                        tensor_count += 1
                grad_norm = sq_sum ** 0.5
                self.logger.info(
                    "[G1_stagewise S1] spatial_grad_norm=%.6f trainable_tensors_with_grad=%s",
                    grad_norm,
                    tensor_count,
                )
                self._s1_grad_norm_logged = True

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
            )
            self._last_stagewise_val_loss_parts = val_parts
        if self.stagewise_enabled:
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
        self.logger.info(
            "[G1_stagewise] saved best-val checkpoint: %s (val_%s=%.4f)",
            save_path,
            metric_name,
            metric,
        )

    def on_training_end(self):
        super().on_training_end()
        if self.stagewise_enabled and self.stagewise_stage == "S1":
            variant = self.stagewise_variant or "G1_stagewise"
            self.logger.info(
                "[G1_stagewise FINAL S1] variant=%s base_variant=%s stage_sequence=T1->T2->T3->S1 "
                "best_val_MAE=%.4f test_MAE@best-val=%.4f test_RMSE@best-val=%.4f saved_checkpoint=%s",
                variant,
                self.stagewise_base_variant,
                float(self._stagewise_best_val_mae or self.best_metrics.get("val_MAE", float("nan"))),
                float(self._stagewise_best_test_mae or float("nan")),
                float(self._stagewise_best_test_rmse or float("nan")),
                self.stagewise_save_checkpoint,
            )

    def on_validating_end(self, train_epoch=None):
        super().on_validating_end(train_epoch)
        if not self.stagewise_enabled:
            return
        diag = self._val_unified_diag
        n = diag["count"]
        if n <= 0:
            return
        train_parts = self._last_stagewise_loss_parts
        val_parts = self._last_stagewise_val_loss_parts
        self.logger.info(
            "[G1_stagewise val epoch=%s stage=%s] train_stage_loss=%s val_stage_loss=%s "
            "val_final_MAE=%.4f",
            train_epoch,
            self.stagewise_stage,
            {k: round(v, 4) for k, v in (train_parts or {}).items()},
            {k: round(v, 4) for k, v in (val_parts or {}).items()},
            diag["mae_final"] / n,
        )

    @torch.no_grad()
    @master_only
    def test(self):
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
                "[G1_stagewise test] stage=%s pred_len=%s valid_horizons=%s",
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
