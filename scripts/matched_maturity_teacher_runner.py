#!/usr/bin/env python3
"""Matched-maturity teacher runner: same forecasting stack, no TEST, history.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
from easytorch.utils.dist import master_only

from basicts.runners.runner_zoo.chain_forecasting_runner import ChainForecastingRunner
from scripts.matched_maturity_lib import (
    get_test_access_count,
    install_test_access_guard,
    sha1_state_dict,
)


class MatchedMaturityTeacherRunner(ChainForecastingRunner):
    """Identical to ChainForecastingRunner except:
    - never builds TEST loader
    - records initial parameter hash + per-epoch history.json
    - asserts train index length / optional expected indices meta
    """

    def __init__(self, cfg: dict):
        install_test_access_guard()
        # ensure TEST section exists for BaseTSFRunner.__init__ but is inert
        if "TEST" not in cfg:
            cfg["TEST"] = {"INTERVAL": 10**9, "DATA": dict(cfg["VAL"]["DATA"]), "USE_GPU": True}
        else:
            cfg["TEST"]["INTERVAL"] = int(cfg["TEST"].get("INTERVAL", 10**9))
            if cfg["TEST"]["INTERVAL"] < 10**6:
                cfg["TEST"]["INTERVAL"] = 10**9
        super().__init__(cfg)
        self._mm_history_path = Path(
            cfg.get("MATCHED_MATURITY_HISTORY_PATH")
            or os.path.join(str(self.ckpt_save_dir), "history.json")
        )
        self._mm_meta_dir = Path(
            cfg.get("MATCHED_MATURITY_META_DIR") or str(self.ckpt_save_dir)
        )
        self._mm_expected_train_len = cfg.get("MATCHED_MATURITY_EXPECTED_TRAIN_LEN")
        self._mm_history: list[dict] = []
        if self._mm_history_path.is_file():
            try:
                self._mm_history = json.loads(self._mm_history_path.read_text())
            except Exception:
                self._mm_history = []

    @master_only
    def init_test(self, cfg: dict):
        """Disable TEST completely — do not instantiate test dataset/loader."""
        self.test_interval = 10**9
        self.test_data_loader = None
        self.logger.info(
            "[matched_maturity] TEST disabled (no loader). TEST_ACCESS_COUNT=%s",
            get_test_access_count(),
        )

    def init_training(self, cfg: dict):
        super().init_training(cfg)
        self._mm_meta_dir.mkdir(parents=True, exist_ok=True)
        init_path = self._mm_meta_dir / "initial_state.json"
        resumed = int(getattr(self, "start_epoch", 0) or 0) > 0
        model = self.model.module if hasattr(self.model, "module") else self.model
        if resumed and init_path.is_file():
            meta = json.loads(init_path.read_text())
            self.logger.info(
                "[matched_maturity] resume detected start_epoch=%s; preserving initial_state.json",
                self.start_epoch,
            )
        else:
            # Fresh-init hash BEFORE any optimizer step (after model+seed construction).
            init_hash = sha1_state_dict(model.state_dict(), n=40)
            meta = {
                "initial_param_sha1": init_hash,
                "optimizer_type": type(self.optim).__name__,
                "scheduler_type": type(self.scheduler).__name__ if self.scheduler is not None else None,
                "num_epochs": int(self.num_epochs),
                "seed": int(cfg.get("ENV", {}).get("SEED", -1)),
                "train_len": int(len(self.train_data_loader.dataset)),
                "TEST_ACCESS_COUNT": get_test_access_count(),
                "resumed": bool(resumed),
            }
            if self._mm_expected_train_len is not None:
                exp = int(self._mm_expected_train_len)
                if meta["train_len"] != exp:
                    raise RuntimeError(
                        f"MATCHED_TRAIN_LEN_MISMATCH actual={meta['train_len']} expected={exp}"
                    )
            if type(self.optim).__name__ != "Adam":
                raise RuntimeError(f"MATCHED_PROTOCOL_MISMATCH optimizer={type(self.optim).__name__}")
            if self.scheduler is None or type(self.scheduler).__name__ != "MultiStepLR":
                raise RuntimeError(
                    f"MATCHED_PROTOCOL_MISMATCH scheduler={type(self.scheduler).__name__ if self.scheduler else None}"
                )
            init_path.write_text(json.dumps(meta, indent=2))
            self.logger.info("[matched_maturity] initial_param_sha1=%s", init_hash)

        # Always re-assert optim/scheduler types
        if type(self.optim).__name__ != "Adam":
            raise RuntimeError(f"MATCHED_PROTOCOL_MISMATCH optimizer={type(self.optim).__name__}")
        if self.scheduler is None or type(self.scheduler).__name__ != "MultiStepLR":
            raise RuntimeError(
                f"MATCHED_PROTOCOL_MISMATCH scheduler={type(self.scheduler).__name__ if self.scheduler else None}"
            )
        self.logger.info(
            "[matched_maturity] optim=%s scheduler=%s train_len=%s start_epoch=%s",
            type(self.optim).__name__,
            type(self.scheduler).__name__ if self.scheduler else None,
            int(len(self.train_data_loader.dataset)),
            getattr(self, "start_epoch", 0),
        )
        train_dir = cfg["TRAIN"]["DATA"]["DIR"]
        val_dir = cfg["VAL"]["DATA"]["DIR"]
        self.logger.info(
            "[matched_maturity] train_loader source DIR=%s (fold purged TRAIN indices)",
            train_dir,
        )
        self.logger.info(
            "[matched_maturity] valid_loader source DIR=%s (official VALID via index['valid'])",
            val_dir,
        )

    def on_epoch_end(self, epoch: int):
        # Do NOT call BaseRunner.on_epoch_end directly: it resets meters before we can log.
        self.test_data_loader = None
        try:
            lr = float(self.meter_pool.get_avg("lr"))
        except Exception:
            try:
                lr = float(self.scheduler.get_last_lr()[0]) if self.scheduler is not None else float("nan")
            except Exception:
                lr = float("nan")
        train_row = {
            "train_loss": _safe_meter(self, "train_MAE"),
            "train_MAE": _safe_meter(self, "train_MAE"),
            "train_RMSE": _safe_meter(self, "train_RMSE"),
            "train_MAPE": _safe_meter(self, "train_MAPE"),
            "lr": lr,
        }
        self.print_epoch_meters("train")
        self.plt_epoch_meters("train", epoch)
        if self.val_data_loader is not None and epoch % self.val_interval == 0:
            self.validate(train_epoch=epoch)
        row = {
            "epoch": int(epoch),
            **train_row,
            "val_MAE": _safe_meter(self, "val_MAE"),
            "val_RMSE": _safe_meter(self, "val_RMSE"),
            "val_MAPE": _safe_meter(self, "val_MAPE"),
            "TEST_ACCESS_COUNT": get_test_access_count(),
        }
        self._mm_history = [r for r in self._mm_history if int(r.get("epoch", -1)) != int(epoch)]
        self._mm_history.append(row)
        self._mm_history.sort(key=lambda r: int(r["epoch"]))
        self._mm_history_path.parent.mkdir(parents=True, exist_ok=True)
        self._mm_history_path.write_text(json.dumps(self._mm_history, indent=2))
        self.save_model(epoch)
        self.reset_epoch_meters()

    def on_training_end(self):
        super().on_training_end()
        best = None
        if self._mm_history:
            # earliest epoch with minimum val_MAE (from logged history, not reset meters)
            ranked = sorted(
                self._mm_history,
                key=lambda r: (float(r.get("val_MAE", 1e9)), int(r["epoch"])),
            )
            best = ranked[0]
        summary = {
            "completed_epochs": len(self._mm_history),
            "num_epochs_target": int(self.num_epochs),
            "best_epoch": None if best is None else int(best["epoch"]),
            "best_val_MAE": None if best is None else float(best["val_MAE"]),
            "optimizer_type": type(self.optim).__name__,
            "scheduler_type": type(self.scheduler).__name__ if self.scheduler else None,
            "TEST_ACCESS_COUNT": get_test_access_count(),
        }
        (self._mm_meta_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
        self.logger.info("[matched_maturity] training_summary=%s", summary)


def _safe_meter(runner, name: str) -> float:
    try:
        return float(runner.meter_pool.get_avg(name))
    except Exception:
        return float("nan")
