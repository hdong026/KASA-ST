"""Build HyperD EasyTorch configs compatible with KASA-ST examples/run.py."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.losses import masked_mae
from basicts.utils import load_adj, load_dataset_desc

from baselines.HyperD.arch import HyperD
from baselines.HyperD.data_prepare import ensure_hyperd_data, init_npy_paths
from baselines.HyperD.hyperd_runner import HyperDRunner
from baselines.HyperD.hyperd_settings import (
    BATCH_SIZE,
    DATASET_META,
    INPUT_LEN,
    LEARNING_RATES,
    MODEL_PARAMS,
    NUM_EPOCHS,
    OUTPUT_LEN,
    TRAIN_VAL_TEST_RATIO,
)


def build_cfg(dataset_name: str) -> EasyDict:
    dataset_name = dataset_name.upper()
    if dataset_name not in DATASET_META:
        raise KeyError(f"Unsupported dataset: {dataset_name}")

    ensure_hyperd_data(dataset_name)
    desc = load_dataset_desc(dataset_name)
    meta = DATASET_META[dataset_name]
    params = MODEL_PARAMS[dataset_name]
    lr = LEARNING_RATES[dataset_name]

    adj_mx, _ = load_adj(meta["adj_path"], "normlap")
    daily_path, weekly_path = init_npy_paths(dataset_name)

    model_param = {
        "seq_len": INPUT_LEN,
        "pred_len": OUTPUT_LEN,
        "num_nodes": meta["num_nodes"],
        "init_path_daily": str(daily_path),
        "init_path_weekly": str(weekly_path),
        "adj": torch.tensor(adj_mx[0]),
        "alpha": params["alpha"],
        "F_low": params["F_low"],
        "embed_size": params["embed_size"],
        "hidden_size": params["hidden_size"],
        "fc_hidden_size": params["fc_hidden_size"],
        "time_of_day_size": 288,
        "day_of_week_size": 7,
    }

    train_steps = math.ceil(desc["num_time_steps"] * TRAIN_VAL_TEST_RATIO[0])

    cfg = EasyDict()
    cfg.DESCRIPTION = f"HyperD official baseline on {dataset_name} 12->12"
    cfg.RUNNER = HyperDRunner
    cfg.DATASET_CLS = None  # HyperDRunner builds dataset directly
    cfg.DATASET_NAME = dataset_name
    cfg.DATASET_TYPE = "Traffic flow"
    cfg.DATASET_INPUT_LEN = INPUT_LEN
    cfg.DATASET_OUTPUT_LEN = OUTPUT_LEN
    cfg.GPU_NUM = 1
    cfg.NULL_VAL = 0.0

    cfg.ENV = EasyDict()
    cfg.ENV.SEED = 1
    cfg.ENV.CUDNN = EasyDict()
    cfg.ENV.CUDNN.ENABLED = True

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = "HyperD"
    cfg.MODEL.ARCH = HyperD
    cfg.MODEL.PARAM = model_param
    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
    cfg.MODEL.TARGET_FEATURES = [0]

    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NULL_VAL = 0.0
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.NUM_EPOCHS = NUM_EPOCHS
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        "baselines",
        f"HyperD_{dataset_name}_12to12",
    )
    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {"lr": lr}
    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    cfg.TRAIN.LR_SCHEDULER.TYPE = "OneCycleLR"
    cfg.TRAIN.LR_SCHEDULER.PARAM = {
        "pct_start": 0.3,
        "epochs": NUM_EPOCHS,
        "steps_per_epoch": train_steps,
        "max_lr": lr,
    }
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    cfg.TRAIN.DATA = EasyDict()
    cfg.TRAIN.DATA.DIR = f"datasets/{dataset_name}"
    cfg.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
    cfg.TRAIN.DATA.PREFETCH = False
    cfg.TRAIN.DATA.SHUFFLE = True
    cfg.TRAIN.DATA.NUM_WORKERS = 2
    cfg.TRAIN.DATA.PIN_MEMORY = False

    cfg.VAL = EasyDict()
    cfg.VAL.INTERVAL = 1
    cfg.VAL.DATA = EasyDict()
    cfg.VAL.DATA.DIR = cfg.TRAIN.DATA.DIR
    cfg.VAL.DATA.BATCH_SIZE = BATCH_SIZE
    cfg.VAL.DATA.PREFETCH = False
    cfg.VAL.DATA.SHUFFLE = False
    cfg.VAL.DATA.NUM_WORKERS = 2
    cfg.VAL.DATA.PIN_MEMORY = False

    cfg.TEST = EasyDict()
    cfg.TEST.INTERVAL = 1
    cfg.TEST.USE_GPU = True
    cfg.TEST.EVALUATION_HORIZONS = [3, 6, 12]
    cfg.TEST.DATA = EasyDict()
    cfg.TEST.DATA.DIR = cfg.TRAIN.DATA.DIR
    cfg.TEST.DATA.BATCH_SIZE = BATCH_SIZE
    cfg.TEST.DATA.PREFETCH = False
    cfg.TEST.DATA.SHUFFLE = False
    cfg.TEST.DATA.NUM_WORKERS = 2
    cfg.TEST.DATA.PIN_MEMORY = False

    return cfg


def write_config_module(dataset_name: str, path: Path) -> Path:
    cfg = build_cfg(dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '"""HyperD official baseline config (auto-generated template)."""',
        "import os",
        "import sys",
        "from easydict import EasyDict",
        "",
        f"ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
        "if ROOT not in sys.path:",
        "    sys.path.insert(0, ROOT)",
        "",
        "from baselines.HyperD.hyperd_config import build_cfg",
        "",
        f"CFG = build_cfg('{dataset_name.upper()}')",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
