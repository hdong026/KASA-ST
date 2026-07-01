import os
import sys
from functools import partial

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from easydict import EasyDict
from basicts.metrics import masked_huber
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import HimNetRunner
from basicts.archs import HimNet

CFG = EasyDict()

# Source: GestaltCogTeam/BasicTS @ eb65f4b (baselines/HimNet/PEMS04.py), 12->12 PeMS04

HIMNET_CONFIG = {
    "lr": 0.001,
    "eps": 0.001,
    "weight_decay": 0.0001,
    "milestones": [30, 50],
    "clip_grad": 5,
    "batch_size": 16,
    "max_epochs": 200,
    "early_stop": 20,
}

CFG.DESCRIPTION = "HimNet model configuration"
CFG.RUNNER = HimNetRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "PEMS04"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "HimNet"
CFG.MODEL.ARCH = HimNet
CFG.MODEL.PARAM = {
    "num_nodes": 307,
    "input_dim": 3,
    "output_dim": 1,
    "tod_embedding_dim": 12,
    "dow_embedding_dim": 4,
    "out_steps": 12,
    "hidden_dim": 64,
    "num_layers": 1,
    "cheb_k": 2,
    "ycov_dim": 2,
    "node_embedding_dim": 16,
    "st_embedding_dim": 16,
    "tf_decay_steps": 4000,
    "use_teacher_forcing": True,
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.SETUP_GRAPH = True

CFG.TRAIN = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
CFG.TRAIN.SETUP_GRAPH = True
CFG.TRAIN.LOSS = partial(masked_huber, reduction="mean", delta=1.0)
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": HIMNET_CONFIG["lr"],
    "eps": HIMNET_CONFIG["eps"],
    "weight_decay": HIMNET_CONFIG["weight_decay"],
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": HIMNET_CONFIG["milestones"],
    "gamma": 0.1,
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": HIMNET_CONFIG["clip_grad"],
}

CFG.TRAIN.NUM_EPOCHS = HIMNET_CONFIG["max_epochs"]
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "baselines", "HimNet_PEMS04")
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TRAIN.DATA.BATCH_SIZE = HIMNET_CONFIG["batch_size"]
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ===== all_baselines_horizon_pems04 overrides (auto-generated) =====
CFG.ENV.SEED = 2
if hasattr(CFG, 'SEED'):
    CFG.SEED = 2
if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):
    CFG.TRAIN.SEED = 2
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints/all_baselines_horizon_pems04/h12/HimNet_seed2")
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.TEST.EVALUATION_HORIZONS = list(range(1, 13))
CFG.MODEL.PARAM["out_steps"] = 12
