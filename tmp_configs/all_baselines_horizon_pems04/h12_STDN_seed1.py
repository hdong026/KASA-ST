import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from easydict import EasyDict
from basicts.losses import masked_mae
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import STDNRunner
from basicts.utils import load_adj
from basicts.archs import STDN
from basicts.archs.arch_zoo.stdn_arch.utils import get_lpls

CFG = EasyDict()

# Source: GestaltCogTeam/BasicTS @ eb65f4b (baselines/STDN/PEMS04.py), 12->12 PeMS04

CFG.DESCRIPTION = "STDN model configuration"
CFG.RUNNER = STDNRunner
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

adj_mx, _ = load_adj("datasets/" + CFG.DATASET_NAME + "/adj_mx.pkl", "original")

model_config = {
    "Data": {
        "dataset_name": CFG.DATASET_NAME,
        "num_of_vertices": 307,
        "time_slice_size": 5,
    },
    "Training": {
        "use_nni": 0,
        "L": 2,
        "K": 16,
        "d": 8,
        "mode": "train",
        "batch_size": 64,
        "epochs": 300,
        "learning_rate": 0.001,
        "patience": 20,
        "decay_epoch": 10,
        "num_his": 12,
        "num_pred": 12,
        "in_channels": 1,
        "out_channels": 1,
        "T_miss_len": 12,
        "node_miss_rate": 0.1,
        "self_weight_dis": 0.05,
        "reference": 3,
        "order": 3,
    },
}

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STDN"
CFG.MODEL.ARCH = STDN
CFG.MODEL.PARAM = {
    "args": model_config,
    "bn_decay": 0.1,
}
CFG.MODEL.LPLS = get_lpls(adj_mx[0])
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.SETUP_GRAPH = True

CFG.TRAIN = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
CFG.TRAIN.SETUP_GRAPH = True
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": model_config["Training"]["learning_rate"],
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "StepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "step_size": model_config["Training"]["decay_epoch"],
    "gamma": 0.9,
}

CFG.TRAIN.NUM_EPOCHS = 300
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "baselines", "STDN_PEMS04")
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TRAIN.DATA.BATCH_SIZE = model_config["Training"]["batch_size"]
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
CFG.ENV.SEED = 1
if hasattr(CFG, 'SEED'):
    CFG.SEED = 1
if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):
    CFG.TRAIN.SEED = 1
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints/all_baselines_horizon_pems04/h12/STDN_seed1")
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.TEST.EVALUATION_HORIZONS = list(range(1, 13))
CFG.MODEL.PARAM["args"]["Training"]["num_his"] = 12
CFG.MODEL.PARAM["args"]["Training"]["num_pred"] = 12
CFG.MODEL.PARAM["args"]["Training"]["T_miss_len"] = 12
