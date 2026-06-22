import os
import sys
# Repo root (examples/baselines/<Model>/)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from easydict import EasyDict
from basicts.archs import AGCRN
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.losses import masked_mae


CFG = EasyDict()

# Source: GestaltCogTeam/BasicTS @ 7a7f970 (examples/*/*_PEMS04.py), 12->12 PeMS04

# ================= general ================= #
CFG.DESCRIPTION = "AGCRN model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "PEMS04"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "AGCRN"
CFG.MODEL.ARCH = AGCRN
CFG.MODEL.PARAM = {
    "num_nodes" : 307,
    "input_dim" : 1,
    "rnn_units" : 64,
    "output_dim": 1,
    "horizon"   : 12,
    "num_layers": 2,
    "default_graph": True,
    "embed_dim" : 10,
    "cheb_k"    : 2
}
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.003,
}

# ================= train ================= #
CFG.TRAIN.NUM_EPOCHS = 200
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "baselines", "AGCRN_PEMS04_" + str(CFG.TRAIN.NUM_EPOCHS))
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ===== all_baselines_horizon_pems04 overrides (auto-generated) =====
CFG.ENV.SEED = 5
if hasattr(CFG, 'SEED'):
    CFG.SEED = 5
if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):
    CFG.TRAIN.SEED = 5
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints/all_baselines_horizon_pems04/h24/AGCRN_seed5")
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 24
CFG.TEST.EVALUATION_HORIZONS = list(range(1, 25))
CFG.MODEL.PARAM["horizon"] = 24
