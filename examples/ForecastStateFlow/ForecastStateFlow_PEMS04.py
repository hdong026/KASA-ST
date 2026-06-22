import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from easydict import EasyDict
from basicts.losses import masked_mae
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import ForecastStateFlowRunner
from basicts.archs import ForecastStateFlow


CFG = EasyDict()

CFG.DESCRIPTION = "Forecast-State Flow Chain on PeMS04 12→12"
CFG.RUNNER = ForecastStateFlowRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "PEMS04"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1

CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "ForecastStateFlow"
CFG.MODEL.ARCH = ForecastStateFlow
CFG.MODEL.PARAM = {
    "node_size": 307,
    "input_len": CFG.DATASET_INPUT_LEN,
    "output_len": CFG.DATASET_OUTPUT_LEN,
    "input_dim": 4,
    "hidden_dim": 64,
    "num_encoder_layers": 2,
    "topk": 20,
    "dropout": 0.1,
    "num_flow_steps": 1,
    "fm_sigma": 0.01,
    "chain_lengths": [3, 6, 12],
    "final_loss_weight": 1.0,
    "state_loss_weight": 0.3,
    "native_loss_weight": 0.1,
    "fm_loss_weight": 0.2,
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]
CFG.MODEL.TARGET_FEATURES = [0]

CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 35, 60, 80, 95],
    "gamma": 0.5,
}

CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)]),
)
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.EVALUATION_HORIZONS = list(range(1, CFG.DATASET_OUTPUT_LEN + 1))
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False
