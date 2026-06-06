import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from easydict import EasyDict
from basicts.losses import masked_mae
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.archs.arch_zoo.KASA_arch_v3_freqgate.KASA_arch import KASA_v3_FreqGate


CFG = EasyDict()

CFG.DESCRIPTION = "KASA v3-freqgate model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
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
CFG.MODEL.NAME = "KASA_v3_freqgate"
CFG.MODEL.ARCH = KASA_v3_FreqGate
CFG.MODEL.PARAM = {
    "node_size": 307,
    "input_len": CFG.DATASET_INPUT_LEN,
    "output_len": CFG.DATASET_OUTPUT_LEN,
    "input_dim": 4,
    "patch_len": 3,
    "stride": 4,
    "td_size": 288,
    "dw_size": 7,
    "d_td": 32,
    "d_dw": 32,
    "d_d": 32,
    "d_spa": 32,
    "if_time_in_day": True,
    "if_day_in_week": True,
    "if_spatial": True,
    "num_layer": 2,
    "adj_mx_path": os.path.join("datasets", CFG.DATASET_NAME, "adj_mx.pkl"),
    "use_gcn": True,
    "gcn_hidden_dim": 64,
    "dyn_hidden_dim": 64,
    "dyn_topk": 20,
    "dyn_tau": 0.5,
    "dyn_static_weight": 0.2,
    "adp_hidden_dim": 32,
    "adp_topk": 20,
    "adp_tau": 0.5,
    "hybrid_alpha": 0.2,
    "use_pre_temporal_spatial_enhancement": False,
    "keep_output_prior_residual": False,
    "use_input_prior_enhancement": False,
    "use_frequency_guided_graph": True,
    "use_freq_conditioned_fusion": True,
    "freq_dim": 16,
    "freq_topk": 20,
    "graph_fusion_hidden": 16,
    "use_cross_st_gate": True,
    "gate_hidden": 16,
    "gate_residual_scale": 1.0,
    "use_spectral_decomp_gate": False,
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
    "gamma": 0.5
}

CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "KASA_v3_freqgate_PEMS04")
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
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ===== kasa_v3_freqgate_ablation overrides (auto-generated) =====
CFG.ENV.SEED = 4
if hasattr(CFG, 'SEED'):
    CFG.SEED = 4
if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):
    CFG.TRAIN.SEED = 4
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "kasa_v3_freqgate", "v3_freq_global_seed4")
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.PARAM["use_frequency_guided_graph"] = True
CFG.MODEL.PARAM["use_freq_conditioned_fusion"] = False
CFG.MODEL.PARAM["use_cross_st_gate"] = False
CFG.MODEL.PARAM["use_spectral_decomp_gate"] = False
CFG.MODEL.PARAM["keep_output_prior_residual"] = False
