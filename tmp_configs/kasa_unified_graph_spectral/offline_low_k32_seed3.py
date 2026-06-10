import os
import sys

import torch

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from easydict import EasyDict
from basicts.losses import masked_mae
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.archs import KASA_v2


CFG = EasyDict()

# PeMS04: official 6:2:2 split, 12→12, graph spectral prior / calibration experiment

# ================= general ================= #
CFG.DESCRIPTION = "KASA graph spectral prior / calibration on PeMS04"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "PEMS04"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "KASA_v2"
CFG.MODEL.ARCH = KASA_v2
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
    "spatial_scheme": "C",
    "adj_mx_path": os.path.join("datasets", CFG.DATASET_NAME, "adj_mx.pkl"),
    "use_gcn": True,
    "gcn_hidden_dim": 64,
    "use_dynamic_spatial": True,
    "dyn_hidden_dim": 64,
    "dyn_topk": 20,
    "dyn_tau": 0.5,
    "dyn_static_weight": 0.2,
    "use_adaptive_adj": True,
    "adp_hidden_dim": 32,
    "adp_topk": 20,
    "adp_tau": 0.5,
    "use_hybrid_graph": True,
    "hybrid_alpha": 0.2,
    "use_lightweight_spatial": False,
    "use_template_lookup": False,
    "prior_mapper_type": "mlp",
    "use_pre_temporal_spatial_enhancement": False,
    "keep_output_prior_residual": False,
    "use_input_prior_enhancement": False,
    "post_spatial_mode": "adaptive_only",
    "use_patch_branch": True,
    "use_downsample_branch": True,
    "use_linear_residual_branch": True,
    "use_extra_prior_input": False,
    "main_input_dim": 3,
    "use_graph_spectral_calibration": False,
    "graph_spectral_calibration_mode": "none",
    "graph_spectral_k": 32,
    "graph_spectral_hidden": 16,
    "graph_spectral_dropout": 0.0,
    "graph_spectral_gamma_init": 0.0,
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
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

# ================= train ================= #
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

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ===== kasa_unified_graph_spectral overrides (auto-generated) =====
CFG.ENV.SEED = 3
if hasattr(CFG, 'SEED'):
    CFG.SEED = 3
if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):
    CFG.TRAIN.SEED = 3
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "kasa_unified_graph_spectral", "offline_low_k32_seed3")
CFG.TRAIN.DATA.DIR = 'datasets/PEMS04_graph_spectral_k32'
CFG.VAL.DATA.DIR = 'datasets/PEMS04_graph_spectral_k32'
CFG.TEST.DATA.DIR = 'datasets/PEMS04_graph_spectral_k32'
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.PARAM["input_dim"] = 4
CFG.MODEL.PARAM["adj_mx_path"] = os.path.join('datasets/PEMS04_graph_spectral_k32', "adj_mx.pkl")
CFG.MODEL.PARAM["use_patch_branch"] = True
CFG.MODEL.PARAM["use_downsample_branch"] = True
CFG.MODEL.PARAM["use_linear_residual_branch"] = True
CFG.MODEL.PARAM["post_spatial_mode"] = 'adaptive_only'
CFG.MODEL.PARAM["use_pre_temporal_spatial_enhancement"] = False
CFG.MODEL.PARAM["keep_output_prior_residual"] = False
CFG.MODEL.PARAM["use_input_prior_enhancement"] = False
CFG.MODEL.PARAM["use_extra_prior_input"] = True
CFG.MODEL.PARAM["main_input_dim"] = 4
CFG.MODEL.PARAM["use_graph_spectral_calibration"] = False
CFG.MODEL.PARAM["graph_spectral_calibration_mode"] = 'none'
CFG.MODEL.PARAM["graph_spectral_k"] = 32
CFG.MODEL.PARAM["graph_spectral_gamma_init"] = 0.0
