import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from easydict import EasyDict
from basicts.losses import masked_mae
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.archs import TFSTGN


def build_model_param(dataset_name: str = "PEMS04") -> dict:
    """Lightweight defaults: small spatial branch + slim temporal trunk."""
    return {
        "num_nodes": 307,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 4,
        "target_dim": 1,
        # spatial / TF branch
        "hidden_dim": 64,
        "embed_dim": 16,
        "n_bands": 4,
        "n_fft": 8,
        "hop_length": 2,
        "win_length": 8,
        "band_mode": "soft",
        "topk": 20,
        "attn_temperature": 1.0,
        "temporal_backbone": "bigru",
        "temporal_layers": 1,
        "dropout": 0.1,
        "use_film": True,
        "use_band_specific_proj": False,
        "use_spectral_gate": True,
        "use_temporal_gate": True,
        "use_spatial_gate": True,
        "use_tf_spatial": True,
        "spatial_alpha_init": -2.0,
        "gate_bias": -2.0,
        "static_hybrid_alpha": 0.2,
        "adj_mx_path": os.path.join("datasets", dataset_name, "adj_mx.pkl"),
        "prediction_head": "shared",
        "decoder_mode": "shared",
        # temporal trunk (KASA-style, slim)
        "patch_len": 3,
        "stride": 4,
        "td_size": 288,
        "dw_size": 7,
        "d_td": 24,
        "d_dw": 24,
        "d_d": 24,
        "d_spa": 24,
        "if_time_in_day": True,
        "if_day_in_week": True,
        "if_spatial": False,
        "num_layer": 1,
        "use_prior_residual": True,
        "prior_mapper_type": "linear",
    }


CFG = EasyDict()

CFG.DESCRIPTION = "TF-STGN lightweight decoupled model on PEMS04"
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
CFG.MODEL.NAME = "TFSTGN"
CFG.MODEL.ARCH = TFSTGN
CFG.MODEL.PARAM = build_model_param(CFG.DATASET_NAME)
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]
CFG.MODEL.TARGET_FEATURES = [0]

CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.004, "weight_decay": 0.0001}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 35, 60, 80, 95], "gamma": 0.5}
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "TFSTGN_PEMS04_full")
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
