import sys
sys.path.insert(0, '/home/dhz/KASA-ST/examples/baselines/GTS')
import os
import sys
# Repo root (examples/baselines/<Model>/)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch
import torch.nn.functional as F
from easydict import EasyDict
from basicts.archs import GTS
from basicts.runners import GTSRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.utils.serialization import load_pkl

_GTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _GTS_DIR not in sys.path:
    sys.path.insert(0, _GTS_DIR)
from loss import gts_loss


def _infer_gts_dim_fc(feats, num_nodes: int) -> int:
    """Derive dim_fc from train node_feats via GTS conv stack (official PEMS04 uses 162976)."""
    conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)
    conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)
    if not isinstance(feats, torch.Tensor):
        feats = torch.tensor(feats, dtype=torch.float32)
    x = feats.transpose(1, 0).view(num_nodes, 1, -1).float()
    x = conv1(x)
    x = F.relu(x)
    x = conv2(x)
    x = F.relu(x)
    return int(x.view(num_nodes, -1).shape[1])


CFG = EasyDict()

# GTS does not allow to load parameters since it creates parameters in the first iteration
resume = False
if not resume:
    import random
    _ = random.randint(-1e6, 1e6)

# ================= general ================= #
CFG.DESCRIPTION = "GTS model configuration"
CFG.RUNNER  = GTSRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "PEMS04"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG._ = _
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "GTS"
CFG.MODEL.ARCH = GTS
node_feats_full = load_pkl("datasets/{0}/data_in{1}_out{2}.pkl".format(CFG.DATASET_NAME, CFG.DATASET_INPUT_LEN, CFG.DATASET_OUTPUT_LEN))["processed_data"][..., 0]
train_index_list = load_pkl("datasets/{0}/index_in{1}_out{2}.pkl".format(CFG.DATASET_NAME, CFG.DATASET_INPUT_LEN, CFG.DATASET_OUTPUT_LEN))["train"]
node_feats = node_feats_full[:train_index_list[-1][-1], ...]
_dim_fc = _infer_gts_dim_fc(node_feats, 307)
CFG.MODEL.PARAM = {
    "cl_decay_steps": 2000,
    "filter_type": "dual_random_walk",
    "horizon": 12,
    "input_dim": 2,
    "l1_decay": 0,
    "max_diffusion_step": 3,
    "num_nodes": 307,
    "num_rnn_layers": 1,
    "output_dim": 1,
    "rnn_units": 64,
    "seq_len": 12,
    "use_curriculum_learning": True,
    "dim_fc": _dim_fc,
    "node_feats": node_feats,
    "temp": 0.5,
    "k": 30
}
CFG.MODEL.SETUP_GRAPH = True
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
CFG.TRAIN.SETUP_GRAPH = True
CFG.TRAIN.LOSS = gts_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "eps": 1e-3
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [20, 30],
    "gamma": 0.1
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}
CFG.TRAIN.NUM_EPOCHS = 200
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "baselines", "GTS_PEMS04_" + str(CFG.TRAIN.NUM_EPOCHS))
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
CFG.ENV.SEED = 3
if hasattr(CFG, 'SEED'):
    CFG.SEED = 3
if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):
    CFG.TRAIN.SEED = 3
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints/all_baselines_horizon_pems04/h48/GTS_seed3")
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 48
CFG.TEST.EVALUATION_HORIZONS = list(range(1, 49))
import sys; sys.path.insert(0, '/home/dhz/KASA-ST/examples/baselines/GTS')
import torch
import torch.nn.functional as F
from basicts.utils.serialization import load_pkl
_gts_data = load_pkl("datasets/PEMS04/data_in12_out48.pkl")
_gts_index = load_pkl("datasets/PEMS04/index_in12_out48.pkl")
_gts_node_feats_full = _gts_data['processed_data'][..., 0]
_gts_train_index = _gts_index['train']
_gts_node_feats = _gts_node_feats_full[:_gts_train_index[-1][-1], ...]

def _infer_gts_dim_fc(feats, num_nodes):
    conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)
    conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)
    if not isinstance(feats, torch.Tensor):
        feats = torch.tensor(feats, dtype=torch.float32)
    x = feats.transpose(1, 0).view(num_nodes, 1, -1).float()
    x = conv1(x); x = F.relu(x); x = conv2(x); x = F.relu(x)
    return int(x.view(num_nodes, -1).shape[1])

CFG.MODEL.PARAM["horizon"] = 48
CFG.MODEL.PARAM["seq_len"] = 12
CFG.MODEL.PARAM["node_feats"] = _gts_node_feats
CFG.MODEL.PARAM["dim_fc"] = _infer_gts_dim_fc(_gts_node_feats, 307)
