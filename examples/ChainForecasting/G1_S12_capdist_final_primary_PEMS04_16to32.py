import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from easydict import EasyDict
from basicts.losses import masked_mae
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import G1S12CapDistFinalPrimaryRunner
from basicts.archs import ChainForecasting


CFG = EasyDict()

CFG.DESCRIPTION = "G1_S12_capdist_final_primary on PeMS04 16→32"
CFG.RUNNER = G1S12CapDistFinalPrimaryRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "PEMS04"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 16
CFG.DATASET_OUTPUT_LEN = 32
CFG.GPU_NUM = 1

CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "ChainForecasting"
CFG.MODEL.ARCH = ChainForecasting
CFG.MODEL.PARAM = {
    "node_size": 307,
    "input_len": CFG.DATASET_INPUT_LEN,
    "output_len": CFG.DATASET_OUTPUT_LEN,
    "input_dim": 3,
    "main_input_dim": 3,
    "patch_len": 4,
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
    "use_patch_branch": True,
    "use_downsample_branch": True,
    "use_linear_residual_branch": True,
    "patch_embedding_mode": "serial_concat",
    "patch_data_input_mode": "all",
    "spatial_placement": "temporal_first_graph_resolution",
    "post_spatial_mode": "adaptive_cluster_mix",
    "use_pre_temporal_spatial_enhancement": False,
    "keep_output_prior_residual": False,
    "use_input_prior_enhancement": False,
    "use_graph_spectral_calibration": False,
    "use_extra_prior_input": False,
    "use_prev_condition": True,
    "chain_lengths": [8, 16, 32],
    "chain_loss_weights": [0.0, 0.0, 0.0],
    "spatial_graph_loss_weights": [0.0, 0.0],
    "unified_aux_loss_mode": "none",
    "aux_eta_temporal": 0.05,
    "aux_eta_spatial": 0.03,
    "aux_temporal_anchor_weight": 0.02,
    "aux_temporal_power": 2.0,
    "aux_spatial_power": 2.0,
    "aux_include_spatial_final": False,
    "aux_mono_margin": 0.0,
    "variant_name": "G1_S12_capdist_final_primary",
    "base_variant": "G1_final_primary_grad_surgery",
    "graph_cluster_method": "capdist_spectral",
    "graph_resolution_ratios": [0.50, 1.00],
    "graph_resolution_capacities": [2, 1],
    "graph_resolution_topks": [8, 16],
    "graph_resolution_alphas": [0.08, 0.10],
    "graph_resolution_betas": [1.0, 1.0],
    "graph_resolution_rhos": [0.50, 1.00],
    "cluster_graph_mix_lambdas": [0.5, 0.3],
    "capdist_sigma_d": 0.5,
    "capdist_lambda_d": 0.05,
    "capdist_use_hard_cutoff": False,
    "capdist_use_road_distance": True,
    "clustering_seed": 0,
    "dataset_name": "PEMS04",
    "cluster_road_distance_path": "datasets/raw_data/PEMS04/adj_PEMS04_distance.pkl",
    "cluster_sigma_d": 0.5,
    "cluster_delta_4": 0.8,
    "cluster_delta_2": 0.5,
    "graph_resolution_skip_final_identity": False,
    "final_primary_grad_surgery": True,
    "aux_grad_max_ratio": 0.2,
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
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
    "G1_S12_capdist_final_primary_PEMS04_16to32",
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
