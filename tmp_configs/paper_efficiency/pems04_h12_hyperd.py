import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.HyperD.hyperd_config import build_cfg

CFG = build_cfg("PEMS04")
CFG.ENV.SEED = 1
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.MODEL.PARAM["seq_len"] = 12
CFG.MODEL.PARAM["pred_len"] = 12
