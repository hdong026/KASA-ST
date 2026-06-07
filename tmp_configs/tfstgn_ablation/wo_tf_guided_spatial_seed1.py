import importlib.util
import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

_loader_path = os.path.join(os.path.dirname(__file__), "_load_full_cfg.py")
_spec = importlib.util.spec_from_file_location("tfstgn_loader", _loader_path)
_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loader)

CFG = _loader.load_full_cfg(root_path)
CFG.MODEL.PARAM["use_film"] = False
CFG.MODEL.PARAM["use_band_specific_proj"] = False
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "TFSTGN_PEMS04_wo_tf_guided_spatial")

# ===== tfstgn_ablation overrides (auto-generated) =====
CFG.ENV.SEED = 1
if hasattr(CFG, 'SEED'):
    CFG.SEED = 1
if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):
    CFG.TRAIN.SEED = 1
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "tfstgn_ablation", "wo_tf_guided_spatial_seed1")
