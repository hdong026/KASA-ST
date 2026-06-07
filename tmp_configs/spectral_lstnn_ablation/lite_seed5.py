import importlib.util
import os
import sys

from basicts.losses import masked_mae

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

_loader_path = os.path.join(os.path.dirname(__file__), "_load_full_cfg.py")
_spec = importlib.util.spec_from_file_location("spectral_lstnn_loader", _loader_path)
_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loader)

CFG = _loader.load_full_cfg(root_path)
CFG.MODEL.PARAM["spatial_scale_init"] = 0.05
CFG.MODEL.PARAM["gate_scale_init"] = 0.05
CFG.MODEL.PARAM["freq_topk"] = 32
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "SpectralLSTNN_PEMS04_lite")

# ===== spectral_lstnn_ablation overrides (auto-generated) =====
CFG.ENV.SEED = 5
if hasattr(CFG, 'SEED'):
    CFG.SEED = 5
if hasattr(CFG, 'TRAIN') and hasattr(CFG.TRAIN, 'SEED'):
    CFG.TRAIN.SEED = 5
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "spectral_lstnn_ablation", "lite_seed5")
