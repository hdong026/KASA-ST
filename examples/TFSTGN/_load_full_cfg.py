import importlib.util
import os


def load_full_cfg(root_path: str):
    cfg_path = os.path.join(root_path, "examples", "TFSTGN", "TFSTGN_PEMS04_full.py")
    spec = importlib.util.spec_from_file_location("tfstgn_pems04_full", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CFG
