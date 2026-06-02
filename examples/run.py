import importlib.util
import os
import sys
from argparse import ArgumentParser

# TODO: remove it when basicts can be installed by pip
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
import torch
from basicts import launch_training
from basicts.data.dataset import TimeSeriesForecastingDataset

torch.set_num_threads(1)  # aviod high cpu avg usage


def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def load_cfg(cfg_path):
    cfg_path = os.path.abspath(cfg_path)
    module_name = os.path.splitext(os.path.basename(cfg_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, cfg_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CFG"):
        raise AttributeError(f"Config file {cfg_path} does not define CFG")
    return module.CFG


def apply_model_param_overrides(cfg, args):
    if args.use_spectral_residual is not None:
        cfg.MODEL.PARAM["use_spectral_residual"] = str2bool(args.use_spectral_residual)

    if args.prior_mapper_type is not None:
        cfg.MODEL.PARAM["prior_mapper_type"] = args.prior_mapper_type

    if args.tag is not None:
        cfg.TRAIN.CKPT_SAVE_DIR = cfg.TRAIN.CKPT_SAVE_DIR + "_" + args.tag


def print_model_param_overrides(cfg):
    print("MODEL.PARAM overrides:")
    print("use_spectral_residual:", cfg.MODEL.PARAM.get("use_spectral_residual"))
    print("prior_mapper_type:", cfg.MODEL.PARAM.get("prior_mapper_type"))
    print("CKPT_SAVE_DIR:", cfg.TRAIN.CKPT_SAVE_DIR)


def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    parser.add_argument("-c", "--cfg", default="examples/LSTNN/LSTNN_PEMS04.py", help="training config")
    parser.add_argument("--gpus", default="1", help="visible gpus")
    parser.add_argument("--use_spectral_residual", type=str, default=None)
    parser.add_argument(
        "--prior_mapper_type",
        type=str,
        default=None,
        choices=["kan", "linear", "mlp"],
    )
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = load_cfg(args.cfg)
    apply_model_param_overrides(cfg, args)
    print_model_param_overrides(cfg)

    launch_training(cfg, args.gpus)
