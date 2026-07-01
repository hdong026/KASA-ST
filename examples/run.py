import os
import sys
from argparse import ArgumentParser
from typing import Optional

# TODO: remove it when basicts can be installed by pip
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    # parser.add_argument("-c", "--cfg", default="examples/DGCRN/DGCRN_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/GWNet/GWNet_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/STID/STID_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/DCRNN/DCRNN_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/GTS/GTS_PEMS03.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/STID/STID_PEMS-BAY.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/HI/HI_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_METR-LA_in96_out96.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_PEMS04_in96_out96.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/FEDformer/FEDformer_METR-LA_in96_out96.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Informer/Informer_ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Informer/Informer_ETTh2.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Informer/Informer_Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/Linear_ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/Linear_ETTh2.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/Linear_Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/DLinear_ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/DLinear_ETTh2.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/DLinear_Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/NLinear_Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/NLinear_ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/NLinear_ETTh2.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_ETTh2.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/FEDformer/FEDformer_ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/FEDformer/FEDformer_ETTh2.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/FEDformer/FEDformer_Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Pyraformer/Pyraformer_ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/FEDformer/FEDformer_Weather.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/STID/STID_ExchangeRate.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/DGCRN/DGCRN_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/MTGNN/MTGNN_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/MegaCRN/MegaCRN_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Informer/Informer_Weather.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Pyraformer/Pyraformer_Weather.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/atfm_04.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/MLP/MLP_METR_LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Crossformer/Crossformer_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/STNorm/STNorm_HangST.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Pyraformer/Pyraformer_METR-LA_in96_out96.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/PatchTST/PatchTST_ETTh1.py", help="training config")
    parser.add_argument("-c", "--cfg", default="examples/LSTNN/LSTNN_PEMS04.py", help="training config")
    parser.add_argument("--gpus", default="1", help="visible gpus (CUDA_VISIBLE_DEVICES)")
    return parser.parse_args()


def resolve_cuda_devices(gpus: Optional[str]) -> Optional[str]:
    """Apply CUDA mask before importing torch; return value for easytorch."""
    if gpus is None:
        return None
    gpus = str(gpus)
    preset = os.environ.get("CUDA_VISIBLE_DEVICES")
    # Parent launchers (horizon/ablation scripts) often pin the physical GPU in the
    # subprocess env and pass --gpus 0 to select logical cuda:0. Do not overwrite.
    if preset and gpus == "0":
        return preset
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    return gpus


def main():
    args = parse_args()
    devices = resolve_cuda_devices(args.gpus)

    import torch
    from basicts import launch_training

    torch.set_num_threads(1)  # aviod high cpu avg usage
    launch_training(args.cfg, devices)


if __name__ == "__main__":
    main()
