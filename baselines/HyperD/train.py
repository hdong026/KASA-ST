#!/usr/bin/env python3
"""Official-style HyperD training entry (BasicTS/EasyTorch)."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from basicts import launch_training


def parse_args():
    parser = ArgumentParser(description="Run HyperD baseline in KASA-ST / BasicTS framework")
    parser.add_argument(
        "-c",
        "--cfg",
        default="baselines/HyperD/PEMS04.py",
        help="HyperD config path relative to repo root",
    )
    parser.add_argument("-g", "--gpus", default="0", help="visible gpus")
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    launch_training(args.cfg, args.gpus, node_rank=0)


if __name__ == "__main__":
    main()
