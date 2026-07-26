import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.HyperD.hyperd_config import build_cfg

CFG = build_cfg("PEMS04")
