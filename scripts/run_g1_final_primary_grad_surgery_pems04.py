#!/usr/bin/env python3
"""G1_final_primary_grad_surgery end-to-end training on PeMS04 (16/32/64)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_PY = ROOT / "examples" / "run.py"
DEFAULT_HORIZONS = [16, 32, 64]
DEFAULT_SEEDS = [1]
DEFAULT_GPUS = ["0"]

HORIZON_CONFIGS: dict[int, dict[str, Any]] = {
    16: {
        "base_cfg": ROOT / "examples" / "ChainForecasting" / "ChainForecasting_PEMS04_16to16.py",
        "chain_lengths": [4, 8, 16],
    },
    32: {
        "base_cfg": ROOT / "examples" / "ChainForecasting" / "ChainForecasting_PEMS04_16to32.py",
        "chain_lengths": [8, 16, 32],
    },
    64: {
        "base_cfg": ROOT / "examples" / "ChainForecasting" / "ChainForecasting_PEMS04_16to64.py",
        "chain_lengths": [16, 32, 64],
    },
}

G1_PRIMARY_GRAD_SURGERY_PARAM: dict[str, Any] = {
    "variant_name": "G1_final_primary_grad_surgery",
    "spatial_placement": "final",
    "post_spatial_mode": "adaptive_only",
    "unified_aux_loss_mode": "none",
    "final_primary_grad_surgery": True,
    "aux_grad_max_ratio": 0.2,
    "use_prev_condition": True,
    "use_extra_prior_input": False,
    "spatial_graph_loss_weights": [0.0, 0.0, 0.0],
    "chain_loss_weights": [0.0, 0.0, 0.0],
}

WORK_DIR = ROOT / "experiments" / "g1_final_primary_grad_surgery"
CKPT_ROOT = ROOT / "checkpoints" / "g1_final_primary_grad_surgery"
LOG_ROOT = ROOT / "logs" / "g1_final_primary_grad_surgery"


def dataset_num_channels(dataset_name: str = "PEMS04") -> int:
    audit = ROOT / "datasets" / dataset_name / "protocol_audit.json"
    if audit.is_file():
        return int(json.loads(audit.read_text(encoding="utf-8"))["num_channels"])
    return 3


def resolve_input_channels(use_extra_prior: bool, num_channels: int) -> tuple[list[int], int]:
    if num_channels >= 4 and use_extra_prior:
        return [0, 1, 2, 3], 4
    return [0, 1, 2], 3


def _py_literal(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, list):
        return repr(v)
    return str(v)


def generate_config(
    horizon: int,
    seed: int,
    work_dir: Path,
    ckpt_root: Path,
) -> Path:
    hspec = HORIZON_CONFIGS[horizon]
    base_cfg = hspec["base_cfg"]
    content = base_cfg.read_text(encoding="utf-8")
    model_param = dict(G1_PRIMARY_GRAD_SURGERY_PARAM)

    num_channels = dataset_num_channels("PEMS04")
    use_prior = bool(model_param.get("use_extra_prior_input", False))
    forward_features, input_dim = resolve_input_channels(use_prior, num_channels)

    ckpt_rel = os.path.join("checkpoints", "g1_final_primary_grad_surgery", f"h{horizon}", f"seed{seed}")
    variant_name = model_param["variant_name"]
    lines = [
        "",
        "# ===== G1_final_primary_grad_surgery overrides (auto-generated) =====",
        "from basicts.runners import G1FinalPrimaryGradSurgeryRunner",
        "CFG.RUNNER = G1FinalPrimaryGradSurgeryRunner",
        f"CFG.ENV.SEED = {seed}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")',
        f'CFG.DESCRIPTION = "{variant_name} PeMS04 h{horizon} seed{seed}"',
        f'CFG.MODEL.FORWARD_FEATURES = {_py_literal(forward_features)}',
        f'CFG.MODEL.PARAM["input_dim"] = {input_dim}',
        f'CFG.MODEL.PARAM["main_input_dim"] = 3',
        f"CFG.TEST.EVALUATION_HORIZONS = list(range(1, {horizon + 1}))",
    ]
    for key, val in model_param.items():
        lines.append(f'CFG.MODEL.PARAM["{key}"] = {_py_literal(val)}')
    lines.append(f'CFG.MODEL.PARAM["chain_lengths"] = {_py_literal(hspec["chain_lengths"])}')

    out_dir = work_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"h{horizon}_{variant_name}_seed{seed}.py"
    if "from easydict import EasyDict" not in content:
        content = "from easydict import EasyDict\n" + content
    out_path.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_one(
    horizon: int,
    seed: int,
    gpu: str,
    work_dir: Path,
    ckpt_root: Path,
    dry_run: bool,
) -> None:
    cfg_path = generate_config(horizon, seed, work_dir, ckpt_root)
    rel_cfg = cfg_path.relative_to(ROOT)
    cmd = [sys.executable, str(RUN_PY), "--cfg", str(rel_cfg), "--gpus", "0"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n=== {G1_PRIMARY_GRAD_SURGERY_PARAM['variant_name']} h{horizon} seed={seed} GPU={gpu} ===")
    print(f"config: {cfg_path}")
    print(f"ckpt: {ckpt_root / f'h{horizon}' / f'seed{seed}'}")
    print("cmd:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1_final_primary_grad_surgery on PeMS04")
    p.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS)
    p.add_argument("--work_dir", type=Path, default=WORK_DIR)
    p.add_argument("--ckpt_root", type=Path, default=CKPT_ROOT)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gpus = args.gpus or ["0"]
    for hi, horizon in enumerate(args.horizons):
        for seed in args.seeds:
            gpu = gpus[hi % len(gpus)]
            run_one(
                horizon=horizon,
                seed=seed,
                gpu=gpu,
                work_dir=args.work_dir,
                ckpt_root=args.ckpt_root,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
