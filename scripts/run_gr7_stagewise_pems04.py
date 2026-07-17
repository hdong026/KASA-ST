#!/usr/bin/env python3
"""GR7 stagewise training on PeMS04 (16/32/64 horizons)."""
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

from basicts.runners.stagewise_training import (  # noqa: E402
    STAGE_ORDER,
    default_stage_ckpt_path,
    resolve_load_checkpoint,
)

RUN_PY = ROOT / "examples" / "run.py"
DEFAULT_WORK_DIR = ROOT / "experiments" / "gr7_stagewise"
DEFAULT_CKPT_ROOT = ROOT / "checkpoints" / "gr7_stagewise"
DEFAULT_HORIZONS = [16, 32, 64]
DEFAULT_SEEDS = [1]
DEFAULT_GPUS = ["0"]

STAGE_EPOCHS: dict[str, int] = {
    "T1": 30,
    "T2": 40,
    "T3": 60,
    "S14": 20,
    "S12": 30,
    "S1": 40,
    "FT": 15,
}

STAGE_MILESTONES: dict[str, list[int]] = {
    "T1": [1, 15, 25],
    "T2": [1, 20, 32],
    "T3": [1, 30, 48],
    "S14": [1, 10, 16],
    "S12": [1, 15, 25],
    "S1": [1, 20, 32],
    "FT": [1, 8, 12],
}

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

GR7_STAGEWISE_PARAM: dict[str, Any] = {
    "variant_name": "GR7_stagewise",
    "spatial_placement": "temporal_first_graph_resolution",
    "post_spatial_mode": "adaptive_only",
    "graph_resolution_ratios": [0.25, 0.50, 1.00],
    "graph_resolution_topks": [4, 8, 16],
    "graph_resolution_alphas": [0.03, 0.06, 0.10],
    "graph_resolution_betas": [1.0, 1.0, 1.0],
    "graph_resolution_rhos": [0.25, 0.50, 1.00],
    "graph_cluster_method": "current",
    "clustering_seed": 0,
    "dataset_name": "PEMS04",
    "cluster_road_distance_path": "datasets/raw_data/PEMS04/adj_PEMS04_distance.pkl",
    "cluster_sigma_d": 0.5,
    "cluster_delta_4": 0.8,
    "cluster_delta_2": 0.5,
    "graph_resolution_skip_final_identity": False,
    "unified_aux_loss_mode": "none",
    "spatial_graph_loss_weights": [0.0, 0.0, 0.0],
    "use_extra_prior_input": False,
}


def dataset_num_channels(dataset_name: str = "PEMS04") -> int:
    audit = ROOT / "datasets" / dataset_name / "protocol_audit.json"
    if audit.is_file():
        return int(json.loads(audit.read_text(encoding="utf-8"))["num_channels"])
    return 3


def resolve_input_channels(use_extra_prior: bool, num_channels: int) -> tuple[list[int], int]:
    """No-prior experiments must not read channel 3 even on 4-channel datasets."""
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


def stage_num_epochs(stage: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    return STAGE_EPOCHS[str(stage).upper()]


def generate_stage_config(
    horizon: int,
    stage: str,
    seed: int,
    work_dir: Path,
    ckpt_root: Path,
    num_epochs: int | None,
    freeze_previous: bool,
    detach_previous: bool,
    fine_tune_lr_scale: float,
    load_checkpoint: str | None,
    save_checkpoint: str | None,
) -> Path:
    stage = str(stage).upper()
    hspec = HORIZON_CONFIGS[horizon]
    base_cfg = hspec["base_cfg"]
    content = base_cfg.read_text(encoding="utf-8")
    epochs = stage_num_epochs(stage, num_epochs)
    milestones = STAGE_MILESTONES[stage]

    if not load_checkpoint:
        load_checkpoint = resolve_load_checkpoint(stage, str(ckpt_root), horizon, seed)
    if not save_checkpoint:
        save_checkpoint = default_stage_ckpt_path(str(ckpt_root), horizon, seed, stage)

    num_channels = dataset_num_channels(GR7_STAGEWISE_PARAM["dataset_name"])
    use_prior = bool(GR7_STAGEWISE_PARAM.get("use_extra_prior_input", False))
    forward_features, input_dim = resolve_input_channels(use_prior, num_channels)

    ckpt_rel = os.path.join("checkpoints", "gr7_stagewise", f"h{horizon}", stage, f"seed{seed}")
    lines = [
        "",
        "# ===== GR7_stagewise overrides (auto-generated) =====",
        f"CFG.ENV.SEED = {seed}",
        f"CFG.TRAIN.NUM_EPOCHS = {epochs}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")',
        f'CFG.DESCRIPTION = "GR7_stagewise PeMS04 h{horizon} stage={stage} seed{seed}"',
        f'CFG.MODEL.FORWARD_FEATURES = {_py_literal(forward_features)}',
        f'CFG.MODEL.PARAM["input_dim"] = {input_dim}',
        f'CFG.MODEL.PARAM["main_input_dim"] = 3',
    ]
    for key, val in GR7_STAGEWISE_PARAM.items():
        lines.append(f'CFG.MODEL.PARAM["{key}"] = {_py_literal(val)}')
    lines.append(f'CFG.MODEL.PARAM["chain_lengths"] = {_py_literal(hspec["chain_lengths"])}')
    lines.append('CFG.MODEL.PARAM["chain_loss_weights"] = [0.0, 0.0, 0.0]')

    base_lr = 0.002
    if stage == "FT":
        lines.append(f'CFG.TRAIN.OPTIM.PARAM["lr"] = {base_lr * fine_tune_lr_scale}')
    else:
        lines.append(f'CFG.TRAIN.OPTIM.PARAM["lr"] = {base_lr}')

    lines.extend(
        [
            f'CFG.TRAIN.LR_SCHEDULER.PARAM["milestones"] = {_py_literal(milestones)}',
            "CFG.TRAIN.STAGEWISE = EasyDict()",
            "CFG.TRAIN.STAGEWISE.enabled = True",
            f"CFG.TRAIN.STAGEWISE.stage = {_py_literal(stage)}",
            f"CFG.TRAIN.STAGEWISE.freeze_previous = {_py_literal(freeze_previous)}",
            f"CFG.TRAIN.STAGEWISE.detach_previous = {_py_literal(detach_previous)}",
            f"CFG.TRAIN.STAGEWISE.fine_tune_lr_scale = {fine_tune_lr_scale}",
            "CFG.TRAIN.STAGEWISE.variant_name = 'GR7_stagewise'",
            f"CFG.TRAIN.STAGEWISE.ckpt_root = {_py_literal(str(ckpt_root))}",
            f"CFG.TRAIN.STAGEWISE.load_checkpoint = {_py_literal(load_checkpoint)}",
            f"CFG.TRAIN.STAGEWISE.save_checkpoint = {_py_literal(save_checkpoint)}",
        ]
    )

    out_dir = work_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"h{horizon}_GR7_stagewise_{stage}_seed{seed}.py"
    if "from easydict import EasyDict" not in content:
        content = "from easydict import EasyDict\n" + content
    out_path.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_stage(
    horizon: int,
    stage: str,
    seed: int,
    gpu: str,
    work_dir: Path,
    ckpt_root: Path,
    num_epochs: int | None,
    freeze_previous: bool,
    detach_previous: bool,
    fine_tune_lr_scale: float,
    load_checkpoint: str | None,
    save_checkpoint: str | None,
    dry_run: bool,
) -> None:
    cfg_path = generate_stage_config(
        horizon=horizon,
        stage=stage,
        seed=seed,
        work_dir=work_dir,
        ckpt_root=ckpt_root,
        num_epochs=num_epochs,
        freeze_previous=freeze_previous,
        detach_previous=detach_previous,
        fine_tune_lr_scale=fine_tune_lr_scale,
        load_checkpoint=load_checkpoint,
        save_checkpoint=save_checkpoint,
    )
    rel_cfg = cfg_path.relative_to(ROOT)
    cmd = [sys.executable, str(RUN_PY), "--cfg", str(rel_cfg), "--gpus", "0"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    epochs = stage_num_epochs(stage, num_epochs)
    print(f"\n=== GR7_stagewise h{horizon} stage={stage} seed={seed} GPU={gpu} epochs={epochs} ===")
    print(f"config: {cfg_path}")
    print(f"load: {load_checkpoint or resolve_load_checkpoint(stage, str(ckpt_root), horizon, seed)}")
    print(f"save: {save_checkpoint or default_stage_ckpt_path(str(ckpt_root), horizon, seed, stage)}")
    print("cmd:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GR7 stagewise training on PeMS04")
    p.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS)
    p.add_argument("--stage", choices=STAGE_ORDER, default=None)
    p.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=None)
    p.add_argument("--run_stagewise_all", action="store_true")
    p.add_argument("--load_checkpoint", default=None)
    p.add_argument("--save_checkpoint", default=None)
    p.add_argument("--freeze_previous", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--detach_previous", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fine_tune_lr_scale", type=float, default=0.1)
    p.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override per-stage default epochs (T1=30,T2=40,T3=60,S14=20,S12=30,S1=40,FT=15)",
    )
    p.add_argument("--work_dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--ckpt_root", type=Path, default=DEFAULT_CKPT_ROOT)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_stagewise_all:
        stages = list(STAGE_ORDER)
    elif args.stages:
        stages = list(args.stages)
    elif args.stage:
        stages = [args.stage]
    else:
        raise SystemExit("Specify --stage, --stages, or --run_stagewise_all")

    gpus = args.gpus or ["0"]
    for hi, horizon in enumerate(args.horizons):
        for seed in args.seeds:
            gpu = gpus[hi % len(gpus)]
            prev_explicit_load = args.load_checkpoint
            for stage in stages:
                load_ckpt = prev_explicit_load if stage == stages[0] else None
                save_ckpt = args.save_checkpoint if len(stages) == 1 else None
                run_stage(
                    horizon=horizon,
                    stage=stage,
                    seed=seed,
                    gpu=gpu,
                    work_dir=args.work_dir,
                    ckpt_root=args.ckpt_root,
                    num_epochs=args.num_epochs,
                    freeze_previous=args.freeze_previous,
                    detach_previous=args.detach_previous,
                    fine_tune_lr_scale=args.fine_tune_lr_scale,
                    load_checkpoint=load_ckpt,
                    save_checkpoint=save_ckpt,
                    dry_run=args.dry_run,
                )
                prev_explicit_load = None


if __name__ == "__main__":
    main()
