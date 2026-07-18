#!/usr/bin/env python3
"""G1 stagewise training on PeMS04 (16/32/64 horizons, T1->T2->T3->S1)."""
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

from basicts.runners.g1_stagewise_training import (  # noqa: E402
    G1_STAGE_CHOICES,
    G1_STAGE_ORDER,
    default_stage_ckpt_path,
    resolve_load_checkpoint,
    stage_best_ckpt_name,
)

RUN_PY = ROOT / "examples" / "run.py"
DEFAULT_HORIZONS = [16, 32, 64]
DEFAULT_SEEDS = [1]
DEFAULT_GPUS = ["0"]

STAGE_EPOCHS: dict[str, int] = {
    "T1": 30,
    "T2": 40,
    "T3": 80,
    "S1": 40,
}

STAGE_MILESTONES: dict[str, list[int]] = {
    "T1": [1, 15, 25],
    "T2": [1, 20, 32],
    "T3": [1, 40, 65],
    "S1": [1, 20, 32],
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

G1_STAGEWISE_PARAM: dict[str, Any] = {
    "variant_name": "G1_stagewise",
    "base_variant": "G1_final_adaptive",
    "spatial_placement": "final",
    "post_spatial_mode": "adaptive_only",
    "unified_aux_loss_mode": "none",
    "use_prev_condition": True,
    "use_extra_prior_input": False,
    "spatial_graph_loss_weights": [0.0, 0.0, 0.0],
}

WORK_DIR = ROOT / "experiments" / "g1_stagewise"
CKPT_ROOT = ROOT / "checkpoints" / "g1_stagewise"


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


def stage_num_epochs(stage: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    return STAGE_EPOCHS[str(stage).upper()]


def stage_milestones(stage: str) -> list[int]:
    return list(STAGE_MILESTONES[str(stage).upper()])


def resolve_stage_list(
    run_stagewise_all: bool,
    stage: str | None,
    stages: list[str] | None,
) -> list[str]:
    if run_stagewise_all:
        return list(G1_STAGE_ORDER)
    if stages:
        return [str(s).upper() for s in stages]
    if stage:
        return [str(stage).upper()]
    raise SystemExit("Specify --stage, --stages, or --run_stagewise_all")


def generate_stage_config(
    horizon: int,
    stage: str,
    seed: int,
    num_epochs: int | None,
    freeze_previous: bool,
    detach_previous: bool,
    load_checkpoint: str | None,
    save_checkpoint: str | None,
    work_dir: Path,
    ckpt_root: Path,
) -> Path:
    stage = str(stage).upper()
    if stage not in G1_STAGE_CHOICES:
        raise ValueError(f"Invalid G1 stagewise stage: {stage}")

    model_param = dict(G1_STAGEWISE_PARAM)
    hspec = HORIZON_CONFIGS[horizon]
    base_cfg = hspec["base_cfg"]
    content = base_cfg.read_text(encoding="utf-8")
    epochs = stage_num_epochs(stage, num_epochs)
    milestones = stage_milestones(stage)

    if not load_checkpoint:
        load_checkpoint = resolve_load_checkpoint(stage, str(ckpt_root), horizon, seed)
    if not save_checkpoint:
        save_checkpoint = default_stage_ckpt_path(str(ckpt_root), horizon, seed, stage)

    num_channels = dataset_num_channels("PEMS04")
    use_prior = bool(model_param.get("use_extra_prior_input", False))
    forward_features, input_dim = resolve_input_channels(use_prior, num_channels)

    ckpt_rel = os.path.join("checkpoints", "g1_stagewise", f"H{horizon}", f"seed{seed}", stage)
    variant_name = model_param["variant_name"]
    lines = [
        "",
        "# ===== G1_stagewise overrides (auto-generated) =====",
        "from basicts.runners import G1StagewiseRunner",
        "CFG.RUNNER = G1StagewiseRunner",
        f"CFG.ENV.SEED = {seed}",
        f"CFG.TRAIN.NUM_EPOCHS = {epochs}",
        f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")',
        f'CFG.DESCRIPTION = "{variant_name} PeMS04 h{horizon} stage={stage} seed{seed}"',
        f'CFG.MODEL.FORWARD_FEATURES = {_py_literal(forward_features)}',
        f'CFG.MODEL.PARAM["input_dim"] = {input_dim}',
        f'CFG.MODEL.PARAM["main_input_dim"] = 3',
    ]
    for key, val in model_param.items():
        lines.append(f'CFG.MODEL.PARAM["{key}"] = {_py_literal(val)}')
    lines.append(f'CFG.MODEL.PARAM["chain_lengths"] = {_py_literal(hspec["chain_lengths"])}')
    lines.append('CFG.MODEL.PARAM["chain_loss_weights"] = [0.0, 0.0, 0.0]')
    lines.append('CFG.TRAIN.OPTIM.PARAM["lr"] = 0.002')
    lines.append(f"CFG.TEST.EVALUATION_HORIZONS = list(range(1, {horizon + 1}))")
    lines.extend(
        [
            f'CFG.TRAIN.LR_SCHEDULER.PARAM["milestones"] = {_py_literal(milestones)}',
            "CFG.TRAIN.STAGEWISE = EasyDict()",
            "CFG.TRAIN.STAGEWISE.enabled = True",
            f"CFG.TRAIN.STAGEWISE.stage = {_py_literal(stage)}",
            f"CFG.TRAIN.STAGEWISE.freeze_previous = {_py_literal(freeze_previous)}",
            f"CFG.TRAIN.STAGEWISE.detach_previous = {_py_literal(detach_previous)}",
            'CFG.TRAIN.STAGEWISE.stage_sequence = "g1"',
            "CFG.TRAIN.STAGEWISE.train_shared_temporal = True",
            f"CFG.TRAIN.STAGEWISE.variant_name = {_py_literal(variant_name)}",
            f"CFG.TRAIN.STAGEWISE.base_variant = {_py_literal(model_param['base_variant'])}",
            f"CFG.TRAIN.STAGEWISE.ckpt_root = {_py_literal(str(ckpt_root))}",
            f"CFG.TRAIN.STAGEWISE.load_checkpoint = {_py_literal(load_checkpoint)}",
            f"CFG.TRAIN.STAGEWISE.save_checkpoint = {_py_literal(save_checkpoint)}",
        ]
    )

    out_dir = work_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"h{horizon}_G1_stagewise_{stage}_seed{seed}.py"
    if "from easydict import EasyDict" not in content:
        content = "from easydict import EasyDict\n" + content
    out_path.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def stage_completed(
    horizon: int,
    seed: int,
    stage: str,
    ckpt_root: Path,
) -> bool:
    ckpt = ckpt_root / f"H{horizon}" / f"seed{seed}" / stage_best_ckpt_name(stage)
    return ckpt.is_file()


def run_stage(
    horizon: int,
    stage: str,
    seed: int,
    gpu: str,
    num_epochs: int | None,
    freeze_previous: bool,
    detach_previous: bool,
    load_checkpoint: str | None,
    save_checkpoint: str | None,
    work_dir: Path,
    ckpt_root: Path,
    dry_run: bool,
    skip_existing: bool,
    show_errors: bool,
) -> int:
    if skip_existing and stage_completed(horizon, seed, stage, ckpt_root):
        print(
            f"[skip] G1_stagewise h{horizon} stage={stage} seed={seed} "
            f"(exists: {ckpt_root / f'H{horizon}' / f'seed{seed}' / stage_best_ckpt_name(stage)})"
        )
        return 0

    cfg_path = generate_stage_config(
        horizon=horizon,
        stage=stage,
        seed=seed,
        num_epochs=num_epochs,
        freeze_previous=freeze_previous,
        detach_previous=detach_previous,
        load_checkpoint=load_checkpoint,
        save_checkpoint=save_checkpoint,
        work_dir=work_dir,
        ckpt_root=ckpt_root,
    )
    rel_cfg = cfg_path.relative_to(ROOT)
    cmd = [sys.executable, str(RUN_PY), "--cfg", str(rel_cfg), "--gpus", "0"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    epochs = stage_num_epochs(stage, num_epochs)
    milestones = stage_milestones(stage)
    load_path = load_checkpoint or resolve_load_checkpoint(stage, str(ckpt_root), horizon, seed)
    save_path = save_checkpoint or default_stage_ckpt_path(str(ckpt_root), horizon, seed, stage)
    print(f"\n=== G1_stagewise h{horizon} stage={stage} seed={seed} GPU={gpu} ===")
    print("base_variant=G1_final_adaptive stage_sequence=T1->T2->T3->S1")
    print(f"epochs={epochs} lr=0.002 milestones={milestones}")
    print(f"evaluation_horizons=1..{horizon}")
    print(f"config: {cfg_path}")
    print(f"load: {load_path}")
    print(f"save_best: {save_path} ({stage_best_ckpt_name(stage)})")
    print("cmd:", " ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    if proc.returncode != 0:
        msg = f"G1_stagewise h{horizon} stage={stage} seed={seed} failed (exit={proc.returncode})"
        if show_errors:
            raise SystemExit(msg)
        print(f"[error] {msg}")
        return proc.returncode
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1 stagewise training on PeMS04 (T1->T2->T3->S1)")
    p.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS)
    p.add_argument("--stage", choices=G1_STAGE_CHOICES, default=None)
    p.add_argument("--stages", nargs="+", choices=G1_STAGE_CHOICES, default=None)
    p.add_argument("--run_stagewise_all", action="store_true")
    p.add_argument("--load_checkpoint", default=None)
    p.add_argument("--save_checkpoint", default=None)
    p.add_argument("--freeze_previous", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--detach_previous", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override per-stage default epochs (T1=30,T2=40,T3=80,S1=40)",
    )
    p.add_argument("--work_dir", type=Path, default=WORK_DIR)
    p.add_argument("--ckpt_root", type=Path, default=CKPT_ROOT)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--show_errors", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stages = resolve_stage_list(args.run_stagewise_all, args.stage, args.stages)
    work_dir = Path(args.work_dir)
    ckpt_root = Path(args.ckpt_root)
    gpus = args.gpus or ["0"]

    exit_code = 0
    for hi, horizon in enumerate(args.horizons):
        for seed in args.seeds:
            gpu = gpus[hi % len(gpus)]
            prev_explicit_load = args.load_checkpoint
            for stage in stages:
                load_ckpt = prev_explicit_load if stage == stages[0] else None
                save_ckpt = args.save_checkpoint if len(stages) == 1 else None
                rc = run_stage(
                    horizon=horizon,
                    stage=stage,
                    seed=seed,
                    gpu=gpu,
                    num_epochs=args.num_epochs,
                    freeze_previous=args.freeze_previous,
                    detach_previous=args.detach_previous,
                    load_checkpoint=load_ckpt,
                    save_checkpoint=save_ckpt,
                    work_dir=work_dir,
                    ckpt_root=ckpt_root,
                    dry_run=args.dry_run,
                    skip_existing=args.skip_existing,
                    show_errors=args.show_errors,
                )
                if rc != 0:
                    exit_code = rc
                    if args.show_errors:
                        raise SystemExit(exit_code)
                prev_explicit_load = None

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
