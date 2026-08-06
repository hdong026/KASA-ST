#!/usr/bin/env python3
"""Budget-conditioned adaptive F2F runner with unique run signatures.

Isolates forced-route experiments so different routes never share
checkpoint / log / temp-config / skip identity.

Does not alter formal ChainForecasting variants' default path layout.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_BASE = ROOT / "scripts" / "run_chain_forecasting_horizon.py"
_spec = importlib.util.spec_from_file_location("run_chain_forecasting_horizon", _BASE)
base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(base)

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    build_run_signature,
    parse_candidate_routes,
    parse_route,
    route_to_key,
    validate_route,
)

BASE_VARIANT = "chain_budget_conditioned_adaptive_f2f_kasa_condition_adapter_token_loss"


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def experiment_paths(dataset: str, horizon: int, experiment_tag: str, seed: int) -> dict[str, Path]:
    """Isolated artifact layout for one budget-F2F run."""
    ckpt = (
        ROOT
        / "checkpoints"
        / dataset
        / f"H{horizon}"
        / "budget_f2f"
        / experiment_tag
        / f"seed{seed}"
    )
    cfg_dir = ROOT / "generated" / f"temp_configs_budget_f2f_{dataset.lower()}"
    log_dir = ROOT / "results" / f"budget_f2f_{dataset.lower()}_logs"
    cfg = cfg_dir / f"H{horizon}_{experiment_tag}_seed{seed}.py"
    log = log_dir / f"H{horizon}_{experiment_tag}_seed{seed}_gpuPLACEHOLDER.log"
    return {
        "ckpt_dir": ckpt,
        "cfg_dir": cfg_dir,
        "log_dir": log_dir,
        "cfg_path": cfg,
        "log_template": log,
    }


def unique_variant_key(experiment_tag: str) -> str:
    """Registerable VARIANT_SPECS key; keeps argparse choices working."""
    return f"{BASE_VARIANT}__{experiment_tag}"


def install_path_hooks(dataset: str, tag_by_variant: dict[str, str]) -> None:
    """Override shared runner path helpers for budget unique variants only."""

    def _tag(variant: str) -> str:
        if variant in tag_by_variant:
            return tag_by_variant[variant]
        if variant.startswith(BASE_VARIANT + "__"):
            return variant[len(BASE_VARIANT) + 2 :]
        return variant

    def ckpt_dir_for(horizon: int, variant: str, seed: int) -> Path:
        if not str(variant).startswith(BASE_VARIANT):
            return base.CKPT_ROOT / f"h{horizon}" / f"{variant}_seed{seed}"
        return experiment_paths(dataset, horizon, _tag(variant), seed)["ckpt_dir"]

    def temp_cfg_path(horizon: int, variant: str, seed: int) -> Path:
        if not str(variant).startswith(BASE_VARIANT):
            base.TEMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
            return base.TEMP_CFG_DIR / f"h{horizon}_{variant}_seed{seed}.py"
        paths = experiment_paths(dataset, horizon, _tag(variant), seed)
        paths["cfg_dir"].mkdir(parents=True, exist_ok=True)
        return paths["cfg_path"]

    def job_name(horizon: int, variant: str, seed: int) -> str:
        if not str(variant).startswith(BASE_VARIANT):
            return f"h{horizon}_{variant}_seed{seed}"
        return f"H{horizon}_{_tag(variant)}_seed{seed}"

    def wrapper_log_path(horizon: int, variant: str, seed: int):
        if not str(variant).startswith(BASE_VARIANT):
            matches = sorted(base.LOG_DIR.glob(f"h{horizon}_{variant}_seed{seed}_gpu*.log"))
            return matches[-1] if matches else None
        log_dir = experiment_paths(dataset, horizon, _tag(variant), seed)["log_dir"]
        matches = sorted(log_dir.glob(f"H{horizon}_{_tag(variant)}_seed{seed}_gpu*.log"))
        return matches[-1] if matches else None

    _orig_generate = base.generate_temp_config

    def generate_temp_config(horizon: int, variant: str, seed: int) -> Path:
        out = _orig_generate(horizon, variant, seed)
        if not str(variant).startswith(BASE_VARIANT):
            return out
        # Force CKPT_SAVE_DIR / DESCRIPTION onto isolated budget paths.
        # Base PEMS04 cfg uses a multi-line os.path.join(...); replace the whole
        # statement instead of only the first line (avoids IndentationError).
        ckpt = ckpt_dir_for(horizon, variant, seed)
        ckpt_rel = os.path.relpath(ckpt, ROOT).replace("\\", "/")
        tag = _tag(variant)
        text = out.read_text(encoding="utf-8")
        import re

        ckpt_stmt = f'CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")'
        pattern = re.compile(
            r"^CFG\.TRAIN\.CKPT_SAVE_DIR\s*=\s*os\.path\.join\([\s\S]*?^\)\s*$",
            re.MULTILINE,
        )
        new_text, n_sub = pattern.subn(ckpt_stmt, text, count=1)
        if n_sub == 0:
            # Fallback: line-wise replace of any CKPT_SAVE_DIR assignment heads,
            # dropping indented continuation lines until a closing ')'.
            lines = []
            skipping_join = False
            replaced = False
            for line in text.splitlines():
                if skipping_join:
                    if line.strip() == ")":
                        skipping_join = False
                    continue
                if "CFG.TRAIN.CKPT_SAVE_DIR" in line:
                    lines.append(ckpt_stmt)
                    replaced = True
                    # If this line opens a multi-line join without closing, skip rest.
                    if "os.path.join(" in line and ")" not in line:
                        skipping_join = True
                    continue
                lines.append(line)
            if not replaced:
                lines.append(ckpt_stmt)
            new_text = "\n".join(lines) + "\n"
        else:
            # Drop any later duplicate CKPT_SAVE_DIR overrides (keep first / our stmt).
            lines = []
            seen = False
            for line in new_text.splitlines():
                if "CFG.TRAIN.CKPT_SAVE_DIR" in line:
                    if seen:
                        continue
                    lines.append(ckpt_stmt)
                    seen = True
                else:
                    lines.append(line)
            new_text = "\n".join(lines) + "\n"

        if f'CFG.DESCRIPTION = "budget_f2f {tag}' not in new_text:
            new_text += f'\nCFG.DESCRIPTION = "budget_f2f {tag} H={horizon} seed={seed}"\n'

        target = temp_cfg_path(horizon, variant, seed)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_text, encoding="utf-8")
        return target

    def run_one(horizon: int, variant: str, cfg_path: Path, gpu: str, seed: int) -> dict:
        """Same as base.run_one but with budget log directory."""
        if not str(variant).startswith(BASE_VARIANT):
            return base.run_one(horizon, variant, cfg_path, gpu, seed)

        paths = experiment_paths(dataset, horizon, _tag(variant), seed)
        paths["log_dir"].mkdir(parents=True, exist_ok=True)
        paths["ckpt_dir"].mkdir(parents=True, exist_ok=True)
        log_file = paths["log_dir"] / f"H{horizon}_{_tag(variant)}_seed{seed}_gpu{gpu}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            sys.executable,
            str(base.RUN_PY),
            "--cfg",
            base.cfg_for_easytorch(cfg_path),
            "--gpus",
            str(gpu),
        ]
        row = base.base_row(horizon, variant, seed, cfg_path)
        row["log_file"] = str(log_file)
        row["ckpt_dir"] = str(paths["ckpt_dir"])
        row["status"] = "running"
        try:
            with open(log_file, "w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT
                )
            metrics = base.parse_metrics(
                base.collect_log_text(horizon, variant, seed, log_file)
            )
            row.update(metrics)
            eval_steps = base.HORIZON_EVAL_STEPS[horizon]
            hz = base.parse_horizon_mae(
                base.collect_log_text(horizon, variant, seed, log_file), eval_steps
            )
            for step in eval_steps:
                row[f"horizon_mae_{step}"] = hz.get(step)
            if proc.returncode != 0:
                row["status"] = f"exit_{proc.returncode}"
                row["failed"] = 1
                err = paths["log_dir"] / f"H{horizon}_{_tag(variant)}_seed{seed}_error_tail.txt"
                if log_file.is_file():
                    err.write_text(
                        "\n".join(log_file.read_text(errors="replace").splitlines()[-200:])
                        + "\n",
                        encoding="utf-8",
                    )
                row["error_tail_file"] = str(err)
            elif metrics["mae"] is None:
                row["status"] = "failed_no_metrics"
                row["failed"] = 1
            else:
                row["status"] = "ok"
                row["failed"] = 0
        except Exception as e:
            row["status"] = f"error:{e}"
            row["failed"] = 1
        return row

    base.ckpt_dir_for = ckpt_dir_for
    base.temp_cfg_path = temp_cfg_path
    base.job_name = job_name
    base.wrapper_log_path = wrapper_log_path
    base.generate_temp_config = generate_temp_config
    base.run_one = run_one


def archive_path(path: Path, reason: str = "overwrite") -> Path | None:
    """Rename an existing path aside (never delete). Returns archive path or None."""
    if not path.exists():
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    archived = path.with_name(f"{path.name}.archived_{reason}_{ts}")
    # Avoid collision
    i = 1
    while archived.exists():
        archived = path.with_name(f"{path.name}.archived_{reason}_{ts}_{i}")
        i += 1
    path.rename(archived)
    print(f"[overwrite] archived {path} -> {archived}")
    return archived


def prepare_fresh_run_artifacts(
    dataset: str,
    horizon: int,
    experiment_tag: str,
    seed: int,
    unique_variant: str,
) -> None:
    """Move prior ckpt/log aside so EasyTorch cannot Resume training."""
    paths = experiment_paths(dataset, horizon, experiment_tag, seed)
    archive_path(paths["ckpt_dir"], reason="overwrite")
    # Wrapper logs for any gpu id
    log_dir = paths["log_dir"]
    if log_dir.is_dir():
        for log in sorted(log_dir.glob(f"H{horizon}_{experiment_tag}_seed{seed}_gpu*.log")):
            archive_path(log, reason="overwrite")
        for err in sorted(log_dir.glob(f"H{horizon}_{experiment_tag}_seed{seed}_error_tail.txt")):
            archive_path(err, reason="overwrite")
    # Also archive any leftover default-layout collision dir if present
    legacy = (
        ROOT
        / "checkpoints"
        / f"fixed_input_horizon_{dataset.lower()}"
        / f"h{horizon}"
        / f"{unique_variant}_seed{seed}"
    )
    if legacy.exists():
        archive_path(legacy, reason="overwrite_legacy")


def enrich_row(row: dict, meta: dict) -> dict:
    out = dict(row)
    out.update(
        {
            "run_signature": meta["run_signature"],
            "experiment_tag": meta["experiment_tag"],
            "forced_route": route_to_key(meta["forced_route"]) if meta["forced_route"] else "",
            "selected_route": route_to_key(meta["forced_route"]) if meta["forced_route"] else "",
            "chain_resolutions": route_to_key(meta["forced_route"]) if meta["forced_route"] else "",
            "actual_stage_count": len(meta["forced_route"]) if meta["forced_route"] else "",
            "route_selection_mode": meta["route_selection_mode"],
            "training_phase": meta["training_phase"],
            "loss_mode": meta["loss_mode"],
            "inference_intensity": meta["inference_intensity"],
            "candidate_routes": meta["candidate_routes_str"],
            "git_commit": meta["git_commit"],
            "checkpoint_dir": row.get("ckpt_dir", ""),
            "config_path": row.get("cfg_path", ""),
            "base_variant": BASE_VARIANT,
        }
    )
    return out


def write_budget_outputs(rows: list[dict], out_csv: Path, out_md: Path) -> None:
    fields = [
        "run_signature",
        "experiment_tag",
        "dataset",
        "horizon",
        "seed",
        "base_variant",
        "forced_route",
        "selected_route",
        "chain_resolutions",
        "actual_stage_count",
        "route_selection_mode",
        "training_phase",
        "loss_mode",
        "inference_intensity",
        "candidate_routes",
        "mae",
        "rmse",
        "mape",
        "params",
        "status",
        "failed",
        "checkpoint_dir",
        "config_path",
        "log_file",
        "git_commit",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = [
        "# Budget-conditioned F2F results\n\n",
        "| experiment_tag | seed | forced_route | mae | rmse | mape | status | checkpoint_dir |\n",
        "|---|---:|---|---:|---:|---:|---|---|\n",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('experiment_tag','')} | {r.get('seed','')} | {r.get('forced_route','')} | "
            f"{_fmt(r.get('mae'))} | {_fmt(r.get('rmse'))} | {_fmt(r.get('mape'))} | "
            f"{r.get('status','')} | `{r.get('checkpoint_dir','')}` |\n"
        )
        lines.append(f"\n- run_signature: `{r.get('run_signature','')}`\n")
        lines.append(f"- config: `{r.get('config_path','')}`\n")
        lines.append(f"- git_commit: `{r.get('git_commit','')}`\n\n")
    out_md.write_text("".join(lines), encoding="utf-8")


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def is_completed_for_signature(
    horizon: int,
    unique_variant: str,
    seed: int,
    cfg_path: Path,
    run_signature: str,
) -> tuple[bool, dict | None]:
    """Skip only when this exact run_signature already finished ok."""
    row = base.summarize_row(horizon, unique_variant, seed, cfg_path)
    if row.get("status") != "ok" or row.get("mae") is None:
        return False, row
    # Require matching signature marker in cfg if present.
    try:
        text = Path(cfg_path).read_text(encoding="utf-8")
    except Exception:
        text = ""
    if "run_signature" in text and run_signature not in text:
        return False, row
    # Also require isolated ckpt dir exists for this experiment tag.
    ckpt = base.ckpt_dir_for(horizon, unique_variant, seed)
    if not ckpt.is_dir():
        return False, row
    return True, row


def build_jobs(args) -> list[dict]:
    base.activate_dataset(args.dataset)
    if BASE_VARIANT not in base.VARIANT_SPECS:
        raise KeyError(f"Missing variant {BASE_VARIANT}")

    horizon0 = int(args.horizons[0])
    if args.candidate_routes is not None:
        routes = parse_candidate_routes(args.candidate_routes, horizon0)
    else:
        routes = parse_candidate_routes(None, horizon0)
    for h in args.horizons:
        for r in routes:
            validate_route(r, horizon=h)

    forced = None
    route_mode = args.route_selection_mode
    if args.forced_route is not None:
        forced = parse_route(args.forced_route)
        for h in args.horizons:
            validate_route(forced, horizon=h)
        if forced not in routes:
            raise ValueError(f"forced-route {forced} not in candidate pool {routes}")
        route_mode = "forced"

    jobs = []
    tag_by_variant: dict[str, str] = {}
    commit = git_commit()
    for h in args.horizons:
        for seed in args.seeds:
            meta_sig = build_run_signature(
                dataset=args.dataset,
                horizon=h,
                seed=seed,
                base_variant=BASE_VARIANT,
                route_selection_mode=route_mode,
                forced_route=forced,
                training_phase=args.training_phase,
                loss_mode=args.loss_mode,
                candidate_routes=routes,
                route_granularity=args.route_granularity,
                inference_intensity=float(args.inference_intensity),
                route_cost_type=args.route_cost_type,
                route_cost_file=args.route_cost_file,
                run_tag=args.run_tag,
            )
            experiment_tag = meta_sig["experiment_tag"]
            unique = unique_variant_key(experiment_tag)
            tag_by_variant[unique] = experiment_tag

            spec = dict(base.VARIANT_SPECS[BASE_VARIANT])
            spec["candidate_routes"] = routes
            spec["training_phase"] = args.training_phase
            spec["route_selection_mode"] = route_mode
            spec["route_granularity"] = args.route_granularity
            spec["inference_intensity"] = float(args.inference_intensity)
            spec["route_cost_type"] = args.route_cost_type
            spec["loss_mode"] = args.loss_mode
            spec["chain_loss_mode"] = args.loss_mode
            spec["route_sampling"] = (
                "none" if forced is not None else args.route_sampling
            )
            spec["freeze_forecasting_backbone"] = bool(args.freeze_forecasting_backbone)
            spec["run_signature"] = meta_sig["run_signature"]
            spec["experiment_tag"] = experiment_tag
            if args.route_cost_file:
                spec["route_cost_file"] = args.route_cost_file
            if args.oracle_file:
                spec["oracle_file"] = args.oracle_file
            if args.init_checkpoint:
                spec["init_checkpoint"] = args.init_checkpoint
            if forced is not None:
                spec["forced_route"] = forced
                spec["route_selection_mode"] = "forced"
            # Keep model_name stable for easytorch; uniqueness is in paths/signature.
            base.VARIANT_SPECS[unique] = spec

            jobs.append(
                {
                    "horizon": h,
                    "seed": seed,
                    "unique_variant": unique,
                    "experiment_tag": experiment_tag,
                    "run_signature": meta_sig["run_signature"],
                    "forced_route": forced,
                    "route_selection_mode": route_mode,
                    "training_phase": args.training_phase,
                    "loss_mode": args.loss_mode,
                    "inference_intensity": float(args.inference_intensity),
                    "candidate_routes_str": "+".join(route_to_key(r) for r in routes),
                    "git_commit": commit,
                    "out": args.out,
                    "markdown": args.markdown,
                }
            )
    return jobs, tag_by_variant


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Budget-conditioned adaptive F2F experiments (signature-isolated)."
    )
    parser.add_argument("--dataset", default="PEMS04", choices=list(base.DATASET_SPECS.keys()))
    parser.add_argument("--horizons", type=int, nargs="+", default=[12])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--prepare_data", "--prepare-data", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument(
        "--training-phase",
        default="supernet",
        choices=["supernet", "planner", "joint", "eval"],
    )
    parser.add_argument("--candidate-routes", nargs="+", default=None)
    parser.add_argument("--forced-route", default=None)
    parser.add_argument(
        "--route-selection-mode",
        default="batch",
        choices=["batch", "sample", "forced"],
    )
    parser.add_argument("--route-granularity", default="batch", choices=["batch", "sample"])
    parser.add_argument("--inference-intensity", type=float, default=0.5)
    parser.add_argument("--route-cost-file", default=None)
    parser.add_argument("--route-cost-type", default="normalized_static_cost")
    parser.add_argument("--oracle-file", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--freeze-forecasting-backbone", action="store_true")
    parser.add_argument(
        "--loss-mode",
        default="dynamic_fair",
        choices=["baseline_compatible", "dynamic_fair"],
    )
    parser.add_argument(
        "--route-sampling",
        default="sandwich",
        choices=["sandwich", "random", "none"],
    )
    parser.add_argument("--run-tag", default=None, help="Optional extra signature tag.")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument(
        "--dry-run-forced-equivalence",
        action="store_true",
        help="Dry-run all candidate routes as forced jobs; verify isolated signatures/ckpts.",
    )
    # Default: do NOT skip (safe for forced-route sweeps).
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=False,
        help="Skip only when full run_signature already completed ok.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Never skip (default).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Force a fresh run: disable skip, and archive existing checkpoint/log "
            "dirs (rename, never delete) so EasyTorch cannot Resume training."
        ),
    )
    # Backward-compatible aliases
    parser.add_argument("--skip_completed", action="store_true", default=None)
    parser.add_argument("--no_skip_completed", action="store_true", default=False)
    args = parser.parse_args()

    if args.no_skip_completed:
        args.skip_existing = False
    elif args.skip_completed:
        args.skip_existing = True
    if args.overwrite:
        args.skip_existing = False

    # Expand forced-equivalence dry-run into one job per candidate route.
    if args.dry_run_forced_equivalence:
        args.dry_run = True
        base.activate_dataset(args.dataset)
        h0 = int(args.horizons[0])
        routes = (
            parse_candidate_routes(args.candidate_routes, h0)
            if args.candidate_routes is not None
            else parse_candidate_routes(None, h0)
        )
        all_jobs = []
        all_tags: dict[str, str] = {}
        saved_forced = args.forced_route
        saved_mode = args.route_selection_mode
        for r in routes:
            args.forced_route = route_to_key(r)
            args.route_selection_mode = "forced"
            jobs_i, tags_i = build_jobs(args)
            all_jobs.extend(jobs_i)
            all_tags.update(tags_i)
        args.forced_route = saved_forced
        args.route_selection_mode = saved_mode
        jobs, tag_by_variant = all_jobs, all_tags
    else:
        jobs, tag_by_variant = build_jobs(args)

    install_path_hooks(args.dataset, tag_by_variant)

    # Ensure base globals for dataset
    base.TEMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    if base.LOG_DIR is not None:
        base.LOG_DIR.mkdir(parents=True, exist_ok=True)
    if base.CKPT_ROOT is not None:
        base.CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    # Materialize configs
    for job in jobs:
        cfg = base.generate_temp_config(job["horizon"], job["unique_variant"], job["seed"])
        # Embed signature into cfg for skip matching.
        text = cfg.read_text(encoding="utf-8")
        if "run_signature" not in text:
            text += (
                f'\nCFG.MODEL.PARAM["run_signature"] = {base._py_literal(job["run_signature"])}\n'
            )
            cfg.write_text(text, encoding="utf-8")
        job["cfg_path"] = cfg
        job["ckpt_dir"] = base.ckpt_dir_for(
            job["horizon"], job["unique_variant"], job["seed"]
        )

    if args.out is None:
        if len(jobs) == 1:
            tag = jobs[0]["experiment_tag"]
            args.out = f"results/pems04_budget_f2f_{tag}_seed{jobs[0]['seed']}.csv"
        else:
            args.out = "results/pems04_budget_f2f_batch.csv"
    if args.markdown is None:
        args.markdown = str(Path(args.out).with_suffix(".md"))
    # Per-job default outs when caller did not pass --out (forced sweeps).
    user_set_out = any(
        a in {"--out", "--markdown"} for a in sys.argv[1:]
    )
    for job in jobs:
        if user_set_out:
            job["out"] = args.out
            job["markdown"] = args.markdown
        else:
            job["out"] = f"results/pems04_budget_f2f_{job['experiment_tag']}_seed{job['seed']}.csv"
            job["markdown"] = str(Path(job["out"]).with_suffix(".md"))
            if len(jobs) == 1:
                args.out = job["out"]
                args.markdown = job["markdown"]
        job["dataset"] = args.dataset

    # ---- dry-run: no training, no data prep ----
    if args.dry_run:
        print("=== budget F2F dry-run (no training) ===\n")
        sigs = []
        ckpts = []
        for job in jobs:
            done, prev = is_completed_for_signature(
                job["horizon"],
                job["unique_variant"],
                job["seed"],
                job["cfg_path"],
                job["run_signature"],
            )
            would_skip = bool(args.skip_existing and not args.overwrite and done)
            print(f"experiment_tag: {job['experiment_tag']}")
            print(f"  run_signature: {job['run_signature']}")
            print(f"  forced_route: {job['forced_route']}")
            print(f"  unique_variant: {job['unique_variant']}")
            print(f"  temp_config: {job['cfg_path']}")
            print(f"  checkpoint_dir: {job['ckpt_dir']}")
            print(f"  output_csv: {job['out']}")
            print(f"  output_md: {job['markdown']}")
            print(f"  would_skip: {would_skip}")
            ckpt_exists = Path(job["ckpt_dir"]).exists()
            print(f"  ckpt_dir_exists: {ckpt_exists}")
            if args.overwrite and ckpt_exists:
                print(
                    "  overwrite_action: ARCHIVE existing ckpt/log "
                    "(rename; EasyTorch will not Resume)"
                )
            elif ckpt_exists and not args.overwrite:
                print(
                    "  WARNING: ckpt_dir exists — without --overwrite, "
                    "EasyTorch may Resume training from it"
                )
            if would_skip and prev is not None:
                print(
                    f"  skip_hit: status={prev.get('status')} mae={prev.get('mae')} "
                    f"log={prev.get('log_file')}"
                )
            print()
            sigs.append(job["run_signature"])
            ckpts.append(str(job["ckpt_dir"]))

        if len(set(sigs)) != len(sigs):
            raise SystemExit("ERROR: duplicate run_signature among jobs")
        if len(set(ckpts)) != len(ckpts):
            raise SystemExit("ERROR: duplicate checkpoint_dir among jobs")
        print(f"unique signatures: {len(set(sigs))} / {len(sigs)}")
        print(f"unique checkpoint dirs: {len(set(ckpts))} / {len(ckpts)}")
        print("dry-run OK")
        return 0

    if args.prepare_data:
        rc = base.ensure_data(args.horizons)
        if rc != 0:
            return rc

    # Multi-job in one process (e.g. future batching) still isolates by signature.
    queue = base.GPUQueue(args.gpus)
    rows: list[dict] = []
    lock = threading.Lock()

    def worker(job: dict):
        cfg_path = job["cfg_path"]
        if args.skip_existing and not args.overwrite:
            done, prev = is_completed_for_signature(
                job["horizon"],
                job["unique_variant"],
                job["seed"],
                cfg_path,
                job["run_signature"],
            )
            if done:
                row = enrich_row(prev or {}, job)
                row["dataset"] = args.dataset
                with lock:
                    rows.append(row)
                print(
                    f"[skip] signature={job['experiment_tag']} seed={job['seed']} "
                    f"(already ok, mae={row.get('mae')})"
                )
                return
        if args.overwrite:
            prepare_fresh_run_artifacts(
                args.dataset,
                job["horizon"],
                job["experiment_tag"],
                job["seed"],
                job["unique_variant"],
            )
        gpu = queue.acquire()
        try:
            print(
                f"[start] {job['experiment_tag']} H={job['horizon']} "
                f"seed={job['seed']} gpu={gpu}"
            )
            print(f"  signature: {job['run_signature']}")
            print(f"  ckpt: {job['ckpt_dir']}")
            if args.overwrite:
                print("  mode: fresh train (overwrite archived prior ckpt/log)")
            raw = base.run_one(
                job["horizon"], job["unique_variant"], cfg_path, gpu, job["seed"]
            )
            row = enrich_row(raw, job)
            row["dataset"] = args.dataset
            # Guard: detect accidental EasyTorch resume from leftover artifacts
            log_file = row.get("log_file") or ""
            if log_file and Path(log_file).is_file():
                text = Path(log_file).read_text(errors="replace")
                if "Resume training" in text and args.overwrite:
                    row["status"] = "error:resumed_despite_overwrite"
                    row["failed"] = 1
                    print(
                        "[error] EasyTorch still resumed after overwrite; "
                        "check archived ckpt paths and CKPT_SAVE_DIR"
                    )
            with lock:
                rows.append(row)
            print(
                f"[done]  {job['experiment_tag']} seed={job['seed']} "
                f"status={row.get('status')} mae={row.get('mae')}"
            )
        finally:
            queue.release(gpu)

    threads = []
    for job in jobs:
        t = threading.Thread(target=worker, args=(job,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    out_csv = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_md = (
        ROOT / args.markdown if not Path(args.markdown).is_absolute() else Path(args.markdown)
    )
    write_budget_outputs(rows, out_csv, out_md)
    print(f"Wrote {out_csv} and {out_md} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
