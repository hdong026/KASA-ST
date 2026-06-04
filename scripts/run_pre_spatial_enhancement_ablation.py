#!/usr/bin/env python3
"""Run pre-temporal spatial embedding enhancement ablation (MLP prior, on vs off)."""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import re
import statistics
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

T_CRIT_N5 = 2.776
DEFAULT_VARIANTS = ("mlp_pre_spatial_on", "mlp_pre_spatial_off")
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
TEMP_CFG_DIR = os.path.join(ROOT, "tmp_configs", "pre_spatial_enhancement_ablation")
RUN_LOG_DIR = os.path.join(ROOT, "logs", "pre_spatial_enhancement_ablation")

RESULT_RE = re.compile(
    r"Result\s*<\s*test\s*>:\s*\[(?P<body>.*?)\]",
    re.IGNORECASE,
)
MAE_RE = re.compile(r"test_MAE:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
RMSE_RE = re.compile(r"test_RMSE:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
MAPE_RE = re.compile(r"test_MAPE:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


@dataclass
class RunResult:
    variant: str
    seed: int
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    ckpt_dir: str = ""
    log_file: str = ""
    status: str = "pending"
    cmd: str = ""
    cfg_path: str = ""


@dataclass
class WorkerState:
    variant: str
    gpu: str
    results: List[RunResult] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MLP prior with/without pre-temporal spatial enhancement."
    )
    parser.add_argument(
        "--cfg",
        default=os.path.join(ROOT, "examples", "KASAST_v2", "KASAST_PEMS04.py"),
        help="Base training config path.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds to evaluate.",
    )
    parser.add_argument("--gpu_with", default="0", help="GPU id for pre_spatial_on runs.")
    parser.add_argument("--gpu_without", default="1", help="GPU id for pre_spatial_off runs.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        choices=DEFAULT_VARIANTS,
        help="Ablation variants to compare.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "results", "pems04_pre_spatial_enhancement_ablation.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--markdown",
        default=os.path.join(ROOT, "results", "pems04_pre_spatial_enhancement_ablation.md"),
        help="Markdown summary output path.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print generated commands/configs without running training.",
    )
    return parser.parse_args()


def cfg_basename(cfg_path: str) -> str:
    return os.path.splitext(os.path.basename(cfg_path))[0]


def temp_cfg_path(base_cfg: str, variant: str, seed: int) -> str:
    short = "on" if variant == "mlp_pre_spatial_on" else "off"
    return os.path.join(
        TEMP_CFG_DIR,
        f"{cfg_basename(base_cfg)}_{short}_seed{seed}.py",
    )


def generate_temp_config(base_cfg: str, variant: str, seed: int) -> str:
    with open(base_cfg, "r", encoding="utf-8") as f:
        content = f.read()

    use_enhancement = variant == "mlp_pre_spatial_on"
    if use_enhancement:
        ckpt_suffix = f"_mlp_pre_spatial_on_seed{seed}"
    else:
        ckpt_suffix = f"_mlp_pre_spatial_off_seed{seed}"

    override = f"""

# ===== pre-temporal spatial enhancement ablation overrides (auto-generated) =====
CFG.MODEL.PARAM["prior_mapper_type"] = "mlp"
CFG.MODEL.PARAM["use_pre_temporal_spatial_enhancement"] = {use_enhancement}
if hasattr(CFG, "SEED"):
    CFG.SEED = {seed}
if hasattr(CFG, "ENV"):
    CFG.ENV.SEED = {seed}
if hasattr(CFG, "TRAIN") and hasattr(CFG.TRAIN, "SEED"):
    CFG.TRAIN.SEED = {seed}
CFG.TRAIN.CKPT_SAVE_DIR = CFG.TRAIN.CKPT_SAVE_DIR + "{ckpt_suffix}"
"""
    out_path = temp_cfg_path(base_cfg, variant, seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.write(override)
    return out_path


def load_ckpt_dir_from_cfg(cfg_path: str) -> str:
    spec = importlib.util.spec_from_file_location("pre_spatial_ablation_cfg", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import config: {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ckpt_dir = module.CFG.TRAIN.CKPT_SAVE_DIR
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(ROOT, ckpt_dir)
    return ckpt_dir


def find_training_log(ckpt_save_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_save_dir):
        return None
    patterns = [
        os.path.join(ckpt_save_dir, "training_log*.log"),
        os.path.join(ckpt_save_dir, "*", "training_log*.log"),
        os.path.join(ckpt_save_dir, "**", "training_log*.log"),
    ]
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def parse_test_metrics(log_path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not log_path or not os.path.exists(log_path):
        return None, None, None

    best: Optional[Tuple[float, float, float]] = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = RESULT_RE.search(line)
            if not match:
                continue
            body = match.group("body")
            mae_match = MAE_RE.search(body)
            rmse_match = RMSE_RE.search(body)
            mape_match = MAPE_RE.search(body)
            if not (mae_match and rmse_match and mape_match):
                continue
            mae = float(mae_match.group(1))
            rmse = float(rmse_match.group(1))
            mape = float(mape_match.group(1))
            if best is None or mae < best[0]:
                best = (mae, rmse, mape)
    if best is None:
        return None, None, None
    return best


def run_single(
    base_cfg: str,
    variant: str,
    seed: int,
    gpu: str,
    dry_run: bool,
) -> RunResult:
    cfg_path = generate_temp_config(base_cfg, variant, seed)
    ckpt_dir = load_ckpt_dir_from_cfg(cfg_path)
    rel_cfg = os.path.relpath(cfg_path, ROOT)
    cmd = [sys.executable, "examples/run.py", "--cfg", rel_cfg, "--gpus", str(gpu)]
    cmd_str = " ".join(cmd)
    result = RunResult(
        variant=variant,
        seed=seed,
        ckpt_dir=ckpt_dir,
        cmd=cmd_str,
        cfg_path=cfg_path,
    )

    print(f"[{variant} seed={seed} gpu={gpu}] {cmd_str}")
    if dry_run:
        result.status = "dry_run"
        return result

    os.makedirs(RUN_LOG_DIR, exist_ok=True)
    short = "on" if variant == "mlp_pre_spatial_on" else "off"
    wrapper_log = os.path.join(
        RUN_LOG_DIR,
        f"{cfg_basename(base_cfg)}_{short}_seed{seed}.log",
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    with open(wrapper_log, "w", encoding="utf-8") as log_f:
        log_f.write(f"command: {cmd_str}\n")
        log_f.write(f"CUDA_VISIBLE_DEVICES={gpu}\n\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        return_code = proc.wait()

    training_log = find_training_log(ckpt_dir)
    result.log_file = training_log or wrapper_log
    mae, rmse, mape = parse_test_metrics(result.log_file)
    result.mae = mae
    result.rmse = rmse
    result.mape = mape

    if return_code != 0:
        result.status = "failed"
    elif mae is None:
        result.status = "failed"
    else:
        result.status = "ok"
    return result


def worker_loop(
    state: WorkerState,
    base_cfg: str,
    seeds: Sequence[int],
    dry_run: bool,
) -> None:
    for seed in seeds:
        result = run_single(base_cfg, state.variant, seed, state.gpu, dry_run)
        with state.lock:
            state.results.append(result)


def ci95(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    std = statistics.stdev(values)
    return T_CRIT_N5 * std / (len(values) ** 0.5)


def summarize_group(results: Sequence[RunResult], metric: str) -> Dict[str, float]:
    values = [
        getattr(r, metric)
        for r in results
        if r.status == "ok" and getattr(r, metric) is not None
    ]
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "std": std, "ci95": ci95(values)}


def fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "NA"
    return f"{x:.{digits}f}"


def write_csv(path: str, results: Sequence[RunResult]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "seed",
                "mae",
                "rmse",
                "mape",
                "ckpt_dir",
                "log_file",
                "status",
                "cmd",
                "cfg_path",
            ],
        )
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x.variant, x.seed)):
            writer.writerow(
                {
                    "variant": r.variant,
                    "seed": r.seed,
                    "mae": r.mae,
                    "rmse": r.rmse,
                    "mape": r.mape,
                    "ckpt_dir": r.ckpt_dir,
                    "log_file": r.log_file,
                    "status": r.status,
                    "cmd": r.cmd,
                    "cfg_path": r.cfg_path,
                }
            )


def write_markdown(path: str, results: Sequence[RunResult], seeds: Sequence[int]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    by_variant: Dict[str, List[RunResult]] = {}
    for r in results:
        by_variant.setdefault(r.variant, []).append(r)

    lines = ["# Pre-Temporal Spatial Embedding Enhancement Ablation", ""]

    lines.append("## Per-run results")
    lines.append("")
    lines.append("| variant | seed | MAE | RMSE | MAPE | status |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for r in sorted(results, key=lambda x: (x.variant, x.seed)):
        lines.append(
            f"| {r.variant} | {r.seed} | {fmt(r.mae)} | {fmt(r.rmse)} | "
            f"{fmt(r.mape)} | {r.status} |"
        )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| variant | n | MAE mean | MAE std | MAE 95% CI | "
        "RMSE mean | RMSE std | RMSE 95% CI | MAPE mean | MAPE std | MAPE 95% CI |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for variant in sorted(by_variant):
        ok = [r for r in by_variant[variant] if r.status == "ok"]
        mae_s = summarize_group(ok, "mae")
        rmse_s = summarize_group(ok, "rmse")
        mape_s = summarize_group(ok, "mape")
        lines.append(
            f"| {variant} | {mae_s['n']} | {fmt(mae_s['mean'])} | {fmt(mae_s['std'])} | "
            f"{fmt(mae_s['ci95'])} | {fmt(rmse_s['mean'])} | {fmt(rmse_s['std'])} | "
            f"{fmt(rmse_s['ci95'])} | {fmt(mape_s['mean'])} | {fmt(mape_s['std'])} | "
            f"{fmt(mape_s['ci95'])} |"
        )
    lines.append("")

    on_by_seed = {
        r.seed: r for r in results if r.variant == "mlp_pre_spatial_on" and r.status == "ok"
    }
    off_by_seed = {
        r.seed: r for r in results if r.variant == "mlp_pre_spatial_off" and r.status == "ok"
    }
    shared_seeds = sorted(set(on_by_seed) & set(off_by_seed) & set(seeds))

    lines.append("## Paired difference: Off - On")
    lines.append("")
    lines.append("Positive diff means removing pre-temporal spatial enhancement is worse.")
    lines.append("")
    if not shared_seeds:
        lines.append("_No paired successful runs available._")
    else:
        lines.append("| metric | mean diff | std diff | 95% CI |")
        lines.append("| --- | ---: | ---: | ---: |")
        for metric in ("mae", "rmse", "mape"):
            diffs = [
                getattr(off_by_seed[s], metric) - getattr(on_by_seed[s], metric)
                for s in shared_seeds
            ]
            mean = statistics.mean(diffs)
            std = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
            lines.append(
                f"| {metric.upper()} | {fmt(mean)} | {fmt(std)} | {fmt(ci95(diffs))} |"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    base_cfg = args.cfg
    if not os.path.isabs(base_cfg):
        base_cfg = os.path.join(ROOT, base_cfg)
    if not os.path.exists(base_cfg):
        raise FileNotFoundError(f"Config not found: {base_cfg}")

    gpu_map = {
        "mlp_pre_spatial_on": args.gpu_with,
        "mlp_pre_spatial_off": args.gpu_without,
    }
    workers: List[Tuple[threading.Thread, WorkerState]] = []

    for variant in args.variants:
        gpu = gpu_map.get(variant, args.gpu_with)
        state = WorkerState(variant=variant, gpu=str(gpu))
        thread = threading.Thread(
            target=worker_loop,
            args=(state, base_cfg, args.seeds, args.dry_run),
            name=f"worker-{variant}",
            daemon=True,
        )
        workers.append((thread, state))

    for thread, _ in workers:
        thread.start()
    for thread, _ in workers:
        thread.join()

    all_results: List[RunResult] = []
    for _, state in workers:
        all_results.extend(state.results)

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    md_path = args.markdown if os.path.isabs(args.markdown) else os.path.join(ROOT, args.markdown)
    write_csv(out_path, all_results)
    write_markdown(md_path, all_results, args.seeds)
    print(f"Wrote CSV: {out_path}")
    print(f"Wrote markdown: {md_path}")


if __name__ == "__main__":
    main()
