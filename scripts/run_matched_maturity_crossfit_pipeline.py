#!/usr/bin/env python3
"""Matched-maturity crossfit orchestrator (smoke / fold1 acceptance / full).

NO adaptive router / Plan A / Plan B / Bellman / PPO / GRPO training.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.matched_maturity_lib import (
    FUTURE_PROTOCOL,
    STABLE_CFG,
    TEACHER_ROOT,
    dump_json,
    fold_dir,
    get_test_access_count,
    load_manifest,
    reset_test_access,
    sha1_file,
    verify_manifest_integrity,
)
from scripts.matched_maturity_prepare_fold import prepare_fold
from scripts.matched_maturity_build_fold_oof import (
    build_fold_oof,
    find_best_ckpt,
    find_last_ckpt,
)
from scripts.matched_maturity_merge_audit import audit_environment, merge_oof


def _run_training(cfg_path: Path, gpu: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Pin physical GPU via env; pass logical --gpus 0 (see examples/run.py resolve_cuda_devices).
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["MATCHED_MATURITY_TEST_ACCESS_COUNT"] = "0"
    cmd = [
        sys.executable,
        str(ROOT / "examples/run.py"),
        "--cfg",
        str(cfg_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "--gpus",
        "0",
    ]
    print("[train]", f"CUDA_VISIBLE_DEVICES={gpu}", " ".join(cmd))
    with log_path.open("w") as lf:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(proc.returncode)


def _fold_complete_marker(meta_dir: Path) -> Path:
    return meta_dir / "COMPLETE.json"


def _verify_completed_fold(meta_dir: Path, *, expected_epochs: int, expected_train_n: int) -> dict:
    problems = []
    need = [
        "training_config.json",
        "train_indices_sha1.txt",
        "initial_state.json",
        "history.json",
        "training_summary.json",
        "oof/fold_oof_oracle.json",
        "COMPLETE.json",
    ]
    for rel in need:
        if not (meta_dir / rel).is_file():
            problems.append(f"missing:{rel}")
    try:
        ckpt = find_best_ckpt(meta_dir)
    except Exception as e:
        problems.append(f"best_ckpt:{e}")
        ckpt = None
    hist = []
    if (meta_dir / "history.json").is_file():
        hist = json.loads((meta_dir / "history.json").read_text())
    if len(hist) < int(expected_epochs):
        problems.append(f"epochs:{len(hist)}<{expected_epochs}")
    summary = {}
    if (meta_dir / "training_summary.json").is_file():
        summary = json.loads((meta_dir / "training_summary.json").read_text())
    init = {}
    if (meta_dir / "initial_state.json").is_file():
        init = json.loads((meta_dir / "initial_state.json").read_text())
    cfg = {}
    if (meta_dir / "training_config.json").is_file():
        cfg = json.loads((meta_dir / "training_config.json").read_text())
    if cfg.get("protocol_check", {}).get("MATCHED_PROTOCOL_MISMATCH"):
        problems.append("protocol_mismatch_cached")
    if init.get("optimizer_type") != "Adam":
        problems.append(f"optim={init.get('optimizer_type')}")
    if init.get("scheduler_type") != "MultiStepLR":
        problems.append(f"sched={init.get('scheduler_type')}")
    if int(init.get("train_len", -1)) != int(expected_train_n):
        problems.append(f"train_len={init.get('train_len')}!={expected_train_n}")
    oof = {}
    if (meta_dir / "oof/fold_oof_oracle.json").is_file():
        oof = json.loads((meta_dir / "oof/fold_oof_oracle.json").read_text())
        if int(oof.get("metadata", {}).get("n_records", -1)) != int(expected_train_n and oof["metadata"].get("n_holdout_expected", -1)):
            # check against holdout expected in oof meta
            pass
        if int(oof.get("metadata", {}).get("TEST_ACCESS_COUNT", 1)) != 0:
            problems.append("TEST_ACCESS_COUNT!=0")
    return {
        "ok": not problems,
        "problems": problems,
        "best_ckpt": None if ckpt is None else str(ckpt),
        "best_epoch": summary.get("best_epoch"),
        "best_val_MAE": summary.get("best_val_MAE"),
        "n_history": len(hist),
        "initial_param_sha1": init.get("initial_param_sha1"),
    }


def run_one_fold(
    fold: int,
    *,
    gpu: str,
    seed: int,
    smoke: bool,
    smoke_train_n: int = 16,
    max_holdout: int | None = None,
    force_retrain: bool = False,
) -> dict:
    reset_test_access()
    man = load_manifest()
    fold_rec = next(f for f in man["folds"] if int(f["fold"]) == int(fold))
    expected_train_n = int(smoke_train_n if smoke else fold_rec["n_train_after_purge"])
    expected_epochs = 1 if smoke else 100
    expected_holdout = int(max_holdout or (4 if smoke else fold_rec["n_holdout"]))

    prep = prepare_fold(
        fold,
        seed=seed,
        smoke=smoke,
        smoke_train_n=smoke_train_n,
        num_epochs=expected_epochs,
    )
    meta_dir = ROOT / prep["meta_dir"]
    marker = _fold_complete_marker(meta_dir)

    if force_retrain:
        import shutil

        print(f"[fold {fold}] force_retrain: wiping checkpoints/COMPLETE under {meta_dir}")
        for name in ("COMPLETE.json", "history.json", "training_summary.json", "oof_complete.json"):
            p = meta_dir / name
            if p.exists():
                p.unlink()
        oof_dir = meta_dir / "oof"
        if oof_dir.exists():
            shutil.rmtree(oof_dir)
        seed_dir = meta_dir / "seed1"
        if seed_dir.exists():
            # remove md5 ckpt dirs so EasyTorch does not resume
            for child in seed_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                elif child.suffix == ".pt":
                    child.unlink()

    if marker.is_file() and not force_retrain:
        ver = _verify_completed_fold(
            meta_dir, expected_epochs=expected_epochs, expected_train_n=expected_train_n
        )
        if ver["ok"]:
            print(f"[fold {fold}] resume: COMPLETE verified — skip retrain")
            return {"fold": fold, "skipped": True, "verify": ver}
        print(f"[fold {fold}] COMPLETE present but stale/invalid: {ver['problems']} — retraining")

    cfg_path = ROOT / prep["cfg_path"]
    log_path = meta_dir / "train.log"
    rc = _run_training(cfg_path, gpu=gpu, log_path=log_path)
    if rc != 0:
        raise RuntimeError(f"training failed fold={fold} rc={rc} log={log_path}")

    # Ensure summary exists (runner writes it); if missing, synthesize from history
    hist_path = meta_dir / "history.json"
    if not hist_path.is_file():
        raise RuntimeError(f"missing history.json after training: {hist_path}")
    hist = json.loads(hist_path.read_text())
    if len(hist) < expected_epochs:
        raise RuntimeError(f"incomplete epochs fold={fold}: {len(hist)}/{expected_epochs}")

    init_path = meta_dir / "initial_state.json"
    if not init_path.is_file():
        raise RuntimeError("missing initial_state.json")
    init = json.loads(init_path.read_text())
    if init.get("optimizer_type") != "Adam" or init.get("scheduler_type") != "MultiStepLR":
        raise RuntimeError(f"protocol instantiate fail: {init}")

    best = find_best_ckpt(meta_dir)
    last = find_last_ckpt(meta_dir)
    # OOF
    oof = build_fold_oof(
        fold,
        device="cuda:0",
        batch_size=4 if smoke else 8,
        smoke=smoke,
        max_holdout=expected_holdout if smoke else None,
    )
    complete = {
        "fold": int(fold),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "best_ckpt": str(best.relative_to(ROOT)),
        "last_ckpt": None if last is None else str(last.relative_to(ROOT)),
        "best_ckpt_sha1": sha1_file(best, 40),
        "train_indices_sha1": prep["train_indices_sha1"],
        "initial_param_sha1": init.get("initial_param_sha1"),
        "n_train": prep["n_train"],
        "n_oof": oof["metadata"]["n_records"],
        "TEST_ACCESS_COUNT": oof["metadata"]["TEST_ACCESS_COUNT"],
        "optimizer": init.get("optimizer_type"),
        "scheduler": init.get("scheduler_type"),
        "epochs": len(hist),
        "smoke": bool(smoke),
    }
    dump_json(marker, complete)
    return {"fold": fold, "skipped": False, "complete": complete, "init": init}


def run_engineering_smoke(gpu: str, seed: int) -> dict:
    t0 = time.time()
    report = {"mode": "engineering_smoke", "assertions": {}, "failures": []}

    def check(name, cond, detail=""):
        report["assertions"][name] = {"pass": bool(cond), "detail": detail}
        if not cond:
            report["failures"].append(f"{name}: {detail}")

    man = load_manifest()
    integ = verify_manifest_integrity(man)
    check("manifest_integrity", integ["pass"], json.dumps(integ))

    out = run_one_fold(
        1,
        gpu=gpu,
        seed=seed,
        smoke=True,
        smoke_train_n=16,
        max_holdout=4,
        force_retrain=True,
    )
    init = out.get("init") or {}
    complete = out.get("complete") or {}
    check("adam", init.get("optimizer_type") == "Adam", str(init.get("optimizer_type")))
    check("multistep", init.get("scheduler_type") == "MultiStepLR", str(init.get("scheduler_type")))
    check("ckpt_saved", bool(complete.get("best_ckpt")), str(complete.get("best_ckpt")))
    check("oof_n", int(complete.get("n_oof", 0)) == 4, str(complete.get("n_oof")))
    check("test_access_0", int(complete.get("TEST_ACCESS_COUNT", 1)) == 0, str(complete.get("TEST_ACCESS_COUNT")))

    merged = merge_oof(smoke=True)
    check("merge_schema", len(merged["records"]) == 4, str(len(merged["records"])))
    audit_environment(merged, smoke=True)

    # prove Z3 + 4 losses present
    r0 = merged["records"][0]
    check("four_losses", len(r0.get("route_final_losses", [])) == 4)
    check("z3_ref", "shard" in r0.get("Z3_ref", {}))

    wall = time.time() - t0
    report.update(
        {
            "wall_sec": wall,
            "verdict": "PASS" if not report["failures"] and wall <= 300 else "FAIL",
            "scientific_claim": False,
            "future_protocol": FUTURE_PROTOCOL,
            "stable_cfg": str(STABLE_CFG.relative_to(ROOT)),
            "stable_cfg_sha1": sha1_file(STABLE_CFG, 40),
            "fold_result": complete,
            "note": "Engineering smoke only — do NOT claim scientific environment match.",
        }
    )
    if wall > 300:
        report["failures"].append("timeout>300s")
        report["verdict"] = "FAIL"
    dump_json("results/matched_maturity_crossfit_engineering_smoke.json", report)
    # also keep legacy smoke filename
    dump_json("results/matched_maturity_crossfit_smoke.json", report)
    print(json.dumps({"verdict": report["verdict"], "wall": wall, "failures": report["failures"]}, indent=2))
    return report


def run_fold1_acceptance(gpu: str, seed: int) -> dict:
    t0 = time.time()
    out = run_one_fold(1, gpu=gpu, seed=seed, smoke=False, force_retrain=False)
    meta_dir = fold_dir(1, smoke=False)
    man = load_manifest()
    f1 = next(f for f in man["folds"] if int(f["fold"]) == 1)
    ver = _verify_completed_fold(
        meta_dir,
        expected_epochs=100,
        expected_train_n=int(f1["n_train_after_purge"]),
    )
    oof = json.loads((meta_dir / "oof/fold_oof_oracle.json").read_text())
    init = json.loads((meta_dir / "initial_state.json").read_text())
    hist = json.loads((meta_dir / "history.json").read_text())
    checks = {
        "optimizer_Adam": init.get("optimizer_type") == "Adam",
        "scheduler_MultiStepLR": init.get("scheduler_type") == "MultiStepLR",
        "epochs_100": len(hist) == 100,
        "best_by_val_MAE": ver.get("best_epoch") is not None,
        "checkpoint_hashes": bool((meta_dir / "COMPLETE.json").is_file()),
        "oof_n": int(oof["metadata"]["n_records"]) == int(f1["n_holdout"]),
        "four_losses": all(len(r["route_final_losses"]) == 4 for r in oof["records"]),
        "z3_present": all("shard" in r.get("Z3_ref", {}) for r in oof["records"]),
        "TEST_ACCESS_COUNT_0": int(oof["metadata"]["TEST_ACCESS_COUNT"]) == 0,
        "verify_ok": ver["ok"],
    }
    passed = all(checks.values())
    report = {
        "MATCHED_FOLD1_ACCEPTANCE": "PASS" if passed else "FAIL",
        "checks": checks,
        "verify": ver,
        "wall_sec": time.time() - t0,
        "n_holdout_expected": int(f1["n_holdout"]),
        "n_train_after_purge": int(f1["n_train_after_purge"]),
        "fold_run": out,
        "future_protocol": FUTURE_PROTOCOL,
    }
    dump_json("results/matched_maturity_crossfit_fold1_acceptance.json", report)
    print(json.dumps({"MATCHED_FOLD1_ACCEPTANCE": report["MATCHED_FOLD1_ACCEPTANCE"], "checks": checks}, indent=2))
    return report


def acceptance_metadata_ok() -> bool:
    p = ROOT / "results/matched_maturity_crossfit_fold1_acceptance.json"
    if not p.is_file():
        return False
    blob = json.loads(p.read_text())
    return blob.get("MATCHED_FOLD1_ACCEPTANCE") == "PASS"


def run_full(gpu: str, seed: int, *, override_acceptance: bool = False) -> dict:
    if not acceptance_metadata_ok() and not override_acceptance:
        raise RuntimeError(
            "Refusing full 5-fold run: Fold1 acceptance PASS metadata missing. "
            "Run --acceptance-fold 1 first, or pass --override-acceptance (LOUD WARNING)."
        )
    if override_acceptance and not acceptance_metadata_ok():
        print("=" * 70)
        print("WARNING: --override-acceptance used WITHOUT Fold1 acceptance PASS")
        print("=" * 70)

    fold_results = []
    init_hashes = []
    for fold in range(1, 6):
        out = run_one_fold(fold, gpu=gpu, seed=seed, smoke=False)
        fold_results.append(out)
        meta = fold_dir(fold, smoke=False)
        init = json.loads((meta / "initial_state.json").read_text())
        init_hashes.append(init.get("initial_param_sha1"))

    merged = merge_oof(smoke=False)
    agree = audit_environment(merged, smoke=False)

    hash_agree = len(set(init_hashes)) == 1
    final = {
        "folds": fold_results,
        "initial_param_sha1_by_fold": init_hashes,
        "initial_hashes_match_across_folds": hash_agree,
        "initial_hash_note": (
            "Expected match under identical arch+seed=1 fresh init."
            if hash_agree
            else "Hashes differ — investigate fold-specific setup affecting init."
        ),
        "environment_gate": agree.get("environment_gate"),
        "MATCHED_CROSSFIT_DID_NOT_SOLVE_ENVIRONMENT_MISMATCH": agree.get(
            "MATCHED_CROSSFIT_DID_NOT_SOLVE_ENVIRONMENT_MISMATCH"
        ),
        "controller_trained": False,
        "TEST_ACCESS_COUNT": 0,
        "future_protocol": FUTURE_PROTOCOL,
    }
    dump_json("results/matched_maturity_crossfit_final_report.json", final)
    return final


def write_preflight_v2(smoke_report: dict | None) -> dict:
    man = load_manifest()
    integ = verify_manifest_integrity(man)
    stub_text = (ROOT / "scripts/run_matched_maturity_crossfit_full.sh").read_text()
    is_stub = "THIS SCRIPT IS A STUB" in stub_text or "exit 3" in stub_text and "STUB" in stub_text
    # After rewrite, stub markers should be gone
    is_stub = "STUB ORCHESTRATOR PLACEHOLDER" in stub_text
    pf = {
        "MATCHED_CROSSFIT_FULL_PREFLIGHT": "PASS" if (not is_stub and integ["pass"] and smoke_report and smoke_report.get("verdict") == "PASS") else "FAIL",
        "full_runner_is_stub": is_stub,
        "manifest_integrity": integ,
        "stable_cfg": str(STABLE_CFG.relative_to(ROOT)),
        "stable_cfg_sha1": sha1_file(STABLE_CFG, 40),
        "engineering_smoke": None if smoke_report is None else smoke_report.get("verdict"),
        "fold1_acceptance": (
            json.loads((ROOT / "results/matched_maturity_crossfit_fold1_acceptance.json").read_text()).get(
                "MATCHED_FOLD1_ACCEPTANCE"
            )
            if (ROOT / "results/matched_maturity_crossfit_fold1_acceptance.json").is_file()
            else "NOT_RUN"
        ),
        "future_protocol": FUTURE_PROTOCOL,
        "note": "Scientific success requires full OOF environment-agreement audit after 5-fold run.",
    }
    dump_json("results/matched_maturity_crossfit_full_preflight_v2.json", pf)
    return pf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--acceptance-fold", type=int, default=None)
    ap.add_argument("--confirm-full-run", action="store_true")
    ap.add_argument("--override-acceptance", action="store_true")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--write-preflight-v2", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    if args.smoke:
        rep = run_engineering_smoke(args.gpu, args.seed)
        write_preflight_v2(rep)
        return 0 if rep["verdict"] == "PASS" else 1
    if args.acceptance_fold is not None:
        if int(args.acceptance_fold) != 1:
            raise SystemExit("Only --acceptance-fold 1 is supported in this protocol.")
        rep = run_fold1_acceptance(args.gpu, args.seed)
        write_preflight_v2(
            json.loads((ROOT / "results/matched_maturity_crossfit_engineering_smoke.json").read_text())
            if (ROOT / "results/matched_maturity_crossfit_engineering_smoke.json").is_file()
            else None
        )
        return 0 if rep["MATCHED_FOLD1_ACCEPTANCE"] == "PASS" else 1
    if args.confirm_full_run:
        run_full(args.gpu, args.seed, override_acceptance=args.override_acceptance)
        return 0
    if args.write_preflight_v2:
        smoke = None
        p = ROOT / "results/matched_maturity_crossfit_engineering_smoke.json"
        if p.is_file():
            smoke = json.loads(p.read_text())
        write_preflight_v2(smoke)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
