#!/usr/bin/env python3
"""Prepare per-fold data dir + training CFG for matched-maturity teachers."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basicts.utils import dump_pkl, load_pkl
from scripts.matched_maturity_lib import (
    DATA_FOLD_ROOT,
    GEN_CFG_ROOT,
    ROUTES,
    STABLE_CFG,
    compare_protocol,
    dump_json,
    extract_protocol_fields,
    fold_dir,
    load_manifest,
    load_stable_cfg,
    sha1_file,
    sha1_indices,
    verify_manifest_integrity,
)


def prepare_fold_data_dir(fold: int, train_indices: list[int], *, smoke: bool = False) -> Path:
    """Create fold-specific data dir: subset TRAIN index + official valid/test lists.

    TEST list is retained in the pkl for BasicTS schema compatibility, but the
    MatchedMaturityTeacherRunner never instantiates the TEST loader.
    """
    official = ROOT / "datasets/PEMS04"
    out = DATA_FOLD_ROOT / ("smoke" if smoke else "formal") / f"fold{int(fold)}"
    out.mkdir(parents=True, exist_ok=True)
    raw = load_pkl(str(official / "index_in12_out12.pkl"))
    out_index = copy.deepcopy(raw)
    train = list(raw["train"])
    out_index["train"] = [train[i] for i in train_indices]
    dump_pkl(out_index, str(out / "index_in12_out12.pkl"))

    for name in ("data_in12_out12.pkl", "scaler_in12_out12.pkl", "adj_mx.pkl"):
        src = official / name
        dst = out / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(os.path.relpath(src, out), dst)
    return out


def write_fold_cfg(
    *,
    fold: int,
    data_dir: Path,
    ckpt_save_dir: Path,
    meta_dir: Path,
    expected_train_len: int,
    seed: int = 1,
    num_epochs: int | None = None,
    smoke: bool = False,
    batch_size: int | None = None,
) -> tuple[Path, dict]:
    cfg, cfg_path = load_stable_cfg()
    actual = extract_protocol_fields(cfg)
    cmp = compare_protocol(actual)
    print(json.dumps({"protocol_comparison": cmp["rows"]}, indent=2))
    if cmp["MATCHED_PROTOCOL_MISMATCH"]:
        raise RuntimeError(
            "MATCHED_PROTOCOL_MISMATCH fields=" + ",".join(cmp["material_mismatches"])
        )

    stable_text = STABLE_CFG.read_text(encoding="utf-8")
    epochs = int(num_epochs if num_epochs is not None else actual["epochs"])
    bs = int(batch_size if batch_size is not None else actual["batch_size"])
    data_rel = os.path.relpath(data_dir, ROOT).replace("\\", "/")
    ckpt_rel = os.path.relpath(ckpt_save_dir, ROOT).replace("\\", "/")
    hist_rel = os.path.relpath(meta_dir / "history.json", ROOT).replace("\\", "/")
    meta_rel = os.path.relpath(meta_dir, ROOT).replace("\\", "/")
    tag = f"matched_maturity_crossfit_v2_fold{fold}" + ("_smoke" if smoke else "")

    append = f"""
# ===== matched-maturity fold overrides (auto) =====
from scripts.matched_maturity_teacher_runner import MatchedMaturityTeacherRunner
CFG.RUNNER = MatchedMaturityTeacherRunner
CFG.ENV.SEED = {int(seed)}
CFG.TRAIN.NUM_EPOCHS = {epochs}
CFG.TRAIN.DATA.BATCH_SIZE = {bs}
CFG.VAL.DATA.BATCH_SIZE = {bs}
CFG.TRAIN.DATA.DIR = "{data_rel}"
CFG.VAL.DATA.DIR = "{data_rel}"
CFG.TEST.DATA.DIR = "{data_rel}"
CFG.TEST.INTERVAL = 1000000000
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("{ckpt_rel}")
CFG.MODEL.PARAM["experiment_tag"] = "{tag}"
CFG.MODEL.PARAM["run_signature"] = CFG.MODEL.PARAM.get("run_signature", "") + "|tag={tag}"
CFG.DESCRIPTION = "matched_maturity {tag} H=12 seed={int(seed)}"
CFG.MATCHED_MATURITY_HISTORY_PATH = "{hist_rel}"
CFG.MATCHED_MATURITY_META_DIR = "{meta_rel}"
CFG.MATCHED_MATURITY_EXPECTED_TRAIN_LEN = {int(expected_train_len)}
CFG.MATCHED_MATURITY_FOLD = {int(fold)}
"""
    GEN_CFG_ROOT.mkdir(parents=True, exist_ok=True)
    out_cfg = GEN_CFG_ROOT / f"H12_{tag}_seed{int(seed)}.py"
    out_cfg.write_text(stable_text + "\n" + append, encoding="utf-8")
    meta = {
        "fold": int(fold),
        "cfg_path": str(out_cfg.relative_to(ROOT)),
        "stable_cfg_path": str(STABLE_CFG.relative_to(ROOT)),
        "stable_cfg_sha1": sha1_file(STABLE_CFG, None),
        "stable_cfg_sha1_16": sha1_file(STABLE_CFG, 16),
        "data_dir": data_rel,
        "ckpt_save_dir": ckpt_rel,
        "expected_train_len": int(expected_train_len),
        "num_epochs": epochs,
        "batch_size": bs,
        "seed": int(seed),
        "smoke": bool(smoke),
        "protocol": actual,
        "protocol_check": cmp,
        "candidate_routes": ROUTES,
    }
    dump_json(meta_dir / "training_config.json", meta)
    return out_cfg, meta


def prepare_fold(
    fold: int,
    *,
    manifest_path: Path | None = None,
    seed: int = 1,
    smoke: bool = False,
    smoke_train_n: int = 16,
    num_epochs: int | None = None,
) -> dict:
    man = load_manifest(manifest_path)
    integrity = verify_manifest_integrity(man)
    if not integrity["pass"] and not smoke:
        raise RuntimeError(f"manifest integrity failed: {integrity}")
    fold_rec = next(f for f in man["folds"] if int(f["fold"]) == int(fold))
    train_indices = list(fold_rec["teacher_train_indices"])
    if smoke:
        train_indices = train_indices[: int(smoke_train_n)]
        num_epochs = 1 if num_epochs is None else num_epochs
    idx_sha = sha1_indices(train_indices, 40)
    meta_dir = fold_dir(fold, smoke=smoke)
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "train_indices_sha1.txt").write_text(idx_sha + "\n")
    dump_json(
        meta_dir / "train_indices.json",
        {
            "fold": int(fold),
            "n": len(train_indices),
            "sha1": idx_sha,
            "indices": train_indices if smoke else None,
            "indices_note": "full indices omitted in formal meta; recover from manifest+sha1",
            "manifest_n_train_after_purge": int(fold_rec["n_train_after_purge"]),
        },
    )
    data_dir = prepare_fold_data_dir(fold, train_indices, smoke=smoke)
    # Assert index length
    written = load_pkl(str(data_dir / "index_in12_out12.pkl"))
    assert len(written["train"]) == len(train_indices)
    ckpt_save_dir = meta_dir / "seed1"
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    cfg_path, cfg_meta = write_fold_cfg(
        fold=fold,
        data_dir=data_dir,
        ckpt_save_dir=ckpt_save_dir,
        meta_dir=meta_dir,
        expected_train_len=len(train_indices),
        seed=seed,
        num_epochs=num_epochs,
        smoke=smoke,
    )
    # Overlap assert (raw timestamps)
    from basicts.archs.arch_zoo.ChainForecasting_arch.temporal_crossfit_refinement import (
        load_split_index,
        sample_raw_span,
        spans_overlap,
    )

    index = load_split_index("datasets/PEMS04/index_in12_out12.pkl", "train")
    hold = list(fold_rec["heldout_sample_indices"])
    if smoke:
        hold = hold[: max(8, min(32, len(hold)))]
    hold_spans = [sample_raw_span(index[i]) for i in hold]
    overlap = 0
    for i in train_indices:
        sp = sample_raw_span(index[i])
        if any(spans_overlap(sp, hs) for hs in hold_spans):
            overlap += 1
    if overlap != 0:
        raise RuntimeError(f"raw-window overlap non-zero: {overlap}")

    out = {
        "fold": int(fold),
        "meta_dir": str(meta_dir.relative_to(ROOT)),
        "cfg_path": str(cfg_path.relative_to(ROOT)),
        "data_dir": str(data_dir.relative_to(ROOT)),
        "train_indices_sha1": idx_sha,
        "n_train": len(train_indices),
        "n_holdout_full": int(fold_rec["n_holdout"]),
        "overlap": overlap,
        "integrity": integrity,
        "cfg_meta": cfg_meta,
    }
    dump_json(meta_dir / "prepare.json", out)
    print(json.dumps({k: out[k] for k in out if k != "cfg_meta"}, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--manifest", default="results/matched_maturity_crossfit_manifest.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-train-n", type=int, default=16)
    ap.add_argument("--num-epochs", type=int, default=None)
    args = ap.parse_args()
    prepare_fold(
        args.fold,
        manifest_path=Path(args.manifest),
        seed=args.seed,
        smoke=args.smoke,
        smoke_train_n=args.smoke_train_n,
        num_epochs=args.num_epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
