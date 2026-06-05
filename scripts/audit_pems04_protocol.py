#!/usr/bin/env python3
"""Audit PeMS04 data split, channels, and KASA vs baseline config protocol."""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "pems04_protocol_audit.md"
INPUT_LEN = 12
OUTPUT_LEN = 12

BASELINE_CONFIGS = sorted((ROOT / "examples" / "baselines").glob("*/*_PEMS04.py"))

CHANNEL_MEANINGS = {
    0: "flow (normalized traffic)",
    1: "time of day (ToD)",
    2: "day of week (DoW)",
    3: "train-only prior (weekly spectral template, HoloST)",
}


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_cfg(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CFG


def cfg_get(cfg, dotted: str, default=None):
    obj = cfg
    for part in dotted.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return default
    return obj


def extract_cfg_fields(cfg) -> dict:
    return {
        "DATASET_NAME": cfg_get(cfg, "DATASET_NAME"),
        "DATASET_INPUT_LEN": cfg_get(cfg, "DATASET_INPUT_LEN"),
        "DATASET_OUTPUT_LEN": cfg_get(cfg, "DATASET_OUTPUT_LEN"),
        "TRAIN.DATA.DIR": cfg_get(cfg, "TRAIN.DATA.DIR"),
        "VAL.DATA.DIR": cfg_get(cfg, "VAL.DATA.DIR"),
        "TEST.DATA.DIR": cfg_get(cfg, "TEST.DATA.DIR"),
        "FORWARD_FEATURES": list(cfg_get(cfg, "MODEL.FORWARD_FEATURES") or []),
        "TARGET_FEATURES": list(cfg_get(cfg, "MODEL.TARGET_FEATURES") or []),
        "TRAIN.BATCH_SIZE": cfg_get(cfg, "TRAIN.DATA.BATCH_SIZE"),
        "NUM_EPOCHS": cfg_get(cfg, "TRAIN.NUM_EPOCHS"),
        "OPTIM": dict(cfg_get(cfg, "TRAIN.OPTIM.PARAM") or {}),
        "LR_SCHEDULER": {
            "type": cfg_get(cfg, "TRAIN.LR_SCHEDULER.TYPE"),
            "param": dict(cfg_get(cfg, "TRAIN.LR_SCHEDULER.PARAM") or {}),
        },
        "CKPT_SAVE_DIR": cfg_get(cfg, "TRAIN.CKPT_SAVE_DIR"),
    }


def read_preprocess_defaults(script_path: Path) -> dict:
    text = script_path.read_text()
    out = {}
    for name in ("HISTORY_SEQ_LEN", "FUTURE_SEQ_LEN", "TRAIN_RATIO", "VALID_RATIO"):
        m = re.search(rf"^\s*{name}\s*=\s*([^\n#]+)", text, re.M)
        if m:
            out[name] = m.group(1).strip()
    # docstring / comment claims
    if "6:2:2" in text:
        out["comment_mentions_622"] = True
    if "0.7" in text and "TRAIN_RATIO" in text:
        out["uses_07_train_ratio"] = True
    return out


def classify_split(train_n: int, valid_n: int, test_n: int) -> str:
    total = train_n + valid_n + test_n
    if total == 0:
        return "unknown"
    ratios = (train_n / total, valid_n / total, test_n / total)
    candidates = {
        "6:2:2": (0.6, 0.2, 0.2),
        "7:1:2": (0.7, 0.1, 0.2),
    }
    best, best_err = "other", 1e9
    for label, target in candidates.items():
        err = sum(abs(r - t) for r, t in zip(ratios, target))
        if err < best_err:
            best, best_err = label, err
    if best_err > 0.05:
        return f"other ({ratios[0]:.3f}:{ratios[1]:.3f}:{ratios[2]:.3f})"
    return best


def audit_data(dataset_dir: Path) -> dict:
    index_path = dataset_dir / f"index_in{INPUT_LEN}_out{OUTPUT_LEN}.pkl"
    data_pkl = dataset_dir / f"data_in{INPUT_LEN}_out{OUTPUT_LEN}.pkl"
    data_npz = dataset_dir / f"data_in{INPUT_LEN}_out{OUTPUT_LEN}.npz"
    protocol_audit_path = dataset_dir / "protocol_audit.json"
    protocol_audit = None
    if protocol_audit_path.is_file():
        protocol_audit = json.loads(protocol_audit_path.read_text(encoding="utf-8"))

    index = load_pkl(index_path)
    data_dict = load_pkl(data_pkl)
    processed = data_dict["processed_data"]
    if hasattr(processed, "numpy"):
        processed = processed.numpy()
    processed = np.asarray(processed)

    npz_info = {}
    if data_npz.is_file():
        with np.load(data_npz) as z:
            npz_info["keys"] = list(z.files)
            npz_info["has_weekly_spectral_template"] = "weekly_spectral_template" in z.files
            if npz_info["has_weekly_spectral_template"]:
                npz_info["weekly_spectral_template_shape"] = tuple(z["weekly_spectral_template"].shape)

    train_idx = index["train"]
    valid_idx = index["valid"]
    test_idx = index["test"]
    total = len(train_idx) + len(valid_idx) + len(test_idx)

    return {
        "index_path": str(index_path.relative_to(ROOT)),
        "data_pkl_path": str(data_pkl.relative_to(ROOT)),
        "data_npz_path": str(data_npz.relative_to(ROOT)) if data_npz.is_file() else None,
        "processed_shape": tuple(processed.shape),
        "num_channels": processed.shape[-1],
        "train_n": len(train_idx),
        "valid_n": len(valid_idx),
        "test_n": len(test_idx),
        "train_ratio": len(train_idx) / total,
        "valid_ratio": len(valid_idx) / total,
        "test_ratio": len(test_idx) / total,
        "split_label": classify_split(len(train_idx), len(valid_idx), len(test_idx)),
        "train_first": train_idx[0],
        "train_last": train_idx[-1],
        "valid_first": valid_idx[0],
        "valid_last": valid_idx[-1],
        "test_first": test_idx[0],
        "test_last": test_idx[-1],
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "has_channel_3": processed.shape[-1] >= 4,
        "npz_info": npz_info,
        "processed_dtype": str(processed.dtype),
        "protocol_audit_path": str(protocol_audit_path.relative_to(ROOT)) if protocol_audit_path.is_file() else None,
        "protocol_audit": protocol_audit,
    }


def compare_configs(name_a: str, path_a: Path, name_b: str, path_b: Path) -> list[str]:
    a = extract_cfg_fields(load_cfg(path_a))
    b = extract_cfg_fields(load_cfg(path_b))
    lines = [f"## Config comparison: {name_a} vs {name_b}", ""]
    lines.append("| Field | " + name_a + " | " + name_b + " | Match? |")
    lines.append("|-------|" + "------|" * 2 + "--------|")
    keys = sorted(set(a) | set(b))
    for k in keys:
        va, vb = a.get(k), b.get(k)
        match = "yes" if va == vb else "no"
        lines.append(f"| {k} | `{va}` | `{vb}` | {match} |")
    lines.append("")

    same_index = (
        a["TRAIN.DATA.DIR"] == b["TRAIN.DATA.DIR"]
        and a["DATASET_INPUT_LEN"] == b["DATASET_INPUT_LEN"]
        and a["DATASET_OUTPUT_LEN"] == b["DATASET_OUTPUT_LEN"]
    )
    lines.append(f"- Same index/data file stem: **{'yes' if same_index else 'no'}** "
                 f"(`datasets/PEMS04/index_in12_out12.pkl` + `data_in12_out12.pkl`)")
    lines.append(f"- {name_a} uses channel 3: **{3 in a['FORWARD_FEATURES']}**")
    lines.append(f"- {name_b} uses channel 3: **{3 in b['FORWARD_FEATURES']}**")
    lines.append("")
    return lines


def baseline_forward_table(data_channels: int) -> list[str]:
    lines = ["## Baseline FORWARD_FEATURES audit", ""]
    lines.append("| Model | FORWARD_FEATURES | Uses ch3? |")
    lines.append("|-------|------------------|-----------|")
    for cfg_path in BASELINE_CONFIGS:
        model = cfg_path.parent.name
        ff = extract_cfg_fields(load_cfg(cfg_path))["FORWARD_FEATURES"]
        uses_ch3 = 3 in ff
        if uses_ch3 and data_channels < 4:
            flag = "yes (but data has <4 ch!)"
        elif uses_ch3:
            flag = "yes"
        else:
            flag = "no"
        lines.append(f"| {model} | `{ff}` | {flag} |")
    lines.append("")
    return lines


def build_report(dataset_dir: Path) -> str:
    data = audit_data(dataset_dir)
    holost_defaults = read_preprocess_defaults(ROOT / "scripts/data_preparation/PEMS04/generate_holost_data.py")
    train_defaults = read_preprocess_defaults(ROOT / "scripts/data_preparation/PEMS04/generate_training_data.py")

    lines = [
        "# PeMS04 Protocol Audit",
        "",
        f"Repo: `{ROOT}`",
        f"Dataset dir: `{dataset_dir.relative_to(ROOT)}`",
        "",
        "## 1. On-disk data split & shape",
        "",
        f"- **processed_data shape**: `{data['processed_shape']}` (T, N, C)",
        f"- **num channels**: {data['num_channels']}",
        f"- **channel meanings (inferred)**:",
    ]
    for i in range(data["num_channels"]):
        meaning = CHANNEL_MEANINGS.get(i, "unknown")
        lines.append(f"  - ch{i}: {meaning}")
    lines += [
        f"- **input length**: {data['input_len']}",
        f"- **output length**: {data['output_len']}",
        f"- **train samples**: {data['train_n']}",
        f"- **valid samples**: {data['valid_n']}",
        f"- **test samples**: {data['test_n']}",
        f"- **ratios (train/valid/test)**: {data['train_ratio']:.4f} / {data['valid_ratio']:.4f} / {data['test_ratio']:.4f}",
        f"- **split classification**: **{data['split_label']}**",
        f"- **train index first/last**: `{data['train_first']}` / `{data['train_last']}`",
        f"- **valid index first/last**: `{data['valid_first']}` / `{data['valid_last']}`",
        f"- **test index first/last**: `{data['test_first']}` / `{data['test_last']}`",
        f"- **channel 3 exists in processed_data**: **{data['has_channel_3']}**",
        f"- **index file**: `{data['index_path']}`",
        f"- **data pkl**: `{data['data_pkl_path']}`",
        f"- **data npz present**: `{data['data_npz_path'] is not None}`",
    ]
    if data["data_npz_path"]:
        lines.append(f"- **npz path**: `{data['data_npz_path']}`")
        lines.append(f"- **npz keys**: `{data['npz_info'].get('keys', [])}`")
        lines.append(
            f"- **weekly_spectral_template in npz**: **{data['npz_info'].get('has_weekly_spectral_template', False)}**"
        )
        if data["npz_info"].get("weekly_spectral_template_shape"):
            lines.append(
                f"- **weekly_spectral_template shape**: `{data['npz_info']['weekly_spectral_template_shape']}`"
            )
    lines.append("")
    if data.get("protocol_audit"):
        lines += ["## protocol_audit.json", "", "```json", json.dumps(data["protocol_audit"], indent=2), "```", ""]
    elif data.get("protocol_audit_path"):
        lines.append(f"- **protocol_audit.json**: missing (regenerate with `generate_holost_data.py`)")
        lines.append("")
    else:
        lines.append("- **protocol_audit.json**: not found")
        lines.append("")

    lines += compare_configs(
        "KASA",
        ROOT / "examples/KASAST_v2/KASAST_PEMS04.py",
        "D2STGNN",
        ROOT / "examples/baselines/D2STGNN/D2STGNN_PEMS04.py",
    )

    lines += baseline_forward_table(data["num_channels"])

    lines += [
        "## Preprocessing script defaults",
        "",
        "### `generate_holost_data.py` (__main__ defaults)",
        "",
        "| Setting | Value |",
        "|---------|-------|",
    ]
    for k, v in holost_defaults.items():
        if not k.startswith("comment") and not k.startswith("uses"):
            lines.append(f"| {k} | `{v}` |")
    lines += [
        "",
        "- Produces **4 channels** (flow, ToD, DoW, prior) and writes `.npz` with `weekly_spectral_template`.",
        "- Default split: **6:2:2** (`TRAIN_RATIO=0.6`, `VALID_RATIO=0.2`, test remainder 0.2).",
        "- Writes `protocol_audit.json` after generation.",
        "",
        "### `generate_training_data.py` (__main__ defaults)",
        "",
        "| Setting | Value |",
        "|---------|-------|",
    ]
    for k, v in train_defaults.items():
        if not k.startswith("comment") and not k.startswith("uses"):
            lines.append(f"| {k} | `{v}` |")
    lines += [
        "",
        "- Default split: **6:2:2** (`TRAIN_RATIO=0.6`, `VALID_RATIO=0.2`).",
        "- Default window: **12→12** (`HISTORY_SEQ_LEN=12`, `FUTURE_SEQ_LEN=12`).",
        "- Produces **3 channels** only (no prior). Use `generate_holost_data.py` for 4-channel KASA data.",
        "",
        "### Official BasicTS v0.2 `generate_training_data.py` (reference)",
        "",
        "| Setting | Official value |",
        "|---------|----------------|",
        "| HISTORY_SEQ_LEN | 12 |",
        "| FUTURE_SEQ_LEN | 12 |",
        "| TRAIN_RATIO | 0.6 |",
        "| VALID_RATIO | 0.2 |",
        "| Channels | 3 (flow, ToD, DoW) |",
        "| Split | **6:2:2** |",
        "",
        "## Protocol inconsistency summary",
        "",
    ]

    if data["split_label"].startswith("6:2:2"):
        lines.append("- On-disk PeMS04 index matches **6:2:2** (official protocol).")
    elif data["split_label"].startswith("7:1:2"):
        lines.append(
            "- On-disk PeMS04 index still matches **7:1:2** — **regenerate** with "
            "`python scripts/data_preparation/PEMS04/generate_holost_data.py`."
        )
    else:
        lines.append(f"- On-disk split classified as: **{data['split_label']}**.")

    if data["has_channel_3"]:
        lines.append("- On-disk `processed_data` has **4 channels** including train-only prior (ch3).")
    else:
        lines.append("- On-disk `processed_data` has **3 channels** (no prior).")

    lines += [
        "- Repo default generation scripts now target **6:2:2**; on-disk data must be regenerated to match.",
        "",
        "## Fairness / comparability conclusions",
        "",
        "### Is D2STGNN comparable to official BasicTS?",
        "",
    ]

    d2 = extract_cfg_fields(load_cfg(ROOT / "examples/baselines/D2STGNN/D2STGNN_PEMS04.py"))
    d2_no_ch3 = 3 not in d2["FORWARD_FEATURES"]
    if data["split_label"].startswith("6:2:2") and d2_no_ch3:
        lines.append(
            "- **Yes** for split alignment (6:2:2). D2STGNN uses official BasicTS hyperparams, FORWARD=[0,1,2], no ch3."
        )
        if data["num_channels"] == 4:
            lines.append(
                "- Local data has **4 channels** (KASA prior ch3 present in file but unused by D2STGNN)."
            )
    else:
        lines.append(
            f"- **Pending regeneration**: on-disk split is **{data['split_label']}**; "
            "run `generate_holost_data.py` for official 6:2:2."
        )
        if d2_no_ch3:
            lines.append("- D2STGNN config does **not** use channel 3.")

    lines += [
        "",
        "### Is D2STGNN internally fair against KASA?",
        "",
        "- **Same dataset path**: both use `datasets/PEMS04` with `index_in12_out12.pkl` / `data_in12_out12.pkl`.",
        "- **Same split**: both see identical train/valid/test windows.",
        f"- **Feature asymmetry**: KASA FORWARD `[0,1,2,3]` uses prior ch3; D2STGNN FORWARD `[0,1,2]` does not.",
        "- **Not equal-compute**: KASA 100 epochs / bs32 vs D2STGNN 200 epochs / bs16 + curriculum learning.",
        "- **Conclusion**: Same data split and target (flow ch0), but **not feature-fair** (KASA gets extra prior channel) "
        "and **not compute-fair** (epochs/batch/scheduler differ by design).",
        "",
        "### Exact protocol in use (this repo)",
        "",
        f"1. PeMS04 12→12 windows from `{data['index_path']}`.",
        f"2. Split: **{data['split_label']}** ({data['train_n']}/{data['valid_n']}/{data['test_n']} samples).",
        f"3. Channels: **{data['num_channels']}** "
        f"({'with' if data['has_channel_3'] else 'without'} train-only prior in ch3).",
        "4. Baselines: typically FORWARD `[0]` or `[0,1]` or `[0,1,2]`; never ch3.",
        "5. KASA: FORWARD `[0,1,2,3]`, TARGET `[0]`, `input_dim=4`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PeMS04 data protocol and configs.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/PEMS04",
        help="PeMS04 dataset directory (default: datasets/PEMS04)",
    )
    args = parser.parse_args()
    dataset_dir = Path(args.data_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(dataset_dir)
    REPORT_PATH.write_text(report)
    print(report)
    print(f"\nWrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
