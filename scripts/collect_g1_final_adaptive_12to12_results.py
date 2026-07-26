#!/usr/bin/env python3
"""Parse G1_final_adaptive 12->12 training logs and write summary tables."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_DIR = ROOT / "results" / "g1_final_adaptive_12to12"
LOG_DIR = ROOT / "logs" / "g1_final_adaptive_12to12"

DATASETS = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]

VAL_RE = re.compile(r"Result\s*<val>:\s*\[.*?val_MAE:\s*([0-9.]+)", re.I)
TEST_BLOCK = re.compile(r"Result\s*<test>:\s*\[(.*?)\]", re.I | re.S)
HORIZON_RE = re.compile(
    r"Evaluate best model on test data for horizon (\d+), Test MAE: ([0-9.]+), Test RMSE: ([0-9.]+), Test MAPE: ([0-9.]+)"
)


def find_training_log(dataset: str) -> Path | None:
    ckpt_root = ROOT / "checkpoints" / f"G1_final_adaptive_{dataset}_12to12"
    if not ckpt_root.is_dir():
        return None
    logs = sorted(ckpt_root.rglob("training_log_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if logs:
        return logs[0]
    wrapper = LOG_DIR / f"g1_final_adaptive_{dataset.lower()}_train.log"
    return wrapper if wrapper.is_file() else None


def parse_summary(dataset: str, log_path: Path, cfg_path: str) -> tuple[dict, list[dict]]:
    text = log_path.read_text(errors="replace")
    val_maes = [float(x) for x in VAL_RE.findall(text)]
    best_val = min(val_maes) if val_maes else float("nan")
    best_val_idx = val_maes.index(best_val) if val_maes else -1

    epoch_matches = re.findall(r"Epoch\s+(\d+)\s+/\s+\d+", text)
    best_epoch = epoch_matches[best_val_idx] if best_val_idx >= 0 and best_val_idx < len(epoch_matches) else ""

    # test at best val: use test block immediately after best val save if possible
    best_test = {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    save_pat = re.compile(rf"ChainForecasting_best_val_MAE\.pt saved")
    saves = list(save_pat.finditer(text))
    if saves:
        last_save = saves[-1].end()
        tail = text[last_save : last_save + 8000]
        m = TEST_BLOCK.search(tail)
        if m:
            chunk = m.group(1)
            for key, pat in [("mae", r"test_MAE:\s*([0-9.]+)"), ("rmse", r"test_RMSE:\s*([0-9.]+)"), ("mape", r"test_MAPE:\s*([0-9.]+)")]:
                mm = re.search(pat, chunk, re.I)
                if mm:
                    best_test[key] = float(mm.group(1))

    # fallback: lowest test MAE block in full log
    if best_test["mae"] != best_test["mae"]:  # nan
        best_mae = None
        for m in TEST_BLOCK.finditer(text):
            chunk = m.group(1)
            mm = re.search(r"test_MAE:\s*([0-9.]+)", chunk, re.I)
            if not mm:
                continue
            v = float(mm.group(1))
            if best_mae is None or v < best_mae:
                best_mae = v
                for key, pat in [("mae", r"test_MAE:\s*([0-9.]+)"), ("rmse", r"test_RMSE:\s*([0-9.]+)"), ("mape", r"test_MAPE:\s*([0-9.]+)")]:
                    mx = re.search(pat, chunk, re.I)
                    best_test[key] = float(mx.group(1)) if mx else float("nan")

    row = {
        "dataset": dataset,
        "model": "G1_final_adaptive",
        "input_len": 12,
        "output_len": 12,
        "val_mae_best": f"{best_val:.4f}" if val_maes else "",
        "test_mae_at_best_val": f"{best_test['mae']:.4f}" if best_test['mae'] == best_test['mae'] else "",
        "test_rmse": f"{best_test['rmse']:.4f}" if best_test['rmse'] == best_test['rmse'] else "",
        "test_mape": f"{best_test['mape']:.4f}" if best_test['mape'] == best_test['mape'] else "",
        "epoch": best_epoch,
        "config_path": cfg_path,
        "log_path": str(log_path),
    }

    # horizon metrics from last test evaluation (best-val checkpoint eval at end of training)
    horizon_rows = []
    horizon_sections = list(HORIZON_RE.finditer(text))
    if horizon_sections:
        # take last occurrence set (final best-val test eval)
        last_horizons = {}
        for m in horizon_sections:
            h = int(m.group(1))
            if h in (3, 6, 12):
                last_horizons[h] = {
                    "dataset": dataset,
                    "horizon": h,
                    "mae": float(m.group(2)),
                    "rmse": float(m.group(3)),
                    "mape": float(m.group(4)),
                }
        horizon_rows = [last_horizons[h] for h in sorted(last_horizons)]

    return row, horizon_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_horizon = []

    for dataset in args.datasets:
        dataset = dataset.upper()
        cfg_path = f"examples/ChainForecasting/G1_final_adaptive_{dataset}_12to12.py"
        log_path = find_training_log(dataset)
        if log_path is None:
            print(f"[skip] {dataset}: no log found")
            continue
        row, horizons = parse_summary(dataset, log_path, cfg_path)
        summary_rows.append(row)
        all_horizon.extend(horizons)

        ds_csv = RESULT_DIR / f"g1_final_adaptive_{dataset.lower()}_12to12.csv"
        with open(ds_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)

    if summary_rows:
        headers = list(summary_rows[0].keys())
        with open(RESULT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(summary_rows)

        with open(RESULT_DIR / "summary.md", "w", encoding="utf-8") as f:
            f.write("# G1_final_adaptive 12→12 summary\n\n")
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for r in summary_rows:
                f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")

    if all_horizon:
        hheaders = ["dataset", "horizon", "mae", "rmse", "mape"]
        with open(RESULT_DIR / "horizon_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hheaders)
            w.writeheader()
            w.writerows(all_horizon)
        with open(RESULT_DIR / "horizon_summary.md", "w", encoding="utf-8") as f:
            f.write("# G1_final_adaptive 12→12 horizon metrics (3/6/12)\n\n")
            f.write("| dataset | horizon | MAE | RMSE | MAPE |\n|---|---|---|---|---|\n")
            for r in all_horizon:
                f.write(f"| {r['dataset']} | {r['horizon']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['mape']:.4f} |\n")

    print(f"Wrote {RESULT_DIR}")


if __name__ == "__main__":
    main()
