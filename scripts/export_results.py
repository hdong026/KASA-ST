#!/usr/bin/env python3
"""Export experiment metrics from KASA-ST / FoRC-ST checkpoint folders."""

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None


RESULT_SUFFIXES = {".log", ".txt", ".csv", ".json", ".yaml", ".yml"}

METRIC_PATTERN = re.compile(
    r"(?i)(?:(test|val|train)[_/])?(?:metric[/_.-]|loss[/_.-])?"
    r"(mae|rmse|mape)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)%?"
)
COMPACT_PATTERN = re.compile(
    r"(?i)(?:best\s+results?\s*:?\s*)?"
    r"mae\s+([0-9]+(?:\.[0-9]+)?)\s+"
    r"rmse\s+([0-9]+(?:\.[0-9]+)?)\s+"
    r"mape\s+([0-9]+(?:\.[0-9]+)?)%?"
)

KNOWN_DATASETS = [
    ("pems-bay", "PEMS-BAY"),
    ("pemsbay", "PEMS-BAY"),
    ("metr-la", "METR-LA"),
    ("metr_la", "METR-LA"),
    ("pems04", "PeMS04"),
    ("pems07", "PeMS07"),
    ("pems08", "PeMS08"),
]
KNOWN_DATASETS.sort(key=lambda item: len(item[0]), reverse=True)

COLUMNS = [
    "experiment",
    "dataset",
    "tag",
    "mae",
    "rmse",
    "mape",
    "source_file",
    "source_line",
    "status",
]


def infer_dataset_and_tag(exp_name):
    lower = exp_name.lower()
    for token, normalized in KNOWN_DATASETS:
        idx = lower.find(token)
        if idx != -1:
            rest = exp_name[idx + len(token):]
            tag = rest.lstrip("_-")
            return normalized, tag
    return "", exp_name


def matches_dataset_filter(exp_name, dataset_filter):
    if not dataset_filter:
        return True
    return dataset_filter.lower() in exp_name.lower()


def detect_phase(line_lower, explicit_phase=None):
    if explicit_phase:
        return explicit_phase.lower()
    if "result <test>" in line_lower or " on test " in line_lower:
        return "test"
    if "result <val>" in line_lower or " validation" in line_lower:
        return "val"
    if "result <train>" in line_lower:
        return "train"
    if re.search(r"(?<![a-z])test(?![a-z])", line_lower):
        return "test"
    if re.search(r"(?<![a-z])val(?![a-z])", line_lower):
        return "val"
    if re.search(r"(?<![a-z])train(?![a-z])", line_lower):
        return "train"
    return "unknown"


def parse_metric_line(line, line_no):
    records = []
    line_lower = line.lower()

    compact = COMPACT_PATTERN.search(line)
    if compact:
        phase = detect_phase(line_lower)
        records.append(
            {
                "mae": float(compact.group(1)),
                "rmse": float(compact.group(2)),
                "mape": float(compact.group(3)),
                "phase": phase,
                "source_file": None,
                "source_line": line_no,
                "line_text": line,
                "is_horizon": "horizon" in line_lower,
                "is_aggregate": "result <test>" in line_lower or "test_mae" in line_lower,
            }
        )
        return records

    metrics = {}
    phases = set()
    for match in METRIC_PATTERN.finditer(line):
        phase_hint, metric_name, value = match.groups()
        metric_key = metric_name.lower()
        metrics[metric_key] = float(value)
        if phase_hint:
            phases.add(phase_hint.lower())

    if not metrics:
        return records

    if len(phases) == 1:
        phase = next(iter(phases))
    elif len(phases) > 1:
        phase = "mixed"
    else:
        phase = detect_phase(line_lower)

    record = {
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "mape": metrics.get("mape"),
        "phase": phase,
        "source_file": None,
        "source_line": line_no,
        "line_text": line,
        "is_horizon": bool(re.search(r"horizon\s+\d+", line_lower)),
        "is_aggregate": (
            "result <test>" in line_lower
            or "result <val>" in line_lower
            or (
                metrics.get("mae") is not None
                and metrics.get("rmse") is not None
                and metrics.get("mape") is not None
                and "test_mae" in line_lower
            )
        ),
    }
    records.append(record)
    return records


def parse_file(file_path):
    records = []
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return records, str(exc)

    for line_no, line in enumerate(text.splitlines(), start=1):
        for record in parse_metric_line(line, line_no):
            record["source_file"] = str(file_path)
            records.append(record)
    return records, None


def is_complete(record):
    return (
        record.get("mae") is not None
        and record.get("rmse") is not None
        and record.get("mape") is not None
    )


def phase_rank(phase):
    if phase == "test":
        return 3
    if phase == "val":
        return 2
    if phase == "train":
        return 1
    return 0


def select_best_record(records):
    if not records:
        return None

    complete = [rec for rec in records if is_complete(rec)]
    if not complete:
        return None

    test_records = [rec for rec in complete if rec.get("phase") == "test"]
    pool = test_records if test_records else complete

    non_horizon = [rec for rec in pool if not rec.get("is_horizon")]
    if non_horizon:
        pool = non_horizon

    aggregate = [rec for rec in pool if rec.get("is_aggregate")]
    if aggregate:
        pool = aggregate

    pool.sort(key=lambda rec: (rec["mae"], rec.get("source_line") or 0))
    return pool[0]


def find_result_files(exp_dir):
    files = []
    for path in exp_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in RESULT_SUFFIXES:
            files.append(path)
    return sorted(files)


def collect_experiment_dirs(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    experiment_dirs = []
    for path in sorted(root.iterdir()):
        if path.is_dir():
            experiment_dirs.append(path)
    return experiment_dirs


def analyze_experiment(exp_dir):
    all_records = []
    errors = []

    for file_path in find_result_files(exp_dir):
        records, error = parse_file(file_path)
        if error:
            errors.append(f"{file_path}: {error}")
        all_records.extend(records)

    best = select_best_record(all_records)
    dataset, tag = infer_dataset_and_tag(exp_dir.name)

    if best is not None:
        return {
            "experiment": exp_dir.name,
            "dataset": dataset,
            "tag": tag,
            "mae": best["mae"],
            "rmse": best["rmse"],
            "mape": best["mape"],
            "source_file": best["source_file"],
            "source_line": best["source_line"],
            "status": "ok",
        }

    if errors:
        return {
            "experiment": exp_dir.name,
            "dataset": dataset,
            "tag": tag,
            "mae": "",
            "rmse": "",
            "mape": "",
            "source_file": errors[0],
            "source_line": "",
            "status": "parse_error",
        }

    return {
        "experiment": exp_dir.name,
        "dataset": dataset,
        "tag": tag,
        "mae": "",
        "rmse": "",
        "mape": "",
        "source_file": "",
        "source_line": "",
        "status": "missing_metrics",
    }


def sort_rows(rows, sort_key, descending):
    def sort_value(row):
        value = row.get(sort_key, "")
        if value == "" or value is None:
            return float("inf") if not descending else float("-inf")
        return float(value)

    return sorted(rows, key=sort_value, reverse=descending)


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        pd.DataFrame(rows, columns=COLUMNS).to_csv(out_path, index=False)
        return

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, md_path):
    md_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(value):
        if value == "" or value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(COLUMNS)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(fmt(row[col]) for col in COLUMNS) + " |")

    content = "\n".join([header, separator] + body) + "\n"
    md_path.write_text(content, encoding="utf-8")


def print_summary(rows, scanned_count, csv_path, md_path=None):
    ok_rows = [row for row in rows if row["status"] == "ok"]
    print(f"Scanned experiment folders: {scanned_count}")
    print(f"Successfully parsed results: {len(ok_rows)}")
    if ok_rows:
        best = min(ok_rows, key=lambda row: float(row["mae"]))
        print(
            "Best experiment by MAE: "
            f"{best['experiment']} (MAE={float(best['mae']):.4f}, "
            f"RMSE={float(best['rmse']):.4f}, MAPE={float(best['mape']):.4f})"
        )
    else:
        print("Best experiment by MAE: N/A")
    print(f"Saved CSV: {csv_path}")
    if md_path is not None:
        print(f"Saved Markdown: {md_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan checkpoint folders and export experiment metric summaries."
    )
    parser.add_argument("--root", type=str, default="checkpoints", help="Checkpoint root directory")
    parser.add_argument("--dataset", type=str, default=None, help="Filter experiment folders by dataset token")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    parser.add_argument("--markdown", type=str, default=None, help="Optional output Markdown table path")
    parser.add_argument("--sort", type=str, default="mae", choices=["mae", "rmse", "mape"], help="Sort key")
    parser.add_argument("--descending", action="store_true", help="Sort in descending order")
    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(args.root)
    experiment_dirs = collect_experiment_dirs(root)
    filtered_dirs = [
        exp_dir for exp_dir in experiment_dirs
        if matches_dataset_filter(exp_dir.name, args.dataset)
    ]

    rows = [analyze_experiment(exp_dir) for exp_dir in filtered_dirs]
    rows = sort_rows(rows, args.sort, args.descending)

    out_path = Path(args.out)
    write_csv(rows, out_path)

    md_path = Path(args.markdown) if args.markdown else None
    if md_path is not None:
        write_markdown(rows, md_path)

    print_summary(rows, len(filtered_dirs), out_path.resolve(), md_path.resolve() if md_path else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
