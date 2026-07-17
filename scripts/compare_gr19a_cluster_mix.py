#!/usr/bin/env python3
"""Compare GR19a S12 cluster_mix results with G1 / GR14 / GR7 baselines."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASELINES = [
    "G1_final_adaptive",
    "GR14_two_level_sparse",
    "GR7_sparse_topk",
    "GR19a_cap_only_spectral_S12_cluster_mix",
]


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_mae(rows: list[dict], horizon: int, variant: str, seed: int) -> dict | None:
    for r in rows:
        if (
            str(r.get("horizon")) == str(horizon)
            and r.get("variant") == variant
            and str(r.get("seed")) == str(seed)
            and r.get("status") == "ok"
        ):
            return r
    return None


def aggregate(rows: list[dict], horizon: int, variant: str) -> dict:
    hits = [
        r for r in rows
        if str(r.get("horizon")) == str(horizon)
        and r.get("variant") == variant
        and r.get("status") == "ok"
        and r.get("test_mae_at_best_val")
    ]
    maes = [float(r["test_mae_at_best_val"]) for r in hits]
    if not maes:
        return {"n": 0, "mean": None, "std": None}
    mean = sum(maes) / len(maes)
    std = (sum((x - mean) ** 2 for x in maes) / len(maes)) ** 0.5 if len(maes) > 1 else 0.0
    return {"n": len(maes), "mean": mean, "std": std}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mix_csv", default="results/pems04_gr19a_s12_cluster_mix_5seeds.csv")
    p.add_argument("--baseline_csv", default="results/pems04_16_32_64_unified_aux_seed1.csv")
    p.add_argument("--out", default="results/pems04_gr19a_s12_cluster_mix_comparison.md")
    args = p.parse_args()

    mix_csv = ROOT / args.mix_csv
    base_csv = ROOT / args.baseline_csv
    out_md = ROOT / args.out

    rows = load_rows(mix_csv) + load_rows(base_csv)

    lines = ["# GR19a S12 Cluster Mix vs Baselines\n\n"]
    for horizon in (32, 64):
        lines.append(f"## Horizon {horizon}\n\n")
        lines.append("| variant | n | MAE mean | MAE std |\n")
        lines.append("|---|---:|---:|---:|\n")
        stats = []
        for variant in BASELINES:
            s = aggregate(rows, horizon, variant)
            stats.append((variant, s))
            mean_s = f"{s['mean']:.4f}" if s["mean"] is not None else "—"
            std_s = f"{s['std']:.4f}" if s["std"] is not None else "—"
            lines.append(f"| {variant} | {s['n']} | {mean_s} | {std_s} |\n")
        ok = [x for x in stats if x[1]["mean"] is not None]
        if ok:
            best = min(ok, key=lambda x: x[1]["mean"])
            lines.append(f"\n最优：**{best[0]}** (MAE={best[1]['mean']:.4f})\n\n")
        else:
            lines.append("\n")

    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
