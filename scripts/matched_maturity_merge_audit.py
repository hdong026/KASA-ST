#!/usr/bin/env python3
"""Merge matched-maturity OOF folds + environment agreement audits (diagnostic only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.matched_maturity_lib import (
    FUTURE_PROTOCOL,
    OLD_CF_AGREE,
    OLD_CF_ORACLE,
    STABLE_TRAIN_ORACLE,
    dump_json,
    fold_dir,
    load_manifest,
)


def _route_losses_dict(rec: dict) -> dict[str, float]:
    if "L12" in rec:
        return {
            "L12": float(rec["L12"]),
            "L6_12": float(rec["L6_12"]),
            "L3_12": float(rec["L3_12"]),
            "L3_6_12": float(rec["L3_6_12"]),
        }
    rfl = rec["route_final_losses"]
    by = {tuple(x["route"]): float(x["final_mae"]) for x in rfl}
    return {
        "L12": by[(12,)],
        "L6_12": by[(6, 12)],
        "L3_12": by[(3, 12)],
        "L3_6_12": by[(3, 6, 12)],
    }


def _gains(L: dict) -> dict[str, float]:
    return {
        "G3": L["L12"] - L["L3_12"],
        "G6": L["L12"] - L["L6_12"],
        "G36": L["L3_12"] - L["L3_6_12"],
    }


def merge_oof(*, smoke: bool = False) -> dict:
    man = load_manifest()
    records = []
    folds_used = []
    for f in man["folds"]:
        fold = int(f["fold"])
        path = fold_dir(fold, smoke=smoke) / "oof" / "fold_oof_oracle.json"
        if not path.is_file():
            if smoke:
                continue
            raise FileNotFoundError(path)
        blob = json.loads(path.read_text())
        records.extend(blob["records"])
        folds_used.append(fold)
    if smoke and not records:
        raise FileNotFoundError("smoke merge: no fold OOF files found")
    by_si = {}
    dups = []
    for r in records:
        si = int(r["sample_index"])
        if si in by_si:
            dups.append(si)
        by_si[si] = r

    idxs = sorted(by_si.keys())
    expected = list(range(10181)) if not smoke else idxs
    if smoke:
        coverage_ok = len(idxs) == len(set(idxs)) and len(dups) == 0
        coverage_msg = "smoke_partial_ok"
    else:
        coverage_ok = idxs == expected and len(dups) == 0
        coverage_msg = "exact_0_10180" if coverage_ok else "OOF_COVERAGE_FAIL"

    if not coverage_ok and not smoke:
        dump_json(
            "results/matched_maturity_crossfit_oof_oracle.json",
            {"error": "OOF_COVERAGE_FAIL", "n": len(idxs), "dups": dups[:20]},
        )
        raise RuntimeError("OOF_COVERAGE_FAIL")

    merged_records = []
    for si in idxs:
        r = by_si[si]
        L = _route_losses_dict(r)
        g = _gains(L)
        if any(k not in r for k in ("fold_id", "Z3_ref", "teacher_checkpoint_sha1")):
            raise RuntimeError(f"missing fields for sample {si}")
        if len(r.get("route_final_losses", [])) != 4:
            raise RuntimeError(f"missing 4 route losses for {si}")
        merged_records.append(
            {
                **r,
                **L,
                **g,
            }
        )

    out = {
        "metadata": {
            "scheme": "matched_maturity_crossfit_v2",
            "n_unique": len(merged_records),
            "min_index": min(idxs) if idxs else None,
            "max_index": max(idxs) if idxs else None,
            "coverage": coverage_msg,
            "folds_used": folds_used,
            "smoke": bool(smoke),
            "future_protocol": FUTURE_PROTOCOL,
            "stable_train_oracle_role": "DIAGNOSTIC_ONLY",
        },
        "records": merged_records,
    }
    dump_json("results/matched_maturity_crossfit_oof_oracle.json", out)
    return out


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    from scipy import stats

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3:
        return float("nan")
    r = stats.spearmanr(a, b)
    return float(r.correlation)


def _sign_agree(a, b, eps: float | None = None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if eps is not None:
        m = np.abs(b) > float(eps)
        a, b = a[m], b[m]
    if len(a) == 0:
        return {"n": 0, "agreement": float("nan")}
    return {
        "n": int(len(a)),
        "agreement": float(np.mean(np.sign(a) == np.sign(b))),
    }


def _bootstrap_ci(a, b, fn, n_boot: int = 400, seed: int = 1):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    if n < 10:
        return {"lo": float("nan"), "hi": float("nan")}
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(fn(a[idx], b[idx]))
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"lo": float("nan"), "hi": float("nan")}
    return {"lo": float(np.quantile(vals, 0.025)), "hi": float(np.quantile(vals, 0.975))}


def _gain_block(gm, gs):
    return {
        "n": int(len(gm)),
        "pearson": _pearson(gm, gs),
        "spearman": _spearman(gm, gs),
        "sign_agreement": _sign_agree(gm, gs),
        "sign_agreement_excl_0p01": _sign_agree(gm, gs, 0.01),
        "sign_agreement_excl_0p05": _sign_agree(gm, gs, 0.05),
        "mae": float(np.mean(np.abs(np.asarray(gm) - np.asarray(gs)))),
        "matched_mean": float(np.mean(gm)),
        "matched_std": float(np.std(gm)),
        "matched_median": float(np.median(gm)),
        "stable_mean": float(np.mean(gs)),
        "stable_std": float(np.std(gs)),
        "stable_median": float(np.median(gs)),
        "spearman_bootstrap_95ci": _bootstrap_ci(gm, gs, _spearman),
        "sign_agreement_bootstrap_95ci": _bootstrap_ci(
            gm, gs, lambda x, y: _sign_agree(x, y)["agreement"]
        ),
    }


def _load_stable_gains() -> dict[int, dict]:
    blob = json.loads(STABLE_TRAIN_ORACLE.read_text())
    out = {}
    for r in blob["records"]:
        si = int(r["sample_index"])
        if si in out:
            continue  # intensities duplicate samples
        L = _route_losses_dict(r)
        out[si] = _gains(L)
        out[si]["L"] = L
    return out


def _load_old_cf_gains() -> dict[int, dict]:
    blob = json.loads(OLD_CF_ORACLE.read_text())
    out = {}
    for r in blob["records"]:
        si = int(r["sample_index"])
        L = _route_losses_dict(r)
        g = _gains(L)
        g["L"] = L
        g["fold"] = int(r.get("teacher_fold", r.get("fold_id", -1)))
        out[si] = g
    return out


def _best_route(L: dict, delta: float = 0.0) -> str:
    order = ["L12", "L6_12", "L3_12", "L3_6_12"]
    costs = {"L12": 0, "L6_12": 1, "L3_12": 2, "L3_6_12": 3}  # cheaper first among near-best
    best = min(order, key=lambda k: L[k])
    if delta <= 0:
        return best
    near = [k for k in order if L[k] <= L[best] + delta]
    return min(near, key=lambda k: (costs[k], L[k]))


def audit_environment(merged: dict | None = None, *, smoke: bool = False) -> dict:
    if merged is None:
        merged = json.loads((ROOT / "results/matched_maturity_crossfit_oof_oracle.json").read_text())
    if smoke:
        report = {
            "smoke": True,
            "note": "Smoke merge/audit schema only — NOT a scientific gate.",
            "n": len(merged["records"]),
            "gate": "SMOKE_NO_SCIENTIFIC_CLAIM",
        }
        dump_json("results/matched_maturity_crossfit_environment_agreement.json", report)
        dump_json(
            "results/matched_maturity_vs_old_crossfit.json",
            {"smoke": True, "note": "not scientific"},
        )
        return report

    stable = _load_stable_gains()
    matched = {}
    for r in merged["records"]:
        si = int(r["sample_index"])
        L = _route_losses_dict(r)
        matched[si] = {**_gains(L), "L": L, "fold_id": int(r["fold_id"])}

    shared = sorted(set(matched) & set(stable))
    by_fold = {k: {"G3": [], "G6": [], "G36": [], "sG3": [], "sG6": [], "sG36": []} for k in range(1, 6)}
    glob = {"G3": [], "G6": [], "G36": [], "sG3": [], "sG6": [], "sG36": []}
    strict_agree = 0
    near_agree = 0
    for si in shared:
        m, s = matched[si], stable[si]
        for name in ("G3", "G6", "G36"):
            glob[name].append(m[name])
            glob["s" + name].append(s[name])
            f = by_fold[m["fold_id"]]
            f[name].append(m[name])
            f["s" + name].append(s[name])
        if _best_route(m["L"], 0.0) == _best_route(s["L"], 0.0):
            strict_agree += 1
        if _best_route(m["L"], 0.05) == _best_route(s["L"], 0.05):
            near_agree += 1

    global_stats = {
        name: _gain_block(glob[name], glob["s" + name]) for name in ("G3", "G6", "G36")
    }
    fold_stats = {}
    for k, f in by_fold.items():
        fold_stats[f"Fold{k}"] = {
            name: _gain_block(f[name], f["s" + name]) for name in ("G3", "G6", "G36")
        }

    # GO / NO-GO gate (engineering)
    def _gate(stats):
        g3s, g6s = stats["G3"]["spearman"], stats["G6"]["spearman"]
        g3a = stats["G3"]["sign_agreement"]["agreement"]
        g6a = stats["G6"]["sign_agreement"]["agreement"]
        if (
            g3s >= 0.30
            and g6s >= 0.30
            and g3a >= 0.70
            and g6a >= 0.70
        ):
            return "STRONG_ENVIRONMENT_MATCH"
        if (
            g3s >= 0.20
            and g6s >= 0.20
            and g3a >= 0.65
            and g6a >= 0.65
        ):
            return "ADEQUATE_ENVIRONMENT_MATCH"
        if (g3s < 0.15 or g3a < 0.60) or (g6s < 0.15 or g6a < 0.60):
            return "FAILED_ENVIRONMENT_MATCH"
        return "INTERMEDIATE_ENVIRONMENT_MATCH"

    gate = _gate(global_stats)
    agree = {
        "n_shared": len(shared),
        "note": "stable TRAIN oracle is DIAGNOSTIC ONLY; never used as supervision",
        "future_protocol": FUTURE_PROTOCOL,
        "global": global_stats,
        "folds": fold_stats,
        "strict_best_route_agreement": strict_agree / max(1, len(shared)),
        "cheapest_near_best_delta_0p05_agreement": near_agree / max(1, len(shared)),
        "environment_gate": gate,
        "MATCHED_CROSSFIT_DID_NOT_SOLVE_ENVIRONMENT_MISMATCH": gate
        == "FAILED_ENVIRONMENT_MATCH",
    }
    dump_json("results/matched_maturity_crossfit_environment_agreement.json", agree)

    # vs old crossfit
    old = _load_old_cf_gains()
    old_shared = sorted(set(old) & set(stable))
    old_stats = {}
    for name in ("G3", "G6", "G36"):
        om = [old[si][name] for si in old_shared]
        os_ = [stable[si][name] for si in old_shared]
        old_stats[name] = {
            "pearson": _pearson(om, os_),
            "spearman": _spearman(om, os_),
            "sign_agreement": _sign_agree(om, os_)["agreement"],
        }
    # prefer archived rootcause numbers if present for consistency
    if OLD_CF_AGREE.is_file():
        archived = json.loads(OLD_CF_AGREE.read_text())
        g = archived.get("global", {})
        for name in ("G3", "G6", "G36"):
            if name in g:
                old_stats[name] = {
                    "pearson": g[name].get("pearson"),
                    "spearman": g[name].get("spearman"),
                    "sign_agreement": g[name].get("sign_agreement", {}).get("agreement"),
                }

    man = load_manifest()
    matched_train_minmax = [
        min(f["n_train_after_purge"] for f in man["folds"]),
        max(f["n_train_after_purge"] for f in man["folds"]),
    ]
    # old expanding sizes from historical knowledge / oracle folds
    old_train_minmax = [2013, 8121]

    # between-fold gain std heterogeneity (G3)
    fold_means = []
    for k in range(1, 6):
        vals = by_fold[k]["G3"]
        if vals:
            fold_means.append(float(np.mean(vals)))
    het = float(np.std(fold_means)) if fold_means else float("nan")

    table = {
        "OLD_CF": old_stats,
        "MATCHED_CF": {
            name: {
                "pearson": global_stats[name]["pearson"],
                "spearman": global_stats[name]["spearman"],
                "sign_agreement": global_stats[name]["sign_agreement"]["agreement"],
            }
            for name in ("G3", "G6", "G36")
        },
        "teacher_train_size_min_max": {
            "OLD_CF": old_train_minmax,
            "MATCHED_CF": matched_train_minmax,
        },
        "between_fold_G3_mean_std_heterogeneity": {
            "MATCHED_CF": het,
        },
        "absolute_improvement": {
            name: {
                "delta_pearson": float(
                    global_stats[name]["pearson"] - (old_stats[name]["pearson"] or 0)
                ),
                "delta_spearman": float(
                    global_stats[name]["spearman"] - (old_stats[name]["spearman"] or 0)
                ),
                "delta_sign_agreement": float(
                    global_stats[name]["sign_agreement"]["agreement"]
                    - (old_stats[name]["sign_agreement"] or 0)
                ),
            }
            for name in ("G3", "G6", "G36")
        },
        "environment_gate": gate,
    }
    dump_json("results/matched_maturity_vs_old_crossfit.json", table)
    return agree


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    merged = merge_oof(smoke=args.smoke)
    if not args.merge_only:
        audit_environment(merged, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
