#!/usr/bin/env python3
"""Build per-fold matched-maturity OOF: Z3 (sharded fp16) + four raw-scale route MAEs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    load_route_costs,
    route_to_key,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    build_refinement_route_index_map,
    gains_from_route_losses,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    SequentialF2FEnvironment,
)
from basicts.data import SCALER_REGISTRY
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from basicts.utils import load_pkl
from scripts.matched_maturity_lib import (
    ROUTES,
    STABLE_CFG,
    dump_json,
    fold_dir,
    get_test_access_count,
    install_test_access_guard,
    load_manifest,
    reset_test_access,
    sha1_file,
    sha1_indices,
)


def _extract_state_dict(obj):
    if not isinstance(obj, dict):
        raise TypeError(type(obj))
    if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        sd = obj["model_state_dict"]
    elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    else:
        sd = obj
    if any(k.startswith("module.") for k in sd):
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    return sd


def per_sample_masked_mae(pred, target, null_val: float = 0.0):
    if null_val != null_val:
        mask = ~torch.isnan(target)
    else:
        mask = ~torch.isclose(
            target,
            torch.tensor(null_val, device=target.device, dtype=target.dtype),
            atol=5e-5,
            rtol=0.0,
        )
    err = (pred - target).abs() * mask.float()
    denom = mask.float().flatten(1).sum(dim=1).clamp_min(1.0)
    return err.flatten(1).sum(dim=1) / denom


def find_best_ckpt(meta_dir: Path) -> Path:
    hits = sorted(meta_dir.rglob("BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"))
    if not hits:
        raise FileNotFoundError(f"no best_val_MAE ckpt under {meta_dir}")
    return hits[-1]


def find_last_ckpt(meta_dir: Path) -> Path | None:
    hits = sorted(meta_dir.rglob("BudgetConditionedAdaptiveF2FNet_*.pt"))
    # prefer non-best last epoch files
    lasts = [p for p in hits if "best_" not in p.name]
    return lasts[-1] if lasts else (hits[-1] if hits else None)


class Z3ShardWriter:
    """Append Z3 tensors as fp16 memmap shards; metadata maps sample_index -> offset."""

    def __init__(self, out_dir: Path, shard_size: int = 512):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.meta: dict[str, dict] = {}
        self._shard_id = 0
        self._buf: list[np.ndarray] = []
        self._indices: list[int] = []
        self.shape = None
        self.dtype = "float16"

    def add(self, sample_index: int, z3: torch.Tensor) -> None:
        arr = z3.detach().float().cpu().numpy().astype(np.float16)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if self.shape is None:
            self.shape = list(arr.shape)
        elif list(arr.shape) != self.shape:
            raise RuntimeError(f"Z3 shape mismatch {arr.shape} vs {self.shape}")
        self._buf.append(arr)
        self._indices.append(int(sample_index))
        if len(self._buf) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        shard_name = f"z3_shard_{self._shard_id:04d}.npy"
        path = self.out_dir / shard_name
        stacked = np.stack(self._buf, axis=0)  # [B, H, N, C]
        np.save(path, stacked)
        for off, si in enumerate(self._indices):
            self.meta[str(si)] = {
                "sample_index": int(si),
                "shard": shard_name,
                "offset": int(off),
                "shape": list(self.shape),
                "dtype": self.dtype,
                "H": int(self.shape[0]),
                "N": int(self.shape[1]),
                "C": int(self.shape[2]),
            }
        self._shard_id += 1
        self._buf = []
        self._indices = []

    def close(self) -> dict:
        self._flush()
        index_path = self.out_dir / "z3_index.json"
        payload = {
            "storage": "sharded_fp16_npy",
            "shape": self.shape,
            "dtype": self.dtype,
            "n": len(self.meta),
            "map": self.meta,
        }
        index_path.write_text(json.dumps(payload, indent=2))
        return payload


def build_fold_oof(
    fold: int,
    *,
    device: str = "cuda:0",
    batch_size: int = 8,
    smoke: bool = False,
    max_holdout: int | None = None,
    manifest_path: Path | None = None,
) -> dict:
    reset_test_access()
    install_test_access_guard()
    man = load_manifest(manifest_path)
    fold_rec = next(f for f in man["folds"] if int(f["fold"]) == int(fold))
    hold = list(fold_rec["heldout_sample_indices"])
    if smoke:
        hold = hold[: int(max_holdout or 4)]
    elif max_holdout is not None:
        hold = hold[: int(max_holdout)]

    meta_dir = fold_dir(fold, smoke=smoke)
    ckpt = find_best_ckpt(meta_dir)
    cfg_meta = json.loads((meta_dir / "training_config.json").read_text())
    cfg_path = ROOT / cfg_meta["cfg_path"]

    from easytorch.config import import_config

    rel = str(cfg_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    cfg = import_config(rel)

    dev = torch.device(device if ("cuda" not in device or torch.cuda.is_available()) else "cpu")
    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM)
    sd = _extract_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model.load_state_dict(sd, strict=False)
    model = model.to(dev).eval()
    model.set_forced_route(None)
    env = SequentialF2FEnvironment(model)

    # Route execution validation before full OOF
    data_dir = ROOT / "datasets/PEMS04"
    base = IndexedTimeSeriesForecastingDataset(
        str(data_dir / "data_in12_out12.pkl"),
        str(data_dir / "index_in12_out12.pkl"),
        "train",
    )
    fut0, hist0, _ = base[hold[0]]
    hist_b = hist0.unsqueeze(0).to(dev)[..., list(cfg.MODEL.FORWARD_FEATURES)]
    route_ok = {}
    with torch.no_grad():
        for r in ROUTES:
            model.set_forced_route(r)
            model.route_selection_mode = "forced"
            out = model(history_data=hist_b, train=False, return_all=False)
            route_ok[route_to_key(r)] = {"ok": True, "pred_shape": list(out.shape)}
        eq = env.sequential_route_equivalence_check(hist_b, atol=1e-6)
    if not (eq.get("quarter_ok") and eq.get("progressive_ok")):
        raise RuntimeError(f"prefix/resume equivalence failed: {eq}")

    scaler = load_pkl(str(data_dir / "scaler_in12_out12.pkl"))
    rescale = SCALER_REGISTRY.get(scaler["func"])
    null_val = float(getattr(cfg.TRAIN, "NULL_VAL", 0.0))
    forward_features = list(cfg.MODEL.FORWARD_FEATURES)
    target_features = list(cfg.MODEL.TARGET_FEATURES)
    ds = Subset(base, hold)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    routes = [list(r) for r in model.candidate_routes]
    assert routes == ROUTES, routes
    costs = load_route_costs(
        None,
        routes,
        int(cfg.DATASET_OUTPUT_LEN),
        cost_type=str(cfg.MODEL.PARAM.get("route_cost_type", "normalized_static_cost")),
    )
    index_map = build_refinement_route_index_map(routes, int(cfg.DATASET_OUTPUT_LEN))

    oof_dir = meta_dir / "oof"
    z3_writer = Z3ShardWriter(oof_dir / "z3", shard_size=256 if not smoke else 8)
    records = []
    summary = json.loads((meta_dir / "training_summary.json").read_text()) if (meta_dir / "training_summary.json").is_file() else {}
    train_sha = (meta_dir / "train_indices_sha1.txt").read_text().strip()
    teacher_sha = sha1_file(ckpt, 40)
    stable_sha = sha1_file(STABLE_CFG, 40)

    with torch.no_grad():
        for future, history, sample_index in loader:
            history = history.to(dev)[..., forward_features]
            future = future.to(dev)
            target = future[..., target_features]
            sis = [int(x) for x in sample_index.view(-1).tolist()]

            # Z3 via quarter prefix (H/4 executes once)
            pref = env.execute_quarter_prefix(history)
            zq = pref["Z_q"]
            for bi, si in enumerate(sis):
                z3_writer.add(si, zq[bi : bi + 1])

            route_mae = []
            for route in routes:
                model.set_forced_route(route)
                model.route_selection_mode = "forced"
                pred = model(history_data=history, train=False, return_all=False)
                y = target[..., : pred.shape[-1]]
                mae_b = per_sample_masked_mae(
                    rescale(pred, **scaler["args"]),
                    rescale(y, **scaler["args"]),
                    null_val=null_val,
                )
                route_mae.append(mae_b.detach().cpu())
            route_mae_t = torch.stack(route_mae, dim=0)
            for bi, si in enumerate(sis):
                route_losses = [
                    {
                        "route_id": rid,
                        "route": list(route),
                        "final_mae": float(route_mae_t[rid, bi].item()),
                        "cost": float(costs[rid]),
                    }
                    for rid, route in enumerate(routes)
                ]
                by_name = {
                    name: route_losses[index_map[name]]["final_mae"]
                    for name in ("direct", "half", "quarter", "progressive")
                }
                g = gains_from_route_losses(by_name)
                records.append(
                    {
                        "sample_index": int(si),
                        "fold_id": int(fold),
                        "split": "matched_maturity_oof_train_holdout",
                        "route_final_losses": route_losses,
                        "L12": by_name["direct"],
                        "L6_12": by_name["half"],
                        "L3_12": by_name["quarter"],
                        "L3_6_12": by_name["progressive"],
                        "G3": g["g3"],
                        "G6": g["g6"],
                        "G36": g["g36"],
                        "Z3_ref": {"sample_index": int(si)},
                        "teacher_checkpoint": str(ckpt.relative_to(ROOT)),
                        "teacher_checkpoint_sha1": teacher_sha,
                        "stable_config_sha1": stable_sha,
                        "train_index_sha1": train_sha,
                        "teacher_best_epoch": summary.get("best_epoch"),
                        "teacher_best_val_MAE": summary.get("best_val_MAE"),
                    }
                )

    z3_meta = z3_writer.close()
    # attach concrete Z3 refs
    for r in records:
        si = str(r["sample_index"])
        r["Z3_ref"] = z3_meta["map"][si]

    test_count = get_test_access_count()
    out = {
        "metadata": {
            "fold_id": int(fold),
            "n_records": len(records),
            "n_holdout_expected": len(hold),
            "loss_scale": "raw_physical_scale",
            "candidate_routes": routes,
            "candidate_routes_order": [route_to_key(r) for r in routes],
            "route_costs": list(map(float, costs)),
            "teacher_checkpoint": str(ckpt.relative_to(ROOT)),
            "teacher_checkpoint_sha1": teacher_sha,
            "stable_config_sha1": stable_sha,
            "train_index_sha1": train_sha,
            "teacher_best_epoch": summary.get("best_epoch"),
            "teacher_best_val_MAE": summary.get("best_val_MAE"),
            "route_execution_validation": {"forced_routes": route_ok, "prefix_resume": eq},
            "z3_storage": {
                "dir": str((oof_dir / "z3").relative_to(ROOT)),
                "strategy": "sharded_fp16_npy",
                "shape": z3_meta["shape"],
                "dtype": z3_meta["dtype"],
            },
            "TEST_ACCESS_COUNT": test_count,
            "smoke": bool(smoke),
        },
        "records": records,
    }
    if len(records) != len(hold):
        raise RuntimeError(f"OOF n mismatch {len(records)} vs {len(hold)}")
    if test_count != 0:
        raise RuntimeError(f"TEST_ACCESS_COUNT={test_count}")
    dump_json(oof_dir / "fold_oof_oracle.json", out)
    dump_json(meta_dir / "oof_complete.json", {"fold": fold, "n": len(records), "ckpt": str(ckpt)})
    print(json.dumps({"fold": fold, "n": len(records), "TEST_ACCESS_COUNT": test_count, "eq": eq}, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-holdout", type=int, default=None)
    ap.add_argument("--manifest", default="results/matched_maturity_crossfit_manifest.json")
    args = ap.parse_args()
    build_fold_oof(
        args.fold,
        device=args.device,
        batch_size=args.batch_size,
        smoke=args.smoke,
        max_holdout=args.max_holdout,
        manifest_path=Path(args.manifest),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
