"""ForecastTrajectory V2 runtime: data, gates, train, cache, policy, eval."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.ChainForecasting_arch import ChainForecasting
from basicts.archs.arch_zoo.ForecastTrajectory_arch.target_resolution import (
    assert_resolution_target_shapes,
    build_resolution_target,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_graph import ForecastTrajectoryGraph
from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_objective import token_normalized_mae
from basicts.archs.arch_zoo.ForecastTrajectoryV2_arch.forecast_trajectory_v2_net import (
    ForecastTrajectoryV2Net,
    expected_dag_transitions,
)
from basicts.archs.arch_zoo.ForecastTrajectoryV2_arch.online_policy_v2 import (
    OnlineTrajectoryPolicyV2,
    exact_prefix_dag_values,
)
from basicts.archs.arch_zoo.ForecastTrajectoryV2_arch.trajectory_cache_v2 import (
    TrajectoryCacheV2,
    TrajectoryCacheV2Writer,
    parse_prefix_key,
    prefix_key,
)
from basicts.data import SCALER_REGISTRY
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.utils import load_pkl

DATA_DIR = ROOT / "datasets" / "PEMS04"
DATA_FILE = DATA_DIR / "data_in12_out12.pkl"
INDEX_FILE = DATA_DIR / "index_in12_out12.pkl"
SCALER_FILE = DATA_DIR / "scaler_in12_out12.pkl"
ADJ_MX = DATA_DIR / "adj_mx.pkl"
CANONICAL_CKPT = (
    ROOT / "checkpoints" / "ChainForecasting_100" / "cd0ad9dcc9dd855c893d064f10450546"
    / "ChainForecasting_best_val_MAE.pt"
)

DEFAULT_STATES = [2, 3, 4, 6, 12]
CANONICAL_TAU = (3, 6, 12)
PRIMARY_PANEL = [(3, 6, 12), (12,), (3, 12), (6, 12), (2, 4, 12), (3, 4, 6, 12)]
ACCEPTANCE_TRAJS = [(12,), (3, 6, 12), (3, 12), (2, 3, 4, 6, 12)]
NULL_VAL = 0.0
FORWARD_FEATURES = [0, 1, 2, 3]
TARGET_FEATURES = [0]
CONTAINMENT_TOL = 0.10
HEADROOM_MIN = 0.02
PATH_PROB_ATOL = 1e-6


def git_head() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
            .decode().strip()
        )
    except Exception:
        return "UNKNOWN"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_f2f_args() -> dict:
    return {
        "node_size": 307,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 4,
        "main_input_dim": 3,
        "patch_len": 3,
        "stride": 4,
        "td_size": 288,
        "dw_size": 7,
        "d_td": 32,
        "d_dw": 32,
        "d_d": 32,
        "d_spa": 32,
        "if_time_in_day": True,
        "if_day_in_week": True,
        "if_spatial": True,
        "num_layer": 2,
        "spatial_scheme": "C",
        "adj_mx_path": str(ADJ_MX),
        "use_gcn": True,
        "gcn_hidden_dim": 64,
        "use_dynamic_spatial": True,
        "dyn_hidden_dim": 64,
        "dyn_topk": 20,
        "dyn_tau": 0.5,
        "dyn_static_weight": 0.2,
        "use_adaptive_adj": True,
        "adp_hidden_dim": 32,
        "adp_topk": 20,
        "adp_tau": 0.5,
        "use_hybrid_graph": True,
        "hybrid_alpha": 0.2,
        "use_patch_branch": True,
        "use_downsample_branch": True,
        "use_linear_residual_branch": True,
        "patch_embedding_mode": "serial_concat",
        "patch_data_input_mode": "all",
        "post_spatial_mode": "adaptive_only",
        "spatial_placement": "final",
        "use_pre_temporal_spatial_enhancement": False,
        "keep_output_prior_residual": False,
        "use_input_prior_enhancement": False,
        "use_graph_spectral_calibration": False,
        "use_extra_prior_input": False,
        "use_prev_condition": True,
        "chain_lengths": [3, 6, 12],
        "chain_loss_weights": [0.2, 0.3, 1.0],
    }


def default_model_args() -> dict:
    return {
        "node_size": 307,
        "input_len": 12,
        "output_len": 12,
        "input_dim": 4,
        "output_dim": 1,
        "patch_len": 3,
        "stride": 4,
        "td_size": 288,
        "dw_size": 7,
        "d_td": 32,
        "d_dw": 32,
        "d_d": 32,
        "d_spa": 32,
        "d_model": 0,
        "num_layer": 2,
        "states": list(DEFAULT_STATES),
        "spatial_scheme": "C",
        "adj_mx_path": str(ADJ_MX),
        "post_spatial_mode": "adaptive_only",
    }


def config_hash(payload: dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run_dir(seed: int, tag: str = "formal") -> Path:
    return ROOT / "results" / "forecast_trajectory_v2_run" / f"{tag}_seed{seed}"


def ckpt_dir(seed: int, tag: str = "formal") -> Path:
    return ROOT / "checkpoints" / "PEMS04" / "H12" / "forecast_trajectory_v2" / f"{tag}_seed{seed}"


def marker_path(rdir: Path, phase: str) -> Path:
    return rdir / "markers" / f"{phase}.done"


def write_marker(rdir: Path, phase: str, payload: dict) -> None:
    p = marker_path(rdir, phase)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))


def marker_ok(rdir: Path, phase: str, expected_hash: str) -> bool:
    p = marker_path(rdir, phase)
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False
    return str(d.get("config_hash", "")) == str(expected_hash)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_scaler():
    scaler = load_pkl(str(SCALER_FILE))
    fn = SCALER_REGISTRY.get(scaler["func"])

    def rescale(x: torch.Tensor) -> torch.Tensor:
        return fn(x, **scaler["args"])

    return scaler, rescale


def select_history(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, :, FORWARD_FEATURES]


def select_target(y: torch.Tensor) -> torch.Tensor:
    return y[:, :, :, TARGET_FEATURES]


class ForecastSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def collate_ft(batch):
    futures, histories, sis = zip(*batch)
    return torch.stack(futures, 0), torch.stack(histories, 0), torch.tensor(sis, dtype=torch.long)


def make_loader(split, indices, batch_size, shuffle, num_workers=0):
    ds = IndexedTimeSeriesForecastingDataset(str(DATA_FILE), str(INDEX_FILE), split)
    if indices is not None:
        ds = ForecastSubset(ds, indices)
        n = len(indices)
    else:
        n = len(ds)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate_ft, drop_last=False, pin_memory=False,
    )
    return loader, n


def finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def build_v2_model(device, warm_start: bool = True) -> tuple[ForecastTrajectoryV2Net, dict]:
    model = ForecastTrajectoryV2Net(**default_model_args()).to(device)
    report = {"warm_start": None, "capacity_warning": False}
    if warm_start and CANONICAL_CKPT.is_file():
        ck = torch.load(str(CANONICAL_CKPT), map_location="cpu", weights_only=False)
        report["warm_start"] = model.warm_start_from_canonical(ck["model_state_dict"])
        model = model.to(device)
    return model, report


def load_canonical_f2f(device) -> tuple[ChainForecasting, int, dict]:
    model = ChainForecasting(**canonical_f2f_args())
    ck = torch.load(str(CANONICAL_CKPT), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model = model.to(device).eval()
    n = count_params(model)
    meta = {"epoch": int(ck.get("epoch", -1)), "best_metrics": ck.get("best_metrics", {}), "params": n}
    print(f"[canonical F2F] params={n} epoch={meta['epoch']} ckpt_val_MAE={meta['best_metrics']}", flush=True)
    return model, n, meta


@torch.no_grad()
def evaluate_canonical_f2f(model, loader, device, rescale) -> dict:
    model.eval()
    maes, rmses, mapes = [], [], []
    n = 0
    any_nonfinite = False
    for future, history, _ in loader:
        x = select_history(history.to(device))
        y = select_target(future.to(device))
        out = model(history_data=x, return_all=True)
        pred = out["pred"] if isinstance(out, dict) else out
        if not finite_tensor(pred):
            any_nonfinite = True
        pr, tg = rescale(pred), rescale(y)
        maes.append(float(masked_mae(pr, tg, NULL_VAL).item()))
        rmses.append(float(masked_rmse(pr, tg, NULL_VAL).item()))
        mapes.append(float(masked_mape(pr, tg, NULL_VAL).item()))
        n += x.shape[0]
    return {
        "n_samples": n,
        "MAE": float(np.mean(maes)) if maes else float("nan"),
        "RMSE": float(np.mean(rmses)) if rmses else float("nan"),
        "MAPE": float(np.mean(mapes)) if mapes else float("nan"),
        "any_nonfinite": any_nonfinite,
        "trajectory": [3, 6, 12],
    }


def traj_mae(pred_h, target_h, rescale) -> dict:
    pr, tg = rescale(pred_h), rescale(target_h)
    return {
        "MAE": float(masked_mae(pr, tg, NULL_VAL).item()),
        "RMSE": float(masked_rmse(pr, tg, NULL_VAL).item()),
        "MAPE": float(masked_mape(pr, tg, NULL_VAL).item()),
    }


def trajectory_token_loss(zs: dict, y: torch.Tensor, rescale) -> torch.Tensor:
    preds, tgts = [], []
    for s, z in zs.items():
        tgt = build_resolution_target(y, int(s))
        preds.append(rescale(z))
        tgts.append(rescale(tgt))
    return token_normalized_mae(preds, tgts, null_val=NULL_VAL)


class ExposureBalancedSampler:
    def __init__(self, graph: ForecastTrajectoryGraph):
        self.graph = graph
        self.taus = graph.terminal_trajectories()
        self.canonical = CANONICAL_TAU
        self.counts = {e: 0 for e in graph.legal_edges()}
        self.incidence = {tau: graph.edges_of_trajectory(tau) for tau in self.taus}

    def sample_b(self) -> tuple[int, ...]:
        best, best_score = self.taus[0], -1e18
        for tau, edges in self.incidence.items():
            hyp = dict(self.counts)
            for e in edges:
                hyp[e] += 1
            vals = list(hyp.values())
            mn, mx = min(vals), max(vals)
            score = float(mn) * 1000.0 - (float(mx) / max(float(mn), 1.0))
            score += sum(1.0 / (self.counts[e] + 1.0) for e in edges)
            if score > best_score:
                best, best_score = tau, score
        return best

    def observe(self, tau: Sequence[int]) -> None:
        for e in self.graph.edges_of_trajectory(tau):
            self.counts[e] += 1

    def ratio(self) -> float:
        vals = list(self.counts.values())
        mn, mx = min(vals), max(vals)
        return float(mx) / float(mn) if mn > 0 else float("inf")

    def report(self) -> dict:
        vals = list(self.counts.values())
        mn, mx = min(vals), max(vals)
        return {
            "exposure": {f"{a}->{b}": int(c) for (a, b), c in self.counts.items()},
            "min": int(mn),
            "max": int(mx),
            "max_min_ratio": float(mx) / float(mn) if mn > 0 else float("inf"),
        }


@torch.no_grad()
def evaluate_trajectories(model, loader, trajectories, device, rescale, max_batches=None, stage_keys=None):
    model.eval()
    acc = {model.graph.trajectory_key(t): {"mae": [], "rmse": [], "mape": []} for t in trajectories}
    stage_acc = {k: [] for k in (stage_keys or [])}
    n = 0
    any_nonfinite = False
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        future, history, _ = batch
        history = select_history(history.to(device))
        y = select_target(future.to(device))
        hist = model.prepare_history(history)
        for tau in trajectories:
            out = model.rollout(history, tau, history=hist)
            z12 = out["Z"][model.graph.H]
            if not finite_tensor(z12):
                any_nonfinite = True
            m = traj_mae(z12, y, rescale)
            key = model.graph.trajectory_key(tau)
            acc[key]["mae"].append(m["MAE"])
            acc[key]["rmse"].append(m["RMSE"])
            acc[key]["mape"].append(m["MAPE"])
            if tuple(tau) == CANONICAL_TAU and stage_keys:
                for s in (3, 6, 12):
                    if s in out["Z"]:
                        tgt = build_resolution_target(y, s)
                        stage_acc[f"Z{s}"].append(traj_mae(out["Z"][s], tgt, rescale)["MAE"])
        n += history.shape[0]
    per = {}
    maes = []
    for tau in trajectories:
        key = model.graph.trajectory_key(tau)
        mae = float(np.mean(acc[key]["mae"])) if acc[key]["mae"] else float("nan")
        per[key] = {
            "MAE": mae,
            "RMSE": float(np.mean(acc[key]["rmse"])) if acc[key]["rmse"] else float("nan"),
            "MAPE": float(np.mean(acc[key]["mape"])) if acc[key]["mape"] else float("nan"),
            "trajectory": list(tau),
        }
        maes.append(mae)
    stages = {k: float(np.mean(v)) if v else float("nan") for k, v in stage_acc.items()}
    return {
        "n_samples": n,
        "per_trajectory": per,
        "mean_final_MAE": float(np.mean(maes)) if maes else float("nan"),
        "canonical_MAE": per.get("3-6-12", {}).get("MAE", float("nan")),
        "stage_MAE": stages,
        "any_nonfinite": any_nonfinite,
    }


def _dummy_batch(device, b=2, n=307):
    x = torch.randn(b, 12, n, 4, device=device)
    x[..., 1] = torch.rand(b, 12, n, device=device)
    x[..., 2] = torch.randint(0, 7, (b, 12, n), device=device).float()
    y = torch.randn(b, 12, n, 1, device=device)
    return x, y


def run_v2_unit_tests(model, device) -> dict:
    model.eval()
    g = model.graph
    g.assert_h12_defaults()
    x, y = _dummy_batch(device, 2, model.node_size)
    report = {"checks": {}, "fail": []}

    def check(name, cond, detail=""):
        report["checks"][name] = {"pass": bool(cond), "detail": detail}
        if not cond:
            report["fail"].append(f"{name}: {detail}")

    check("n_traj_16", len(g.terminal_trajectories()) == 16, str(len(g.terminal_trajectories())))
    check("n_edges_15", len(g.legal_edges()) == 15, str(len(g.legal_edges())))
    check("keep_2_3", g.is_legal_edge(2, 3), "")
    check("keep_3_4", g.is_legal_edge(3, 4), "")
    check("keep_4_6", g.is_legal_edge(4, 6), "")
    check("one_shared_core", not isinstance(getattr(model, "transitions", None), nn.ModuleDict), "")
    check("kasa_core", model.uses_kasa_temporal_core(), "")
    ids = model.transition_parameter_ids()
    check("shared_ids", len(ids) == len(set(ids)), f"n={len(ids)}")

    model.reset_history_encode_count()
    out = model.rollout(x, [2, 3, 4, 6, 12])
    check("hist_once", model.history_encode_count == 1, str(model.history_encode_count))
    check("H_propagated", all(s in out["H"] and out["H"][s] is not None for s in [2, 3, 4, 6, 12]), "")
    check("Z_propagated", all(s in out["Z"] and out["Z"][s] is not None for s in [2, 3, 4, 6, 12]), "")
    for s in [2, 3, 4, 6, 12]:
        check(f"Z{s}_shape", tuple(out["Z"][s].shape) == (2, s, model.node_size, 1), str(tuple(out["Z"][s].shape)))
        check(f"H{s}_shape", out["H"][s].shape[1] == s, str(tuple(out["H"][s].shape)))
        check(f"finite_Z{s}", finite_tensor(out["Z"][s]), "")

    hist = model.prepare_history(x)
    h3, z3, _ = model.transition(hist, None, None, 0, 3)
    h4, z4, aux34 = model.transition(hist, h3, z3, 3, 4)
    z_interp = aux34["z_interp"]
    diff = float((z4 - z_interp).abs().mean().item())
    check("3to4_not_interp_identity", diff > 1e-6, f"mean|z-interp|={diff} gate={aux34['gate_mean']}")
    check("3to4_gate_learned_path", aux34["gate_mean"] > 0.5, str(aux34["gate_mean"]))
    h6, z6, aux46 = model.transition(hist, h4, z4, 4, 6)
    check("4to6_finite", finite_tensor(z6), "")
    check("4to6_gate", aux46["gate_mean"] > 0.5, str(aux46["gate_mean"]))

    expected = expected_dag_transitions(g)
    _, _, n_trans = model.rollout_prefix_dag(x)
    check("dag_transition_count", n_trans == expected, f"{n_trans} vs unique prefixes {expected}")

    edge_ok = {}
    dense = model.rollout(x, g.dense_trajectory(), history=hist)
    zmap, hmap = {0: None}, {0: None}
    zmap.update(dense["Z"])
    hmap.update(dense["H"])
    for sp, sn in g.legal_edges():
        hp = None if sp == 0 else hmap.get(sp)
        zp = None if sp == 0 else zmap.get(sp)
        if sp != 0 and zp is None:
            hp, zp, _ = model.transition(hist, None, None, 0, sp)
            hmap[sp], zmap[sp] = hp, zp
        hn, zn, _ = model.transition(hist, hp, zp, sp, sn)
        edge_ok[f"{sp}->{sn}"] = bool(finite_tensor(zn) and zn.shape[1] == sn)
        hmap[sn], zmap[sn] = hn, zn
    check("all_15_edges", all(edge_ok.values()), json.dumps(edge_ok))
    report["edge_execution"] = edge_ok
    report["pass"] = len(report["fail"]) == 0
    return report


def policy_dead_param_test(policy, model, device) -> dict:
    policy.train()
    x, y = _dummy_batch(device, 2, model.node_size)
    hist, states, _ = model.rollout_prefix_dag(x)
    taus = model.graph.terminal_trajectories()
    rescale = lambda t: t
    y = y
    losses = []
    for tau in taus:
        z = states[tuple(tau)][1]
        losses.append(trajectory_token_loss({model.graph.H: z}, y, rescale).detach())
    L = torch.stack(losses, dim=0).mean().expand(x.shape[0], len(taus)).contiguous()
    # per-sample dummy losses
    L = torch.zeros(x.shape[0], len(taus), device=device)
    for ti, tau in enumerate(taus):
        z = states[tuple(tau)][1]
        L[:, ti] = (z - build_resolution_target(y, 12)).abs().mean(dim=(1, 2, 3))
    total_ms = torch.ones_like(L)
    lam = torch.zeros(x.shape[0], device=device)
    nb = torch.ones(x.shape[0], device=device)
    pol_states = {p: (h, z) for p, (h, z) in states.items()}
    policy.zero_grad(set_to_none=True)
    out = exact_prefix_dag_values(
        policy, model.graph, hist.pooled, pol_states, L, total_ms, lam, None, nb,
        {}, {}, 0.0, 1.0,
    )
    out["expected"].backward()
    dead = []
    for n, p in policy.named_parameters():
        if p.grad is None or float(p.grad.abs().sum().item()) == 0:
            dead.append(n)
    ok = len(dead) == 0
    return {"pass": ok, "dead": dead, "path_err": out["max_path_err"], "min_regret": out["min_regret"]}


def run_preflight(device) -> dict:
    g = ForecastTrajectoryGraph(H=12, states=DEFAULT_STATES)
    g.assert_h12_defaults()
    y = torch.randn(2, 12, 4, 1)
    assert_resolution_target_shapes(y, DEFAULT_STATES)
    model, warm = build_v2_model(device, warm_start=True)
    mrep = run_v2_unit_tests(model, device)
    policy = OnlineTrajectoryPolicyV2(model.graph, d_model=model.d_model).to(device)
    prep = policy_dead_param_test(policy, model, device)
    f2f, n_f2f, f2f_meta = load_canonical_f2f(device)
    br = model.param_breakdown()
    ratio = br["total"] / max(n_f2f, 1)
    cap_warn = ratio < 0.5
    if cap_warn:
        print("CAPACITY_COMPRESSION_WARNING", flush=True)
        print(
            "V2 shares one KASA TemporalStep-width trunk across all edges instead of "
            f"3 horizon-specific TemporalSteps. original={n_f2f} V2={br['total']} "
            f"ratio={ratio:.3f}. This is parameter sharing, not a tiny QKV replacement.",
            flush=True,
        )
    passed = mrep["pass"] and prep["pass"]
    return {
        "pass": passed,
        "model": mrep,
        "policy_dead": prep,
        "warm_start": warm,
        "f2f_params": n_f2f,
        "f2f_meta": f2f_meta,
        "v2_params": br,
        "capacity_ratio": ratio,
        "CAPACITY_COMPRESSION_WARNING": cap_warn,
        "total_param_count": br["total"],
        "model_obj": model,
        "f2f_obj": f2f,
    }


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn, device):
    if device.type == "cuda":
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize(device)
        return float(a.elapsed_time(b))
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def _summarize(xs):
    if not xs:
        return {"mean": None, "median": None, "p95": None, "n": 0}
    ys = sorted(xs)
    k = min(len(ys) - 1, max(0, int(round(0.95 * (len(ys) - 1)))))
    return {"mean": float(statistics.fmean(xs)), "median": float(statistics.median(xs)), "p95": float(ys[k]), "n": len(xs)}


@torch.no_grad()
def profile_latency(model, policy, x_one, device, warmup=20, iters=40) -> dict:
    model.eval()
    if policy is not None:
        policy.eval()
    x_one = x_one.to(device)
    for _ in range(warmup):
        model.prepare_history(x_one)
    _sync(device)
    hist_s = [_time_ms(lambda: model.prepare_history(x_one), device) for _ in range(iters)]
    history = model.prepare_history(x_one)
    dense = model.rollout(x_one, model.graph.dense_trajectory(), history=history)
    hmap, zmap = {0: None}, {0: None}
    hmap.update(dense["H"])
    zmap.update(dense["Z"])
    edges = {}
    for sp, sn in model.graph.legal_edges():
        hp = None if sp == 0 else hmap[sp]
        zp = None if sp == 0 else zmap[sp]

        def fn(sp=sp, sn=sn, hp=hp, zp=zp):
            model.transition(history, hp, zp, sp, sn)

        for _ in range(max(3, warmup // 4)):
            fn()
        edges[f"{sp}->{sn}"] = _summarize([_time_ms(fn, device) for _ in range(iters)])
    pol = None
    if policy is not None:
        h = history.pooled
        b = h.shape[0]
        lam = torch.zeros(b, device=device)
        rem = torch.ones(b, device=device)
        nb = torch.ones(b, device=device)

        def pfn():
            policy(h, None, None, 0, lam, rem, nb)

        for _ in range(warmup):
            pfn()
        pol = _summarize([_time_ms(pfn, device) for _ in range(iters)])
    t_hist = float(statistics.median(hist_s))
    t_pol = float(pol["median"]) if pol and pol["median"] is not None else 0.0
    edge_med = {k: float(v["median"]) for k, v in edges.items()}
    per_tau = {}
    for tau in model.graph.terminal_trajectories():
        s = 0
        trans = 0.0
        n_dec = 0
        for sn in tau:
            trans += edge_med[f"{s}->{sn}"]
            n_dec += 1
            s = sn
        tot = t_hist + trans + n_dec * t_pol
        per_tau[model.graph.trajectory_key(tau)] = {
            "transition_only_ms": trans,
            "total_ms": tot,
            "n_decisions": n_dec,
        }
    totals = [v["total_ms"] for v in per_tau.values()]
    return {
        "source": "cuda_events" if device.type == "cuda" else "cpu_perf_counter",
        "history": _summarize(hist_s),
        "policy": pol,
        "edges": edges,
        "per_trajectory": per_tau,
        "lookup": {
            "history_median_ms": t_hist,
            "policy_step_median_ms": t_pol,
            "edges_median_ms": edge_med,
            "total_ms_min": min(totals) if totals else None,
            "total_ms_max": max(totals) if totals else None,
        },
        "eta_used": False,
        "proxy_cost": False,
    }


def trajectory_total_ms(tau, lookup) -> dict:
    t_hist = float(lookup["history_median_ms"] or 0.0)
    t_pol = float(lookup.get("policy_step_median_ms") or 0.0)
    s = 0
    trans = 0.0
    n_dec = 0
    for sn in tau:
        trans += float(lookup["edges_median_ms"][f"{s}->{sn}"])
        n_dec += 1
        s = int(sn)
    return {"transition_only_ms": trans, "total_ms": t_hist + trans + n_dec * t_pol, "n_decisions": n_dec}


def train_transition(
    *,
    device,
    seed,
    epochs,
    batch_size,
    train_indices,
    valid_indices,
    test_indices,
    out_ckpt: Path,
    history_json: Path,
    phase: str,
    canonical_baseline_mae: Optional[float],
    acceptance: bool,
    eval_test: bool,
    max_epochs: Optional[int] = None,
    init_ckpt: Optional[Path] = None,
):
    seed_all(seed)
    _, rescale = load_scaler()
    model, warm = build_v2_model(device, warm_start=True)
    if init_ckpt is not None and Path(init_ckpt).is_file():
        ck = torch.load(str(init_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print(f"[V2] loaded init ckpt {init_ckpt} epoch={ck.get('epoch')}", flush=True)
    br = model.param_breakdown()
    opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1, 35, 60, 80, 95], gamma=0.5)
    train_loader, _ = make_loader("train", train_indices, batch_size, True)
    valid_loader, _ = make_loader("valid", valid_indices, batch_size, False)
    test_loader = None
    if eval_test:
        test_loader, _ = make_loader("test", test_indices, batch_size, False)
    sampler = ExposureBalancedSampler(model.graph)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_can = float("inf")
    best_epoch = 0
    rows = []
    last_loss = float("nan")
    last_grad = 0.0
    last_updated = False
    nan_inf = False
    containment_pass = False
    n_ep = int(epochs)
    panel = PRIMARY_PANEL if not acceptance else ACCEPTANCE_TRAJS
    all16 = model.graph.terminal_trajectories()

    for epoch in range(1, n_ep + 1):
        model.train()
        losses = []
        for future, history, _ in train_loader:
            history = select_history(history.to(device))
            y = select_target(future.to(device))
            hist = model.prepare_history(history)
            tau_a = CANONICAL_TAU
            if phase == "containment" or acceptance:
                # acceptance still exercises arbitrary 3->4 via sampled B every other batch
                tau_b = sampler.sample_b() if acceptance else CANONICAL_TAU
            else:
                tau_b = sampler.sample_b()
            opt.zero_grad(set_to_none=True)
            traj_losses = []
            for tau in (tau_a, tau_b):
                out = model.rollout(history, tau, history=hist)
                traj_losses.append(trajectory_token_loss(out["Z"], y, rescale))
                sampler.observe(tau)
            loss = 0.5 * traj_losses[0] + 0.5 * traj_losses[1]
            if not torch.isfinite(loss):
                nan_inf = True
                continue
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
            opt.step()
            last_updated = any(
                not torch.equal(p.detach(), b)
                for p, b in zip([p for p in model.parameters() if p.requires_grad], before)
            )
            last_grad = gn
            losses.append(float(loss.item()))
        sched.step()
        last_loss = float(np.mean(losses)) if losses else float("nan")
        use_all = (not acceptance) and (epoch % 5 == 0)
        ev_trajs = list(all16) if use_all else list(panel)
        val = evaluate_trajectories(
            model, valid_loader, ev_trajs, device, rescale, stage_keys=["Z3", "Z6", "Z12"]
        )
        can_mae = float(val.get("canonical_MAE") or val["per_trajectory"].get("3-6-12", {}).get("MAE", 1e9))
        mean_mae = float(val["mean_final_MAE"])
        test_m = None
        if eval_test and test_loader is not None:
            test_m = evaluate_trajectories(model, test_loader, panel, device, rescale)
        ratio = sampler.ratio()
        if phase == "curriculum" and epoch >= 3 and ratio > 2.5:
            print("EDGE_BALANCING_FAIL", flush=True)
            raise RuntimeError(f"EDGE_BALANCING_FAIL ratio={ratio}")
        valid_ckpt = True
        if canonical_baseline_mae is not None and not acceptance:
            valid_ckpt = can_mae <= float(canonical_baseline_mae) + CONTAINMENT_TOL
            if phase == "containment" and valid_ckpt:
                containment_pass = True
            if phase == "curriculum" and not valid_ckpt:
                print(
                    f"[reject ckpt] canonical MAE {can_mae:.4f} > baseline "
                    f"{canonical_baseline_mae:.4f}+{CONTAINMENT_TOL}",
                    flush=True,
                )
        score = mean_mae
        better = valid_ckpt and (
            score < best - 1e-8 or (abs(score - best) <= 1e-8 and can_mae < best_can)
        )
        if better:
            best, best_can, best_epoch = score, can_mae, epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "canonical_MAE": can_mae,
                    "mean_final_MAE": mean_mae,
                    "phase": phase,
                },
                out_ckpt,
            )
        row = {
            "epoch": epoch, "train_loss": last_loss, "valid": val, "test": test_m,
            "edge_ratio": ratio, "grad_norm": last_grad,
        }
        rows.append(row)
        print(
            f"[V2 {phase}] epoch={epoch} loss={last_loss:.4f} canonical_MAE={can_mae:.4f} "
            f"mean_MAE={mean_mae:.4f} edge_ratio={ratio:.3f}",
            flush=True,
        )
        if phase == "containment" and not acceptance and containment_pass:
            print("BACKBONE_CONTAINMENT_PASS", flush=True)
            break

    if out_ckpt.is_file():
        ck = torch.load(out_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        best_epoch = int(ck.get("epoch", best_epoch))
        best_can = float(ck.get("canonical_MAE", best_can))
    val_final = evaluate_trajectories(
        model, valid_loader, list(all16 if not acceptance else panel), device, rescale,
        stage_keys=["Z3", "Z6", "Z12"],
    )
    test_final = None
    if eval_test and test_loader is not None:
        test_final = evaluate_trajectories(model, test_loader, panel, device, rescale)
    if canonical_baseline_mae is not None and not acceptance:
        gap = float(val_final["canonical_MAE"]) - float(canonical_baseline_mae)
        containment_pass = gap <= CONTAINMENT_TOL
        if not containment_pass:
            print("BACKBONE_CONTAINMENT_FAIL", flush=True)
    summary = {
        "phase": phase,
        "best_epoch": best_epoch,
        "train_loss": last_loss,
        "grad_norm": last_grad,
        "optimizer_updated": last_updated,
        "nan_inf": nan_inf or val_final["any_nonfinite"],
        "valid": val_final,
        "test": test_final,
        "param_breakdown": br,
        "warm_start": warm,
        "edge_exposure": sampler.report(),
        "containment_pass": containment_pass,
        "canonical_baseline_mae": canonical_baseline_mae,
        "canonical_MAE": val_final.get("canonical_MAE"),
    }
    dump_json(history_json, {"rows": rows, "summary": summary})
    return {"model": model, "summary": summary}


@torch.no_grad()
def build_prefix_dag_cache(model, loader, device, rescale, latency, out_dir: Path) -> dict:
    model.eval()
    graph = model.graph
    expected = expected_dag_transitions(graph)
    writer = TrajectoryCacheV2Writer(out_dir, graph)
    lookup = latency["lookup"]
    n_bad = 0
    for future, history, sis in loader:
        history = select_history(history.to(device))
        y = select_target(future.to(device))
        hist, states, n_trans = model.rollout_prefix_dag(history)
        if int(n_trans) != int(expected):
            n_bad += 1
        b = history.shape[0]
        for i in range(b):
            prefix_h, prefix_z = {}, {}
            for pref, (h, z) in states.items():
                if h is None:
                    prefix_h[pref] = None
                else:
                    prefix_h[pref] = h[i].mean(dim=0)  # [N,D] time pool
                prefix_z[pref] = None if z is None else z[i]
            metrics = {}
            for tau in graph.terminal_trajectories():
                z12 = states[tuple(tau)][1][i : i + 1]
                m = traj_mae(z12, y[i : i + 1], rescale)
                c = trajectory_total_ms(tau, lookup)
                key = graph.trajectory_key(tau)
                metrics[key] = {**m, **c, "cost_ms": c["total_ms"], "cost_norm": c["total_ms"]}
            writer.add(
                int(sis[i].item()),
                hist.pooled[i],
                prefix_h,
                prefix_z,
                metrics,
                int(n_trans),
            )
    man = writer.close({"expected_transitions": expected, "n_mismatch": n_bad})
    print(f"[V2 cache] samples={man['n_samples']} mean_transitions={man['mean_transitions']} expected={expected}", flush=True)
    return man


def oracle_analysis(cache: TrajectoryCacheV2, graph: ForecastTrajectoryGraph) -> dict:
    keys = [graph.trajectory_key(t) for t in graph.terminal_trajectories()]
    L = []
    for si in cache.sample_indices():
        rec = cache.get(si)
        L.append([float(rec["traj_metrics"][k]["MAE"]) for k in keys])
    L = np.asarray(L, dtype=np.float64)
    mean_tau = L.mean(axis=0)
    best_i = int(np.argmin(mean_tau))
    L_fixed = float(mean_tau[best_i])
    L_oracle = float(L.min(axis=1).mean())
    delta = L_fixed - L_oracle
    # pairwise output / MAE variance
    var = float(L.var(axis=1).mean())
    pair = float(np.mean(np.abs(L[:, :, None] - L[:, None, :])))
    hist = Counter(keys[int(i)] for i in L.argmin(axis=1))
    return {
        "best_fixed_trajectory": keys[best_i],
        "best_fixed_MAE": L_fixed,
        "sample_wise_oracle_MAE": L_oracle,
        "Delta_adaptive": float(delta),
        "trajectory_mae_variance": var,
        "pairwise_mae_diff": pair,
        "mean_mae_per_trajectory": {k: float(v) for k, v in zip(keys, mean_tau)},
        "sample_optimal_histogram": dict(hist),
        "lambda_scale": {"lambda_max": float(max(L_fixed, 1.0) / 50.0)},
        "n": int(L.shape[0]),
    }


def _states_from_rec(rec, device):
    states = {}
    for k, h in rec["prefix_h"].items():
        pref = parse_prefix_key(k)
        ht = None if h is None else h.float().to(device)
        if ht is not None and ht.ndim == 2:
            ht = ht.unsqueeze(0).unsqueeze(0)  # [1,1,N,D]
        z = rec["prefix_z"].get(k)
        zt = None if z is None else z.float().to(device)
        if zt is not None and zt.ndim == 3:
            zt = zt.unsqueeze(0)
        states[pref] = (ht, zt)
    return states


def stack_prefix_states(recs, device, graph):
    """Stack cached prefixes to batch [B,...] for one policy eval per unique prefix."""
    prefs = []
    seen = set()
    for tau in graph.terminal_trajectories():
        for i in range(len(tau) + 1):
            p = tuple(tau[:i])
            if p not in seen:
                seen.add(p)
                prefs.append(p)
    b = len(recs)
    hx = torch.stack([r["history_summary"].float() for r in recs], 0).to(device)
    batched = {}
    for pref in prefs:
        key = prefix_key(pref)
        hs, zs = [], []
        has_h = has_z = True
        for r in recs:
            h = r["prefix_h"].get(key)
            z = r["prefix_z"].get(key)
            if h is None:
                has_h = False
            else:
                hs.append(h.float())
            if z is None:
                has_z = False
            else:
                zs.append(z.float())
        ht = torch.stack(hs, 0).to(device).unsqueeze(1) if has_h and hs else None  # [B,1,N,D]
        zt = torch.stack(zs, 0).to(device) if has_z and zs else None
        batched[pref] = (ht, zt)
    return hx, batched


def policy_vectorized_loss(policy, graph, recs, device, lam, remaining_init, no_budget, edge_cost, min_finish, extra, c_dense):
    keys = [graph.trajectory_key(t) for t in graph.terminal_trajectories()]
    b = len(recs)
    n_tau = len(keys)
    L = torch.zeros(b, n_tau, device=device)
    T = torch.zeros(b, n_tau, device=device)
    for i, r in enumerate(recs):
        for j, k in enumerate(keys):
            L[i, j] = float(r["traj_metrics"][k]["MAE"])
            T[i, j] = float(r["traj_metrics"][k]["total_ms"])
    hx, states = stack_prefix_states(recs, device, graph)
    return exact_prefix_dag_values(
        policy, graph, hx, states, L, T, lam, remaining_init, no_budget,
        edge_cost, min_finish, extra, c_dense,
    )


@torch.no_grad()
def eval_policy_panel(policy, graph, cache, indices, device, lambdas, B_list, edge_cost, min_finish, extra, c_dense, c_hist, batch_size=32):
    policy.eval()
    t0 = time.perf_counter()
    rows = []
    regrets = []
    lats = []
    path_err = 0.0
    min_reg = 0.0
    ids = list(indices)
    for lam in lambdas:
        for B in B_list:
            batch_reg, batch_lat = [], []
            for i in range(0, len(ids), batch_size):
                recs = [cache.get(j) for j in ids[i : i + batch_size]]
                bsz = len(recs)
                lam_t = torch.full((bsz,), float(lam), device=device)
                if B is None:
                    nb = torch.ones(bsz, device=device)
                    rem = None
                else:
                    nb = torch.zeros(bsz, device=device)
                    rem = torch.full((bsz,), max(float(B) - float(c_hist), 0.0), device=device)
                out = policy_vectorized_loss(
                    policy, graph, recs, device, lam_t, rem, nb, edge_cost, min_finish, extra, c_dense
                )
                path_err = max(path_err, out["max_path_err"])
                min_reg = min(min_reg, out["min_regret"])
                batch_reg.append(float(out["regret"].mean().item()))
                pp = out["path_probs"]
                keys = [graph.trajectory_key(t) for t in graph.terminal_trajectories()]
                T = torch.tensor(
                    [[float(r["traj_metrics"][k]["total_ms"]) for k in keys] for r in recs],
                    device=device,
                )
                batch_lat.append(float((pp * T).sum(-1).mean().item()))
            mean_reg = float(np.mean(batch_reg)) if batch_reg else float("nan")
            mean_lat = float(np.mean(batch_lat)) if batch_lat else float("nan")
            regrets.append(mean_reg)
            lats.append(mean_lat)
            rows.append({"lambda": lam, "B_ms": B, "mean_regret": mean_reg, "mean_latency_ms": mean_lat})
    dt = time.perf_counter() - t0
    return {
        "primary_mean_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "panel": rows,
        "path_prob_max_error": path_err,
        "min_regret": min_reg,
        "wall_s": dt,
        "vectorized": True,
    }


def train_policy(
    *,
    model,
    cache,
    split,
    device,
    latency,
    oracle,
    out_ckpt: Path,
    history_json: Path,
    acceptance=False,
    min_epochs=20,
    max_epochs=80,
    patience=15,
    batch_size=16,
):
    graph = model.graph
    policy = OnlineTrajectoryPolicyV2(graph, d_model=model.d_model).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4, weight_decay=1e-4)
    lambda_max = float(oracle["lambda_scale"]["lambda_max"])
    lookup = latency["lookup"]
    edge_cost = {}
    for k, v in lookup["edges_median_ms"].items():
        a, b = k.split("->")
        edge_cost[(int(a), int(b))] = float(v)
    extra = float(lookup.get("policy_step_median_ms") or 0.0)
    min_finish = graph.min_finish_edge_cost(edge_cost, extra_per_edge=extra)
    c_hist = float(lookup["history_median_ms"] or 0.0)
    totals = [float(v["total_ms"]) for v in latency["per_trajectory"].values()]
    c_dense = max(totals) if totals else 1.0
    unique_B = sorted({float(v["total_ms"]) for v in latency["per_trajectory"].values()})
    B_low, B_mid, B_high = unique_B[max(0, len(unique_B)//5)], unique_B[len(unique_B)//2], unique_B[min(len(unique_B)-1, 4*len(unique_B)//5)]
    val_lams = [0.0, 0.5 * lambda_max, lambda_max]
    val_Bs = [None, B_low, B_mid, B_high]
    train_ids = split.get("policy_train") or cache.sample_indices()
    valid_ids = split.get("policy_valid") or train_ids[: max(1, len(train_ids) // 5)]
    best = float("inf")
    best_lat = 1e18
    best_epoch = 0
    hist = []
    epochs = 1 if acceptance else max_epochs
    path_err = 0.0
    min_reg = 0.0
    t_epoch = []
    invalid = False

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        policy.train()
        losses = []
        ids = list(train_ids)
        random.shuffle(ids)
        t_io = t_fwd = t_dp = 0.0
        for i in range(0, len(ids), batch_size):
            chunk = ids[i : i + batch_size]
            t1 = time.perf_counter()
            recs = [cache.get(j) for j in chunk]
            t_io += time.perf_counter() - t1
            bsz = len(recs)
            lam = torch.empty(bsz, device=device).uniform_(0.0, lambda_max)
            nb = torch.zeros(bsz, device=device)
            rem = torch.zeros(bsz, device=device)
            for j in range(bsz):
                if random.random() < 0.5:
                    nb[j] = 1.0
                    rem[j] = max(c_dense - c_hist, 0.0)
                else:
                    nb[j] = 0.0
                    rem[j] = max(float(random.choice(unique_B)) - c_hist, 0.0)
            opt.zero_grad(set_to_none=True)
            t2 = time.perf_counter()
            out = policy_vectorized_loss(
                policy, graph, recs, device, lam, rem, nb, edge_cost, min_finish, extra, c_dense
            )
            t_dp += time.perf_counter() - t2
            path_err = max(path_err, out["max_path_err"])
            min_reg = min(min_reg, out["min_regret"])
            if out["min_regret"] < -1e-6:
                print("POLICY_OBJECTIVE_INVALID", flush=True)
                invalid = True
                break
            loss = out["expected"]
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        if invalid:
            break
        t_val0 = time.perf_counter()
        val = eval_policy_panel(
            policy, graph, cache, valid_ids, device, val_lams, val_Bs,
            edge_cost, min_finish, extra, c_dense, c_hist, batch_size=batch_size,
        )
        t_val = time.perf_counter() - t_val0
        epoch_s = time.perf_counter() - t0
        t_epoch.append(epoch_s)
        primary = val["primary_mean_regret"]
        mean_lat = float(np.mean([r["mean_latency_ms"] for r in val["panel"]]))
        print(
            f"[V2 policy] epoch={epoch} loss={np.mean(losses) if losses else float('nan'):.5f} "
            f"regret={primary:.5f} wall={epoch_s:.1f}s io={t_io:.2f}s dp={t_dp:.2f}s val={t_val:.2f}s",
            flush=True,
        )
        hist.append({
            "epoch": epoch, "train_loss": float(np.mean(losses) if losses else float("nan")),
            "valid": val, "timing": {"epoch_s": epoch_s, "cache_io_s": t_io, "dp_s": t_dp, "validation_s": t_val},
        })
        if primary < best - 1e-8 or (abs(primary - best) <= 1e-8 and mean_lat < best_lat):
            best, best_lat, best_epoch = primary, mean_lat, epoch
            torch.save({"epoch": epoch, "state_dict": policy.state_dict(), "primary_regret": primary}, out_ckpt)
        if (not acceptance) and epoch >= min_epochs and epoch - best_epoch >= patience:
            break

    if out_ckpt.is_file():
        ck = torch.load(out_ckpt, map_location=device, weights_only=False)
        policy.load_state_dict(ck["state_dict"])
        best_epoch = int(ck.get("epoch", best_epoch))
        best = float(ck.get("primary_regret", best))
    val_final = eval_policy_panel(
        policy, graph, cache, valid_ids, device, val_lams, val_Bs,
        edge_cost, min_finish, extra, c_dense, c_hist, batch_size=batch_size,
    )
    if val_final["min_regret"] < -1e-6 or path_err > PATH_PROB_ATOL:
        print("POLICY_OBJECTIVE_INVALID", flush=True)
        invalid = True
    summary = {
        "best_epoch": best_epoch,
        "best_valid_regret": best,
        "param_count": policy.parameter_count(),
        "path_prob_max_error": max(path_err, val_final["path_prob_max_error"]),
        "min_regret": min(min_reg, val_final["min_regret"]),
        "policy_epoch_wall_s": float(np.mean(t_epoch)) if t_epoch else None,
        "valid": val_final,
        "lambda_max": lambda_max,
        "val_lambdas": val_lams,
        "val_Bs": val_Bs,
        "invalid": invalid,
        "vectorized": True,
    }
    dump_json(history_json, {"rows": hist, "summary": summary})
    return {"policy": policy, "summary": summary}


@torch.no_grad()
def run_online_policy(model, policy, loader, device, rescale, latency, lam, B_ms):
    model.eval()
    policy.eval()
    graph = model.graph
    lookup = latency["lookup"]
    edge_cost = {}
    for k, v in lookup["edges_median_ms"].items():
        a, b = k.split("->")
        edge_cost[(int(a), int(b))] = float(v)
    extra = float(lookup.get("policy_step_median_ms") or 0.0)
    min_finish = graph.min_finish_edge_cost(edge_cost, extra_per_edge=extra)
    c_hist = float(lookup["history_median_ms"] or 0.0)
    c_dense = float(lookup.get("total_ms_max") or 1.0)
    maes, lats, n_trans = [], [], []
    tau_hist = Counter()
    any_nonfinite = False
    n = 0
    for future, history, _ in loader:
        history = select_history(history.to(device))
        y = select_target(future.to(device))
        for i in range(history.shape[0]):
            xi, yi = history[i : i + 1], y[i : i + 1]
            hist = model.prepare_history(xi)
            s, h_cur, z = 0, None, None
            chosen = []
            remaining = None if B_ms is None else (float(B_ms) - c_hist)
            nb = torch.ones(1, device=device) if B_ms is None else torch.zeros(1, device=device)
            while s != graph.H:
                rem_norm = torch.ones(1, device=device)
                feas = None
                if remaining is not None:
                    rem_norm = torch.tensor([remaining / max(c_dense, 1e-6)], device=device)
                    feas_list = [
                        cand for cand in graph.successors(s)
                        if graph.edge_feasible(s, cand, remaining, edge_cost, min_finish, extra)
                    ]
                    if not feas_list:
                        feas_list = graph.successors(s)
                    feas = torch.zeros(1, policy.n_dest, dtype=torch.bool, device=device)
                    for cand in feas_list:
                        feas[0, policy.state_to_index[int(cand)]] = True
                out = policy(
                    hist.pooled, h_cur, z, s,
                    torch.tensor([lam], device=device), rem_norm, nb, feasible_next=feas,
                )
                s_next = int(policy.dest_states[int(out["probs"][0].argmax().item())])
                if not graph.is_legal_edge(s, s_next):
                    s_next = graph.successors(s)[-1]
                h_cur, z, _ = model.transition(hist, h_cur, z, s, s_next)
                if not finite_tensor(z):
                    any_nonfinite = True
                chosen.append(s_next)
                if remaining is not None:
                    remaining -= float(edge_cost[(s, s_next)]) + extra
                s = s_next
            m = traj_mae(z, yi, rescale)
            maes.append(m["MAE"])
            c = trajectory_total_ms(chosen, lookup)
            lats.append(c["total_ms"])
            n_trans.append(len(chosen))
            tau_hist[graph.trajectory_key(chosen)] += 1
            n += 1
    return {
        "n": n,
        "MAE": float(np.mean(maes)) if maes else float("nan"),
        "avg_latency_ms": float(np.mean(lats)) if lats else float("nan"),
        "avg_n_transitions": float(np.mean(n_trans)) if n_trans else float("nan"),
        "trajectory_histogram": dict(tau_hist),
        "lambda": lam,
        "B_ms": B_ms,
        "any_nonfinite": any_nonfinite,
    }


def chronological_policy_split(sample_indices, train_index, ratio=0.8):
    sis = sorted(int(i) for i in sample_indices)
    if len(sis) <= 1:
        return {"policy_train": sis, "policy_valid": sis, "purged": []}
    cut = min(max(int(math.floor(len(sis) * ratio)), 1), len(sis) - 1)
    left, right = sis[:cut], sis[cut:]
    spans = []
    for idx in train_index:
        spans.append((int(idx[0]), int(idx[2])))
    right_spans = [spans[i] for i in right]
    purged, kept = [], []
    for i in left:
        a, c = spans[i]
        if any(not (c <= ra or rc <= a) for ra, rc in right_spans):
            purged.append(i)
        else:
            kept.append(i)
    return {"policy_train": kept, "policy_valid": right, "purged": purged}


def print_terminal_summary(bundle: dict) -> None:
    print("==== ForecastTrajectory V2 TERMINAL SUMMARY ====", flush=True)
    print(f"canonical original F2F VALID MAE: {bundle.get('canonical_f2f_valid_mae')}", flush=True)
    print(f"V2 canonical [3,6,12] VALID MAE: {bundle.get('v2_canonical_valid_mae')}", flush=True)
    print(f"containment gap: {bundle.get('containment_gap')}", flush=True)
    print(f"BACKBONE_CONTAINMENT_PASS/FAIL: {bundle.get('containment_verdict')}", flush=True)
    print(f"V2 params: {bundle.get('v2_params')}", flush=True)
    print(f"shared transition params: {bundle.get('shared_transition_params')}", flush=True)
    print(f"state adapter params: {bundle.get('state_adapter_params')}", flush=True)
    print(f"all legal edges: {bundle.get('n_edges')}", flush=True)
    print(f"all terminal trajectories: {bundle.get('n_traj')}", flush=True)
    print(f"edge exposure ratio: {bundle.get('edge_exposure_ratio')}", flush=True)
    print(f"VALID best fixed MAE: {bundle.get('valid_best_fixed_mae')}", flush=True)
    print(f"VALID sample oracle MAE: {bundle.get('valid_oracle_mae')}", flush=True)
    print(f"VALID adaptive headroom: {bundle.get('valid_adaptive_headroom')}", flush=True)
    print(f"history latency: {bundle.get('history_latency')}", flush=True)
    print(f"policy latency: {bundle.get('policy_latency')}", flush=True)
    print(f"total latency range: {bundle.get('total_latency_range')}", flush=True)
    print(f"policy epoch wall time: {bundle.get('policy_epoch_wall')}", flush=True)
    print(f"path probability max error: {bundle.get('path_prob_max_error')}", flush=True)
    print(f"minimum observed regret: {bundle.get('min_regret')}", flush=True)
    print(f"joint fine-tuning retained?: {bundle.get('joint_retained')}", flush=True)
    print(f"final VALID: {bundle.get('final_valid')}", flush=True)
    print(f"final TEST: {bundle.get('final_test')}", flush=True)
    print("No eta: MUST BE YES", flush=True)
    print("No old proxy cost: MUST BE YES", flush=True)
