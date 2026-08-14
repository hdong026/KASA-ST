"""Shared runtime for ForecastTrajectory: data, tests, train, cache, policy, eval."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
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

from basicts.data import SCALER_REGISTRY
from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.utils import load_pkl
from basicts.archs.arch_zoo.ForecastTrajectory_arch.forecast_trajectory_net import (
    ForecastTrajectoryNet,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.online_trajectory_policy import (
    OnlineTrajectoryPolicy,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.target_resolution import (
    assert_resolution_target_shapes,
    build_resolution_target,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_cache import (
    TrajectoryCache,
    TrajectoryCacheWriter,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_graph import (
    ForecastTrajectoryGraph,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_latency import (
    profile_transition_latency,
    trajectory_cost_ms,
)
from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_objective import (
    PATH_PROB_ATOL,
    exact_policy_loss,
    exact_trajectory_probs,
    token_normalized_mae,
)

DATA_DIR = ROOT / "datasets" / "PEMS04"
DATA_FILE = DATA_DIR / "data_in12_out12.pkl"
INDEX_FILE = DATA_DIR / "index_in12_out12.pkl"
SCALER_FILE = DATA_DIR / "scaler_in12_out12.pkl"
ADJ_MX = DATA_DIR / "adj_mx.pkl"

DEFAULT_STATES = [2, 3, 4, 6, 12]
H_DEFAULT = 12
VALIDATION_PANEL = [(12,), (2, 12), (3, 12), (6, 12), (2, 3, 4, 6, 12)]
ACCEPTANCE_TRAJECTORIES = [(12,), (3, 12), (6, 12), (2, 3, 4, 6, 12)]

NULL_VAL = 0.0
FORWARD_FEATURES = [0, 1, 2, 3]
TARGET_FEATURES = [0]


def git_head() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        "d_model": 64,
        "cond_dim": 64,
        "num_layer": 2,
        "n_heads": 4,
        "if_time_in_day": True,
        "if_day_in_week": True,
        "if_spatial": True,
        "states": list(DEFAULT_STATES),
        "spatial_scheme": "C",
        "adj_mx_path": str(ADJ_MX),
        "post_spatial_mode": "adaptive_only",
        "adp_hidden_dim": 32,
        "adp_topk": 20,
        "adp_tau": 0.5,
        "adp_alpha": 0.1,
        "hybrid_alpha": 0.2,
        "use_hybrid_graph": True,
    }


def config_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def run_dir(seed: int, tag: str = "formal") -> Path:
    return ROOT / "results" / "forecast_trajectory_run" / f"{tag}_seed{seed}"


def ckpt_dir(seed: int, tag: str = "formal") -> Path:
    return ROOT / "checkpoints" / "PEMS04" / "H12" / "forecast_trajectory" / f"{tag}_seed{seed}"


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


def load_scaler():
    scaler = load_pkl(str(SCALER_FILE))
    fn = SCALER_REGISTRY.get(scaler["func"])

    def rescale(x: torch.Tensor) -> torch.Tensor:
        return fn(x, **scaler["args"])

    return scaler, rescale


def build_model(device: torch.device) -> ForecastTrajectoryNet:
    model = ForecastTrajectoryNet(**default_model_args())
    return model.to(device)


def select_history(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, :, FORWARD_FEATURES]


def select_target(y: torch.Tensor) -> torch.Tensor:
    return y[:, :, :, TARGET_FEATURES]


class ForecastSubset(torch.utils.data.Dataset):
    def __init__(self, base: IndexedTimeSeriesForecastingDataset, indices: list[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def collate_ft(batch):
    futures, histories, sis = zip(*batch)
    return (
        torch.stack(futures, dim=0),
        torch.stack(histories, dim=0),
        torch.tensor(sis, dtype=torch.long),
    )


def make_loader(
    split: str,
    indices: Optional[list[int]],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> tuple[DataLoader, int]:
    ds = IndexedTimeSeriesForecastingDataset(str(DATA_FILE), str(INDEX_FILE), split)
    if indices is not None:
        ds = ForecastSubset(ds, indices)
        n = len(indices)
    else:
        n = len(ds)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_ft,
        drop_last=False,
        pin_memory=False,
    )
    return loader, n


def split_raw_windows(index_list) -> list[tuple[int, int]]:
    spans = []
    for idx in index_list:
        a, b, c = int(idx[0]), int(idx[1]), int(idx[2])
        spans.append((a, c))
    return spans


def chronological_policy_split(
    sample_indices: list[int],
    train_index,
    ratio: float = 0.8,
) -> dict:
    """First 80% / last 20% of TRAIN cache indices with raw-window purge."""
    sis = sorted(int(i) for i in sample_indices)
    if not sis:
        return {"policy_train": [], "policy_valid": [], "purged": []}
    cut = int(math.floor(len(sis) * ratio))
    cut = min(max(cut, 1), len(sis) - 1) if len(sis) > 1 else len(sis)
    left = sis[:cut]
    right = sis[cut:]
    spans = split_raw_windows(train_index)
    right_spans = [spans[i] for i in right]
    purged = []
    kept_left = []
    for i in left:
        a, c = spans[i]
        overlap = False
        for ra, rc in right_spans:
            if not (c <= ra or rc <= a):
                overlap = True
                break
        if overlap:
            purged.append(i)
        else:
            kept_left.append(i)
    return {
        "policy_train": kept_left,
        "policy_valid": right,
        "purged": purged,
        "cut": cut,
        "n_before_purge": len(left),
        "n_after_purge": len(kept_left),
    }


def finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def run_graph_unit_tests() -> dict:
    g = ForecastTrajectoryGraph(H=12, states=[2, 3, 4, 6, 12])
    g.assert_h12_defaults()
    return {
        "n_trajectories": len(g.terminal_trajectories()),
        "n_edges": len(g.legal_edges()),
        "nodes": list(g.nodes),
        "pass": True,
    }


def run_target_unit_tests() -> dict:
    y = torch.arange(12, dtype=torch.float32).view(1, 12, 1, 1).expand(2, 12, 4, 1).clone()
    y = y + torch.randn_like(y) * 0.01
    assert_resolution_target_shapes(y, [2, 3, 4, 6, 12])
    y12 = build_resolution_target(y, 12)
    ok = torch.allclose(y12, y, atol=1e-6, rtol=1e-6)
    shapes = {s: list(build_resolution_target(y, s).shape) for s in [2, 3, 4, 6, 12]}
    return {"shapes": shapes, "s12_equals_y": bool(ok), "pass": bool(ok)}


@torch.no_grad()
def _dummy_batch(device, b=2, n=307):
    x = torch.randn(b, 12, n, 4, device=device)
    x[..., 1] = torch.rand(b, 12, n, device=device)
    x[..., 2] = torch.randint(0, 7, (b, 12, n), device=device).float()
    y = torch.randn(b, 12, n, 1, device=device)
    return x, y


def run_model_unit_tests(model: ForecastTrajectoryNet, device: torch.device) -> dict:
    model.eval()
    g = model.graph
    g.assert_h12_defaults()
    x, y = _dummy_batch(device, b=2, n=model.node_size)
    report = {"checks": {}, "fail": []}

    def check(name, cond, detail=""):
        report["checks"][name] = {"pass": bool(cond), "detail": detail}
        if not cond:
            report["fail"].append(f"{name}: {detail}")

    check("n_trajectories_16", len(g.terminal_trajectories()) == 16, str(len(g.terminal_trajectories())))
    check("n_edges_15", len(g.legal_edges()) == 15, str(len(g.legal_edges())))
    ids = model.transition_parameter_ids()
    check("shared_transition_param_ids", len(ids) == len(set(ids)), f"n={len(ids)} unique={len(set(ids))}")
    check("one_transition_core", True, f"count={model.transition_parameter_count()}")

    model.reset_history_encode_count()
    out = model.rollout(x, [2, 3, 4, 6, 12])
    check("history_encoder_once_per_rollout", model.history_encode_count == 1, str(model.history_encode_count))
    for s, z in out.items():
        check(f"shape_s{s}", tuple(z.shape) == (2, int(s), model.node_size, 1), str(tuple(z.shape)))
        check(f"finite_s{s}", finite_tensor(z), "")
    check("s12_shape", tuple(out[12].shape) == (2, 12, model.node_size, 1), str(tuple(out[12].shape)))

    for tau in ACCEPTANCE_TRAJECTORIES:
        model.reset_history_encode_count()
        r = model.rollout(x, tau)
        check(f"run_{g.trajectory_key(tau)}", 12 in r and finite_tensor(r[12]), "")
        check(f"hist_once_{g.trajectory_key(tau)}", model.history_encode_count == 1, str(model.history_encode_count))

    # all 15 edges execute
    hist = model.prepare_history(x)
    z_prev_map = {0: None}
    # get some Z via dense
    dense = model.rollout(x, g.dense_trajectory(), history=hist)
    z_prev_map.update(dense)
    edge_ok = {}
    for sp, sn in g.legal_edges():
        zp = None if sp == 0 else z_prev_map.get(sp)
        if sp != 0 and zp is None:
            zp = model.transition(hist, None, 0, sp)
            z_prev_map[sp] = zp
        z = model.transition(hist, zp, sp, sn)
        edge_ok[f"{sp}->{sn}"] = bool(finite_tensor(z) and z.shape[1] == sn)
    check("all_15_edges_execute", all(edge_ok.values()), json.dumps(edge_ok))
    report["edge_execution"] = edge_ok
    report["pass"] = len(report["fail"]) == 0
    return report


def run_edge_gradient_coverage_test(model: ForecastTrajectoryNet, device: torch.device) -> dict:
    model.train()
    x, y = _dummy_batch(device, b=2, n=model.node_size)
    y_full = y
    covered = {}
    for sp, sn in model.graph.legal_edges():
        model.zero_grad(set_to_none=True)
        hist = model.prepare_history(x)
        zp = None
        if sp > 0:
            zp = model.transition(hist, None, 0, sp)
        z = model.transition(hist, zp, sp, sn)
        tgt = build_resolution_target(y_full, sn)
        loss = (z - tgt).abs().mean()
        loss.backward()
        n_nz = 0
        n_p = 0
        for p in model.transition_core.parameters():
            n_p += 1
            if p.grad is not None and float(p.grad.abs().sum().item()) > 0:
                n_nz += 1
        covered[f"{sp}->{sn}"] = {"nonzero_grad_tensors": n_nz, "n_tensors": n_p, "loss": float(loss.item())}
    ok = all(v["nonzero_grad_tensors"] > 0 for v in covered.values())
    return {"edges": covered, "pass": ok}


def run_preflight_unit_tests(device: torch.device) -> dict:
    graph = run_graph_unit_tests()
    targets = run_target_unit_tests()
    model = build_model(device)
    model_rep = run_model_unit_tests(model, device)
    grad_rep = run_edge_gradient_coverage_test(model, device)
    passed = (
        graph["pass"]
        and targets["pass"]
        and model_rep["pass"]
        and grad_rep["pass"]
    )
    return {
        "graph": graph,
        "targets": targets,
        "model": model_rep,
        "edge_gradients": grad_rep,
        "transition_param_count": model.transition_parameter_count(),
        "total_param_count": count_params(model),
        "pass": passed,
    }


# ---------------------------------------------------------------------------
# Trajectory sampling
# ---------------------------------------------------------------------------


class EdgeBalancedSampler:
    def __init__(self, graph: ForecastTrajectoryGraph):
        self.graph = graph
        self.taus = graph.terminal_trajectories()
        self.counts = {e: 0 for e in graph.legal_edges()}
        self._direct = graph.direct_trajectory()
        self._dense = graph.dense_trajectory()
        self._a_toggle = 0

    def sample_a(self) -> tuple[int, ...]:
        tau = self._direct if (self._a_toggle % 2 == 0) else self._dense
        self._a_toggle += 1
        return tau

    def sample_b(self) -> tuple[int, ...]:
        weights = []
        for tau in self.taus:
            edges = self.graph.edges_of_trajectory(tau)
            score = sum(1.0 / (self.counts[e] + 1.0) for e in edges)
            weights.append(score)
        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        chosen = self.taus[-1]
        for tau, w in zip(self.taus, weights):
            acc += w
            if r <= acc:
                chosen = tau
                break
        return chosen

    def observe(self, tau: Sequence[int]) -> None:
        for e in self.graph.edges_of_trajectory(tau):
            self.counts[e] += 1

    def report(self) -> dict:
        vals = list(self.counts.values())
        mn = min(vals) if vals else 0
        mx = max(vals) if vals else 0
        ratio = (float(mx) / float(mn)) if mn > 0 else float("inf")
        return {
            "exposure": {f"{a}->{b}": int(c) for (a, b), c in self.counts.items()},
            "min": int(mn),
            "max": int(mx),
            "max_min_ratio": float(ratio),
        }


# ---------------------------------------------------------------------------
# Metrics / loss
# ---------------------------------------------------------------------------


def rescale_pair(pred, target, rescale):
    return rescale(pred), rescale(target)


def raw_panel_metrics(pred_h, target_h, rescale, null_val=NULL_VAL) -> dict:
    pr, tg = rescale_pair(pred_h, target_h, rescale)
    return {
        "MAE": float(masked_mae(pr, tg, null_val).item()),
        "RMSE": float(masked_rmse(pr, tg, null_val).item()),
        "MAPE": float(masked_mape(pr, tg, null_val).item()),
    }


def transition_batch_loss(
    model: ForecastTrajectoryNet,
    history_x: torch.Tensor,
    future_y: torch.Tensor,
    trajectories: list[tuple[int, ...]],
    rescale,
) -> tuple[torch.Tensor, dict]:
    """Token-normalized MAE over all visited states of sampled trajectories. No detach."""
    y = select_target(future_y)
    hist = model.prepare_history(select_history(history_x) if history_x.shape[-1] > 3 else history_x)
    preds = []
    tgts = []
    for tau in trajectories:
        z_prev = None
        s_prev = 0
        for s_next in tau:
            z_prev = model.transition(hist, z_prev, s_prev, s_next)
            tgt_s = build_resolution_target(y, int(s_next))
            pr, tg = rescale_pair(z_prev, tgt_s, rescale)
            preds.append(pr)
            tgts.append(tg)
            s_prev = int(s_next)
    loss = token_normalized_mae(preds, tgts, null_val=NULL_VAL)
    return loss, {"n_traj": len(trajectories), "n_supervised_pairs": len(preds)}


@torch.no_grad()
def evaluate_trajectories(
    model: ForecastTrajectoryNet,
    loader: DataLoader,
    trajectories: list[tuple[int, ...]],
    device: torch.device,
    rescale,
    max_batches: Optional[int] = None,
) -> dict:
    model.eval()
    acc = {model.graph.trajectory_key(t): {"mae": [], "rmse": [], "mape": []} for t in trajectories}
    n = 0
    any_nonfinite = False
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        future, history, _sis = batch
        history = select_history(history.to(device))
        y = select_target(future.to(device))
        hist = model.prepare_history(history)
        for tau in trajectories:
            states = model.rollout(history, tau, history=hist)
            z12 = states[model.graph.H]
            if not finite_tensor(z12):
                any_nonfinite = True
            m = raw_panel_metrics(z12, y, rescale)
            key = model.graph.trajectory_key(tau)
            acc[key]["mae"].append(m["MAE"])
            acc[key]["rmse"].append(m["RMSE"])
            acc[key]["mape"].append(m["MAPE"])
        n += history.shape[0]
    out = {}
    maes = []
    for tau in trajectories:
        key = model.graph.trajectory_key(tau)
        mae = float(np.mean(acc[key]["mae"])) if acc[key]["mae"] else float("nan")
        out[key] = {
            "MAE": mae,
            "RMSE": float(np.mean(acc[key]["rmse"])) if acc[key]["rmse"] else float("nan"),
            "MAPE": float(np.mean(acc[key]["mape"])) if acc[key]["mape"] else float("nan"),
            "trajectory": list(tau),
        }
        maes.append(mae)
    primary = float(np.mean(maes)) if maes else float("nan")
    return {
        "n_samples": n,
        "per_trajectory": out,
        "primary_mean_final_MAE": primary,
        "any_nonfinite": any_nonfinite,
    }


def should_extend_training(val_history: list[float], best_epoch: int, current_epoch: int) -> bool:
    if not val_history:
        return True
    if current_epoch - int(best_epoch) <= 15:
        return True
    last = val_history[-10:] if len(val_history) >= 10 else val_history
    if len(last) < 3:
        return True
    # negative slope of primary MAE => improving
    xs = np.arange(len(last), dtype=np.float64)
    ys = np.array(last, dtype=np.float64)
    slope = float(np.polyfit(xs, ys, 1)[0])
    rel = (float(last[0]) - float(last[-1])) / max(abs(float(last[0])), 1e-6)
    return slope < -1e-4 or rel > 0.002


def optimizer_nonzero_update(model, opt, clip=5.0) -> dict:
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return {"grad_norm": 0.0, "updated": False}
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    opt.step()
    changed = False
    for p, b in zip([p for p in model.parameters() if p.requires_grad], before):
        if not torch.equal(p.detach(), b):
            changed = True
            break
    return {"grad_norm": grad_norm, "updated": changed}


# ---------------------------------------------------------------------------
# Transition training
# ---------------------------------------------------------------------------


def train_transition(
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    train_indices: Optional[list[int]],
    valid_indices: Optional[list[int]],
    test_indices: Optional[list[int]],
    out_ckpt: Path,
    history_json: Path,
    acceptance: bool = False,
    auto_extend: bool = True,
    planned_epochs: int = 100,
    max_epochs: int = 250,
    eval_test: bool = False,
) -> dict:
    seed_all(seed)
    scaler, rescale = load_scaler()
    model = build_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[1, 35, 60, 80, 95], gamma=0.5
    )
    train_loader, n_train = make_loader("train", train_indices, batch_size, shuffle=True)
    valid_loader, n_valid = make_loader("valid", valid_indices, batch_size, shuffle=False)
    test_loader, n_test = (None, 0)
    if eval_test:
        test_loader, n_test = make_loader("test", test_indices, batch_size, shuffle=False)

    sampler = EdgeBalancedSampler(model.graph)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    history_json.parent.mkdir(parents=True, exist_ok=True)

    best_metric = float("inf")
    best_epoch = 0
    val_curve = []
    hist_rows = []
    last_grad = 0.0
    last_train_loss = float("nan")
    last_update = False
    current_max = int(epochs if acceptance else planned_epochs)
    if acceptance:
        current_max = int(epochs)
        auto_extend = False
    epoch = 0
    verdict = "TRANSITION_TRAINING_CONVERGED"
    nan_inf = False

    while epoch < current_max:
        epoch += 1
        model.train()
        losses = []
        for future, history, _sis in train_loader:
            history = history.to(device)
            future = future.to(device)
            tau_a = sampler.sample_a()
            tau_b = sampler.sample_b()
            sampler.observe(tau_a)
            sampler.observe(tau_b)
            opt.zero_grad(set_to_none=True)
            loss, _meta = transition_batch_loss(
                model, history, future, [tau_a, tau_b], rescale
            )
            if not torch.isfinite(loss):
                nan_inf = True
                continue
            loss.backward()
            upd = optimizer_nonzero_update(model, opt, clip=5.0)
            last_grad = upd["grad_norm"]
            last_update = last_update or upd["updated"]
            losses.append(float(loss.item()))
        sched.step()
        last_train_loss = float(np.mean(losses)) if losses else float("nan")
        val = evaluate_trajectories(
            model, valid_loader, list(VALIDATION_PANEL), device, rescale
        )
        primary = val["primary_mean_final_MAE"]
        val_curve.append(primary)
        if primary < best_metric:
            best_metric = primary
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "primary_valid_mae": primary,
                    "model_args": default_model_args(),
                },
                out_ckpt,
            )
        row = {
            "epoch": epoch,
            "train_loss": last_train_loss,
            "valid_primary_mae": primary,
            "valid": val,
            "lr": float(opt.param_groups[0]["lr"]),
            "edge_exposure": sampler.report(),
        }
        if epoch % 10 == 0 and not acceptance:
            audit_loader, _ = make_loader(
                "valid",
                (valid_indices[:256] if valid_indices is not None else list(range(256))),
                batch_size,
                False,
            )
            row["full_16_audit"] = evaluate_trajectories(
                model,
                audit_loader,
                list(model.graph.terminal_trajectories()),
                device,
                rescale,
            )
        hist_rows.append(row)
        print(
            f"[transition] epoch={epoch}/{current_max} loss={last_train_loss:.4f} "
            f"valid_primary_MAE={primary:.4f} best={best_metric:.4f}@{best_epoch} "
            f"edge_ratio={sampler.report()['max_min_ratio']:.3f}",
            flush=True,
        )
        if (not acceptance) and auto_extend and epoch == current_max:
            if should_extend_training(val_curve, best_epoch, epoch) and current_max < max_epochs:
                current_max = min(max_epochs, current_max + 50)
                print(f"[transition] auto-extend to {current_max}", flush=True)
            elif epoch >= max_epochs and should_extend_training(val_curve, best_epoch, epoch):
                verdict = "TRANSITION_TRAINING_HIT_MAX_WHILE_IMPROVING"

    # restore best
    if out_ckpt.is_file():
        ckpt = torch.load(out_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        best_epoch = int(ckpt.get("epoch", best_epoch))
        best_metric = float(ckpt.get("primary_valid_mae", best_metric))

    valid_final = evaluate_trajectories(
        model, valid_loader, list(VALIDATION_PANEL), device, rescale
    )
    test_final = None
    if test_loader is not None:
        test_final = evaluate_trajectories(
            model, test_loader, list(ACCEPTANCE_TRAJECTORIES), device, rescale
        )

    payload = {
        "best_epoch": best_epoch,
        "best_valid_primary_mae": best_metric,
        "epochs_run": epoch,
        "planned_epochs": planned_epochs,
        "max_epochs": max_epochs,
        "verdict": verdict if not acceptance else "ACCEPTANCE_1EPOCH",
        "train_loss": last_train_loss,
        "valid": valid_final,
        "test": test_final,
        "grad_norm": last_grad,
        "optimizer_updated": last_update,
        "nan_inf": nan_inf,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "edge_exposure": sampler.report(),
        "param_count": count_params(model),
        "transition_param_count": model.transition_parameter_count(),
        "history_param_count": model.history_parameter_count(),
        "checkpoint": str(out_ckpt),
    }
    history_json.write_text(json.dumps({"rows": hist_rows, "summary": payload}, indent=2, default=str))
    return {"model": model, "summary": payload, "history": hist_rows}


# ---------------------------------------------------------------------------
# Latency / cache / oracle
# ---------------------------------------------------------------------------


def build_latency_table(model, loader, device, warmup, iters, policy=None) -> dict:
    future, history, _ = next(iter(loader))
    x = select_history(history.to(device))[:1]
    return profile_transition_latency(
        model, x, device, warmup=warmup, iters=iters, policy=policy
    )


def all_traj_costs(graph, latency, include_policy=False) -> dict:
    dense = graph.dense_trajectory()
    dense_c = trajectory_cost_ms(dense, graph, latency["lookup"], include_policy=include_policy)
    c_dense = dense_c["c_ms"]
    rows = {}
    for tau in graph.terminal_trajectories():
        rec = trajectory_cost_ms(tau, graph, latency["lookup"], include_policy=include_policy)
        rec["c_norm"] = rec["c_ms"] / max(c_dense, 1e-9)
        rec["c_transition_norm"] = rec["c_transition_ms"] / max(dense_c["c_transition_ms"], 1e-9)
        rows[graph.trajectory_key(tau)] = rec
    return {
        "dense": dense_c,
        "c_dense_ms": c_dense,
        "per_trajectory": rows,
    }


@torch.no_grad()
def build_trajectory_cache(
    model: ForecastTrajectoryNet,
    loader: DataLoader,
    device: torch.device,
    rescale,
    latency_table: dict,
    out_dir: Path,
    max_samples: Optional[int] = None,
) -> dict:
    model.eval()
    graph = model.graph
    costs = all_traj_costs(graph, latency_table, include_policy=False)
    writer = TrajectoryCacheWriter(out_dir, graph, shard_size=128)
    n = 0
    prefixes = graph.all_nonterminal_prefixes()
    taus = graph.terminal_trajectories()
    for future, history, sis in loader:
        history = select_history(history.to(device))
        y = select_target(future.to(device))
        sis = sis.tolist()
        for i in range(history.shape[0]):
            if max_samples is not None and n >= max_samples:
                break
            x_i = history[i : i + 1]
            y_i = y[i : i + 1]
            hist = model.prepare_history(x_i)
            prefix_z: dict[tuple[int, ...], Optional[torch.Tensor]] = {(): None}
            # rollout every terminal trajectory; reuse prefixes when possible
            traj_metrics = {}
            computed: dict[tuple[int, ...], dict[int, torch.Tensor]] = {}
            for tau in taus:
                states = model.rollout(x_i, tau, history=hist)
                computed[tau] = states
                z12 = states[graph.H]
                m = raw_panel_metrics(z12, y_i, rescale)
                key = graph.trajectory_key(tau)
                c = costs["per_trajectory"][key]
                traj_metrics[key] = {
                    "MAE": m["MAE"],
                    "RMSE": m["RMSE"],
                    "MAPE": m["MAPE"],
                    "cost_ms": c["c_transition_ms"],
                    "cost_norm": c["c_transition_norm"],
                    "trajectory": list(tau),
                }
                # fill prefixes
                pref: tuple[int, ...] = ()
                for s in tau:
                    pref = pref + (int(s),)
                    if pref[-1] != graph.H:
                        prefix_z[pref] = states[int(s)].squeeze(0)
            writer.add(
                sample_index=int(sis[i]),
                history_summary=hist.pooled.squeeze(0),
                prefix_z=prefix_z,
                traj_metrics=traj_metrics,
            )
            n += 1
        if max_samples is not None and n >= max_samples:
            break
    man = writer.close(
        extra_manifest={
            "split_note": "fp16 prefix Z + history summary; trajectory metrics in shards",
            "n_written": n,
            "latency_source": latency_table.get("source"),
        }
    )
    return man


def oracle_analysis(cache: TrajectoryCache, graph: ForecastTrajectoryGraph) -> dict:
    taus = graph.terminal_trajectories()
    keys = [graph.trajectory_key(t) for t in taus]
    losses = []
    costs_ms = []
    costs_norm = []
    sample_ids = cache.sample_indices()
    for si in sample_ids:
        rec = cache.get(si)
        tm = rec["traj_metrics"]
        losses.append([float(tm[k]["MAE"]) for k in keys])
        costs_ms.append([float(tm[k]["cost_ms"]) for k in keys])
        costs_norm.append([float(tm[k]["cost_norm"]) for k in keys])
    L = np.array(losses, dtype=np.float64)
    C = np.array(costs_ms, dtype=np.float64)
    Cn = np.array(costs_norm, dtype=np.float64)
    mean_L = L.mean(axis=0)
    best_fixed_idx = int(np.argmin(mean_L))
    L_fixed = float(mean_L[best_fixed_idx])
    L_oracle = float(L.min(axis=1).mean())
    delta = L_fixed - L_oracle
    opt_idx = L.argmin(axis=1)
    hist = Counter(int(i) for i in opt_idx)
    dist = {keys[k]: int(v) for k, v in sorted(hist.items())}

    unique_B = sorted(set(float(x) for x in C.reshape(-1)))
    budget_rows = []
    for B in unique_B:
        feas = C <= (B + 1e-9)
        # fixed: among trajectories with mean cost <= B? Spec: L_fixed(B) = min over tau with C(tau)<=B of mean_i L_i(tau)
        # and L_oracle(B) = mean_i min_{tau: C_i(tau)<=B} L_i(tau)
        mean_cost = C.mean(axis=0)
        fixed_ok = mean_cost <= (B + 1e-9)
        if not fixed_ok.any():
            continue
        L_fixed_B = float(mean_L[fixed_ok].min())
        sample_best = []
        for i in range(L.shape[0]):
            ok = feas[i]
            if not ok.any():
                continue
            sample_best.append(float(L[i, ok].min()))
        if not sample_best:
            continue
        L_oracle_B = float(np.mean(sample_best))
        budget_rows.append(
            {
                "B_ms": B,
                "L_fixed": L_fixed_B,
                "L_oracle": L_oracle_B,
                "Delta_adaptive": L_fixed_B - L_oracle_B,
                "n_samples_feasible": len(sample_best),
            }
        )

    # lambda scale from TRAIN: typical loss gap / typical C_norm gap
    loss_gap = float(np.percentile(L.max(axis=1) - L.min(axis=1), 75))
    cn_gap = float(np.percentile(Cn.max(axis=1) - Cn.min(axis=1), 75))
    lambda_max = float(loss_gap / max(cn_gap, 1e-6))
    return {
        "n_samples": int(L.shape[0]),
        "trajectories": keys,
        "mean_loss_per_trajectory": {k: float(v) for k, v in zip(keys, mean_L)},
        "best_fixed_trajectory": keys[best_fixed_idx],
        "L_fixed": L_fixed,
        "L_oracle": L_oracle,
        "Delta_adaptive": float(delta),
        "sample_optimal_trajectory_histogram": dist,
        "budget_curve": budget_rows,
        "lambda_scale": {
            "loss_gap_p75": loss_gap,
            "c_norm_gap_p75": cn_gap,
            "lambda_max": lambda_max,
            "lambda_min": 0.0,
            "derived_from": "TRAIN_oracle_not_TEST",
        },
        "eta_used": False,
        "crossfit_used": False,
    }


def derive_lambda_grid(lambda_max: float) -> list[float]:
    lam_max = max(float(lambda_max), 1e-6)
    return [0.0, 0.25 * lam_max, 0.5 * lam_max, lam_max]


# ---------------------------------------------------------------------------
# Policy training
# ---------------------------------------------------------------------------


def _z_from_rec(rec, prefix: tuple[int, ...], device):
    key = "start" if not prefix else "-".join(str(s) for s in prefix)
    z = rec["prefix_z"].get(key)
    if z is None:
        return None
    return z.to(device=device, dtype=torch.float32).unsqueeze(0)


def policy_batch_objective(
    policy: OnlineTrajectoryPolicy,
    graph: ForecastTrajectoryGraph,
    recs: list[dict],
    device: torch.device,
    lam_vec: torch.Tensor,
    no_budget: torch.Tensor,
    remaining_ms: Optional[torch.Tensor],
    remaining_norm: torch.Tensor,
    edge_cost: dict,
    min_finish: dict,
    c_dense: float,
    extra_per_edge: float,
) -> tuple[torch.Tensor, dict]:
    h = torch.stack([r["history_summary"].float() for r in recs], dim=0).to(device)
    keys = [graph.trajectory_key(t) for t in graph.terminal_trajectories()]
    L = torch.tensor(
        [[float(r["traj_metrics"][k]["MAE"]) for k in keys] for r in recs],
        device=device,
        dtype=torch.float32,
    )
    Cn = torch.tensor(
        [[float(r["traj_metrics"][k]["cost_norm"]) for k in keys] for r in recs],
        device=device,
        dtype=torch.float32,
    )
    # z_by_prefix cannot be stacked easily across samples when we loop trajectories
    # inside exact_trajectory_probs — we run sample-wise for correctness (tiny).
    probs_all = []
    feas_all = []
    path_err = []
    invalid = False
    for i, rec in enumerate(recs):
        zmap = {}
        for pref in graph.all_nonterminal_prefixes():
            zmap[pref] = _z_from_rec(rec, pref, device)
        rem_ms = None if remaining_ms is None else remaining_ms[i : i + 1]
        out = exact_trajectory_probs(
            policy,
            graph,
            h[i : i + 1],
            zmap,
            lam_vec[i : i + 1],
            remaining_norm[i : i + 1],
            no_budget[i : i + 1],
            edge_cost,
            min_finish,
            c_dense,
            rem_ms,
            extra_per_edge_ms=extra_per_edge,
        )
        probs_all.append(out["probs"])
        feas_all.append(out["feas"])
        path_err.append(float((out["path_sum"] - 1.0).abs().item()))
        if not out["valid"]:
            invalid = True
    probs = torch.cat(probs_all, dim=0)
    feas = torch.cat(feas_all, dim=0)
    loss = exact_policy_loss(probs, feas, L, Cn, lam_vec)
    return loss, {
        "path_sum_err_max": max(path_err) if path_err else 0.0,
        "path_invalid": invalid,
    }


def sample_lambda_batch(batch: int, lambda_max: float, device) -> torch.Tensor:
    out = []
    for _ in range(batch):
        if random.random() < 0.25:
            out.append(0.0)
        else:
            out.append(random.uniform(0.0, float(lambda_max)))
    return torch.tensor(out, device=device, dtype=torch.float32)


@torch.no_grad()
def eval_policy_regret(
    policy,
    graph,
    cache: TrajectoryCache,
    indices: list[int],
    device,
    lambda_grid: list[float],
    B_list: list[Optional[float]],
    edge_cost,
    min_finish,
    c_dense,
    extra_per_edge,
    c_hist: float = 0.0,
) -> dict:
    policy.eval()
    keys = [graph.trajectory_key(t) for t in graph.terminal_trajectories()]
    regrets = []
    latencies = []
    rows = []
    for lam in lambda_grid:
        for B in B_list:
            batch_reg = []
            batch_lat = []
            for si in indices:
                rec = cache.get(si)
                L = np.array([float(rec["traj_metrics"][k]["MAE"]) for k in keys])
                C = np.array([float(rec["traj_metrics"][k]["cost_ms"]) for k in keys])
                Cn = np.array([float(rec["traj_metrics"][k]["cost_norm"]) for k in keys])
                if B is None:
                    feas = np.ones_like(L, dtype=bool)
                    rem_ms = None
                    nb = torch.ones(1, device=device)
                    rem_norm = torch.ones(1, device=device)
                    remaining_init = None
                else:
                    feas = C <= (float(B) + 1e-9)
                    nb = torch.zeros(1, device=device)
                    remaining_init = torch.tensor(
                        [max(float(B) - float(c_hist), 0.0)], device=device
                    )
                    rem_norm = remaining_init / max(float(c_dense), 1e-6)
                zmap = {p: _z_from_rec(rec, p, device) for p in graph.all_nonterminal_prefixes()}
                h = rec["history_summary"].float().unsqueeze(0).to(device)
                out = exact_trajectory_probs(
                    policy,
                    graph,
                    h,
                    zmap,
                    torch.tensor([lam], device=device),
                    rem_norm if B is not None else torch.ones(1, device=device),
                    nb if B is not None else torch.ones(1, device=device),
                    edge_cost,
                    min_finish,
                    c_dense,
                    remaining_init if B is not None else None,
                    extra_per_edge_ms=extra_per_edge,
                )
                p = out["probs"][0].detach().cpu().numpy()
                obj = L + float(lam) * Cn
                if B is not None:
                    obj = np.where(feas, obj, np.inf)
                oracle = float(np.min(obj))
                expected = float(np.sum(p * (L + float(lam) * Cn)))
                batch_reg.append(expected - oracle)
                batch_lat.append(float(np.sum(p * C)))
            mean_reg = float(np.mean(batch_reg)) if batch_reg else float("nan")
            mean_lat = float(np.mean(batch_lat)) if batch_lat else float("nan")
            regrets.append(mean_reg)
            latencies.append(mean_lat)
            rows.append(
                {
                    "lambda": lam,
                    "B_ms": B,
                    "mean_regret": mean_reg,
                    "mean_latency_ms": mean_lat,
                    "no_budget": B is None,
                }
            )
    primary = float(np.mean(regrets)) if regrets else float("nan")
    return {"primary_mean_regret": primary, "panel": rows}


def train_policy(
    *,
    model: ForecastTrajectoryNet,
    cache: TrajectoryCache,
    split: dict,
    device: torch.device,
    latency_table: dict,
    oracle: dict,
    out_ckpt: Path,
    history_json: Path,
    min_epochs: int = 20,
    max_epochs: int = 150,
    patience: int = 25,
    batch_size: int = 16,
    acceptance: bool = False,
) -> dict:
    graph = model.graph
    policy = OnlineTrajectoryPolicy(graph, d_history=model.d_model).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4, weight_decay=1e-4)
    lambda_max = float(oracle["lambda_scale"]["lambda_max"])
    costs = all_traj_costs(graph, latency_table, include_policy=True)
    edge_cost = {}
    for k, v in latency_table["lookup"]["edges_median_ms"].items():
        a, b = k.split("->")
        edge_cost[(int(a), int(b))] = float(v)
    extra = float(latency_table["lookup"].get("policy_step_median_ms") or 0.0)
    min_finish = graph.min_finish_edge_cost(edge_cost, extra_per_edge=extra)
    c_dense = float(costs["c_dense_ms"])
    c_hist = float(latency_table["lookup"]["history_median_ms"] or 0.0)
    unique_B = sorted({float(v["c_ms"]) for v in costs["per_trajectory"].values()})
    lambda_grid = derive_lambda_grid(lambda_max)
    B_list: list[Optional[float]] = [None] + unique_B

    train_ids = split["policy_train"]
    valid_ids = split["policy_valid"]
    if not train_ids:
        train_ids = cache.sample_indices()
    if not valid_ids:
        valid_ids = train_ids[: max(1, len(train_ids) // 5)] or train_ids

    def batches(ids):
        random.shuffle(ids)
        for i in range(0, len(ids), batch_size):
            chunk = ids[i : i + batch_size]
            yield [cache.get(j) for j in chunk]

    best = float("inf")
    best_epoch = 0
    bad = 0
    hist = []
    init_loss = None
    epochs = 1 if acceptance else max_epochs
    min_ep = 1 if acceptance else min_epochs
    last_loss = float("nan")
    path_err = 0.0
    path_invalid = False

    for epoch in range(1, epochs + 1):
        policy.train()
        losses = []
        for recs in batches(list(train_ids)):
            bsz = len(recs)
            lam = sample_lambda_batch(bsz, lambda_max, device)
            # mix no-budget and a random B
            nb = torch.zeros(bsz, device=device)
            rem_ms = torch.zeros(bsz, device=device)
            for i in range(bsz):
                if random.random() < 0.5:
                    nb[i] = 1.0
                    rem_ms[i] = max(c_dense - c_hist, 0.0)
                else:
                    nb[i] = 0.0
                    rem_ms[i] = max(float(random.choice(unique_B)) - c_hist, 0.0)
            rem_norm = rem_ms / max(c_dense, 1e-6)
            opt.zero_grad(set_to_none=True)
            loss, meta = policy_batch_objective(
                policy, graph, recs, device, lam, nb, rem_ms, rem_norm,
                edge_cost, min_finish, c_dense, extra,
            )
            path_err = max(path_err, float(meta["path_sum_err_max"]))
            path_invalid = path_invalid or bool(meta["path_invalid"])
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        last_loss = float(np.mean(losses)) if losses else float("nan")
        if init_loss is None:
            init_loss = last_loss
        val = eval_policy_regret(
            policy, graph, cache, valid_ids, device, lambda_grid, B_list,
            edge_cost, min_finish, c_dense, extra, c_hist=c_hist,
        )
        primary = val["primary_mean_regret"]
        hist.append({"epoch": epoch, "train_loss": last_loss, "valid": val})
        print(
            f"[policy] epoch={epoch} loss={last_loss:.5f} valid_regret={primary:.5f}",
            flush=True,
        )
        improved = primary < best - 1e-8 or (
            abs(primary - best) <= 1e-8
            and float(np.mean([r["mean_latency_ms"] for r in val["panel"]]))
            < (best * 0 + 1e18)
        )
        # tie-break lower real latency among panel
        mean_lat = float(np.mean([r["mean_latency_ms"] for r in val["panel"]]))
        if primary < best - 1e-8 or (
            abs(primary - best) <= 1e-8 and mean_lat < getattr(train_policy, "_best_lat", 1e18)
        ):
            best = primary
            best_epoch = epoch
            train_policy._best_lat = mean_lat
            bad = 0
            out_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": policy.state_dict(),
                    "primary_regret": primary,
                    "lambda_max": lambda_max,
                    "lambda_grid": lambda_grid,
                    "unique_B": unique_B,
                },
                out_ckpt,
            )
        else:
            bad += 1
        if (not acceptance) and epoch >= min_ep and bad >= patience:
            break

    if out_ckpt.is_file():
        ckpt = torch.load(out_ckpt, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["state_dict"])
        best_epoch = int(ckpt.get("epoch", best_epoch))
        best = float(ckpt.get("primary_regret", best))

    val_final = eval_policy_regret(
        policy, graph, cache, valid_ids, device, lambda_grid, B_list,
        edge_cost, min_finish, c_dense, extra, c_hist=c_hist,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_valid_regret": best,
        "init_loss": init_loss,
        "final_loss": last_loss,
        "path_sum_err_max": path_err,
        "path_probability_invalid": path_invalid or path_err > PATH_PROB_ATOL,
        "param_count": policy.parameter_count(),
        "lambda_max": lambda_max,
        "lambda_grid": lambda_grid,
        "unique_B": unique_B,
        "valid": val_final,
        "checkpoint": str(out_ckpt),
    }
    history_json.parent.mkdir(parents=True, exist_ok=True)
    history_json.write_text(json.dumps({"rows": hist, "summary": summary}, indent=2, default=str))
    return {"policy": policy, "summary": summary}


# ---------------------------------------------------------------------------
# Online policy inference on a dataloader (VALID/TEST)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_online_policy(
    model: ForecastTrajectoryNet,
    policy: OnlineTrajectoryPolicy,
    loader: DataLoader,
    device: torch.device,
    rescale,
    latency_table: dict,
    lam: float,
    B_ms: Optional[float],
    extra_per_edge: float,
) -> dict:
    model.eval()
    policy.eval()
    graph = model.graph
    edge_cost = {}
    for k, v in latency_table["lookup"]["edges_median_ms"].items():
        a, b = k.split("->")
        edge_cost[(int(a), int(b))] = float(v)
    min_finish = graph.min_finish_edge_cost(edge_cost, extra_per_edge=extra_per_edge)
    c_hist = float(latency_table["lookup"]["history_median_ms"] or 0.0)
    c_dense_pack = all_traj_costs(graph, latency_table, include_policy=True)
    c_dense = float(c_dense_pack["c_dense_ms"])
    maes, rmses, mapes, lats, n_trans = [], [], [], [], []
    tau_hist = Counter()
    first_hist = Counter()
    edge_hist = Counter()
    any_nonfinite = False
    n = 0
    for future, history, _sis in loader:
        history = select_history(history.to(device))
        y = select_target(future.to(device))
        for i in range(history.shape[0]):
            x_i = history[i : i + 1]
            y_i = y[i : i + 1]
            hist = model.prepare_history(x_i)
            s = 0
            z = None
            chosen = []
            remaining = None if B_ms is None else (float(B_ms) - c_hist)
            nb = torch.ones(1, device=device) if B_ms is None else torch.zeros(1, device=device)
            while s != graph.H:
                rem_norm = torch.ones(1, device=device)
                if remaining is not None:
                    rem_norm = torch.tensor([remaining / max(c_dense, 1e-6)], device=device)
                feas = None
                if remaining is not None:
                    feas = [
                        cand
                        for cand in graph.successors(s)
                        if graph.edge_feasible(
                            s, cand, remaining, edge_cost, min_finish, extra_per_edge
                        )
                    ]
                    if not feas:
                        feas = graph.successors(s)
                out = policy(
                    h_history=hist.pooled,
                    z_current=z,
                    s_current=s,
                    lam=torch.tensor([lam], device=device),
                    remaining_norm=rem_norm,
                    no_budget=nb,
                    H=graph.H,
                    feasible_dest=feas,
                )
                idx = int(out["probs"][0].argmax().item())
                s_next = int(policy.dest_states[idx])
                if not graph.is_legal_edge(s, s_next):
                    s_next = graph.successors(s)[-1]
                z = model.transition(hist, z, s, s_next)
                if not finite_tensor(z):
                    any_nonfinite = True
                edge_hist[f"{s}->{s_next}"] += 1
                if not chosen:
                    first_hist[str(s_next)] += 1
                chosen.append(s_next)
                if remaining is not None:
                    remaining -= float(edge_cost[(s, s_next)]) + extra_per_edge
                s = s_next
            m = raw_panel_metrics(z, y_i, rescale)
            maes.append(m["MAE"])
            rmses.append(m["RMSE"])
            mapes.append(m["MAPE"])
            key = graph.trajectory_key(chosen)
            tau_hist[key] += 1
            c = trajectory_cost_ms(chosen, graph, latency_table["lookup"], include_policy=True)
            lats.append(c["c_ms"])
            n_trans.append(len(chosen))
            n += 1
    lat_arr = np.array(lats) if lats else np.array([float("nan")])
    return {
        "n": n,
        "MAE": float(np.mean(maes)) if maes else float("nan"),
        "RMSE": float(np.mean(rmses)) if rmses else float("nan"),
        "MAPE": float(np.mean(mapes)) if mapes else float("nan"),
        "avg_latency_ms": float(np.mean(lat_arr)),
        "p95_latency_ms": float(np.percentile(lat_arr, 95)) if len(lat_arr) else float("nan"),
        "avg_n_transitions": float(np.mean(n_trans)) if n_trans else float("nan"),
        "trajectory_histogram": dict(tau_hist),
        "first_state_histogram": dict(first_hist),
        "state_transition_histogram": dict(edge_hist),
        "lambda": lam,
        "B_ms": B_ms,
        "any_nonfinite": any_nonfinite,
    }


def dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
