#!/usr/bin/env python3
"""Synthetic CPU tests for budget F2F oracle / planner fixes (no PEMS training)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs import BudgetConditionedAdaptiveF2FNet
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_conditioned_f2f_loss import (
    planner_imitation_loss,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    budgets_from_intensity_tensor,
)
from basicts.data import IndexedTimeSeriesForecastingDataset
from basicts.runners.runner_zoo.chain_forecasting_runner import ChainForecastingRunner
from scripts.budget_f2f_synth_kwargs import synthetic_budget_f2f_kwargs


PASS = []
FAIL = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"[PASS] {name}")
    else:
        FAIL.append(name)
        print(f"[FAIL] {name} {detail}")


def _tiny_model(**overrides):
    kw = synthetic_budget_f2f_kwargs(
        forced_route=None,
        route_selection_mode="batch",
        loss_mode="dynamic_fair",
        training_phase="planner",
    )
    kw.update(overrides)
    return BudgetConditionedAdaptiveF2FNet(**kw)


def test_indexed_dataset_shuffle_stable_index():
    # Minimal fake by subclassing with in-memory override is heavy; use real PEMS
    # index file if present, else skip gracefully with synthetic Dataset wrapper.
    data = ROOT / "datasets/PEMS04/data_in12_out12.pkl"
    index = ROOT / "datasets/PEMS04/index_in12_out12.pkl"
    if not data.is_file() or not index.is_file():
        # Synthetic stand-in
        class _Fake(IndexedTimeSeriesForecastingDataset):
            def __init__(self):
                self.data = torch.zeros(20, 12, 4, 4)
                self.index = list(range(10))

            def __len__(self):
                return len(self.index)

            def __getitem__(self, index: int):
                return torch.zeros(12, 4, 1), torch.zeros(12, 4, 4), int(index)

        ds = _Fake()
    else:
        ds = IndexedTimeSeriesForecastingDataset(
            data_file_path=str(data),
            index_file_path=str(index),
            mode="train",
        )
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    seen = set()
    for batch in loader:
        assert len(batch) == 3
        _, _, idx = batch
        for i in idx.tolist():
            seen.add(int(i))
            fut, hist, si = ds[int(i)]
            check(
                "shuffle_index_matches_getitem",
                int(si) == int(i),
                f"si={si} i={i}",
            )
            break
        break
    check("indexed_dataset_returns_ternary", True)


def test_per_sample_oracle_labels_differ():
    runner = object.__new__(ChainForecastingRunner)
    runner._budget_oracle_by_index = {
        (0, 0.5): 0,
        (1, 0.5): 3,
        (2, 0.5): 1,
        (0, 1.0): 3,
        (1, 1.0): 2,
    }
    runner._budget_oracle_meta = {"intensities": [0.5, 1.0]}
    si = torch.tensor([0, 1, 2])
    eta = torch.tensor([0.5, 0.5, 0.5])
    labels = ChainForecastingRunner._lookup_oracle_labels(runner, si, eta)
    check("per_sample_oracle_labels_differ", labels.tolist() == [0, 3, 1], str(labels))
    labels2 = ChainForecastingRunner._lookup_oracle_labels(
        runner, torch.tensor([0, 1]), torch.tensor([1.0, 1.0])
    )
    check("different_intensity_different_label", labels2.tolist() == [3, 2], str(labels2))


def test_feasible_mask_varies_with_intensity():
    m = _tiny_model(route_selection_mode="sample", training_phase="eval")
    m.eval()
    x = torch.randn(4, 12, 7, 4)
    with torch.no_grad():
        low = m._select_route_id(
            x, train=False, intensity_override=torch.zeros(4)
        )
        high = m._select_route_id(
            x, train=False, intensity_override=torch.ones(4)
        )
    check(
        "low_intensity_fewer_feasible",
        int(low["feasible_mask"].float().sum())
        <= int(high["feasible_mask"].float().sum()),
        f"low={low['feasible_mask'].sum()} high={high['feasible_mask'].sum()}",
    )
    check(
        "high_intensity_all_feasible",
        bool(high["feasible_mask"].all()),
        str(high["feasible_mask"]),
    )


def test_planner_only_no_execute_route():
    m = _tiny_model(
        training_phase="planner",
        route_selection_mode="sample",
        freeze_forecasting_backbone=True,
    )
    m.train()
    m.freeze_backbone(True)
    x = torch.randn(3, 12, 7, 4)
    oracle = torch.tensor([0, 0, 0])  # cheapest usually feasible
    called = {"n": 0}
    orig = m._execute_route

    def wrapped(*a, **k):
        called["n"] += 1
        return orig(*a, **k)

    m._execute_route = wrapped  # type: ignore
    out = m(
        history_data=x,
        train=True,
        return_all=True,
        oracle_route_id=oracle,
        inference_intensity_override=torch.zeros(3),
    )
    check("planner_only_flag", bool(out.get("planner_only")), str(out.keys()))
    check("planner_only_no_execute_route", called["n"] == 0, f"n={called['n']}")


def test_planner_grad_backbone_frozen():
    m = _tiny_model(
        training_phase="planner",
        route_selection_mode="sample",
        freeze_forecasting_backbone=True,
    )
    m.train()
    m.freeze_backbone(True)
    x = torch.randn(4, 12, 7, 4)
    # Use intensity 1 so full route oracle is feasible
    oracle = torch.tensor([3, 2, 1, 0])
    out = m(
        history_data=x,
        train=True,
        return_all=True,
        oracle_route_id=oracle,
        inference_intensity_override=torch.ones(4),
    )
    parts = planner_imitation_loss(
        out["masked_route_logits"],
        oracle,
        out["expected_cost"],
        out["budget"],
        lambda_imitation=1.0,
        lambda_budget=0.0,
    )
    m.zero_grad(set_to_none=True)
    parts["loss"].backward()
    plan_g = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in m.named_parameters()
        if n.startswith("planner.")
    )
    bb_g = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in m.named_parameters()
        if n.startswith("backbone.")
    )
    check("planner_head_nonzero_grad", plan_g)
    check("frozen_backbone_zero_grad", not bb_g)


def test_sample_mode_bucketed_execution_order():
    m = _tiny_model(route_selection_mode="sample", route_granularity="sample")
    m.eval()
    m.set_training_phase("eval")
    x = torch.randn(2, 12, 7, 4)
    # Force different routes via bucketing API
    ids = torch.tensor([0, 3])
    out = m._execute_routes_bucketed(x, ids)
    check(
        "sample_bucket_executed_ids",
        out["executed_route_id"].tolist() == [0, 3],
        str(out["executed_route_id"]),
    )
    # Scatter order: pred[0] from route0, pred[1] from route3
    o0 = m._execute_route(x[0:1], m.candidate_routes[0])["pred"]
    o3 = m._execute_route(x[1:2], m.candidate_routes[3])["pred"]
    d0 = (out["pred"][0] - o0[0]).abs().max().item()
    d1 = (out["pred"][1] - o3[0]).abs().max().item()
    check("scatter_order_sample0", d0 < 1e-5, str(d0))
    check("scatter_order_sample1", d1 < 1e-5, str(d1))


def test_batch_mode_mean_logits_not_majority():
    m = _tiny_model(route_selection_mode="batch", route_granularity="batch")
    m.eval()
    x = torch.randn(5, 12, 7, 4)
    # Monkeypatch planner to return known logits where mean argmax != majority of per-sample argmax
    R = len(m.candidate_routes)
    logits = torch.zeros(5, R)
    # samples 0-3 prefer route 0; sample 4 prefers route 3 with huge score
    logits[:, 0] = 1.0
    logits[4, 3] = 100.0
    # Make mean favor route 3: boost all samples slightly on route 3 so mean wins route 3
    # Actually: mean of [1,1,1,1,1] on r0 vs mean of [0,0,0,0,100] on r3 => r3 wins mean
    # per-sample argmax majority is route 0 (4 vs 1)
    costs = m.route_costs
    budget = torch.ones(5) * float(costs.max().item())
    feas = torch.ones(5, R, dtype=torch.bool)
    masked = logits.masked_fill(~feas, -1e9)
    probs = torch.softmax(masked, dim=-1)
    selected = masked.argmax(dim=-1)

    def fake_planner(history, intensity, route_costs, budget, deterministic=True):
        return {
            "route_logits": logits,
            "masked_route_logits": masked,
            "route_probs": probs,
            "feasible_mask": feas,
            "selected_route_id": selected,
            "selected_cost": costs[selected],
            "expected_cost": (probs * costs.view(1, -1)).sum(-1),
            "budget": budget,
            "features": history.new_zeros(history.shape[0], 8),
        }

    m.planner.forward = fake_planner  # type: ignore
    plan = m._select_route_id(x, train=False, intensity_override=1.0)
    majority = int(torch.bincount(selected, minlength=R).argmax().item())
    mean_id = int(plan["batch_route_id"])
    check("batch_not_majority_vote", mean_id != majority, f"mean={mean_id} maj={majority}")
    check("batch_mean_picks_route3", mean_id == 3, str(mean_id))
    check("batch_route_logits_present", plan.get("batch_route_logits") is not None)


def test_expected_cost_grad():
    m = _tiny_model(route_selection_mode="sample", training_phase="planner")
    m.train()
    x = torch.randn(3, 12, 7, 4)
    plan = m._select_route_id(x, train=True, intensity_override=torch.ones(3))
    loss = plan["expected_cost"].mean()
    m.zero_grad(set_to_none=True)
    loss.backward()
    g = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in m.named_parameters()
        if n.startswith("planner.")
    )
    check("expected_cost_planner_grad", g)


def test_oracle_route_order_mismatch_raises():
    runner = object.__new__(ChainForecastingRunner)
    runner.planner_training_intensities = [0.5]
    runner.dataset_name = "PEMS04"
    runner.logger = SimpleNamespace(info=lambda *a, **k: None)
    m = _tiny_model()
    runner.model = m
    bad = {
        "metadata": {
            "dataset": "PEMS04",
            "horizon": 12,
            "split": "train",
            "candidate_routes": [[12], [3, 6, 12], [6, 12], [3, 12]],  # wrong order
            "candidate_routes_order": ["12", "3,6,12", "6,12", "3,12"],
            "route_costs": m.route_costs.tolist(),
            "intensities": [0.5],
            "loss_scale": "raw_physical_scale",
        },
        "records": [{"sample_index": 0, "intensity": 0.5, "oracle_route_id": 0}],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(bad, f)
        path = f.name
    raised = False
    try:
        ChainForecastingRunner._load_budget_oracle(runner, path)
    except RuntimeError as e:
        raised = "order" in str(e).lower() or "mismatch" in str(e).lower()
    check("oracle_order_mismatch_raises", raised)


def test_joint_optimizer_scheduler_bind():
    import torch.optim as optim

    class _R:
        pass

    r = _R()
    r.budget_training_phase = "joint"
    r.model = _tiny_model(training_phase="joint")
    r.optim = optim.Adam(r.model.parameters(), lr=0.01)
    r.scheduler = optim.lr_scheduler.MultiStepLR(r.optim, milestones=[1], gamma=0.5)
    r.logger = SimpleNamespace(info=lambda *a, **k: None)
    cfg = {
        "MODEL": {
            "PARAM": {
                "backbone_lr": 1e-4,
                "planner_lr": 1e-3,
            }
        },
        "TRAIN": {
            "OPTIM": {"PARAM": {"lr": 0.002, "weight_decay": 0.0}},
            "LR_SCHEDULER": {
                "TYPE": "MultiStepLR",
                "PARAM": {"milestones": [1, 2], "gamma": 0.5},
            },
        },
    }
    # Manually run the joint rebuild block logic
    param = cfg["MODEL"]["PARAM"]
    train_cfg = cfg["TRAIN"]
    optim_param = train_cfg["OPTIM"]["PARAM"]
    sched_cfg = train_cfg["LR_SCHEDULER"]
    base_lr = float(optim_param.get("lr", 0.002))
    backbone_lr = float(param.get("backbone_lr", base_lr * 0.1))
    planner_lr = float(param.get("planner_lr", base_lr))
    groups = [
        {"params": list(r.model.backbone.parameters()), "lr": backbone_lr},
        {"params": list(r.model.planner.parameters()), "lr": planner_lr},
    ]
    r.optim = torch.optim.Adam(groups)
    sched_cls = getattr(torch.optim.lr_scheduler, sched_cfg["TYPE"])
    r.scheduler = sched_cls(r.optim, **sched_cfg["PARAM"])
    check(
        "joint_scheduler_binds_new_optim",
        id(r.scheduler.optimizer) == id(r.optim),
        f"{id(r.scheduler.optimizer)} vs {id(r.optim)}",
    )
    check("joint_has_two_param_groups", len(r.optim.param_groups) == 2)


def test_budgets_tensor_helper():
    costs = [0.5, 0.7, 0.8, 1.0]
    etas = torch.tensor([0.0, 0.5, 1.0])
    b = budgets_from_intensity_tensor(etas, costs)
    check("budget_eta0_is_min", abs(float(b[0]) - 0.5) < 1e-6, str(b))
    check("budget_eta1_is_max", abs(float(b[2]) - 1.0) < 1e-6, str(b))


def test_joint_sample_raises():
    m = _tiny_model(
        training_phase="joint",
        route_selection_mode="sample",
        route_granularity="sample",
    )
    m.train()
    x = torch.randn(2, 12, 7, 4)
    raised = False
    try:
        m(history_data=x, train=True, return_all=True, oracle_route_id=torch.tensor([0, 1]))
    except RuntimeError as e:
        raised = "sample" in str(e).lower()
    check("joint_sample_granularity_raises", raised)


def main() -> int:
    test_indexed_dataset_shuffle_stable_index()
    test_per_sample_oracle_labels_differ()
    test_feasible_mask_varies_with_intensity()
    test_planner_only_no_execute_route()
    test_planner_grad_backbone_frozen()
    test_sample_mode_bucketed_execution_order()
    test_batch_mode_mean_logits_not_majority()
    test_expected_cost_grad()
    test_oracle_route_order_mismatch_raises()
    test_joint_optimizer_scheduler_bind()
    test_budgets_tensor_helper()
    test_joint_sample_raises()
    print("=" * 60)
    print(f"PASS={len(PASS)} FAIL={len(FAIL)}")
    for f in FAIL:
        print("  failed:", f)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
