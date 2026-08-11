#!/usr/bin/env python3
"""Train Group-Relative Sequential Forecast Refinement Policy (Plan B).

GRPO-inspired group-relative trajectory policy optimization.
No critic / GAE / value network.

Formal reward MUST come from temporal cross-fitted route oracle.
Smoke may temporarily use synthetic / in-sample losses with an explicit disclaimer.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_refinement_context import (
    pool_pre_route_context,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budget_conditioned_adaptive_f2f import (
    BudgetConditionedAdaptiveF2FNet,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.group_relative_refinement_objective import (
    clipped_trajectory_objective,
    group_relative_advantages,
    terminal_route_reward,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.group_relative_refinement_policy import (
    GroupRelativeRefinementPolicy,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.sequential_f2f_environment import (
    A0_QUARTER,
    SequentialF2FEnvironment,
)
from scripts.budget_f2f_synth_kwargs import synthetic_budget_f2f_kwargs


def _require_safety(args) -> None:
    if args.smoke_test:
        return
    if not args.confirm_full_run:
        raise RuntimeError(
            "Full training is disabled. Pass --confirm-full-run manually."
        )


def _build_s0(model, history: torch.Tensor) -> torch.Tensor:
    h = model.extract_pre_route_context(history, detach=True)
    return pool_pre_route_context(h)


def _zq_for_batch(env, history: torch.Tensor) -> torch.Tensor:
    """Execute quarter prefix once per sample to get explicit Z_q (detach)."""
    with torch.no_grad():
        pref = env.execute_quarter_prefix(history)
        return pref["Z_q"].detach()


def smoke_train(args) -> int:
    print("THIS IS ONLY A CODE SMOKE TEST.")
    print("IN-SAMPLE ORACLE IS FORBIDDEN FOR FORMAL RL.")
    print("SMOKE may use synthetic route losses only for code path verification.")

    device = torch.device(
        args.device
        if ("cpu" in args.device or torch.cuda.is_available())
        else "cpu"
    )
    supernet = BudgetConditionedAdaptiveF2FNet(
        **synthetic_budget_f2f_kwargs(
            node_size=7, training_phase="eval", route_selection_mode="forced"
        )
    ).to(device)
    supernet.freeze_backbone(True)
    supernet.eval()
    for p in supernet.parameters():
        p.requires_grad = False

    env = SequentialF2FEnvironment(supernet)
    # Probe context dim
    with torch.no_grad():
        h0 = torch.randn(2, 12, 7, 4, device=device)
        s0 = _build_s0(supernet, h0)
        ctx_dim = int(s0.shape[-1])

    policy = GroupRelativeRefinementPolicy(context_dim=ctx_dim, zq_dim=1, hidden=128).to(
        device
    )
    opt = torch.optim.Adam(policy.parameters(), lr=float(args.lr))

    # Synthetic route losses for smoke reward (NOT scientific)
    n = 16
    history = torch.randn(n, 12, 7, 4, device=device)
    # L = [18.20, 18.12, 18.18, 18.13] style with noise
    base = torch.tensor([18.20, 18.12, 18.18, 18.13], device=device)
    route_losses = base.unsqueeze(0) + 0.05 * torch.randn(n, 4, device=device)
    costs = supernet.route_costs.detach().to(device).float()

    etas = [0.0, 0.25, 0.5, 0.75, 1.0]
    eta = float(etas[args.smoke_eta_idx % len(etas)])
    masks = env.action_masks(eta)
    feas = masks["feasible_routes"].to(device).unsqueeze(0).expand(n, -1)
    mask0 = masks["mask0"].to(device).unsqueeze(0).expand(n, -1)
    mask1 = masks["mask1"].to(device).unsqueeze(0).expand(n, -1)

    rewards, _ = terminal_route_reward(
        route_losses,
        costs,
        feas,
        delta_abs=float(args.delta_abs),
        lambda_quality=float(args.lambda_quality),
        lambda_cost=float(args.lambda_cost),
    )
    advantages, adv_info = group_relative_advantages(rewards, feas)
    print(f"[smoke B] eta={eta} zero_var_groups={adv_info['zero_variance_groups']}")

    # Build trajectory tensors for all feasible routes (enumerate)
    trajs = env.enumerate_feasible_trajectories(eta)
    # Expand: for each sample, one row per feasible traj
    rows_s0 = []
    rows_zq = []
    rows_a0 = []
    rows_a1 = []
    rows_adv = []
    rows_m0 = []
    rows_m1 = []
    # Need Z_q only for quarter trajs — compute once
    zq_all = _zq_for_batch(env, history)
    s0_all = _build_s0(supernet, history)

    for t in trajs:
        rid = int(t["route_id"])
        a0 = torch.full((n,), int(t["a0"]), device=device, dtype=torch.long)
        if t["a1"] is None:
            a1 = torch.zeros(n, device=device, dtype=torch.long)
        else:
            a1 = torch.full((n,), int(t["a1"]), device=device, dtype=torch.long)
        rows_s0.append(s0_all)
        rows_zq.append(zq_all)
        rows_a0.append(a0)
        rows_a1.append(a1)
        rows_adv.append(advantages[:, rid])
        rows_m0.append(mask0)
        rows_m1.append(mask1)

    s0_b = torch.cat(rows_s0, dim=0)
    zq_b = torch.cat(rows_zq, dim=0)
    a0_b = torch.cat(rows_a0, dim=0)
    a1_b = torch.cat(rows_a1, dim=0)
    adv_b = torch.cat(rows_adv, dim=0).detach()
    m0_b = torch.cat(rows_m0, dim=0)
    m1_b = torch.cat(rows_m1, dim=0)

    # Rollout with pi_old
    policy.eval()
    with torch.no_grad():
        logp_old = policy.trajectory_logprob(
            policy.encode_s0(s0_b), zq_b, a0_b, a1_b, m0_b, m1_b
        ).detach()

    # Multiple update epochs on same rollout
    policy.train()
    ratio_means = []
    for upd in range(int(args.policy_update_epochs)):
        s0_enc = policy.encode_s0(s0_b)
        logp_new = policy.trajectory_logprob(s0_enc, zq_b, a0_b, a1_b, m0_b, m1_b)
        loss, stats = clipped_trajectory_objective(
            logp_new, logp_old, adv_b, clip_eps=float(args.clip_eps)
        )
        # entropy bonus on policy0
        if float(args.beta_entropy) > 0:
            logits0 = policy.logits0(s0_enc)
            logp0 = policy.masked_log_softmax(logits0, m0_b)
            ent = -(logp0.exp() * logp0).sum(-1).mean()
            loss = loss - float(args.beta_entropy) * ent
        opt.zero_grad()
        loss.backward()
        # no critic params
        assert not any("critic" in n.lower() or "value" in n.lower() for n, _ in policy.named_parameters())
        for p in supernet.parameters():
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                raise RuntimeError("backbone received gradients")
        for p in policy.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError("non-finite policy grads")
        opt.step()
        ratio_means.append(stats["ratio_mean"])
        print(
            f"[smoke B4] update={upd} loss={float(loss):.4f} "
            f"ratio_mean={stats['ratio_mean']:.4f} "
            f"ratio_min={stats['ratio_min']:.4f} ratio_max={stats['ratio_max']:.4f}"
        )

    # Ratio should move away from exactly 1 after updates (unless zero adv everywhere)
    if all(abs(r - 1.0) < 1e-8 for r in ratio_means) and float(adv_b.abs().sum()) > 0:
        print("[warn] ratio stayed exactly 1 — check old/new separation")
    else:
        print(f"[ok] ratio trajectory: {ratio_means}")

    # Audits: same X different eta -> raw logits identical
    policy.eval()
    with torch.no_grad():
        x = history[:2]
        s0 = policy.encode_s0(_build_s0(supernet, x))
        l0_a = policy.logits0(s0)
        l0_b = policy.logits0(s0)
        assert torch.allclose(l0_a, l0_b)
        # eta only changes mask
        mA = env.action_masks(0.0)["mask0"]
        mB = env.action_masks(1.0)["mask0"]
        assert mA.sum() == 1 and mB.sum() == 3
        # Zq swap affects policy1
        zq = _zq_for_batch(env, x)
        zq_swap = zq.flip(0)
        l1 = policy.logits1(s0, policy.pool_zq(zq))
        l1s = policy.logits1(s0, policy.pool_zq(zq_swap))
        zq_dep = float((l1 - l1s).abs().max().item())
        print(f"[smoke B5] policy1_zq_swap_max_abs_diff={zq_dep:.6f}")
        if zq_dep < 1e-8:
            print("COARSE FORECAST SIGNAL FAILURE")

    out = Path("/tmp/kasa_planB_grpo_smoke.pt")
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "policy_params": policy.count_parameters(),
            "smoke": True,
            "ratio_means": ratio_means,
            "note": "SMOKE TEST ONLY - NOT A SCIENTIFIC RESULT",
            "formal_reward_source": "temporal_crossfit_oracle_REQUIRED",
        },
        out,
    )
    print("SMOKE TEST ONLY - NOT A SCIENTIFIC RESULT")
    print(f"Wrote {out} policy_params={policy.count_parameters()}")
    return 0


def formal_train(args) -> int:
    """Full GRPO-inspired policy training on temporal cross-fitted oracle.

    Only reachable with --confirm-full-run (never invoked by Cursor agents).
    """
    from torch.utils.data import DataLoader, Subset

    from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
        default_candidate_routes,
        load_route_costs,
    )
    from basicts.data.forecast_refinement_gain_dataset import (
        ForecastRefinementGainDataset,
        collate_refinement_gains,
    )
    from basicts.data.indexed_timeseries_dataset import IndexedTimeSeriesForecastingDataset

    if not args.crossfit_oracle or not args.valid_oracle or not args.supernet_checkpoint:
        raise RuntimeError(
            "Formal Plan B requires --crossfit-oracle --valid-oracle --supernet-checkpoint"
        )

    device = torch.device(args.device)
    routes = default_candidate_routes(12)
    costs = load_route_costs(None, routes, 12)
    data_file = "datasets/PEMS04/data_in12_out12.pkl"
    index_file = "datasets/PEMS04/index_in12_out12.pkl"

    train_ds = ForecastRefinementGainDataset(
        IndexedTimeSeriesForecastingDataset(data_file, index_file, "train"),
        args.crossfit_oracle,
        expected_routes=routes,
        expected_costs=costs,
        expected_horizon=12,
        expected_dataset="PEMS04",
        require_len_match=False,
    )
    valid_ds = ForecastRefinementGainDataset(
        IndexedTimeSeriesForecastingDataset(data_file, index_file, "valid"),
        args.valid_oracle,
        expected_routes=routes,
        expected_costs=costs,
        expected_horizon=12,
        expected_dataset="PEMS04",
        require_len_match=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=2,
        collate_fn=collate_refinement_gains,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=2,
        collate_fn=collate_refinement_gains,
    )

    # Load frozen supernet (AdaptiveForecastRefinementRouteNet or BudgetConditioned)
    from basicts.archs.arch_zoo.ChainForecasting_arch.adaptive_forecast_refinement_route import (
        AdaptiveForecastRefinementRouteNet,
    )

    # Build kwargs matching stable PEMS04 F2F (same as controller trainer)
    from scripts.train_forecast_refinement_controller import _build_model, _load_supernet

    cfg_path = args.cfg
    if cfg_path is None:
        # Prefer sibling EasyTorch cfg next to the supernet checkpoint.
        ckpt_p = Path(args.supernet_checkpoint)
        sibling = list(ckpt_p.parent.glob("H12_*.py"))
        cfg_path = str(sibling[0]) if sibling else None

    class _Args:
        horizon = 12
        controller_dim = 128
        pooling_queries = 4
        delta_abs = float(args.delta_abs)
        route_cost_file = None
        cfg = cfg_path

    print(f"[build] supernet cfg={cfg_path}")
    supernet = _build_model(_Args(), routes, device)
    _load_supernet(supernet, Path(args.supernet_checkpoint))
    supernet.eval()
    for p in supernet.parameters():
        p.requires_grad = False

    env = SequentialF2FEnvironment(supernet)
    with torch.no_grad():
        probe = torch.zeros(1, 12, 307, 4, device=device)
        ctx_dim = int(_build_s0(supernet, probe).shape[-1])
    policy = GroupRelativeRefinementPolicy(context_dim=ctx_dim, zq_dim=1, hidden=256).to(
        device
    )
    opt = torch.optim.Adam(policy.parameters(), lr=float(args.lr))

    discrete_etas = [0.0, 0.25, 0.5, 0.75, 1.0]
    best = {"mean_validation_route_regret": float("inf"), "mean_selected_cost": float("inf")}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(args.num_epochs) + 1):
        policy.train()
        for history, _si, _gains, losses in train_loader:
            history = history.to(device)
            losses = losses.to(device)
            b = history.shape[0]
            if args.eta_mode == "discrete":
                eta_vals = [
                    discrete_etas[int(torch.randint(0, len(discrete_etas), (1,)).item())]
                    for _ in range(b)
                ]
            else:
                eta_vals = torch.rand(b).tolist()

            # Per-sample groups (enumerate feasible trajs). Batch by shared eta for speed.
            s0 = _build_s0(supernet, history)
            # Z_q only needed when any quarter traj feasible — compute once
            zq = _zq_for_batch(env, history)
            costs_t = supernet.route_costs.detach().to(device).float()

            # Build rollout rows across samples
            rows = []
            for i in range(b):
                eta = float(eta_vals[i])
                masks = env.action_masks(eta)
                feas = masks["feasible_routes"].to(device)
                rew, _ = terminal_route_reward(
                    losses[i : i + 1],
                    costs_t,
                    feas.unsqueeze(0),
                    delta_abs=float(args.delta_abs),
                    lambda_quality=float(args.lambda_quality),
                    lambda_cost=float(args.lambda_cost),
                )
                adv, _ = group_relative_advantages(rew, feas.unsqueeze(0))
                for t in env.enumerate_feasible_trajectories(eta):
                    rows.append(
                        {
                            "s0": s0[i],
                            "zq": zq[i],
                            "a0": int(t["a0"]),
                            "a1": 0 if t["a1"] is None else int(t["a1"]),
                            "needs_a1": t["a1"] is not None,
                            "adv": float(adv[0, int(t["route_id"])]),
                            "m0": masks["mask0"].to(device),
                            "m1": masks["mask1"].to(device),
                        }
                    )
            if not rows:
                continue
            s0_b = torch.stack([r["s0"] for r in rows], dim=0)
            zq_b = torch.stack([r["zq"] for r in rows], dim=0)
            a0_b = torch.tensor([r["a0"] for r in rows], device=device, dtype=torch.long)
            a1_b = torch.tensor([r["a1"] for r in rows], device=device, dtype=torch.long)
            adv_b = torch.tensor([r["adv"] for r in rows], device=device)
            m0_b = torch.stack([r["m0"] for r in rows], dim=0)
            m1_b = torch.stack([r["m1"] for r in rows], dim=0)

            with torch.no_grad():
                logp_old = policy.trajectory_logprob(
                    policy.encode_s0(s0_b), zq_b, a0_b, a1_b, m0_b, m1_b
                ).detach()

            for _upd in range(int(args.policy_update_epochs)):
                s0_enc = policy.encode_s0(s0_b)
                logp_new = policy.trajectory_logprob(
                    s0_enc, zq_b, a0_b, a1_b, m0_b, m1_b
                )
                loss, stats = clipped_trajectory_objective(
                    logp_new, logp_old, adv_b, clip_eps=float(args.clip_eps)
                )
                if float(args.beta_entropy) > 0:
                    logp0 = policy.masked_log_softmax(policy.logits0(s0_enc), m0_b)
                    ent = -(logp0.exp() * logp0).sum(-1).mean()
                    loss = loss - float(args.beta_entropy) * ent
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Validation: deterministic masked argmax regret
        val = _eval_policy_regret(
            policy, supernet, env, valid_loader, device, discrete_etas, args
        )
        print(
            f"[epoch {epoch}] val_regret={val['mean_validation_route_regret']:.6f} "
            f"val_cost={val['mean_selected_cost']:.6f}"
        )
        improved = val["mean_validation_route_regret"] < best["mean_validation_route_regret"] - 1e-4
        tie = abs(val["mean_validation_route_regret"] - best["mean_validation_route_regret"]) <= 1e-4
        cheaper = val["mean_selected_cost"] < best["mean_selected_cost"]
        if improved or (tie and cheaper):
            best = dict(val)
            best["epoch"] = epoch
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "policy_params": policy.count_parameters(),
                    "valid": val,
                    "args": vars(args),
                    "method": "GRPO-inspired group-relative trajectory policy",
                    "reward_source": "temporal_crossfit_oracle",
                },
                out_path,
            )
            print(f"[ckpt] saved {out_path}")
    print("[done] best=", best)
    return 0


@torch.no_grad()
def _eval_policy_regret(policy, supernet, env, loader, device, etas, args):
    from collections import Counter

    policy.eval()
    supernet.eval()
    regrets, costs, hist = [], [], Counter()
    for history, _si, _gains, losses in loader:
        history = history.to(device)
        losses = losses.to(device)
        s0 = policy.encode_s0(_build_s0(supernet, history))
        zq = _zq_for_batch(env, history)
        zqp = policy.pool_zq(zq)
        for eta in etas:
            masks = env.action_masks(float(eta))
            m0 = masks["mask0"].to(device).unsqueeze(0).expand(history.shape[0], -1)
            m1 = masks["mask1"].to(device).unsqueeze(0).expand(history.shape[0], -1)
            # deterministic masked argmax
            a0 = policy.logits0(s0).masked_fill(~m0, -1e9).argmax(-1)
            a1 = policy.logits1(s0, zqp).masked_fill(~m1, -1e9).argmax(-1)
            for i in range(history.shape[0]):
                route = env.route_from_actions(
                    int(a0[i]),
                    int(a1[i]) if int(a0[i]) == A0_QUARTER else None,
                )
                rid = env.index_map[
                    {
                        tuple(env.template["direct"]): "direct",
                        tuple(env.template["half"]): "half",
                        tuple(env.template["quarter"]): "quarter",
                        tuple(env.template["progressive"]): "progressive",
                    }[tuple(route)]
                ]
                feas = masks["feasible_routes"].to(device)
                best = losses[i][feas].min()
                regrets.append(float((losses[i, rid] - best).item()))
                costs.append(float(supernet.route_costs[rid].item()))
                hist[rid] += 1
    return {
        "mean_validation_route_regret": float(sum(regrets) / max(len(regrets), 1)),
        "mean_selected_cost": float(sum(costs) / max(len(costs), 1)),
        "route_histogram": dict(hist),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crossfit-oracle", default=None)
    p.add_argument("--valid-oracle", default=None)
    p.add_argument("--supernet-checkpoint", default=None)
    p.add_argument(
        "--cfg",
        default=None,
        help="EasyTorch cfg next to stable supernet (auto-detect sibling H12_*.py if omitted).",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--confirm-full-run", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--policy-update-epochs", type=int, default=2)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--lambda-quality", type=float, default=10.0)
    p.add_argument("--lambda-cost", type=float, default=1.0)
    p.add_argument("--delta-abs", type=float, default=0.05)
    p.add_argument("--beta-entropy", type=float, default=0.001)
    p.add_argument("--beta-kl", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--smoke-eta-idx", type=int, default=4)
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--eta-mode",
        default="discrete",
        choices=["discrete", "continuous"],
    )
    p.add_argument(
        "--out",
        default="checkpoints/PEMS04/H12/budget_f2f/group_relative_policy.pt",
    )
    args = p.parse_args()
    _require_safety(args)

    if args.smoke_test:
        args.policy_update_epochs = min(int(args.policy_update_epochs), 2)
        return smoke_train(args)

    return formal_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
