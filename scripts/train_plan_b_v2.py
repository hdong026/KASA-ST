#!/usr/bin/env python3
"""Train Plan B-v2: Exact Full-Information Sequential Forecast Refinement Policy.

NOT original GRPO/GSPO. Uses exact terminal expected utility with mean-centered
advantages, global utility scale, dual-view consistency, and real proximal KL.

Formal training requires --confirm-full-run (never invoked by Cursor agents).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.budget_route_utils import (
    default_candidate_routes,
    load_route_costs,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.exact_trajectory_policy_objective import (
    action_masks_from_feasible,
    compute_global_utility_scale,
    exact_expected_utility,
    exact_terminal_route_probs,
    mean_centered_advantages,
    rewards_from_losses,
    scale_advantages,
    terminal_entropy,
    terminal_kl,
    unique_nontrivial_feasibility_regimes,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.forecast_refinement_routes import (
    build_refinement_route_index_map,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.group_relative_refinement_policy_v2 import (
    GroupRelativeRefinementPolicyV2,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.plan_b_v2_state_cache import (
    PlanBV2StateCache,
    build_dual_view_cache,
)


class CachedOOFDataset(Dataset):
    def __init__(self, cache: PlanBV2StateCache, sample_indices: list[int] | None = None):
        self.cache = cache
        self.indices = sample_indices or cache.sample_indices()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        rec = self.cache.get(self.indices[i])
        return (
            rec["H_teacher"].float(),
            rec["Zq_teacher"].float(),
            rec["H_stable"].float(),
            rec["Zq_stable"].float(),
            rec["route_losses"].float(),
            int(rec["fold_id"]),
            int(rec["sample_index"]),
        )


def collate_cache(batch):
    Ht, Zt, Hs, Zs, losses, folds, sis = zip(*batch)
    return (
        torch.stack(Ht),
        torch.stack(Zt),
        torch.stack(Hs),
        torch.stack(Zs),
        torch.stack(losses),
        torch.tensor(folds, dtype=torch.long),
        torch.tensor(sis, dtype=torch.long),
    )


def _require_safety(args) -> None:
    if args.smoke_test or args.build_cache_only:
        return
    if not args.confirm_full_run:
        raise RuntimeError("Full Plan B-v2 training disabled. Pass --confirm-full-run.")


def compute_utility_scale_from_cache(
    cache: PlanBV2StateCache,
    costs: torch.Tensor,
    regimes: list[dict],
    *,
    delta_abs: float,
    lambda_quality: float,
    lambda_cost: float,
) -> float:
    """TRAIN OOF only — average centered utilities over nontrivial regimes."""
    centered_all = []
    feas_all = []
    for si in cache.sample_indices():
        rec = cache.get(si)
        losses = rec["route_losses"].float().unsqueeze(0)
        for reg in regimes:
            feas = reg["feasible_mask"].unsqueeze(0)
            rew = rewards_from_losses(
                losses, costs, feas,
                delta_abs=delta_abs, lambda_quality=lambda_quality, lambda_cost=lambda_cost,
            )
            adv, _ = mean_centered_advantages(rew, feas)
            centered_all.append(adv)
            feas_all.append(feas)
    centered = torch.cat(centered_all, dim=0)
    feas = torch.cat(feas_all, dim=0)
    return compute_global_utility_scale(centered, feas)


def batch_route_probs(policy, H, Zq, feas, index_map):
    return policy.forward_terminal_probs(H, Zq, feas, index_map)["route_probs"]


def train_step(
    policy,
    policy_old,
    batch,
    regimes,
    costs,
    index_map,
    *,
    utility_scale: float,
    lambda_view: float,
    beta_kl: float,
    beta_entropy: float,
    delta_abs: float,
    lambda_quality: float,
    lambda_cost: float,
):
    Ht, Zt, Hs, Zs, losses, _folds, _sis = batch
    device = Ht.device
    costs_t = costs.to(device)

    # Encode once per view (reuse across regimes)
    s0_t = policy.encode_state0(Ht)
    z_t = policy.encode_zq(Zt)
    s0_s = policy.encode_state0(Hs)
    z_s = policy.encode_zq(Zs)
    hidden_t, zqh_t = s0_t["state0_hidden"], z_t["zq_hidden"]
    hidden_s, zqh_s = s0_s["state0_hidden"], z_s["zq_hidden"]

    if policy_old is not None and beta_kl > 0:
        with torch.no_grad():
            s0_old = policy_old.encode_state0(Ht)
            z_old = policy_old.encode_zq(Zt)
            hidden_old, zqh_old = s0_old["state0_hidden"], z_old["zq_hidden"]
    else:
        hidden_old = zqh_old = None

    j_teacher_list = []
    j_stable_list = []
    kl_view_list = []
    kl_old_list = []
    ent_list = []

    for reg in regimes:
        feas = reg["feasible_mask"].to(device).unsqueeze(0).expand(Ht.shape[0], -1)
        feas_1d = reg["feasible_mask"].to(device)
        rew = rewards_from_losses(
            losses,
            costs_t,
            feas,
            delta_abs=delta_abs,
            lambda_quality=lambda_quality,
            lambda_cost=lambda_cost,
        )
        adv_c, _ = mean_centered_advantages(rew, feas)
        adv = scale_advantages(adv_c, utility_scale)

        masks = action_masks_from_feasible(feas_1d, index_map=index_map)
        m0 = masks["mask0"].to(device)
        m1 = masks["mask1"].to(device)

        def _probs(hidden, zqh):
            log0 = policy.masked_log_softmax(policy.logits0(hidden), m0)
            log1 = policy.masked_log_softmax(policy.logits1(hidden, zqh), m1)
            return exact_terminal_route_probs(
                log0, log1, m0, m1, index_map=index_map, n_routes=int(feas_1d.numel())
            )

        p_t = _probs(hidden_t, zqh_t)
        p_s = _probs(hidden_s, zqh_s)
        j_teacher_list.append(exact_expected_utility(p_t, adv, feas))
        j_stable_list.append(exact_expected_utility(p_s, adv, feas))
        kl_view_list.append(terminal_kl(p_t.detach(), p_s, feas))
        ent_list.append(terminal_entropy(p_t, feas))

        if hidden_old is not None:
            with torch.no_grad():
                log0_o = policy_old.masked_log_softmax(policy_old.logits0(hidden_old), m0)
                log1_o = policy_old.masked_log_softmax(
                    policy_old.logits1(hidden_old, zqh_old), m1
                )
                p_old = exact_terminal_route_probs(
                    log0_o, log1_o, m0, m1, index_map=index_map, n_routes=int(feas_1d.numel())
                )
            kl_old_list.append(terminal_kl(p_old, p_t, feas))

    j_teacher = torch.stack(j_teacher_list).mean()
    j_stable = torch.stack(j_stable_list).mean()
    l_view = torch.stack(kl_view_list).mean()
    ent = torch.stack(ent_list).mean()
    l_kl = torch.stack(kl_old_list).mean() if kl_old_list else torch.zeros((), device=device)

    loss = (
        -j_teacher
        + float(lambda_view) * l_view
        + float(beta_kl) * l_kl
        - float(beta_entropy) * ent
    )
    stats = {
        "loss": float(loss.detach().item()),
        "J_teacher": float(j_teacher.detach().item()),
        "J_stable": float(j_stable.detach().item()),
        "L_view": float(l_view.detach().item()),
        "L_kl": float(l_kl.detach().item()),
        "entropy": float(ent.detach().item()),
    }
    return loss, stats


@torch.no_grad()
def eval_regret_nontrivial(
    policy,
    loader,
    regimes,
    costs,
    index_map,
    device,
    *,
    view: str = "stable",
):
    policy.eval()
    regrets, costs_sel, hist = [], [], Counter()
    for batch in loader:
        Ht, Zt, Hs, Zs, losses, folds, sis = [x.to(device) if torch.is_tensor(x) else x for x in batch]
        H = Hs if view == "stable" else Ht
        Z = Zs if view == "stable" else Zt
        for reg in regimes:
            feas = reg["feasible_mask"].to(device)
            out = policy.select_actions_deterministic(H, Z, feas, index_map)
            rids = out["route_ids"]
            for i in range(H.shape[0]):
                rid = int(rids[i].item())
                m = feas
                best = losses[i][m].min()
                regrets.append(float((losses[i, rid] - best).item()))
                costs_sel.append(float(costs[rid].item()))
                hist[rid] += 1
    return {
        "mean_regret_nontrivial": float(sum(regrets) / max(len(regrets), 1)),
        "mean_selected_cost": float(sum(costs_sel) / max(len(costs_sel), 1)),
        "route_histogram": dict(hist),
        "view": view,
    }


def formal_train(args) -> int:
    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)
    if not (cache_dir / "manifest.json").is_file():
        print("[cache] building full dual-view OOF cache...")
        build_dual_view_cache(
            oracle_path=args.crossfit_oracle,
            stable_ckpt=args.supernet_checkpoint,
            out_dir=cache_dir,
            device=device,
            max_per_fold=None,
            time_budget_sec=None,
            use_fp16=True,
        )
    cache = PlanBV2StateCache(cache_dir)
    routes = default_candidate_routes(12)
    costs = torch.tensor(load_route_costs(None, routes, 12), dtype=torch.float32)
    index_map = build_refinement_route_index_map(routes, 12)
    regimes = unique_nontrivial_feasibility_regimes(costs)
    print(f"[regimes] {[ (r['name'], r['n_feasible'], r['example_eta']) for r in regimes ]}")

    utility_scale = compute_utility_scale_from_cache(
        cache, costs, regimes,
        delta_abs=float(args.delta_abs),
        lambda_quality=float(args.lambda_quality),
        lambda_cost=float(args.lambda_cost),
    )
    print(f"[utility_scale] {utility_scale:.8f}")

    # probe dims
    rec0 = cache.get(cache.sample_indices()[0])
    h_dim = int(rec0["H_teacher"].shape[-1])
    z_ch = int(rec0["Zq_teacher"].shape[-1])
    policy = GroupRelativeRefinementPolicyV2(h_dim=h_dim, z_channels=z_ch).to(device)
    opt = torch.optim.AdamW(
        policy.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    ds = CachedOOFDataset(cache)
    loader = DataLoader(
        ds, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_cache, num_workers=0
    )

    # VALID uses stable view + valid oracle losses — loaded separately if provided
    best = {"mean_regret_nontrivial": float("inf"), "mean_selected_cost": float("inf")}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    policy_old = None
    batch_i = 0

    for epoch in range(1, int(args.num_epochs) + 1):
        policy.train()
        # snapshot old policy once per epoch (before updates)
        policy_old = copy.deepcopy(policy).eval()
        for p in policy_old.parameters():
            p.requires_grad = False

        for batch in loader:
            batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            loss, stats = train_step(
                policy, policy_old, batch, regimes, costs, index_map,
                utility_scale=utility_scale,
                lambda_view=float(args.lambda_view),
                beta_kl=float(args.beta_kl),
                beta_entropy=float(args.beta_entropy),
                delta_abs=float(args.delta_abs),
                lambda_quality=float(args.lambda_quality),
                lambda_cost=float(args.lambda_cost),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # clip_grad_norm_ returns the pre-clip total norm; do NOT backward twice.
            gn_before = float(
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float(args.grad_clip)
                ).item()
            )
            gn_after = min(gn_before, float(args.grad_clip))
            opt.step()
            if batch_i % 50 == 0:
                print(
                    f"[train] epoch={epoch} "
                    f"loss={stats['loss']:.4f} J_t={stats['J_teacher']:.4f} "
                    f"gn_before={gn_before:.4f} gn_after={gn_after:.4f} "
                    f"L_view={stats['L_view']:.4f} L_kl={stats['L_kl']:.4f} "
                    f"H={stats['entropy']:.4f}"
                )
            batch_i += 1

        # selection on train cache stable-view nontrivial (formal VALID path in eval script)
        val_loader = DataLoader(
            ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_cache
        )
        val = eval_regret_nontrivial(
            policy, val_loader, regimes, costs.to(device), index_map, device, view="stable"
        )
        print(
            f"[epoch {epoch}] nontrivial_regret={val['mean_regret_nontrivial']:.6f} "
            f"cost={val['mean_selected_cost']:.6f} hist={val['route_histogram']}"
        )
        improved = val["mean_regret_nontrivial"] < best["mean_regret_nontrivial"] - 1e-4
        tie = abs(val["mean_regret_nontrivial"] - best["mean_regret_nontrivial"]) <= 1e-4
        cheaper = val["mean_selected_cost"] < best["mean_selected_cost"]
        if improved or (tie and cheaper):
            best = dict(val)
            best["epoch"] = epoch
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "policy_params": policy.count_parameters(),
                    "utility_scale": utility_scale,
                    "regimes": [
                        {
                            "name": r["name"],
                            "n_feasible": r["n_feasible"],
                            "example_eta": r["example_eta"],
                            "feasible_mask": r["feasible_mask"].tolist(),
                        }
                        for r in regimes
                    ],
                    "args": vars(args),
                    "method": "PlanB-v2 Exact Full-Information Trajectory Policy Optimization",
                    "not_grpo_gspo": True,
                    "reward_source": "temporal_crossfit_oracle",
                    "valid": val,
                },
                out_path,
            )
            print(f"[ckpt] saved {out_path}")
    print("[done] best=", best)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirm-full-run", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--build-cache-only", action="store_true")
    p.add_argument("--crossfit-oracle", default="results/pems04_temporal_crossfit_refinement_oracle.json")
    p.add_argument("--valid-oracle", default="results/pems04_budget_f2f_oracle_valid_rawscale.json")
    p.add_argument(
        "--supernet-checkpoint",
        default=(
            "checkpoints/PEMS04/H12/budget_f2f/"
            "supernet_eta0p50_dynamic_fair_rawscale_loss_v2_60f53aa1c6/seed1/"
            "b5678fda5e8d94ed028c6c8bb073461d/BudgetConditionedAdaptiveF2FNet_best_val_MAE.pt"
        ),
    )
    p.add_argument("--cache-dir", default="results/planB_v2_oof_state_cache")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--delta-abs", type=float, default=0.05)
    p.add_argument("--lambda-quality", type=float, default=10.0)
    p.add_argument("--lambda-cost", type=float, default=1.0)
    p.add_argument("--lambda-view", type=float, default=0.5)
    p.add_argument("--beta-kl", type=float, default=0.05)
    p.add_argument("--beta-entropy", type=float, default=0.005)
    p.add_argument(
        "--out",
        default="checkpoints/PEMS04/H12/budget_f2f/plan_b_v2_exact_policy.pt",
    )
    p.add_argument("--max-per-fold-cache", type=int, default=None)
    args = p.parse_args()
    _require_safety(args)

    if args.build_cache_only:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        info = build_dual_view_cache(
            oracle_path=args.crossfit_oracle,
            stable_ckpt=args.supernet_checkpoint,
            out_dir=args.cache_dir,
            device=device,
            max_per_fold=args.max_per_fold_cache,
            time_budget_sec=None if args.confirm_full_run else 120.0,
        )
        print(json.dumps({k: info[k] for k in info if k != "teachers"}, indent=2, default=str))
        return 0

    if args.smoke_test:
        print("Plan B-v2 smoke: use scripts/audit_plan_b_v2.py for diagnostics")
        return 0

    return formal_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
