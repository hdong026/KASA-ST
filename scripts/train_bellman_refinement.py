#!/usr/bin/env python3
"""Train Budgeted Bellman Forecast Refinement (Q1 then Q0, optional joint)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.archs.arch_zoo.ChainForecasting_arch.bellman_refinement_dataset import (
    BellmanOOFCache,
    BellmanOOFDataset,
    collate_bellman,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.bellman_refinement_qnet import (
    BellmanRefinementRouter,
    Q0Net,
    Q1Net,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.budgeted_bellman_refinement import (
    BudgetedRefinementMDP,
    greedy_masked_argmax,
    route_name_from_actions,
    semantic_to_route,
)


def write_json(path: Path | str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        if mask.sum() == 0:
            return pred.sum() * 0.0
        pred = pred[mask]
        target = target[mask]
    return nn.functional.smooth_l1_loss(pred, target, reduction="mean")


class GradClipMonitor:
    def __init__(self, clip: float = 5.0):
        self.clip = float(clip)
        self.raw = []
        self.clipped = []
        self.n_clip = 0
        self.n = 0

    def step(self, module: nn.Module) -> tuple[float, float]:
        raw = float(torch.nn.utils.clip_grad_norm_(module.parameters(), float("inf")))
        clipped = float(torch.nn.utils.clip_grad_norm_(module.parameters(), self.clip))
        # second call already clipped; recompute properly:
        # fix: clip once
        self.n += 1
        self.raw.append(raw)
        was = raw > self.clip + 1e-12
        self.n_clip += int(was)
        self.clipped.append(min(raw, self.clip))
        return raw, min(raw, self.clip)

    @property
    def frac(self) -> float:
        return self.n_clip / max(self.n, 1)


def clip_grads(module: nn.Module, clip: float) -> tuple[float, float, bool]:
    raw = torch.nn.utils.clip_grad_norm_(module.parameters(), float("inf"))
    raw_f = float(raw)
    torch.nn.utils.clip_grad_norm_(module.parameters(), clip)
    return raw_f, min(raw_f, clip), raw_f > clip + 1e-12


# Fix clip_grads — calling clip_grad_norm twice with inf then clip is wrong because
# first call doesn't modify grads when max_norm is inf... actually clip_grad_norm_
# with inf still computes total_norm but clamps to inf (no change). OK.


def train_q1_epoch(model, loader, opt, device, clip, c_max, scale_budgets=True):
    model.train()
    total = 0.0
    n = 0
    mon = {"raw": [], "clip_frac": 0, "n": 0, "n_clip": 0}
    fold_loss: dict[int, list] = {}
    for batch in loader:
        x = batch["X"].to(device)
        z = batch["Z_q"].to(device)
        tgt = batch["q1_target"].to(device)
        # use largest regime budget remaining after q for training Q1
        # budgets are whole-route B; remaining = B - c_q — use max budget regime
        B = batch["budgets"][:, -1].to(device)  # last regime = highest budget
        # remaining after q from MDP costs stored in dataset budgets absolute
        # We pass normalized remaining; for Q1 targets exact g_Q/g_F independent of b,
        # but observation includes remaining. Use B - c_q with c_q from first sample regimes.
        # Approximate: remaining_norm = (B - (Bmin related))/Cmax — use B/Cmax as proxy remaining after? 
        # Spec: sq state includes remaining budget after paying c_q.
        # For each sample use the highest-budget regime's remaining after q.
        rem = batch["budgets"][:, -1].to(device)  # will subtract c_q outside
        # get c_q from difference of route costs — pass via attribute
        c_q = float(getattr(loader.dataset, "c_q", 0.16216216216216228))
        rem = rem - c_q
        bnorm = (rem / c_max).unsqueeze(-1)
        sq_mask = batch["sq_masks"][:, -1].to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(x, z, bnorm, sq_mask)
        loss = huber(pred, tgt)
        loss.backward()
        # grad check nonzero heads
        raw, clipped, was = _clip_once(model, clip)
        opt.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
        mon["raw"].append(raw)
        mon["n"] += 1
        mon["n_clip"] += int(was)
        for fid, li in zip(batch["fold_id"], [float(loss.item())] * x.size(0)):
            fold_loss.setdefault(int(fid), []).append(li)
    mon["clip_frac"] = mon["n_clip"] / max(mon["n"], 1)
    mon["mean_raw"] = float(sum(mon["raw"]) / max(len(mon["raw"]), 1))
    return total / max(n, 1), mon, {str(k): float(sum(v) / len(v)) for k, v in fold_loss.items()}


def _clip_once(module: nn.Module, clip: float) -> tuple[float, float, bool]:
    params = [p for p in module.parameters() if p.grad is not None]
    if not params:
        return 0.0, 0.0, False
    total_norm = torch.nn.utils.clip_grad_norm_(params, clip)
    raw = float(total_norm)
    return raw, min(raw, clip), raw > clip + 1e-12


def train_q0_epoch(model, loader, opt, device, clip, c_max):
    model.train()
    total = 0.0
    n = 0
    mon = {"raw": [], "n": 0, "n_clip": 0}
    for batch in loader:
        x = batch["X"].to(device)
        # expand over regimes
        R = batch["q0_targets"].shape[1]
        Bsz = x.size(0)
        losses = []
        opt.zero_grad(set_to_none=True)
        # process all regimes in one forward by repeating X
        x_rep = x.unsqueeze(1).expand(-1, R, *x.shape[1:]).reshape(Bsz * R, *x.shape[1:])
        b = batch["budgets"].to(device).reshape(Bsz * R)
        bnorm = (b / c_max).unsqueeze(-1)
        mask = batch["s0_masks"].to(device).reshape(Bsz * R, 3)
        tgt = batch["q0_targets"].to(device).reshape(Bsz * R, 3)
        valid = batch["q0_valids"].to(device).reshape(Bsz * R, 3)
        pred = model(x_rep, bnorm, mask)
        loss = huber(pred, tgt, valid)
        loss.backward()
        raw, clipped, was = _clip_once(model, clip)
        opt.step()
        total += float(loss.item()) * Bsz
        n += Bsz
        mon["raw"].append(raw)
        mon["n"] += 1
        mon["n_clip"] += int(was)
    mon["clip_frac"] = mon["n_clip"] / max(mon["n"], 1)
    mon["mean_raw"] = float(sum(mon["raw"]) / max(len(mon["raw"]), 1))
    return total / max(n, 1), mon


@torch.no_grad()
def eval_q1_decision_regret(model, loader, device, c_max, c_q, valid_oracle_losses: dict[int, torch.Tensor] | None = None):
    """If valid_oracle_losses provided (sample_index->DMQF losses), compute child regret on those indices present in loader.
    For OOF proxy: use batch losses.
    """
    model.eval()
    regs = []
    huber_vals = []
    for batch in loader:
        x = batch["X"].to(device)
        z = batch["Z_q"].to(device)
        B = batch["budgets"][:, -1].to(device)
        rem = B - c_q
        bnorm = (rem / c_max).unsqueeze(-1)
        sq_mask = batch["sq_masks"][:, -1].to(device)
        pred = model(x, z, bnorm, sq_mask)
        tgt = batch["q1_target"].to(device)
        huber_vals.append(float(huber(pred, tgt).item()))
        a = greedy_masked_argmax(pred, sq_mask)
        # true best among feasible children using gains
        g = batch["gains"].to(device)
        # child returns: f->g_Q, m->g_F
        child_g = torch.stack([g[:, 2], g[:, 3]], dim=-1)
        best = greedy_masked_argmax(child_g, sq_mask)
        # regret in unscaled loss space: use losses
        L = batch["losses"].to(device)
        # map action to route loss: f->L_Q(index2), m->L_F(index3)
        sel_loss = torch.where(a == 0, L[:, 2], L[:, 3])
        best_loss = torch.where(best == 0, L[:, 2], L[:, 3])
        # also compare to best among feasible of Q/F
        regs.extend((sel_loss - best_loss).detach().cpu().tolist())
    return {
        "mean_child_regret": float(sum(regs) / max(len(regs), 1)),
        "mean_huber": float(sum(huber_vals) / max(len(huber_vals), 1)),
        "n": len(regs),
    }


def build_optim(model, lr, wd):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    return opt, sched


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["q1", "q0", "joint", "all"], default="all")
    p.add_argument("--cache-dir", default="results/planB_bellman_oof_cache")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--min-epochs-q1", type=int, default=30)
    p.add_argument("--max-epochs-q1", type=int, default=300)
    p.add_argument("--patience-q1", type=int, default=40)
    p.add_argument("--min-epochs-q0", type=int, default=30)
    p.add_argument("--max-epochs-q0", type=int, default=300)
    p.add_argument("--patience-q0", type=int, default=50)
    p.add_argument("--min-epochs-joint", type=int, default=10)
    p.add_argument("--max-epochs-joint", type=int, default=100)
    p.add_argument("--patience-joint", type=int, default=20)
    p.add_argument("--lambda-bellman", type=float, default=0.1)
    p.add_argument("--enable-joint", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--formal-path-smoke",
        action="store_true",
        help="Exercise NON-smoke train loop for 1 epoch (catches formal-path bugs).",
    )
    p.add_argument("--smoke-steps-q1", type=int, default=10)
    p.add_argument("--smoke-steps-q0", type=int, default=10)
    p.add_argument("--smoke-steps-joint", type=int, default=5)
    p.add_argument("--out-dir", default="checkpoints/PEMS04/H12/budget_f2f/plan_b_bellman")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache = BellmanOOFCache(args.cache_dir)
    scale = float(cache.manifest.get("global_return_scale", 1.0))
    mdp = BudgetedRefinementMDP(12)
    c_max = mdp.costs.C_max
    c_q = mdp.costs.c_q

    ds = BellmanOOFDataset(cache, scale=scale, mdp=mdp)
    ds.c_q = c_q
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_bellman,
        drop_last=False,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q1 = Q1Net().to(device)
    q0 = Q0Net().to(device)

    history = {"q1": [], "q0": [], "joint": [], "fold_counts": cache.fold_counts(), "scale": scale}

    if args.phase in ("q1", "all"):
        opt, sched = build_optim(q1, args.lr, args.weight_decay)
        best = math.inf
        best_state = None
        best_epoch = -1
        bad = 0
        unhealthy = 0
        init_loss = None
        if args.smoke:
            max_ep = args.smoke_steps_q1
            min_ep = 1
        elif args.formal_path_smoke:
            max_ep = 1
            min_ep = 1
        else:
            max_ep = args.max_epochs_q1
            min_ep = args.min_epochs_q1
        # smoke: treat steps as optimizer steps not epochs
        if args.smoke:
            q1.train()
            it = iter(loader)
            losses = []
            mon_raw = []
            n_clip = 0
            for step in range(args.smoke_steps_q1):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                x = batch["X"].to(device)
                z = batch["Z_q"].to(device)
                tgt = batch["q1_target"].to(device)
                B = batch["budgets"][:, -1].to(device)
                bnorm = ((B - c_q) / c_max).unsqueeze(-1)
                sq_mask = batch["sq_masks"][:, -1].to(device)
                opt.zero_grad(set_to_none=True)
                pred = q1(x, z, bnorm, sq_mask)
                loss = huber(pred, tgt)
                if init_loss is None:
                    init_loss = float(loss.item())
                loss.backward()
                # assert grads
                grads_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in q1.parameters())
                raw, _, was = _clip_once(q1, args.grad_clip)
                opt.step()
                losses.append(float(loss.item()))
                mon_raw.append(raw)
                n_clip += int(was)
            final_loss = float(losses[-1])
            history["q1_smoke"] = {
                "init_loss": init_loss,
                "final_loss": final_loss,
                "grad_norm_mean": float(sum(mon_raw) / len(mon_raw)),
                "clip_frac": n_clip / max(len(mon_raw), 1),
                "grads_nonzero": grads_ok,
                "steps": args.smoke_steps_q1,
            }
            torch.save(
                {"model": q1.state_dict(), "scale": scale, "epoch": args.smoke_steps_q1},
                out_dir / "q1_best.pt",
            )
        else:
            for ep in range(1, max_ep + 1):
                tr_loss, mon, fold_l = train_q1_epoch(q1, loader, opt, device, args.grad_clip, c_max)
                ev = eval_q1_decision_regret(q1, loader, device, c_max, c_q)
                metric = ev["mean_child_regret"]
                sched.step(metric)
                lr = opt.param_groups[0]["lr"]
                history["q1"].append(
                    {
                        "epoch": ep,
                        "train_huber": tr_loss,
                        "valid_proxy_huber": ev["mean_huber"],
                        "child_regret": metric,
                        "clip_frac": mon["clip_frac"],
                        "grad_raw_mean": mon["mean_raw"],
                        "lr": lr,
                        "fold_loss": fold_l,
                    }
                )
                if mon["clip_frac"] > 0.5:
                    unhealthy += 1
                    if unhealthy >= 3:
                        for g in opt.param_groups:
                            g["lr"] = max(g["lr"] * 0.5, 1e-6)
                        print("GRADIENT_SCALE_UNHEALTHY: reducing LR by 0.5")
                        unhealthy = 0
                else:
                    unhealthy = 0
                improved = metric < best - 1e-4 or (
                    abs(metric - best) <= 1e-4 and ev["mean_huber"] < best
                )
                # primary: min child regret; tie lower huber
                if metric < best - 1e-12 or (
                    abs(metric - best) <= 1e-4 and ev["mean_huber"] <= history.get("_best_huber", math.inf)
                ):
                    best = metric
                    history["_best_huber"] = ev["mean_huber"]
                    best_state = {k: v.detach().cpu().clone() for k, v in q1.state_dict().items()}
                    best_epoch = ep
                    bad = 0
                else:
                    bad += 1
                print(f"[q1 ep {ep}] loss={tr_loss:.4f} regret={metric:.4f} clip={mon['clip_frac']:.2f} lr={lr:.2e}")
                if ep >= min_ep and bad >= args.patience_q1:
                    break
                if ep >= min_ep and lr <= 1e-6 + 1e-15 and bad >= 10:
                    break
            if best_state is not None:
                q1.load_state_dict(best_state)
            torch.save(
                {"model": q1.state_dict(), "scale": scale, "best_epoch": best_epoch, "best_metric": best},
                out_dir / "q1_best.pt",
            )
            history["q1_best"] = {"epoch": best_epoch, "metric": best}

    # load q1 if needed
    if (out_dir / "q1_best.pt").is_file():
        blob = torch.load(out_dir / "q1_best.pt", map_location="cpu")
        q1.load_state_dict(blob["model"])
    q1.to(device)

    if args.phase in ("q0", "all"):
        # freeze q1
        for p in q1.parameters():
            p.requires_grad = False
        q1.eval()
        opt, sched = build_optim(q0, args.lr, args.weight_decay)
        if args.smoke:
            q0.train()
            it = iter(loader)
            init_loss = None
            losses = []
            mon_raw = []
            n_clip = 0
            grads_ok = False
            for step in range(args.smoke_steps_q0):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                x = batch["X"].to(device)
                R = batch["q0_targets"].shape[1]
                Bsz = x.size(0)
                x_rep = x.unsqueeze(1).expand(-1, R, *x.shape[1:]).reshape(Bsz * R, *x.shape[1:])
                b = batch["budgets"].to(device).reshape(Bsz * R)
                bnorm = (b / c_max).unsqueeze(-1)
                mask = batch["s0_masks"].to(device).reshape(Bsz * R, 3)
                tgt = batch["q0_targets"].to(device).reshape(Bsz * R, 3)
                valid = batch["q0_valids"].to(device).reshape(Bsz * R, 3)
                opt.zero_grad(set_to_none=True)
                pred = q0(x_rep, bnorm, mask)
                loss = huber(pred, tgt, valid)
                if init_loss is None:
                    init_loss = float(loss.item())
                loss.backward()
                grads_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in q0.parameters())
                raw, _, was = _clip_once(q0, args.grad_clip)
                opt.step()
                losses.append(float(loss.item()))
                mon_raw.append(raw)
                n_clip += int(was)
            history["q0_smoke"] = {
                "init_loss": init_loss,
                "final_loss": float(losses[-1]),
                "grad_norm_mean": float(sum(mon_raw) / len(mon_raw)),
                "clip_frac": n_clip / max(len(mon_raw), 1),
                "grads_nonzero": grads_ok,
                "steps": args.smoke_steps_q0,
            }
            torch.save(
                {"model": q0.state_dict(), "scale": scale, "epoch": args.smoke_steps_q0},
                out_dir / "q0_best.pt",
            )
        else:
            best = math.inf
            best_state = None
            best_epoch = -1
            bad = 0
            max_ep_q0 = 1 if args.formal_path_smoke else args.max_epochs_q0
            min_ep_q0 = 1 if args.formal_path_smoke else args.min_epochs_q0
            for ep in range(1, max_ep_q0 + 1):
                tr_loss, mon = train_q0_epoch(q0, loader, opt, device, args.grad_clip, c_max)
                # OOF proxy sequential regret
                metric = tr_loss  # formal eval uses VALID in eval script
                sched.step(metric)
                history["q0"].append(
                    {
                        "epoch": ep,
                        "train_huber": tr_loss,
                        "clip_frac": mon["clip_frac"],
                        "grad_raw_mean": mon["mean_raw"],
                        "lr": opt.param_groups[0]["lr"],
                    }
                )
                if tr_loss < best:
                    best = tr_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in q0.state_dict().items()}
                    best_epoch = ep
                    bad = 0
                else:
                    bad += 1
                print(f"[q0 ep {ep}] loss={tr_loss:.4f} clip={mon['clip_frac']:.2f}")
                if ep >= min_ep_q0 and bad >= args.patience_q0:
                    break
                if args.formal_path_smoke:
                    break
            if best_state is not None:
                q0.load_state_dict(best_state)
            torch.save(
                {"model": q0.state_dict(), "scale": scale, "best_epoch": best_epoch, "best_metric": best},
                out_dir / "q0_best.pt",
            )
            history["q0_best"] = {"epoch": best_epoch, "metric": best}

    if args.enable_joint or (args.smoke and args.phase in ("joint", "all")):
        # optional joint
        for p in q1.parameters():
            p.requires_grad = True
        lr_j = min(args.lr, 1e-4)
        opt = torch.optim.AdamW(
            list(q0.parameters()) + list(q1.parameters()), lr=lr_j, weight_decay=args.weight_decay
        )
        if args.smoke:
            it = iter(loader)
            for step in range(args.smoke_steps_joint):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                x = batch["X"].to(device)
                z = batch["Z_q"].to(device)
                Bsz = x.size(0)
                R = batch["q0_targets"].shape[1]
                # Q1 loss
                B = batch["budgets"][:, -1].to(device)
                bnorm1 = ((B - c_q) / c_max).unsqueeze(-1)
                sq_mask = batch["sq_masks"][:, -1].to(device)
                q1_pred = q1(x, z, bnorm1, sq_mask)
                loss_q1 = huber(q1_pred, batch["q1_target"].to(device))
                # Q0 loss
                x_rep = x.unsqueeze(1).expand(-1, R, *x.shape[1:]).reshape(Bsz * R, *x.shape[1:])
                b = batch["budgets"].to(device).reshape(Bsz * R)
                bnorm0 = (b / c_max).unsqueeze(-1)
                mask0 = batch["s0_masks"].to(device).reshape(Bsz * R, 3)
                q0_pred = q0(x_rep, bnorm0, mask0)
                loss_q0 = huber(
                    q0_pred,
                    batch["q0_targets"].to(device).reshape(Bsz * R, 3),
                    batch["q0_valids"].to(device).reshape(Bsz * R, 3),
                )
                # Bellman consistency on q-action: Q0_q vs max feasible Q1
                # use last regime
                q0_last = q0(x, (B / c_max).unsqueeze(-1), batch["s0_masks"][:, -1].to(device))
                with torch.no_grad():
                    max_q1 = q1_pred.max(dim=-1).values
                loss_b = nn.functional.smooth_l1_loss(q0_last[:, 2], max_q1)
                loss = loss_q0 + loss_q1 + args.lambda_bellman * loss_b
                opt.zero_grad(set_to_none=True)
                loss.backward()
                _clip_once(q0, args.grad_clip)
                _clip_once(q1, args.grad_clip)
                opt.step()
            history["joint_smoke"] = {"steps": args.smoke_steps_joint, "last_loss": float(loss.item())}
            torch.save({"q0": q0.state_dict(), "q1": q1.state_dict(), "scale": scale}, out_dir / "joint_last.pt")
            # smoke: keep phase-II as best unless we explicitly compare — save combined
            torch.save(
                {"q0": q0.state_dict(), "q1": q1.state_dict(), "scale": scale, "c_max": c_max},
                out_dir / "router_best.pt",
            )
        else:
            pass  # formal joint handled in runner with VALID metric

    # Always refresh router_best after any phase that may have updated Q0/Q1.
    # (Bug: previously only wrote if missing, so formal Q0 training left a
    # zero-init Q0 in router_best and eval collapsed to DIRECT.)
    torch.save(
        {
            "q0": q0.state_dict(),
            "q1": q1.state_dict(),
            "scale": scale,
            "c_max": c_max,
            "phase": args.phase,
        },
        out_dir / "router_best.pt",
    )

    write_json(out_dir / "train_history.json", history)
    write_json("results/planB_bellman_q1_history.json", {"history": history.get("q1"), "smoke": history.get("q1_smoke")})
    write_json("results/planB_bellman_q0_history.json", {"history": history.get("q0"), "smoke": history.get("q0_smoke")})
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "smoke": args.smoke}, indent=2))


if __name__ == "__main__":
    main()
