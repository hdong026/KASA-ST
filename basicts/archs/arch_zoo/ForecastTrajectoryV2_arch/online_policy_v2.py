"""V2 online policy: prefix-DAG exact expectation, no probability leakage."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_graph import ForecastTrajectoryGraph


class OnlineTrajectoryPolicyV2(nn.Module):
    def __init__(self, graph: ForecastTrajectoryGraph, d_model: int = 160, hidden: int = 128):
        super().__init__()
        self.graph = graph
        self.dest_states = list(graph.states)
        self.n_dest = len(self.dest_states)
        self.state_to_index = {int(s): i for i, s in enumerate(self.dest_states)}
        self.node_query_h = nn.Parameter(torch.randn(d_model) * 0.02)
        self.node_query_z = nn.Parameter(torch.randn(d_model) * 0.02)
        self.h_proj = nn.Linear(d_model, hidden)
        self.z_proj = nn.Linear(1, d_model)
        self.hx_mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.SiLU())
        self.hc_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        self.zc_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        self.state_emb = nn.Embedding(len(graph.nodes), hidden)
        self.ctx = nn.Sequential(nn.Linear(3, hidden), nn.SiLU())
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_dest),
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _pool_nodes(self, tok: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # tok: [B, T, N, D] or [B, N, D]
        if tok.ndim == 3:
            tok = tok.unsqueeze(1)
        scores = (tok * query.view(1, 1, 1, -1)).sum(-1)  # [B,T,N]
        w = torch.softmax(scores.mean(dim=1), dim=-1)  # [B,N]
        pooled = torch.einsum("bn,bnd->bd", w, tok.mean(dim=1))
        return pooled

    def encode(
        self,
        h_x: torch.Tensor,
        h_cur: Optional[torch.Tensor],
        z_cur: Optional[torch.Tensor],
        s_current: int,
        lam: torch.Tensor,
        remaining_norm: torch.Tensor,
        no_budget: torch.Tensor,
    ) -> torch.Tensor:
        b = h_x.shape[0]
        device, dtype = h_x.device, h_x.dtype
        hx = self.hx_mlp(h_x)
        if h_cur is None:
            hc = hx.new_zeros(b, hx.shape[-1])
        else:
            hc = self.hc_mlp(self.h_proj(self._pool_nodes(h_cur, self.node_query_h)))
        if z_cur is None:
            zc = hx.new_zeros(b, hx.shape[-1])
        else:
            zt = self.z_proj(z_cur)
            zc = self.zc_mlp(self.h_proj(self._pool_nodes(zt, self.node_query_z)))
        s_idx = 0 if int(s_current) == 0 else 1 + self.graph.states.index(int(s_current)) if int(s_current) in self.graph.states else 0
        se = self.state_emb(torch.full((b,), int(s_idx), device=device, dtype=torch.long))
        ctx = self.ctx(
            torch.stack(
                [lam.reshape(b), remaining_norm.reshape(b), no_budget.reshape(b)], dim=-1
            ).to(dtype)
        )
        return torch.cat([hx, hc, zc, se, ctx], dim=-1)

    def forward(
        self,
        h_x,
        h_cur,
        z_cur,
        s_current: int,
        lam,
        remaining_norm,
        no_budget,
        legal_next: Optional[list[int]] = None,
        feasible_next: Optional[torch.Tensor] = None,
    ) -> dict:
        fused = self.encode(h_x, h_cur, z_cur, s_current, lam, remaining_norm, no_budget)
        logits = self.fuse(fused)
        b = logits.shape[0]
        device = logits.device
        mask = torch.zeros(self.n_dest, dtype=torch.bool, device=device)
        succ = set(self.graph.successors(int(s_current)))
        if legal_next is not None:
            succ = succ.intersection(set(int(x) for x in legal_next))
        for s, i in self.state_to_index.items():
            mask[i] = s in succ
        mask = mask.unsqueeze(0).expand(b, -1).contiguous()
        if feasible_next is not None:
            mask = mask & feasible_next.bool()
        none = (~mask).all(dim=-1, keepdim=True)
        if none.any():
            legal = torch.zeros_like(mask)
            for s, i in self.state_to_index.items():
                if s in set(self.graph.successors(int(s_current))):
                    legal[:, i] = True
            mask = torch.where(none, legal, mask)
        neg = torch.finfo(logits.dtype).min / 4
        masked = logits.masked_fill(~mask, neg)
        probs = torch.softmax(masked, dim=-1) * mask.float()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        logp = torch.log(probs.clamp_min(1e-12))
        return {"logits": logits, "probs": probs, "log_probs": logp, "mask": mask}


def exact_prefix_dag_values(
    policy: OnlineTrajectoryPolicyV2,
    graph: ForecastTrajectoryGraph,
    h_x: torch.Tensor,
    states: dict,
    forecast_loss: torch.Tensor,
    total_ms: torch.Tensor,
    lam: torch.Tensor,
    remaining_init: Optional[torch.Tensor],
    no_budget: torch.Tensor,
    edge_cost: dict,
    min_finish: dict,
    extra_policy: float,
    c_dense: float,
) -> dict:
    """Exact V(p) by backward DP. forecast_loss/total_ms: [B, n_tau]."""
    taus = graph.terminal_trajectories()
    n_tau = len(taus)
    b = h_x.shape[0]
    device = h_x.device
    dtype = h_x.dtype
    obj = forecast_loss + lam.reshape(b, 1) * total_ms.reshape(b, n_tau)
    # terminal values indexed by trajectory
    # DP on prefixes: V[pref] [B]
    prefixes = []
    seen = set()
    for tau in taus:
        for i in range(len(tau) + 1):
            p = tuple(tau[:i])
            if p not in seen:
                seen.add(p)
                prefixes.append(p)
    prefixes.sort(key=lambda p: (len(p), p))
    V: dict[tuple[int, ...], torch.Tensor] = {}
    # terminals
    tau_index = {tuple(t): i for i, t in enumerate(taus)}
    for tau in taus:
        V[tuple(tau)] = obj[:, tau_index[tuple(tau)]]
    path_probs = h_x.new_zeros(b, n_tau)
    # store pi at each prefix
    pi_store = {}
    # forward to get pi
    for pref in prefixes:
        if pref and pref[-1] == graph.H:
            continue
        s = 0 if not pref else int(pref[-1])
        h_cur, z_cur = states.get(pref, (None, None))
        rem_norm = torch.ones(b, device=device, dtype=dtype)
        feas_mask = None
        if remaining_init is not None:
            paid = 0.0
            sp = 0
            for sx in pref:
                paid += float(edge_cost[(sp, int(sx))]) + float(extra_policy)
                sp = int(sx)
            rem = remaining_init - paid
            rem_norm = rem / max(float(c_dense), 1e-6)
            feas = torch.zeros(b, policy.n_dest, dtype=torch.bool, device=device)
            use_hard = no_budget.reshape(b) < 0.5
            for cand, idx in policy.state_to_index.items():
                if not graph.is_legal_edge(s, cand):
                    continue
                need = float(edge_cost[(s, cand)]) + float(extra_policy) + float(min_finish[cand])
                ok = rem >= (need - 1e-9)
                legal = torch.ones(b, dtype=torch.bool, device=device)
                feas[:, idx] = torch.where(use_hard, ok & legal, legal)
            feas_mask = feas
        out = policy(
            h_x, h_cur, z_cur, s, lam, rem_norm, no_budget, feasible_next=feas_mask
        )
        pi_store[pref] = out
        row_sum = out["probs"].sum(dim=-1)
        if (row_sum - 1).abs().max() > 1e-5:
            pass
    # backward
    for pref in reversed(prefixes):
        if pref in V:
            continue
        s = 0 if not pref else int(pref[-1])
        out = pi_store[pref]
        acc = h_x.new_zeros(b)
        for cand, idx in policy.state_to_index.items():
            child = pref + (int(cand),)
            if child not in V:
                continue
            acc = acc + out["probs"][:, idx] * V[child]
        V[pref] = acc
    # path probs via product
    for ti, tau in enumerate(taus):
        logp = h_x.new_zeros(b)
        pref: tuple[int, ...] = ()
        for s_next in tau:
            out = pi_store[pref]
            logp = logp + out["log_probs"][:, policy.state_to_index[int(s_next)]]
            pref = pref + (int(s_next),)
        path_probs[:, ti] = logp.exp()
    path_sum = path_probs.sum(dim=-1)
    expected = V[()]
    oracle = obj.min(dim=-1).values
    regret = expected - oracle
    return {
        "expected": expected.mean(),
        "path_probs": path_probs,
        "path_sum": path_sum,
        "regret": regret,
        "V0": V[()],
        "max_path_err": float((path_sum - 1).abs().max().item()),
        "min_regret": float(regret.min().item()),
    }
