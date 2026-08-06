"""One-shot resolution program planner (single call per forward)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

MAX_OPTIONAL_INTERMEDIATE_STEPS = 2


class CallCounter:
    """Tiny helper to assert single-call invariants in tests."""

    def __init__(self):
        self.count = 0

    def tick(self) -> None:
        self.count += 1

    def reset(self) -> None:
        self.count = 0


class SharedHistoryEncoder(nn.Module):
    """Encode history once; executor/planner share outputs."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.call_counter = CallCounter()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history [B,P,N,Cx] → [B,P,N,H]."""
        self.call_counter.tick()
        return self.net(history)

    def reset_counter(self) -> None:
        self.call_counter.reset()


class OneShotResolutionProgramPlanner(nn.Module):
    """Emit full K-step resolution program logits in one forward."""

    def __init__(
        self,
        hidden_dim: int,
        n_temporal_nodes: int,
        n_spatial_nodes: int,
        k_steps: int = MAX_OPTIONAL_INTERMEDIATE_STEPS,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_temporal_nodes = int(n_temporal_nodes)
        self.n_spatial_nodes = int(n_spatial_nodes)
        self.k_steps = int(k_steps)
        self.temperature = float(temperature)
        self.call_counter = CallCounter()

        self.history_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.intensity_emb = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.step_query = nn.Embedding(max(k_steps, 1), hidden_dim)
        # Tree node scoring: concat global + node meta(6) + step
        self.temporal_score = nn.Sequential(
            nn.Linear(hidden_dim + 6 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.spatial_score = nn.Sequential(
            nn.Linear(hidden_dim + 6 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.budget_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Mild positive bias so early training prefers refinement
        nn.init.constant_(self.temporal_score[-1].bias, 0.5)
        nn.init.constant_(self.spatial_score[-1].bias, 0.5)
        nn.init.constant_(self.continue_head[-1].bias, 0.0)

    def reset_counter(self) -> None:
        self.call_counter.reset()

    def forward(
        self,
        encoded_history: torch.Tensor,
        temporal_node_meta: torch.Tensor,
        spatial_node_meta: torch.Tensor,
        thinking_intensity: float | torch.Tensor,
        total_optional_budget: torch.Tensor,
        temporal_leaf_mask: torch.Tensor,
        spatial_leaf_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Args:
            encoded_history: [B,P,N,H]
            temporal_node_meta: [Tnodes, 6]
            spatial_node_meta: [Snodes, 6]
            total_optional_budget: [B]
            temporal/spatial_leaf_mask: [Tnodes]/[Snodes] bool (cannot split leaves)
        """
        self.call_counter.tick()
        b = encoded_history.shape[0]
        device = encoded_history.device
        dtype = encoded_history.dtype
        # Global history summary
        g = encoded_history.mean(dim=(1, 2))  # [B,H]
        g = self.history_proj(g)
        if not torch.is_tensor(thinking_intensity):
            intens = encoded_history.new_full((b,), float(thinking_intensity))
        else:
            intens = thinking_intensity.to(device=device, dtype=dtype).expand(b)
        budget = total_optional_budget.to(device=device, dtype=dtype)
        if budget.ndim == 0:
            budget = budget.expand(b)
        ib = self.intensity_emb(torch.stack([intens, budget], dim=-1))
        global_ctx = g + ib  # [B,H]

        t_logits = []
        s_logits = []
        c_logits = []
        budgets = []
        expected_costs = []

        t_meta = temporal_node_meta.to(device=device, dtype=dtype)
        s_meta = spatial_node_meta.to(device=device, dtype=dtype)
        t_leaf = temporal_leaf_mask.to(device=device)
        s_leaf = spatial_leaf_mask.to(device=device)

        for k in range(self.k_steps):
            q = self.step_query(
                torch.tensor(k, device=device, dtype=torch.long)
            ).unsqueeze(0).expand(b, -1)
            # Temporal scores
            g_t = global_ctx.unsqueeze(1).expand(-1, self.n_temporal_nodes, -1)
            q_t = q.unsqueeze(1).expand(-1, self.n_temporal_nodes, -1)
            tm = t_meta.unsqueeze(0).expand(b, -1, -1)
            t_in = torch.cat([g_t, tm, q_t], dim=-1)
            tl = self.temporal_score(t_in).squeeze(-1) / max(self.temperature, 1e-4)
            # Mask leaves
            tl = tl.masked_fill(t_leaf.unsqueeze(0), -1e4)
            # Spatial scores
            g_s = global_ctx.unsqueeze(1).expand(-1, self.n_spatial_nodes, -1)
            q_s = q.unsqueeze(1).expand(-1, self.n_spatial_nodes, -1)
            sm = s_meta.unsqueeze(0).expand(b, -1, -1)
            s_in = torch.cat([g_s, sm, q_s], dim=-1)
            sl = self.spatial_score(s_in).squeeze(-1) / max(self.temperature, 1e-4)
            sl = sl.masked_fill(s_leaf.unsqueeze(0), -1e4)

            cont_in = torch.cat([global_ctx, q], dim=-1)
            cl = self.continue_head(cont_in).squeeze(-1)
            bal = torch.softmax(self.budget_head(cont_in), dim=-1)  # [B,2] T/S alloc
            # Expected cost proxy from soft split mass
            soft_t = torch.sigmoid(tl) * (~t_leaf).unsqueeze(0).to(dtype)
            soft_s = torch.sigmoid(sl) * (~s_leaf).unsqueeze(0).to(dtype)
            exp_cost = 0.05 + 0.1 * soft_t.mean(-1) + 0.1 * soft_s.mean(-1)

            t_logits.append(tl)
            s_logits.append(sl)
            c_logits.append(cl)
            budgets.append(bal)
            expected_costs.append(exp_cost)

        confidence = torch.sigmoid(self.confidence_head(global_ctx).squeeze(-1))
        return {
            "temporal_split_logits": torch.stack(t_logits, dim=1),  # [B,K,T]
            "spatial_split_logits": torch.stack(s_logits, dim=1),
            "continue_logits": torch.stack(c_logits, dim=1),  # [B,K]
            "budget_allocation": torch.stack(budgets, dim=1),  # [B,K,2]
            "expected_cost": torch.stack(expected_costs, dim=1),  # [B,K]
            "planner_confidence": confidence,
            "global_ctx": global_ctx,
        }
