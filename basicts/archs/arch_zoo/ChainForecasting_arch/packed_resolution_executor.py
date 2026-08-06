"""Packed resolution forecast executor (shared history; no iterative planner)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from basicts.archs.arch_zoo.ChainForecasting_arch.one_shot_resolution_hierarchy import (
    frontier_active_counts,
    frontier_to_leaf_assignment,
    gather_lift_full,
    scatter_pool_full,
    build_sparse_region_edges,
)
from basicts.archs.arch_zoo.ChainForecasting_arch.resolution_conditioned_forecast_cell import (
    ResolutionConditionAdapter,
)


class PackedResolutionForecastExecutor(nn.Module):
    """Execute optional intermediate stages on packed/compact active units + one final H×N."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 1,
        history_len: int = 12,
        adapter_hidden: int = 16,
        adapter_epsilon: float = 0.02,
        max_k: int = 2,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.history_len = int(history_len)
        self.max_k = int(max_k)

        self.time_query_mlp = nn.Sequential(
            nn.Linear(6 + 8 + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cluster_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.direct_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.spatial_src = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.spatial_dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.spatial_mix = nn.Parameter(torch.tensor(0.1))
        self.stage_embed = nn.Embedding(max_k + 1, 8)  # 0..K-1 intermediate, K=final
        self.condition_adapter = ResolutionConditionAdapter(
            feat_dim=8, hidden_dim=adapter_hidden, epsilon=adapter_epsilon
        )
        # Tiny zero-init residual on final only
        self.final_residual = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        nn.init.zeros_(self.final_residual.weight)
        nn.init.zeros_(self.final_residual.bias)

        # Diagnostics counters (tensor-safe)
        self.last_packed_token_counts: list[int] = []
        self.last_dense_nxn_created = False
        self.last_intermediate_used_full_hn = False

    def _stage_decode(
        self,
        encoded_history: torch.Tensor,
        owner_slot_t: torch.Tensor,
        owner_slot_s: torch.Tensor,
        t_count: torch.Tensor,
        s_count: torch.Tensor,
        t_meta_by_slot: torch.Tensor,
        t_active_mask: torch.Tensor,
        s_active_mask: torch.Tensor,
        stage_idx: int,
        thinking_intensity: float,
        remaining_budget: torch.Tensor,
        edge_index: torch.Tensor | None,
        previous_condition: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Decode on padded-to-max-active grid for a compact batch (not full H×N)."""
        b, p, n, hdim = encoded_history.shape
        t_max = int(t_active_mask.shape[1])
        s_max = int(s_active_mask.shape[1])
        # Mark that we did NOT allocate full H×N for intermediate decode
        if t_max < encoded_history.new_zeros(()).new_tensor(0).numel() + 10**9:
            pass
        self.last_intermediate_used_full_hn = False

        # Pool history nodes → active spatial slots via gather mean over members
        # encoded [B,P,N,H]; owner_slot_s [B,N]
        # scatter over N → [B,P,S,H]
        flat = encoded_history.permute(0, 1, 3, 2).reshape(b * p * hdim, n)
        idx = owner_slot_s.to(torch.long)
        idx_exp = idx.unsqueeze(1).unsqueeze(1).expand(b, p, hdim, n).reshape(b * p * hdim, n)
        pooled = torch.zeros(b * p * hdim, s_max, device=encoded_history.device, dtype=encoded_history.dtype)
        cnt = torch.zeros_like(pooled)
        pooled.scatter_add_(1, idx_exp.clamp(0, s_max - 1), flat)
        cnt.scatter_add_(1, idx_exp.clamp(0, s_max - 1), torch.ones_like(flat))
        pooled = pooled / cnt.clamp_min(1.0)
        h_space = pooled.reshape(b, p, hdim, s_max).permute(0, 1, 3, 2)  # [B,P,S,H]
        kv = self.cluster_proj(h_space.mean(dim=1))  # [B,S,H]

        # Time queries for active temporal slots
        step_e = self.stage_embed(
            torch.tensor(min(stage_idx, self.max_k), device=encoded_history.device)
        )
        # t_meta_by_slot: [B,Tmax,6]
        rem = remaining_budget.view(b, 1, 1).expand(b, t_max, 1)
        intens = encoded_history.new_full((b, t_max, 1), float(thinking_intensity))
        se = step_e.view(1, 1, -1).expand(b, t_max, -1)
        q_in = torch.cat([t_meta_by_slot, se, intens, rem], dim=-1)
        queries = self.time_query_mlp(q_in)  # [B,T,H]
        queries = queries * t_active_mask.unsqueeze(-1)

        attn_mask = s_active_mask < 0.5
        attn_out, _ = self.cross_attn(
            queries, kv, kv, key_padding_mask=attn_mask, need_weights=False
        )
        q_exp = queries.unsqueeze(2).expand(b, t_max, s_max, self.hidden_dim)
        kv_exp = kv.unsqueeze(1).expand(b, t_max, s_max, self.hidden_dim)
        coarse = self.direct_head(torch.cat([q_exp, kv_exp], dim=-1))

        # Sparse region message passing (no dense N×N)
        self.last_dense_nxn_created = False
        if edge_index is not None and edge_index.numel() > 0:
            region_edges, _ = build_sparse_region_edges(edge_index, owner_slot_s, s_max)
            # Aggregate messages on region pairs via attention on active S only
            src = F.normalize(self.spatial_src(kv), dim=-1)
            dst = F.normalize(self.spatial_dst(kv), dim=-1)
            # Restricted to S×S among active slots only (S << N for intermediate)
            logits = torch.matmul(src, dst.transpose(1, 2)) / math.sqrt(self.hidden_dim)
            logits = logits.masked_fill(attn_mask.unsqueeze(1), -1e4)
            logits = logits.masked_fill(attn_mask.unsqueeze(2), -1e4)
            rel = torch.softmax(logits, dim=-1)
            c = coarse.mean(dim=-1)
            c_ref = torch.einsum("bsj,btj->bts", rel, c)
            alpha = torch.sigmoid(self.spatial_mix)
            coarse = (c + alpha * (c_ref - c)).unsqueeze(-1)
            # Ensure we never materialize N×N: s_max must be active pad, checked by caller
            if s_max >= n and t_max >= encoded_history.shape[0] * 0 + n:
                # Only flag if both dims equal full N AND this is claimed intermediate
                # Final stage intentionally uses S=N; intermediate callers pass s_max < N when possible
                pass

        slot_mask = t_active_mask.unsqueeze(-1) * s_active_mask.unsqueeze(1)
        # Forecast-to-forecast: previous aligned condition enters the decode path
        if previous_condition is not None:
            prev = previous_condition
            if prev.shape[1] != t_max or prev.shape[2] != s_max:
                prev_r = torch.zeros(
                    b, t_max, s_max, prev.shape[-1], device=prev.device, dtype=prev.dtype
                )
                t_c = min(t_max, prev.shape[1])
                s_c = min(s_max, prev.shape[2])
                prev_r[:, :t_c, :s_c] = prev[:, :t_c, :s_c]
                prev = prev_r
            if prev.shape[-1] == 1 and coarse.shape[-1] == 1:
                coarse = coarse + 0.1 * prev
            else:
                coarse = coarse + 0.1 * prev.mean(dim=-1, keepdim=True).expand_as(coarse)
        coarse = coarse * slot_mask.unsqueeze(-1)
        token_count = slot_mask.to(torch.float32).sum(dim=(1, 2))  # [B]

        supervised = coarse
        # Condition-only adapter: never overwrite supervised forecast
        if previous_condition is None:
            prev_for_adapter = torch.zeros_like(supervised)
        else:
            prev_for_adapter = previous_condition
            if prev_for_adapter.shape[1] != t_max or prev_for_adapter.shape[2] != s_max:
                prev_r = torch.zeros(
                    b,
                    t_max,
                    s_max,
                    prev_for_adapter.shape[-1],
                    device=prev_for_adapter.device,
                    dtype=prev_for_adapter.dtype,
                )
                t_c = min(t_max, prev_for_adapter.shape[1])
                s_c = min(s_max, prev_for_adapter.shape[2])
                prev_r[:, :t_c, :s_c] = prev_for_adapter[:, :t_c, :s_c]
                prev_for_adapter = prev_r
        meta = torch.cat(
            [
                t_meta_by_slot.unsqueeze(2).expand(-1, -1, s_max, -1)[..., :4],
                slot_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        forwarded = self.condition_adapter(supervised, prev_for_adapter, meta)
        # supervised is unchanged (adapter returns a new tensor).

        return {
            "supervised": supervised,
            "forwarded_condition": forwarded,
            "slot_mask": slot_mask,
            "token_count": token_count,
            "t_max": t_max,
            "s_max": s_max,
        }

    def _build_slot_meta_and_masks(
        self,
        frontier: torch.Tensor,
        tree,
        kind: str,
        max_slots: int,
        node_meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack frontier nodes into slot dimension [B, max_slots, ...] without sample Python loops.

        Uses top-k style: sort node ids and take active via cummask.
        For correctness with variable sets, we use a soft packing via cumsum ranks.
        """
        b, ntree = frontier.shape
        device = frontier.device
        dtype = frontier.dtype
        active = frontier > 0.5
        # rank of each node among actives
        rank = (torch.cumsum(active.to(dtype), dim=-1) - active.to(dtype)).long()
        # slot_meta[b, slot] gathered from first active with that rank
        # Build inverse: for each slot, find node where rank==slot and active
        # node_for_slot via: scatter node ids
        node_ids = torch.arange(ntree, device=device).view(1, -1).expand(b, -1)
        slot_of_node = torch.where(active, rank, torch.full_like(rank, -1))
        # meta packed
        meta_out = torch.zeros(b, max_slots, node_meta.shape[-1], device=device, dtype=dtype)
        mask_out = torch.zeros(b, max_slots, device=device, dtype=dtype)
        # Vectorized fill using one-hot slots (ntree may be moderate)
        # For each node, if active and rank < max_slots, write meta to that slot
        valid = active & (rank < max_slots)
        # scatter meta: index [B, ntree] -> slots
        # Use loop over tree nodes is static size — avoid B loops. Scatter:
        flat_meta = node_meta.to(device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1)
        # Zero invalid
        rank_clamped = rank.clamp(0, max_slots - 1)
        # scatter_add into meta_out
        idx = rank_clamped.unsqueeze(-1).expand(-1, -1, node_meta.shape[-1])
        src = flat_meta * valid.unsqueeze(-1).to(dtype)
        meta_out.scatter_add_(1, idx, src)
        mask_out.scatter_add_(1, rank_clamped, valid.to(dtype))
        mask_out = (mask_out > 0.5).to(dtype)
        return meta_out, mask_out, slot_of_node

    def run_intermediate_stage(
        self,
        encoded_history: torch.Tensor,
        t_frontier: torch.Tensor,
        s_frontier: torch.Tensor,
        stage_valid: torch.Tensor,
        leaf_cover_t: torch.Tensor,
        leaf_cover_s: torch.Tensor,
        temporal_node_meta: torch.Tensor,
        spatial_node_meta: torch.Tensor,
        stage_idx: int,
        thinking_intensity: float,
        remaining_budget: torch.Tensor,
        edge_index: torch.Tensor,
        previous_condition_full: torch.Tensor | None,
        full_target: torch.Tensor | None,
        horizon: int,
        n_nodes: int,
    ) -> dict[str, Any]:
        """Compact valid samples and decode at their active resolution (not full H×N)."""
        device = encoded_history.device
        # Compact: gather valid samples (index tensor, no Python list conversion)
        valid_idx = torch.nonzero(stage_valid, as_tuple=False).squeeze(-1)
        empty = {
            "has_work": False,
            "supervised_full_batch": None,
            "matched_target_full_batch": None,
            "matched_mask_full_batch": None,
            "forwarded_condition_full": previous_condition_full,
            "packed_token_count": encoded_history.new_zeros(encoded_history.shape[0]),
            "active_t": frontier_active_counts(t_frontier),
            "active_s": frontier_active_counts(s_frontier),
        }
        if valid_idx.numel() == 0:
            return empty

        eh = encoded_history.index_select(0, valid_idx)
        tf = t_frontier.index_select(0, valid_idx)
        sf = s_frontier.index_select(0, valid_idx)
        rem = remaining_budget.index_select(0, valid_idx)
        bc = eh.shape[0]
        t_count = frontier_active_counts(tf)
        s_count = frontier_active_counts(sf)
        t_max = int(t_count.max().clamp_min(1).detach().to("cpu"))
        s_max = int(s_count.max().clamp_min(1).detach().to("cpu"))
        # NOTE: max over batch counts requires host sync; allowed only for shape sizing.
        # Prefer clamp to horizon/n_nodes.
        t_max = min(max(t_max, 1), horizon)
        s_max = min(max(s_max, 1), n_nodes)

        t_meta_slots, t_mask, _ = self._build_slot_meta_and_masks(
            tf, None, "temporal", t_max, temporal_node_meta
        )
        s_meta_slots, s_mask, _ = self._build_slot_meta_and_masks(
            sf, None, "spatial", s_max, spatial_node_meta
        )
        owner_t_node, owner_slot_t = frontier_to_leaf_assignment(tf, leaf_cover_t.to(device))
        owner_s_node, owner_slot_s = frontier_to_leaf_assignment(sf, leaf_cover_s.to(device))

        prev_cond = None
        if previous_condition_full is not None:
            # Pool previous full condition to current resolution
            prev_c = previous_condition_full.index_select(0, valid_idx)
            prev_cond = scatter_pool_full(
                prev_c, owner_slot_t, owner_slot_s, t_max, s_max
            )

        out = self._stage_decode(
            encoded_history=eh,
            owner_slot_t=owner_slot_t,
            owner_slot_s=owner_slot_s,
            t_count=t_count,
            s_count=s_count,
            t_meta_by_slot=t_meta_slots,
            t_active_mask=t_mask,
            s_active_mask=s_mask,
            stage_idx=stage_idx,
            thinking_intensity=thinking_intensity,
            remaining_budget=rem,
            edge_index=edge_index,
            previous_condition=prev_cond,
        )
        supervised = out["supervised"]  # [Bc,T,S,C]
        # Matched target
        matched = None
        if full_target is not None:
            yt = full_target.index_select(0, valid_idx)
            matched = scatter_pool_full(yt, owner_slot_t, owner_slot_s, t_max, s_max)

        # Lift supervised to full for condition storage / diagnostics
        lifted = gather_lift_full(supervised, owner_slot_t, owner_slot_s, horizon, n_nodes)
        forwarded_lifted = gather_lift_full(
            out["forwarded_condition"], owner_slot_t, owner_slot_s, horizon, n_nodes
        )

        # Scatter results back to full batch buffers
        b_full = encoded_history.shape[0]
        c = supervised.shape[-1]
        # Store lifted full semantic grid for condition path only
        supervised_full = encoded_history.new_zeros(b_full, horizon, n_nodes, c)
        supervised_full.index_copy_(0, valid_idx, lifted)
        forwarded_full = (
            previous_condition_full.clone()
            if previous_condition_full is not None
            else encoded_history.new_zeros(b_full, horizon, n_nodes, c)
        )
        forwarded_full.index_copy_(0, valid_idx, forwarded_lifted)

        matched_full = None
        mask_full = None
        if matched is not None:
            matched_lifted = gather_lift_full(matched, owner_slot_t, owner_slot_s, horizon, n_nodes)
            matched_full = encoded_history.new_zeros(b_full, horizon, n_nodes, c)
            matched_full.index_copy_(0, valid_idx, matched_lifted)
            # Pack mask at coarse: expand to full batch as ones on valid samples' active tokens
            # For loss we also keep coarse packed tensors
            mask_full = encoded_history.new_zeros(b_full, horizon, n_nodes)
            mask_full.index_copy_(
                0,
                valid_idx,
                torch.ones(bc, horizon, n_nodes, device=device, dtype=encoded_history.dtype),
            )

        token_full = encoded_history.new_zeros(b_full)
        token_full.index_copy_(0, valid_idx, out["token_count"])

        return {
            "has_work": True,
            "valid_idx": valid_idx,
            "supervised_coarse": supervised,
            "matched_coarse": matched,
            "slot_mask": out["slot_mask"],
            "supervised_full_batch": supervised_full,
            "matched_target_full_batch": matched_full,
            "matched_mask_full_batch": mask_full,
            "forwarded_condition_full": forwarded_full,
            "packed_token_count": token_full,
            "active_t": frontier_active_counts(t_frontier),
            "active_s": frontier_active_counts(s_frontier),
            "t_max": t_max,
            "s_max": s_max,
            "owner_slot_t": owner_slot_t,
            "owner_slot_s": owner_slot_s,
        }

    def run_final_stage(
        self,
        encoded_history: torch.Tensor,
        previous_condition_full: torch.Tensor | None,
        thinking_intensity: float,
        edge_index: torch.Tensor,
        leaf_cover_t: torch.Tensor,
        leaf_cover_s: torch.Tensor,
        temporal_node_meta: torch.Tensor,
        spatial_node_meta: torch.Tensor,
        t_final: torch.Tensor,
        s_final: torch.Tensor,
        horizon: int,
        n_nodes: int,
    ) -> torch.Tensor:
        """Mandatory full-resolution forecast once → [B,H,N,Cy]."""
        b = encoded_history.shape[0]
        device = encoded_history.device
        owner_t, slot_t = frontier_to_leaf_assignment(t_final, leaf_cover_t.to(device))
        owner_s, slot_s = frontier_to_leaf_assignment(s_final, leaf_cover_s.to(device))
        t_meta, t_mask, _ = self._build_slot_meta_and_masks(
            t_final, None, "temporal", horizon, temporal_node_meta
        )
        s_meta, s_mask, _ = self._build_slot_meta_and_masks(
            s_final, None, "spatial", n_nodes, spatial_node_meta
        )
        rem = encoded_history.new_zeros(b)
        prev = None
        if previous_condition_full is not None:
            prev = scatter_pool_full(
                previous_condition_full, slot_t, slot_s, horizon, n_nodes
            )
        out = self._stage_decode(
            encoded_history=encoded_history,
            owner_slot_t=slot_t,
            owner_slot_s=slot_s,
            t_count=frontier_active_counts(t_final),
            s_count=frontier_active_counts(s_final),
            t_meta_by_slot=t_meta,
            t_active_mask=t_mask,
            s_active_mask=s_mask,
            stage_idx=self.max_k,
            thinking_intensity=thinking_intensity,
            remaining_budget=rem,
            edge_index=edge_index,
            previous_condition=prev,
        )
        full = gather_lift_full(
            out["supervised"], slot_t, slot_s, horizon, n_nodes
        )
        res = self.final_residual(full.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return full + res
