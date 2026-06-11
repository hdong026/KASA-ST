"""Building blocks for ChainForecasting architecture."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenTransformerBlock(nn.Module):
    """Transformer block over sequence tokens (batch_first)."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class NodeCrossAttentionBlock(nn.Module):
    """Cross-attention from future queries to history memory, per node."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, F, N, d]
            memory: [B, M, N, d]
        Returns:
            [B, F, N, d]
        """
        batch_size, future_len, num_nodes, d_model = query.shape
        mem_len = memory.shape[1]
        q = query.reshape(batch_size * num_nodes, future_len, d_model)
        mem = memory.reshape(batch_size * num_nodes, mem_len, d_model)
        attn_out, _ = self.cross_attn(q, mem, mem, need_weights=False)
        out = self.norm1(q + self.dropout(attn_out))
        out = self.norm2(out + self.dropout(self.ffn(out)))
        return out.reshape(batch_size, future_len, num_nodes, d_model)


class HistoryPatchEncoder(nn.Module):
    """Patch-style history encoder producing [B, M, N, d]."""

    def __init__(
        self,
        input_dim: int,
        patch_len: int,
        patch_stride: int,
        d_model: int,
        num_nodes: int,
        max_patches: int = 8,
        num_heads: int = 4,
        ffn_dim: int = 128,
        use_temporal_transformer: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.input_dim = input_dim
        flat_dim = patch_len * input_dim
        self.patch_proj = nn.Linear(flat_dim, d_model)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, d_model))
        self.patch_pos_emb = nn.Parameter(torch.empty(max_patches, d_model))
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.patch_pos_emb)
        self.use_temporal_transformer = use_temporal_transformer
        if use_temporal_transformer:
            self.temporal_block = TokenTransformerBlock(
                d_model, num_heads, ffn_dim, dropout=dropout
            )

    def forward(self, x_main: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_main: [B, H, N, C]
        Returns:
            patch_memory: [B, M, N, d]
        """
        patches = x_main.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        # [B, M, N, P, C]
        patches = patches.permute(0, 1, 2, 3, 4)
        batch_size, num_patches, num_nodes, patch_len, channels = patches.shape
        flat = patches.reshape(batch_size, num_patches, num_nodes, patch_len * channels)
        mem = self.patch_proj(flat)
        mem = mem + self.node_emb.unsqueeze(0).unsqueeze(0)
        mem = mem + self.patch_pos_emb[:num_patches].unsqueeze(0).unsqueeze(2)
        if self.use_temporal_transformer:
            tokens = mem.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, num_patches, -1)
            tokens = self.temporal_block(tokens)
            mem = tokens.reshape(batch_size, num_nodes, num_patches, -1).permute(0, 2, 1, 3)
        return mem


class DownsampleMemoryEncoder(nn.Module):
    """Average-pool history by factor 2 and project to d_model."""

    def __init__(self, input_dim: int, d_model: int, num_nodes: int, pool_factor: int = 2):
        super().__init__()
        self.pool_factor = pool_factor
        self.proj = nn.Linear(input_dim, d_model)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, d_model))
        nn.init.xavier_uniform_(self.node_emb)

    def forward(self, x_main: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_main: [B, H, N, C]
        Returns:
            down_memory: [B, M_ds, N, d]
        """
        batch_size, hist_len, num_nodes, channels = x_main.shape
        x = x_main.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, channels, hist_len)
        pooled_len = math.ceil(hist_len / self.pool_factor)
        x = F.avg_pool1d(x, kernel_size=self.pool_factor, stride=self.pool_factor)
        if x.shape[-1] > pooled_len:
            x = x[..., :pooled_len]
        x = x.reshape(batch_size, num_nodes, channels, -1).permute(0, 3, 1, 2)
        mem = self.proj(x)
        mem = mem + self.node_emb.unsqueeze(0).unsqueeze(0)
        return mem


class FutureQueryEmbedding(nn.Module):
    """Build future queries for a given forecast length."""

    def __init__(self, future_len: int, d_model: int, num_nodes: int, td_size: int = 288):
        super().__init__()
        self.future_len = future_len
        self.td_size = td_size
        self.future_pos_emb = nn.Parameter(torch.empty(future_len, d_model))
        self.tod_emb = nn.Embedding(td_size, d_model)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, d_model))
        nn.init.xavier_uniform_(self.future_pos_emb)
        nn.init.xavier_uniform_(self.node_emb)

    def forward(self, history_main: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_main: [B, H, N, C] with ToD at channel 1
        Returns:
            queries: [B, F, N, d]
        """
        batch_size, _, num_nodes, _ = history_main.shape
        tod = history_main[:, -1, :, 1]
        if float(tod.max()) <= 1.0 + 1e-6:
            start_idx = torch.floor(tod * self.td_size).long()
        else:
            start_idx = tod.long()
        start_idx = start_idx.clamp(0, self.td_size - 1)
        step = max(1, self.td_size // max(self.future_len, 1))
        offsets = torch.arange(self.future_len, device=history_main.device).view(1, 1, -1)
        future_idx = (start_idx.unsqueeze(-1) + offsets * step) % self.td_size
        tod_q = self.tod_emb(future_idx).permute(0, 2, 1, 3)
        pos_q = self.future_pos_emb.unsqueeze(0).unsqueeze(2)
        node_q = self.node_emb.unsqueeze(0).unsqueeze(0)
        return tod_q + pos_q + node_q


class CoarseForecastDecoder(nn.Module):
    """Initial coarse decoder: cross-attn from future queries to history memory."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.block = NodeCrossAttentionBlock(d_model, num_heads, ffn_dim, dropout=dropout)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.block(queries, memory)


class ForecastTransitionBlock(nn.Module):
    """Transition from previous forecast state to next resolution."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = NodeCrossAttentionBlock(d_model, num_heads, ffn_dim, dropout=dropout)
        self.temporal_block = TokenTransformerBlock(d_model, num_heads, ffn_dim, dropout=dropout)

    def forward(
        self,
        prev_state: torch.Tensor,
        queries: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prev_state: [B, F_prev, N, d]
            queries: [B, F_next, N, d]
            memory: [B, M, N, d]
        """
        z = prev_state + queries
        g = self.cross_attn(z, memory)
        batch_size, future_len, num_nodes, d_model = g.shape
        tokens = g.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, future_len, d_model)
        tokens = self.temporal_block(tokens)
        return tokens.reshape(batch_size, num_nodes, future_len, d_model).permute(0, 2, 1, 3)


def upsample_state(state: torch.Tensor, target_len: int) -> torch.Tensor:
    """Interpolate forecast state along future length."""
    batch_size, src_len, num_nodes, d_model = state.shape
    x = state.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, d_model, src_len)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.reshape(batch_size, num_nodes, d_model, target_len).permute(0, 3, 1, 2)


def pool_target_to_length(y: torch.Tensor, target_len: int) -> torch.Tensor:
    """Average-pool future target [B, F, N, 1] to target_len."""
    batch_size, future_len, num_nodes, channels = y.shape
    if future_len == target_len:
        return y
    if future_len % target_len == 0:
        group = future_len // target_len
        return y.reshape(batch_size, target_len, group, num_nodes, channels).mean(dim=2)
    x = y.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, channels, future_len)
    x = F.adaptive_avg_pool1d(x, target_len)
    return x.reshape(batch_size, num_nodes, channels, target_len).permute(0, 3, 1, 2)
