"""Module 2: Frequency-binned Spatial Feature Extractor (FS-Extractor)."""

import os
import pickle
from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_topk(logits: torch.Tensor, topk: Optional[int]) -> torch.Tensor:
    n = logits.shape[-1]
    if topk is None or not (0 < topk < n):
        return logits
    idx = torch.topk(logits, k=topk, dim=-1).indices
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep.scatter_(-1, idx, True)
    return logits.masked_fill(~keep, float("-inf"))


def _row_normalize(adj: torch.Tensor) -> torch.Tensor:
    row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return adj / row_sum


def load_adj_from_pickle(adj_mx_path: Optional[str]) -> Optional[torch.Tensor]:
    if not adj_mx_path or not os.path.exists(adj_mx_path):
        return None
    with open(adj_mx_path, "rb") as f:
        try:
            adj_obj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            adj_obj = pickle.load(f, encoding="latin1")
    if isinstance(adj_obj, (list, tuple)):
        adj_obj = adj_obj[-1]
    adj = torch.as_tensor(adj_obj, dtype=torch.float32)
    if adj.ndim == 3:
        adj = adj[0]
    return _row_normalize(adj)


class FrequencySpatialExtractor(nn.Module):
    """Build E^(k,τ)-conditioned dynamic graphs and band-wise spatial propagation."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        n_bands: int = 4,
        embed_dim: int = 32,
        topk: Optional[int] = None,
        attn_temperature: float = 1.0,
        dropout: float = 0.1,
        use_film: bool = True,
        use_band_specific_proj: bool = True,
        adj_mx_path: Optional[str] = None,
        static_hybrid_alpha: float = 0.2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_bands = n_bands
        self.embed_dim = embed_dim
        self.topk = topk
        self.attn_temperature = attn_temperature
        self.use_film = use_film
        self.use_band_specific_proj = use_band_specific_proj
        self.static_hybrid_alpha = static_hybrid_alpha

        self.node_emb = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.spec_proj = nn.Linear(input_dim, embed_dim)
        self.spec_norm = nn.LayerNorm(embed_dim)
        self.gamma_proj = nn.Linear(embed_dim, embed_dim)
        self.beta_proj = nn.Linear(embed_dim, embed_dim)

        n_proj = n_bands if use_band_specific_proj else 1
        self.spatial_proj = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(n_proj)]
        )
        self.band_alpha = nn.Parameter(torch.zeros(n_bands))
        self.dropout = nn.Dropout(dropout)

        static_adj = load_adj_from_pickle(adj_mx_path)
        if static_adj is not None:
            self.register_buffer("static_adj", static_adj, persistent=False)
        else:
            eye = torch.eye(num_nodes, dtype=torch.float32)
            self.register_buffer("static_adj", eye, persistent=False)

        self.latest_band_alpha = None
        self.latest_adj_sample = None

    def _node_embedding(self, a_k_tau: torch.Tensor) -> torch.Tensor:
        # a_k_tau: [B*, N, C_tf] -> E_tilde: [B*, N, embed_dim]
        b_star, n, _ = a_k_tau.shape
        if not self.use_film:
            return self.node_emb.unsqueeze(0).expand(b_star, -1, -1)

        m = self.spec_norm(self.spec_proj(a_k_tau))
        gamma = 0.1 * torch.tanh(self.gamma_proj(m))
        beta = 0.1 * torch.tanh(self.beta_proj(m))
        e0 = self.node_emb.unsqueeze(0).expand(b_star, -1, -1)
        return (1.0 + gamma) * e0 + beta

    def _dynamic_adj(self, e_tilde: torch.Tensor) -> torch.Tensor:
        # e_tilde: [B*, N, d] -> [B*, N, N]
        logits = torch.bmm(e_tilde, e_tilde.transpose(1, 2)) / sqrt(self.embed_dim)
        logits = _mask_topk(logits, self.topk)
        temp = max(self.attn_temperature, 1e-6)
        dyn_adj = torch.softmax(logits / temp, dim=-1)

        if self.static_hybrid_alpha > 0:
            static = self.static_adj.unsqueeze(0).expand(e_tilde.shape[0], -1, -1)
            alpha = self.static_hybrid_alpha
            return (1.0 - alpha) * dyn_adj + alpha * static
        return dyn_adj

    def forward(self, x_frame: torch.Tensor, band_amp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_frame: [B, T_prime, N, C_tf]
            band_amp: [B, N, C_tf, K, T_prime]
        Returns:
            Z: [B, T_prime, N, hidden_dim]
        """
        b, t_prime, n, c_tf = x_frame.shape
        k_bands = band_amp.shape[3]
        alpha = F.softmax(self.band_alpha, dim=0)
        self.latest_band_alpha = alpha.detach()

        # Vectorize over (k, tau): [B, K, T', N, C]
        band_perm = band_amp.permute(0, 3, 4, 1, 2).contiguous()
        x_expand = x_frame.unsqueeze(1).expand(-1, k_bands, -1, -1, -1).contiguous()

        bk = b * k_bands * t_prime
        a_flat = band_perm.reshape(bk, n, c_tf)
        x_flat = x_expand.reshape(bk, n, c_tf)

        e_tilde = self._node_embedding(a_flat)
        a_dyn = self._dynamic_adj(e_tilde)
        if self.latest_adj_sample is None:
            self.latest_adj_sample = a_dyn[0].detach()

        propagated = torch.bmm(a_dyn, x_flat)

        if self.use_band_specific_proj:
            z_all = torch.empty(bk, n, self.hidden_dim, device=x_frame.device, dtype=x_frame.dtype)
            k_idx = (
                torch.arange(k_bands, device=x_frame.device)
                .view(1, k_bands, 1)
                .expand(b, -1, t_prime)
                .reshape(bk)
            )
            for k in range(k_bands):
                mask = k_idx == k
                if mask.any():
                    z_all[mask] = self.spatial_proj[k](propagated[mask])
        else:
            z_all = self.spatial_proj[0](propagated)

        z_reshaped = z_all.view(b, k_bands, t_prime, n, self.hidden_dim)
        z_out = (z_reshaped * alpha.view(1, k_bands, 1, 1, 1)).sum(dim=1)
        return self.dropout(z_out)
