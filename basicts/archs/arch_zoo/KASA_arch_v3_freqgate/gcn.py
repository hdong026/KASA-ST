"""Frequency-guided spatial module for KASA v3-freqgate."""

from math import sqrt
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicts.archs.arch_zoo.KASA_arch_v3_freqgate.frequency import FrequencyDescriptor


def normalize_adj(adj):
    adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


def row_normalize(adj):
    row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return adj / row_sum


def mask_topk(logits, topk):
    if not (0 < topk < logits.shape[-1]):
        return logits
    topk_index = torch.topk(logits, k=topk, dim=-1).indices
    keep_mask = torch.zeros_like(logits, dtype=torch.bool)
    keep_mask.scatter_(-1, topk_index, True)
    return logits.masked_fill(~keep_mask, float("-inf"))


def mask_topk_keep_diag(logits, topk):
    """Top-k sparsification while always retaining diagonal (self-loop)."""
    n = logits.shape[-1]
    if not (0 < topk < n):
        return logits
    diag_idx = torch.arange(n, device=logits.device)
    keep_mask = torch.zeros_like(logits, dtype=torch.bool)
    keep_mask[..., diag_idx, diag_idx] = True
    topk_index = torch.topk(logits, k=topk, dim=-1).indices
    keep_mask.scatter_(-1, topk_index, True)
    return logits.masked_fill(~keep_mask, float("-inf"))


def load_adj_from_pickle(adj_mx_path):
    if not adj_mx_path or not os.path.exists(adj_mx_path):
        return None
    with open(adj_mx_path, "rb") as f:
        try:
            adj_obj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            adj_obj = pickle.load(f, encoding="latin1")
    if isinstance(adj_obj, (list, tuple)):
        for item in reversed(adj_obj):
            if hasattr(item, "shape") and len(item.shape) == 2:
                adj_obj = item
                break
    return normalize_adj(torch.as_tensor(adj_obj, dtype=torch.float32))


def _zero_init_last_linear(seq: nn.Sequential, bias_value: float = 0.0) -> None:
    last = seq[-1]
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, bias_value)


def _conservative_init_gate_last_linear(seq: nn.Sequential) -> None:
    _zero_init_last_linear(seq, bias_value=-2.0)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activate = nn.ReLU()

    def forward(self, x, adj):
        return self.activate(torch.mm(adj, self.linear(x)))


class FreqGateSpatialModule(nn.Module):
    """Post-temporal frequency-guided graph fusion and cross ST gating."""

    def __init__(
        self,
        node_size,
        input_len,
        d_spa,
        if_spatial,
        adj_mx_path=None,
        use_gcn=False,
        gcn_hidden_dim=64,
        dyn_hidden_dim=64,
        dyn_topk=20,
        dyn_tau=0.5,
        dyn_static_weight=0.2,
        adp_hidden_dim=32,
        adp_topk=20,
        adp_tau=0.5,
        hybrid_alpha=0.2,
        freq_dim=16,
        freq_topk=20,
        freq_eps=1e-6,
        use_frequency_guided_graph=True,
        use_freq_conditioned_fusion=True,
        graph_fusion_hidden=16,
        use_cross_st_gate=True,
        gate_hidden=16,
        gate_residual_scale=1.0,
        use_spectral_decomp_gate=False,
    ):
        super().__init__()
        self.node_size = node_size
        self.input_len = input_len
        self.d_spa = d_spa
        self.if_spatial = if_spatial

        self.dyn_hidden_dim = dyn_hidden_dim
        self.dyn_topk = dyn_topk
        self.dyn_tau = dyn_tau
        self.dyn_static_weight = dyn_static_weight
        self.adp_hidden_dim = adp_hidden_dim
        self.adp_topk = adp_topk
        self.adp_tau = adp_tau
        self.hybrid_alpha = hybrid_alpha

        self.freq_dim = freq_dim
        self.freq_topk = freq_topk
        self.use_frequency_guided_graph = use_frequency_guided_graph
        self.use_freq_conditioned_fusion = use_freq_conditioned_fusion
        self.use_cross_st_gate = use_cross_st_gate
        self.gate_residual_scale = gate_residual_scale
        self.use_spectral_decomp_gate = use_spectral_decomp_gate

        self.register_buffer("adj_mx", load_adj_from_pickle(adj_mx_path))

        self.gcn1 = None
        self.gcn2 = None
        if use_gcn and self.adj_mx is not None and self.if_spatial:
            self.gcn1 = GCNLayer(d_spa, gcn_hidden_dim)
            self.gcn2 = GCNLayer(gcn_hidden_dim, d_spa)

        self.dynamic_query = nn.Linear(input_len, dyn_hidden_dim, bias=False)
        self.dynamic_key = nn.Linear(input_len, dyn_hidden_dim, bias=False)
        self.adaptive_src = nn.Parameter(torch.empty(node_size, adp_hidden_dim))
        self.adaptive_dst = nn.Parameter(torch.empty(node_size, adp_hidden_dim))
        nn.init.xavier_uniform_(self.adaptive_src)
        nn.init.xavier_uniform_(self.adaptive_dst)

        num_graphs = 4 if use_frequency_guided_graph else 3
        if use_frequency_guided_graph:
            base_init = torch.tensor([0.0, 0.0, 0.0, -2.0])
        else:
            base_init = torch.zeros(num_graphs)
        self.base_graph_logits = nn.Parameter(base_init)

        self.graph_fusion_mlp = None
        if use_frequency_guided_graph and use_freq_conditioned_fusion:
            self.graph_fusion_mlp = nn.Sequential(
                nn.Linear(freq_dim, graph_fusion_hidden),
                nn.SiLU(),
                nn.Linear(graph_fusion_hidden, 4),
            )
            _zero_init_last_linear(self.graph_fusion_mlp)

        self.freq_descriptor = None
        if use_frequency_guided_graph:
            self.freq_descriptor = FrequencyDescriptor(freq_dim, eps=freq_eps)

        self.freq_gate_proj = None
        self.gate_mlp = None
        self.gate_low_mlp = None
        self.gate_high_mlp = None
        if use_cross_st_gate:
            if use_frequency_guided_graph:
                self.freq_gate_proj = nn.Linear(freq_dim, 1)
            if use_spectral_decomp_gate:
                self.gate_low_mlp = nn.Sequential(
                    nn.Linear(3, gate_hidden),
                    nn.SiLU(),
                    nn.Linear(gate_hidden, 1),
                )
                self.gate_high_mlp = nn.Sequential(
                    nn.Linear(3, gate_hidden),
                    nn.SiLU(),
                    nn.Linear(gate_hidden, 1),
                )
                _conservative_init_gate_last_linear(self.gate_low_mlp)
                _conservative_init_gate_last_linear(self.gate_high_mlp)
            else:
                self.gate_mlp = nn.Sequential(
                    nn.Linear(3, gate_hidden),
                    nn.SiLU(),
                    nn.Linear(gate_hidden, 1),
                )
                _conservative_init_gate_last_linear(self.gate_mlp)

        self.last_graph_fusion_weights = None
        self.last_gate_mean = None
        self.last_gate_low_mean = None
        self.last_gate_high_mean = None

    def get_enhanced_spatial_embedding(self, spa_codebook):
        if not self.if_spatial or spa_codebook is None:
            return None
        if self.gcn1 is not None and self.adj_mx is not None:
            emb = self.gcn1(spa_codebook, self.adj_mx)
            return self.gcn2(emb, self.adj_mx)
        return None

    def _build_dynamic_adj(self, history_flow):
        node_signal = history_flow.permute(0, 2, 1)
        query = F.normalize(self.dynamic_query(node_signal), p=2, dim=-1)
        key = F.normalize(self.dynamic_key(node_signal), p=2, dim=-1)
        logits = torch.matmul(query, key.transpose(-1, -2)) / sqrt(self.dyn_hidden_dim)
        if self.adj_mx is not None:
            logits = logits + self.dyn_static_weight * row_normalize(self.adj_mx).unsqueeze(0)
        logits = mask_topk(logits, self.dyn_topk)
        return torch.softmax(logits / max(self.dyn_tau, 1e-6), dim=-1)

    def _build_adaptive_adj(self):
        src = F.normalize(self.adaptive_src, p=2, dim=-1)
        dst = F.normalize(self.adaptive_dst, p=2, dim=-1)
        logits = torch.matmul(src, dst.transpose(0, 1)) / sqrt(self.adp_hidden_dim)
        logits = mask_topk(logits, self.adp_topk)
        return torch.softmax(logits / max(self.adp_tau, 1e-6), dim=-1)

    def _build_static_adj_batch(self, batch_size, device):
        if self.adj_mx is not None:
            return row_normalize(self.adj_mx).unsqueeze(0).expand(batch_size, -1, -1)
        eye = torch.eye(self.node_size, device=device)
        return eye.unsqueeze(0).expand(batch_size, -1, -1)

    def _build_freq_adj(self, freq_emb):
        d = freq_emb.shape[-1]
        sim = torch.matmul(freq_emb, freq_emb.transpose(-1, -2)) / sqrt(d)
        sim = mask_topk_keep_diag(sim, self.freq_topk)
        return torch.softmax(sim, dim=-1)

    def _graph_fusion_weights(self, freq_emb, batch_size):
        base = self.base_graph_logits
        if self.use_frequency_guided_graph and self.use_freq_conditioned_fusion and self.graph_fusion_mlp is not None:
            if freq_emb is None:
                raise ValueError("freq_emb is required for frequency-conditioned fusion")
            freq_context = freq_emb.mean(dim=1)
            delta = self.graph_fusion_mlp(freq_context)
            return torch.softmax(base.unsqueeze(0) + delta, dim=-1)
        return torch.softmax(base, dim=0).unsqueeze(0).expand(batch_size, -1)

    def _fuse_graphs(self, history_flow, freq_emb):
        batch_size = history_flow.shape[0]
        device = history_flow.device
        a_static = self._build_static_adj_batch(batch_size, device)
        a_adaptive = self._build_adaptive_adj().unsqueeze(0).expand(batch_size, -1, -1)
        a_dynamic = self._build_dynamic_adj(history_flow)
        weights = self._graph_fusion_weights(freq_emb, batch_size)
        self.last_graph_fusion_weights = weights.detach()

        if self.use_frequency_guided_graph:
            if freq_emb is None:
                raise ValueError("freq_emb is required when use_frequency_guided_graph=True")
            a_freq = self._build_freq_adj(freq_emb)
            return (
                weights[:, 0:1, None] * a_static
                + weights[:, 1:2, None] * a_adaptive
                + weights[:, 2:3, None] * a_dynamic
                + weights[:, 3:4, None] * a_freq
            )

        if self.use_freq_conditioned_fusion:
            raise ValueError(
                "use_freq_conditioned_fusion=True requires use_frequency_guided_graph=True"
            )
        return (
            weights[:, 0:1, None] * a_static
            + weights[:, 1:2, None] * a_adaptive
            + weights[:, 2:3, None] * a_dynamic
        )

    def _spatial_propagate(self, y_temporal, a_hybrid):
        y = y_temporal.squeeze(-1)
        y_spatial = torch.einsum("bij,bhj->bhi", a_hybrid, y)
        return y_spatial.unsqueeze(-1)

    def _lowpass_horizon(self, x):
        b, h, n, _ = x.shape
        y = x.squeeze(-1).permute(0, 2, 1).reshape(b * n, 1, h)
        low = F.avg_pool1d(y, kernel_size=3, stride=1, padding=1)
        low = low.reshape(b, n, h).permute(0, 2, 1).unsqueeze(-1)
        return low, x - low

    def _freq_gate_feature(self, freq_emb, horizon, batch_size, num_nodes, device, dtype):
        if freq_emb is not None and self.freq_gate_proj is not None:
            gate = self.freq_gate_proj(freq_emb)
            return gate.unsqueeze(1).expand(-1, horizon, -1, -1)
        return torch.zeros(batch_size, horizon, num_nodes, 1, device=device, dtype=dtype)

    def _apply_cross_gate(self, y_temporal, y_spatial, freq_emb):
        b, h, n, _ = y_temporal.shape
        freq_gate = self._freq_gate_feature(
            freq_emb, h, b, n, y_temporal.device, y_temporal.dtype
        )
        if self.use_spectral_decomp_gate:
            yt_low, yt_high = self._lowpass_horizon(y_temporal)
            ys_low, ys_high = self._lowpass_horizon(y_spatial)
            gate_low = torch.sigmoid(
                self.gate_low_mlp(torch.cat([yt_low, ys_low, freq_gate], dim=-1))
            )
            gate_high = torch.sigmoid(
                self.gate_high_mlp(torch.cat([yt_high, ys_high, freq_gate], dim=-1))
            )
            self.last_gate_low_mean = gate_low.detach().mean()
            self.last_gate_high_mean = gate_high.detach().mean()
            return y_temporal + self.gate_residual_scale * (gate_low * ys_low + gate_high * ys_high)

        gate = torch.sigmoid(
            self.gate_mlp(torch.cat([y_temporal, y_spatial, freq_gate], dim=-1))
        )
        self.last_gate_mean = gate.detach().mean()
        return y_temporal + self.gate_residual_scale * gate * y_spatial

    def refine_prediction(self, y_temporal, history_flow, tod=None, dow=None):
        """
        Post-temporal frequency-guided spatial refinement.

        Args:
            y_temporal: [B, H, N, 1]
            history_flow: [B, T, N]
        """
        del tod, dow

        freq_emb = None
        if self.use_frequency_guided_graph:
            if self.freq_descriptor is None:
                raise ValueError("freq_descriptor is required when use_frequency_guided_graph=True")
            freq_emb, _ = self.freq_descriptor(history_flow)

        a_hybrid = self._fuse_graphs(history_flow, freq_emb)
        y_spatial = self._spatial_propagate(y_temporal, a_hybrid)

        if self.use_cross_st_gate and (
            self.gate_mlp is not None or self.gate_low_mlp is not None
        ):
            return self._apply_cross_gate(y_temporal, y_spatial, freq_emb)
        return y_temporal + self.hybrid_alpha * y_spatial

    def get_diagnostics(self) -> dict:
        diag = {}
        if self.last_graph_fusion_weights is not None:
            w = self.last_graph_fusion_weights
            if w.dim() == 2:
                diag["graph_fusion_static_mean"] = float(w[:, 0].mean().cpu())
                diag["graph_fusion_adaptive_mean"] = float(w[:, 1].mean().cpu())
                diag["graph_fusion_dynamic_mean"] = float(w[:, 2].mean().cpu())
                if w.shape[1] > 3:
                    diag["graph_fusion_frequency_mean"] = float(w[:, 3].mean().cpu())
            else:
                diag["graph_fusion_static"] = float(w[0].cpu())
                diag["graph_fusion_adaptive"] = float(w[1].cpu())
                diag["graph_fusion_dynamic"] = float(w[2].cpu())
                if w.numel() > 3:
                    diag["graph_fusion_frequency"] = float(w[3].cpu())
        if self.last_gate_mean is not None:
            diag["gate_mean"] = float(self.last_gate_mean.cpu())
        if self.last_gate_low_mean is not None:
            diag["gate_low_mean"] = float(self.last_gate_low_mean.cpu())
        if self.last_gate_high_mean is not None:
            diag["gate_high_mean"] = float(self.last_gate_high_mean.cpu())
        return diag
