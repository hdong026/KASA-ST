"""Spatial graph modules for KASA v3 (frequency-guided graph + cross ST gates)."""

from math import sqrt
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicts.archs.arch_zoo.KASA_arch_v3.frequency import FrequencyDescriptor


def normalize_adj(adj):
    """Symmetric normalization D^{-1/2}(A+I)D^{-1/2}."""
    adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


def row_normalize(adj):
    row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return adj / row_sum


def apply_adj(x, adj):
    """Graph propagation on temporal features.

    Args:
        x: [B, T, N]
        adj: [N, N] or [B, N, N]
    """
    if adj.dim() == 2:
        return torch.einsum("ij,btj->bti", adj, x)
    return torch.einsum("bij,btj->bti", adj, x)


def mask_topk(logits, topk):
    if not (0 < topk < logits.shape[-1]):
        return logits
    topk_index = torch.topk(logits, k=topk, dim=-1).indices
    keep_mask = torch.zeros_like(logits, dtype=torch.bool)
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
    adj_tensor = torch.as_tensor(adj_obj, dtype=torch.float32)
    return normalize_adj(adj_tensor)


class GCNLayer(nn.Module):
    """Single GCN layer H = ReLU(A X W)."""

    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activate = nn.ReLU()

    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.mm(adj, support)
        return self.activate(output)


class ABCDSpatialModule(nn.Module):
    """Unified spatial module implementing schemes A/B/C/D with v3 extensions."""

    def __init__(
        self,
        node_size,
        input_len,
        d_spa,
        if_spatial,
        spatial_scheme="LEGACY",
        adj_mx_path=None,
        use_gcn=False,
        gcn_hidden_dim=64,
        use_dynamic_spatial=False,
        dyn_hidden_dim=64,
        dyn_topk=20,
        dyn_tau=0.5,
        dyn_alpha=0.15,
        dyn_static_weight=0.2,
        use_adaptive_adj=False,
        adp_hidden_dim=32,
        adp_topk=20,
        adp_tau=0.5,
        adp_alpha=0.1,
        use_hybrid_graph=False,
        hybrid_alpha=0.2,
        use_lightweight_spatial=False,
        light_alpha=0.05,
        freq_dim=16,
        use_frequency_guided_graph=True,
        freq_topk=20,
        use_cross_st_gate=False,
        use_spectral_decomp_gate=False,
        gate_hidden=8,
    ):
        super().__init__()
        self.node_size = node_size
        self.input_len = input_len
        self.d_spa = d_spa
        self.if_spatial = if_spatial
        self.spatial_scheme = str(spatial_scheme).upper()

        self.use_gcn = use_gcn
        self.use_dynamic_spatial = use_dynamic_spatial
        self.use_adaptive_adj = use_adaptive_adj
        self.use_hybrid_graph = use_hybrid_graph
        self.use_lightweight_spatial = use_lightweight_spatial

        if self.spatial_scheme in {"A", "B", "C", "D"}:
            self.use_gcn = self.spatial_scheme in {"A", "C"}
            self.use_dynamic_spatial = self.spatial_scheme in {"B", "C"}
            self.use_adaptive_adj = self.spatial_scheme in {"C"}
            self.use_hybrid_graph = self.spatial_scheme in {"C"}
            self.use_lightweight_spatial = self.spatial_scheme in {"D"}
        if self.use_hybrid_graph:
            self.use_dynamic_spatial = True
            self.use_adaptive_adj = True

        self.dyn_hidden_dim = dyn_hidden_dim
        self.dyn_topk = dyn_topk
        self.dyn_tau = dyn_tau
        self.dyn_alpha = dyn_alpha
        self.dyn_static_weight = dyn_static_weight

        self.adp_hidden_dim = adp_hidden_dim
        self.adp_topk = adp_topk
        self.adp_tau = adp_tau
        self.adp_alpha = adp_alpha

        self.hybrid_alpha = hybrid_alpha
        self.light_alpha = light_alpha

        self.freq_dim = freq_dim
        self.use_frequency_guided_graph = use_frequency_guided_graph
        self.freq_topk = freq_topk
        self.use_cross_st_gate = use_cross_st_gate
        self.use_spectral_decomp_gate = use_spectral_decomp_gate
        self.gate_hidden = gate_hidden

        self.register_buffer("adj_mx", None)
        need_static_adj = (
            self.use_gcn
            or self.use_dynamic_spatial
            or self.use_lightweight_spatial
            or self.use_hybrid_graph
        )
        if need_static_adj:
            self.adj_mx = load_adj_from_pickle(adj_mx_path)

        self.gcn1 = None
        self.gcn2 = None
        if self.use_gcn and self.adj_mx is not None and self.if_spatial:
            self.gcn1 = GCNLayer(self.d_spa, gcn_hidden_dim)
            self.gcn2 = GCNLayer(gcn_hidden_dim, self.d_spa)

        self.dynamic_query = None
        self.dynamic_key = None
        if self.use_dynamic_spatial:
            self.dynamic_query = nn.Linear(self.input_len, self.dyn_hidden_dim, bias=False)
            self.dynamic_key = nn.Linear(self.input_len, self.dyn_hidden_dim, bias=False)

        self.adaptive_src = None
        self.adaptive_dst = None
        if self.use_adaptive_adj:
            self.adaptive_src = nn.Parameter(torch.empty(self.node_size, self.adp_hidden_dim))
            self.adaptive_dst = nn.Parameter(torch.empty(self.node_size, self.adp_hidden_dim))
            nn.init.xavier_uniform_(self.adaptive_src)
            nn.init.xavier_uniform_(self.adaptive_dst)

        self.hybrid_logits = None
        if self.use_hybrid_graph and not self.use_frequency_guided_graph:
            self.hybrid_logits = nn.Parameter(torch.zeros(3))

        self.freq_hybrid_logits = None
        if self.use_hybrid_graph and self.use_frequency_guided_graph:
            self.freq_hybrid_logits = nn.Parameter(torch.zeros(4))

        self.freq_descriptor = None
        if self.use_frequency_guided_graph:
            self.freq_descriptor = FrequencyDescriptor(self.freq_dim)

        self.freq_gate_proj = None
        self.cross_gate_mlp = None
        self.spectral_gate_mlp_low = None
        self.spectral_gate_mlp_high = None
        if self.use_cross_st_gate:
            self.freq_gate_proj = nn.Linear(self.freq_dim, 1)
            if self.use_spectral_decomp_gate:
                self.spectral_gate_mlp_low = nn.Sequential(
                    nn.Linear(3, self.gate_hidden),
                    nn.SiLU(),
                    nn.Linear(self.gate_hidden, 1),
                )
                self.spectral_gate_mlp_high = nn.Sequential(
                    nn.Linear(3, self.gate_hidden),
                    nn.SiLU(),
                    nn.Linear(self.gate_hidden, 1),
                )
            else:
                self.cross_gate_mlp = nn.Sequential(
                    nn.Linear(3, self.gate_hidden),
                    nn.SiLU(),
                    nn.Linear(self.gate_hidden, 1),
                )

        self._last_diagnostics: dict = {}

    def _build_dynamic_adj(self, history_flow):
        node_signal = history_flow.permute(0, 2, 1)
        query = F.normalize(self.dynamic_query(node_signal), p=2, dim=-1)
        key = F.normalize(self.dynamic_key(node_signal), p=2, dim=-1)
        logits = torch.matmul(query, key.transpose(-1, -2)) / sqrt(self.dyn_hidden_dim)

        if self.adj_mx is not None:
            static_adj = row_normalize(self.adj_mx)
            logits = logits + self.dyn_static_weight * static_adj.unsqueeze(0)

        logits = mask_topk(logits, self.dyn_topk)
        dyn_adj = torch.softmax(logits / max(self.dyn_tau, 1e-6), dim=-1)
        return dyn_adj

    def _build_adaptive_adj(self):
        src = F.normalize(self.adaptive_src, p=2, dim=-1)
        dst = F.normalize(self.adaptive_dst, p=2, dim=-1)
        logits = torch.matmul(src, dst.transpose(0, 1)) / sqrt(self.adp_hidden_dim)
        logits = mask_topk(logits, self.adp_topk)
        adp_adj = torch.softmax(logits / max(self.adp_tau, 1e-6), dim=-1)
        return adp_adj

    def _build_freq_adj(self, freq_emb):
        d = freq_emb.shape[-1]
        sim = torch.matmul(freq_emb, freq_emb.transpose(-1, -2)) / sqrt(d)
        sim = mask_topk(sim, self.freq_topk)
        return torch.softmax(sim, dim=-1)

    def _static_adj_batch(self, history_flow, batch_size):
        if self.adj_mx is not None:
            return row_normalize(self.adj_mx).unsqueeze(0).expand(batch_size, -1, -1)
        return torch.eye(self.node_size, device=history_flow.device).unsqueeze(0).expand(batch_size, -1, -1)

    def _build_hybrid_adj(self, history_flow, freq_emb=None):
        batch_size = history_flow.shape[0]
        dynamic_adj = self._build_dynamic_adj(history_flow)
        adaptive_adj = self._build_adaptive_adj().unsqueeze(0).expand(batch_size, -1, -1)
        static_adj = self._static_adj_batch(history_flow, batch_size)

        if self.use_frequency_guided_graph and freq_emb is not None:
            freq_adj = self._build_freq_adj(freq_emb)
            weights = torch.softmax(self.freq_hybrid_logits, dim=0)
            self._last_diagnostics["fusion_weights"] = {
                "static": float(weights[0].detach().cpu()),
                "adaptive": float(weights[1].detach().cpu()),
                "dynamic": float(weights[2].detach().cpu()),
                "frequency": float(weights[3].detach().cpu()),
            }
            return (
                weights[0] * static_adj
                + weights[1] * adaptive_adj
                + weights[2] * dynamic_adj
                + weights[3] * freq_adj
            )

        weights = torch.softmax(self.hybrid_logits, dim=0)
        self._last_diagnostics["fusion_weights"] = {
            "static": float(weights[0].detach().cpu()),
            "adaptive": float(weights[1].detach().cpu()),
            "dynamic": float(weights[2].detach().cpu()),
            "frequency": None,
        }
        return weights[0] * static_adj + weights[1] * adaptive_adj + weights[2] * dynamic_adj

    def _lowpass_horizon(self, x):
        """Moving-average low-pass along horizon H. x: [B, H, N, 1]."""
        b, h, n, _ = x.shape
        y = x.squeeze(-1).permute(0, 2, 1).reshape(b * n, 1, h)
        low = F.avg_pool1d(y, kernel_size=3, stride=1, padding=1)
        low = low.reshape(b, n, h).permute(0, 2, 1).unsqueeze(-1)
        high = x - low
        return low, high

    def _expand_freq_gate(self, freq_emb, horizon):
        gate = self.freq_gate_proj(freq_emb)  # [B, N, 1]
        gate = gate.unsqueeze(1).expand(-1, horizon, -1, -1)
        return gate

    def _apply_cross_gate(self, temporal_output, spatial_refine, freq_emb):
        freq_gate = self._expand_freq_gate(freq_emb, temporal_output.shape[1])
        if self.use_spectral_decomp_gate:
            yt_low, yt_high = self._lowpass_horizon(temporal_output)
            ys_low, ys_high = self._lowpass_horizon(spatial_refine)
            gate_low = torch.sigmoid(
                self.spectral_gate_mlp_low(torch.cat([yt_low, ys_low, freq_gate], dim=-1))
            )
            gate_high = torch.sigmoid(
                self.spectral_gate_mlp_high(torch.cat([yt_high, ys_high, freq_gate], dim=-1))
            )
            self._last_diagnostics["gate_low_mean"] = float(gate_low.mean().detach().cpu())
            self._last_diagnostics["gate_high_mean"] = float(gate_high.mean().detach().cpu())
            return temporal_output + gate_low * ys_low + gate_high * ys_high

        gate_input = torch.cat([temporal_output, spatial_refine, freq_gate], dim=-1)
        gate = torch.sigmoid(self.cross_gate_mlp(gate_input))
        self._last_diagnostics["gate_mean"] = float(gate.mean().detach().cpu())
        return temporal_output + gate * spatial_refine

    def get_enhanced_spatial_embedding(self, spa_codebook):
        if not self.if_spatial or spa_codebook is None:
            return None
        if self.use_gcn and self.adj_mx is not None and self.gcn1 is not None:
            emb = self.gcn1(spa_codebook, self.adj_mx)
            emb = self.gcn2(emb, self.adj_mx)
            return emb
        return None

    def refine_prediction(self, output, history_flow, return_diagnostics=False):
        """Apply spatial refinement with optional frequency graph and cross ST gate."""
        x = output.squeeze(-1)
        freq_emb = None
        if self.use_frequency_guided_graph and self.freq_descriptor is not None:
            freq_emb = self.freq_descriptor(history_flow)

        if self.use_hybrid_graph:
            hybrid_adj = self._build_hybrid_adj(history_flow, freq_emb=freq_emb)
            refine = apply_adj(x, hybrid_adj).unsqueeze(-1)
            if self.use_cross_st_gate and freq_emb is not None:
                output = self._apply_cross_gate(output, refine, freq_emb)
            else:
                output = output + self.hybrid_alpha * refine
            if return_diagnostics:
                return output, dict(self._last_diagnostics)
            return output

        if self.use_dynamic_spatial:
            dyn_adj = self._build_dynamic_adj(history_flow)
            refine = apply_adj(x, dyn_adj).unsqueeze(-1)
            if self.use_cross_st_gate and freq_emb is not None:
                output = self._apply_cross_gate(output, refine, freq_emb)
            else:
                output = output + self.dyn_alpha * refine
            if return_diagnostics:
                return output, dict(self._last_diagnostics)
            return output

        if self.use_adaptive_adj:
            adp_adj = self._build_adaptive_adj()
            refine = apply_adj(x, adp_adj).unsqueeze(-1)
            if self.use_cross_st_gate and freq_emb is not None:
                output = self._apply_cross_gate(output, refine, freq_emb)
            else:
                output = output + self.adp_alpha * refine
            if return_diagnostics:
                return output, dict(self._last_diagnostics)
            return output

        if self.use_lightweight_spatial and self.adj_mx is not None:
            static_adj = row_normalize(self.adj_mx)
            smooth = apply_adj(x, static_adj).unsqueeze(-1) - output
            output = output + self.light_alpha * smooth

        if return_diagnostics:
            return output, dict(self._last_diagnostics)
        return output

    def get_last_diagnostics(self):
        return dict(self._last_diagnostics)
