"""Spatial graph modules for HoloST (A/B/C/D schemes)."""

from math import sqrt
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def build_adaptive_adj_with_self(
    src_embedding: torch.Tensor,
    dst_embedding: torch.Tensor,
    topk: int,
    tau: float,
) -> torch.Tensor:
    """Build row-stochastic adaptive adjacency that always includes self-loops.

    Each row keeps exactly ``topk`` nonzeros: the self node plus top-(k-1)
    non-self neighbors (or only self when ``topk == 1``).
    """
    src = F.normalize(src_embedding, p=2, dim=-1)
    dst = F.normalize(dst_embedding, p=2, dim=-1)

    logits = torch.matmul(src, dst.transpose(0, 1))
    logits = logits / sqrt(src.shape[-1])

    num_nodes = logits.shape[0]
    if topk < 1 or topk > num_nodes:
        raise ValueError(f"topk must be in [1, {num_nodes}], got {topk}")

    self_index = torch.arange(num_nodes, device=logits.device)

    if topk == 1:
        selected_logits = logits[self_index, self_index].unsqueeze(-1)
        selected_indices = self_index.unsqueeze(-1)
    else:
        non_self_logits = logits.clone()
        non_self_logits[self_index, self_index] = float("-inf")
        neighbor_logits, neighbor_indices = torch.topk(
            non_self_logits,
            k=topk - 1,
            dim=-1,
        )
        self_logits = logits[self_index, self_index].unsqueeze(-1)
        selected_logits = torch.cat([self_logits, neighbor_logits], dim=-1)
        selected_indices = torch.cat(
            [self_index.unsqueeze(-1), neighbor_indices],
            dim=-1,
        )

    selected_weights = torch.softmax(
        selected_logits / max(float(tau), 1e-6),
        dim=-1,
    )
    adjacency = torch.zeros_like(logits)
    adjacency.scatter_(
        dim=-1,
        index=selected_indices,
        src=selected_weights,
    )
    return adjacency


def load_adj_from_pickle(adj_mx_path):
    if not adj_mx_path or not os.path.exists(adj_mx_path):
        return None
    with open(adj_mx_path, "rb") as f:
        try:
            adj_obj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            adj_obj = pickle.load(f, encoding='latin1')
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


class STAESpatialAttentionBlock(nn.Module):
    """STAEformer-style spatial self-attention over the node dimension.

    Operates on [B, T, N, 1] forecasts: project to d_model, add a learnable
    spatiotemporal adaptive embedding, apply node-wise MHA (+ FFN) with
    internal residuals, then project back to 1 channel. The block output is
    the final prediction (no outer residual onto Y_T).
    """

    def __init__(
        self,
        node_size: int,
        time_len: int,
        d_model: int = 32,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"spatial_attn_dim={d_model} must be divisible by "
                f"spatial_attn_heads={num_heads}."
            )
        if num_layers < 1:
            raise ValueError(f"spatial_attn_layers must be >= 1, got {num_layers}.")

        self.node_size = int(node_size)
        self.time_len = int(time_len)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)

        self.input_proj = nn.Linear(1, self.d_model)
        self.adaptive_embedding = nn.Parameter(
            torch.empty(self.time_len, self.node_size, self.d_model)
        )
        nn.init.xavier_uniform_(self.adaptive_embedding)

        self.attn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for _ in range(self.num_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=self.d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.norm1_layers.append(nn.LayerNorm(self.d_model))
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(self.d_model, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, self.d_model),
                )
            )
            self.norm2_layers.append(nn.LayerNorm(self.d_model))

        self.output_proj = nn.Linear(self.d_model, 1)

    def forward(self, y_temporal_final: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_temporal_final: [B, T, N, 1]
        Returns:
            prediction: [B, T, N, 1]
        """
        if y_temporal_final.ndim != 4 or y_temporal_final.shape[-1] != 1:
            raise ValueError(
                "y_temporal_final must have shape [B, T, N, 1], "
                f"got {tuple(y_temporal_final.shape)}."
            )

        batch_size, time_len, num_nodes, _ = y_temporal_final.shape
        if num_nodes != self.node_size:
            raise ValueError(
                f"Expected num_nodes={self.node_size}, got {num_nodes}."
            )
        if time_len != self.time_len:
            raise ValueError(
                f"Expected time_len={self.time_len}, got {time_len}."
            )

        # [B, T, N, d]
        hidden = self.input_proj(y_temporal_final)
        hidden = hidden + self.adaptive_embedding.unsqueeze(0)

        # Node-dimension attention: treat each (B, T) as a batch item over N tokens.
        hidden = hidden.reshape(batch_size * time_len, num_nodes, self.d_model)

        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers,
            self.norm1_layers,
            self.ffn_layers,
            self.norm2_layers,
        ):
            attn_out, _ = attn(hidden, hidden, hidden, need_weights=False)
            hidden = norm1(hidden + self.dropout(attn_out))
            ffn_out = ffn(hidden)
            hidden = norm2(hidden + self.dropout(ffn_out))

        prediction = self.output_proj(hidden)
        prediction = prediction.reshape(batch_size, time_len, num_nodes, 1)
        return prediction


class ABCDSpatialModule(nn.Module):
    """Unified spatial module implementing schemes A/B/C/D."""

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
        post_spatial_mode="hybrid",
        adaptive_ms_topks=None,
        adaptive_ms_alpha=0.10,
        adaptive_ms_fusion="softmax",
        adaptive_ms_share_logits=True,
        adaptive_ms_init="favor_largest",
        cluster_adj=None,
        cluster_graph_mix_lambda=0.5,
        spatial_attn_dim=32,
        spatial_attn_heads=4,
        spatial_attn_ffn_dim=128,
        spatial_attn_layers=1,
        spatial_attn_dropout=0.1,
        stae_time_len=None,
    ):
        super().__init__()
        self.node_size = node_size
        self.input_len = input_len
        self.d_spa = d_spa
        self.if_spatial = if_spatial
        self.spatial_scheme = str(spatial_scheme).upper()
        self.post_spatial_mode = str(post_spatial_mode).lower()

        # Control flags (explicit flags + scheme override).
        self.use_gcn = use_gcn
        self.use_dynamic_spatial = use_dynamic_spatial
        self.use_adaptive_adj = use_adaptive_adj
        self.use_hybrid_graph = use_hybrid_graph
        self.use_lightweight_spatial = use_lightweight_spatial

        if self.post_spatial_mode == "adaptive_multiscale_only":
            self.use_adaptive_adj = True
            self.use_dynamic_spatial = False
            self.use_hybrid_graph = False
            self.use_gcn = False
            self.use_lightweight_spatial = False

        if self.post_spatial_mode == "adaptive_cluster_mix":
            self.use_adaptive_adj = True
            self.use_dynamic_spatial = False
            self.use_hybrid_graph = False
            self.use_gcn = False
            self.use_lightweight_spatial = False

        if self.post_spatial_mode == "adaptive_self_replace":
            # Independent G1-self-replace path: Y_hat = A_self Y_T.
            self.use_adaptive_adj = True
            self.use_dynamic_spatial = False
            self.use_hybrid_graph = False
            self.use_gcn = False
            self.use_lightweight_spatial = False

        if self.post_spatial_mode == "stae_spatial_attention":
            # STAEformer-style latent spatial attention; no graph averaging.
            self.use_adaptive_adj = False
            self.use_dynamic_spatial = False
            self.use_hybrid_graph = False
            self.use_gcn = False
            self.use_lightweight_spatial = False

        if self.post_spatial_mode not in {
            "adaptive_multiscale_only",
            "adaptive_cluster_mix",
            "adaptive_self_replace",
            "stae_spatial_attention",
        } and self.spatial_scheme in {"A", "B", "C", "D"}:
            self.use_gcn = self.spatial_scheme in {"A", "C"}
            self.use_dynamic_spatial = self.spatial_scheme in {"B", "C"}
            self.use_adaptive_adj = self.spatial_scheme in {"C"}
            self.use_hybrid_graph = self.spatial_scheme in {"C"}
            self.use_lightweight_spatial = self.spatial_scheme in {"D"}
        if self.use_hybrid_graph:
            self.use_dynamic_spatial = True
            self.use_adaptive_adj = True

        # Hyper-params.
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
        self.cluster_graph_mix_lambda = float(cluster_graph_mix_lambda)

        self.adaptive_ms_topks = list(adaptive_ms_topks or [8, 16, 32])
        self.adaptive_ms_alpha = float(adaptive_ms_alpha)
        self.adaptive_ms_fusion = str(adaptive_ms_fusion).lower()
        self.adaptive_ms_share_logits = bool(adaptive_ms_share_logits)
        self.adaptive_ms_init = str(adaptive_ms_init).lower()

        # Static adjacency buffer.
        self.register_buffer("adj_mx", None)
        need_static_adj = (
            self.post_spatial_mode not in {
                "adaptive_multiscale_only",
                "adaptive_self_replace",
                "stae_spatial_attention",
            }
            and (
                self.use_gcn
                or self.use_dynamic_spatial
                or self.use_lightweight_spatial
                or self.use_hybrid_graph
            )
        )
        if need_static_adj:
            self.adj_mx = load_adj_from_pickle(adj_mx_path)

        self.stae_spatial_block = None
        if self.post_spatial_mode == "stae_spatial_attention":
            self.stae_spatial_block = STAESpatialAttentionBlock(
                node_size=self.node_size,
                time_len=int(stae_time_len or input_len),
                d_model=int(spatial_attn_dim),
                num_heads=int(spatial_attn_heads),
                ffn_dim=int(spatial_attn_ffn_dim),
                num_layers=int(spatial_attn_layers),
                dropout=float(spatial_attn_dropout),
            )

        # Scheme A: static GCN on spatial codebook.
        self.gcn1 = None
        self.gcn2 = None
        if self.use_gcn and self.adj_mx is not None and self.if_spatial:
            self.gcn1 = GCNLayer(self.d_spa, gcn_hidden_dim)
            self.gcn2 = GCNLayer(gcn_hidden_dim, self.d_spa)

        # Scheme B: dynamic graph from flow windows.
        self.dynamic_query = None
        self.dynamic_key = None
        if self.use_dynamic_spatial:
            self.dynamic_query = nn.Linear(self.input_len, self.dyn_hidden_dim, bias=False)
            self.dynamic_key = nn.Linear(self.input_len, self.dyn_hidden_dim, bias=False)

        # Scheme C: adaptive graph parameters + hybrid fusion.
        self.adaptive_src = None
        self.adaptive_dst = None
        if self.use_adaptive_adj:
            self.adaptive_src = nn.Parameter(torch.empty(self.node_size, self.adp_hidden_dim))
            self.adaptive_dst = nn.Parameter(torch.empty(self.node_size, self.adp_hidden_dim))
            nn.init.xavier_uniform_(self.adaptive_src)
            nn.init.xavier_uniform_(self.adaptive_dst)

        self.hybrid_logits = None
        if self.use_hybrid_graph:
            self.hybrid_logits = nn.Parameter(torch.zeros(3))

        self.adaptive_ms_logits = None
        self.adaptive_ms_src_list = None
        self.adaptive_ms_dst_list = None
        if self.post_spatial_mode == "adaptive_multiscale_only":
            num_scales = len(self.adaptive_ms_topks)
            if self.adaptive_ms_fusion != "softmax":
                raise ValueError(
                    f"Unsupported adaptive_ms_fusion: {self.adaptive_ms_fusion}. "
                    "Only 'softmax' is supported in v1."
                )
            self.adaptive_ms_logits = nn.Parameter(
                self._init_adaptive_ms_logits(num_scales, self.adaptive_ms_init)
            )
            if not self.adaptive_ms_share_logits:
                self.adaptive_ms_src_list = nn.ParameterList()
                self.adaptive_ms_dst_list = nn.ParameterList()
                for _ in range(num_scales):
                    src = nn.Parameter(torch.empty(self.node_size, self.adp_hidden_dim))
                    dst = nn.Parameter(torch.empty(self.node_size, self.adp_hidden_dim))
                    nn.init.xavier_uniform_(src)
                    nn.init.xavier_uniform_(dst)
                    self.adaptive_ms_src_list.append(src)
                    self.adaptive_ms_dst_list.append(dst)

        self.last_adaptive_adj = None
        self.last_adaptive_ms_weights = None
        self.last_adaptive_ms_entropy = None
        self.last_cluster_adj = None
        self.last_mix_adj = None
        self.last_adp_density = None
        self.last_cluster_density = None
        self.last_adj_mean_abs_diff = None
        self.g1_stagewise_s1_gate = None
        if cluster_adj is not None:
            self.register_buffer("cluster_adj", cluster_adj.float())
        else:
            self.cluster_adj = None

    def enable_g1_stagewise_s1_gate(self) -> None:
        """Zero-init gate so G1 S1 starts as pred_final ~= pred_T_full."""
        device = None
        if self.adaptive_src is not None:
            device = self.adaptive_src.device
        elif self.g1_stagewise_s1_gate is not None:
            device = self.g1_stagewise_s1_gate.device
        if self.g1_stagewise_s1_gate is None:
            self.g1_stagewise_s1_gate = nn.Parameter(
                torch.zeros(1, device=device) if device is not None else torch.zeros(1)
            )
        else:
            with torch.no_grad():
                self.g1_stagewise_s1_gate.zero_()
                if device is not None:
                    self.g1_stagewise_s1_gate.data = self.g1_stagewise_s1_gate.data.to(device)

    def _build_dynamic_adj(self, history_flow):
        node_signal = history_flow.permute(0, 2, 1)  # [B, N, L]
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

    def _build_adaptive_adj_with_self(self):
        """Adaptive adjacency with an explicit self-loop in every row."""
        return build_adaptive_adj_with_self(
            src_embedding=self.adaptive_src,
            dst_embedding=self.adaptive_dst,
            topk=int(self.adp_topk),
            tau=float(self.adp_tau),
        )

    @staticmethod
    def _init_adaptive_ms_logits(num_scales: int, init_mode: str = "favor_largest") -> torch.Tensor:
        if num_scales <= 0:
            raise ValueError("adaptive_ms_topks must be non-empty.")
        if init_mode == "uniform":
            return torch.zeros(num_scales)
        if init_mode == "favor_largest":
            if num_scales == 1:
                return torch.zeros(1)
            return torch.linspace(-2.94, 0.0, num_scales)
        raise ValueError(
            f"Unsupported adaptive_ms_init: {init_mode}. "
            "Expected 'uniform' or 'favor_largest'."
        )

    def _build_shared_adaptive_logits(self) -> torch.Tensor:
        src = F.normalize(self.adaptive_src, p=2, dim=-1)
        dst = F.normalize(self.adaptive_dst, p=2, dim=-1)
        return torch.matmul(src, dst.transpose(0, 1)) / sqrt(self.adp_hidden_dim)

    def _build_scale_adaptive_logits(self, scale_idx: int) -> torch.Tensor:
        src = F.normalize(self.adaptive_ms_src_list[scale_idx], p=2, dim=-1)
        dst = F.normalize(self.adaptive_ms_dst_list[scale_idx], p=2, dim=-1)
        return torch.matmul(src, dst.transpose(0, 1)) / sqrt(self.adp_hidden_dim)

    def _build_adaptive_adj_at_k(self, logits: torch.Tensor, topk: int) -> torch.Tensor:
        masked = mask_topk(logits, int(topk))
        return torch.softmax(masked / max(self.adp_tau, 1e-6), dim=-1)

    def _fuse_adaptive_ms_deltas(self, deltas: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.adaptive_ms_logits, dim=0)
        stacked = torch.stack(deltas, dim=0)
        fused = torch.einsum("k,kbtn->btn", weights, stacked)
        entropy = -(weights * (weights + 1e-12).log()).sum()
        self.last_adaptive_ms_weights = weights.detach()
        self.last_adaptive_ms_entropy = float(entropy.detach().item())
        return fused, weights

    def _refine_adaptive_multiscale(self, output: torch.Tensor, history_flow: torch.Tensor) -> torch.Tensor:
        del history_flow  # adaptive-only: no dynamic/static/hybrid graph input.
        x = output.squeeze(-1)
        deltas: list[torch.Tensor] = []
        if self.adaptive_ms_share_logits:
            logits_s = self._build_shared_adaptive_logits()
            self.last_adaptive_adj = self._build_adaptive_adj_at_k(
                logits_s, self.adaptive_ms_topks[-1]
            ).detach()
            for topk in self.adaptive_ms_topks:
                adj_k = self._build_adaptive_adj_at_k(logits_s, topk)
                deltas.append(apply_adj(x, adj_k))
        else:
            for scale_idx, topk in enumerate(self.adaptive_ms_topks):
                logits_k = self._build_scale_adaptive_logits(scale_idx)
                adj_k = self._build_adaptive_adj_at_k(logits_k, topk)
                if scale_idx == len(self.adaptive_ms_topks) - 1:
                    self.last_adaptive_adj = adj_k.detach()
                deltas.append(apply_adj(x, adj_k))

        fused_delta, _ = self._fuse_adaptive_ms_deltas(deltas)
        return output + self.adaptive_ms_alpha * fused_delta.unsqueeze(-1)

    def get_adaptive_ms_diagnostics(self) -> dict:
        weights = self.last_adaptive_ms_weights
        if weights is None:
            weights = torch.softmax(self.adaptive_ms_logits, dim=0).detach()
        return {
            "adaptive_ms_weights": weights,
            "adaptive_ms_topks": list(self.adaptive_ms_topks),
            "adaptive_ms_alpha": self.adaptive_ms_alpha,
            "adaptive_ms_entropy": self.last_adaptive_ms_entropy,
        }

    def _build_hybrid_adj(self, history_flow):
        batch_size = history_flow.shape[0]
        dynamic_adj = self._build_dynamic_adj(history_flow)
        adaptive_adj = self._build_adaptive_adj().unsqueeze(0).expand(batch_size, -1, -1)
        if self.adj_mx is not None:
            static_adj = row_normalize(self.adj_mx).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            static_adj = torch.eye(self.node_size, device=history_flow.device).unsqueeze(0).expand(batch_size, -1, -1)

        weights = torch.softmax(self.hybrid_logits, dim=0)
        return weights[0] * static_adj + weights[1] * adaptive_adj + weights[2] * dynamic_adj

    def _static_adj_batch(self, history_flow, batch_size):
        if self.adj_mx is not None:
            return row_normalize(self.adj_mx).unsqueeze(0).expand(batch_size, -1, -1)
        return torch.eye(self.node_size, device=history_flow.device).unsqueeze(0).expand(batch_size, -1, -1)

    def _build_mode_adj(self, history_flow, mode: str):
        batch_size = history_flow.shape[0]
        static_adj = self._static_adj_batch(history_flow, batch_size)

        if mode == "static_only":
            return static_adj

        adaptive_adj = self._build_adaptive_adj().unsqueeze(0).expand(batch_size, -1, -1)
        if mode == "adaptive_only":
            return adaptive_adj

        dynamic_adj = self._build_dynamic_adj(history_flow)
        if mode == "dynamic_only":
            return dynamic_adj

        if mode == "hybrid":
            return self._build_hybrid_adj(history_flow)

        if mode == "static_adaptive":
            return 0.5 * static_adj + 0.5 * adaptive_adj

        if mode == "static_dynamic":
            return 0.5 * static_adj + 0.5 * dynamic_adj

        if mode == "adaptive_dynamic":
            return 0.5 * adaptive_adj + 0.5 * dynamic_adj

        raise ValueError(f"Unsupported post_spatial_mode: {mode}")

    def get_enhanced_spatial_embedding(self, spa_codebook):
        """Scheme A/C: enhance spatial codebook before encoders."""
        if not self.if_spatial or spa_codebook is None:
            return None
        if self.use_gcn and self.adj_mx is not None and self.gcn1 is not None:
            emb = self.gcn1(spa_codebook, self.adj_mx)
            emb = self.gcn2(emb, self.adj_mx)
            return emb
        return None

    def refine_prediction(self, output, history_flow):
        """Apply B/C/D or adaptive-only refinement on prediction output.

        Args:
            output: [B, T, N, 1]
            history_flow: [B, L, N]
        """
        if self.post_spatial_mode == "none":
            return output

        if self.post_spatial_mode == "adaptive_multiscale_only":
            return self._refine_adaptive_multiscale(output, history_flow)

        if self.post_spatial_mode == "stae_spatial_attention":
            # Strict replace: Y_final = Projection(SpatialTransformer(Embed(Y_T)+E_adp)).
            # Internal residuals keep self information; do not add Y_T outside.
            if self.stae_spatial_block is None:
                raise RuntimeError(
                    "stae_spatial_block is required for stae_spatial_attention mode."
                )
            del history_flow
            return self.stae_spatial_block(output)

        x = output.squeeze(-1)

        if self.post_spatial_mode == "adaptive_self_replace":
            # Strict replace: Y_hat = A_self Y_T  (no residual, no hybrid_alpha).
            adp_adj = self._build_adaptive_adj_with_self()
            self.last_adaptive_adj = adp_adj.detach()
            return apply_adj(x, adp_adj).unsqueeze(-1)

        if self.post_spatial_mode == "adaptive_cluster_mix":
            if self.cluster_adj is None:
                raise RuntimeError("cluster_adj is required for adaptive_cluster_mix mode.")
            adp_adj = self._build_adaptive_adj()
            if adp_adj.shape != self.cluster_adj.shape:
                raise RuntimeError(
                    f"adaptive_cluster_mix shape mismatch: A_adp {tuple(adp_adj.shape)} "
                    f"vs A_cluster {tuple(self.cluster_adj.shape)}"
                )
            self.last_adaptive_adj = adp_adj.detach()
            self.last_cluster_adj = self.cluster_adj.detach()
            lam = self.cluster_graph_mix_lambda
            a_mix = (1.0 - lam) * adp_adj + lam * self.cluster_adj
            self.last_mix_adj = a_mix.detach()
            with torch.no_grad():
                self.last_adp_density = float((adp_adj > 1e-6).float().mean().item())
                self.last_cluster_density = float((self.cluster_adj > 1e-6).float().mean().item())
                self.last_adj_mean_abs_diff = float(
                    (adp_adj - self.cluster_adj).abs().mean().item()
                )
            adj = a_mix.unsqueeze(0).expand(history_flow.shape[0], -1, -1)
            refine = apply_adj(x, adj).unsqueeze(-1)
            return output + self.hybrid_alpha * refine

        explicit_modes = {
            "hybrid",
            "static_only",
            "adaptive_only",
            "dynamic_only",
            "static_adaptive",
            "static_dynamic",
            "adaptive_dynamic",
        }
        if self.post_spatial_mode in explicit_modes:
            if self.post_spatial_mode == "adaptive_only":
                adp_adj = self._build_adaptive_adj()
                # Detach for safe graph spectral eigendecomposition (first version).
                self.last_adaptive_adj = adp_adj.detach()
                adj = adp_adj.unsqueeze(0).expand(history_flow.shape[0], -1, -1)
            else:
                adj = self._build_mode_adj(history_flow, self.post_spatial_mode)
            refine = apply_adj(x, adj).unsqueeze(-1)
            if self.g1_stagewise_s1_gate is not None:
                spatial_residual = refine - output
                return output + self.hybrid_alpha * self.g1_stagewise_s1_gate * spatial_residual
            return output + self.hybrid_alpha * refine

        if self.use_hybrid_graph:
            hybrid_adj = self._build_hybrid_adj(history_flow)
            refine = apply_adj(x, hybrid_adj).unsqueeze(-1)
            return output + self.hybrid_alpha * refine

        if self.use_dynamic_spatial:
            dyn_adj = self._build_dynamic_adj(history_flow)
            refine = apply_adj(x, dyn_adj).unsqueeze(-1)
            return output + self.dyn_alpha * refine

        if self.use_adaptive_adj:
            adp_adj = self._build_adaptive_adj()
            self.last_adaptive_adj = adp_adj.detach()
            refine = apply_adj(x, adp_adj).unsqueeze(-1)
            return output + self.adp_alpha * refine

        if self.use_lightweight_spatial and self.adj_mx is not None:
            static_adj = row_normalize(self.adj_mx)
            smooth = apply_adj(x, static_adj).unsqueeze(-1) - output
            return output + self.light_alpha * smooth

        return output

    def get_cluster_mix_diagnostics(self) -> dict[str, float]:
        return {
            "a_adp_density": float(self.last_adp_density or 0.0),
            "a_cluster_density": float(self.last_cluster_density or 0.0),
            "a_cluster_adp_mean_abs_diff": float(self.last_adj_mean_abs_diff or 0.0),
            "cluster_graph_mix_lambda": float(self.cluster_graph_mix_lambda),
        }

    def get_adaptive_adj(self):
        """Return adaptive adjacency used in adaptive-only post-spatial refinement."""
        if self.last_adaptive_adj is not None:
            return self.last_adaptive_adj
        if self.adaptive_src is not None and self.post_spatial_mode == "adaptive_self_replace":
            return self._build_adaptive_adj_with_self().detach()
        if self.use_adaptive_adj or self.adaptive_src is not None:
            return self._build_adaptive_adj().detach()
        raise RuntimeError("Adaptive adjacency is not available for the current spatial configuration.")
