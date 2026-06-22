"""Temporal and graph-spectral operators for ST-FSF."""
from __future__ import annotations

import os
import pickle

import torch
import torch.nn.functional as F


def temporal_resize(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """x: [B, T, N, C] -> [B, target_len, N, C]."""
    b, t, n, c = x.shape
    if t == target_len:
        return x
    y = x.permute(0, 2, 3, 1).reshape(b * n, c, t)
    y = F.interpolate(y, size=target_len, mode="linear", align_corners=False)
    return y.reshape(b, n, c, target_len).permute(0, 3, 1, 2)


def temporal_pool(y: torch.Tensor, target_len: int) -> torch.Tensor:
    """y: [B, F, N, C] -> [B, target_len, N, C]."""
    b, f_len, n, c = y.shape
    if f_len == target_len:
        return y
    if f_len % target_len == 0:
        g = f_len // target_len
        return y.reshape(b, target_len, g, n, c).mean(dim=2)
    z = y.permute(0, 2, 3, 1).reshape(b * n, c, f_len)
    z = F.adaptive_avg_pool1d(z, target_len)
    return z.reshape(b, n, c, target_len).permute(0, 3, 1, 2)


def graph_project(x: torch.Tensor, v: torch.Tensor, q: int) -> torch.Tensor:
    """Low graph-frequency projection. x: [B,T,N,C], V: [N,N] -> [B,T,N,C]."""
    if q is None or q >= v.size(0):
        return x
    vq = v[:, :q].to(device=x.device, dtype=x.dtype)
    coeff = torch.einsum("nq,btnc->btqc", vq, x)
    return torch.einsum("nq,btqc->btnc", vq, coeff)


def native_graph_coeff(x: torch.Tensor, v: torch.Tensor, q: int) -> torch.Tensor:
    """x: [B,T,N,C], V: [N,N] -> [B,T,q,C]."""
    if q is None or q >= v.size(0):
        return x
    vq = v[:, :q].to(device=x.device, dtype=x.dtype)
    return torch.einsum("nq,btnc->btqc", vq, x)


def st_project(x: torch.Tensor, v: torch.Tensor, r: int, q: int, full_len: int) -> torch.Tensor:
    """Project to joint (r,q) resolution and lift back to [B,F,N,C]."""
    x_t = temporal_resize(temporal_pool(x, r), full_len)
    return graph_project(x_t, v, q)


def build_normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """Return L = I - D^{-1/2} A D^{-1/2}."""
    adj = adj.float()
    adj = torch.clamp(adj, min=0.0)
    adj = 0.5 * (adj + adj.T)

    n = adj.size(0)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    d_inv_sqrt = torch.diag(deg_inv_sqrt)

    eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
    return eye - d_inv_sqrt @ adj @ d_inv_sqrt


def compute_graph_fourier_basis(adj: torch.Tensor) -> torch.Tensor:
    """Return eigenvectors ordered by ascending eigenvalues."""
    l = build_normalized_laplacian(adj)
    evals, evecs = torch.linalg.eigh(l)
    idx = torch.argsort(evals)
    return evecs[:, idx].contiguous()


def load_adj_matrix(adj_mx_path: str, node_size: int) -> torch.Tensor:
    """Load adjacency matrix from pickle; raise if missing or invalid."""
    if not adj_mx_path:
        raise FileNotFoundError(
            "adj_mx_path is required for STForecastStateFlow but was not provided."
        )
    if not os.path.exists(adj_mx_path):
        raise FileNotFoundError(f"adj_mx not found at {adj_mx_path}")

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

    if not hasattr(adj_obj, "shape") or len(adj_obj.shape) != 2:
        raise ValueError(f"Invalid adjacency at {adj_mx_path}: expected 2D matrix.")

    adj = torch.as_tensor(adj_obj, dtype=torch.float32)
    if adj.size(0) != node_size or adj.size(1) != node_size:
        raise ValueError(
            f"adj shape {tuple(adj.shape)} != ({node_size}, {node_size}) at {adj_mx_path}"
        )
    return adj


def lift_native_to_full(native_z: torch.Tensor, v: torch.Tensor, full_len: int) -> torch.Tensor:
    """native_z: [B,r,q,1], V: [N,N] -> [B,full_len,N,1]."""
    _, _, q, _ = native_z.shape
    vq = v[:, :q].to(device=native_z.device, dtype=native_z.dtype)
    x = torch.einsum("nq,brqc->brnc", vq, native_z)
    return temporal_resize(x, full_len)
