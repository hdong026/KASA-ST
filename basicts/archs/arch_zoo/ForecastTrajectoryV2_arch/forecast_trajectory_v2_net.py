"""ForecastTrajectoryV2: shared KASA state-conditioned transition with latent H + explicit Z."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn

from basicts.archs.arch_zoo.ChainForecasting_arch.gcn import ABCDSpatialModule
from basicts.archs.arch_zoo.ChainForecasting_arch.kasa_temporal_step import interpolate_forecast
from basicts.archs.arch_zoo.ChainForecasting_arch.mlp import MultiLayerPerceptron
from basicts.archs.arch_zoo.ForecastTrajectory_arch.trajectory_graph import ForecastTrajectoryGraph
from basicts.archs.arch_zoo.ForecastTrajectoryV2_arch.kasa_token_encoder import (
    KASATokenDownsampEncoder,
    KASATokenPatchEncoder,
    _prepare_patch_inputs,
)


def _cross_attend(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """Per-node cross-attention. q: [B,Tq,N,D], kv: [B,Tk,N,D] -> [B,Tq,N,D]."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    attn = torch.einsum("btnd,bknd->bntk", q, kv) * scale
    w = torch.softmax(attn, dim=-1)
    return torch.einsum("bntk,bknd->btnd", w, kv)


class StateHyperNetwork(nn.Module):
    """One shared hypernet: source/dest embeddings + continuous features -> FiLM + gate."""

    def __init__(self, n_states: int, d_model: int, cond_dim: int = 64):
        super().__init__()
        self.emb_prev = nn.Embedding(n_states + 1, cond_dim)  # include START=0
        self.emb_next = nn.Embedding(n_states + 1, cond_dim)
        self.cont = nn.Sequential(nn.Linear(4, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.fuse = nn.Sequential(nn.Linear(cond_dim * 3, cond_dim), nn.SiLU())
        self.film = nn.Linear(cond_dim, 2 * d_model)
        self.gate = nn.Linear(cond_dim, 1)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        with torch.no_grad():
            self.film.bias[:d_model].fill_(1.0)
            self.gate.bias.fill_(2.0)  # sigmoid ~ 0.88: learned path dominates interpolation

    def forward(self, idx_prev: int, idx_next: int, s_prev: int, s_next: int, H: int, batch: int, device, dtype):
        sp = torch.tensor([int(idx_prev)], device=device)
        sn = torch.tensor([int(idx_next)], device=device)
        e_prev = self.emb_prev(sp).expand(batch, -1)
        e_next = self.emb_next(sn).expand(batch, -1)
        r1 = float(s_prev) / float(H)
        r2 = float(s_next) / float(H)
        r3 = (float(s_next) - float(s_prev)) / float(H)
        r4 = math.log((float(s_next) + 1.0) / (float(s_prev) + 1.0))
        cont = self.cont(torch.tensor([[r1, r2, r3, r4]], device=device, dtype=dtype).expand(batch, -1))
        h = self.fuse(torch.cat([e_prev, e_next, cont], dim=-1))
        gb = self.film(h)
        gamma, beta = gb.chunk(2, dim=-1)
        g = torch.sigmoid(self.gate(h)).view(batch, 1, 1, 1)
        return h, gamma, beta, g


@dataclass
class HistoryState:
    tokens: torch.Tensor
    pooled: torch.Tensor
    history_flow: torch.Tensor
    history_data: torch.Tensor


@dataclass
class ForecastState:
    H: Optional[torch.Tensor]
    Z: Optional[torch.Tensor]
    s: int


class KASAStateTransitionCell(nn.Module):
    """Shared KASA temporal cell + dest queries + learned source alignment."""

    def __init__(self, d_in: int, d_model: int, cy: int, num_layer: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.cy = cy
        self.in_proj = nn.Linear(d_in, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.dest_mlp = nn.Sequential(
            *[MultiLayerPerceptron(d_model, d_model) for _ in range(num_layer)]
        )
        self.z_in = nn.Linear(cy, d_model)
        self.head = nn.Linear(d_model, cy)
        self.start_token = nn.Parameter(torch.zeros(d_model))
        self.pos_proj = nn.Linear(16, d_model)

    def dest_queries(self, s_next: int, batch: int, n: int, device, dtype, spa: Optional[torch.Tensor]):
        j = torch.arange(s_next, device=device, dtype=dtype)
        u = (j + 0.5) / float(s_next)
        freqs = (2 * math.pi) * torch.arange(1, 9, device=device, dtype=dtype)
        feat = torch.cat([torch.sin(u[:, None] * freqs), torch.cos(u[:, None] * freqs)], dim=-1)
        q = self.pos_proj(feat).view(1, s_next, 1, -1).expand(batch, -1, n, -1)
        if spa is not None:
            q = q + spa.view(1, 1, n, -1)
        return q

    def forward(
        self,
        h_x: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        z_prev: Optional[torch.Tensor],
        s_prev: int,
        s_next: int,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        gate: torch.Tensor,
        spa_proj: Optional[torch.Tensor],
        history_flow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        b, _, n, _ = h_x.shape
        device, dtype = h_x.device, h_x.dtype
        q = self.dest_queries(s_next, b, n, device, dtype, spa_proj)
        if int(s_prev) == 0:
            q = q + self.start_token.view(1, 1, 1, -1)
        hx = self.in_proj(h_x)
        hist_ctx = _cross_attend(self.q_proj(q), hx)
        if h_prev is not None:
            src_ctx = _cross_attend(self.q_proj(q), self.in_proj(h_prev) if h_prev.shape[-1] != self.d_model else h_prev)
        else:
            src_ctx = torch.zeros_like(hist_ctx)
        if z_prev is not None:
            z_tok = self.z_in(z_prev)
            a_z = _cross_attend(self.q_proj(q), z_tok)
            z_interp = interpolate_forecast(z_prev, s_next)
        else:
            a_z = torch.zeros_like(hist_ctx)
            z_interp = hist_ctx.new_zeros(b, s_next, n, self.cy)
        fused = hist_ctx + src_ctx + a_z
        while gamma.ndim < fused.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        fused = gamma * fused + beta
        h = self.dest_mlp(fused.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        z_learned = self.head(h)
        z_next = gate * z_learned + (1.0 - gate) * z_interp
        aux = {
            "gate_mean": float(gate.detach().mean().item()),
            "z_learned": z_learned,
            "z_interp": z_interp,
        }
        return h, z_next, aux


class ForecastTrajectoryV2Net(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.node_size = int(model_args.get("node_size", 307))
        self.input_len = int(model_args.get("input_len", 12))
        self.output_len = int(model_args.get("output_len", 12))
        self.input_dim = int(model_args.get("input_dim", 3))
        self.output_dim = int(model_args.get("output_dim", 1))
        self.patch_len = int(model_args.get("patch_len", 3))
        self.stride = int(model_args.get("stride", 4))
        self.td_size = int(model_args.get("td_size", 288))
        self.dw_size = int(model_args.get("dw_size", 7))
        self.d_td = int(model_args.get("d_td", 32))
        self.d_dw = int(model_args.get("d_dw", 32))
        self.d_d = int(model_args.get("d_d", 32))
        self.d_spa = int(model_args.get("d_spa", 32))
        self.num_layer = int(model_args.get("num_layer", 2))
        requested_d = model_args.get("d_model", None)
        self.if_time_in_day = True
        self.if_day_in_week = True
        self.if_spatial = True
        states = list(model_args.get("states", [2, 3, 4, 6, 12]))
        self.graph = ForecastTrajectoryGraph(H=self.output_len, states=states)
        self.state_to_idx = {0: 0, **{s: i + 1 for i, s in enumerate(self.graph.states)}}

        self.td_codebook = nn.Parameter(torch.empty(self.td_size, self.d_td))
        self.dw_codebook = nn.Parameter(torch.empty(self.dw_size, self.d_dw))
        self.spa_codebook = nn.Parameter(torch.empty(self.node_size, self.d_spa))
        nn.init.xavier_uniform_(self.td_codebook)
        nn.init.xavier_uniform_(self.dw_codebook)
        nn.init.xavier_uniform_(self.spa_codebook)

        enc_kw = dict(
            td_size=self.td_size,
            dw_size=self.dw_size,
            td_codebook=self.td_codebook,
            dw_codebook=self.dw_codebook,
            spa_codebook=self.spa_codebook,
            if_time_in_day=True,
            if_day_in_week=True,
            if_spatial=True,
            patch_len=self.patch_len,
            stride=self.stride,
            d_d=self.d_d,
            d_td=self.d_td,
            d_dw=self.d_dw,
            d_spa=self.d_spa,
            output_len=self.output_len,
            num_layer=self.num_layer,
        )
        self.patch_hist = KASATokenPatchEncoder(
            input_dim=3, patch_data_input_mode="all", patch_embedding_mode="serial_concat", **enc_kw
        )
        self.down_hist = KASATokenDownsampEncoder(input_dim=3, **enc_kw)
        self.patch_cond = KASATokenPatchEncoder(
            input_dim=4, patch_data_input_mode="all", patch_embedding_mode="serial_concat", **enc_kw
        )
        self.down_cond = KASATokenDownsampEncoder(input_dim=4, **enc_kw)
        token_dim = int(self.patch_hist.hidden_dim + self.d_spa)
        # Match KASA temporal_encoder width (hidden_dim + d_spa = 192) unless overridden.
        self.d_model = int(requested_d) if requested_d not in (None, 0) else token_dim
        if self.d_model == token_dim:
            self.token_proj = nn.Identity()
        else:
            self.token_proj = nn.Linear(token_dim, self.d_model)
        self.hyper = StateHyperNetwork(len(self.graph.states), self.d_model)
        self.cell = KASAStateTransitionCell(self.d_model, self.d_model, self.output_dim, self.num_layer)
        self.hist_skip_scale = nn.Parameter(torch.zeros(1))
        self.spa_proj = nn.Linear(self.d_spa, self.d_model)
        self.spatial = ABCDSpatialModule(
            node_size=self.node_size,
            input_len=self.input_len,
            d_spa=self.d_spa,
            if_spatial=True,
            spatial_scheme=model_args.get("spatial_scheme", "C"),
            adj_mx_path=model_args.get("adj_mx_path"),
            use_adaptive_adj=True,
            adp_hidden_dim=32,
            adp_topk=20,
            adp_tau=0.5,
            adp_alpha=0.1,
            use_hybrid_graph=True,
            hybrid_alpha=0.2,
            post_spatial_mode="adaptive_only",
        )
        self._hist_count = 0
        n_core = sum(p.numel() for p in list(self.patch_hist.parameters()) + list(self.down_hist.parameters())
                     + list(self.patch_cond.parameters()) + list(self.down_cond.parameters())
                     + list(self.cell.dest_mlp.parameters()))
        print(f"[ForecastTrajectoryV2] unique shared KASA-core params ≈ {n_core}")
        print(f"[ForecastTrajectoryV2] total params = {sum(p.numel() for p in self.parameters())}")

    def reset_history_encode_count(self):
        self._hist_count = 0

    @property
    def history_encode_count(self):
        return self._hist_count

    def encode_history_tokens(self, x: torch.Tensor, use_cond: bool, z_aux: Optional[torch.Tensor]):
        if use_cond and z_aux is not None:
            cond = interpolate_forecast(z_aux, self.input_len)
            xin = torch.cat([x[..., :3], cond], dim=-1)
            patch, down = _prepare_patch_inputs(xin, self.input_len, self.patch_len, self.stride)
            t_p = self.patch_cond.forward_tokens(patch, self.spa_codebook)
            t_d = self.down_cond.forward_tokens(down, self.spa_codebook)
        else:
            xin = x[..., :3]
            patch, down = _prepare_patch_inputs(xin, self.input_len, self.patch_len, self.stride)
            t_p = self.patch_hist.forward_tokens(patch, self.spa_codebook)
            t_d = self.down_hist.forward_tokens(down, self.spa_codebook)
        tokens = torch.cat([t_p, t_d], dim=1)
        return self.token_proj(tokens)

    def prepare_history(self, X: torch.Tensor) -> HistoryState:
        self._hist_count += 1
        tokens = self.encode_history_tokens(X, False, None)
        return HistoryState(
            tokens=tokens,
            pooled=tokens.mean(dim=1).mean(dim=1),
            history_flow=X[..., 0],
            history_data=X,
        )

    def transition(
        self,
        history: HistoryState,
        H_prev: Optional[torch.Tensor],
        Z_prev: Optional[torch.Tensor],
        s_prev: int,
        s_next: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if not self.graph.is_legal_edge(int(s_prev), int(s_next)):
            raise ValueError(f"illegal {s_prev}->{s_next}")
        b = history.tokens.shape[0]
        # auxiliary cond encoding of history+Z (original F2F prev-condition idea)
        if Z_prev is not None:
            hx = 0.5 * history.tokens + 0.5 * self.encode_history_tokens(
                history.history_data, True, Z_prev
            )
        else:
            hx = history.tokens
        e, gamma, beta, g = self.hyper(
            self.state_to_idx[int(s_prev)],
            self.state_to_idx[int(s_next)],
            int(s_prev),
            int(s_next),
            self.graph.H,
            b,
            hx.device,
            hx.dtype,
        )
        spa = self.spa_proj(self.spa_codebook)
        h_next, z_next, aux = self.cell(
            hx, H_prev, Z_prev, int(s_prev), int(s_next), gamma, beta, g, spa, history.history_flow
        )
        hist_skip = interpolate_forecast(history.history_flow.unsqueeze(-1), int(s_next))
        z_next = z_next + self.hist_skip_scale * hist_skip
        z_next = self.spatial.refine_prediction(z_next, history.history_flow)
        aux["e_transition"] = e
        aux["kasa_temporal_core"] = True
        return h_next, z_next, aux

    def rollout(self, X: torch.Tensor, trajectory: Sequence[int], history: Optional[HistoryState] = None):
        if history is None:
            history = self.prepare_history(X)
        out_h, out_z = {}, {}
        h_prev, z_prev, s_prev = None, None, 0
        for s_next in [int(s) for s in trajectory]:
            h_prev, z_prev, _ = self.transition(history, h_prev, z_prev, s_prev, s_next)
            out_h[s_next] = h_prev
            out_z[s_next] = z_prev
            s_prev = s_next
        return {"H": out_h, "Z": out_z, "history": history}

    def rollout_prefix_dag(self, X: torch.Tensor):
        """Compute every unique prefix exactly once. Returns states + n_transitions."""
        history = self.prepare_history(X)
        prefixes = _all_prefixes(self.graph)
        states: dict[tuple[int, ...], tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            (): (None, None)
        }
        n_trans = 0
        for pref in prefixes:
            if not pref:
                continue
            parent = pref[:-1]
            s_prev = 0 if not parent else int(parent[-1])
            s_next = int(pref[-1])
            h_p, z_p = states[parent]
            h_n, z_n, _ = self.transition(history, h_p, z_p, s_prev, s_next)
            states[pref] = (h_n, z_n)
            n_trans += 1
        return history, states, n_trans

    def param_breakdown(self) -> dict:
        def n(mod):
            return sum(p.numel() for p in mod.parameters())
        kasa_enc = n(self.patch_hist) + n(self.down_hist) + n(self.patch_cond) + n(self.down_cond)
        kasa = kasa_enc + n(self.cell.dest_mlp)
        adapt = n(self.hyper) + n(self.cell.q_proj) + n(self.cell.pos_proj) + n(self.cell.z_in)
        if not isinstance(self.token_proj, nn.Identity):
            adapt += n(self.token_proj)
        return {
            "total": n(self),
            "shared_kasa_core": kasa,
            "shared_transition": kasa_enc + n(self.cell) + n(self.spatial),
            "state_adapters": adapt,
            "spatial": n(self.spatial),
            "uses_module_dict_transitions": False,
            "uses_kasa_temporal_core": True,
        }

    def transition_parameter_ids(self) -> list[int]:
        return [id(p) for p in self.cell.parameters()] + [id(p) for p in self.hyper.parameters()]

    def uses_kasa_temporal_core(self) -> bool:
        from basicts.archs.arch_zoo.ChainForecasting_arch.mlp import MultiLayerPerceptron
        from basicts.archs.arch_zoo.ChainForecasting_arch.patch_emb import PatchEncoder
        from basicts.archs.arch_zoo.ChainForecasting_arch.downsamp_emb import DownsampEncoder
        ok_mlp = any(isinstance(m, MultiLayerPerceptron) for m in self.cell.dest_mlp.modules())
        ok_patch = isinstance(self.patch_hist, PatchEncoder)
        ok_down = isinstance(self.down_hist, DownsampEncoder)
        return bool(ok_mlp and ok_patch and ok_down)

    def warm_start_from_canonical(self, teacher_sd: dict) -> dict:
        """Copy compatible KASA trunks / spatial / codebooks. Print per-group status."""
        own = self.state_dict()
        copied, partial, new_init = [], [], []
        groups = {
            "codebooks": ["td_codebook", "dw_codebook", "spa_codebook"],
            "spatial": [k for k in own if k.startswith("spatial.")],
            "patch_hist": [k for k in own if k.startswith("patch_hist.") and "projection1" not in k],
            "down_hist": [k for k in own if k.startswith("down_hist.") and "projection1" not in k],
            "patch_cond": [k for k in own if k.startswith("patch_cond.") and "projection1" not in k],
            "down_cond": [k for k in own if k.startswith("down_cond.") and "projection1" not in k],
            "dest_mlp": [k for k in own if k.startswith("cell.dest_mlp.")],
            "token_proj": [k for k in own if k.startswith("token_proj.")],
            "hyper": [k for k in own if k.startswith("hyper.")],
            "cell_align": [
                k for k in own
                if k.startswith("cell.") and not k.startswith("cell.dest_mlp.")
            ],
            "spa_proj": [k for k in own if k.startswith("spa_proj.")],
            "hist_skip": [k for k in own if k.startswith("hist_skip")],
        }
        src_map = {
            "codebooks": {
                "td_codebook": "td_codebook",
                "dw_codebook": "dw_codebook",
                "spa_codebook": "spa_codebook",
            },
            "spatial": None,  # prefix remap spatial. <- spatial_module.
            "patch_hist": "temporal_steps.0.patch_encoder.",
            "down_hist": "temporal_steps.0.downsamp_encoder.",
            "patch_cond": "temporal_steps.1.patch_encoder_cond.",
            "down_cond": "temporal_steps.1.downsamp_encoder_cond.",
            "dest_mlp": "temporal_steps.2.patch_encoder.temporal_encoder.",
        }
        loaded = {}
        # codebooks
        n_ok = n_try = 0
        for dst, src in src_map["codebooks"].items():
            n_try += 1
            if src in teacher_sd and tuple(teacher_sd[src].shape) == tuple(own[dst].shape):
                loaded[dst] = teacher_sd[src]
                n_ok += 1
        _record_group("codebooks", n_ok, n_try, copied, partial, new_init)
        # spatial
        n_ok = n_try = 0
        for dst in groups["spatial"]:
            src = "spatial_module." + dst[len("spatial.") :]
            n_try += 1
            if src in teacher_sd and tuple(teacher_sd[src].shape) == tuple(own[dst].shape):
                loaded[dst] = teacher_sd[src]
                n_ok += 1
        _record_group("spatial", n_ok, n_try, copied, partial, new_init)
        for gname, prefix in [
            ("patch_hist", src_map["patch_hist"]),
            ("down_hist", src_map["down_hist"]),
            ("patch_cond", src_map["patch_cond"]),
            ("down_cond", src_map["down_cond"]),
        ]:
            n_ok = n_try = 0
            for dst in groups[gname]:
                src = prefix + dst.split(".", 1)[1]
                n_try += 1
                if src in teacher_sd and tuple(teacher_sd[src].shape) == tuple(own[dst].shape):
                    loaded[dst] = teacher_sd[src]
                    n_ok += 1
            _record_group(gname, n_ok, n_try, copied, partial, new_init)
        n_ok = n_try = 0
        dest_prefix = src_map["dest_mlp"]
        for dst in groups["dest_mlp"]:
            src = dest_prefix + dst.split("cell.dest_mlp.", 1)[1]
            n_try += 1
            if src in teacher_sd and tuple(teacher_sd[src].shape) == tuple(own[dst].shape):
                loaded[dst] = teacher_sd[src]
                n_ok += 1
        _record_group("dest_mlp_from_F12_temporal_encoder", n_ok, n_try, copied, partial, new_init)
        for gname in ["token_proj", "hyper", "cell_align", "spa_proj", "hist_skip"]:
            n_try = len(groups[gname])
            _record_group(gname, 0, n_try, copied, partial, new_init)
        missing, unexpected = self.load_state_dict(loaded, strict=False)
        report = {
            "COPIED": copied,
            "PARTIALLY_COPIED": partial,
            "NEW_INIT": new_init,
            "n_tensors_copied": len(loaded),
            "missing_after_load": len(missing),
        }
        print("[V2 warm-start]", json_safe(report), flush=True)
        for row in copied + partial + new_init:
            print(f"  {row['status']:16s} {row['group']} ({row['copied']}/{row['tried']})", flush=True)
        return report


def _record_group(name, n_ok, n_try, copied, partial, new_init):
    if n_try <= 0:
        new_init.append({"group": name, "status": "NEW_INIT", "copied": 0, "tried": 0})
        return
    if n_ok == n_try:
        copied.append({"group": name, "status": "COPIED", "copied": n_ok, "tried": n_try})
    elif n_ok > 0:
        partial.append({"group": name, "status": "PARTIALLY_COPIED", "copied": n_ok, "tried": n_try})
    else:
        new_init.append({"group": name, "status": "NEW_INIT", "copied": 0, "tried": n_try})


def json_safe(obj):
    import json as _json
    return _json.dumps(obj, default=str)


def _all_prefixes(graph: ForecastTrajectoryGraph) -> list[tuple[int, ...]]:
    seen = set()
    out = []
    for tau in graph.terminal_trajectories():
        for i in range(len(tau) + 1):
            p = tuple(int(s) for s in tau[:i])
            if p not in seen:
                seen.add(p)
                out.append(p)
    out.sort(key=lambda p: (len(p), p))
    return out


def expected_dag_transitions(graph: ForecastTrajectoryGraph) -> int:
    return sum(1 for p in _all_prefixes(graph) if p)
