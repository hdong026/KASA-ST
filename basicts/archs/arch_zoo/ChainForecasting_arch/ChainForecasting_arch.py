"""ChainForecasting: chain of future prediction states at increasing resolutions."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from basicts.archs.arch_zoo.KASA_arch_v2.gcn import ABCDSpatialModule
from basicts.archs.arch_zoo.ChainForecasting_arch.chain_modules import (
    CoarseForecastDecoder,
    DownsampleMemoryEncoder,
    ForecastTransitionBlock,
    FutureQueryEmbedding,
    HistoryPatchEncoder,
    pool_target_to_length,
    upsample_state,
)


class ChainForecasting(nn.Module):
    """Forecast chain X -> Y_hat_3 -> Y_hat_6 -> Y_hat_12 (configurable)."""

    def __init__(self, **model_args):
        super().__init__()
        self.node_size = model_args["node_size"]
        self.input_len = model_args["input_len"]
        self.output_len = model_args["output_len"]
        self.input_dim = model_args.get("input_dim", 4)
        self.main_input_dim = model_args.get("main_input_dim", 3)

        self.chain_lengths: List[int] = list(model_args.get("chain_lengths", [3, 6, 12]))
        if self.chain_lengths[-1] != self.output_len:
            raise ValueError(
                f"chain_lengths must end with output_len={self.output_len}, got {self.chain_lengths}"
            )

        self.d_model = model_args.get("d_model", 64)
        self.num_heads = model_args.get("num_heads", 4)
        self.ffn_dim = model_args.get("ffn_dim", 128)
        self.patch_len = model_args.get("patch_len", 3)
        self.patch_stride = model_args.get("patch_stride", 3)
        self.use_downsample_memory = model_args.get("use_downsample_memory", True)
        self.use_final_spatial_refine = model_args.get("use_final_spatial_refine", True)
        self.post_spatial_mode = model_args.get("post_spatial_mode", "adaptive_only")
        self.td_size = model_args.get("td_size", 288)
        self.dropout = model_args.get("dropout", 0.0)

        max_patches = max(4, (self.input_len - self.patch_len) // self.patch_stride + 1)
        self.history_encoder = HistoryPatchEncoder(
            input_dim=self.main_input_dim,
            patch_len=self.patch_len,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            num_nodes=self.node_size,
            max_patches=max_patches,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            use_temporal_transformer=True,
            dropout=self.dropout,
        )

        self.downsample_encoder = None
        if self.use_downsample_memory:
            self.downsample_encoder = DownsampleMemoryEncoder(
                input_dim=self.main_input_dim,
                d_model=self.d_model,
                num_nodes=self.node_size,
                pool_factor=2,
            )

        self.future_queries = nn.ModuleList([
            FutureQueryEmbedding(future_len=flen, d_model=self.d_model, num_nodes=self.node_size, td_size=self.td_size)
            for flen in self.chain_lengths
        ])

        self.coarse_decoder = CoarseForecastDecoder(
            self.d_model, self.num_heads, self.ffn_dim, dropout=self.dropout
        )
        self.transition_blocks = nn.ModuleList([
            ForecastTransitionBlock(self.d_model, self.num_heads, self.ffn_dim, dropout=self.dropout)
            for _ in range(len(self.chain_lengths) - 1)
        ])
        self.readout = nn.Linear(self.d_model, 1)

        self.spatial_module = None
        if self.use_final_spatial_refine:
            self.spatial_module = ABCDSpatialModule(
                node_size=self.node_size,
                input_len=self.input_len,
                d_spa=self.d_model,
                if_spatial=True,
                spatial_scheme="C",
                adj_mx_path=model_args.get("adj_mx_path"),
                use_gcn=False,
                use_dynamic_spatial=False,
                use_adaptive_adj=True,
                adp_hidden_dim=model_args.get("adp_hidden_dim", 32),
                adp_topk=model_args.get("adp_topk", 20),
                adp_tau=model_args.get("adp_tau", 0.5),
                use_hybrid_graph=False,
                post_spatial_mode=self.post_spatial_mode,
            )

    @staticmethod
    def build_chain_targets(y: torch.Tensor, chain_lengths: List[int]) -> List[torch.Tensor]:
        """Pool future target y [B, F, N, 1] to each chain level."""
        final_len = y.shape[1]
        targets = []
        for clen in chain_lengths:
            if clen == final_len:
                targets.append(y)
            else:
                targets.append(pool_target_to_length(y, clen))
        return targets

    def _encode_history(self, history_data: torch.Tensor) -> torch.Tensor:
        x_main = history_data[..., : self.main_input_dim]
        patch_mem = self.history_encoder(x_main)
        if self.downsample_encoder is not None:
            down_mem = self.downsample_encoder(x_main)
            memory = torch.cat([patch_mem, down_mem], dim=1)
        else:
            memory = patch_mem
        return memory

    def _decode_chain(self, history_data: torch.Tensor, memory: torch.Tensor):
        chain_states = []
        chain_preds = []

        q0 = self.future_queries[0](history_data[..., : self.main_input_dim])
        s0 = self.coarse_decoder(q0, memory)
        chain_states.append(s0)
        chain_preds.append(self.readout(s0))

        prev_state = s0
        for level_idx in range(1, len(self.chain_lengths)):
            prev_len = self.chain_lengths[level_idx - 1]
            next_len = self.chain_lengths[level_idx]
            prev_up = upsample_state(prev_state, next_len)
            q_next = self.future_queries[level_idx](history_data[..., : self.main_input_dim])
            s_next = self.transition_blocks[level_idx - 1](prev_up, q_next, memory)
            chain_states.append(s_next)
            chain_preds.append(self.readout(s_next))
            prev_state = s_next

        return chain_states, chain_preds

    def forward(
        self,
        history_data: torch.Tensor,
        future_data=None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = True,
        return_all: bool = False,
        **kwargs,
    ):
        """
        Args:
            history_data: [B, H, N, C]
            return_all: if True, return dict with pred, chain_preds, chain_states
        """
        memory = self._encode_history(history_data)
        chain_states, chain_preds = self._decode_chain(history_data, memory)

        y_final = chain_preds[-1]
        if self.use_final_spatial_refine and self.spatial_module is not None:
            history_flow = history_data[..., 0]
            y_final = self.spatial_module.refine_prediction(y_final, history_flow)

        if return_all:
            return {
                "pred": y_final,
                "chain_preds": chain_preds,
                "chain_states": chain_states,
            }
        return y_final
