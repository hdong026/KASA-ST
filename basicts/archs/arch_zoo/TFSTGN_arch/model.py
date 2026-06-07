"""TF-STGN: Time-Frequency Guided Spatio-Temporal Gating Network (decoupled)."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicts.archs.arch_zoo.TFSTGN_arch.fs_extractor import FrequencySpatialExtractor
from basicts.archs.arch_zoo.TFSTGN_arch.horizon_decoder import HorizonDecoder
from basicts.archs.arch_zoo.TFSTGN_arch.st_spectral_gate import STSpectralGate
from basicts.archs.arch_zoo.TFSTGN_arch.temporal_backbone import build_temporal_backbone
from basicts.archs.arch_zoo.TFSTGN_arch.temporal_trunk import KASATemporalTrunk
from basicts.archs.arch_zoo.TFSTGN_arch.tf_analyzer import TFAnalyzer


class TFSTGN(nn.Module):
    """
    Decoupled TF-STGN:
      - KASA-style temporal trunk (patch + downsample + flow residual + ToD/DoW)
      - TF-guided spatial refinement branch (TF-Analyzer + FS-Extractor + ST-Gate)
    """

    def __init__(
        self,
        num_nodes: int = None,
        input_len: int = 12,
        output_len: int = 12,
        input_dim: int = 4,
        target_dim: int = 1,
        hidden_dim: int = 64,
        embed_dim: int = 16,
        n_bands: int = 4,
        n_fft: int = 8,
        hop_length: int = 2,
        win_length: int = 8,
        band_mode: str = "soft",
        topk: int = 20,
        attn_temperature: float = 1.0,
        temporal_backbone: str = "bigru",
        temporal_layers: int = 1,
        dropout: float = 0.1,
        use_film: bool = True,
        use_band_specific_proj: bool = False,
        use_spectral_gate: bool = True,
        use_temporal_gate: bool = True,
        use_spatial_gate: bool = True,
        use_tf_spatial: bool = True,
        spatial_alpha_init: float = -2.0,
        gate_bias: float = -2.0,
        static_hybrid_alpha: float = 0.2,
        adj_mx_path: str = None,
        patch_len: int = 3,
        stride: int = 4,
        td_size: int = 288,
        dw_size: int = 7,
        d_td: int = 32,
        d_dw: int = 32,
        d_d: int = 32,
        d_spa: int = 32,
        if_time_in_day: bool = True,
        if_day_in_week: bool = True,
        if_spatial: bool = False,
        num_layer: int = 1,
        use_prior_residual: bool = True,
        prior_mapper_type: str = "linear",
        prediction_head: str = "shared",
        decoder_mode: str = "shared",
        **kwargs,
    ):
        super().__init__()
        if num_nodes is None:
            num_nodes = kwargs.get("node_size")
        if num_nodes is None:
            raise ValueError("TFSTGN requires num_nodes or node_size")
        if "output_dim" in kwargs and target_dim == 1:
            target_dim = kwargs["output_dim"]

        if adj_mx_path is None:
            dataset = kwargs.get("dataset_name", "PEMS04")
            adj_mx_path = os.path.join("datasets", dataset, "adj_mx.pkl")

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.use_tf_spatial = use_tf_spatial
        self.prediction_head = prediction_head

        self.temporal_trunk = KASATemporalTrunk(
            node_size=num_nodes,
            input_len=input_len,
            output_len=output_len,
            input_dim=input_dim,
            patch_len=patch_len,
            stride=stride,
            td_size=td_size,
            dw_size=dw_size,
            d_td=d_td,
            d_dw=d_dw,
            d_d=d_d,
            d_spa=d_spa,
            if_time_in_day=if_time_in_day,
            if_day_in_week=if_day_in_week,
            if_spatial=if_spatial,
            num_layer=num_layer,
            use_prior_residual=use_prior_residual,
            prior_mapper_type=prior_mapper_type,
        )

        self.tf_analyzer = TFAnalyzer(
            input_len=input_len,
            target_dim=target_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_bands=n_bands,
            band_mode=band_mode,
        )
        self.fs_extractor = FrequencySpatialExtractor(
            num_nodes=num_nodes,
            input_dim=target_dim,
            hidden_dim=hidden_dim,
            n_bands=n_bands,
            embed_dim=embed_dim,
            topk=topk,
            attn_temperature=attn_temperature,
            dropout=dropout,
            use_film=use_film,
            use_band_specific_proj=use_band_specific_proj,
            adj_mx_path=adj_mx_path,
            static_hybrid_alpha=static_hybrid_alpha,
        )
        self.spectral_gate = STSpectralGate(
            use_spectral_gate=use_spectral_gate,
            use_temporal_gate=use_temporal_gate,
            use_spatial_gate=use_spatial_gate,
            gate_bias=gate_bias,
        )
        self.spatial_temporal = build_temporal_backbone(
            temporal_backbone,
            hidden_dim,
            temporal_layers,
            dropout,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_len, num_nodes, target_dim)
            band_amp, _, _ = self.tf_analyzer(dummy)
            t_prime = band_amp.shape[-1]

        dec_mode = decoder_mode if prediction_head in ("horizon", "shared") else "shared"
        self.horizon_decoder = HorizonDecoder(
            hidden_dim=hidden_dim,
            output_len=output_len,
            target_dim=target_dim,
            t_prime=t_prime,
            dropout=dropout,
            mode=dec_mode,
        )
        self.spatial_alpha = nn.Parameter(torch.tensor(float(spatial_alpha_init)))

        # Legacy fallback head (ablation / backward compat)
        self.legacy_pred_head = nn.Linear(hidden_dim, output_len * target_dim)

    def _align_frames(self, x: torch.Tensor, t_prime: int) -> torch.Tensor:
        b, t, n, c = x.shape
        x_perm = x.permute(0, 2, 3, 1).reshape(b * n * c, 1, t)
        x_align = F.interpolate(x_perm, size=t_prime, mode="linear", align_corners=False)
        return x_align.reshape(b, n, c, t_prime).permute(0, 3, 1, 2)

    def _spatial_branch(self, history_data: torch.Tensor):
        x_flow = history_data[..., : self.target_dim]
        band_amp, amplitude, phase = self.tf_analyzer(x_flow)
        t_prime = band_amp.shape[-1]

        x_align = self._align_frames(x_flow, t_prime)
        z = self.fs_extractor(x_align, band_amp)
        z = self.spectral_gate(z, band_amp)
        h = self.spatial_temporal(z)

        if self.prediction_head == "last":
            h_pool = h[:, -1]
            b, n, d = h_pool.shape
            delta = self.legacy_pred_head(h_pool)
            delta = delta.reshape(b, n, self.output_len, self.target_dim).permute(0, 2, 1, 3)
        elif self.prediction_head == "mean":
            h_pool = h.mean(dim=1)
            b, n, d = h_pool.shape
            delta = self.legacy_pred_head(h_pool)
            delta = delta.reshape(b, n, self.output_len, self.target_dim).permute(0, 2, 1, 3)
        else:
            delta = self.horizon_decoder(h)

        return delta, band_amp, amplitude, phase

    def forward(
        self,
        history_data: torch.Tensor,
        future_data=None,
        batch_seen=None,
        epoch=None,
        train=None,
        **kwargs,
    ):
        y_temporal = self.temporal_trunk(history_data)

        if not self.use_tf_spatial:
            if kwargs.get("return_intermediates", False):
                return {
                    "prediction": y_temporal,
                    "band_amp": None,
                    "amplitude": None,
                    "phase": None,
                    "temporal_gate": None,
                    "spatial_gate": None,
                    "band_alpha": None,
                    "y_temporal": y_temporal,
                    "delta_spatial": None,
                }
            return y_temporal

        delta_spatial, band_amp, amplitude, phase = self._spatial_branch(history_data)
        alpha = torch.sigmoid(self.spatial_alpha)
        pred = y_temporal + alpha * delta_spatial

        if kwargs.get("return_intermediates", False):
            return {
                "prediction": pred,
                "band_amp": band_amp,
                "amplitude": amplitude,
                "phase": phase,
                "temporal_gate": self.spectral_gate.latest_temporal_gate,
                "spatial_gate": self.spectral_gate.latest_spatial_gate,
                "band_alpha": self.fs_extractor.latest_band_alpha,
                "y_temporal": y_temporal,
                "delta_spatial": delta_spatial,
                "spatial_alpha": alpha.detach(),
            }
        return pred
