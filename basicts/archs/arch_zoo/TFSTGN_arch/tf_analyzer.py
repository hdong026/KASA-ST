"""Module 1: TF-Analyzer — STFT-based time-frequency decomposition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TFAnalyzer(nn.Module):
    """Extract band-wise amplitude spectrograms along the temporal axis."""

    def __init__(
        self,
        input_len: int,
        target_dim: int = 1,
        n_fft: int = 8,
        hop_length: int = 2,
        win_length: int = 8,
        n_bands: int = 4,
        band_mode: str = "soft",
        window_type: str = "hann",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.input_len = input_len
        self.target_dim = target_dim
        self.n_fft = min(n_fft, input_len)
        self.hop_length = hop_length
        self.win_length = min(win_length, self.n_fft)
        self.n_bands = n_bands
        self.band_mode = band_mode
        self.eps = eps
        self.f_total = self.n_fft // 2 + 1

        if window_type == "hamming":
            window = torch.hamming_window(self.win_length)
        else:
            window = torch.hann_window(self.win_length)
        self.register_buffer("window", window, persistent=False)

        if band_mode == "soft":
            self.band_weights = nn.Parameter(torch.randn(n_bands, self.f_total))
        else:
            self.band_weights = None

    def _window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.window.device != device or self.window.dtype != dtype:
            if self.window.shape[0] == self.win_length:
                return self.window.to(device=device, dtype=dtype)
        return self.window

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, N, C]
        Returns:
            band_amp: [B, N, C_tf, K, T_prime]
            amplitude: [B, N, C_tf, F, T_prime]
            phase: [B, N, C_tf, F, T_prime] or None
        """
        x_tf = x[..., : self.target_dim]  # [B, T, N, C_tf]
        b, t, n, c_tf = x_tf.shape
        x_flat = x_tf.permute(0, 2, 3, 1).reshape(b * n * c_tf, t)  # [B*N*C_tf, T]

        stft = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window(x_flat.device, x_flat.dtype),
            center=True,
            return_complex=True,
        )
        t_prime = stft.shape[-1]
        amplitude = stft.abs().view(b, n, c_tf, self.f_total, t_prime)  # [B,N,C_tf,F,T']
        phase = stft.angle().view(b, n, c_tf, self.f_total, t_prime)

        if self.band_mode == "soft" and self.band_weights is not None:
            band_w = F.softmax(self.band_weights, dim=-1)  # [K, F]
            band_amp = torch.einsum("bncft,kf->bnckt", amplitude, band_w)
        else:
            edges = torch.linspace(0, self.f_total, self.n_bands + 1, device=x.device).long()
            bands = []
            for k in range(self.n_bands):
                seg = amplitude[:, :, :, edges[k] : edges[k + 1], :]
                bands.append(seg.mean(dim=3))
            band_amp = torch.stack(bands, dim=3)  # [B,N,C_tf,K,T']

        return band_amp, amplitude, phase
