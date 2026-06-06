# KASA v3-freqgate Design

## Motivation

Recent spatio-temporal forecasting work suggests that **excessive or poorly timed spatial operations can harm temporal forecasting**. We do not believe spatial information is inherently harmful. The problem is **unconditional, state-agnostic, and overly early spatial mixing**.

KASA v3-freqgate addresses this by using **time-frequency descriptors from history flow** to decide:

1. **Which graphs** to trust (static / adaptive / dynamic / frequency).
2. **How much spatial signal** to inject into the temporal prediction.

Spatial activation happens **after** the temporal backbone, not as a direct input/output prior residual.

## Why prior residual was removed from the default path

Under PeMS04 6:2:2, direct output-side prior residual (channel 3) did not consistently improve performance. v3-freqgate therefore:

- Uses `FORWARD_FEATURES=[0,1,2,3]` for dataset compatibility.
- Uses only `history_data[..., :3]` for the temporal backbone.
- Uses `history_data[..., 0]` for frequency descriptors.
- Ignores channel 3 unless `keep_output_prior_residual=True` (default **False**).

## Architecture overview

```
flow + ToD + DoW
  → patch encoder + downsample encoder + linear residual
  → y_temporal
  → FrequencyDescriptor(history_flow)
  → build A_static, A_adaptive, A_dynamic, A_freq
  → frequency-conditioned graph fusion → A_hybrid
  → spatial propagation → y_spatial
  → Cross ST Gate (or hybrid_alpha fallback)
  → y_final
```

## Main modules

### 1. FrequencyDescriptor

- Input: `history_flow [B, T, N]`
- Detrend per node: subtract temporal mean before FFT
- rFFT → amplitude → drop DC → split low/mid/high bands
- Normalize band energies per node (relative spectral distribution)
- MLP + LayerNorm → `freq_emb [B, N, freq_dim]`

### 2. Frequency-guided dynamic graph

- `A_freq = softmax(topk(freq_emb @ freq_emb^T / sqrt(d)))`
- Batch-specific, history-only; no future flow or channel 3 prior

### 3. Frequency-conditioned graph fusion

- **Not** a single global 4-way weight (weak v3 design).
- `freq_context = mean(freq_emb, dim=1)` → MLP → `weights [B, 4]`
- `A_hybrid = Σ_k weights_k * A_k`
- Fallback: `use_freq_conditioned_fusion=False` uses learnable global logits (ablation `v3_freq_global`)

### 4. Cross Spatial-Temporal Gate

- Gate input: `[y_temporal, y_spatial, freq_gate]`
- `output = y_temporal + gate_residual_scale * sigmoid(MLP(...)) * y_spatial`
- Pointwise over `(B, H, N)` — not full attention over `N×T`

### 5. Optional spectral decomposition gate

- Low/high split via moving average along horizon `H`
- Separate gates for low and high spatial branches
- Default **off** (`use_spectral_decomp_gate=False`)

## Complexity

| Component | Cost |
|-----------|------|
| FFT | `O(B × T log T × N)` |
| Frequency graph | `O(B × N²)` with top-k sparsification |
| Graph fusion MLP | `O(B × freq_dim)` |
| Cross gate | `O(B × H × N)` pointwise MLP |
| Full `N×T` attention | **Not used** |

## Default config

`examples/KASAST_v3_freqgate/KASAST_v3_freqgate_PEMS04.py`

- `use_frequency_guided_graph=True`
- `use_freq_conditioned_fusion=True`
- `use_cross_st_gate=True`
- `use_spectral_decomp_gate=False`
- Prior switches all **False**

Checkpoint: `checkpoints/KASA_v3_freqgate_PEMS04`

## Ablation variants

| Variant | Fusion | Cross gate | Spectral gate |
|---------|--------|------------|---------------|
| `v2_clean` | v2 hybrid (3 graphs) | scalar α | — |
| `v3_freq_global` | global 4-way | off | off |
| `v3_freq_conditioned` | freq-conditioned | off | off |
| `v3_freq_gate` | freq-conditioned | on | off |
| `v3_spectral_gate` | freq-conditioned | on | on |

Runner: `scripts/run_kasa_v3_freqgate_ablation.py`
