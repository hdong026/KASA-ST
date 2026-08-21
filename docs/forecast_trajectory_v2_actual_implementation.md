# ForecastTrajectory V2 — actual implementation

V1 (`ForecastTrajectory_arch`) is scientifically invalid and is preserved as failure evidence.
This namespace is new: `basicts/archs/arch_zoo/ForecastTrajectoryV2_arch/`.
Master runner: `scripts/run_forecast_trajectory_v2.sh`.

## Formulation

```
(H_next, Z_next) = F_theta(H_X, H_prev, Z_prev, state_prev, state_next)
```

- `H_X` is encoded once per rollout (hist KASA patch+downsample trunks).
- `H_s` latent tokens `[B,s,N,D]` and `Z_s` explicit forecast `[B,s,N,Cy]` are both propagated.
- Legal graph: START=0, states `[2,3,4,6,12]`, all increasing edges kept (including 2→3, 3→4, 4→6).
- No eta. Latency is real CUDA-event time. `T(tau)=T_history + sum edge T + (#decisions)*T_policy`. Hard B uses `total_ms`.

## Why V1 failed (not resumed)

- Tiny QKV `StateConditionedTemporalStep` replaced KASA TemporalStep (~151k vs ~2.16M).
- Forced `Z_next = Interpolate(Z_prev)+residual` as the identity path.
- Only `Z` was recurrent memory.
- Concatenated-token loss overweight long trajectories.
- Cache reran every terminal trajectory from START.
- Policy `probs * obj * feas` without softmax renormalization; `del pooled` dead node pooling.
- Nested lambda×B×sample×traj×step validation (~1.8h/epoch).

## V2 architecture

- Shared KASA trunks: `KASATokenPatchEncoder` / `KASATokenDownsampEncoder` subclass original Patch/Downsamp encoders; `forward_tokens` skips horizon `projection1`.
- `KASAStateTransitionCell`: destination Fourier queries, cross-attention to `H_X`, `H_prev`, `Z_prev`; dest MLP is original `MultiLayerPerceptron`; optional interpolation is a **gated auxiliary**, gate bias +2 so learned path dominates (3→4 is not interpolate-identity).
- One `StateHyperNetwork` (src/dst embeddings + continuous s/H, delta/H, log-ratio) → FiLM + gate. No `ModuleDict` of full transition nets.
- Spatial: original `ABCDSpatialModule` adaptive-only.
- Canonical warm-start from `checkpoints/ChainForecasting_100/.../ChainForecasting_best_val_MAE.pt` with COPIED / PARTIALLY_COPIED / NEW_INIT prints.

## Training

1. Phase 1: only `[3,6,12]` until `V2 VALID MAE <= canonical F2F VALID MAE + 0.10`, else `BACKBONE_CONTAINMENT_FAIL` and stop (no policy).
2. Phase 2: every batch `0.5 L_canonical + 0.5 L_sampled` with per-trajectory token MAE. Exposure-balanced trajectory B. Checkpoint rejected if canonical MAE exceeds +0.10.
3. Prefix-DAG cache computes each unique prefix once (`n_transitions = #unique non-start prefixes`).
4. Policy: mask infeasible **logits before softmax**; exact prefix-DAG DP; `sum p = 1`; `regret >= -1e-6`. Vectorized validation panel (3 λ × 4 B).

## Commands

```bash
bash scripts/run_forecast_trajectory_v2.sh --acceptance-1epoch --gpu 0 --seed 1
bash scripts/run_forecast_trajectory_v2.sh --full --gpu 0 --seed 1
```

Acceptance writes `results/forecast_trajectory_v2_acceptance.json`.
Full uses `results/forecast_trajectory_v2_run/formal_seed1/` and never reuses V1 markers/caches.
