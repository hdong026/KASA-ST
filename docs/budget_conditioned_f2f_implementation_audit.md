# Budget-Conditioned Adaptive F2F — Implementation Audit

## Confirmed facts

1. Formal variant `chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss`
   uses `ChainForecasting` with H=12 `chain_lengths=[3,6,12]`.
2. Each stage is a `KASATemporalStep(output_len=k)` with patch / downsample / linear residual.
3. `ForecastStateAdapter(mode=condition_only)` only rewrites forwarded condition when
   `previous_state is not None` and the stage is not final (formal code: `step_idx==1`).
4. Progressive spatial keeps node count `N` fixed; only `topk` / `alpha` / hidden dims change.
5. Runner uses `token_normalized` → `forecast_state_token_mae` over all stage tokens.
6. One-shot model uses `PackedResolutionForecastExecutor` (tiny MHA/MLP), **not**
   `KASATemporalStep`; history is mean-pooled; ~0.02M-scale params vs multi-stage KASA.
7. One-shot `dual_update` is not wired in `_one_shot_resolution_token_loss`.

## Current one-shot problems (not reused as backbone)

- Forecasting backbone replaced by packed direct head.
- Cross-attn / mean-over-time collapses temporal structure.
- Planner credit assignment weak; dual unused in runner.
- Spatial clustering / variable slots change semantics vs fixed-N F2F.

## Must reuse as-is

- `KASATemporalStep`, `interpolate_forecast`
- `ForecastStateAdapter` (condition_only)
- `ABCDSpatialModule` progressive configs
- `ChainForecasting.pool_target`
- `forecast_state_token_mae(..., rescale_pair=...)` for both `token_normalized`
  and `baseline_compatible` (raw physical scale + null mask after inverse transform)
- Formal VARIANT entry untouched

## This implementation’s fix

- New variant wraps a full-route `ChainForecasting` supernet (`[H/4,H/2,H]`).
- Executes a **selected subset route** by indexing the same stage modules by resolution.
- Planner only chooses among `{[H],[H/2,H],[H/4,H],[H/4,H/2,H]}`.
- No node pooling / clustering.
- Scheme A for missing condition: first stage of any route uses `prev_forecast=None`
  (native KASA path); later stages use interpolate + optional condition_only adapter
  whenever `previous_state is not None` and stage is not final (matches formal [3,6,12]).
- Forecasting losses are always computed in the **runner** with `self._rescale_pair`;
  model-side `loss_terms` are diagnostics only.

## Choice: Scheme A

Zero/absent condition uses the existing unconditioned KASA encoders; conditioned stages
use the existing cond encoders. No dynamic Parameter rebuild; closest to formal F2F.

## Module / parameter correspondence (forced [3,6,12])

| Component | Formal F2FNet | New forced-full-route |
|-----------|---------------|------------------------|
| Temporal stages | `temporal_steps` ×3 (KASA) | same modules via shared `ChainForecasting` supernet |
| Progressive spatial | stage idx 0/1/2 | same modules indexed by resolution tier |
| Condition adapter | condition_only at middle stage | condition_only on non-final stages with previous |
| Loss | `forecast_state_token_mae` + runner `rescale_pair` | same helper via `_token_mae_for_resolutions` |
| Extra params | — | lightweight `BudgetRoutePlanner` (~few K) |
| Node set N | fixed | fixed (no clustering) |

Expected param scale: multi-million (KASA multi-stage), **not** ~0.02M packed executor.

## Loss audit caveat

Do not treat “same helper name” without `rescale_pair` as runner equivalence.
Normalized-scale comparisons are false positives; raw-scale scalar, gradient, and
null-mask probes are required.
