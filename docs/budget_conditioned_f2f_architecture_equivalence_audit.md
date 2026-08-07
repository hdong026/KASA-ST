# Forced-route architecture + raw-scale loss equivalence audit

## Verdict (synthetic weight-copy)

Under matched kwargs, after copying `ChainForecasting` weights into
`BudgetConditionedAdaptiveF2FNet.backbone`:

**forced `[3,6,12]` forward tensors match the original full chain**
(historically `max_abs_diff=0` after weight copy).

Therefore the KASA / progressive-spatial / condition_only adapter **path is
reused**. Performance gaps on PEMS04 are **not** explained by “forgot KASA”.

## Forced `[12]` is NOT original single-stage

| | `ChainForecasting(chain_lengths=[12])` | `forced_route=[12]` |
|--|--|--|
| temporal modules | 1 | uses supernet index 2 of 3 |
| progressive spatial | `_fit_stage_list([0.25,0.5,1.0],1)=[0.25]` LIGHT | index 2 → ratio 1.0 FULL |
| intermediate supervision | none | none (but unused stage 3/6 modules exist) |
| first-stage condition | `prev_forecast=None` → base encoders | same (no zero condition) |

Do **not** claim forced `[12]` equals a dedicated `ChainForecasting([12])`.

## Loss equivalence (corrected)

### Forward architecture equivalence
forced `[3,6,12]` after weight copy: tensor alignment holds.

### Old loss-audit false positive
Earlier audits compared:

- `forecast_state_token_mae(... without rescale_pair)`
- `baseline_compatible_token_mae(... without rescale_pair)`

Both sides computed **normalized-scale** MAE, so `abs_diff=0` did **not** prove
equivalence to the original runner (which always passes `rescale_pair`).

### Fixed standard (raw physical scale)
Audits must pass all of:

1. scalar loss with synthetic / runner `rescale_pair` (`abs_diff < 1e-7`)
2. per-stage prediction gradients under the same rescale (`max_abs_diff < 1e-7`)
3. null-mask semantics: normalized target `0` is **valid** after inverse transform;
   only physical `null_val` (e.g. raw `0`) is masked. Canonical probe expects loss `40.0`.

Runner unification: `_token_mae_loss` and `_baseline_compatible_loss` both call
`_token_mae_for_resolutions(..., rescale_pair=self._rescale_pair)`.

## Forced mode priority

- Runner: sandwich disabled when `forced_route` set or `route_selection_mode=forced`.
- Model: `sandwich_routes` + forced → `RuntimeError`.
- First-batch log: `executed_routes / chain_resolutions / actual_stage_count`.

## Config / checkpoint isolation

`experiment_tag` now includes `phase`, `loss_mode`, and a short SHA1 of the full
`run_signature` (including `code_version` = git commit + source fingerprint), so
normalized-loss runs and raw-scale runs do not share `.../forced_3-6-12/seed1`.
