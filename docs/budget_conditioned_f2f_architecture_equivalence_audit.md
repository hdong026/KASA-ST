# Forced-route architecture equivalence audit

## Verdict (synthetic weight-copy)

Under matched kwargs, after copying `ChainForecasting` weights into
`BudgetConditionedAdaptiveF2FNet.backbone`:

**forced `[3,6,12]` forward tensors match the original full chain to `max_abs_diff=0`.**

Therefore the KASA / progressive-spatial / condition_only adapter **path is
reused**. Performance gaps on PEMS04 are **not** explained by “forgot KASA”.

## Forced `[12]` is NOT original single-stage

| | `ChainForecasting(chain_lengths=[12])` | `forced_route=[12]` |
|--|--|--|
| temporal modules | 1 | uses supernet index 2 of 3 |
| progressive spatial | `_fit_stage_list([0.25,0.5,1.0],1)=[0.25]` LIGHT | index 2 → ratio 1.0 FULL |
| intermediate supervision | none | none (but unused stage 3/6 modules exist) |
| first-stage condition | `prev_forecast=None` → base encoders | same (no zero condition) |

## Why PEMS MAE can still be worse

1. **forced `[12]`** trains a different object than formal `[3,6,12]` (missing mid-stage losses; full spatial on a lone stage).
2. Extra **planner** params exist (not used in forced forward) — negligible for eval.
3. Training dynamics / random init of unused stages still affect shared codebooks.
4. Compare apples-to-apples: only **forced `[3,6,12]` + weight-aligned init** is architecture-equivalent; random-init training is not guaranteed to match formal seed-1 numbers.

## Forced mode priority

- Runner: sandwich disabled when `forced_route` set or `route_selection_mode=forced`.
- Model: `sandwich_routes` + forced → `RuntimeError`.
- First-batch log: `executed_routes / chain_resolutions / actual_stage_count`.

## Loss

`baseline_compatible_token_mae` → `forecast_state_token_mae` (abs diff 0 on synthetic).

## Config

PEMS formal vs forced generated configs share d_*, progressive lists, lr, milestones, bs.
Diffs: `MODEL.NAME/ARCH`, `chain_loss_mode` name (`token_normalized` vs `baseline_compatible`), budget-only keys.
