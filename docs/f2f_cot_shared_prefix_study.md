# F2FCoT shared-prefix continuation depth: 3→12 vs 3→6→12

## Decision

After the same executed `Z_3`, taking one extra shared-core call through `Z_6`
does **not** create a stable, observable continuation-depth frontier.  The extra
call is locally progressive as an explicit Forecast-to-Forecast state, but the
induced change in the final `Z_12` is highly redundant and slightly
anti-corrective, in the same qualitative family as Stage III `12→12`
refinement.

No dynamic stop/continue controller was implemented.  The selected fixed
operating point is the prefix-compatible short continuation `3 → 12`.

All methodology and checkpoint decisions used TRAIN/VALID only.  TEST was
opened once after that freeze.

## Starting point

Code: `F2FCoTMultiDepthNet` in `basicts/archs/arch_zoo/F2FCoT_arch/`, still
exactly one `SharedKASAReasoningCore`.  Parameters remain 2,098,006 (97.03% of
the protected 2,162,184-parameter F2FNet).  Zero new forecasting parameters
were added.

Warm start: Stage III
`checkpoints/PEMS04/H12/f2f_cot_depth/formal_v1_seed1/multidepth_best.pt`.

Protected artifacts were not overwritten:

- original F2FNet / canonical ChainForecasting checkpoint
- F2FCoT `extra_best.pt`
- Stage III `multidepth_best.pt`
- `basicts/archs/arch_zoo/F2FCoT_arch/f2f_cot.py`

Stage III never trained `3 → 12`.  Its schedules were `[12]`, `[6,12]`,
`[3,6,12]`, `[3,4,6,12]`, `[3,6,12,12]`, `[2,3,4,6,12]`, `[3,6,12,12,12]`.
A new run was therefore required.  Zero-shot insertion of `3 → 12` into the
Stage III checkpoint was measured first and **not** used as the scientific
verdict.

## Shared-prefix construction

`F2FCoTMultiDepthNet.rollout_shared_prefix_pair` executes the prefix once:

```
state0 = begin_reasoning(X)
prefix = reason_step(state0, 3)          # one shared-core call → Z_3
short  = continue_from(prefix, [12])     # Z_3 → Z_12
long   = continue_from(prefix, [6, 12])  # Z_3 → Z_6 → Z_12
```

`reason_step` is functionally pure, so both continuations read the identical
`ForecastReasoningState` object.  On every VALID/TRAIN/TEST batch:

- `short.forecasts[0] is prefix.latest_forecast`
- `long.forecasts[0] is prefix.latest_forecast`
- prefix memory is the same object at the fork

Object-identity fractions were 1.0 on 107 VALID batches, 319 TRAIN batches,
and 107 TEST batches.  Independent rollouts would recompute `Z_3` separately
and were never used for the comparison.

The extra cost is exactly one additional invocation of the same shared KASA
reasoner (remaining calls: 1 vs 2).

## Training

Protocol, matching the original F2F philosophy:

- PEMS04 in12/out12 split and raw-scale masked MAE
- canonical weights `[0.2, 0.3, 1.0]` on `Z_3`, `Z_6`, `Z_12` of the long path
- 60% of minibatches: canonical `3 → 6 → 12` only
- 40% of minibatches: paired shared-prefix loss, both continuations from the
  same executed `Z_3`, with short-final weight 1.0
- Adam, weight decay `1e-4`, batch 32
- learning rate `5e-5` (a `1.25e-4` probe left containment; it was aborted)
- VALID selection: `0.5 * long MAE + 0.5 * short MAE`, subject to long MAE
  within +0.10 of protected F2FCoT 17.9451
- TEST not loaded during training or selection

Selected epoch 12,
`checkpoints/PEMS04/H12/f2f_cot_shared_prefix/formal_v1_seed1/shared_prefix_best.pt`.

Zero-shot Stage III checkpoint, before this exposure:

| Continuation | Calls | VALID MAE |
|---|---:|---:|
| `3 → 12` | 2 | 17.9709 |
| `3 → 6 → 12` | 3 | 17.9678 |

Zero-shot `3 → 12` already transferred reasonably, but it had never been an
explicit training program.  After paired exposure it became a legitimate,
slightly stronger completion of the same `Z_3`.

## Fixed VALID operating points

Headline MAE is the original batch-averaged raw-scale metric.  Latency is
batch 1 on RTX 4090, 100 repetitions.

| Program | Calls | VALID MAE | Median latency |
|---|---:|---:|---:|
| `3 → 12` | 2 | **17.9325** | 7.523 ms |
| `3 → 6 → 12` | 3 | 17.9571 | 11.302 ms |
| `6 → 12` (reference only) | 2 | 17.9419 | — |

Containment: long VALID 17.9571 is +0.0120 vs protected F2FCoT and +0.0180 vs
original F2FNet 17.9391.  Short VALID 17.9325 is actually 0.0066 **better**
than original F2FNet VALID, with one fewer shared-core call than canonical
`3 → 6 → 12`.  The extra call costs 3.78 ms, matching one core invocation
(~3.76 ms/call).

## Sample-wise crossover (VALID)

Positive gain means the extra `Z_6` call helps the final `Z_12`.  CI is a
paired 2,000-sample bootstrap.

| Quantity | Value |
|---|---|
| Longer helps | 40.81% |
| Mean gain when helpful | 0.0523 |
| Mean harm when harmful | 0.0777 |
| Net average gain | **−0.0247** |
| 95% CI | [−0.0282, −0.0213] |

The extra call is worse on average.  The interval excludes zero.

## Prefix-compatible oracle

Only the two shared-prefix continuations are eligible.  This is not a
multi-program route oracle.

| Split | Best fixed | Best-fixed per-sample MAE | Oracle MAE | Oracle gain |
|---|---|---:|---:|---:|
| VALID | `3 → 12` | 18.0123 | 17.9909 | **0.0213** |
| TRAIN | `3 → 6 → 12` | 16.4667 | 16.4564 | 0.0103 |

0.0213 is the same scale as Stage III's prefix-compatible ceiling (0.0188
among programs sharing initial `Z_3`, 0.0180 for canonical plus `12→12`).
It is not a meaningful continuation-depth frontier.

## TRAIN→VALID stability

| | TRAIN | VALID |
|---|---:|---:|
| Extra call helps | 66.78% | 40.81% |
| Net gain | +0.0315 | −0.0247 |

Help-rate shift 26.0 points.  Gain-distribution KS p-value `2×10^{-183}`.
On TRAIN the extra `Z_6` step looks useful; on VALID it is harmful.  The
benefiting subset is not a stable property of samples.

## What the extra `Z_6` call actually changes

Local reasoning state `Z_3 → Z_6` remains a real F2F step:

| Local `Z_3 → Z_6` (VALID) | Value |
|---|---:|
| Mean abs update | 2.790 |
| Forecast cosine | 0.99974 |
| Correction / residual cosine | **+0.065** |
| Projected MAE gain | **+0.449** |
| Helps | **96.02%** |

The intermediate state is meaningful.  The *final-answer* update caused by
spending that extra call is not:

| Final `Z_12(short)` vs `Z_12(long)` | Extra `Z_6` | Stage III `12→12` |
|---|---:|---:|
| Mean abs update | 0.959 | 0.559 |
| Forecast cosine | 0.99996 | 0.99999 |
| Correction / residual cosine | **−0.110** | −0.107 |
| Net MAE gain | **−0.025** | −0.025 |
| Helps | 40.8% | 34.0% |

The extra call writes a locally better `Z_6`, then the shared reasoner's
`Z_6 → Z_12` lands almost collinear with `Z_3 → Z_12` and slightly away from
the remaining target residual.  It is not a stronger corrective computation
than repeated final-resolution refinement.

## Observability at the `Z_3` decision point

A diagnosis-only logistic probe was trained on TRAIN and evaluated on VALID.
Features are inference-safe and available after the executed `Z_3` only:
history, last delta, `Z_3`, projected `Z_3`, prefix memory, persistence gap,
and prefix KASA branch disagreement (57 dimensions).  Target `Y` is not used.

| Probe (VALID) | Value |
|---|---:|
| ROC AUC | 0.548 |
| Balanced accuracy @ 0.5 | 0.531 |
| Brier | 0.249 |
| Spearman of predicted gain | 0.112 |
| Regression R² | −0.265 |
| Recovered oracle headroom @ 0.5 | **−19.5%** |
| Recovered oracle headroom, TRAIN-tuned | **−110%** |

The probe is slightly worse than always choosing the best fixed continuation.
Generic or `Z_3`-conditioned summaries do not say whether the extra `Z_6`
step will improve the final forecast.

## Controller gate

Predeclared VALID gates were:

- A: oracle gain ≥ 0.03, help fraction in [0.40, 0.85], TRAIN/VALID help
  shift ≤ 0.12, and the extra call less redundant or more corrective than
  `12→12`
- B: AUC ≥ 0.68, recovered headroom ≥ 0.25, selected MAE no worse than
  best-fixed + 0.005

Observed: oracle gain 0.0213, help shift 0.260, AUC 0.548, recovered
headroom negative.  Both gates fail.  A stop/continue head would optimize
split noise on a flat, anti-aligned final update.

## TEST, after the freeze

No controller.  Selected checkpoint and the short continuation as the
operating point were frozen from VALID.

| Program | TEST MAE | TEST RMSE | TEST MAPE |
|---|---:|---:|---:|
| `3 → 12` | **18.0323** | 27.9448 | 0.12348 |
| `3 → 6 → 12` | 18.0984 | 28.1340 | 0.12441 |

Protected original F2FNet TEST MAE is 18.0387; protected F2FCoT `3→6→12` is
18.0406.  Prefix-compatible TEST oracle gain is only 0.0124, and the extra
call helps just 27.0% of TEST samples (net −0.066).

## Answer to the research question

After the same F2F reasoner has produced the same explicit `Z_3`,

- finishing immediately with `Z_3 → Z_12` is the better *fixed* policy;
- spending one additional `Z_6` step does not, on average, improve the
  final forecast;
- sample-wise disagreements exist, but they are TRAIN-overfit, tiny in
  oracle terms, and not predictable from inference-safe `Z_3` context.

The LLM-style question “stop after the current reasoning state, or think
one more intermediate step?” is therefore currently answered by **stop**.
The useful operational result is a 2-call prefix-compatible program
`3 → 12` that preserves original F2F quality.

This does not contradict Stage III's locally progressive `Z_3 → Z_6`
states.  It says that local state improvement is not the same as a
valuable extra call from an already available prefix when the shared
reasoner can already complete `Z_3 → Z_12`.

## Reproducibility

- architecture fork: `basicts/archs/arch_zoo/F2FCoT_arch/f2f_cot_multidepth.py`
- training/evaluation: `scripts/f2f_cot_shared_prefix_study.py`
- launcher: `scripts/run_f2f_cot_shared_prefix_study.sh`
- full JSON: `results/f2f_cot_shared_prefix/formal_v1_seed1/shared_prefix_report.json`
- selected checkpoint: `checkpoints/PEMS04/H12/f2f_cot_shared_prefix/formal_v1_seed1/shared_prefix_best.pt`
