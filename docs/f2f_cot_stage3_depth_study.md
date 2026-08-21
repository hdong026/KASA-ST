# F2FCoT Stage III: reasoning-depth methodology study

## Decision

Multi-depth training is necessary and succeeds at making shallow shared-core
programs useful without destroying the protected `3 -> 6 -> 12` model.  It does
**not**, however, create a meaningful *continuation-depth* frontier beyond the
canonical forecast trace.  Additional resolution-coupled states are locally
progressive, but final same-resolution calls are highly redundant and mildly
anti-corrective.  No dynamic-depth controller was implemented.

This decision was made from TRAIN/VALID only.  TEST was evaluated once after
the epoch-34 model and the no-controller conclusion were fixed.

## Why the previous experiment was insufficient

The previous Stage III comparison contained only `3 -> 6 -> 12` and
`3 -> 4 -> 6 -> 12`.  It therefore confounded:

1. an extra call to the shared reasoner;
2. insertion of temporal resolution 4;
3. a schedule that received less training exposure; and
4. reasoning depth with resolution depth.

Its 0.00875 sample-oracle VALID gain was evidence about those two schedules,
not evidence that recurrent reasoning depth was intrinsically flat.

## Implementation and protected artifacts

`F2FCoTMultiDepthNet` is a new class in
`basicts/archs/arch_zoo/F2FCoT_arch/f2f_cot_multidepth.py`.  It permits
non-decreasing transitions, so `12 -> 12` is an explicit forecast refinement.
Repeated states are retained positionally as `Z_12^(1), Z_12^(2), ...`; they
are not collapsed by a resolution-keyed dictionary.

The model adds **zero parameters**.  The same `SharedKASAReasoningCore`, spatial
refinement, resolution conditioner, and forecast-trace GRU memory are reused on
every call.  Total parameters remain 2,098,006 (97.03% of the protected
2,162,184-parameter F2FNet).  Equal-resolution calls differ only through the
evolved explicit forecast and memory context, not a separate refinement net.

Protected files remained unchanged:

- F2FCoT SHA-256: `b987a03781cd6801d83ae9a1bb95e09bc1c8059e6ca80b4fc1155f3da3ff04b0`
- canonical F2FNet SHA-256: `4dcf9bc01fbe29767ea6ee94898c09bd586f05f5e85e3b6ea8c34fa6a688a5b0`

The new selected checkpoint is
`checkpoints/PEMS04/H12/f2f_cot_depth/formal_v1_seed1/multidepth_best.pt`.

## Schedule family

The study uses one representative at call depths 1--3 and two organization
controls at depths 4 and 5:

| Name | Explicit forecast trace | Purpose |
|---|---|---|
| direct_d1 | `12` | one-call lower bound |
| coarse_d2 | `6 -> 12` | two-call coarse-to-final |
| canonical_d3 | `3 -> 6 -> 12` | protected containment path |
| coupled_d4 | `3 -> 4 -> 6 -> 12` | extra call coupled to a new resolution |
| refine_d4 | `3 -> 6 -> 12 -> 12` | one explicit same-resolution refinement |
| dense_d5 | `2 -> 3 -> 4 -> 6 -> 12` | dense resolution reasoning |
| refine_d5 | `3 -> 6 -> 12 -> 12 -> 12` | two final-resolution refinements |

This is intentionally not an enumeration of every graph path.  It tests call
count and, at matched call counts, whether computation is organized as new
resolutions or repeated target-space refinement.

## Apples-to-apples multi-depth training

The run kept the original PEMS04 in12/out12 data and split (10,181 TRAIN, 3,394
VALID, 3,394 TEST), original target construction, explicit
resolution-matched targets, physical/raw-scale masked MAE, Adam, weight decay
`1e-4`, and the original canonical loss weights `[0.2, 0.3, 1.0]`.

The protected F2FCoT checkpoint warm-started a separate run.  Sixty percent of
TRAIN minibatches used the exact canonical objective; the remaining 40% were
balanced across the other six programs.  For non-canonical routes, total
intermediate supervision mass was fixed at 0.5 and the final state weight at
1.0, preventing longer routes from receiving more objective weight merely for
having more states.  Training used learning rate `1.25e-4` and the same
50-epoch/milestone style as the earlier extra-step curriculum.

Full schedule panels were evaluated every two epochs.  Selection minimized
`0.5 * canonical MAE + 0.5 * mean(alternative MAEs)` subject to canonical MAE
being within 0.10 of protected F2FCoT.  Epoch 34 was selected.  No TEST loader
was used during training or selection.

Zero-exposure results demonstrate why multi-depth training was necessary:
direct/coarse VALID MAE was 26.4067/19.4120 before exposure and
18.0920/17.9669 afterward.  Canonical changed only from 17.9451 to 17.9678.

## Fixed operating points

MAE below is the same batch-averaged raw-scale metric used by the original
runtime.  Latency is batch 1 on RTX 4090, 100 repetitions.

| Program | Calls | VALID MAE | Median latency | TEST MAE |
|---|---:|---:|---:|---:|
| direct_d1 | 1 | 18.0920 | 3.751 ms | 18.0820 |
| coarse_d2 | 2 | **17.9669** | 7.468 ms | **18.0284** |
| canonical_d3 | 3 | 17.9678 | 11.197 ms | 18.0708 |
| coupled_d4 | 4 | 17.9868 | 14.967 ms | 18.1137 |
| refine_d4 | 4 | 17.9925 | 14.944 ms | 18.1222 |
| dense_d5 | 5 | 18.0011 | 18.728 ms | 18.1432 |
| refine_d5 | 5 | 18.0136 | 18.713 ms | 18.1632 |

The selected canonical gap is +0.0227 MAE versus protected F2FCoT and +0.0287
versus original F2FNet VALID, comfortably preserving containment.  The
interesting new frontier is **cheaper reasoning**, not more reasoning: two
calls match three calls while saving 33% of recurrent calls and 3.73 ms.

For representative adjacent depths, average marginal VALID gains are:

- 1 to 2 calls: +0.1251 MAE;
- 2 to 3 calls: -0.0009 MAE (statistically indistinguishable);
- canonical 3 to best 4-call program: -0.0190 MAE;
- best 4 to best 5-call program: -0.0143 MAE.

Training cost is also nearly linear.  Batch-32 forward + raw weighted loss +
backward costs 83.2, 168.1, 253.8, 338.7--340.9, and 423.5--427.2 ms for
1--5 calls, or 83.2--85.4 ms amortized per shared-core call.  Peak allocated
memory rises from 3,026 MiB at one call to 14,469 MiB at five because training
retains each unrolled call's activations.

## Sample-wise crossover

All comparisons use per-sample MAE.  Positive gain means the deeper program is
better.  Confidence intervals are paired 2,000-sample bootstrap intervals.

| Comparison | Deeper helps | Gain when helpful | Harm when harmful | Net gain | 95% CI |
|---|---:|---:|---:|---:|---:|
| direct_d1 -> coarse_d2 | 78.08% | 0.1898 | 0.1045 | +0.1253 | [0.1182, 0.1321] |
| coarse_d2 -> canonical_d3 | 49.47% | 0.0668 | 0.0673 | -0.0010 | [-0.0045, 0.0025] |
| canonical_d3 -> coupled_d4 | 33.29% | 0.0291 | 0.0432 | -0.0191 | [-0.0211, -0.0173] |
| canonical_d3 -> refine_d4 | 33.97% | 0.0357 | 0.0561 | -0.0249 | [-0.0274, -0.0226] |
| canonical_d3 -> dense_d5 | 31.44% | 0.0423 | 0.0682 | -0.0335 | [-0.0365, -0.0306] |
| canonical_d3 -> refine_d5 | 31.91% | 0.0550 | 0.0935 | -0.0461 | [-0.0499, -0.0421] |

The apparent benefiting subsets are not stable across splits.  For example,
coupled_d4 helps 45.85% on TRAIN but 33.29% on VALID; refine_d4 shifts from
43.90% to 33.97%.  Gain distributions differ significantly (KS p-values below
`1e-57`) with 7.4--12.6 percentage-point help-rate shifts for deeper programs.

## Oracle frontier and the important prefix correction

Per-sample MAE and oracle MAE differ slightly from the batch-averaged headline
metric because samples have different valid-value counts.

| Call budget | Best fixed per-sample MAE | Oracle MAE | Oracle gain |
|---:|---:|---:|---:|
| 1 | 18.1732 | 18.1732 | 0.0000 |
| 2 | 18.0480 | 18.0251 | 0.0229 |
| 3 | 18.0480 | 17.9921 | 0.0559 |
| 4 | 18.0480 | 17.9790 | 0.0689 |
| 5 | 18.0480 | 17.9721 | 0.0759 |

The 0.0759 number is *not* pure continuation-depth headroom.  It lets an oracle
switch among programs with different first states (`12`, `6`, `3`, or `2`), so
it includes program/route selection.  The operational CoT questions must use
prefix-compatible programs:

- canonical plus one/two `12 -> 12` continuations: 0.0180 VALID oracle gain;
- all programs sharing initial `Z_3`: 0.0188 VALID oracle gain;
- the corresponding TEST upper bound is only 0.0128.

Thus the stronger full oracle diagnoses some schedule complementarity, but it
does not establish that another call from the current trace has meaningful
value.

## Forecast-trace diagnostics

The canonical resolution chain is locally progressive:

- projected `Z_3 -> Z_6` gains 0.4612 MAE and helps 96.55% of samples;
- projected `Z_6 -> Z_12` gains 0.0882 and helps 79.32%;
- coupled `Z_3 -> Z_4` gains 0.3123 and helps 90.72%;
- dense intermediate steps help 83.06--94.28% locally.

This shows the explicit states are meaningful F2F reasoning states.  It also
explains why local refinement percentages must not be confused with route-level
final superiority: a different earlier state changes all downstream calls.

Same-resolution continuation is qualitatively different:

| Call | Mean absolute update | Forecast cosine | Correction/residual cosine | Local MAE gain | Helps |
|---|---:|---:|---:|---:|---:|
| first `12 -> 12` | 0.5590 | 0.999988 | -0.1066 | -0.0249 | 33.97% |
| second `12 -> 12` | 0.3134 | 0.999995 | -0.1030 | -0.0212 | 30.08% |

The updates shrink rapidly, are nearly collinear with the existing forecast,
and point slightly away from its remaining target residual.  They mostly
duplicate computation rather than add new forecast information.

Context ablations confirm that the trained model causally uses both the latest
explicit forecast and accumulated memory: removing memory raises canonical
VALID MAE by 7.30, removing the latest forecast by 42.20, and removing both by
16.60.  These are deliberately out-of-distribution ablations, so their
magnitude should be read as dependence/sensitivity, not an estimate of the
context components' standalone achievable quality.

## Controller gate

Target-free logistic probes were trained on TRAIN trace summaries and evaluated
on VALID only.  AUC for predicting whether a particular deeper program beats
canonical is just 0.558--0.621; balanced accuracy is 0.532--0.582.  A five-way
next-program diagnostic has 0.239 balanced accuracy and 0.317 top-1 accuracy.

Taken with the 0.0188 prefix-compatible oracle ceiling, unstable crossover
rates, negative average marginal gains, and anti-aligned repeated refinements,
this is not a statistically or operationally meaningful continuation-depth
frontier.  PPO, a stopping head, and a next-resolution controller were therefore
not implemented.  Doing so would optimize selection noise on a flat frontier.

The useful Stage III result is instead a two-call `6 -> 12` operating point that
matches the protected-quality three-call model at lower latency.  A future
dynamic study should first redesign training so successive *prefix-compatible*
states have a positive value-of-computation objective; controller work should
remain gated until that architecture produces substantially more than 0.0188
VALID oracle headroom from an actually available current trace.

## Reproducibility artifacts

- training/evaluation runtime: `scripts/f2f_cot_depth_study.py`
- post-selection diagnostics: `scripts/f2f_cot_depth_diagnostics.py`
- launcher: `scripts/run_f2f_cot_depth_study.sh`
- full report: `results/f2f_cot_depth/formal_v1_seed1/stage3_depth_report.json`
- diagnostic report: `results/f2f_cot_depth/formal_v1_seed1/depth_diagnostics.json`
- per-sample arrays: `results/f2f_cot_depth/formal_v1_seed1/*_arrays.npz`
- selected checkpoint: `checkpoints/PEMS04/H12/f2f_cot_depth/formal_v1_seed1/multidepth_best.pt`

