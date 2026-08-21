# F2FCoT methodology iteration

Status: fixed-depth forecasting validated; dynamic routing intentionally not implemented.

## Scientific question and answer

Can one sufficiently expressive F2F-style forecasting unit be called repeatedly on an
evolving explicit forecast context without losing the original F2FNet's fixed
`3 -> 6 -> 12` quality?

**Yes on the seed-1 PEMS04 containment experiment.** The scratch-trained fixed model
reached VALID MAE 18.0054 versus 17.9391 for the protected canonical checkpoint
(+0.0663, within the predeclared +0.10 band). After a VALID-only two-trajectory
curriculum, the selected checkpoint reached 17.9451 on `3 -> 6 -> 12` and 17.9646
on `3 -> 4 -> 6 -> 12`.

This result does not establish that more steps are always better. The four-step route
is slightly worse on mean MAE, but the additional calls behave as stable forecast
refinement and retain essentially all canonical forecasting quality.

## Exact original unit reused

The source unit is the original
`ChainForecasting_arch.kasa_temporal_step.KASATemporalStep`. Its successful temporal
mechanism is:

1. `PatchEncoder`: KASA patch embedding, spatial/data MLPs, temporal/time-of-day/
   day-of-week representation, and horizon projection;
2. `DownsampEncoder`: the complementary stride-offset temporal representation with
   the same KASA MLP structure;
3. `Conv2d(input_len, output_len, 1)`: linear history residual;
4. additive branch fusion;
5. the canonical `ABCDSpatialModule` adaptive spatial forecast refinement.

`SharedKASAReasoningCore` retains this structure. It is widened from 32 to 64 data/
spatial channels where appropriate and deepened from two to four internal MLP blocks.
It always produces a 12-slot forecast canvas, which is average-pooled to the requested
explicit resolution. The canvas is discarded; only the emitted `Z_r` enters the next
reasoning context.

This avoids V2's replacement of the KASA horizon projection with a small QKV/query
decoder.

## What is shared

Exactly one instance of each large forecasting component exists:

- one KASA patch encoder;
- one KASA downsample encoder;
- one linear history residual;
- one branch-fusion modulation;
- one ABCD adaptive spatial module;
- one compact forecast-trace memory updater.

The same Python module and parameter IDs execute every transition, including
`START -> 3`, `3 -> 4`, `3 -> 6`, `4 -> 6`, and `6 -> 12`. There is no per-edge
`ModuleDict` and no independent forecasting network per resolution.

## Resolution-specific parts

`ResolutionConditioner` has learned source and destination embedding rows plus one
shared continuous conditioner over

`(r_current/H, r_next/H, delta/H, log((r_next+1)/(r_current+1)))`.

It produces eight broadcast input planes and a bounded modulation of the three KASA
branch scales. Only 384 parameters are resolution-specific embedding rows. All MLPs
that consume those rows are shared.

## Forecast Chain-of-Thought context

Every reasoning state is an actual target-space forecast:

- `Z_3`: `[B,3,N,1]`;
- `Z_4`: `[B,4,N,1]` when requested;
- `Z_6`: `[B,6,N,1]`;
- `Z_12`: `[B,12,N,1]`.

For each new call, the model consumes:

- the original ordered history `X`;
- the latest explicit forecast, linearly aligned to the 12-step input axis;
- a 16-channel recurrent memory updated only from previous explicit `Z_r` values and
  their resolution code;
- current/next-resolution conditioning.

The recurrent memory is not an independently evolved latent forecaster: it can change
only when an explicit forecast is appended to the trace. The returned forecast canvas
is not carried forward.

## Capacity

| Component | Parameters |
|---|---:|
| Original protected F2FNet | 2,162,184 |
| Shared KASA reasoning core | 2,055,748 |
| Forecast-trace memory | 2,500 |
| Shared resolution conditioning | 10,251 |
| Of which resolution-specific embedding rows | 384 |
| Shared spatial refinement | 29,507 |
| **F2FCoT total** | **2,098,006** |

The new total is 97.03% of the original. The containment result therefore cannot be
explained by removing most model capacity. The failed ForecastTrajectory V2 had
1,043,158 parameters (48.25% of original) and ended +0.6121 MAE behind canonical.

## Apples-to-apples containment protocol

Kept identical or directly equivalent to the original F2FNet:

- PEMS04 `data_in12_out12.pkl` and `index_in12_out12.pkl` split;
- history 12, target 12, 307 nodes, features `[0,1,2,3]`, target feature `[0]`;
- the exact average-pooled `Z_3`, `Z_6`, and full `Z_12` targets;
- raw-physical-scale masked MAE after the same scaler;
- weights `[0.2,0.3,1.0]` for fixed containment;
- Adam, weight decay `1e-4`, batch size 32, gradient clipping, 100 epochs;
- the same late milestone family and per-epoch full VALID evaluation;
- final 12-step VALID MAE for checkpoint selection;
- TEST unopened until the methodology checkpoint was fixed.

Recorded recurrent-specific optimization adjustment: learning rate `5e-4` rather
than the original initial `2e-3`. Exact-rate smoke probes caused feedback blow-up
because the same unit immediately consumes its own forecast. Patch/downsample heads
were initialized near zero and the linear residual at last-observation persistence.
No target, loss, split, supervision weight, or selection rule was changed.

The extra-step curriculum starts from the selected containment checkpoint, alternates
the fixed and four-step trajectories by minibatch, uses `[0.15,0.15,0.3,1.0]` on the
four explicit states, and accepts a checkpoint only if fixed-route VALID MAE stays
within canonical +0.10. Its selection score is the mean final VALID MAE of the two
routes.

## Results

### VALID

| Model / trajectory | Calls | MAE | RMSE | MAPE |
|---|---:|---:|---:|---:|
| Protected original F2FNet `3-6-12` | 3 | 17.9391 | 28.4917 | 0.12155 |
| Scratch fixed F2FCoT `3-6-12` | 3 | 18.0054 | 28.4321 | 0.12233 |
| Selected curriculum F2FCoT `3-6-12` | 3 | 17.9451 | 28.3243 | 0.12033 |
| Selected curriculum F2FCoT `3-4-6-12` | 4 | 17.9646 | 28.4021 | 0.12065 |

The curriculum checkpoint was selected at epoch 45 using VALID only.

### One-time TEST after methodology fixation

| Model / trajectory | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Protected original F2FNet `3-6-12` | 18.0387 | 28.1596 | 0.12511 |
| F2FCoT `3-6-12` | 18.0406 | 27.9608 | 0.12368 |
| F2FCoT `3-4-6-12` | 18.0870 | 28.0882 | 0.12436 |

No TEST result influenced architecture, training, or checkpoint selection.

## Do additional calls behave as refinement?

On VALID for `3 -> 4 -> 6 -> 12`, projected full-resolution MAE improves across:

| Transition | Mean gain | Improve fraction | Median gain |
|---|---:|---:|---:|
| `3 -> 4` | +0.3200 | 93.61% | +0.2228 |
| `4 -> 6` | +0.1570 | 88.04% | +0.1028 |
| `6 -> 12` | +0.0648 | 79.32% | +0.0473 |

The old optional bridge graph improved on only 49.1% (`3 -> 4`) and 50.1%
(`4 -> 6`) of VALID samples and had negative mean gains. This is a major improvement
in transition semantics and predictability.

The four-step final forecast differs from the three-step forecast by 0.4394 physical
units on average. It is better for 31.47% of VALID samples, and the two-route
sample-wise oracle has 0.00875 MAE headroom over the better fixed route under the
per-sample metric. On TEST the extra route is better for 16.71% and oracle headroom is
0.00417. Thus an additional call is meaningful but its adaptive payoff is small and
does not yet justify RL.

## Cost

RTX 4090, batch 1, synchronized CUDA, selected checkpoint:

- each inference reasoning call: about 3.79--3.81 ms median;
- `3 -> 6 -> 12`: 11.46 ms median total;
- `3 -> 4 -> 6 -> 12`: 15.30 ms median total.

Batch-32 training profiling (forward + raw weighted loss + backward, no optimizer
step):

- three-call route: 252.88 ms mean, 84.29 ms amortized per call, 8,781 MiB peak;
- four-call route: 338.68 ms mean, 84.67 ms amortized per call, 11,639 MiB peak.

Cost is therefore approximately linear in reasoning depth, as intended.

## Dynamic reasoning-depth decision

Do not add PPO/RL yet. The forecasting gate passed, but the two-route adaptive
headroom is only 0.00875 VALID MAE and shrinks on TEST.

The next justified experiment is a small supervised, inference-observable next-step
head attached to the current shared reasoning state after `Z_3`. It should predict the
signed benefit of inserting `Z_4` using `X`, `Z_3`, forecast-trace memory summaries,
resolution code, and remaining measured milliseconds. It must be trained on TRAIN,
selected on VALID, and compared against always-skip/always-refine fixed policies.
Only if that head recovers material VALID oracle headroom should online optimal
stopping or policy optimization be considered.

## Artifacts

- Model: `basicts/archs/arch_zoo/F2FCoT_arch/f2f_cot.py`
- Training/evaluation: `scripts/f2f_cot_runtime.py`
- Structural tests: `scripts/test_f2f_cot.py`
- Post-fixed TEST/cost audit: `scripts/audit_f2f_cot_fixed_methodology.py`
- Selected checkpoint:
  `checkpoints/PEMS04/H12/f2f_cot/formal_v1_seed1/extra_best.pt`
- Main report: `results/f2f_cot/formal_v1_seed1/final_report.json`
- Post-fixed audit: `results/f2f_cot/formal_v1_seed1/postfixed_audit.json`
