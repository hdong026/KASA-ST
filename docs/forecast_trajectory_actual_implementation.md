# ForecastTrajectory — actual implementation

Working name: **ForecastTrajectory**

Formulation: **Universal State-Conditioned Forecast Transition + Online Trajectory Policy**

This method is **not** Plan A, Plan B-v1, Plan B-v2, Bellman Q0/Q1, eta routing, or the old 4-route budget system. Those modules are untouched.

## What was built

New package:

- `basicts/archs/arch_zoo/ForecastTrajectory_arch/`

New runners:

- `scripts/run_forecast_trajectory_learning.sh` — **the only user-facing command**
- `scripts/run_forecast_trajectory_pipeline.py` — phase orchestrator
- `scripts/forecast_trajectory_runtime.py` — shared train/cache/policy/eval
- `scripts/train_forecast_trajectory_transition.py`
- `scripts/build_forecast_trajectory_cache.py`
- `scripts/train_forecast_trajectory_policy.py`
- `scripts/eval_forecast_trajectory.py`

## Architecture

There is **one** shared `UniversalTransitionCore`. There are **not** separate F2/F3/F4/F6/F12 modules and **not** per-edge modules.

```
history = E_θ(X)          # executed once per sample
Z_next  = F_θ(history, Z_prev, s_prev, s_next)
```

Resolution states for H=12:

```
START = 0
STATES = [2, 3, 4, 6, 12]
```

Legal edges: strictly increasing, 15 directed edges. Terminal trajectories: all subsets of `{2,3,4,6}` followed by `12` → **16** paths, enumerated from the graph (not hard-coded route IDs).

`3→12`, `4→12`, `2→6`, and `0→12` are ordinary shared-parameter edges.

### Transition

- Shared KASA-style history encoder (patch + downsample) **without** horizon-specific `Conv2d(in=L, out=s)` heads.
- Destination queries `u_j = (j+0.5)/s_next` with Fourier features.
- Previous forecast aligned with the existing F2F linear interpolation helper.
- Residual: `Z_next = Z_bar + Δ`, with `Z_bar = 0` at START (plus a learned START token on queries).
- FiLM from continuous resolution features `(s_prev/H, s_next/H, Δs/H, log((s_next+1)/(s_prev+1)))`.
- One shared adaptive spatial residual module.

Gradients flow through the full sampled trajectory (no detach).

Loss: **token-normalized MAE** over every visited state of the sampled trajectories. No `.2/.3/1.0` stage weights. No latency term in `L_transition`.

Sampling: 2 trajectories / batch — A alternates `[12]` and `[2,3,4,6,12]`; B is edge-balanced over all 15 edges.

Checkpoint: mean final MAE on the fixed panel `{[12],[2,12],[3,12],[6,12],[2,3,4,6,12]}`. Auto-extend 100→+50…≤250 if still improving.

### Latency

Real CUDA-event profiling (history encoder + every edge + policy step). Median is the training-time lookup. **No eta. No 0.54/0.70/0.84/1.0 proxy costs.**

`C_ms(τ) = C_history + Σ edge + C_policy × #decisions`  
`C_norm(τ) = C_ms(τ) / C_ms(dense=[2,3,4,6,12])`

### Policy

Lightweight `π_φ(s_next | h_X, Z_s, s, λ, remaining budget)` using the **same** pooled history encoder output. Illegal and hard-budget-infeasible destinations are masked. Hard remaining budget uses DP `min_finish_cost(s)` on the latency graph.

`λ` is a continuous tradeoff weight, **not eta**. 25% of policy samples use `λ=0`; the rest are drawn from `[0, λ_max]` with `λ_max` derived from **TRAIN** oracle scale.

Because there are only 16 terminal trajectories, the objective is the **exact** expected cost

```
Σ_τ  p_φ(τ) [ L(τ) + λ C_norm(τ) ]
```

No REINFORCE / PPO / GRPO / actor-critic / DQN. Path probabilities must sum to 1 (±1e-6).

Transition is frozen while the policy is trained. No joint fine-tune in this implementation. No crossfit teachers.

## Commands

Acceptance (engineering, not for tuning):

```bash
cd /home/dhz/KASA-ST
bash scripts/run_forecast_trajectory_learning.sh --acceptance-1epoch --gpu 0 --seed 1
```

Full scientific pipeline (resumable; one command):

```bash
cd /home/dhz/KASA-ST
bash scripts/run_forecast_trajectory_learning.sh --full --gpu 0 --seed 1
```

## Outputs

- `results/forecast_trajectory_acceptance_1epoch.json`
- `results/forecast_trajectory_transition_history.json`
- `results/forecast_trajectory_latency_table.json`
- `results/forecast_trajectory_train_cache_manifest.json`
- `results/forecast_trajectory_valid_cache_manifest.json`
- `results/forecast_trajectory_oracle_analysis.json`
- `results/forecast_trajectory_policy_history.json`
- `results/forecast_trajectory_valid_eval.json`
- `results/forecast_trajectory_test_eval.json`
- `results/forecast_trajectory_final_report.json`
