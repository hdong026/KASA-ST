# Plan B — Budgeted Bellman Forecast Refinement

**Method name:** Budgeted Bellman Forecast Refinement  
**Status after this document:** Smoke pipeline `BELLMAN_SMOKE_PASS`  
**Not:** GRPO / PPO / policy gradient / actor-critic / entropy routing / classification routing

## Scientific form

Finite-horizon **Budgeted MDP** on the F2F refinement DAG, trained by **cross-fitted full-information backward Q regression**.

### Stages (horizon H divisible by 4)

- `q = H/4`, `m = H/2`, `f = H`
- Terminal routes: `[f]`, `[m,f]`, `[q,f]`, `[q,m,f]`
- H=12 → `[12]`, `[6,12]`, `[3,12]`, `[3,6,12]`
- H=24/48 analogous

### Additive stage costs (derived, not invented)

From whole-route normalized static costs:

- `c_f = C[f]`
- `c_m = C[m,f] - C[f]`
- `c_q = C[q,f] - C[f]`

Verified: `c_q + c_m + c_f == C[q,m,f]` (tol 1e-6), all positive.

H=12 values:

| stage | cost |
|-------|------|
| `c_f` | 0.5405405405405405 |
| `c_m` | 0.29729729729729737 |
| `c_q` | 0.16216216216216228 |

### Budget

`B(η) = C_min + η (C_max - C_min)`  
Action feasible iff `c(a) + min_finish(successor) ≤ B + ε` (recursive, not immediate-cost-only).

### Returns (primary Bellman objective)

Cost is a **hard constraint**. Terminal quality = MAE.

Baseline-centered returns (preserve argmin L ↔ argmax g):

- `g_D = 0`
- `g_M = L_D - L_M`
- `g_Q = L_D - L_Q`
- `g_F = L_D - L_F`

**Not** used: V2 reward `-10 q - cost`, δ-tolerance, cost penalty in return.

### Exact targets

**Q1** (after paying `c_q`):

- `target_Q1_f = g_Q`
- `target_Q1_m = g_F`

**Q0** (per unique nontrivial budget regime):

- `target_Q0_f = 0`
- `target_Q0_m = g_M` if `[m,f]` feasible
- `target_Q0_q = g_Q` if only jump child feasible
- `target_Q0_q = max(g_Q, g_F)` if both children feasible

Targets come from OOF counterfactual returns — **not** from bootstrapped / learned Q1.

### Networks

- **Q0:** history encoder on raw `X` + normalized budget + feasible-mask embedding → 3 scalars
- **Q1:** independent history encoder + `Z_q` encoder + remaining budget → 2 scalars
- No softmax / entropy / route quotas
- Observation ≈ mathematical state `(X[, Z_q], remaining budget)`; embedding is not claimed to be the Markov state

### OOF state discipline

- Labels / `Z_q` from **same fold teacher**
- Raw history `X` is teacher-independent (loaded live)
- **Do not** use stable supernet `H_shared` as OOF Q-state

### Training phases

1. Phase I: train Q1 (Huber / SmoothL1), select by VALID child-route regret  
2. Phase II: freeze Q1, train Q0 with exact targets, select by VALID sequential strict regret  
3. Phase III (optional): joint consistency with `L_Bellman`; keep only if VALID improves ≥ 1e-4

Global return scale: one TRAIN-OOF robust scale (`IQR/1.349`, MAD fallback) on `g_M,g_Q,g_F`.

### Inference

Greedy masked argmax on Q0 → if `q`, execute quarter **once**, then greedy Q1 child, resume suffix. Prefix/resume max abs diff must be `< 1e-6`.

## Files (new only; V1/V2 untouched)

| Path | Role |
|------|------|
| `basicts/.../budgeted_bellman_refinement.py` | MDP, costs, targets, regimes |
| `basicts/.../bellman_refinement_dataset.py` | OOF cache / dataset |
| `basicts/.../bellman_refinement_qnet.py` | Q0/Q1 nets |
| `scripts/train_bellman_refinement.py` | Phase train |
| `scripts/eval_bellman_refinement.py` | Sequential eval + BasicTS metrics |
| `scripts/audit_bellman_refinement.py` | Cost/dataset/frontier audits |
| `scripts/run_bellman_smoke_pipeline.py` | Full smoke S0–S10 |
| `scripts/run_plan_b_bellman_smoke.sh` | Smoke entry (300s timeout) |
| `scripts/run_plan_b_bellman.sh` | Formal runner (`--confirm-full-run`) |

## Smoke

```bash
bash scripts/run_plan_b_bellman_smoke.sh --gpu 0
```

## Formal (user-launched; not auto-run in coding turn)

```bash
cd /home/dhz/KASA-ST
bash scripts/run_plan_b_bellman.sh --confirm-full-run --gpu 0 --seed 1
```
