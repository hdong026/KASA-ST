# Plan B-v2 — Actual Implementation

**Method name:** Exact Full-Information Sequential Forecast Refinement Policy Optimization  
**Not:** original GRPO / GSPO / clipped PPO trajectory surrogate.

This document describes what was implemented. Formal training is **not** run by agents; use:

```bash
bash scripts/run_plan_b_v2_full.sh --confirm-full-run --gpu 0
```

---

## Preserved (unchanged scientific behavior)

| Artifact | Status |
|----------|--------|
| `scripts/train_group_relative_refinement_policy.py` | untouched |
| `scripts/run_plan_b_full.sh` | untouched |
| `checkpoints/.../group_relative_policy.pt` | not overwritten |
| Plan A controller / teachers / stable F2F supernet | not retrained |
| Reward `(δ=0.05, λq=10, λc=1)` | identical to V1 |

---

## New modules

| Path | Role |
|------|------|
| `basicts/.../exact_trajectory_policy_objective.py` | Mean-centered advantages, global utility scale, unique nontrivial regimes, exact terminal probs, entropy/KL |
| `basicts/.../group_relative_refinement_policy_v2.py` | Structured state0 / Z_q encoders; zero-init action heads |
| `basicts/.../plan_b_v2_state_cache.py` | Dual-view OOF cache (teacher + stable), strict supernet load |
| `scripts/train_plan_b_v2.py` | Formal trainer (`--confirm-full-run`) |
| `scripts/eval_plan_b_v2.py` | Eval with H/4 once + resume |
| `scripts/run_plan_b_v2_full.sh` | One-click formal runner |
| `scripts/audit_plan_b_v2.py` | Mini diagnostics only |

---

## Mathematical objective

For each sample and each nontrivial feasibility regime \(F\):

\[
A(r) = R(r) - \mathrm{mean}_{r'\in F} R(r'),\quad
A_{\mathrm{scaled}} = A / \mathrm{utility\_scale}
\]

\[
p([H])=\pi_0(D),\ 
p([H/2,H])=\pi_0(H),\ 
p([H/4,H])=\pi_0(Q)\pi_1(J),\ 
p([H/4,H/2,H])=\pi_0(Q)\pi_1(R)
\]

\[
J = \sum_r p(r)\, A_{\mathrm{scaled}}(r)
\]

\[
\mathcal{L} = -J_{\mathrm{teacher}}
 + \lambda_{\mathrm{view}}\mathrm{KL}(p_T\|p_S)
 + \beta_{\mathrm{KL}}\mathrm{KL}(p_{\mathrm{old}}\|p)
 - \beta_{\mathrm{entropy}} H(p)
\]

No PPO ratio, no group-std, no critic, no trajectory row enumeration.

**Defaults:** `λ_view=0.5`, `β_KL=0.05`, `β_entropy=0.005`, AdamW `lr=3e-4`, `wd=1e-4`, grad clip `1.0`, one update per batch.

---

## State paths (PEMS04 H=12)

**state0**

```
H_shared [B,M,N,D=64]
  -> per-node [mean,last,abs_var] -> U0 [B,N,3D]
  -> Linear(3D->128)+GELU+LN
  -> 4-query × 4-head cross-node attention
  -> residual MLP -> state0_hidden [B,256]
```

**state1**

```
Z_q [B,T=H/4,N,C=1]   # never scalar-pooled first
  -> per-node [mean,last,slope,abs_var,std] -> [B,N,5C]
  -> Linear -> 64 + query attn + mean/std summary
  -> zq_hidden [B,128]
  -> concat(s0_proj, zq_hidden) -> MLP -> policy1 logits [B,2]
```

Action heads: **weight=bias=0** → raw probs `(1/3,1/3,1/3)` and `(1/2,1/2)`.

---

## Dual-view OOF cache

For each crossfit sample \(i\) from fold teacher \(k\):

- `H_teacher`, `Zq_teacher` from teacher checkpoint (hash-matched to oracle)
- `H_stable`, `Zq_stable` from final stable supernet
- route losses reused from oracle (not recomputed)

Sharded fp16 on disk. Full cache ≈ **2.4 GiB** (estimate). Tiny diagnostic cache: ≤32/fold under `/tmp`.

Supernet load: `missing=36` all `gain_controller.*` (Plan A) — allowed; any other missing/unexpected → `RuntimeError`.

---

## Feasibility regimes (derived from costs, not hardcoded IDs)

For H12 normalized costs, unique nontrivial masks:

| Regime | Feasible routes | example η |
|--------|-----------------|-----------|
| F1 | `[12]`, `[3,12]` | ~0.36 |
| F2 | `[12]`, `[6,12]`, `[3,12]` | ~0.65 |
| F3 | all four | 1.0 |

Training averages the objective over **all three** every sample. η∉ features.

---

## Execution

`PlanBV2EvalNet`: if QUARTER, run H/4 **once**, then `resume_quarter_to_final` / `resume_quarter_to_progressive`.  
Audit: `H4_calls=1` for both `[3,12]` and `[3,6,12]`; `max_abs_diff=0` vs full route.

---

## Mini-audit outcome (this turn)

See `results/planB_v2_*_audit.json`.

Acceptance checklist passed → recommendation:

**`READY_FOR_FORMAL_V2_RUN`**

(User must run formal training manually with `--confirm-full-run`.)
