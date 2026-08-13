# Plan B-v2 — Diagnosis Findings & Design Note

**Status:** DESIGN ONLY — do not train Plan B-v2 yet.  
**Diagnosis date:** 2026-08-11  
**Formal Plan B-v1:** complete DIRECT collapse (`route [12]=100%` all etas; val regret frozen at `0.083255`).  
**Artifacts:** `results/planB_v1_*.json`, evidence under `results/planB_v1_diagnosis_evidence/`.

This note does **not** claim “GRPO failed.” It asks whether the *current* Plan B-v1 formulation is a valid and suitable group-relative trajectory policy optimization for adaptive F2F computation.

---

## 1. Confirmed diagnosis (tested hypotheses)

| ID | Hypothesis | Result |
|----|------------|--------|
| H1 | `STATE_REWARD_ENVIRONMENT_MISMATCH` | **TRUE (CRITICAL)** |
| H2 | `STATE0_AGGRESSIVE_POOLING` | **TRUE** |
| H3 | `COARSE_FORECAST_SCALAR_BOTTLENECK` | **TRUE** |
| H4 | `CURRENT_ENUMERATED_SURROGATE_MISMATCH` | **TRUE (formulation)**; init cosine(A,B)≈0.97 but not sampling-consistent; \|\|g_C\|\|/\|\|g_A\|\|≈14 |
| H5 | `GROUP_STD_ERASES_MARGIN_MAGNITUDE` | **TRUE** (η=0.5 → advantages ≈[+1,−1]) |
| H6 | Structural zero-gradient ≈40% | **TRUE** (η∈{0,0.25} always 1 route) |
| H7 | Current reward often favors DIRECT | **TRUE** (~38–53% by η) |
| H8 | Quarter prefix recomputed at inference | **TRUE** (H/4 runs twice) |
| H9 | Fold reward heterogeneity | **TRUE** (Fold1 immature) |
| H10 | `beta_kl` unused | **TRUE** |

### Computation graph (PEMS04 H=12)

```
history X [B,12,307,4]
  -> BudgetConditionedAdaptiveF2FNet.extract_pre_route_context
       H_shared [B,4,307,64]
  -> pool_pre_route_context  # mean over patches AND nodes
       s0 [B,64]
  -> policy.encode_s0 / logits0
       a0 ∈ {DIRECT, HALF, QUARTER}

QUARTER branch:
  -> SequentialF2FEnvironment.execute_quarter_prefix
       Z_q [B,3,307,1]
  -> GroupRelativeRefinementPolicy.pool_zq  # mean over time+nodes
       zq_pooled [B,1]
  -> logits1 -> a1 -> terminal route
```

Flags:

- `STATE0_AGGRESSIVE_POOLING = TRUE`
- `COARSE_FORECAST_SCALAR_BOTTLENECK = TRUE`
- `STATE_REWARD_ENVIRONMENT_MISMATCH = TRUE`  
  state = **stable full-train supernet**; reward = **fold-specific OOF teacher** losses.

Fold teacher vs stable pooled-s0 relative L2 (n=32/fold): Fold1≈0.67, Fold2≈0.62, Fold3≈0.44, Fold4≈0.92.

---

## 2. Why v1 collapses (ranked)

1. **Environment mismatch (CRITICAL)** — policy features and oracle rewards come from different forecasting models. Supervision is off-policy w.r.t. the state generator.
2. **Collapse dynamics (CRITICAL)** — reward already selects DIRECT on 38–53% of OOF; group-std turns every η=0.5 duel into ±1; 40% of η draws give zero advantage; init is seed-unstable and often DIRECT-heavy; formal eval metric dilutes nontrivial η with η∈{0,0.25}.
3. **Information bottleneck (HIGH)** — global-mean s0 (effective rank ~3 for 95% var) and scalar mean(Z_q) discard node/temporal structure needed to decide refine vs jump.

Objective issues (enumeration, branch multiplicity, unused KL) are real but secondary to (1)+(2)+(3) for the observed total DIRECT collapse.

---

## 3. Candidate V2-A — Exact Full-Information Group-Relative Trajectory Policy Optimization

**Motivation:** For each `(x, η)`, the feasible terminal set has size ≤4 and rewards are known from the oracle. Stochastic sampling is unnecessary.

### Objective

For sample \(i\) with feasible trajectories \(\mathcal{T}_i\):

\[
J_i = \sum_{\tau \in \mathcal{T}_i} \pi_\theta(\tau \mid s_i)\, A_i(\tau)
\]

with **mean-centered** (not group-std) utilities:

\[
A_i(\tau) = R_i(\tau) - \frac{1}{|\mathcal{T}_i|}\sum_{\tau'} R_i(\tau')
\]

Maximize \(\mathbb{E}_i[J_i]\) (equivalently minimize \(-J\)).

Optional trust region / KL:

\[
\mathcal{L} = -J + \beta_{\mathrm{KL}}\, \mathrm{KL}(\pi_{\mathrm{old}} \,\|\, \pi_\theta)
\]

or reverse KL / TV ball — pick one and actually wire `beta_kl` into the loss.

### Properties

- No critic / GAE / value head.
- No pretence that uniform enumeration equals \(\pi_{\mathrm{old}}\) sampling.
- Branch probabilities factor correctly: \(\pi(\text{QUARTER}\to\cdot)=\pi_0(Q)\,\pi_1(\cdot\mid Z_q)\); optimize the **trajectory** distribution, not duplicated rows.
- Compatible with hard η masks (eta never in features).

### When to prefer V2-A

Default recommendation after this diagnosis: **V2-A**.

---

## 4. Candidate V2-B — Proper sampled trajectory-level GSPO-style optimization

If a sampled RL story is desired:

1. Sample \(\tau \sim \pi_{\mathrm{old}}(\cdot\mid s)\) (respecting masks).
2. Importance ratio at **trajectory** level: \(\rho=\exp(\log\pi_\theta(\tau)-\log\pi_{\mathrm{old}}(\tau))\).
3. Clip \(\rho\) and apply group-relative advantage computed on a **fresh group** of samples from \(\pi_{\mathrm{old}}\) for the same state (true GSPO), not on a uniform enumeration of all terminals.

**Do not** keep “enumerate all terminals → mean over rows → call it GRPO/GSPO.”

V2-B is more complex and higher variance; only choose it if experiments show V2-A under-explores or if online non-oracle rewards appear later.

---

## 5. State design candidate (modules only; not implemented)

### state0 (pre-route)

Do **not** use global mean only.

From the same `H_shared` tap (no new forecasting backbone), preserve:

- temporal mean / last / absolute variation (per node)
- spatial summary: mean + std (and/or lightweight query attention)

Diagnostic structured features already show ~2.15× variance trace vs global-mean s0; PCA of current s0 collapses to ~3 dims.

### state1 (after quarter)

Do **not** use scalar `mean(Z_q)`.

Encode Z_q structure:

- temporal trajectory (mean / last / slope)
- node heterogeneity (spatial std / max-min or query attention)

Diagnostic probe (chronological OOF split): predicting prefer `[3,6,12]` over `[3,12]`:

- scalar Z_q AUC ≈ **0.53**
- structured Z_q AUC ≈ **0.66**

---

## 6. Environment-consistent OOF state (strong recommendation)

Precompute for each crossfit sample from the **same fold teacher** that produced its route losses:

- \(H_{\mathrm{shared}}^{(-k)}\)
- \(Z_q^{(-k)}\)
- route losses \(L^{(-k)}\) (already in oracle)

Policy training then uses **one environment** for state and reward.

Diagnosis shows this is **necessary**, not optional: stable vs teacher pooled-s0 relative L2 is large (especially Fold4 ≈0.92). Implementation waits until Plan B-v2 scaffolding is approved; do not silently mix teachers again.

Fold1 teacher maturity (`n_teacher=2013`) remains noisier (G36 std≈0.30 vs ~0.15–0.17). Quantified as `FOLD_REWARD_HETEROGENEITY`; do not drop Fold1 yet — first fix environment consistency and objective.

---

## 7. Reward / η / validation hygiene (v2 checklist)

Without yet retuning λ:

- Keep reporting STRICT MAE oracle, δ-tolerance oracle, and reward-argmax separately.
- Sample η from **nontrivial** set `{0.5, 0.75, 1.0}` for gradient updates (or reweight); keep `{0, 0.25}` only for evaluation of mask legality.
- Prefer mean-centered advantages for full-info V2-A; treat group-std as optional diagnostic.
- Model selection on nontrivial η only (diagnosis: all-η regret 0.083 vs nontrivial-only ≈0.139 under collapsed DIRECT policy).

Theoretical MAE improvement vs DIRECT required under current (λq=10, λc=1, δ=0.05) when r is best:

| Route | min ΔMAE to beat DIRECT on reward |
|-------|-------------------------------------|
| `[6,12]` | ≈0.0797 |
| `[3,12]` | ≈0.0662 |
| `[3,6,12]` | ≈0.0959 |

---

## 8. Execution hygiene

`PlanBPolicyEvalNet.select_route_ids` runs quarter prefix; `forward` → `_execute_routes_bucketed` runs H/4 again.

- `QUARTER_PREFIX_RECOMPUTED_AT_INFERENCE = TRUE`
- prefix+resume vs full route: `max_abs_diff = 0` (safe to cache/resume)

Fix in v2 eval path: resume from cached `prev_forecast` / skip duplicate H/4.

---

## 9. Recommended next step

**Direction: V2-A** (+ environment-consistent OOF states + structured state0/state1 + nontrivial η sampling).

Do **not** retrain Plan B-v1. Do **not** implement speculative training until this design is accepted.

Explicit non-goals until approved:

- no formal Plan B training
- no Plan A / supernet / crossfit teacher retrain
- no TEST oracle
- no overwrite of `group_relative_policy.pt`
