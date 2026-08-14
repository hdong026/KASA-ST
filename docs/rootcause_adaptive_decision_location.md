# Adaptive Decision Location — Root-Cause Design Note

**Status:** diagnosis only. Do **not** implement Bellman-v2 / policy-v3 / GRPO / PPO / new controller yet.

## Scientific context

Repeated adaptive methods (gain regression, GRPO-like, exact policy opt, Bellman Q) share the same symptom:

- η=0.5 / 0.75 often fail to form useful **TEST** Pareto points.

Root-cause audit supports **BOTH_COUPLED**:

1. **CROSSFIT_ENVIRONMENT_MISMATCH** — rolling teachers (train sizes ~2013…8121) produce OOF route gains that are nearly uncorrelated with the final stable supernet’s TRAIN gains.
2. **PRE_ROUTE_INFORMATION_BOTTLENECK** — initial D/M/Q decisions occur **before any explicit future forecast**; there is **no zero-overhead shared neural state** reused by all routes; even an *extra* `extract_pre_route_context` probe only yields **WEAK** G3/G6 predictability. After observing **Z3**, G36 becomes **USEFUL**.

## Observability facts (code-traced)

| η | Initial decision | Z3 available? | Z3 can affect initial D/M/Q? | Z3 can affect Q vs F? |
|---|------------------|---------------|-----------------------------|------------------------|
| 0.5 | D vs Q | only if Q chosen later | **NO** | N/A (F infeasible) |
| 0.75 | D/M/Q | only after Q | **NO** | **NO** (F still infeasible) |
| 1.0 | D/M/Q then maybe Q→F | after Q | **NO** for initial | **YES** for continuation |

## Branch A — retain pre-route adaptivity

**Gate:** a **true zero-overhead** common state must exist **and** show USEFUL G3/G6 predictability under chronological VALID probes.

Today’s audit:

- Verdict: `NO_ZERO_OVERHEAD_COMMON_STATE`
- Best pre-route probes: **WEAK**

Therefore Branch A is **not unlocked**.

If a future executor redesign makes a shared representation *actually reused* by `[12]`, `[6,12]`, `[3,12]` with no extra compute, re-run Part 5+8 before allowing pre-route policies that consume it.

## Branch B — post-Z3 forecast-state-conditioned refinement / optimal stopping

**Evidence favoring Branch B:**

- Post-Z3 G36 signal: **USEFUL**
- ΔSpearman(X+Z3 − X) ≈ **+0.27**; ΔAUC ≈ **+0.07**
- Mid-η decisions that matter most for cost–accuracy tradeoffs currently cannot condition on Z3

**Design sketch (not implemented):**

1. Low-budget η (≤0.5): **conservative / near-deterministic** DIRECT or fixed cheap refinement — do not pretend sample-adaptive D/Q is identified.
2. When budget allows quarter: **execute Z3 once**, then decide jump-to-final vs continue-to-middle using forecast-state features (optimal stopping / Q1-only refinement).
3. Supervision must use a **matched-maturity** OOF environment (see below), not the current rolling immature Fold1-heavy crossfit.

## Matched-maturity crossfit v2 (environment fix first)

Blocked LOO over K=5 TRAIN blocks → each teacher ≈80% data, matched schedule/protocol to stable supernet, purge derived from `P+H` window overlap (not hardcoded 23).

- Manifest: `results/matched_maturity_crossfit_manifest.json`
- Smoke: `scripts/run_matched_maturity_crossfit_smoke.sh` (execution only)
- Full (user-launched later): `scripts/run_matched_maturity_crossfit_full.sh --confirm-full-run ...`

**Order of operations:**

1. `RUN_MATCHED_CROSSFIT_FULL`
2. Re-check teacher↔stable gain agreement
3. Only then `DESIGN_POST_Z3_OPTIMAL_STOPPING` (Branch B)

Do not keep iterating pre-route RL algorithms under the current non-stationary OOF labels.
