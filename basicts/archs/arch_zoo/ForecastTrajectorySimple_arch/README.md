# ForecastTrajectorySimple

This implementation asks one question first: after adding only the forecasting
transitions missing from a mature F2F model, do different samples actually have
different best trajectories?

## Current explicit X--Z relation controller

The active research method remains the original multi-resolution graph over
`2,3,4,6,12`. Run it with:

    python -m basicts.archs.arch_zoo.ForecastTrajectorySimple_arch.run_online_sequential_rl \
      --controller-state xz_relation ...

This option changes only the controller encoder. Ordered history X and the
explicit forecast Z_r produced by the latest executed transition are mapped by
one shared temporal encoder. The actor receives learned signed contrast,
absolute contrast, multiplicative matching, boundary/trend/volatility mismatch,
and low-rank spatial projections of that relation. It does not receive an
unexecuted future forecast. Action feasibility and PPO/RLOO training are
otherwise identical to the previous genuine online controller.

The completed unchanged-graph experiment is at
`checkpoints/ForecastTrajectorySimpleOnlineXZRelation/full_v1/final_report.json`.
It is a negative result: the policy selected `3->12` at the lowest budget and
`3->6->12` at all larger budgets for every VALID/TEST sample, matching the
previous controller while adding approximately 2--3 ms p90 overhead on routes
with online decisions.

## Archived refinement-mechanism experiments

The following modules are retained only as earlier diagnostics:

    python -m basicts.archs.arch_zoo.ForecastTrajectorySimple_arch.diagnose_progressive_graph ...
    python -m basicts.archs.arch_zoo.ForecastTrajectorySimple_arch.diagnose_learned_consistency ...
    python -m basicts.archs.arch_zoo.ForecastTrajectorySimple_arch.run_corrective_terminal_transition ...
    python -m basicts.archs.arch_zoo.ForecastTrajectorySimple_arch.diagnose_optional_z6_observability ...
    python -m basicts.archs.arch_zoo.ForecastTrajectorySimple_arch.run_online_corrective_controller ...

`diagnose_progressive_graph` compares every actually reached forecast to its
pre-transition state under common full-resolution stopping semantics. It
reports signed per-sample MAE change, correction/error alignment, and whether
the previous change predicts the next edge's value on TRAIN and VALID.
`diagnose_learned_consistency` is a diagnosis-only upper-bound probe: its input
is raw history, the reached explicit forecast, and the previous-to-current
change. Its labels may use TRAIN targets, but it is never deployed and never
supervises the online policy.

The original optional graph failed this gate: on VALID, `3->4` and the following
`4->6` improved only 49.1% and 50.1% of samples, respectively, and had negative
mean gains. A locally harm-penalized bridge experiment made these individual
edges more correction-like, but destabilized their composition with frozen
downstream edges and did not produce a canonical-beating route.

`run_corrective_terminal_transition` tested a minimal protected
mechanism. Skip remains the exact native `3->6->12`. The optional action executes
that same frozen terminal transition and adds a zero-initialized bounded
residual derived from explicit X--Z6--Z12 shape mismatch. Only this new residual
module is trained. `run_online_corrective_controller` then learns at the
actually reached Z6 state with bandit feedback: the sampled action is executed
immediately, and reward is its signed per-sample improvement over the canonical
provisional forecast. No unchosen outcome or route cache enters training.

The completed PEMS04 experiment is intentionally a negative result. The
diagnostic probe improved VALID over canonical by 0.0078 MAE, but correlation
with signed benefit remained only 0.057. Genuine on-policy learning consequently
collapsed to the corrected fixed action, recovering 4.96% of VALID oracle
headroom and degrading TEST. The report is at
`checkpoints/ForecastTrajectorySimpleOnlineCorrective/full_v1/final_report.json`.
That controller is archived and is not used by the multi-resolution method.

## One-command PEMS04 experiments

Inference-observable refinement diagnostics (diagnosis only, never policy
supervision):

    cd /home/dhz/KASA-ST
    conda run -n basicts python -m \
      basicts.archs.arch_zoo.ForecastTrajectorySimple_arch.diagnose_refinement_observability \
      --output checkpoints/ForecastTrajectorySimpleObservability/z3.json
    conda run -n basicts python -m \
      basicts.archs.arch_zoo.ForecastTrajectorySimple_arch.diagnose_start_observability \
      --output checkpoints/ForecastTrajectorySimpleObservability/start.json

The first command observes signals already computed by the actually executed
native `START->3` transition: KASA branch disagreement, internal activation
energy, forecast consistency, and history frequency/volatility. It evaluates
only continuations legal from that reached state. The second command evaluates
history-only frequency and cheap multi-view trend disagreement before the first
edge. Both use TRAIN/VALID route losses only as diagnostic labels. They are a
go/no-go gate: if a strong supervised probe using inference-available inputs
cannot beat canonical VALID, those features are not added to the online actor
and another expensive RL run is not justified.

Sequential resolution decisions with physical computation budgets:

    cd /home/dhz/KASA-ST
    GPU=0 bash basicts/archs/arch_zoo/ForecastTrajectorySimple_arch/run_sequential_budget_pems04.sh

This command trains the current budgeted research version: a real on-policy
sequential actor--critic. No route, loss, state-feature, or cost-to-go cache is
read or generated for training. At START the actor observes history and samples
a first resolution. The frozen forecasting model immediately executes that
transition. If the result is not at resolution 12, the actor observes the newly
produced explicit forecast and samples the next legal edge, which is likewise
executed immediately. The terminal physical-scale MAE is the policy return.

TRAIN uses several independently sampled trajectories for each example. Every
one of those trajectories is executed through the real forecast graph; their
leave-one-out returns are only a variance-reduction baseline for REINFORCE, not
counterfactual labels. A fresh execution of the unchanged canonical forecaster
provides an action-independent within-sample control variate, so the return asks
whether the sampled route improved this sample rather than trying to predict
its absolute difficulty. Several clipped PPO updates reuse each freshly
generated batch, after which it is discarded. The critic is auxiliary and is
trained only on returns from sampled, executed trajectories.

Online features retain sensor identity: history contributes `10*N+48` values,
and the currently available explicit forecast adds `10*N`. A shared learned
low-rank spatial projection compresses those sensor-specific values before the
small actor heads; history features are computed once and reused at later
decisions. Deployment calls an actor-only greedy path, avoiding critic and
categorical-distribution kernels; reports include the measured p90 difference
from forecast-only execution for every route.

The command profiles batch-1 synchronized CUDA latency. Fixed and adaptive
executions of each route are alternated in one paired profiling loop so GPU
clock or thermal drift cannot favor one system. The primary constraint combines
a hard physical completion mask with an average-budget primal--dual term.
Deployment greedily chooses the highest-scoring feasible action;
`--stochastic-eval` is an explicit sampling ablation. Actor quality scores do
not depend on `B`, so increasing a hard budget only adds feasible opportunities
and never forces refinement. In both formulations the quantities are measured p90
milliseconds, including actor overhead, rather than an artificial eta:

- hard per-sample: the selected route's measured end-to-end p90 cost, including
  online policy overhead, must not exceed user budget `B` in milliseconds;
- average: TRAIN updates a dual price from the observed cost of sampled online
  trajectories to keep mean p90 route cost at or below `B`.

Reports compare common millisecond budgets against fixed paths profiled without
policy overhead. Average-budget evaluation also includes the lower convex
envelope of sample-independent fixed-path mixtures and a label-only sample-wise
allocation oracle under the same physical budget. The complete route panel is
computed only on VALID/TEST for baselines and headroom analysis; it never enters
the training objective. TEST is evaluated once after VALID checkpoint selection.

Train the progressive selector from the already-trained bridge checkpoint,
select its checkpoint by VALID learned-selection MAE, and evaluate TEST once:

    cd /home/dhz/KASA-ST
    GPU=0 bash basicts/archs/arch_zoo/ForecastTrajectorySimple_arch/run_selector_pems04.sh

The selector freezes both `model.f2f` and `model.bridges`. It learns two small
pairwise benefit heads from TRAIN trajectory losses: history selects the first
state (2/3/4), and history plus the explicit native `Z3` selects 4/6/12 if state
3 was reached. VALID targets are used only for checkpoint selection; TEST is
evaluated only after that checkpoint has been restored.

Full bridge training followed by validation/test trajectory enumeration:

    cd /home/dhz/KASA-ST
    GPU=0 bash basicts/archs/arch_zoo/ForecastTrajectorySimple_arch/run_pems04.sh

The default validated schedule is 50 epochs with validation early stopping
(patience 12), intermediate explicit-state weight 0.25, and normalized bridge
correction limit 2.0.

Pipeline smoke test (one epoch, two batches per split):

    cd /home/dhz/KASA-ST
    GPU=0 bash basicts/archs/arch_zoo/ForecastTrajectorySimple_arch/run_pems04.sh --smoke

The script activates the `basicts` conda environment itself. Results are saved
under `checkpoints/ForecastTrajectorySimple/seed1_<timestamp>/`, including
`bridges_best.pt`, `history.json`, and `headroom_report.json`.

## Invariants

- The complete original ChainForecasting is stored as model.f2f.
- [3, 6, 12] calls model.f2f.forward directly. No bridge, adapter, or new
  arithmetic is inserted into the canonical path.
- Native edges are exactly START->3, 3->6, and 6->12.
- Every requested non-native edge owns an independent residual KASA bridge.
  It reads history and the current explicit forecast, anchors on a resampled
  forecast (or last-observation persistence for a start edge), and predicts a
  bounded correction. It shares frozen codebooks, not a forecasting trunk.
- Every intermediate bridge state is supervised as an explicit pooled-target
  forecast. It cannot silently become a latent code for the next transition.
- The mature F2F is always frozen and kept in eval mode.
- The online actor state is limited to current history/forecast summaries,
  reached resolution, measured consumed prefix cost, and remaining budget.
  Targets and unexecuted future forecasts are never policy inputs.
- The actor and bridges are separate: online policy training freezes the mature
  F2F and all bridge parameters. The canonical route remains an exact direct
  call to the original F2F implementation.

The alternate-trajectory executor targets the verified canonical
spatial_placement="final" semantics (and also accepts "none"). Every route
ending at 12 uses the same mature final spatial refine. Unsupported spatial
organizations fail at construction instead of silently changing semantics.

## Minimal use

    from basicts.archs import ForecastTrajectorySimple

    model = ForecastTrajectorySimple(
        **canonical_model_args,
        trajectories=[
            [3, 6, 12], [3, 12], [2, 4, 12], [4, 12], [3, 4, 6, 12]
        ],
    )
    model.load_pretrained_f2f("path/to/ChainForecasting_best_val_MAE.pt")

    # Exact original computation.
    y_canonical = model(history, trajectory=[3, 6, 12])

    # A graph trajectory containing learned bridges.
    trace = model(history, trajectory=[3, 4, 6, 12], return_all=True)

    # Optimizer scope: bridges only.
    optimizer = torch.optim.Adam(model.bridge_parameters(), lr=2e-3)

Passing trajectories=[route_for_sample_0, ...] executes sample-specific routes
by grouping equal routes inside the batch. This is an execution API, not a
policy.

## Recommended experiment order

1. Load and freeze a mature F2F checkpoint. Assert exact canonical equality.
2. Train only bridges over a declared trajectory pool. Use final forecast loss
   as the primary objective (trajectory_supervision_loss); optional intermediate
   bridge supervision defaults to zero.
3. On held-out data, execute every trajectory for every sample and call
   headroom_from_predictions. Report canonical, best fixed route, sample-wise
   oracle, oracle route counts, and both oracle improvements. The oracle uses
   labels and is analysis only.
4. Only if oracle headroom is real, measure each route with
   profile_trajectory_latency. Apply quality_latency_objective using measured
   time and continuous lambda. A real deployment ceiling only masks routes.
5. Train `ProgressiveTrajectorySelector` separately. Its first decision sees
   history only; its second decision sees history and the current explicit
   `Z3`. Neither decision sees targets or an unexecuted future state.
