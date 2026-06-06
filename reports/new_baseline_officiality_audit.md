# New Baseline Officiality Audit

Branch: `feb9_best_rebuild`. Source commit: **GestaltCogTeam/BasicTS `eb65f4b`** (pre-v1.0, last commit with all four models + PeMS04 configs).

Smoke test: `python scripts/smoke_test_new_baselines.py` — **all four OK** (2026-06-06, after `PyWavelets` install via Aliyun mirror).

## Audit table

| Model | Arch source | Config source | Runner | Epochs | Batch size | LR | Scheduler | Forward features | Target features | Compatibility changes | Paper-use ready? |
|-------|-------------|---------------|--------|--------|------------|-----|-----------|------------------|-----------------|----------------------|:----------------:|
| STAEformer | BasicTS `eb65f4b` `baselines/STAEformer/arch/staeformer_arch.py` | BasicTS `eb65f4b` `baselines/STAEformer/PEMS04.py` | `SimpleTimeSeriesForecastingRunner` | 100 | 16 | 0.001 | MultiStepLR milestones [20,25] γ=0.1 | [0,1,2] | [0] | v0.2 CFG layout; `sys.path` root; checkpoint dir `checkpoints/baselines/STAEformer_PEMS04`; `TRAIN.NULL_VAL` | **true** |
| STWave | BasicTS `eb65f4b` `baselines/STWave/arch/stwave_arch.py` | BasicTS `eb65f4b` `baselines/STWave/PEMS04.py` | `SimpleTimeSeriesForecastingRunner` | 80 | 64 | 0.001 | MultiStepLR [65,70,75] γ=0.1 | [0,1,2] | [0] | + `stwave_masked_mae` loss; precompute `adj_gat`/`graphwave` in config; grad clip 5; requires **PyWavelets** (`pywt`) | **true** |
| STDN | BasicTS `eb65f4b` `baselines/STDN/arch/model.py` + `utils.py` | BasicTS `eb65f4b` `baselines/STDN/PEMS04.py` | `STDNRunner` (ported to v0.2 tuple API) | 300 | 64 | 0.001 | StepLR step=10 γ=0.9 | [0,1,2] | [0] | Custom runner in `basicts/runners/runner_zoo/stdn_runner.py`; `MODEL.LPLS` from `get_lpls(adj)`; `SETUP_GRAPH`; requires **torch_geometric** (official STDN dep) | **true** |
| HimNet | BasicTS `eb65f4b` `baselines/HimNet/arch/model/HimNet.py` | BasicTS `eb65f4b` `baselines/HimNet/PEMS04.py` | `HimNetRunner` (ported to v0.2 tuple API) | 200 | 16 | 0.001 | MultiStepLR [30,50] γ=0.1 | [0,1,2] | [0] | + `masked_huber` metric/loss; custom runner; `SETUP_GRAPH`; grad clip 5 | **true** |

## Notes

- **BasicTS master (v1.0+)** and **v0.2 (`7a7f970`)** do not ship these four models; sources are from historical BasicTS commit `eb65f4b`, which matches README author links to official repos.
- Early BasicTS `STAEformer_PEMS04.py` (`2531b2a`) incorrectly used PEMS08 hyperparams (`num_nodes=170`); **`eb65f4b` config is used** (307 nodes, full `MODEL_PARAM`).
- **No channel 3 prior** in any baseline config.
- **12 input → 12 output** confirmed in all configs.
- Author-repo cross-check (not used as primary source): STAEformer `STAEformer.yaml` PEMS04; STDN `PEMSD4_1dim_12.conf` (epochs=200 vs BasicTS 300 — **BasicTS config preserved**).

## Dependency

```bash
pip install PyWavelets -i https://mirrors.aliyun.com/pypi/simple/
# STDN also needs torch_geometric (already present in basicts env if prior graph models work)
```

## UNVERIFIED_DO_NOT_USE_FOR_PAPER

None of the four models above — all marked paper-ready pending full training sanity on PeMS04 data.
