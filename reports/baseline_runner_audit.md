# Baseline Runner & Config Audit (PeMS04 12→12)

Branch: `feb9_best_rebuild`  
Official source: [GestaltCogTeam/BasicTS @ 7a7f970](https://github.com/GestaltCogTeam/BasicTS/tree/7a7f970)

## Runner alignment (failed baselines fixed)

| Model | Previous runner | Fixed runner | Official BasicTS runner | Status | Notes |
|-------|-----------------|--------------|----------------------|--------|-------|
| DCRNN | `DCRNNRunner` | `SimpleTimeSeriesForecastingRunner` | `SimpleTimeSeriesForecastingRunner` | Fixed | + `MODEL.SETUP_GRAPH`; `dcrnn_cell` adj→GPU; removed inference assert in `dcrnn_arch` |
| DGCRN | `DGCRNRunner` (3-tuple) | `DGCRNRunner` (extends `SimpleTimeSeriesForecastingRunner`) | `DGCRNRunner` | Fixed | Replaced local runner with official implementation |
| GWNet | `SimpleTimeSeriesForecastingRunner` | `SimpleTimeSeriesForecastingRunner` | `SimpleTimeSeriesForecastingRunner` | Fixed | `gwnet_arch`: Conv1d→Conv2d for 4D tensors |
| GTS | `GTSRunner` | `GTSRunner` | `GTSRunner` | Fixed | `dim_fc` derived from train `node_feats`; + `MODEL.SETUP_GRAPH` |
| STNorm | `SimpleTimeSeriesForecastingRunner` | `SimpleTimeSeriesForecastingRunner` | `SimpleTimeSeriesForecastingRunner` | Fixed | `stnorm_arch`: Conv1d→Conv2d for 4D tensors |

## Runnable baselines (unchanged)

| Model | Runner | Official match |
|-------|--------|----------------|
| STID | `SimpleTimeSeriesForecastingRunner` | Yes |
| D2STGNN | `SimpleTimeSeriesForecastingRunner` | Yes |
| AGCRN | `SimpleTimeSeriesForecastingRunner` | Yes |
| STGCN | `SimpleTimeSeriesForecastingRunner` | Yes |
| MTGNN | `MTGNNRunner` | Yes |
| StemGNN | `SimpleTimeSeriesForecastingRunner` | Yes |

## Compatibility patches (not hyperparameter changes)

| File | Change |
|------|--------|
| `basicts/archs/arch_zoo/dcrnn_arch/dcrnn_cell.py` | Move adjacency to GPU in `_gconv` |
| `basicts/archs/arch_zoo/dcrnn_arch/dcrnn_arch.py` | Remove strict inference assert (align official) |
| `basicts/archs/arch_zoo/gwnet_arch/gwnet_arch.py` | Conv1d→Conv2d |
| `basicts/archs/arch_zoo/stnorm_arch/stnorm_arch.py` | Conv1d→Conv2d |
| `basicts/runners/runner_zoo/dgcrn_runner.py` | Official BasicTS runner |
| `examples/baselines/DCRNN/DCRNN_PEMS04.py` | Runner + `MODEL.SETUP_GRAPH` |
| `examples/baselines/GTS/GTS_PEMS04.py` | `dim_fc` inference from local data |

## Known runner-script issue (multi-seed)

`scripts/run_baselines_pems04.py` `make_seed_config()` corrupts `CKPT_SAVE_DIR` lines with `+ str(NUM_EPOCHS)`. Use `--seeds 1` until regex is fixed.

## Smoke test

```bash
python scripts/smoke_test_failed_baselines.py
```

All five previously failed models pass forward smoke test after fixes.
