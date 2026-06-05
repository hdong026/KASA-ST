# PeMS04 12→12 Baseline Inventory

Generated for branch `feb9_best_rebuild`. Architecture code under `basicts/archs/arch_zoo/` was **not** modified during this inventory (only `basicts/archs/__init__.py` exports were restored for baseline imports).

**STD-MAE excluded intentionally** from the predictor-only baseline suite (pretraining framework; handle separately).

## Summary

| Category | Count |
|----------|------:|
| A: arch exists + verified PeMS04 config | 11 |
| B: missing arch / no official PeMS04 12→12 config | 10 |
| Ready to run (smoke test instantiate OK) | 11 |

Official config source for all ready A-list models: **GestaltCogTeam/BasicTS** commit [`7a7f970`](https://github.com/GestaltCogTeam/BasicTS/commit/7a7f97088df685d39342a33b4424193b3d927c2f) (`examples/<Model>/<Model>_PEMS04.py`), BasicTS v0.2.0.

Local copies: `examples/baselines/<Model>/<Model>_PEMS04.py`. Adaptations vs upstream: repo `sys.path` root, `CFG.TRAIN.NULL_VAL`, `checkpoints/baselines/…` save dir, DCRNN/GTS use dedicated runners + `CFG.TRAIN.SETUP_GRAPH` (upstream used `CFG.MODEL.SETUP_GRAPH` with `SimpleTimeSeriesForecastingRunner`). Hyperparameters unchanged.

---

## A. Models with local architecture

| Model | Arch exists? | Arch path | Import works? | PeMS04 config exists? | Official/BasicTS config found? | Action | Notes |
|-------|:------------:|-----------|:-------------:|:---------------------:|:------------------------------:|--------|-------|
| STID | Yes | `basicts/archs/arch_zoo/stid_arch` | Yes | Yes | Yes — BasicTS `examples/STID/STID_PEMS04.py` @ 7a7f970; also [zezhishao/STID `stid/PEMS04.py`](https://github.com/zezhishao/STID) | **Ready** | `FORWARD_FEATURES=[0,1,2]`, `TARGET=[0]`; 200 epochs, lr 0.002 |
| D2STGNN | Yes | `basicts/archs/arch_zoo/d2stgnn_arch` | Yes | Yes | Yes — BasicTS + [D2STGNN `configs/PEMS04.yaml`](https://github.com/zezhishao/D2STGNN) | **Ready** | Needs `adj_mx.pkl`; CL in `CFG.TRAIN.CL` |
| AGCRN | Yes | `basicts/archs/arch_zoo/agcrn_arch` | Yes | Yes | Yes — BasicTS; hyperparams match [LeiBAI/AGCRN `PEMSD4_AGCRN.conf`](https://github.com/LeiBAI/AGCRN) | **Ready** | Flow-only `[0]`; lr 0.003, batch 64 |
| GWNet | Yes | `basicts/archs/arch_zoo/gwnet_arch` (`GraphWaveNet`) | Yes | Yes | Yes — BasicTS `examples/GWNet/GWNet_PEMS04.py` | **Ready** | `FORWARD_FEATURES=[0,1]` |
| STGCN | Yes | `basicts/archs/arch_zoo/stgcn_arch` | Yes | Yes | Yes — BasicTS | **Ready** | Flow-only `[0]`; `normlap` adj |
| DCRNN | Yes | `basicts/archs/arch_zoo/dcrnn_arch` | Yes | Yes | Yes — BasicTS | **Ready** | `DCRNNRunner`, `SETUP_GRAPH`; `[0,1]` |
| DGCRN | Yes | `basicts/archs/arch_zoo/dgcrn_arch` | Yes | Yes | Yes — BasicTS | **Ready** | `DGCRNRunner`, CL |
| MTGNN | Yes | `basicts/archs/arch_zoo/mtgnn_arch` | Yes | Yes | Yes — BasicTS | **Ready** | `MTGNNRunner`, 100 epochs |
| StemGNN | Yes | `basicts/archs/arch_zoo/stemgnn_arch` | Yes | Yes | Yes — BasicTS | **Ready** | Flow-only `[0]`; RMSprop |
| GTS | Yes | `basicts/archs/arch_zoo/gts_arch` | Yes | Yes | Yes — BasicTS + `examples/baselines/GTS/loss.py` | **Ready** | `GTSRunner`; loads train `node_feats` from pkl |
| STNorm | Yes | `basicts/archs/arch_zoo/stnorm_arch` | Yes | Yes | Yes — BasicTS | **Ready** | `[0,1]`; 100 epochs |

## B. Models not in local arch_zoo (or no verified PeMS04 config)

| Model | Arch exists? | Arch path | Import works? | PeMS04 config exists? | Official/BasicTS config found? | Action | Notes |
|-------|:------------:|-----------|:-------------:|:---------------------:|:------------------------------:|--------|-------|
| STAEformer | No | — | — | No | No in BasicTS @ master or v0.2 | **TODO** | Not in GestaltCogTeam/BasicTS examples; add from official STAEformer repo when verified |
| STWave | No | — | — | No | No | **TODO** | — |
| STDN | No | — | — | No | No in BasicTS (local LSTNN checkpoint config is **unverified** for paper) | **TODO** | Do not use LSTNN `STDN_PEMS04.py` without official cross-check |
| HimNet | No | — | — | No | No | **TODO** | — |
| DFDGCN | No | — | — | No | No | **TODO** | — |
| STPGNN | No | — | — | No | No | **TODO** | — |
| BigST | No | — | — | No | No | **TODO** | — |
| STEP | No | — | — | No | No | **TODO** | Pretraining-style; separate from plain baselines |
| STGODE | No | — | — | No | No | **TODO** | — |
| MegaCRN | No (local) | — | — | No | Partial — BasicTS v0.2 has `basicts/archs/arch_zoo/megacrn` + `MegaCRN_METR-LA.py` only | **TODO** | No official PeMS04 12→12 config; do not guess |

## C. Explicitly deprioritized

| Model | Action | Notes |
|-------|--------|-------|
| STD-MAE / STDMAE | **Excluded** | Pretraining framework |
| PatchTST, DLinear, iTransformer | Later | Only if clean BasicTS PeMS04 12→12 configs appear |
| PDFormer, GMAN, DST-Mamba | Later | Low priority unless official configs are clean |

## Other arch_zoo entries (not in PeMS04 baseline target list)

Present under `basicts/archs/arch_zoo/`: `autoformer_arch`, `fedformer_arch`, `informer_arch`, `linear_arch`, `hi_arch`, `pyraformer_arch`, `KASA_arch_v2` (do not modify for this task).

## Scanned paths

- `basicts/archs/arch_zoo/` — all target A architectures present
- `examples/baselines/` — 11 verified configs added
- `examples/KASAST_v2/` — KASA configs (untouched)
- `scripts/smoke_test_baselines.py`, `scripts/run_baselines_pems04.py` — added

## Smoke test (2026-06-01)

```bash
/home/dhz/miniconda3/envs/basicts/bin/python scripts/smoke_test_baselines.py
```

All 11 ready baselines: **import + instantiate OK** (no training).

## Data note

`datasets/PEMS04/` contains `data_in12_out12.pkl`, `adj_mx.pkl`, `scaler_in12_out12.pkl` (4-channel KASA-style data possible). Baseline configs use `FORWARD_FEATURES` without channel 3 (prior).
