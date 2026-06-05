# Failed Baseline Official Config Audit

Official reference: [GestaltCogTeam/BasicTS @ 7a7f970](https://github.com/GestaltCogTeam/BasicTS/tree/7a7f970)

---

## DCRNN

| Item | Official | Local | Diff |
|------|----------|-------|------|
| Source | `examples/DCRNN/DCRNN_PEMS04.py` | `examples/baselines/DCRNN/DCRNN_PEMS04.py` | — |
| Runner | `SimpleTimeSeriesForecastingRunner` | `DCRNNRunner` | **Runner mismatch** |
| Model | `DCRNN` | `DCRNN` | Same |
| SETUP_GRAPH | `CFG.MODEL.SETUP_GRAPH = True` | `CFG.TRAIN.SETUP_GRAPH = True` | Key differs; local TRAIN flag works with `base_runner` |
| Params | `input_dim=2, horizon=12, rnn_units=64, ...` | Same | Same |
| Optimizer | Adam lr=0.003, eps=1e-3 | Same | Same |
| Scheduler | MultiStepLR milestones=[80], gamma=0.3 | Same | Same |
| Epochs | 200 | 200 | Same |
| Batch size | 64 | 64 | Same |
| FORWARD_FEATURES | `[0, 1]` | `[0, 1]` | Same |
| TARGET_FEATURES | `[0]` | `[0]` | Same |
| CKPT dir | `checkpoints/DCRNN_200` | `checkpoints/baselines/DCRNN_PEMS04_200` | Intentional local layout |

---

## DGCRN

| Item | Official | Local | Diff |
|------|----------|-------|------|
| Source | `examples/DGCRN/DGCRN_PEMS04.py` | `examples/baselines/DGCRN/DGCRN_PEMS04.py` | — |
| Runner | `DGCRNRunner` (extends `SimpleTimeSeriesForecastingRunner`) | `DGCRNRunner` (custom 3-tuple) | **Runner implementation mismatch** |
| Model | `DGCRN` | `DGCRN` | Same |
| Params | `in_dim=2, seq_length=12, rnn_size=64, ...` | Same | Same |
| Optimizer | Adam lr=0.001, wd=1e-4 | Same | Same |
| Scheduler | MultiStepLR [100,150], gamma=0.5 | Same | Same |
| Epochs | 200 | 200 | Same |
| Batch size | 32 (train) | 32 | Same |
| CL | WARM=0, CL_EPOCHS=6, PRED_LEN=12 | Same | Same |
| FORWARD/TARGET | `[0,1]` / `[0]` | Same | Same |

---

## GWNet

| Item | Official | Local | Diff |
|------|----------|-------|------|
| Source | `examples/GWNet/GWNet_PEMS04.py` | `examples/baselines/GWNet/GWNet_PEMS04.py` | — |
| Runner | `SimpleTimeSeriesForecastingRunner` | Same | Same |
| Model | `GraphWaveNet` | Same | Same |
| Params | `in_dim=2, out_dim=12, blocks=4, layers=2, ...` | Same | Same |
| Optimizer | Adam lr=0.002, wd=1e-4 | Same | Same |
| Scheduler | MultiStepLR [1,50,100], gamma=0.5 | Same | Same |
| Epochs | 200 | 200 | Same |
| Batch size | 64 | 64 | Same |
| FORWARD/TARGET | `[0,1]` / `[0]` | Same | Same |

---

## GTS

| Item | Official | Local | Diff |
|------|----------|-------|------|
| Source | `examples/GTS/GTS_PEMS04.py` | `examples/baselines/GTS/GTS_PEMS04.py` | — |
| Runner | `GTSRunner` | Same | Same |
| Model | `GTS` | Same | Same |
| SETUP_GRAPH | `CFG.MODEL.SETUP_GRAPH = True` | `CFG.TRAIN.SETUP_GRAPH = True` | Key differs |
| Params | `dim_fc=162976`, `node_feats` from train slice | Same hard-coded `dim_fc` | **dim_fc incompatible with local data length** |
| Optimizer | Adam lr=0.001, eps=1e-3 | Same | Same |
| Scheduler | MultiStepLR [20,30], gamma=0.1 | Same | Same |
| Epochs | 200 | 200 | Same |
| Batch size | 64 | 64 | Same |
| Loss | `gts_loss` (local `loss.py`) | Same pattern | Same |
| FORWARD/TARGET | `[0,1]` / `[0]` | Same | Same |

---

## STNorm

| Item | Official | Local | Diff |
|------|----------|-------|------|
| Source | `examples/STNorm/STNorm_PEMS04.py` | `examples/baselines/STNorm/STNorm_PEMS04.py` | — |
| Runner | `SimpleTimeSeriesForecastingRunner` | Same | Same |
| Model | `STNorm` | Same | Same |
| Params | `in_dim=2, out_dim=12, channels=32, blocks=4, layers=2` | Same | Same |
| Optimizer | Adam lr=0.002, wd=1e-4 | Same | Same |
| Scheduler | MultiStepLR [1,50], gamma=0.5 | Same | Same |
| Epochs | 100 | 100 | Same |
| Batch size | 64 | 64 | Same |
| FORWARD/TARGET | `[0,1]` / `[0]` | Same | Same |

---

## Conclusion

- **Config hyperparameters** already match official BasicTS for all five models.
- Required fixes are **runner alignment** (DCRNN, DGCRN), **arch compatibility patches** (DCRNN adj device, GWNet/STNorm Conv2d), and **GTS dim_fc derivation** from local `node_feats` shape.
