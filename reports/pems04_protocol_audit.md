# PeMS04 Protocol Audit

Repo: `/home/dhz/KASA-ST`
Dataset dir: `datasets/PEMS04`

## 1. On-disk data split & shape

- **processed_data shape**: `(16992, 307, 4)` (T, N, C)
- **num channels**: 4
- **channel meanings (inferred)**:
  - ch0: flow (normalized traffic)
  - ch1: time of day (ToD)
  - ch2: day of week (DoW)
  - ch3: train-only prior (weekly spectral template, HoloST)
- **input length**: 12
- **output length**: 12
- **train samples**: 11878
- **valid samples**: 1697
- **test samples**: 3394
- **ratios (train/valid/test)**: 0.7000 / 0.1000 / 0.2000
- **split classification**: **7:1:2**
- **train index first/last**: `(0, 12, 24)` / `(11877, 11889, 11901)`
- **valid index first/last**: `(11878, 11890, 11902)` / `(13574, 13586, 13598)`
- **test index first/last**: `(13575, 13587, 13599)` / `(16968, 16980, 16992)`
- **channel 3 exists in processed_data**: **True**
- **index file**: `datasets/PEMS04/index_in12_out12.pkl`
- **data pkl**: `datasets/PEMS04/data_in12_out12.pkl`
- **data npz present**: `True`
- **npz path**: `datasets/PEMS04/data_in12_out12.npz`
- **npz keys**: `['processed_data', 'weekly_spectral_template', 'slots_per_day', 'slots_per_week']`
- **weekly_spectral_template in npz**: **True**
- **weekly_spectral_template shape**: `(2016, 307, 1)`

- **protocol_audit.json**: not found

## Config comparison: KASA vs D2STGNN

| Field | KASA | D2STGNN | Match? |
|-------|------|------|--------|
| CKPT_SAVE_DIR | `checkpoints/KASA_v2_100` | `checkpoints/baselines/D2STGNN_PEMS04_200` | no |
| DATASET_INPUT_LEN | `12` | `12` | yes |
| DATASET_NAME | `PEMS04` | `PEMS04` | yes |
| DATASET_OUTPUT_LEN | `12` | `12` | yes |
| FORWARD_FEATURES | `[0, 1, 2, 3]` | `[0, 1, 2]` | no |
| LR_SCHEDULER | `{'type': 'MultiStepLR', 'param': {'milestones': [1, 35, 60, 80, 95], 'gamma': 0.5}}` | `{'type': 'MultiStepLR', 'param': {'milestones': [1, 30, 38, 46, 54, 150], 'gamma': 0.5}}` | no |
| NUM_EPOCHS | `100` | `200` | no |
| OPTIM | `{'lr': 0.002, 'weight_decay': 0.0001}` | `{'lr': 0.002, 'weight_decay': 1e-05, 'eps': 1e-08}` | no |
| TARGET_FEATURES | `[0]` | `[0]` | yes |
| TEST.DATA.DIR | `datasets/PEMS04` | `datasets/PEMS04` | yes |
| TRAIN.BATCH_SIZE | `32` | `16` | no |
| TRAIN.DATA.DIR | `datasets/PEMS04` | `datasets/PEMS04` | yes |
| VAL.DATA.DIR | `datasets/PEMS04` | `datasets/PEMS04` | yes |

- Same index/data file stem: **yes** (`datasets/PEMS04/index_in12_out12.pkl` + `data_in12_out12.pkl`)
- KASA uses channel 3: **True**
- D2STGNN uses channel 3: **False**

## Baseline FORWARD_FEATURES audit

| Model | FORWARD_FEATURES | Uses ch3? |
|-------|------------------|-----------|
| AGCRN | `[0]` | no |
| D2STGNN | `[0, 1, 2]` | no |
| DCRNN | `[0, 1]` | no |
| DGCRN | `[0, 1]` | no |
| GTS | `[0, 1]` | no |
| GWNet | `[0, 1]` | no |
| MTGNN | `[0, 1]` | no |
| STGCN | `[0]` | no |
| STID | `[0, 1, 2]` | no |
| STNorm | `[0, 1]` | no |
| StemGNN | `[0]` | no |

## Preprocessing script defaults

### `generate_holost_data.py` (__main__ defaults)

| Setting | Value |
|---------|-------|
| HISTORY_SEQ_LEN | `12` |
| FUTURE_SEQ_LEN | `12` |
| TRAIN_RATIO | `0.6` |
| VALID_RATIO | `0.2` |

- Produces **4 channels** (flow, ToD, DoW, prior) and writes `.npz` with `weekly_spectral_template`.
- Default split: **6:2:2** (`TRAIN_RATIO=0.6`, `VALID_RATIO=0.2`, test remainder 0.2).
- Writes `protocol_audit.json` after generation.

### `generate_training_data.py` (__main__ defaults)

| Setting | Value |
|---------|-------|
| HISTORY_SEQ_LEN | `12` |
| FUTURE_SEQ_LEN | `12` |
| TRAIN_RATIO | `0.6` |
| VALID_RATIO | `0.2` |

- Default split: **6:2:2** (`TRAIN_RATIO=0.6`, `VALID_RATIO=0.2`).
- Default window: **12→12** (`HISTORY_SEQ_LEN=12`, `FUTURE_SEQ_LEN=12`).
- Produces **3 channels** only (no prior). Use `generate_holost_data.py` for 4-channel KASA data.

### Official BasicTS v0.2 `generate_training_data.py` (reference)

| Setting | Official value |
|---------|----------------|
| HISTORY_SEQ_LEN | 12 |
| FUTURE_SEQ_LEN | 12 |
| TRAIN_RATIO | 0.6 |
| VALID_RATIO | 0.2 |
| Channels | 3 (flow, ToD, DoW) |
| Split | **6:2:2** |

## Protocol inconsistency summary

- On-disk PeMS04 index still matches **7:1:2** — **regenerate** with `python scripts/data_preparation/PEMS04/generate_holost_data.py`.
- On-disk `processed_data` has **4 channels** including train-only prior (ch3).
- Repo default generation scripts now target **6:2:2**; on-disk data must be regenerated to match.

## Fairness / comparability conclusions

### Is D2STGNN comparable to official BasicTS?

- **Pending regeneration**: on-disk split is **7:1:2**; run `generate_holost_data.py` for official 6:2:2.
- D2STGNN config does **not** use channel 3.

### Is D2STGNN internally fair against KASA?

- **Same dataset path**: both use `datasets/PEMS04` with `index_in12_out12.pkl` / `data_in12_out12.pkl`.
- **Same split**: both see identical train/valid/test windows.
- **Feature asymmetry**: KASA FORWARD `[0,1,2,3]` uses prior ch3; D2STGNN FORWARD `[0,1,2]` does not.
- **Not equal-compute**: KASA 100 epochs / bs32 vs D2STGNN 200 epochs / bs16 + curriculum learning.
- **Conclusion**: Same data split and target (flow ch0), but **not feature-fair** (KASA gets extra prior channel) and **not compute-fair** (epochs/batch/scheduler differ by design).

### Exact protocol in use (this repo)

1. PeMS04 12→12 windows from `datasets/PEMS04/index_in12_out12.pkl`.
2. Split: **7:1:2** (11878/1697/3394 samples).
3. Channels: **4** (with train-only prior in ch3).
4. Baselines: typically FORWARD `[0]` or `[0,1]` or `[0,1,2]`; never ch3.
5. KASA: FORWARD `[0,1,2,3]`, TARGET `[0]`, `input_dim=4`.
