# PeMS04 Data Protocols

## Current default (repository standard)

The repository now uses **PeMS04 12→12** with the **official 6:2:2** train/validation/test split.

| Item | Value |
|------|-------|
| Input length | 12 |
| Output length | 12 |
| Train / valid / test | 60% / 20% / 20% |
| Output directory | `datasets/PEMS04` |
| Index file | `index_in12_out12.pkl` |
| Data files | `data_in12_out12.pkl`, `data_in12_out12.npz` |
| Scaler | `scaler_in12_out12.pkl` |

### Channels (4)

| Channel | Meaning |
|---------|---------|
| 0 | Normalized traffic flow |
| 1 | Time of day (ToD) |
| 2 | Day of week (DoW) |
| 3 | Train-only FFT-filtered weekly prior |

Channel 3 is computed **only from the training split** (weekly average → FFT low-pass → iFFT → tile → normalize with train stats). Validation and test flows are never used to build the prior.

### Generation script

```bash
python scripts/data_preparation/PEMS04/generate_holost_data.py
```

After generation, `datasets/PEMS04/protocol_audit.json` records the protocol metadata.

### Audit

```bash
python scripts/audit_pems04_protocol.py --data_dir datasets/PEMS04
```

Report: `reports/pems04_protocol_audit.md`

---

## Previous development protocol (deprecated)

The earlier default was **7:1:2** (`TRAIN_RATIO=0.7`, `VALID_RATIO=0.1`). That split is **no longer the repository default** and should not be used for new experiments or external comparisons.

If on-disk data still shows 7:1:2 sample counts, regenerate with `generate_holost_data.py`.

---

## Model usage

### KASA (`examples/KASAST_v2/KASAST_PEMS04.py`)

- `FORWARD_FEATURES = [0, 1, 2, 3]` — uses train-only prior (ch3)
- `TARGET_FEATURES = [0]` — predicts flow
- `prior_mapper_type = "mlp"`
- `use_pre_temporal_spatial_enhancement = False`

### Baselines (`examples/baselines/*/*_PEMS04.py`)

- Share the same `datasets/PEMS04` index and data files
- `TARGET_FEATURES = [0]` for all ready baselines
- **Do not use channel 3** (official input features only, e.g. `[0]`, `[0,1]`, or `[0,1,2]`)

### D2STGNN

- Official BasicTS hyperparameters unchanged
- `FORWARD_FEATURES = [0, 1, 2]`, `TARGET_FEATURES = [0]`
- Comparable to external BasicTS/D2STGNN results **after** data is regenerated to 6:2:2

---

## Fairness notes

- **Same split**: KASA and all baselines read identical train/valid/test windows from `datasets/PEMS04`.
- **Feature asymmetry**: Only KASA consumes ch3 prior; baselines do not.
- **Not equal-compute**: Epochs, batch size, and schedulers differ per official configs.

---

## Related scripts

| Script | Purpose |
|--------|---------|
| `generate_holost_data.py` | **Primary** — 4-channel 6:2:2 data for KASA + shared baselines |
| `generate_training_data.py` | 3-channel 6:2:2 (no prior); legacy BasicTS-style helper |
| `audit_pems04_protocol.py` | Verify on-disk split, channels, and config alignment |
