# PeMS04 12→12 Predictor Baselines

**Data protocol:** `datasets/PEMS04` uses official **6:2:2** split, 12→12 windows, 4 channels (baselines do **not** use ch3 prior). See `docs/pems04_protocols.md`.

Verified configs copied from **GestaltCogTeam/BasicTS** v0.2 (`7a7f970`).

Entry point:

```bash
python examples/run.py --cfg examples/baselines/<Model>/<Model>_PEMS04.py --gpus 0
```

Use the `basicts` conda env (or any env with `easytorch` + project deps).

## Ready baselines

| Model | Ready? | Source | Config | Command | Notes |
|-------|:------:|--------|--------|---------|-------|
| STID | Yes | BasicTS v0.2 | `examples/baselines/STID/STID_PEMS04.py` | `python examples/run.py --cfg examples/baselines/STID/STID_PEMS04.py --gpus 0` | `[0,1,2]` → target `[0]` |
| D2STGNN | Yes | BasicTS v0.2 | `examples/baselines/D2STGNN/D2STGNN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/D2STGNN/D2STGNN_PEMS04.py --gpus 0` | needs `adj_mx.pkl` |
| AGCRN | Yes | BasicTS v0.2 | `examples/baselines/AGCRN/AGCRN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/AGCRN/AGCRN_PEMS04.py --gpus 0` | flow-only `[0]` |
| GWNet | Yes | BasicTS v0.2 | `examples/baselines/GWNet/GWNet_PEMS04.py` | `python examples/run.py --cfg examples/baselines/GWNet/GWNet_PEMS04.py --gpus 0` | `SimpleTimeSeriesForecastingRunner`; arch patch Conv1d→Conv2d |
| STGCN | Yes | BasicTS v0.2 | `examples/baselines/STGCN/STGCN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/STGCN/STGCN_PEMS04.py --gpus 0` | flow-only `[0]` |
| DCRNN | Yes | BasicTS v0.2 | `examples/baselines/DCRNN/DCRNN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/DCRNN/DCRNN_PEMS04.py --gpus 0` | `SimpleTimeSeriesForecastingRunner` + `SETUP_GRAPH`; adj GPU patch |
| DGCRN | Yes | BasicTS v0.2 | `examples/baselines/DGCRN/DGCRN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/DGCRN/DGCRN_PEMS04.py --gpus 0` | `DGCRNRunner` (official BasicTS impl) |
| MTGNN | Yes | BasicTS v0.2 | `examples/baselines/MTGNN/MTGNN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/MTGNN/MTGNN_PEMS04.py --gpus 0` | `MTGNNRunner` |
| StemGNN | Yes | BasicTS v0.2 | `examples/baselines/StemGNN/StemGNN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/StemGNN/StemGNN_PEMS04.py --gpus 0` | flow-only `[0]` |
| GTS | Yes | BasicTS v0.2 | `examples/baselines/GTS/GTS_PEMS04.py` | `python examples/run.py --cfg examples/baselines/GTS/GTS_PEMS04.py --gpus 0` | `GTSRunner` + `loss.py`; `dim_fc` derived from train `node_feats` |
| STNorm | Yes | BasicTS v0.2 | `examples/baselines/STNorm/STNorm_PEMS04.py` | `python examples/run.py --cfg examples/baselines/STNorm/STNorm_PEMS04.py --gpus 0` | `SimpleTimeSeriesForecastingRunner`; arch patch Conv1d→Conv2d |

Checkpoints save under `checkpoints/baselines/<Model>_PEMS04_<epochs>/` (or `_seed<N>` when using the batch script).

## Not ready / TODO

| Model | Status |
|-------|--------|
| STAEformer | Arch + official PeMS04 12→12 config |
| STWave | Arch + config |
| STDN | Arch + verified config |
| HimNet | Arch + config |
| DFDGCN | Arch + config |
| STPGNN | Arch + config |
| BigST | Arch + config |
| STEP | Arch + config |
| STGODE | Arch + config |
| MegaCRN | PeMS04 config (BasicTS only has METR-LA) |
| STD-MAE | Intentionally excluded for now |

See `reports/baseline_inventory.md` for full audit.

## Fairness note

Baseline configs follow official/BasicTS recommended hyperparameters when available. All methods share `datasets/PEMS04` under the **6:2:2** protocol. Baselines use official input features only (no channel 3). This provides a protocol-fair comparison under the same split and metrics, but not a strictly equal-compute comparison. Training epochs and runtime may differ across models and should be reported separately.

## Utilities

**Summary only** (parse existing logs, no training):

```bash
python scripts/run_baselines_pems04.py \
  --models STID D2STGNN AGCRN GWNet STGCN DCRNN DGCRN MTGNN StemGNN GTS STNorm \
  --gpus 0 1 \
  --seeds 5 \
  --summary_only \
  --out results/pems04_baselines.csv \
  --markdown results/pems04_baselines.md
```

**Normal batch run:**

```bash
python scripts/run_baselines_pems04.py \
  --models STID D2STGNN AGCRN GWNet STGCN DCRNN DGCRN MTGNN StemGNN GTS STNorm \
  --gpus 0 1 \
  --seeds 5 \
  --out results/pems04_baselines.csv \
  --markdown results/pems04_baselines.md
```

- `--seeds 5` means seeds **1, 2, 3, 4, 5** (not seed 5 only).
- `--seed-list 1 2 3 4 5` can be used for explicit seeds.

Outputs:

- `results/pems04_baselines.csv` — per-run rows (sorted by model, seed)
- `results/pems04_baselines_summary.csv` — mean / std / 95% CI per model
- `results/pems04_baselines.md` — markdown tables
- `results/baseline_logs/` — wrapper logs

**Single model:**

```bash
python examples/run.py --cfg examples/baselines/STID/STID_PEMS04.py --gpus 0
```

Use a **repo-relative** `--cfg` path. Absolute paths like `/home/.../STID_PEMS04.py` break EasyTorch import.
