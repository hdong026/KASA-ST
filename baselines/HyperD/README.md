# HyperD Baseline (KASA-ST integration)

Official reference: [ll121202/HyperD](https://github.com/ll121202/HyperD)

This directory integrates HyperD as an **external baseline only**. KASA / ChainForecasting code is untouched.

## Setting

- Datasets: PEMS03 / PEMS04 / PEMS07 / PEMS08
- Horizon: **12 -> 12** (official)
- Split: **6:2:2 on time steps** (HyperD `data.dat` protocol)
- Loss: MAE + official **DVA (dual-view) loss**
- Initialization: `daily_init.npy`, `weekly_init.npy`

## Data

HyperD uses BasicTS-style files under `datasets/{NAME}/`:

- `desc.json` (auto-generated from repo metadata)
- `data.dat` (auto-generated memmap from `datasets/raw_data/{NAME}/{NAME}.npz`, not duplicated in git)
- `adj_mx.pkl` (reuses existing KASA-ST graph)
- `daily_init.npy`, `weekly_init.npy` (from `Initialization.py`)

PEMS03 requires raw data under `datasets/raw_data/PEMS03/` (not bundled in this repo).

## Commands

```bash
# 1) Statistical prior initialization
python baselines/HyperD/Initialization.py -d PEMS04

# 2) Official-style training entry
python baselines/HyperD/train.py -c baselines/HyperD/PEMS04.py -g 0

# 3) Unified launcher (init check + train + result table)
python scripts/run_hyperd_baseline.py --datasets PEMS04 PEMS07 PEMS08 --gpus 0

# Dry run (print commands only)
python scripts/run_hyperd_baseline.py --datasets PEMS04 --gpus 0 --dry_run
```

Results are written to `results/baselines/hyperd/`.

## Notes

- Do **not** mix official 12->12 numbers with adapted 16->32 runs.
- `examples/run.py` also works: `python examples/run.py -c baselines/HyperD/PEMS04.py --gpus 0`
