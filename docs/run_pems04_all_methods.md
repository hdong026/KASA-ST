# PeMS04 All-Methods Runner

Unified orchestration for **KASA-ST** and ready **PeMS04 12→12 baselines** in one command.

Script: `scripts/run_pems04_all_methods.py`

## Purpose

- Run the final KASA-ST setting and multiple baselines with the same seeds/GPUs
- Generate per-seed temp configs (original configs untouched)
- Parse test metrics from wrapper + checkpoint logs
- Write CSV, summary CSV, and Markdown tables (mean / std / 95% CI)

## Final KASA-ST setting (temp config overrides)

```python
CFG.MODEL.PARAM["prior_mapper_type"] = "mlp"
CFG.MODEL.PARAM["use_pre_temporal_spatial_enhancement"] = False
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]
CFG.MODEL.TARGET_FEATURES = [0]
```

Base config: `examples/KASAST_v2/KASAST_PEMS04.py`

## Default baseline list

`STID D2STGNN AGCRN STGCN MTGNN StemGNN`

Configs: `examples/baselines/<Model>/<Model>_PEMS04.py`

Optional (pass explicitly if fixed): `DCRNN DGCRN GWNet GTS STNorm`

Paper-ready strong baselines (after smoke test): `STAEformer STWave STDN HimNet`

Extended example (KASA + all ready baselines):

```bash
python scripts/run_pems04_all_methods.py \
  --methods KASA D2STGNN STID STAEformer STWave STDN HimNet AGCRN STGCN MTGNN StemGNN \
  --seeds 1 2 3 4 5 \
  --gpus 0 1 \
  --out results/pems04_all_methods_extended.csv \
  --markdown results/pems04_all_methods_extended.md
```

New baselines one-seed sanity:

```bash
python scripts/run_pems04_all_methods.py \
  --methods STAEformer STWave STDN HimNet \
  --seeds 1 \
  --gpus 0 1 \
  --out results/pems04_new_baselines_sanity.csv \
  --markdown results/pems04_new_baselines_sanity.md
```

Baselines are **not** modified beyond seed + deterministic checkpoint dir. They do **not** use channel 3.

## Paths

| Item | Location |
|------|----------|
| Temp configs | `tmp_configs/pems04_all_methods/<Method>_seed<N>.py` |
| Checkpoints | `checkpoints/pems04_all_methods/<Method>_seed<N>/` |
| Wrapper logs | `results/pems04_all_method_logs/<Method>_seed<N>_gpu<G>.log` |
| Error tails | `results/pems04_all_method_logs/<Method>_seed<N>_error_tail.txt` |
| Results CSV | `results/pems04_all_methods.csv` |
| Summary CSV | `results/pems04_all_methods_summary.csv` |
| Markdown | `results/pems04_all_methods.md` |

## Commands

### Dry run

```bash
python scripts/run_pems04_all_methods.py \
  --methods KASA STID D2STGNN AGCRN STGCN MTGNN StemGNN \
  --seeds 1 2 3 4 5 \
  --gpus 0 1 \
  --out results/pems04_all_methods.csv \
  --markdown results/pems04_all_methods.md \
  --dry_run
```

### Normal run

```bash
python scripts/run_pems04_all_methods.py \
  --methods KASA STID D2STGNN AGCRN STGCN MTGNN StemGNN \
  --seeds 1 2 3 4 5 \
  --gpus 0 1 \
  --out results/pems04_all_methods.csv \
  --markdown results/pems04_all_methods.md
```

### Summary only (no training)

```bash
python scripts/run_pems04_all_methods.py \
  --methods KASA STID D2STGNN AGCRN STGCN MTGNN StemGNN \
  --seeds 1 2 3 4 5 \
  --gpus 0 1 \
  --summary_only \
  --out results/pems04_all_methods.csv \
  --markdown results/pems04_all_methods.md
```

### Tmux background run

```bash
tmux new -s pems04_all -d \
  'cd /path/to/KASA-ST && source ~/miniconda3/etc/profile.d/conda.sh && conda activate basicts && \
   python scripts/run_pems04_all_methods.py \
     --methods KASA STID D2STGNN AGCRN STGCN MTGNN StemGNN \
     --seeds 1 2 3 4 5 --gpus 0 1 \
     --out results/pems04_all_methods.csv \
     --markdown results/pems04_all_methods.md \
   2>&1 | tee results/pems04_all_methods_run.log'
```

## Resume

The script does not implement manual checkpoint loading. Re-running the same command uses EasyTorch native resume from `checkpoints/pems04_all_methods/<Method>_seed<N>/` when checkpoints exist.

## Fairness note

- All methods share the same PeMS04 data, split, and metrics under `datasets/PEMS04` (official **6:2:2**, 12→12).
- KASA uses the train-only prior channel (ch3) as a proposed module.
- Baselines use official/BasicTS input features only and **do not** use channel 3.
- Baseline hyperparameters (optimizer, scheduler, epochs, batch size) are preserved from their configs.
- This is protocol-fair on data/split/metrics, not equal-compute across models.
