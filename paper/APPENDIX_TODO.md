# Appendix TODO (author-facing; not for reviewers)

## Blocker: missing `supplementary.tex`

As of 2026-08-01, `supplementary.tex` is **not present** on this machine under `/home/dhz` (same situation as the earlier Methodology rewrite). Provide the absolute path to the full supplementary source (or copy it into `KASA-ST/paper/supplementary.tex`) so Appendix D can be deleted in-place and subsequent appendices renumbered via `\appendix` / `\section` (no hard-coded letters).

A drop-in revised sensitivity appendix draft lives at:

- `paper/sections/sensitivity_and_efficiency.tex`
- Compilable stub: `paper/supplementary.tex` (contains only the revised sensitivity appendix until the full file is available)

## Reminders moved out of deleted Appendix D (do not put in PDF)

- Do **not** describe unexecuted patch-length / phase-stride / adapter / node-count scaling ablations to reviewers.
- Do **not** use old variants `chain_interleaved_progressive_spatial_light` / `_strong` as final sensitivity controls (they lack condition-only adapter + token-normalized loss).
- Stage-schedule efficiency belongs in the **main manuscript** table (`tab:efficiency`); the appendix only discusses it and points to that table.
- Fill spatial-sensitivity numerical cells only from verified PEMS04 H=12 seed-1 logs/checkpoints + `results/paper_efficiency.csv` (or a fresh profile of light/strong). Until then, keep numbers here (or as TeX `%` comments), not as visible missing boxes.

## Spatial sensitivity runs (PEMS04, H=12, seed 1)

Command:

```bash
python scripts/run_chain_forecasting_horizon.py \
  --dataset PEMS04 \
  --horizons 12 \
  --variants \
  chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss_light \
  chain_interleaved_progressive_spatial_state_adapter_fixed_token_loss_strong \
  --seeds 1 \
  --gpus 0 1
```

Status (updated 2026-07-31):

| Setting | Variant | ratios | top-k | alphas | MAE (seed 1) | Params (M) | Peak Mem. (MiB) | Infer. (ms/batch) |
|---------|---------|--------|-------|--------|--------------|------------|-----------------|-------------------|
| Light | `..._fixed_token_loss_light` | [0.25, 0.5, 1.0] | [4, 8, 16] | [0.02, 0.04, 0.08] | **training** (retry after OOM; healthy as of epoch 4) | 2.212 (2212002) | pending profile | pending profile |
| Default | `..._fixed_token_loss` | [0.25, 0.5, 1.0] | [8, 16, 32] | [0.03, 0.06, 0.10] | **missing exact PEMS04 seed-1 MAE on this machine** | 2.212 (2212002) | 2596.5 | 14.413 |
| Strong | `..._fixed_token_loss_strong` | [0.5, 0.75, 1.0] | [16, 24, 40] | [0.05, 0.08, 0.12] | **training** (healthy; ETA ~16:23–16:55 local) | 2.223 (2222594) | pending profile | pending profile |

Note: Light/Default share identical trainable parameter counts (topk/alpha are not extra weights); Strong is slightly larger (2.223M) under the official model constructor with the Strong spatial schedule. Peak memory / inference for Light/Strong will be measured with `scripts/profile_paper_efficiency.py` after GPUs are free (same protocol as Default).

Logs:

- Light (failed OOM): `results/fixed_input_horizon_pems04_logs/h12_..._light_seed1_gpu0.log`
- Light retry runner: `results/pems04_spatial_sensitivity_light_seed1_runner.log`
- Strong: `results/fixed_input_horizon_pems04_logs/h12_..._strong_seed1_gpu1.log`
- Combined orchestrator: `results/pems04_spatial_sensitivity_light_strong_seed1_runner.log`

Default efficiency numbers above are from `results/paper_efficiency.csv` (F2FNet row), `memory_measurement_mode=training_step`, `latency_measurement_mode=model_forward_with_h2d`. Re-profile Light/Strong with the same script after checkpoints exist (or with initialized weights for compute-only cells).

## Efficiency protocol (verified; for appendix prose)

From `scripts/profile_paper_efficiency.py` / `results/paper_efficiency_environment.txt`:

- Device: single NVIDIA GeForce RTX 4090 (GPU 0 for a given worker subprocess)
- Batch size: 32
- Precision: fp32
- Warm-up: 20 train steps (not timed); inference warm-up `max(20, warmup_steps)`
- Timed: up to 40 train steps (for peak memory) and 100 inference steps
- Synchronization: `torch.cuda.synchronize()` before/after each timed step
- Peak memory: `torch.cuda.reset_peak_memory_stats()` after train warm-up; report `torch.cuda.max_memory_allocated()` over timed train steps
- Latency: median ms/batch; timed region includes H2D + model forward (not checkpoint load / metric aggregation / saving)
- Each model runs in an isolated subprocess with `gc.collect()`, `empty_cache()`, peak reset, synchronize before measurement
