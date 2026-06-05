# Failed PeMS04 Baseline Error Report

Branch: `feb9_best_rebuild`  
Date: 2026-06-05

## Summary

Recent batch logs (`results/baseline_logs/*_error_tail.txt`) show **SyntaxError** from `scripts/run_baselines_pems04.py` `make_seed_config()` when `--seeds > 1` (corrupts `CKPT_SAVE_DIR` line). That is a runner-script issue, not a model issue.

Reproduced **real model/runtime errors** below via `runner.forward()` on one train batch (seed-1 configs, no temp files).

---

## DCRNN

| Field | Value |
|-------|-------|
| Config | `examples/baselines/DCRNN/DCRNN_PEMS04.py` |
| Runner (local) | `DCRNNRunner` |
| Model | `basicts.archs.DCRNN` |
| Log error (multi-seed) | `SyntaxError: unmatched ')'` in temp seed config |
| Runtime error | `RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cuda:0, different from other tensors on cpu` |
| Traceback tail | `dcrnn_cell.py` `_gconv` → `torch.mm(support, x0)` with CPU `adj_mx` |
| Likely cause | Adjacency matrices passed as CPU tensors in `CFG.MODEL.PARAM`; `_gconv` does not move supports to GPU (official BasicTS uses `support.to(x0.device)`) |
| Proposed fix | Patch `dcrnn_cell._gconv` to `.to(device)` supports; align config runner with official `SimpleTimeSeriesForecastingRunner` + `TRAIN.SETUP_GRAPH` |

---

## DGCRN

| Field | Value |
|-------|-------|
| Config | `examples/baselines/DGCRN/DGCRN_PEMS04.py` |
| Runner (local) | `DGCRNRunner` (custom, expects 3-tuple batch) |
| Model | `basicts.archs.DGCRN` |
| Log error (multi-seed) | `SyntaxError: unmatched ')'` in temp seed config |
| Runtime error | `ValueError: not enough values to unpack (expected 3, got 2)` |
| Traceback tail | `dgcrn_runner.py:64` `future_data, history_data, _ = data` |
| Likely cause | Local `DGCRNRunner` diverges from official BasicTS; official runner extends `SimpleTimeSeriesForecastingRunner` and unpacks 2-tuple `(future, history)` |
| Proposed fix | Replace `dgcrn_runner.py` with official BasicTS implementation |

---

## GWNet

| Field | Value |
|-------|-------|
| Config | `examples/baselines/GWNet/GWNet_PEMS04.py` |
| Runner (local) | `SimpleTimeSeriesForecastingRunner` (matches official) |
| Model | `basicts.archs.GraphWaveNet` |
| Log error (multi-seed) | `SyntaxError: unmatched ')'` in temp seed config |
| Runtime error | `RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [2, 32, 307, 13]` |
| Traceback tail | `gwnet_arch.py:200` `gate = self.gate_convs[i](residual)` |
| Likely cause | `gate_convs` / `residual_convs` / `skip_convs` use `nn.Conv1d` on 4D tensors `[B, C, N, L]`; current PyTorch requires `Conv2d` for this layout |
| Proposed fix | Minimal arch patch: `Conv1d` → `Conv2d` with `kernel_size=(1, k)` for gate/residual/skip layers |

---

## GTS

| Field | Value |
|-------|-------|
| Config | `examples/baselines/GTS/GTS_PEMS04.py` |
| Runner (local) | `GTSRunner` (matches official) |
| Model | `basicts.archs.GTS` |
| Log error (multi-seed) | `SyntaxError: unmatched ')'` in temp seed config |
| Runtime error | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (307x190128 and 162976x100)` |
| Traceback tail | `gts_arch.py:270` `x = self.fc(x)` |
| Likely cause | Hard-coded `dim_fc=162976` from official PeMS04 config does not match local `node_feats` time length (local → 190128) |
| Proposed fix | Derive `dim_fc` from train `node_feats` via GTS conv stack (same formula as arch, not hyperparameter tuning); add `CFG.MODEL.SETUP_GRAPH = True` per official |

---

## STNorm

| Field | Value |
|-------|-------|
| Config | `examples/baselines/STNorm/STNorm_PEMS04.py` |
| Runner (local) | `SimpleTimeSeriesForecastingRunner` (matches official) |
| Model | `basicts.archs.STNorm` |
| Log error (multi-seed) | `SyntaxError: unmatched ')'` in temp seed config |
| Runtime error | `RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [2, 96, 307, 13]` |
| Traceback tail | `stnorm_arch.py:168` `gate = self.gate_convs[i](x)` |
| Likely cause | Same as GWNet: `Conv1d` on 4D spatiotemporal tensors |
| Proposed fix | Minimal arch patch: `Conv1d` → `Conv2d` for gate/residual/skip layers |

---

## Runner-script note (out of scope for config fixes)

`make_seed_config()` regex breaks lines like:

```python
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "baselines", "DCRNN_PEMS04_" + str(CFG.TRAIN.NUM_EPOCHS))
```

Use `--seeds 1` for sanity until runner regex is fixed.

---

## Applied fixes (2026-06-05)

| Model | Files changed |
|-------|---------------|
| DCRNN | `DCRNN_PEMS04.py` (runner→Simple + SETUP_GRAPH), `dcrnn_cell.py`, `dcrnn_arch.py` |
| DGCRN | `dgcrn_runner.py` (official BasicTS) |
| GWNet | `gwnet_arch.py` (Conv2d) |
| GTS | `GTS_PEMS04.py` (`dim_fc` inference) |
| STNorm | `stnorm_arch.py` (Conv2d) |

Smoke test (`scripts/smoke_test_failed_baselines.py`): all five OK.
Sanity training (`--seeds 1`): DCRNN/DGCRN confirmed training past epoch 1 without `exit_1` (GWNet/GTS/STNorm queued on 2 GPUs).
