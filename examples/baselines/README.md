# PeMS04 12→12 Predictor Baselines

Verified configs copied from **GestaltCogTeam/BasicTS** v0.2 (`7a7f970`): `examples/<Model>/<Model>_PEMS04.py`.

Entry point (unchanged):

```bash
python examples/run.py --cfg examples/baselines/<Model>/<Model>_PEMS04.py --gpus 0
```

Use the `basicts` conda env (or any env with `easytorch` + project deps).

## Ready baselines

| Model | Ready? | Source | Config | Command | Notes |
|-------|:------:|--------|--------|---------|-------|
| STID | Yes | BasicTS v0.2 `STID_PEMS04.py` | `examples/baselines/STID/STID_PEMS04.py` | `python examples/run.py --cfg examples/baselines/STID/STID_PEMS04.py --gpus 0` | `[0,1,2]` → target `[0]` |
| D2STGNN | Yes | BasicTS v0.2 | `examples/baselines/D2STGNN/D2STGNN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/D2STGNN/D2STGNN_PEMS04.py --gpus 0` | Requires `datasets/PEMS04/adj_mx.pkl` |
| AGCRN | Yes | BasicTS v0.2 (+ AGCRN official PEMSD4 conf) | `examples/baselines/AGCRN/AGCRN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/AGCRN/AGCRN_PEMS04.py --gpus 0` | Flow-only `[0]` |
| GWNet | Yes | BasicTS v0.2 | `examples/baselines/GWNet/GWNet_PEMS04.py` | `python examples/run.py --cfg examples/baselines/GWNet/GWNet_PEMS04.py --gpus 0` | Class `GraphWaveNet` |
| STGCN | Yes | BasicTS v0.2 | `examples/baselines/STGCN/STGCN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/STGCN/STGCN_PEMS04.py --gpus 0` | Flow-only `[0]` |
| DCRNN | Yes | BasicTS v0.2 | `examples/baselines/DCRNN/DCRNN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/DCRNN/DCRNN_PEMS04.py --gpus 0` | `DCRNNRunner` |
| DGCRN | Yes | BasicTS v0.2 | `examples/baselines/DGCRN/DGCRN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/DGCRN/DGCRN_PEMS04.py --gpus 0` | `DGCRNRunner` |
| MTGNN | Yes | BasicTS v0.2 | `examples/baselines/MTGNN/MTGNN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/MTGNN/MTGNN_PEMS04.py --gpus 0` | `MTGNNRunner` |
| StemGNN | Yes | BasicTS v0.2 | `examples/baselines/StemGNN/StemGNN_PEMS04.py` | `python examples/run.py --cfg examples/baselines/StemGNN/StemGNN_PEMS04.py --gpus 0` | Flow-only `[0]` |
| GTS | Yes | BasicTS v0.2 | `examples/baselines/GTS/GTS_PEMS04.py` | `python examples/run.py --cfg examples/baselines/GTS/GTS_PEMS04.py --gpus 0` | `GTSRunner` + local `loss.py` |
| STNorm | Yes | BasicTS v0.2 | `examples/baselines/STNorm/STNorm_PEMS04.py` | `python examples/run.py --cfg examples/baselines/STNorm/STNorm_PEMS04.py --gpus 0` | `[0,1]` |

Checkpoints save under `checkpoints/baselines/<Model>_PEMS04_<epochs>/`.

## Not ready (TODO)

| Model | Ready? | Missing |
|-------|:------:|---------|
| STAEformer | No | Arch + official PeMS04 12→12 config |
| STWave | No | Arch + config |
| STDN | No | Arch + verified config (do not use ad-hoc LSTNN configs for papers) |
| HimNet | No | Arch + config |
| DFDGCN | No | Arch + config |
| STPGNN | No | Arch + config |
| BigST | No | Arch + config |
| STEP | No | Arch + config (pretraining-style) |
| STGODE | No | Arch + config |
| MegaCRN | No | PeMS04 config (BasicTS only ships METR-LA) |
| STD-MAE | No | Intentionally excluded from this suite |

See `reports/baseline_inventory.md` for full audit.

## Utilities

**Smoke test (no training):**

```bash
python scripts/smoke_test_baselines.py
```

**Batch run ready models:**

```bash
python scripts/run_baselines_pems04.py \
  --models STID D2STGNN AGCRN GWNet STGCN DCRNN DGCRN MTGNN StemGNN GTS STNorm \
  --gpus 0 1 \
  --seeds 1 \
  --out results/pems04_baselines.csv \
  --markdown results/pems04_baselines.md
```

**Multiple seeds**

- `--seeds 5` → 跑 seed **1, 2, 3, 4, 5**（不是只跑 seed 5）
- 或显式指定：`--seed-list 1 2 3 4 5`
- 多种子时 checkpoint 会写到 `checkpoints/baselines/<Model>_PEMS04_<epochs>_seed<N>/`

示例（STID 跑 5 个 seed，单卡）：

```bash
python scripts/run_baselines_pems04.py \
  --models STID \
  --gpus 0 \
  --seeds 5
```

示例（指定 seed 列表）：

```bash
python scripts/run_baselines_pems04.py \
  --models STID AGCRN \
  --gpus 0 \
  --seed-list 1 3 5 7 9
```

注意：`--cfg` 必须是**相对仓库根目录**的路径（脚本已自动处理）；不要写成 `/home/.../STID_PEMS04.py`，否则 EasyTorch 会报 `No module named '.home'`。

`--gpus 0 1` 时脚本会把第 1 个任务交给 EasyTorch `--gpus 0`、第 2 个交给 `--gpus 1`（由 EasyTorch 设置 `CUDA_VISIBLE_DEVICES`）。日志里若看到 `Use devices 0`，表示**该进程可见 GPU 列表中的第 0 号**，在 `--gpus 1` 的进程里对应**物理 GPU 1**。

Logs: `results/baseline_logs/`. Unverified configs are **not** included by default.
