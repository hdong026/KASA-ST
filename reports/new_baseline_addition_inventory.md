# New Baseline Addition Inventory (STAEformer / STWave / STDN / HimNet)

Branch: `feb9_best_rebuild`. Generated before any code copy.

## Source scan summary

| Repo | Branch/commit | Notes |
|------|---------------|-------|
| GestaltCogTeam/BasicTS `master` (v1.0+) | `c2bb6e3` | **No** STAEformer/STWave/STDN/HimNet in `src/basicts/models/`; models removed in v1.0 refactor |
| GestaltCogTeam/BasicTS v0.2 | `7a7f970` | **No** four models in `examples/` or `basicts/archs/arch_zoo/` |
| GestaltCogTeam/BasicTS pre-v1 | `eb65f4b` | **All four** present under `baselines/<Model>/` with PeMS04 12→12 configs |
| XDZhelheim/STAEformer | local `/home/dhz/STAEformer` | Official author repo; `model/STAEformer.yaml` PEMS04 section |
| roarer008/STDN | local `/home/dhz/STDN` | Official author repo; `conf/PEMSD4_1dim_12.conf` |
| LMissher/STWave | — | Not cloned locally; BasicTS `eb65f4b` used |
| XDZhelheim/HimNet | — | Not cloned locally; BasicTS `eb65f4b` used |

## Per-model inventory

| Model | BasicTS arch found? | BasicTS PeMS04 config found? | Official repo found? | Source chosen | Action | Notes |
|-------|:---------------------:|:----------------------------:|:--------------------:|---------------|--------|-------|
| STAEformer | Yes (not on master) | Yes — `baselines/STAEformer/PEMS04.py` @ `eb65f4b` | Yes — https://github.com/XDZhelheim/STAEformer | **BasicTS `eb65f4b`** arch + config | Copy arch + adapt config to KASA v0.2 runner | Early BasicTS copy (`2531b2a` `STAEformer_PEMS04.py`) had **PEMS08 hyperparams** (`num_nodes=170`); `eb65f4b` fixes to 307 + full `MODEL_PARAM`. Official yaml: `STAEformer/model/STAEformer.yaml` PEMS04 block |
| STWave | Yes (not on master) | Yes — `baselines/STWave/PEMS04.py` @ `eb65f4b` | Yes — https://github.com/LMissher/STWave | **BasicTS `eb65f4b`** arch + config + `loss.py` | Copy arch, loss, config | `stwave_masked_mae` loss; precompute `adj_gat` + `graphwave` in config; `[0,1,2]` features |
| STDN | Yes (not on master) | Yes — `baselines/STDN/PEMS04.py` @ `eb65f4b` | Yes — https://github.com/roarer008/STDN | **BasicTS `eb65f4b`** arch + runner + config | Copy arch (`model.py`, `utils.py`), `STDNRunner`, config | Custom runner; `get_lpls(adj)`; 300 epochs, StepLR γ=0.9 step=10, batch 64; author conf `PEMSD4_1dim_12.conf` epochs=200 — **BasicTS uses 300** |
| HimNet | Yes (not on master) | Yes — `baselines/HimNet/PEMS04.py` @ `eb65f4b` | Yes — https://github.com/XDZhelheim/HimNet | **BasicTS `eb65f4b`** arch + runner + config | Copy arch, `HimNetRunner`, config | `masked_huber` loss; 200 epochs; `HimNetRunner` teacher forcing |

## Exact source paths (BasicTS commit `eb65f4b`)

### STAEformer
- Arch: `baselines/STAEformer/arch/staeformer_arch.py`
- Config: `baselines/STAEformer/PEMS04.py`
- URL: https://github.com/GestaltCogTeam/BasicTS/blob/eb65f4b/baselines/STAEformer/PEMS04.py

### STWave
- Arch: `baselines/STWave/arch/stwave_arch.py`
- Loss: `baselines/STWave/loss.py`
- Config: `baselines/STWave/PEMS04.py`
- URL: https://github.com/GestaltCogTeam/BasicTS/blob/eb65f4b/baselines/STWave/PEMS04.py

### STDN
- Arch: `baselines/STDN/arch/model.py`, `baselines/STDN/arch/utils.py`
- Runner: `baselines/STDN/runner/stdn_runner.py`
- Config: `baselines/STDN/PEMS04.py`
- URL: https://github.com/GestaltCogTeam/BasicTS/blob/eb65f4b/baselines/STDN/PEMS04.py

### HimNet
- Arch: `baselines/HimNet/arch/model/HimNet.py`
- Runner: `baselines/HimNet/runner/himnet_runner.py`
- Config: `baselines/HimNet/PEMS04.py`
- URL: https://github.com/GestaltCogTeam/BasicTS/blob/eb65f4b/baselines/HimNet/PEMS04.py

## BasicTS master / v0.2 status

- **BasicTS `master` (v1.0+):** README lists all four models with author links, but implementation **not** in current tree.
- **BasicTS `7a7f970` (v0.2, used by existing KASA baselines):** None of the four models present.

## Compatibility plan (minimal)

1. Place arch under `basicts/archs/arch_zoo/{staeformer,stwave,stdn,himnet}_arch/`.
2. Adapt PeMS04 configs to KASA-ST v0.2 `CFG` layout (`DATASET_CLS`, `TRAIN.DATA.DIR`, etc.) — preserve hyperparameters from `eb65f4b`.
3. Port `STDNRunner` / `HimNetRunner` to v0.2 tuple `forward` API (like `DCRNNRunner`).
4. Add `stwave_masked_mae` and `masked_huber` to `basicts/losses` / `basicts/metrics`.
5. Checkpoint dirs: `checkpoints/baselines/<Model>_PEMS04`.
6. Never use `FORWARD_FEATURES` containing channel 3.

## Not used (unverified)

- LSTNN local `STDN_Adapter` / checkpoint config — third-party adaptation, not official BasicTS.
