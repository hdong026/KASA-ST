# Baseline Syntax Check

Date: 2026-06-05  
Branch: `feb9_best_rebuild`

## Command

```bash
python -m py_compile scripts/run_baselines_pems04.py
python -m py_compile examples/baselines/<Model>/<Model>_PEMS04.py
```

## Results

| File | Status |
|------|--------|
| `scripts/run_baselines_pems04.py` | OK |
| `examples/baselines/STID/STID_PEMS04.py` | OK |
| `examples/baselines/D2STGNN/D2STGNN_PEMS04.py` | OK |
| `examples/baselines/AGCRN/AGCRN_PEMS04.py` | OK |
| `examples/baselines/STGCN/STGCN_PEMS04.py` | OK |
| `examples/baselines/MTGNN/MTGNN_PEMS04.py` | OK |
| `examples/baselines/StemGNN/StemGNN_PEMS04.py` | OK |
| `examples/baselines/GWNet/GWNet_PEMS04.py` | OK |
| `examples/baselines/DCRNN/DCRNN_PEMS04.py` | OK |
| `examples/baselines/DGCRN/DGCRN_PEMS04.py` | OK |
| `examples/baselines/GTS/GTS_PEMS04.py` | OK |
| `examples/baselines/STNorm/STNorm_PEMS04.py` | OK |

No formatting or syntax fixes were required in config files for this pass.
