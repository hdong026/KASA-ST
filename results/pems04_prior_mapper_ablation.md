# Prior Mapper Seed Ablation

## Per-run results

| mapper | seed | MAE | RMSE | MAPE | status |
| --- | ---: | ---: | ---: | ---: | --- |
| kan | 1 | 18.1169 | 29.5294 | 0.1262 | ok |
| kan | 2 | 18.0805 | 29.5591 | 0.1253 | ok |
| kan | 3 | 18.0886 | 29.5302 | 0.1258 | ok |
| kan | 4 | 18.0954 | 29.5607 | 0.1259 | ok |
| kan | 5 | 18.1029 | 29.6354 | 0.1243 | ok |
| mlp | 1 | 18.0224 | 29.4310 | 0.1251 | ok |
| mlp | 2 | 18.0531 | 29.4749 | 0.1263 | ok |
| mlp | 3 | 18.0565 | 29.5245 | 0.1260 | ok |
| mlp | 4 | 18.1354 | 29.5950 | 0.1263 | ok |
| mlp | 5 | 18.0801 | 29.5803 | 0.1249 | ok |

## Summary

| mapper | n | MAE mean | MAE std | MAE 95% CI | RMSE mean | RMSE std | RMSE 95% CI | MAPE mean | MAPE std | MAPE 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kan | 5 | 18.0969 | 0.0139 | 0.0173 | 29.5630 | 0.0432 | 0.0536 | 0.1255 | 0.0007 | 0.0009 |
| mlp | 5 | 18.0695 | 0.0422 | 0.0523 | 29.5211 | 0.0693 | 0.0861 | 0.1257 | 0.0007 | 0.0008 |

## Paired difference: MLP - KAN

Positive diff means MLP is worse than KAN.

| metric | mean diff | std diff | 95% CI |
| --- | ---: | ---: | ---: |
| MAE | -0.0274 | 0.0477 | 0.0592 |
| RMSE | -0.0418 | 0.0554 | 0.0688 |
| MAPE | 0.0002 | 0.0008 | 0.0010 |
