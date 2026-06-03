# V3 Base Mapper Plus Prior Seed Ablation

## Per-run results

| variant | mapper | seed | MAE | RMSE | MAPE | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v3_base_mapper_plus_prior_kan | kan | 1 | 24.2326 | 38.0804 | 0.1684 | ok |
| v3_base_mapper_plus_prior_kan | kan | 2 | 26.5947 | 42.7825 | 0.1894 | ok |
| v3_base_mapper_plus_prior_kan | kan | 3 | 19.5732 | 31.4739 | 0.1412 | ok |
| v3_base_mapper_plus_prior_kan | kan | 4 | 19.5260 | 31.6695 | 0.1401 | ok |
| v3_base_mapper_plus_prior_kan | kan | 5 | 19.5716 | 31.5865 | 0.1389 | ok |
| v3_base_mapper_plus_prior_mlp | mlp | 1 | 19.2458 | 31.0670 | 0.1379 | ok |
| v3_base_mapper_plus_prior_mlp | mlp | 2 | 19.2680 | 31.1269 | 0.1373 | ok |
| v3_base_mapper_plus_prior_mlp | mlp | 3 | 19.2088 | 31.0323 | 0.1380 | ok |
| v3_base_mapper_plus_prior_mlp | mlp | 4 | 19.0667 | 30.8234 | 0.1341 | ok |
| v3_base_mapper_plus_prior_mlp | mlp | 5 | 19.1851 | 31.0058 | 0.1366 | ok |

## Summary

| mapper | n | MAE mean | MAE std | MAE 95% CI | RMSE mean | RMSE std | RMSE 95% CI | MAPE mean | MAPE std | MAPE 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kan | 5 | 21.8996 | 3.3148 | 4.1153 | 35.1186 | 5.1275 | 6.3656 | 0.1556 | 0.0225 | 0.0280 |
| mlp | 5 | 19.1949 | 0.0785 | 0.0975 | 31.0111 | 0.1143 | 0.1419 | 0.1368 | 0.0016 | 0.0020 |

## Paired difference: MLP - KAN

Positive diff means MLP is worse than KAN.

| metric | mean diff | std diff | 95% CI |
| --- | ---: | ---: | ---: |
| MAE | -2.7047 | 3.2582 | 4.0450 |
| RMSE | -4.1075 | 5.0481 | 6.2670 |
| MAPE | -0.0188 | 0.0219 | 0.0272 |
