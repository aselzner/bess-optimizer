Backtest 2025-01-01 to 2025-12-31, 10 MW / 40 MWh, RTE 85%, SoC 10%-90%, degradation $3/MWh. Revenue is net of degradation cost.

| Zone | Strategy | $/kW-year | Capture ratio | Cycles/day | MAE ($/MWh) | RMSE ($/MWh) |
|---|---|---:|---:|---:|---:|---:|
| WEST | perfect | 29.7 | 100.0% | 0.90 |  |  |
| WEST | naive_d1 | 22.9 | 77.1% | 0.90 | 9.66 | 16.69 |
| WEST | naive_d7 | 22.4 | 75.3% | 0.90 | 18.81 | 33.06 |
| WEST | regression | 23.2 | 78.1% | 0.94 | 9.69 | 15.88 |
| CAPITL | perfect | 50.1 | 100.0% | 1.00 |  |  |
| CAPITL | naive_d1 | 40.1 | 80.0% | 1.00 | 13.33 | 27.79 |
| CAPITL | naive_d7 | 39.1 | 78.0% | 1.00 | 26.49 | 50.45 |
| CAPITL | regression | 38.9 | 77.6% | 1.06 | 14.62 | 28.36 |
| N.Y.C. | perfect | 42.4 | 100.0% | 0.94 |  |  |
| N.Y.C. | naive_d1 | 35.5 | 83.6% | 0.94 | 12.19 | 23.47 |
| N.Y.C. | naive_d7 | 34.6 | 81.6% | 0.94 | 25.06 | 46.76 |
| N.Y.C. | regression | 34.8 | 81.9% | 0.93 | 12.97 | 22.95 |
| LONGIL | perfect | 50.0 | 100.0% | 0.96 |  |  |
| LONGIL | naive_d1 | 42.6 | 85.2% | 0.96 | 13.46 | 26.06 |
| LONGIL | naive_d7 | 41.7 | 83.5% | 0.96 | 26.05 | 47.02 |
| LONGIL | regression | 42.6 | 85.2% | 0.93 | 14.42 | 26.12 |
