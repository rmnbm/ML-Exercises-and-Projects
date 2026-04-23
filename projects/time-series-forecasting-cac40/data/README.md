# Data Files

This folder contains the cached data and exported result tables used by the notebook.

Main files:

- `cac40_daily.csv`: cached daily CAC 40 market data used to avoid downloading the same series repeatedly
- `model_comparison.csv`: final comparison table of the forecasting models
- `test_predictions.csv`: test-period predictions exported by the notebook

The notebook can reuse the cached CSV for reproducibility, or download the CAC 40 series again from `yfinance` if needed.
