# Time Series Forecasting of the CAC 40

This project presents a financial time-series forecasting study based on the daily CAC 40 index.

The notebook was prepared as a complete course submission and includes:

- data collection and cleaning
- exploratory analysis
- decomposition and stationarity checks
- a baseline forecast
- a SARIMAX model
- three machine learning regressors
- an LSTM model
- final comparison with `RMSE`, `MAE`, and `MAPE`

## Main Files

- `cac40_time_series_forecasting.ipynb`
- `requirements.txt`

## Data

The `data` folder contains the cached CAC 40 series and the exported prediction tables used in the notebook.

## Reproducibility

The notebook can reuse the cached CSV file for reproducibility, or download the series again with `yfinance` if needed.

