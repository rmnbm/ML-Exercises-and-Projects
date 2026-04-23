# Machine Learning Projects

This repository gathers several semester machine learning projects and a set of solved practical exercises.

## Repository Structure

```text
machine-learning-projects/
  projects/
    airbnb-price-prediction/
    credit-card-fraud-detection/
    time-series-forecasting-cac40/
  exercises/
```

## Projects

### 1. Airbnb Price Prediction

This project predicts Airbnb listing prices with a tabular regression pipeline based on feature engineering, model comparison, and hyperparameter tuning.

Main topics:

- feature engineering from listing metadata
- exploratory analysis
- regression model comparison
- CatBoost and XGBoost tuning
- prediction export

Folder: `projects/airbnb-price-prediction`

### 2. Credit Card Fraud Detection

This project studies fraud detection as a highly imbalanced classification problem. The notebook follows an incremental strategy, starting from simple baselines and progressively moving toward more suitable approaches for rare-event detection.

Main topics:

- baseline classification
- cost-sensitive learning
- imbalance handling with oversampling
- ensemble methods
- model evaluation under class imbalance
- threshold tuning for operational use

Folder: `projects/credit-card-fraud-detection`


### 3. Time Series Forecasting of the CAC 40

This project focuses on financial time-series forecasting using the daily CAC 40 index. It combines exploratory analysis, a statistical forecasting model, several machine learning regressors, and an LSTM model.

Main topics:

- exploratory time-series analysis
- stationarity checks
- SARIMAX forecasting
- regression models
- LSTM forecasting
- model comparison with error metrics

Folder: `projects/time-series-forecasting-cac40`

## Exercises

The `exercises` folder contains fully commented solution notebooks for the practical sessions:

- dimensionality reduction
- cancer diagnosis under class imbalance
- anomaly detection
- credit card customer clustering
- financial time-series forecasting

Folder: `exercises`

## Data Policy

Large raw datasets are not versioned in full as they would unnecessarily bloat the repository. A note is provided in the relevant folder to explain where the data should be placed locally.

## Recommended Workflow

1. Create a virtual environment.
2. Install the dependencies required by the project you want to run.
3. Open the corresponding notebook.
4. Run the notebook from its own folder so local relative paths work correctly.

