# Airbnb Price Prediction

This project predicts Airbnb listing `log_price` values from structured listing features such as room type, city, accommodation capacity, review information, and host attributes.

The notebook follows a full tabular regression workflow:
- data loading from local course files,
- feature cleaning and engineering,
- exploratory data analysis,
- validation split and model benchmark,
- hyperparameter tuning,
- feature importance analysis,
- final prediction export.

## Project Files

- `airbnb_price_prediction.ipynb`: cleaned English notebook prepared as the main project deliverable
- `archive/coursework_original_airbnb_price_prediction.ipynb`: preserved copy of the original coursework notebook
- `artifacts/airbnb_predictions.csv`: saved prediction file produced during the original project work
- `requirements.txt`: project dependencies

## Expected Local Data

The original data CSV files are not included in the repository. To run the notebook locally, place the following files in `data/`:

- `airbnb_train.csv`
- `airbnb_test.csv`
- `prediction_example.csv`

## Modeling Summary

Five regression models are compared:
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost

The notebook reports that the tuned CatBoost model is the strongest final candidate, with:
- validation `R-squared` of about `0.6554`
- validation `MAPE` of about `31.78%`

These results are coherent with the nature of the dataset, because CatBoost handles mixed numerical and categorical features particularly well.
