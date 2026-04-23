# Credit Card Fraud Detection

This project studies fraud detection on the European cardholders dataset under extreme class imbalance.

It compares several supervised approaches, selects a high-recall candidate on a validation split, and then tunes the decision threshold to obtain a more realistic precision/recall trade-off on the final test set.

## Main Files

- `credit_card_fraud_detection.ipynb`
- `model_benchmark.py`

## Project Workflow

The final workflow includes:

1. data loading and imbalance audit
2. baseline comparison
3. standard logistic regression
4. cost-sensitive logistic regression
5. SMOTE and Borderline-SMOTE logistic regression
6. balanced random forest and gradient boosting comparison
7. validation-based threshold tuning
8. final test-set analysis with saved figures

## Main Result

The final selected model is a **SMOTE-based logistic regression with tuned threshold**.

On the saved test benchmark, it reaches approximately:

- `Precision = 0.732`
- `Recall = 0.811`
- `F1 = 0.769`
- `PR AUC = 0.794`
- `Specificity = 0.9995`


## Reproducibility

The benchmark script exports the project artifacts to `artifacts/`:

- validation metrics
- test metrics
- class distribution figure
- precision-recall comparison
- confusion matrices
- logistic coefficient plot

The notebook then presents those saved results in a narrative format.

## Dataset

The project expects the standard fraud dataset here:

- `projects/credit-card-fraud-detection/data/creditcard.csv`


