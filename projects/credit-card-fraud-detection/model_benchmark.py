from __future__ import annotations

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"
OUTPUT_DIR = PROJECT_DIR / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "scaled",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(range(30)),
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def metric_bundle(y_true, y_pred, y_score) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    raise AttributeError("Model does not expose probabilities or a decision function.")


def save_class_distribution(y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=["Legitimate", "Fraud"],
        y=[int(counts.get(0, 0)), int(counts.get(1, 0))],
        ax=ax,
    )
    ax.set_title("Class Distribution in the Full Dataset")
    ax.set_ylabel("Number of Transactions")
    for idx, val in enumerate([int(counts.get(0, 0)), int(counts.get(1, 0))]):
        ax.text(idx, val, f"{val:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)


def save_precision_recall_plot(curves: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, (precision, recall, ap_score) in curves.items():
        ax.plot(recall, precision, linewidth=2, label=f"{label} (AP={ap_score:.3f})")
    ax.set_title("Validation Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "precision_recall_curves.png", dpi=180)
    plt.close(fig)


def save_confusion_matrices(y_true, default_pred, tuned_pred) -> None:
    cms = [
        ("Best Model - Default Threshold", confusion_matrix(y_true, default_pred)),
        ("Best Model - Tuned Threshold", confusion_matrix(y_true, tuned_pred)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, cm) in zip(axes, cms):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def save_logistic_coefficients(model, feature_names: list[str], title: str) -> None:
    coefs = pd.Series(model.named_steps["clf"].coef_[0], index=feature_names)
    top = (
        pd.concat([coefs.nlargest(8), coefs.nsmallest(8)])
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    top.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "logistic_coefficients.png", dpi=180)
    plt.close(fig)


def evaluate_model(name: str, model, X_train, y_train, X_valid, y_valid):
    start = time.time()
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    valid_scores = predict_scores(fitted, X_valid)
    valid_pred = (valid_scores >= 0.5).astype(int)
    metrics = metric_bundle(y_valid, valid_pred, valid_scores)
    metrics["model"] = name
    metrics["fit_seconds"] = round(time.time() - start, 2)
    return fitted, metrics, valid_scores


def tune_threshold(y_true, y_score) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    beta = 2.0
    scores = (1 + beta**2) * precision[:-1] * recall[:-1] / (
        beta**2 * precision[:-1] + recall[:-1] + 1e-12
    )
    best_idx = int(np.nanargmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def main():
    df = load_data()
    save_class_distribution(df["Class"])
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)

    preprocess = make_preprocessor()
    logistic = Pipeline(
        [
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    weighted_logistic = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    smote_logistic = ImbPipeline(
        [
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    borderline_smote_logistic = ImbPipeline(
        [
            ("preprocess", preprocess),
            ("smote", BorderlineSMOTE(random_state=RANDOM_STATE)),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    random_forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )
    hist_gb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=250,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    models = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": logistic,
        "weighted_logistic": weighted_logistic,
        "smote_logistic": smote_logistic,
        "borderline_smote_logistic": borderline_smote_logistic,
        "random_forest_balanced": random_forest,
        "hist_gradient_boosting": hist_gb,
    }

    fitted_models: dict[str, object] = {}
    valid_scores_map: dict[str, np.ndarray] = {}
    pr_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    rows: list[dict[str, float]] = []

    for name, model in models.items():
        fitted, metrics, valid_scores = evaluate_model(
            name, model, X_train, y_train, X_valid, y_valid
        )
        fitted_models[name] = fitted
        valid_scores_map[name] = valid_scores
        rows.append(metrics)
        precision, recall, _ = precision_recall_curve(y_valid, valid_scores)
        pr_curves[name] = (precision, recall, metrics["pr_auc"])
        print(name, metrics)

    valid_df = pd.DataFrame(rows).sort_values(
        ["recall", "pr_auc", "precision"], ascending=[False, False, False]
    )
    valid_df.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    save_precision_recall_plot(pr_curves)

    best_name = valid_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    threshold, f2_score = tune_threshold(y_valid, valid_scores_map[best_name])

    test_scores = predict_scores(best_model, X_test)
    test_pred_default = (test_scores >= 0.5).astype(int)
    test_pred_tuned = (test_scores >= threshold).astype(int)

    test_default_metrics = metric_bundle(y_test, test_pred_default, test_scores)
    test_default_metrics["model"] = f"{best_name}_default_threshold"
    test_tuned_metrics = metric_bundle(y_test, test_pred_tuned, test_scores)
    test_tuned_metrics["model"] = f"{best_name}_tuned_threshold"
    test_tuned_metrics["threshold"] = threshold
    test_tuned_metrics["validation_f2"] = f2_score

    pd.DataFrame([test_default_metrics, test_tuned_metrics]).to_csv(
        OUTPUT_DIR / "test_metrics.csv", index=False
    )
    save_confusion_matrices(y_test, test_pred_default, test_pred_tuned)

    if best_name in {"logistic_regression", "weighted_logistic", "smote_logistic", "borderline_smote_logistic"}:
        save_logistic_coefficients(best_model, list(X_train.columns), "Top Logistic Coefficients")

    summary = {
        "dataset_rows": int(len(df)),
        "fraud_cases": int(df["Class"].sum()),
        "fraud_rate_pct": round(float(df["Class"].mean() * 100), 4),
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "test_rows": int(len(X_test)),
        "best_model_validation": best_name,
        "best_threshold": threshold,
        "best_validation_f2": f2_score,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
