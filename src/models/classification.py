"""
Classification pipeline for arrest prediction (Task 1).

Trains Logistic Regression, Random Forest, and SVM on the feature-engineered
Chicago Crime Dataset and reports evaluation metrics.

Usage:
    python -m src.models.classification
    python -m src.models.classification --input data/processed/chicago_features.csv
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

from src.data.features import get_feature_matrix, engineer_features


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """
    Returns instantiated classifiers.
    class_weight='balanced' addresses the ~25–30% arrest-rate imbalance.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
    }


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def _compute_metrics(model, X_test: np.ndarray, y_test: np.ndarray,
                     train_time: float) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "Accuracy":        round(accuracy_score(y_test, y_pred), 4),
        "Precision":       round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":          round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":              round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":         round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
        "Train Time (s)":  round(train_time, 2),
    }


def train_evaluate_all(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str = "outputs/models",
    svm_subset: int = 30000,
) -> dict:
    """
    Trains all three classifiers, evaluates on a held-out 20% test set,
    prints classification reports, saves models, and returns a results dict.

    Args:
        X: Feature matrix (scaled inside this function).
        y: Target vector (0/1).
        output_dir: Directory to save .joblib model files.
        svm_subset: Max training rows for SVM (O(n²) complexity).

    Returns:
        dict mapping model name → metrics dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Scale all features (required for LR and SVM; harmless for RF)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        # SVM subset due to O(n²) complexity
        if "SVM" in name and len(X_train) > svm_subset:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_train), size=svm_subset, replace=False)
            X_tr, y_tr = X_train[idx], y_train[idx]
            print(f"  (SVM: using {svm_subset:,}/{len(X_train):,} training rows)")
        else:
            X_tr, y_tr = X_train, y_train

        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0

        print(classification_report(y_test, model.predict(X_test),
                                    target_names=["No Arrest", "Arrest"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, model.predict(X_test)))

        metrics = _compute_metrics(model, X_test, y_test, train_time)
        metrics["model"] = model
        results[name] = metrics

        model_path = os.path.join(output_dir, f"{name.replace(' ', '_')}.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved → {model_path}")

    return results, X_test, y_test


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict:
    """
    Stratified k-fold cross-validation with Accuracy, F1, and ROC-AUC.

    Args:
        model: A scikit-learn estimator (will be cloned internally).
        X: Feature matrix (un-scaled; scaling applied inside pipeline).
        y: Target vector.
        cv: Number of folds.

    Returns:
        dict of cross_validate results.
    """
    from sklearn.pipeline import make_pipeline

    scaler = StandardScaler()
    pipeline = make_pipeline(scaler, model)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1", "roc_auc"]

    print(f"\nCross-validation ({cv}-fold) for {type(model).__name__}:")
    cv_results = cross_validate(
        pipeline, X, y,
        cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False
    )
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        print(f"  {metric:12s}: {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 3,
) -> RandomForestClassifier:
    """
    Grid-search hyperparameter tuning for Random Forest.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.
        cv: Number of cross-validation folds.

    Returns:
        Best RandomForestClassifier estimator.
    """
    param_grid = {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [10, 15, 20, None],
        "min_samples_leaf": [2, 5, 10],
    }
    rf = RandomForestClassifier(
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring="f1",
        n_jobs=-1, verbose=2, refit=True
    )
    print("\nHyperparameter tuning for Random Forest...")
    grid_search.fit(X_train, y_train)
    print(f"Best params : {grid_search.best_params_}")
    print(f"Best F1     : {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str = "data/processed/chicago_features.csv",
    output_dir: str = "outputs/models",
    reports_dir: str = "outputs/reports",
) -> dict:
    """End-to-end classification pipeline."""
    print(f"Loading features from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    X, y, feature_names = get_feature_matrix(df)
    print(f"Feature matrix: {X.shape}  |  Arrest rate: {y.mean():.1%}")

    results, X_test, y_test = train_evaluate_all(X, y, output_dir=output_dir)

    # Save metrics table
    os.makedirs(reports_dir, exist_ok=True)
    rows = []
    for name, m in results.items():
        row = {k: v for k, v in m.items() if k != "model"}
        row["Model"] = name
        rows.append(row)
    df_metrics = pd.DataFrame(rows).set_index("Model")
    metrics_path = os.path.join(reports_dir, "classification_results.csv")
    df_metrics.to_csv(metrics_path)
    print(f"\nMetrics saved → {metrics_path}")
    print(df_metrics.to_string())

    return results, X_test, y_test, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifiers for arrest prediction")
    parser.add_argument("--input",   type=str,
                        default="data/processed/chicago_features.csv")
    parser.add_argument("--output",  type=str, default="outputs/models")
    parser.add_argument("--reports", type=str, default="outputs/reports")
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.reports)
