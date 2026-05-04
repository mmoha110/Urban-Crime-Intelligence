"""Training and evaluation for arrest prediction models."""

from __future__ import annotations

import os
import time
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from imblearn.over_sampling import SMOTE
except Exception:  # pragma: no cover - optional dependency safety
    SMOTE = None


def get_models() -> Dict[str, object]:
    """Return baseline model instances with class-imbalance handling."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
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


def _maybe_apply_smote(X_train, y_train, use_smote: bool):
    """Apply SMOTE only when requested and dependency is available."""
    if not use_smote:
        return X_train, y_train

    if SMOTE is None:
        print("SMOTE requested but imbalanced-learn is unavailable; continuing without SMOTE.")
        return X_train, y_train

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"Applied SMOTE: {len(y_train)} -> {len(y_res)} training rows")
    return X_res, y_res


def _tune_random_forest(X_train, y_train):
    """Tune random forest hyperparameters with a compact randomized search."""
    param_dist = {
        "n_estimators": [150, 250, 350, 500],
        "max_depth": [10, 15, 20, 30, None],
        "min_samples_leaf": [1, 2, 4, 6, 10],
        "max_features": ["sqrt", "log2", None],
    }

    base_rf = RandomForestClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=16,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print("Best RF params:", search.best_params_)
    return search.best_estimator_, search.best_params_, search.best_score_


def _tune_logistic_regression(X_train, y_train):
    """Tune logistic regression regularization strength and penalty."""
    param_dist = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
    }

    base_lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )

    search = RandomizedSearchCV(
        estimator=base_lr,
        param_distributions=param_dist,
        n_iter=8,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print("Best LR params:", search.best_params_)
    return search.best_estimator_, search.best_params_, search.best_score_


def _tune_svm(X_train, y_train, subsample: int = 12_000):
    """Tune SVM hyperparameters on a stratified subsample for tractability."""
    X_arr = np.asarray(X_train)
    y_arr = np.asarray(y_train)

    if len(X_arr) > subsample:
        rng = np.random.default_rng(42)
        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]

        pos_frac = len(pos_idx) / len(y_arr)
        n_pos = max(1, int(round(subsample * pos_frac)))
        n_neg = subsample - n_pos
        n_pos = min(n_pos, len(pos_idx))
        n_neg = min(n_neg, len(neg_idx))

        pos_sample = rng.choice(pos_idx, size=n_pos, replace=False)
        neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)
        sample_idx = np.concatenate([pos_sample, neg_sample])
        rng.shuffle(sample_idx)

        X_sub = X_arr[sample_idx]
        y_sub = y_arr[sample_idx]
        print(f"SVM tuning on stratified subsample: {len(sample_idx)} rows")
    else:
        X_sub, y_sub = X_arr, y_arr

    param_dist = {
        "C": [0.5, 1.0, 5.0],
        "gamma": ["scale", 0.01, 0.1],
    }

    base_svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42,
    )

    search = RandomizedSearchCV(
        estimator=base_svm,
        param_distributions=param_dist,
        n_iter=6,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_sub, y_sub)
    print("Best SVM params:", search.best_params_)
    return search.best_estimator_, search.best_params_, search.best_score_


def train_evaluate_all(
    X,
    y,
    output_dir: str = "outputs/models",
    tune_rf: bool = True,
    tune_lr: bool = True,
    tune_svm: bool = False,
    calibrate_rf: bool = True,
    use_smote: bool = True,
) -> dict:
    """Train all classifiers, save artifacts, and return metric summaries."""
    os.makedirs(output_dir, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)

    X_train_bal, y_train_bal = _maybe_apply_smote(X_train, y_train_arr, use_smote)

    models = get_models()
    results = {}
    metadata = {
        "smote_applied": bool(use_smote and SMOTE is not None),
    }

    if tune_rf:
        tuned_rf, best_params, cv_roc_auc = _tune_random_forest(X_train_bal, y_train_bal)
        models["Random Forest"] = tuned_rf
        metadata["rf_tuning"] = {
            "best_params": best_params,
            "cv_roc_auc": float(cv_roc_auc),
        }

    if tune_lr:
        tuned_lr, lr_params, lr_cv = _tune_logistic_regression(X_train_bal, y_train_bal)
        models["Logistic Regression"] = tuned_lr
        metadata["lr_tuning"] = {
            "best_params": lr_params,
            "cv_roc_auc": float(lr_cv),
        }

    if tune_svm:
        tuned_svm, svm_params, svm_cv = _tune_svm(X_train, y_train_arr)
        models["SVM"] = tuned_svm
        metadata["svm_tuning"] = {
            "best_params": svm_params,
            "cv_roc_auc": float(svm_cv),
        }

    for name, model in models.items():
        print(f"\nTraining {name}...")

        if "SVM" in name and len(X_train) > 30_000:
            idx = np.random.choice(len(X_train), size=30_000, replace=False)
            X_tr, y_tr = X_train[idx], y_train_arr[idx]
        elif name in {"Random Forest", "Logistic Regression"}:
            X_tr, y_tr = X_train_bal, y_train_bal
        else:
            X_tr, y_tr = X_train, y_train_arr

        start = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - start

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "Accuracy": accuracy_score(y_test_arr, y_pred),
            "Precision": precision_score(y_test_arr, y_pred, zero_division=0),
            "Recall": recall_score(y_test_arr, y_pred, zero_division=0),
            "F1": f1_score(y_test_arr, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test_arr, y_prob) if y_prob is not None else None,
            "Train Time (s)": round(train_time, 2),
            "model": model,
        }

        print(classification_report(y_test_arr, y_pred))
        print(f"Confusion Matrix:\n{confusion_matrix(y_test_arr, y_pred)}")

        model_path = os.path.join(output_dir, f"{name.replace(' ', '_')}.joblib")
        joblib.dump(model, model_path)
        results[name] = metrics

    if calibrate_rf and "Random Forest" in results:
        print("\nCalibrating Random Forest probabilities...")
        base_rf = results["Random Forest"]["model"]
        calibrator = CalibratedClassifierCV(base_rf, method="sigmoid", cv=3)
        calibrator.fit(X_train_bal, y_train_bal)

        start = time.time()
        y_pred = calibrator.predict(X_test)
        y_prob = calibrator.predict_proba(X_test)[:, 1]
        eval_time = time.time() - start

        calibrated_name = "Random Forest (Calibrated)"
        results[calibrated_name] = {
            "Accuracy": accuracy_score(y_test_arr, y_pred),
            "Precision": precision_score(y_test_arr, y_pred, zero_division=0),
            "Recall": recall_score(y_test_arr, y_pred, zero_division=0),
            "F1": f1_score(y_test_arr, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test_arr, y_prob),
            "Train Time (s)": round(eval_time, 2),
            "model": calibrator,
        }
        joblib.dump(calibrator, os.path.join(output_dir, "Random_Forest_Calibrated.joblib"))
        metadata["rf_calibration"] = "sigmoid"

    results_df = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()}).T

    return {
        "results": results,
        "results_df": results_df,
        "X_test": X_test,
        "y_test": y_test,
        "models": {name: meta["model"] for name, meta in results.items()},
        "scaler": scaler,
        "metadata": metadata,
    }


def cross_validate_model(model, X, y, cv: int = 5):
    """Run stratified cross-validation and print mean +/- std for key metrics."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scoring = ["accuracy", "f1", "roc_auc"]
    cv_results = cross_validate(
        model,
        X_scaled,
        y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        n_jobs=-1,
    )

    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        print(f"{metric}: {scores.mean():.4f} +/- {scores.std():.4f}")

    return cv_results


def run_cross_validation_report(
    X,
    y,
    cv: int = 5,
    report_path: str = "outputs/reports/cv_results.csv",
) -> pd.DataFrame:
    """Run 5-fold CV for all classifiers and export a comparison CSV.

    Uses baseline (non-tuned) models so the report is comparable across runs.
    SVM is cross-validated on a stratified subsample to keep the runtime bounded.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_arr = np.asarray(y)
    scoring = ["accuracy", "f1", "roc_auc"]
    rows = []

    models_for_cv = get_models()

    for name, model in models_for_cv.items():
        if name == "SVM" and len(X_scaled) > 15_000:
            rng = np.random.default_rng(42)
            pos_idx = np.where(y_arr == 1)[0]
            neg_idx = np.where(y_arr == 0)[0]
            n_total = 15_000
            n_pos = max(1, int(round(n_total * len(pos_idx) / len(y_arr))))
            n_pos = min(n_pos, len(pos_idx))
            n_neg = min(n_total - n_pos, len(neg_idx))
            idx = np.concatenate(
                [
                    rng.choice(pos_idx, size=n_pos, replace=False),
                    rng.choice(neg_idx, size=n_neg, replace=False),
                ]
            )
            rng.shuffle(idx)
            X_cv = X_scaled[idx]
            y_cv = y_arr[idx]
            cv_note = f"stratified subsample n={len(idx)}"
        else:
            X_cv = X_scaled
            y_cv = y_arr
            cv_note = f"full dataset n={len(y_arr)}"

        print(f"\nCross-validating {name} ({cv_note})...")
        cv_results = cross_validate(
            model,
            X_cv,
            y_cv,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring,
            n_jobs=-1,
        )
        row = {"model": name, "cv_folds": cv, "subset": cv_note}
        for metric in scoring:
            scores = cv_results[f"test_{metric}"]
            row[f"{metric}_mean"] = float(scores.mean())
            row[f"{metric}_std"] = float(scores.std(ddof=0))
            print(f"  {metric}: {scores.mean():.4f} +/- {scores.std(ddof=0):.4f}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(report_path, index=False)
    print(f"Saved CV report -> {report_path}")
    return df


def tune_random_forest(X_train, y_train):
    """Backwards-compatible wrapper around the upgraded RF tuning helper."""
    best_model, best_params, _ = _tune_random_forest(X_train, y_train)
    print("Best params:", best_params)
    return best_model
