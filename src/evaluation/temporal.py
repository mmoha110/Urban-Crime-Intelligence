"""Temporal validation and distribution drift reporting."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def _safe_roc_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def _feature_drift_report(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        train_col = train_df[col].astype(float)
        test_col = test_df[col].astype(float)

        train_mean = float(train_col.mean())
        test_mean = float(test_col.mean())
        train_std = float(train_col.std(ddof=0))
        test_std = float(test_col.std(ddof=0))

        pooled_std = np.sqrt((train_std**2 + test_std**2) / 2) if (train_std > 0 or test_std > 0) else 0.0
        smd = abs(train_mean - test_mean) / pooled_std if pooled_std > 0 else 0.0

        rows.append(
            {
                "feature": col,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "mean_delta": test_mean - train_mean,
                "train_std": train_std,
                "test_std": test_std,
                "std_mean_difference": smd,
            }
        )

    return pd.DataFrame(rows).sort_values("std_mean_difference", ascending=False).reset_index(drop=True)


def run_temporal_validation_and_drift(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    report_dir: str = "outputs/reports",
    max_rows: int = 60_000,
):
    """Train on older data, test on recent data, and export drift summaries."""
    os.makedirs(report_dir, exist_ok=True)

    required = set(feature_cols + ["Arrest", "Date"])
    missing = [c for c in required if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing required columns for temporal validation: {missing}")

    temp_df = df_feat.sort_values("Date").reset_index(drop=True)
    if len(temp_df) > max_rows:
        temp_df = temp_df.iloc[-max_rows:].reset_index(drop=True)
    split_idx = int(len(temp_df) * 0.8)
    if split_idx <= 0 or split_idx >= len(temp_df):
        raise ValueError("Temporal split failed due to insufficient data.")

    train_df = temp_df.iloc[:split_idx].copy()
    test_df = temp_df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)
    y_train = train_df["Arrest"].astype(int)
    y_test = test_df["Arrest"].astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    temporal_metrics = pd.DataFrame(
        [
            {
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "split_date": test_df["Date"].min(),
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": _safe_roc_auc(y_test, y_prob),
            }
        ]
    )

    yearly_rows = []
    for year in sorted(temp_df["Year"].dropna().astype(int).unique()):
        yearly = temp_df[temp_df["Year"].astype(int) == year]
        if yearly.empty:
            continue
        yearly_rows.append(
            {
                "year": int(year),
                "rows": int(len(yearly)),
                "arrest_rate": float(yearly["Arrest"].mean()),
            }
        )

    yearly_df = pd.DataFrame(yearly_rows)
    drift_df = _feature_drift_report(train_df, test_df, feature_cols)

    temporal_path = os.path.join(report_dir, "temporal_validation.csv")
    drift_path = os.path.join(report_dir, "feature_drift_report.csv")
    yearly_path = os.path.join(report_dir, "yearly_arrest_rate.csv")

    temporal_metrics.to_csv(temporal_path, index=False)
    drift_df.to_csv(drift_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)

    return {
        "temporal_metrics": temporal_metrics,
        "feature_drift": drift_df,
        "yearly_arrest": yearly_df,
        "paths": {
            "temporal": temporal_path,
            "drift": drift_path,
            "yearly_arrest": yearly_path,
        },
    }
