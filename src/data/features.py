"""
Feature engineering for the Chicago Crime Dataset.

Adds temporal and encoded categorical features to a cleaned DataFrame,
then produces an (X, y, feature_names) tuple ready for modeling.

Usage:
    python -m src.data.features
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Season mapping by month
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring",  4: "Spring",  5: "Spring",
    6: "Summer",  7: "Summer",  8: "Summer",
    9: "Fall",   10: "Fall",   11: "Fall",
}

# Feature columns used for model input
FEATURE_COLS = [
    "Hour", "DayOfWeek", "Month", "Year",
    "IsWeekend", "IsNight",
    "PrimaryType_enc", "LocationDesc_enc",
    "Season_enc", "Domestic_enc",
    "District", "Community Area",
    "Latitude", "Longitude",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds temporal and encoded categorical features in-place.

    New columns added:
        Temporal  : Hour, DayOfWeek, Month, Year, IsWeekend, Season, IsNight
        Encoded   : PrimaryType_enc, LocationDesc_enc, Season_enc, Domestic_enc

    Args:
        df: Cleaned DataFrame (output of clean.clean()).

    Returns:
        DataFrame with new feature columns appended.
    """
    df = df.copy()

    # ---- Temporal features ----
    df["Hour"]      = df["Date"].dt.hour
    df["DayOfWeek"] = df["Date"].dt.dayofweek   # 0 = Monday
    df["Month"]     = df["Date"].dt.month
    df["Year"]      = df["Date"].dt.year
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Season"]    = df["Month"].map(SEASON_MAP)
    df["IsNight"]   = ((df["Hour"] >= 22) | (df["Hour"] <= 5)).astype(int)

    # ---- Categorical encoding ----
    le = LabelEncoder()
    df["PrimaryType_enc"]  = le.fit_transform(df["Primary Type"].astype(str))
    df["LocationDesc_enc"] = le.fit_transform(
        df["Location Description"].astype(str)
    )
    df["Season_enc"] = le.fit_transform(df["Season"].astype(str))
    df["Domestic_enc"] = (
        df["Domestic"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(int)
    )

    # Ensure numeric district / community area
    for col in ["District", "Community Area"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extracts model-ready (X, y, feature_names) from an engineered DataFrame.

    Args:
        df: DataFrame with features already added (via engineer_features).
        feature_cols: Override the default feature column list.

    Returns:
        X: Feature matrix (numpy array).
        y: Target array (numpy array of 0/1).
        feature_names: List of column names corresponding to X's columns.
    """
    cols = feature_cols if feature_cols else FEATURE_COLS
    cols = [c for c in cols if c in df.columns]
    X = df[cols].fillna(0).values.astype(float)
    y = df["Arrest"].values.astype(int)
    return X, y, cols


def run_pipeline(
    input_path: str = "data/samples/chicago_sample.csv",
    output_path: str = "data/processed/chicago_features.csv",
) -> pd.DataFrame:
    """Load cleaned data, engineer features, and save."""
    df = pd.read_csv(input_path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df_feat = engineer_features(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_feat.to_csv(output_path, index=False)
    print(f"Feature-engineered data saved → {output_path}  ({df_feat.shape})")
    return df_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engineer features for Chicago Crime")
    parser.add_argument("--input",  type=str,
                        default="data/samples/chicago_sample.csv")
    parser.add_argument("--output", type=str,
                        default="data/processed/chicago_features.csv")
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
