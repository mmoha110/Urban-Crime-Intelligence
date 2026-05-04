"""Feature engineering for supervised and unsupervised modeling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def engineer_features(
    df: pd.DataFrame,
    persist_encoders: bool = True,
    encoders_path: str = "outputs/models/encoders.joblib",
) -> pd.DataFrame:
    """Create model-ready engineered features from a cleaned dataframe.

    When ``persist_encoders`` is true the fitted LabelEncoders are saved to
    ``encoders_path`` so inference and notebook workflows can reuse the
    exact same category-to-integer mapping.
    """
    out = df.copy()

    out["Hour"] = out["Date"].dt.hour
    out["DayOfWeek"] = out["Date"].dt.dayofweek
    out["Month"] = out["Date"].dt.month
    out["Year"] = out["Date"].dt.year
    out["IsWeekend"] = (out["DayOfWeek"] >= 5).astype(int)
    out["Season"] = out["Month"].map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Fall",
            10: "Fall",
            11: "Fall",
        }
    )
    out["IsNight"] = ((out["Hour"] >= 22) | (out["Hour"] <= 5)).astype(int)

    le_primary = LabelEncoder()
    le_location = LabelEncoder()
    le_season = LabelEncoder()

    out["PrimaryType_enc"] = le_primary.fit_transform(out["Primary Type"].astype(str))
    out["LocationDesc_enc"] = le_location.fit_transform(out["Location Description"].astype(str))
    out["Season_enc"] = le_season.fit_transform(out["Season"].astype(str))
    out["Domestic_enc"] = (
        out["Domestic"]
        .astype(str)
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0})
        .fillna(0)
        .astype(int)
    )

    if persist_encoders:
        encoders: Dict[str, LabelEncoder] = {
            "primary_type": le_primary,
            "location_description": le_location,
            "season": le_season,
        }
        out_path = Path(encoders_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, out_path)

    return out


def load_encoders(encoders_path: str = "outputs/models/encoders.joblib") -> Dict[str, LabelEncoder]:
    """Load previously persisted LabelEncoders, or raise if missing."""
    path = Path(encoders_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Encoder file not found at {path}. Run engineer_features() with persist_encoders=True first."
        )
    return joblib.load(path)


def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Return X, y, and the feature column list."""
    feature_cols = [
        "Hour",
        "DayOfWeek",
        "Month",
        "Year",
        "IsWeekend",
        "IsNight",
        "PrimaryType_enc",
        "LocationDesc_enc",
        "Season_enc",
        "Domestic_enc",
        "District",
        "Community Area",
        "Latitude",
        "Longitude",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df["Arrest"].astype(int)
    return X, y, feature_cols
