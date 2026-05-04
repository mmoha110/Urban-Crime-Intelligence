"""Data cleaning and sampling helpers for the Chicago crimes dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


COLUMN_MAP = {
    "id": "ID",
    "case_number": "Case Number",
    "date": "Date",
    "block": "Block",
    "iucr": "IUCR",
    "primary_type": "Primary Type",
    "description": "Description",
    "location_description": "Location Description",
    "arrest": "Arrest",
    "domestic": "Domestic",
    "beat": "Beat",
    "district": "District",
    "ward": "Ward",
    "community_area": "Community Area",
    "fbi_code": "FBI Code",
    "x_coordinate": "X Coordinate",
    "y_coordinate": "Y Coordinate",
    "year": "Year",
    "updated_on": "Updated On",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "location": "Location",
}

DROP_COLS = [
    "ID",
    "Case Number",
    "IUCR",
    "FBI Code",
    "X Coordinate",
    "Y Coordinate",
    "Updated On",
    "Location",
    "Beat",
    "Ward",
    "Block",
    "Description",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map API-style lowercase columns to project standard column names."""
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in COLUMN_MAP:
            rename_map[col] = COLUMN_MAP[key]
    return df.rename(columns=rename_map)


def load_raw(path: str = "data/raw/chicago_crimes.csv") -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline and return a cleaned dataframe."""
    out = _normalize_columns(df.copy())

    # Normalize column names from Socrata API snake_case to Title Case
    out = out.rename(columns=COLUMN_MAP)

    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns], errors="ignore")
    out = out.dropna(subset=["Arrest"])
    out = out.dropna(subset=["Latitude", "Longitude"])

    # Supports both Open Data CSV datetime format and Socrata API ISO timestamps.
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])

    out["Arrest"] = (
        out["Arrest"]
        .astype(str)
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0})
        .fillna(0)
        .astype(int)
    )

    if "District" in out.columns and not out["District"].mode().empty:
        out["District"] = out["District"].fillna(out["District"].mode()[0])
    if "Community Area" in out.columns and not out["Community Area"].mode().empty:
        out["Community Area"] = out["Community Area"].fillna(out["Community Area"].mode()[0])

    out["Primary Type"] = out.get("Primary Type", "UNKNOWN")
    out["Primary Type"] = out["Primary Type"].fillna("UNKNOWN")

    out["Location Description"] = out.get("Location Description", "OTHER")
    out["Location Description"] = out["Location Description"].fillna("OTHER")

    out = out[(out["Latitude"] >= 41.6) & (out["Latitude"] <= 42.1)]
    out = out[(out["Longitude"] >= -87.95) & (out["Longitude"] <= -87.5)]

    return out.reset_index(drop=True)


def sample(df: pd.DataFrame, n: int = 100_000, random_state: int = 42) -> pd.DataFrame:
    """Create a stratified sample preserving Arrest class ratio."""
    if n >= len(df):
        return df.copy()

    _, sample_df = train_test_split(
        df,
        test_size=n / len(df),
        stratify=df["Arrest"],
        random_state=random_state,
    )
    return sample_df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: str = "data/processed/chicago_cleaned.csv") -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned data: {df.shape} -> {out_path}")
    return out_path


if __name__ == "__main__":
    raw_df = load_raw()
    cleaned_df = clean(raw_df)
    save_processed(cleaned_df)
