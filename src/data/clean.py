"""
Data cleaning and preprocessing pipeline for the Chicago Crime Dataset.

Usage:
    python -m src.data.clean
    python -m src.data.clean --input data/raw/chicago_crimes.csv --sample 100000
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Columns we don't need for modeling or EDA
DROP_COLS = [
    "ID", "Case Number", "IUCR", "FBI Code",
    "X Coordinate", "Y Coordinate", "Updated On",
    "Location", "Beat", "Ward", "Block", "Description",
]

# Bounding box for Chicago proper
LAT_MIN, LAT_MAX = 41.6, 42.1
LON_MIN, LON_MAX = -87.95, -87.5


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to Title Case with spaces.
    The Socrata API returns lowercase snake_case headers
    (e.g. 'arrest', 'primary_type', 'location_description'),
    while manual CSV exports use Title Case with spaces
    (e.g. 'Arrest', 'Primary Type', 'Location Description').
    This function maps both forms to the Title Case variant.
    """
    rename_map = {
        # Socrata API names → canonical names
        "id":                    "ID",
        "case_number":           "Case Number",
        "date":                  "Date",
        "block":                 "Block",
        "iucr":                  "IUCR",
        "primary_type":          "Primary Type",
        "description":           "Description",
        "location_description":  "Location Description",
        "arrest":                "Arrest",
        "domestic":              "Domestic",
        "beat":                  "Beat",
        "district":              "District",
        "ward":                  "Ward",
        "community_area":        "Community Area",
        "fbi_code":              "FBI Code",
        "x_coordinate":          "X Coordinate",
        "y_coordinate":          "Y Coordinate",
        "year":                  "Year",
        "updated_on":            "Updated On",
        "latitude":              "Latitude",
        "longitude":             "Longitude",
        "location":              "Location",
    }
    # Only rename columns that are actually present
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def load_raw(path: str = "data/raw/chicago_crimes.csv") -> pd.DataFrame:
    """Load raw CSV from disk."""
    print(f"Loading raw data from {path}...")
    df = pd.read_csv(path, low_memory=False)
    df = _normalize_columns(df)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline.

    Steps:
        1. Drop unused columns
        2. Drop rows missing target (Arrest) or coordinates
        3. Parse datetime
        4. Encode Arrest as 0/1
        5. Fill remaining nulls
        6. Filter to Chicago bounding box
        7. Remove duplicates

    Returns:
        Cleaned DataFrame.
    """
    original_len = len(df)

    # 1. Drop unused columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 2. Drop rows missing target or coordinates
    df = df.dropna(subset=["Arrest"])
    df = df.dropna(subset=["Latitude", "Longitude"])

    # 3. Parse datetime — handle two common formats
    df["Date"] = pd.to_datetime(
        df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    df = df.dropna(subset=["Date"])

    # 4. Encode boolean target
    df["Arrest"] = (
        df["Arrest"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
    )
    df = df.dropna(subset=["Arrest"])
    df["Arrest"] = df["Arrest"].astype(int)

    # 5. Fill remaining nulls in key columns
    for col in ["District", "Community Area"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    df["Primary Type"] = df["Primary Type"].fillna("UNKNOWN")
    df["Location Description"] = df["Location Description"].fillna("OTHER")
    if "Domestic" in df.columns:
        df["Domestic"] = df["Domestic"].fillna(False)

    # 6. Filter to Chicago bounding box
    df = df[
        (df["Latitude"] >= LAT_MIN) & (df["Latitude"] <= LAT_MAX) &
        (df["Longitude"] >= LON_MIN) & (df["Longitude"] <= LON_MAX)
    ]

    # 7. Remove duplicates
    df = df.drop_duplicates()

    print(
        f"Cleaning complete: {original_len:,} → {len(df):,} rows "
        f"({original_len - len(df):,} removed)"
    )
    return df.reset_index(drop=True)


def sample_stratified(df: pd.DataFrame, n: int = 100000,
                      random_state: int = 42) -> pd.DataFrame:
    """
    Stratified sample preserving Arrest class ratio.
    Returns the full dataset if n >= len(df).
    """
    if n >= len(df):
        print(f"Dataset has only {len(df):,} rows — returning full dataset.")
        return df
    _, df_sample = train_test_split(
        df, test_size=n / len(df), stratify=df["Arrest"],
        random_state=random_state
    )
    arrest_rate = df_sample["Arrest"].mean()
    print(
        f"Sampled {len(df_sample):,} rows "
        f"(arrest rate: {arrest_rate:.1%})"
    )
    return df_sample.reset_index(drop=True)


def save_processed(df: pd.DataFrame,
                   path: str = "data/processed/chicago_cleaned.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data ({df.shape[0]:,} rows, {df.shape[1]} cols) → {path}")


def run_pipeline(raw_path: str = "data/raw/chicago_crimes.csv",
                 n_sample: int = 100000,
                 output_path: str = "data/processed/chicago_cleaned.csv",
                 sample_path: str = "data/samples/chicago_sample.csv") -> pd.DataFrame:
    """End-to-end: load → clean → sample → save."""
    df_raw = load_raw(raw_path)
    df_clean = clean(df_raw)
    save_processed(df_clean, output_path)

    df_sample = sample_stratified(df_clean, n=n_sample)
    save_processed(df_sample, sample_path)
    return df_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Chicago Crime Dataset")
    parser.add_argument("--input", type=str,
                        default="data/raw/chicago_crimes.csv")
    parser.add_argument("--sample", type=int, default=100000,
                        help="Sample size for fast iteration")
    parser.add_argument("--output", type=str,
                        default="data/processed/chicago_cleaned.csv")
    parser.add_argument("--sample_output", type=str,
                        default="data/samples/chicago_sample.csv")
    args = parser.parse_args()
    run_pipeline(args.input, args.sample, args.output, args.sample_output)
