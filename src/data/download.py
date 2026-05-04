"""Utilities for downloading Chicago crime data."""

from __future__ import annotations

import os
from pathlib import Path

import requests

DATA_DIR = Path("data/raw")
DATASET_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"


def download_sample(
    limit: int = 200_000,
    output_file: str = "chicago_crimes.csv",
    min_year: int = 2018,
    timeout: int = 120,
) -> Path:
    """Download a recent subset of records from the Socrata API."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        "$limit": limit,
        "$order": "date DESC",
        "$where": f"year >= {min_year}",
    }

    response = requests.get(DATASET_URL, params=params, timeout=timeout)
    response.raise_for_status()

    out_path = DATA_DIR / output_file
    out_path.write_bytes(response.content)
    print(f"Downloaded {limit} records to {out_path}")
    return out_path


if __name__ == "__main__":
    env_limit = int(os.getenv("CRIME_DOWNLOAD_LIMIT", "200000"))
    download_sample(limit=env_limit)
