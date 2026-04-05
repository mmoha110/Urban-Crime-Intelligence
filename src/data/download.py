"""
Download Chicago Crime Dataset from the Socrata Open Data API.
Dataset ID: ijzp-q8t2
Full dataset is large (~1.8GB). For development, download a time-bounded subset.

Usage:
    python -m src.data.download
    python -m src.data.download --limit 100000 --year_from 2020
"""

import requests
import os
import argparse
from tqdm import tqdm

DATA_DIR = "data/raw"
DATASET_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"


def download_sample(limit: int = 200000, year_from: int = 2018,
                    output_file: str = "chicago_crimes.csv") -> str:
    """
    Downloads up to `limit` records from the Chicago Crime API.

    Args:
        limit: Maximum number of records to download.
        year_from: Only include crimes from this year onward.
        output_file: Output filename under data/raw/.

    Returns:
        Path to the saved CSV file.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, output_file)

    if os.path.exists(filepath):
        print(f"File already exists: {filepath}. Delete it to re-download.")
        return filepath

    params = {
        "$limit": limit,
        "$order": "date DESC",
        "$where": f"year >= {year_from}",
    }

    print(f"Downloading up to {limit:,} records from {year_from} onward...")
    response = requests.get(DATASET_URL, params=params, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(filepath, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=output_file
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"Saved {limit:,} records → {filepath}")
    return filepath


def download_by_year_range(year_from: int = 2018, year_to: int = 2023,
                            batch_size: int = 50000,
                            output_file: str = "chicago_crimes.csv") -> str:
    """
    Downloads records year by year and concatenates into one file.
    Useful when the API has per-request row limits.
    """
    import pandas as pd

    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, output_file)

    if os.path.exists(filepath):
        print(f"File already exists: {filepath}. Delete it to re-download.")
        return filepath

    all_frames = []
    for year in range(year_from, year_to + 1):
        print(f"Fetching year {year}...")
        offset = 0
        year_frames = []
        while True:
            params = {
                "$limit": batch_size,
                "$offset": offset,
                "$where": f"year = {year}",
                "$order": "date DESC",
            }
            resp = requests.get(DATASET_URL, params=params, timeout=120)
            resp.raise_for_status()
            df_chunk = pd.read_csv(
                __import__("io").StringIO(resp.text), low_memory=False
            )
            if df_chunk.empty:
                break
            year_frames.append(df_chunk)
            offset += batch_size
            if len(df_chunk) < batch_size:
                break
        if year_frames:
            all_frames.append(pd.concat(year_frames, ignore_index=True))
            print(f"  Year {year}: {sum(len(f) for f in year_frames):,} records")

    df_all = pd.concat(all_frames, ignore_index=True)
    df_all.to_csv(filepath, index=False)
    print(f"Saved {len(df_all):,} total records → {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Chicago Crime Dataset")
    parser.add_argument("--limit", type=int, default=200000,
                        help="Max records to download (default: 200000)")
    parser.add_argument("--year_from", type=int, default=2018,
                        help="Earliest year to include (default: 2018)")
    parser.add_argument("--output", type=str, default="chicago_crimes.csv",
                        help="Output filename under data/raw/")
    args = parser.parse_args()
    download_sample(limit=args.limit, year_from=args.year_from,
                    output_file=args.output)
