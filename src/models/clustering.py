"""
Clustering pipeline for geographic crime hotspot detection (Task 2).

Implements K-Means and DBSCAN on lat/lon coordinates and evaluates
cluster quality with Silhouette Score and Davies-Bouldin Index.

Usage:
    python -m src.models.clustering
    python -m src.models.clustering --input data/samples/chicago_sample.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------

def run_kmeans(
    df: pd.DataFrame,
    k_values: list[int] | None = None,
    sample_size_sil: int = 10000,
) -> tuple[dict, list[dict]]:
    """
    Runs K-Means for each k in k_values and picks the best by silhouette score.

    Args:
        df: DataFrame with Latitude and Longitude columns.
        k_values: List of k values to try.
        sample_size_sil: Sample size used for silhouette score estimation.

    Returns:
        best: Dict with keys k, silhouette, davies_bouldin, model, labels, scaler.
        all_results: List of the same dict for every k.
    """
    if k_values is None:
        k_values = [5, 8, 10, 12, 15]

    coords = df[["Latitude", "Longitude"]].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    all_results = []
    print("\nK-Means clustering:")
    print(f"{'k':>4}  {'Silhouette':>12}  {'Davies-Bouldin':>16}  {'Inertia':>12}")
    print("-" * 50)

    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(coords_scaled)

        sil = silhouette_score(
            coords_scaled, labels,
            sample_size=min(sample_size_sil, len(coords_scaled)),
            random_state=42,
        )
        db = davies_bouldin_score(coords_scaled, labels)

        print(f"{k:>4}  {sil:>12.4f}  {db:>16.4f}  {km.inertia_:>12.1f}")
        all_results.append({
            "k": k,
            "silhouette":      sil,
            "davies_bouldin":  db,
            "inertia":         km.inertia_,
            "model":           km,
            "labels":          labels,
            "scaler":          scaler,
        })

    best = max(all_results, key=lambda r: r["silhouette"])
    print(f"\nBest k = {best['k']}  (Silhouette = {best['silhouette']:.4f})")
    return best, all_results


def elbow_data(all_results: list[dict]) -> pd.DataFrame:
    """Returns a DataFrame with k, inertia, and silhouette for elbow plotting."""
    rows = [{"k": r["k"], "inertia": r["inertia"], "silhouette": r["silhouette"]}
            for r in all_results]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

def run_dbscan(
    df: pd.DataFrame,
    eps: float = 0.005,
    min_samples: int = 50,
    sample_size_sil: int = 10000,
) -> tuple[np.ndarray, int]:
    """
    DBSCAN using the haversine metric on lat/lon coordinates.

    Args:
        df: DataFrame with Latitude and Longitude columns.
        eps: Neighbourhood radius in radians (0.005 rad ≈ 550 m).
        min_samples: Min points to form a core point.
        sample_size_sil: Sample size for silhouette estimation.

    Returns:
        labels: Array of cluster labels (-1 = noise).
        n_clusters: Number of clusters found (excluding noise).
    """
    coords = df[["Latitude", "Longitude"]].values
    coords_rad = np.radians(coords)

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
        n_jobs=-1,
    )
    labels = db.fit_predict(coords_rad)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    noise_pct  = n_noise / len(labels) * 100

    print(f"\nDBSCAN (eps={eps}, min_samples={min_samples}):")
    print(f"  Clusters : {n_clusters}")
    print(f"  Noise    : {n_noise:,} points ({noise_pct:.1f}%)")

    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 1:
            sample_n = min(sample_size_sil, mask.sum())
            rng = np.random.default_rng(42)
            idx = rng.choice(np.where(mask)[0], size=sample_n, replace=False)
            sil = silhouette_score(coords[idx], labels[idx])
            print(f"  Silhouette (non-noise, n={sample_n}): {sil:.4f}")

    return labels, n_clusters


def tune_dbscan(
    df: pd.DataFrame,
    eps_values: list[float] | None = None,
    min_samples_values: list[int] | None = None,
) -> pd.DataFrame:
    """
    Grid search over eps and min_samples for DBSCAN.

    Returns a DataFrame summarising n_clusters, noise%, and silhouette.
    """
    if eps_values is None:
        eps_values = [0.003, 0.005, 0.008, 0.01]
    if min_samples_values is None:
        min_samples_values = [30, 50, 100]

    coords = df[["Latitude", "Longitude"]].values
    coords_rad = np.radians(coords)
    rows = []

    print("\nDBSCAN grid search:")
    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms, algorithm="ball_tree",
                        metric="haversine", n_jobs=-1)
            labels = db.fit_predict(coords_rad)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_pct  = (labels == -1).mean() * 100
            mask = labels != -1
            sil = None
            if n_clusters > 1 and mask.sum() > 1:
                s = min(10000, mask.sum())
                rng = np.random.default_rng(42)
                idx = rng.choice(np.where(mask)[0], size=s, replace=False)
                try:
                    sil = round(silhouette_score(coords[idx], labels[idx]), 4)
                except Exception:
                    pass
            rows.append({"eps": eps, "min_samples": ms,
                         "n_clusters": n_clusters, "noise_pct": round(noise_pct, 1),
                         "silhouette": sil})
            print(f"  eps={eps}, min_samples={ms}: "
                  f"{n_clusters} clusters, {noise_pct:.1f}% noise, sil={sil}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cluster analysis
# ---------------------------------------------------------------------------

def analyze_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    name: str = "KMeans",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Attaches cluster labels to the DataFrame and computes per-cluster stats.

    Args:
        df: Feature-engineered DataFrame.
        labels: Cluster label array (same length as df).
        name: Algorithm name for display.

    Returns:
        df_labeled: Copy of df with 'Cluster' column added.
        cluster_stats: Per-cluster summary DataFrame.
    """
    df = df.copy()
    df["Cluster"] = labels

    # Exclude DBSCAN noise
    df_clustered = df[df["Cluster"] != -1]

    cluster_stats = (
        df_clustered
        .groupby("Cluster")
        .agg(
            crime_count  =("Cluster",         "count"),
            arrest_rate  =("Arrest",           "mean"),
            lat_center   =("Latitude",         "mean"),
            lon_center   =("Longitude",        "mean"),
            top_crime    =("Primary Type",     lambda x: x.mode()[0]),
        )
        .reset_index()
        .sort_values("crime_count", ascending=False)
    )

    print(f"\n{name} Cluster Summary (top 10 by crime count):")
    print(cluster_stats.head(10).to_string(index=False))
    return df, cluster_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str = "data/samples/chicago_sample.csv",
    output_dir: str = "outputs/models",
    reports_dir: str = "outputs/reports",
) -> dict:
    """End-to-end clustering pipeline."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    df = df.dropna(subset=["Latitude", "Longitude"])
    print(f"  {len(df):,} rows with valid coordinates")

    # ---- K-Means ----
    best_km, all_km = run_kmeans(df)
    df_km, km_stats = analyze_clusters(df, best_km["labels"], name="KMeans")

    # Save elbow data
    os.makedirs(reports_dir, exist_ok=True)
    elbow_data(all_km).to_csv(
        os.path.join(reports_dir, "kmeans_elbow.csv"), index=False
    )
    km_stats.to_csv(
        os.path.join(reports_dir, "kmeans_cluster_stats.csv"), index=False
    )

    # Save best K-Means model
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best_km["model"],  os.path.join(output_dir, "kmeans_best.joblib"))
    joblib.dump(best_km["scaler"], os.path.join(output_dir, "kmeans_scaler.joblib"))
    print(f"\nK-Means model saved → {output_dir}/kmeans_best.joblib")

    # ---- DBSCAN ----
    db_labels, n_clusters = run_dbscan(df)
    df_db, db_stats = analyze_clusters(df, db_labels, name="DBSCAN")
    db_stats.to_csv(
        os.path.join(reports_dir, "dbscan_cluster_stats.csv"), index=False
    )

    return {
        "kmeans": {"best": best_km, "all": all_km,
                   "df_labeled": df_km, "stats": km_stats},
        "dbscan": {"labels": db_labels, "n_clusters": n_clusters,
                   "df_labeled": df_db, "stats": db_stats},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster crime locations")
    parser.add_argument("--input",   type=str,
                        default="data/samples/chicago_sample.csv")
    parser.add_argument("--output",  type=str, default="outputs/models")
    parser.add_argument("--reports", type=str, default="outputs/reports")
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.reports)
