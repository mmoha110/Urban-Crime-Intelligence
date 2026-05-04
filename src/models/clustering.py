"""Hotspot detection with K-Means and DBSCAN."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def run_kmeans(df: pd.DataFrame, k_values=None):
    """Run K-Means for several k values and select by silhouette score."""
    if k_values is None:
        k_values = [5, 8, 10, 12, 15]

    coords = df[["Latitude", "Longitude"]].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    results = []
    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(coords_scaled)

        sil_sample = min(10_000, len(coords_scaled))
        sil = silhouette_score(coords_scaled, labels, sample_size=sil_sample, random_state=42)
        db = davies_bouldin_score(coords_scaled, labels)

        results.append(
            {
                "k": k,
                "silhouette": sil,
                "davies_bouldin": db,
                "model": km,
                "labels": labels,
                "scaler": scaler,
            }
        )
        print(f"K={k}: Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}")

    best = max(results, key=lambda r: r["silhouette"])
    print(f"\nBest k={best['k']} (Silhouette={best['silhouette']:.4f})")
    return best, results


def run_dbscan(df: pd.DataFrame, eps: float = 0.01, min_samples: int = 50):
    """Run DBSCAN with haversine distance on latitude/longitude in radians."""
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
    n_noise = (labels == -1).sum()
    noise_pct = (n_noise / len(labels)) * 100

    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points ({noise_pct:.1f}%)")

    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 1:
            sil_sample = min(10_000, mask.sum())
            sil = silhouette_score(coords[mask], labels[mask], sample_size=sil_sample, random_state=42)
            print(f"Silhouette (non-noise): {sil:.4f}")

    return labels, n_clusters


def tune_dbscan(
    df: pd.DataFrame,
    eps_grid=None,
    min_samples_grid=None,
    report_path: str = "outputs/reports/dbscan_tuning.csv",
    max_rows: int = 50_000,
):
    """Sweep DBSCAN eps / min_samples and report silhouette + noise percentage.

    Returns a dict with the results dataframe and the best parameter combo
    (selected by silhouette on non-noise points, tiebreaker: lower noise_pct).
    """
    if eps_grid is None:
        eps_grid = [0.00005, 0.0001, 0.0003, 0.0005, 0.001]
    if min_samples_grid is None:
        min_samples_grid = [30, 50, 100]

    work_df = df.copy()
    if len(work_df) > max_rows:
        work_df = work_df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    coords = work_df[["Latitude", "Longitude"]].values
    coords_rad = np.radians(coords)

    rows = []
    for eps in eps_grid:
        for min_samples in min_samples_grid:
            db = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                algorithm="ball_tree",
                metric="haversine",
                n_jobs=-1,
            )
            labels = db.fit_predict(coords_rad)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            noise_pct = float((n_noise / len(labels)) * 100)

            sil = np.nan
            if n_clusters > 1:
                mask = labels != -1
                if mask.sum() > 1:
                    sil_sample = min(10_000, mask.sum())
                    sil = float(
                        silhouette_score(coords[mask], labels[mask], sample_size=sil_sample, random_state=42)
                    )

            rows.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": int(n_clusters),
                    "noise_points": n_noise,
                    "noise_pct": noise_pct,
                    "silhouette": sil,
                }
            )
            print(
                f"DBSCAN eps={eps}, min_samples={min_samples}: "
                f"clusters={n_clusters}, noise={noise_pct:.1f}%, silhouette={sil}"
            )

    df_results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df_results.to_csv(report_path, index=False)

    scored = df_results.dropna(subset=["silhouette"])
    if not scored.empty:
        best_row = scored.sort_values(
            ["silhouette", "noise_pct"], ascending=[False, True]
        ).iloc[0]
        best = {
            "eps": float(best_row["eps"]),
            "min_samples": int(best_row["min_samples"]),
            "silhouette": float(best_row["silhouette"]),
            "n_clusters": int(best_row["n_clusters"]),
            "noise_pct": float(best_row["noise_pct"]),
        }
        print(
            f"\nBest DBSCAN: eps={best['eps']}, min_samples={best['min_samples']} "
            f"(silhouette={best['silhouette']:.4f}, noise={best['noise_pct']:.1f}%)"
        )
    else:
        best = None
        print("\nDBSCAN tuning: no configuration produced >1 cluster.")

    return {"results": df_results, "best": best, "path": report_path}


def analyze_clusters(df: pd.DataFrame, labels, name: str = "KMeans"):
    """Attach labels and report per-cluster summary statistics."""
    out = df.copy()
    out["Cluster"] = labels

    clustered = out[out["Cluster"] != -1]
    if clustered.empty:
        return out, pd.DataFrame(columns=["Cluster", "crime_count", "arrest_rate", "lat_center", "lon_center", "top_crime"])

    cluster_stats = (
        clustered.groupby("Cluster")
        .agg(
            crime_count=("Cluster", "count"),
            arrest_rate=("Arrest", "mean"),
            lat_center=("Latitude", "mean"),
            lon_center=("Longitude", "mean"),
            top_crime=("Primary Type", lambda x: x.mode().iloc[0] if not x.mode().empty else "UNKNOWN"),
        )
        .reset_index()
        .sort_values("crime_count", ascending=False)
    )

    print(f"\n{name} Cluster Summary:")
    print(cluster_stats.to_string(index=False))
    return out, cluster_stats


def clustering_stability_report(df: pd.DataFrame, k: int, seeds=None):
    """Estimate K-Means stability by repeating across random seeds."""
    if seeds is None:
        seeds = [11, 21, 42, 84, 126]

    coords = df[["Latitude", "Longitude"]].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    rows = []
    for seed in seeds:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
        labels = km.fit_predict(coords_scaled)

        sil_sample = min(10_000, len(coords_scaled))
        sil = silhouette_score(coords_scaled, labels, sample_size=sil_sample, random_state=seed)
        db = davies_bouldin_score(coords_scaled, labels)
        rows.append({"seed": seed, "silhouette": sil, "davies_bouldin": db})

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "k": k,
                "silhouette_mean": float(detail["silhouette"].mean()),
                "silhouette_std": float(detail["silhouette"].std(ddof=0)),
                "davies_bouldin_mean": float(detail["davies_bouldin"].mean()),
                "davies_bouldin_std": float(detail["davies_bouldin"].std(ddof=0)),
            }
        ]
    )
    return detail, summary


def compare_kmeans_dbscan(
    df: pd.DataFrame,
    best_kmeans: dict,
    report_dir: str = "outputs/reports",
    dbscan_eps: float = 0.01,
    dbscan_min_samples: int = 50,
    max_rows: int = 50_000,
    best_dbscan: dict | None = None,
):
    """Create exported comparison/stability reports for K-Means and DBSCAN.

    When ``best_dbscan`` is provided (from :func:`tune_dbscan`) its eps and
    min_samples override the defaults so the comparison reflects the tuned
    configuration.
    """
    os.makedirs(report_dir, exist_ok=True)

    if best_dbscan is not None:
        dbscan_eps = float(best_dbscan.get("eps", dbscan_eps))
        dbscan_min_samples = int(best_dbscan.get("min_samples", dbscan_min_samples))

    work_df = df.copy()
    if len(work_df) > max_rows:
        work_df = work_df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    kmeans_rows = [
        {
            "method": "KMeans",
            "k": int(best_kmeans["k"]),
            "silhouette": float(best_kmeans["silhouette"]),
            "davies_bouldin": float(best_kmeans["davies_bouldin"]),
            "n_clusters": int(best_kmeans["k"]),
            "noise_points": 0,
            "noise_pct": 0.0,
        }
    ]

    labels_dbscan, n_clusters_dbscan = run_dbscan(work_df, eps=dbscan_eps, min_samples=dbscan_min_samples)
    n_noise = int((labels_dbscan == -1).sum())
    noise_pct = float((n_noise / len(labels_dbscan)) * 100)

    dbscan_row = {
        "method": "DBSCAN",
        "k": np.nan,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "n_clusters": int(n_clusters_dbscan),
        "noise_points": n_noise,
        "noise_pct": noise_pct,
    }

    if n_clusters_dbscan > 1:
        mask = labels_dbscan != -1
        if mask.sum() > 1:
            coords = work_df[["Latitude", "Longitude"]].values
            sil_sample = min(10_000, mask.sum())
            dbscan_row["silhouette"] = float(
                silhouette_score(coords[mask], labels_dbscan[mask], sample_size=sil_sample, random_state=42)
            )

    detail_df, stability_summary = clustering_stability_report(work_df, k=int(best_kmeans["k"]))
    comparison_df = pd.DataFrame(kmeans_rows + [dbscan_row])

    comparison_path = os.path.join(report_dir, "clustering_comparison.csv")
    stability_detail_path = os.path.join(report_dir, "kmeans_stability_detail.csv")
    stability_summary_path = os.path.join(report_dir, "kmeans_stability_summary.csv")

    comparison_df.to_csv(comparison_path, index=False)
    detail_df.to_csv(stability_detail_path, index=False)
    stability_summary.to_csv(stability_summary_path, index=False)

    return {
        "comparison": comparison_df,
        "stability_detail": detail_df,
        "stability_summary": stability_summary,
        "paths": {
            "comparison": comparison_path,
            "stability_detail": stability_detail_path,
            "stability_summary": stability_summary_path,
        },
    }
