"""
Interactive Folium map generation for crime hotspot visualization.

Outputs HTML files (openable in any browser) to outputs/figures/.

Usage:
    python -m src.visualization.maps
    python -m src.visualization.maps --input data/samples/chicago_sample.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster

# Chicago approximate centre
CHICAGO_CENTER = [41.85, -87.65]
DEFAULT_ZOOM   = 11


# ---------------------------------------------------------------------------
# Heatmap — all crimes
# ---------------------------------------------------------------------------

def make_heatmap(
    df: pd.DataFrame,
    output_path: str = "outputs/figures/crime_heatmap.html",
    max_points: int = 100000,
) -> folium.Map:
    """
    Folium heatmap of all crime locations.

    Args:
        df: DataFrame with Latitude and Longitude columns.
        output_path: Where to save the HTML file.
        max_points: Downsample for browser performance.

    Returns:
        folium.Map object.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_valid = df.dropna(subset=["Latitude", "Longitude"])
    if len(df_valid) > max_points:
        df_valid = df_valid.sample(max_points, random_state=42)

    m = folium.Map(location=CHICAGO_CENTER, zoom_start=DEFAULT_ZOOM,
                   tiles="CartoDB positron")
    heat_data = df_valid[["Latitude", "Longitude"]].values.tolist()
    HeatMap(heat_data, radius=8, blur=10, max_zoom=13).add_to(m)

    m.save(output_path)
    print(f"Crime heatmap saved → {output_path}")
    return m


# ---------------------------------------------------------------------------
# Cluster map — coloured by cluster ID
# ---------------------------------------------------------------------------

def make_cluster_map(
    df_with_labels: pd.DataFrame,
    cluster_stats: pd.DataFrame,
    output_path: str = "outputs/figures/cluster_map.html",
    points_per_cluster: int = 200,
) -> folium.Map:
    """
    Folium map with cluster-coloured scatter and centroid pop-ups.

    Args:
        df_with_labels: DataFrame with Latitude, Longitude, Cluster, Arrest,
                        Primary Type columns.
        cluster_stats: Output of clustering.analyze_clusters().
        output_path: Where to save the HTML file.
        points_per_cluster: Max sample per cluster for browser performance.

    Returns:
        folium.Map object.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    m = folium.Map(location=CHICAGO_CENTER, zoom_start=DEFAULT_ZOOM,
                   tiles="CartoDB positron")

    # Sample points per cluster
    df_plot = df_with_labels[df_with_labels["Cluster"] != -1]
    sample_df = (
        df_plot
        .groupby("Cluster", group_keys=False)
        .apply(lambda x: x.sample(min(points_per_cluster, len(x)), random_state=42))
        .reset_index(drop=True)
    )

    n_clusters = sample_df["Cluster"].nunique()
    cmap = cm.get_cmap("tab20", max(n_clusters, 1))

    for cluster_id in sample_df["Cluster"].unique():
        subset = sample_df[sample_df["Cluster"] == cluster_id]
        color = mcolors.to_hex(cmap(int(cluster_id) % n_clusters))
        for _, row in subset.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.5,
                weight=0,
            ).add_to(m)

    # Centroid markers with cluster stats
    for _, row in cluster_stats.iterrows():
        popup_html = (
            f"<b>Cluster {int(row['Cluster'])}</b><br>"
            f"Crimes: {int(row['crime_count']):,}<br>"
            f"Arrest Rate: {row['arrest_rate']:.1%}<br>"
            f"Top Crime: {row['top_crime']}"
        )
        folium.Marker(
            location=[row["lat_center"], row["lon_center"]],
            popup=folium.Popup(popup_html, max_width=220),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    m.save(output_path)
    print(f"Cluster map saved → {output_path}")
    return m


# ---------------------------------------------------------------------------
# Arrest-rate heatmap
# ---------------------------------------------------------------------------

def make_arrest_rate_map(
    df: pd.DataFrame,
    output_path: str = "outputs/figures/arrest_rate_map.html",
    max_points: int = 50000,
) -> folium.Map:
    """
    Heatmap of arrest locations only (shows geographic arrest concentration).

    Args:
        df: DataFrame with Latitude, Longitude, and Arrest columns.
        output_path: Where to save the HTML file.
        max_points: Downsample for browser performance.

    Returns:
        folium.Map object.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    arrest_df = df[df["Arrest"] == 1].dropna(subset=["Latitude", "Longitude"])
    if len(arrest_df) > max_points:
        arrest_df = arrest_df.sample(max_points, random_state=42)

    m = folium.Map(location=CHICAGO_CENTER, zoom_start=DEFAULT_ZOOM,
                   tiles="CartoDB dark_matter")
    heat_data = arrest_df[["Latitude", "Longitude"]].values.tolist()
    HeatMap(heat_data, radius=10, blur=15).add_to(m)

    m.save(output_path)
    print(f"Arrest-rate map saved → {output_path}")
    return m


# ---------------------------------------------------------------------------
# Crime-type marker-cluster map
# ---------------------------------------------------------------------------

def make_crime_type_map(
    df: pd.DataFrame,
    crime_type: str = "HOMICIDE",
    output_path: str | None = None,
    max_points: int = 5000,
) -> folium.Map:
    """
    MarkerCluster map for a specific crime type.

    Args:
        df: Cleaned DataFrame.
        crime_type: Value from 'Primary Type' to filter on.
        output_path: HTML output path; auto-generated if None.
        max_points: Max markers to add.

    Returns:
        folium.Map object.
    """
    if output_path is None:
        safe_name = crime_type.lower().replace(" ", "_")
        output_path = f"outputs/figures/crime_type_{safe_name}.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    subset = df[df["Primary Type"].str.upper() == crime_type.upper()].dropna(
        subset=["Latitude", "Longitude"]
    )
    if len(subset) > max_points:
        subset = subset.sample(max_points, random_state=42)

    m = folium.Map(location=CHICAGO_CENTER, zoom_start=DEFAULT_ZOOM,
                   tiles="CartoDB positron")
    mc = MarkerCluster().add_to(m)
    for _, row in subset.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"{row['Primary Type']} — {row.get('Date', '')}",
            icon=folium.Icon(color="orange", icon="exclamation-sign"),
        ).add_to(mc)

    m.save(output_path)
    print(f"Crime-type map ({crime_type}) saved → {output_path}")
    return m


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str = "data/samples/chicago_sample.csv",
    cluster_labels: np.ndarray | None = None,
    cluster_stats: pd.DataFrame | None = None,
) -> None:
    """Generate all standard maps from a sample CSV."""
    df = pd.read_csv(input_path, low_memory=False)
    if "Arrest" in df.columns:
        df["Arrest"] = pd.to_numeric(df["Arrest"], errors="coerce").fillna(0).astype(int)

    make_heatmap(df)
    make_arrest_rate_map(df)

    if cluster_labels is not None and cluster_stats is not None:
        df["Cluster"] = cluster_labels
        make_cluster_map(df, cluster_stats)

    print("\nAll maps generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate crime hotspot maps")
    parser.add_argument("--input", type=str,
                        default="data/samples/chicago_sample.csv")
    args = parser.parse_args()
    run_pipeline(args.input)
