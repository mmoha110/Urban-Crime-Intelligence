"""Map generation utilities for hotspot visualization."""

from __future__ import annotations

import os

import folium
import pandas as pd
from folium.plugins import HeatMap


def make_heatmap(df, output_path: str = "outputs/figures/crime_heatmap.html"):
    """Generate a Folium heatmap of all crime locations."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m = folium.Map(location=[41.85, -87.65], zoom_start=11, tiles="CartoDB positron")
    heat_data = df[["Latitude", "Longitude"]].dropna().values.tolist()
    HeatMap(heat_data, radius=8, blur=10, max_zoom=13).add_to(m)
    m.save(output_path)
    print(f"Heatmap saved -> {output_path}")
    return m


def make_cluster_map(df_with_labels, cluster_stats, output_path: str = "outputs/figures/cluster_map.html"):
    """Generate a Folium map with sampled cluster points and centroid markers."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    m = folium.Map(location=[41.85, -87.65], zoom_start=11, tiles="CartoDB positron")

    if "Cluster" not in df_with_labels.columns:
        m.save(output_path)
        return m

    clustered = df_with_labels[df_with_labels["Cluster"] != -1].copy()
    if clustered.empty:
        m.save(output_path)
        return m

    sampled_groups = []
    for _, group in clustered.groupby("Cluster"):
        sample = group.sample(n=min(200, len(group)), random_state=42)
        sampled_groups.append(sample)

    if not sampled_groups:
        m.save(output_path)
        return m

    sample_df = pd.concat(sampled_groups, ignore_index=True)

    if sample_df.empty:
        m.save(output_path)
        return m

    n_clusters = sample_df["Cluster"].nunique()
    cmap = plt.get_cmap("tab20", n_clusters)

    for cluster_id in sample_df["Cluster"].unique():
        subset = sample_df[sample_df["Cluster"] == cluster_id]
        color = colors.to_hex(cmap(int(cluster_id) % max(n_clusters, 1)))
        for _, row in subset.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.5,
            ).add_to(m)

    for _, row in cluster_stats.iterrows():
        folium.Marker(
            location=[row["lat_center"], row["lon_center"]],
            popup=folium.Popup(
                (
                    f"Cluster {int(row['Cluster'])}<br>"
                    f"Crimes: {int(row['crime_count'])}<br>"
                    f"Arrest Rate: {row['arrest_rate']:.1%}<br>"
                    f"Top Crime: {row['top_crime']}"
                ),
                max_width=220,
            ),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    m.save(output_path)
    print(f"Cluster map saved -> {output_path}")
    return m


def make_arrest_rate_map(df, output_path: str = "outputs/figures/arrest_rate_map.html"):
    """Generate a heatmap emphasizing locations with arrests."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m = folium.Map(location=[41.85, -87.65], zoom_start=11, tiles="CartoDB dark_matter")
    arrest_only = df[df["Arrest"] == 1][["Latitude", "Longitude"]].dropna().values.tolist()
    HeatMap(arrest_only, radius=10, blur=15).add_to(m)
    m.save(output_path)
    print(f"Arrest map saved -> {output_path}")
    return m


def make_filterable_intelligence_map(
    df,
    probability_df: pd.DataFrame | None = None,
    output_path: str = "outputs/figures/filterable_intelligence_map.html",
    top_n_crime_types: int = 5,
):
    """Generate a map with toggle layers for crime types and arrest probability bins."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    m = folium.Map(location=[41.85, -87.65], zoom_start=11, tiles="CartoDB positron")

    base_df = df[["Latitude", "Longitude", "Primary Type", "Arrest"]].dropna().copy()
    if base_df.empty:
        m.save(output_path)
        return m

    sample_size = min(6000, len(base_df))
    base_df = base_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    all_fg = folium.FeatureGroup(name="All Crimes Heatmap", show=True)
    HeatMap(base_df[["Latitude", "Longitude"]].values.tolist(), radius=8, blur=10, max_zoom=13).add_to(all_fg)
    all_fg.add_to(m)

    arrest_fg = folium.FeatureGroup(name="Arrest-only Heatmap", show=False)
    arrest_points = base_df[base_df["Arrest"] == 1][["Latitude", "Longitude"]].values.tolist()
    if arrest_points:
        HeatMap(arrest_points, radius=10, blur=14, max_zoom=13).add_to(arrest_fg)
    arrest_fg.add_to(m)

    top_types = base_df["Primary Type"].value_counts().head(top_n_crime_types).index.tolist()
    for crime_type in top_types:
        fg = folium.FeatureGroup(name=f"Crime Type: {crime_type}", show=False)
        subset = base_df[base_df["Primary Type"] == crime_type]
        for _, row in subset.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=2,
                color="#1f77b4",
                fill=True,
                fill_opacity=0.35,
                weight=0,
            ).add_to(fg)
        fg.add_to(m)

    if probability_df is not None and not probability_df.empty:
        prob_df = probability_df.dropna(subset=["Latitude", "Longitude", "ArrestProb"]).copy()
        prob_df = prob_df.sample(n=min(4000, len(prob_df)), random_state=42)

        bins = [
            ("Arrest Prob: Low (<0.30)", prob_df[prob_df["ArrestProb"] < 0.30], "#2ca02c"),
            ("Arrest Prob: Medium (0.30-0.60)", prob_df[(prob_df["ArrestProb"] >= 0.30) & (prob_df["ArrestProb"] < 0.60)], "#ff7f0e"),
            ("Arrest Prob: High (>=0.60)", prob_df[prob_df["ArrestProb"] >= 0.60], "#d62728"),
        ]

        for layer_name, layer_df, color in bins:
            fg = folium.FeatureGroup(name=layer_name, show=False)
            for _, row in layer_df.iterrows():
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=2,
                    color=color,
                    fill=True,
                    fill_opacity=0.5,
                    weight=0,
                ).add_to(fg)
            fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_path)
    print(f"Filterable intelligence map saved -> {output_path}")
    return m
