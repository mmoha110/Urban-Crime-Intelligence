"""Evaluation plots and report export helpers."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve


def plot_confusion_matrices(models, X_test, y_test, output_dir: str = "outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))

    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            ax=ax,
            display_labels=["No Arrest", "Arrest"],
        )
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150)
    plt.close(fig)


def plot_roc_curves(models, X_test, y_test, output_dir: str = "outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close(fig)


def plot_metrics_comparison(results_dict: dict, output_dir: str = "outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    clean_results = {
        model_name: {k: v for k, v in metrics.items() if k != "model"}
        for model_name, metrics in results_dict.items()
    }

    df = pd.DataFrame(clean_results).T[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    fig = df.plot(kind="bar", figsize=(10, 6), ylim=(0, 1)).get_figure()
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150)
    plt.close(fig)


def save_metrics_csv(results_dict: dict, path: str = "outputs/reports/classification_results.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results_dict).T.drop(columns=["model"], errors="ignore")
    df.to_csv(path)
    print(f"Saved metrics -> {path}")


def plot_feature_importance(rf_model, feature_names, top_n: int = 15, output_dir: str = "outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    importances = rf_model.feature_importances_
    n_plot = min(top_n, len(importances), len(feature_names))
    indices = importances.argsort()[::-1][:n_plot]

    fig = plt.figure(figsize=(10, 6))
    plt.bar(range(n_plot), importances[indices])
    plt.xticks(range(n_plot), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.title("Top Feature Importances - Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)


def plot_silhouette_vs_k(kmeans_results, output_dir: str = "outputs/figures"):
    """Plot silhouette score and Davies-Bouldin index across tested k values."""
    os.makedirs(output_dir, exist_ok=True)

    ks = [r["k"] for r in kmeans_results]
    sils = [r["silhouette"] for r in kmeans_results]
    dbs = [r["davies_bouldin"] for r in kmeans_results]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ks, sils, marker="o", color="#1f77b4", label="Silhouette (higher is better)")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Silhouette score", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(ks, dbs, marker="s", color="#d62728", label="Davies-Bouldin (lower is better)")
    ax2.set_ylabel("Davies-Bouldin index", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    best_k = ks[int(np.argmax(sils))]
    ax1.axvline(best_k, linestyle="--", color="gray", alpha=0.6)
    ax1.set_title(f"K-Means selection curve (best k = {best_k})")

    fig.tight_layout()
    out_path = os.path.join(output_dir, "silhouette_vs_k.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved silhouette-vs-k plot -> {out_path}")


def summarize_clusters_to_report(
    cluster_stats: pd.DataFrame,
    top_n: int = 3,
    output_path: str = "outputs/reports/cluster_narrative.md",
) -> str:
    """Write a markdown narrative of the top-N hotspot clusters."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if cluster_stats is None or cluster_stats.empty:
        body = "# Cluster Narrative\n\nNo clusters were discovered.\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(body)
        return output_path

    ranked = cluster_stats.sort_values("crime_count", ascending=False).head(top_n)
    total_crimes = int(cluster_stats["crime_count"].sum())
    overall_arrest_rate = float((cluster_stats["arrest_rate"] * cluster_stats["crime_count"]).sum() / max(total_crimes, 1))

    lines = [
        "# Cluster Narrative",
        "",
        f"Total clustered incidents: {total_crimes:,}",
        f"Weighted mean arrest rate across clusters: {overall_arrest_rate:.2%}",
        "",
        f"## Top {len(ranked)} Hotspot Clusters",
        "",
        "| Rank | Cluster | Crimes | Arrest Rate | Centroid (lat, lon) | Top Crime |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        lines.append(
            f"| {rank} | {int(row['Cluster'])} | {int(row['crime_count']):,} | "
            f"{float(row['arrest_rate']):.2%} | "
            f"({float(row['lat_center']):.4f}, {float(row['lon_center']):.4f}) | "
            f"{row['top_crime']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")

    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        delta = float(row["arrest_rate"]) - overall_arrest_rate
        direction = "above" if delta >= 0 else "below"
        lines.append(
            f"- **Cluster {int(row['Cluster'])}** (rank {rank}) concentrates "
            f"{int(row['crime_count']):,} incidents around "
            f"({float(row['lat_center']):.4f}, {float(row['lon_center']):.4f}). "
            f"The dominant offense is **{row['top_crime']}** and the arrest rate is "
            f"{float(row['arrest_rate']):.2%}, which is {abs(delta):.2%} {direction} the "
            f"weighted mean across all clusters."
        )

    body = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"Saved cluster narrative -> {output_path}")
    return output_path
