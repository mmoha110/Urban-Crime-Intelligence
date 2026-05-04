"""Reusable plotting wrappers for project reports."""

from .metrics import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_metrics_comparison,
    plot_roc_curves,
    plot_silhouette_vs_k,
    summarize_clusters_to_report,
)

__all__ = [
    "plot_confusion_matrices",
    "plot_roc_curves",
    "plot_metrics_comparison",
    "plot_feature_importance",
    "plot_silhouette_vs_k",
    "summarize_clusters_to_report",
]
