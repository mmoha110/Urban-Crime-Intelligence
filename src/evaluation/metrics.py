"""
Evaluation utilities: plots and metric tables for classification and clustering.

All plotting functions save figures to outputs/figures/ by default and also
call plt.show() so they render inline in Jupyter notebooks.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay, roc_curve, auc,
    RocCurveDisplay,
)


# ---------------------------------------------------------------------------
# Classification plots
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "outputs/figures",
) -> None:
    """
    Side-by-side confusion matrices for every model in `models`.

    Args:
        models: Dict of {model_name: fitted_estimator}.
        X_test: Scaled test features.
        y_test: True labels.
        output_dir: Directory for saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            display_labels=["No Arrest", "Arrest"],
            ax=ax, colorbar=False,
        )
        ax.set_title(name, fontsize=11)

    plt.suptitle("Confusion Matrices", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


def plot_roc_curves(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "outputs/figures",
) -> None:
    """
    Overlaid ROC curves for all models that support predict_proba.

    Args:
        models: Dict of {model_name: fitted_estimator}.
        X_test: Scaled test features.
        y_test: True labels.
        output_dir: Directory for saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name}  (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — All Models", fontsize=13)
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved → {path}")


def plot_metrics_comparison(
    results_dict: dict,
    output_dir: str = "outputs/figures",
) -> None:
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC.

    Args:
        results_dict: Output of classification.train_evaluate_all().
        output_dir: Directory for saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    rows = []
    for name, m in results_dict.items():
        row = {col: m.get(col) for col in metric_cols}
        row["Model"] = name
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")[metric_cols]
    ax = df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05), width=0.75)
    ax.set_title("Model Performance Comparison", fontsize=13)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_xticklabels(df.index, rotation=15, ha="right")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved → {path}")


def plot_feature_importance(
    rf_model,
    feature_names: list[str],
    top_n: int = 15,
    output_dir: str = "outputs/figures",
) -> None:
    """
    Horizontal bar chart of the top-n Random Forest feature importances.

    Args:
        rf_model: Fitted RandomForestClassifier.
        feature_names: List of feature column names.
        top_n: Number of features to display.
        output_dir: Directory for saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    importances = rf_model.feature_importances_
    indices = importances.argsort()[::-1][:top_n]
    names   = [feature_names[i] for i in indices]
    values  = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), values[::-1], align="center")
    plt.yticks(range(top_n), names[::-1])
    plt.xlabel("Feature Importance (Gini)", fontsize=12)
    plt.title(f"Top {top_n} Feature Importances — Random Forest", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Clustering plots
# ---------------------------------------------------------------------------

def plot_elbow_curve(
    elbow_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
) -> None:
    """
    Elbow curve (inertia vs k) with silhouette scores on a secondary axis.

    Args:
        elbow_df: DataFrame with columns [k, inertia, silhouette].
        output_dir: Directory for saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_inertia = "#1f77b4"
    ax1.plot(elbow_df["k"], elbow_df["inertia"], "o-",
             color=color_inertia, label="Inertia", lw=2)
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertia", color=color_inertia, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_inertia)

    ax2 = ax1.twinx()
    color_sil = "#ff7f0e"
    ax2.plot(elbow_df["k"], elbow_df["silhouette"], "s--",
             color=color_sil, label="Silhouette", lw=2)
    ax2.set_ylabel("Silhouette Score", color=color_sil, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_sil)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("K-Means Elbow Curve", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "kmeans_elbow.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved → {path}")


def plot_cluster_scatter(
    df_labeled: pd.DataFrame,
    cluster_col: str = "Cluster",
    title: str = "Crime Clusters",
    output_dir: str = "outputs/figures",
    filename: str = "cluster_scatter.png",
    max_points: int = 50000,
) -> None:
    """
    Scatter plot of lat/lon coloured by cluster label.

    Args:
        df_labeled: DataFrame with Latitude, Longitude, and cluster_col columns.
        cluster_col: Name of the cluster label column.
        title: Plot title.
        output_dir: Directory for saved PNG.
        filename: Output filename.
        max_points: Downsample to this many points for fast rendering.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_plot = df_labeled[df_labeled[cluster_col] != -1]
    if len(df_plot) > max_points:
        df_plot = df_plot.sample(max_points, random_state=42)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        df_plot["Longitude"], df_plot["Latitude"],
        c=df_plot[cluster_col], cmap="tab20",
        s=1, alpha=0.4,
    )
    plt.colorbar(scatter, label="Cluster ID")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title, fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Metric saving
# ---------------------------------------------------------------------------

def save_metrics_csv(
    results_dict: dict,
    path: str = "outputs/reports/classification_results.csv",
) -> None:
    """Save classification metrics to CSV (excludes model objects)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for name, m in results_dict.items():
        row = {k: v for k, v in m.items() if k != "model"}
        row["Model"] = name
        rows.append(row)
    pd.DataFrame(rows).set_index("Model").to_csv(path)
    print(f"Metrics saved → {path}")
