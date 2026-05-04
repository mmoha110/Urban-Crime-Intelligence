"""End-to-end runner for cleaning, feature engineering, modeling, and reports."""

from __future__ import annotations

import argparse

import pandas as pd

from src.data.clean import clean, load_raw, save_processed
from src.data.features import engineer_features, get_feature_matrix
from src.evaluation.metrics import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_metrics_comparison,
    plot_roc_curves,
    plot_silhouette_vs_k,
    save_metrics_csv,
    summarize_clusters_to_report,
)
from src.evaluation.temporal import run_temporal_validation_and_drift
from src.models.classification import run_cross_validation_report, train_evaluate_all
from src.models.clustering import (
    analyze_clusters,
    compare_kmeans_dbscan,
    run_kmeans,
    tune_dbscan,
)
from src.visualization.maps import (
    make_arrest_rate_map,
    make_cluster_map,
    make_filterable_intelligence_map,
    make_heatmap,
)


def _print_final_summary(
    df_clean_shape,
    results,
    best_kmeans,
    cluster_stats,
    clustering_compare,
    cv_report,
    temporal_out,
    metadata,
):
    """Print a single consolidated summary at the end of the pipeline."""
    print("\n" + "=" * 72)
    print("URBAN CRIME INTELLIGENCE - PIPELINE SUMMARY")
    print("=" * 72)
    print(f"Cleaned dataset shape: {df_clean_shape}")

    print("\nClassification results (test set):")
    if results:
        metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train Time (s)"]
        header = f"  {'Model':<28}" + "".join(f"{c:>14}" for c in metric_cols)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, metrics in results.items():
            row = f"  {name:<28}"
            for c in metric_cols:
                v = metrics.get(c)
                if isinstance(v, float):
                    row += f"{v:>14.4f}"
                elif v is None:
                    row += f"{'n/a':>14}"
                else:
                    row += f"{str(v):>14}"
            print(row)

    if cv_report is not None and not cv_report.empty:
        print("\n5-fold Cross-Validation (mean ROC-AUC):")
        for _, row in cv_report.iterrows():
            print(
                f"  {row['model']:<28}"
                f"  roc_auc={row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f}"
                f"  f1={row['f1_mean']:.4f} +/- {row['f1_std']:.4f}"
            )

    print("\nClustering:")
    print(f"  Best K-Means k: {best_kmeans['k']}  (silhouette={best_kmeans['silhouette']:.4f})")
    print(f"  Total clusters produced: {cluster_stats.shape[0]}")
    if clustering_compare and "comparison" in clustering_compare:
        print("\n  KMeans vs DBSCAN comparison:")
        print(clustering_compare["comparison"].to_string(index=False))

    if temporal_out:
        tm = temporal_out.get("temporal_metrics")
        if tm is not None and not tm.empty:
            print("\nTemporal validation (train on older, test on newer):")
            print(tm.to_string(index=False))

    if metadata:
        print("\nTuning metadata:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")
    print("=" * 72 + "\n")


def run_pipeline(
    raw_path: str = "data/raw/chicago_crimes.csv",
    tune_rf: bool = True,
    tune_lr: bool = True,
    tune_svm: bool = False,
    calibrate_rf: bool = True,
    use_smote: bool = True,
    make_filterable_map: bool = True,
    run_cv: bool = False,
    tune_dbscan_flag: bool = True,
):
    print("Loading and cleaning data...")
    df_raw = load_raw(raw_path)
    df_clean = clean(df_raw)
    save_processed(df_clean)

    print("Engineering features...")
    df_feat = engineer_features(df_clean)
    X, y, feature_names = get_feature_matrix(df_feat)

    print("Training classification models...")
    cls_output = train_evaluate_all(
        X,
        y,
        tune_rf=tune_rf,
        tune_lr=tune_lr,
        tune_svm=tune_svm,
        calibrate_rf=calibrate_rf,
        use_smote=use_smote,
    )
    results = cls_output["results"]
    models = cls_output["models"]
    X_test = cls_output["X_test"]
    y_test = cls_output["y_test"]
    scaler = cls_output["scaler"]
    metadata = cls_output.get("metadata", {})

    save_metrics_csv(results)
    plot_metrics_comparison(results)
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)

    if "Random Forest" in models:
        plot_feature_importance(models["Random Forest"], feature_names)

    cv_report = None
    if run_cv:
        print("Running 5-fold cross-validation report...")
        cv_report = run_cross_validation_report(X, y)

    print("Running temporal validation and drift reports...")
    temporal_out = run_temporal_validation_and_drift(df_feat, feature_names)

    print("Running clustering analysis and DBSCAN comparison...")
    best_kmeans, kmeans_all_results = run_kmeans(df_clean)
    plot_silhouette_vs_k(kmeans_all_results)

    df_clustered, cluster_stats = analyze_clusters(df_clean, best_kmeans["labels"], name="KMeans")
    summarize_clusters_to_report(cluster_stats)

    best_dbscan = None
    if tune_dbscan_flag:
        print("Tuning DBSCAN over eps and min_samples grid...")
        dbscan_tuning = tune_dbscan(df_clean)
        best_dbscan = dbscan_tuning.get("best")

    clustering_compare = compare_kmeans_dbscan(
        df_clean, best_kmeans, best_dbscan=best_dbscan
    )

    print("Generating maps...")
    make_heatmap(df_clean)
    make_arrest_rate_map(df_clean)
    make_cluster_map(df_clustered, cluster_stats)

    if make_filterable_map:
        map_model = models.get("Random Forest (Calibrated)", models.get("Random Forest"))
        if map_model is not None:
            map_n = min(4000, len(df_clean))
            map_idx = df_clean.sample(n=map_n, random_state=42).index
            X_map = X.loc[map_idx]
            X_map_scaled = scaler.transform(X_map)
            arrest_prob = map_model.predict_proba(X_map_scaled)[:, 1]

            prob_df = pd.DataFrame(
                {
                    "Latitude": df_clean.loc[map_idx, "Latitude"].values,
                    "Longitude": df_clean.loc[map_idx, "Longitude"].values,
                    "Primary Type": df_clean.loc[map_idx, "Primary Type"].values,
                    "ArrestProb": arrest_prob,
                }
            )
            make_filterable_intelligence_map(df_clean, probability_df=prob_df)

    _print_final_summary(
        df_clean.shape,
        results,
        best_kmeans,
        cluster_stats,
        clustering_compare,
        cv_report,
        temporal_out,
        metadata,
    )

    return {
        "clean_shape": df_clean.shape,
        "classification_models": list(results.keys()),
        "best_k": best_kmeans["k"],
        "best_dbscan": best_dbscan,
        "n_clusters": int(cluster_stats.shape[0]),
        "temporal_report_paths": temporal_out["paths"],
        "clustering_report_paths": clustering_compare["paths"],
        "training_metadata": metadata,
        "cv_report_rows": None if cv_report is None else cv_report.to_dict(orient="records"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Urban Crime Intelligence pipeline")
    parser.add_argument("--raw-path", default="data/raw/chicago_crimes.csv", help="Path to raw crime CSV")
    parser.add_argument("--no-tune-rf", action="store_true", help="Skip RF hyperparameter tuning")
    parser.add_argument("--no-tune-lr", action="store_true", help="Skip LR hyperparameter tuning")
    parser.add_argument("--tune-svm", action="store_true", help="Enable SVM hyperparameter tuning (slow)")
    parser.add_argument("--no-calibrate-rf", action="store_true", help="Skip RF probability calibration")
    parser.add_argument("--no-smote", action="store_true", help="Skip SMOTE oversampling")
    parser.add_argument("--no-filterable-map", action="store_true", help="Skip filterable intelligence map")
    parser.add_argument("--cv", action="store_true", help="Run 5-fold cross-validation for all models")
    parser.add_argument("--no-tune-dbscan", action="store_true", help="Skip DBSCAN grid search")
    args = parser.parse_args()

    summary = run_pipeline(
        raw_path=args.raw_path,
        tune_rf=not args.no_tune_rf,
        tune_lr=not args.no_tune_lr,
        tune_svm=args.tune_svm,
        calibrate_rf=not args.no_calibrate_rf,
        use_smote=not args.no_smote,
        make_filterable_map=not args.no_filterable_map,
        run_cv=args.cv,
        tune_dbscan_flag=not args.no_tune_dbscan,
    )
    print("Pipeline summary:", summary)
