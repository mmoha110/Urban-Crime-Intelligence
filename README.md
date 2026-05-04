# Urban Crime Intelligence

End-to-end machine learning on the [City of Chicago Crimes dataset](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) (Socrata `ijzp-q8t2`). Built for **CS-483 Big Data Mining**: **supervised** arrest prediction and **unsupervised** geographic hotspot clustering, with reproducible pipelines and interactive maps.

## What you get

- **Classification:** Logistic Regression, Random Forest (with optional sigmoid calibration), and RBF SVM; `RandomizedSearchCV` for RF/LR; class weights, SMOTE on the training fold only, and stratified evaluation.
- **Clustering:** K-Means with silhouette-based `k` selection, multi-seed stability; DBSCAN (haversine) with grid search; K-Means vs DBSCAN comparison exports.
- **Evaluation:** Hold-out metrics, 5-fold stratified CV, chronological train-old / test-new validation, feature drift table.
- **Maps:** Folium heatmaps (all incidents, arrests-only), K-Means cluster map with centroids, filterable layer map (crime types + calibrated arrest-probability bins).

**Python:** 3.10+ recommended (tested on 3.12).

## Quick start

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Download data (writes `data/raw/chicago_crimes.csv`; CSVs are gitignored—clone with download step):

```bash
python -m src.data.download
```

Optional row cap (example: 50k):

```powershell
# PowerShell
$env:CRIME_DOWNLOAD_LIMIT="50000"; python -m src.data.download
```

```bash
# bash
CRIME_DOWNLOAD_LIMIT=50000 python -m src.data.download
```

Run the full pipeline (figures, CSV reports, maps; models written to `outputs/models/`):

```bash
python -m src.main --cv
```

Use `--no-tune-rf` / `--no-tune-lr` for faster iteration. Enable `--tune-svm` only if you accept long SVM tuning runs.

## Project layout

```text
urban-crime-intelligence/
├── data/
│   ├── raw/           # chicago_crimes.csv from Socrata (not in Git)
│   ├── processed/     # cleaned CSV from pipeline (not in Git)
│   └── samples/       # optional stratified samples (not in Git)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classification.ipynb
│   └── 04_clustering.ipynb
├── src/
│   ├── data/          # download, clean, features
│   ├── models/        # classification, clustering
│   ├── evaluation/    # metrics, temporal validation
│   └── visualization/ # Folium maps
├── outputs/
│   ├── figures/       # PNGs + *.html maps (tracked when present)
│   ├── models/        # *.joblib (gitignored; regenerate via main)
│   └── reports/       # metric CSVs + cluster_narrative.md
├── reports/           # placeholder for local course write-ups only (.gitkeep)
├── requirements.txt
└── README.md
```

The `reports/` directory is kept in the tree for local use (final papers, slide decks, scripts). Those files are **listed in `.gitignore`** and are not part of this public repository.

## Generated artifacts

Produced by `python -m src.main` (paths under `outputs/`).

### Figures (`outputs/figures/`)

| Artifact | Description |
| --- | --- |
| `model_comparison.png` | Accuracy, Precision, Recall, F1, ROC-AUC bars |
| `confusion_matrices.png` | Side-by-side confusion matrices |
| `roc_curves.png` | ROC curves per classifier |
| `feature_importance.png` | Random Forest importances |
| `silhouette_vs_k.png` | Silhouette and Davies-Bouldin vs `k` |
| `crime_heatmap.html` | Heatmap of all incidents |
| `arrest_rate_map.html` | Heatmap of arrests only |
| `cluster_map.html` | Sampled points + K-Means centroids |
| `filterable_intelligence_map.html` | Layer control: crime types + prob bins |

### Metric reports (`outputs/reports/`)

CSV/MD exports: `classification_results.csv`, `cv_results.csv`, `clustering_comparison.csv`, `kmeans_stability_*.csv`, `dbscan_tuning.csv`, `temporal_validation.csv`, `feature_drift_report.csv`, `yearly_arrest_rate.csv`, `cluster_narrative.md`.

### Models (`outputs/models/`)

`Logistic_Regression.joblib`, `Random_Forest.joblib`, `Random_Forest_Calibrated.joblib`, `SVM.joblib`, `scaler.joblib`, `encoders.joblib`. **These binaries are gitignored;** run the pipeline to regenerate them.

## CLI reference

| Flag | Default | Effect |
| --- | --- | --- |
| `--raw-path PATH` | `data/raw/chicago_crimes.csv` | Input CSV |
| `--no-tune-rf` | off | Skip RF `RandomizedSearchCV` |
| `--no-tune-lr` | off | Skip LR `RandomizedSearchCV` |
| `--tune-svm` | off | Enable SVM search (slow) |
| `--no-calibrate-rf` | off | Skip RF probability calibration |
| `--no-smote` | off | Skip SMOTE |
| `--no-filterable-map` | off | Skip filterable map |
| `--no-tune-dbscan` | off | Skip DBSCAN grid |
| `--cv` | off | Run 5-fold CV for classifiers |

## Core modules

| Module | Role |
| --- | --- |
| `src/data/download.py` | Socrata download |
| `src/data/clean.py` | Cleaning, sampling |
| `src/data/features.py` | Features + encoder persistence |
| `src/models/classification.py` | LR, RF, SVM, tuning, CV export |
| `src/models/clustering.py` | K-Means, DBSCAN, stability |
| `src/evaluation/metrics.py` | Plots, cluster narrative |
| `src/evaluation/temporal.py` | Temporal split, drift |
| `src/visualization/maps.py` | Folium maps |
| `src/main.py` | Orchestration |

## Design notes

- All classifiers use `class_weight="balanced"`.
- SMOTE applies only to the training fold (never the test set).
- Calibrated Random Forest (`CalibratedClassifierCV`, sigmoid) improves probability estimates for the filterable map.
- SVM training may subsample rows for tractability.
- `random_state=42` where applicable for reproducibility.

## Authors

Tejaswi Tiyyagura, Sanjith Jayasankar, Srinath Ganesh, Muneeb Mohammed (Northeastern CS-483).

## Repository hygiene

Course deliverables (final papers, slide decks, presentation scripts) are maintained **locally** by the team and excluded via `.gitignore`. If you previously committed any of those paths, stop tracking them without deleting your working copy:

```bash
git rm --cached reports/CS483_Final_Report.md reports/final_report.md reports/presentation_script.md reports/slides_outline.md outputs/reports/progress_report.md
git rm --cached reports/*.pdf reports/*.pptx 2>/dev/null || true
git commit -m "Stop tracking local course write-ups"
```

Then verify with `git status` before pushing.
