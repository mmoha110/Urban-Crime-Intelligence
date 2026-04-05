# Urban Crime Intelligence

CS483 Big Data Mining — University of Illinois Chicago

Predictive and descriptive analytics on the Chicago Crime Dataset using supervised classification (arrest prediction) and unsupervised clustering (geographic hotspot detection).

## Group Members

| Name | Role |
|---|---|
| Tejaswi Tiyyagura | Project Lead & Modeling |
| Sanjith Jayasankar | Data Engineer |
| Srinath Ganesh | Visualization & Analysis |
| Muneeb Mohammed | Model Evaluation & Optimization |

## Research Question

> Can spatio-temporal features significantly improve arrest prediction accuracy, and how do different clustering algorithms capture geographic crime concentration patterns?

## Tasks

| Task | Type | Algorithms |
|---|---|---|
| Task 1 — Arrest Prediction | Supervised Classification | Logistic Regression, Random Forest, SVM |
| Task 2 — Hotspot Detection | Unsupervised Clustering | K-Means, DBSCAN |

## Repository Structure

```
urban-crime-intelligence/
├── data/
│   ├── raw/           # Raw downloaded CSVs (not committed)
│   ├── processed/     # Cleaned & feature-engineered data (not committed)
│   └── samples/       # 100k-row stratified sample (not committed)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classification.ipynb
│   └── 04_clustering.ipynb
├── src/
│   ├── data/
│   │   ├── download.py     # Dataset download utility
│   │   ├── clean.py        # Preprocessing pipeline
│   │   └── features.py     # Feature engineering
│   ├── models/
│   │   ├── classification.py
│   │   └── clustering.py
│   ├── evaluation/
│   │   └── metrics.py      # Plots and metric tables
│   └── visualization/
│       └── maps.py         # Folium interactive maps
├── outputs/
│   ├── figures/   # Saved plots and HTML maps
│   ├── models/    # Serialised .joblib model files
│   └── reports/   # CSV metric summaries
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### 1. Download data
```bash
python -m src.data.download --limit 200000 --year_from 2018
```
Or manually download from the [Chicago Open Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) and place the CSV in `data/raw/chicago_crimes.csv`.

### 2. Clean and sample
```bash
python -m src.data.clean --sample 100000
```

### 3. Feature engineering
```bash
python -m src.data.features
```

### 4. Run classification (Task 1)
```bash
python -m src.models.classification
```

### 5. Run clustering (Task 2)
```bash
python -m src.models.clustering
```

### 6. Generate maps
```bash
python -m src.visualization.maps
```

### 7. Notebooks (recommended)
Open Jupyter and run the notebooks in order:
```bash
jupyter notebook notebooks/
```

## Dataset

- **Source:** [Chicago Open Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
- **Records:** 2001–present, millions of rows, ~22 attributes
- **Target:** `Arrest` (Boolean → 0/1)
- **Key features:** Date/time, primary crime type, location description, district, community area, latitude, longitude, domestic flag

## Key Features Engineered

| Feature | Description |
|---|---|
| `Hour` | Hour of day (0–23) |
| `DayOfWeek` | Day of week (0=Mon) |
| `Month` | Month (1–12) |
| `IsWeekend` | 1 if Saturday/Sunday |
| `IsNight` | 1 if hour ≤5 or ≥22 |
| `Season` | Spring/Summer/Fall/Winter |
| `PrimaryType_enc` | Label-encoded crime type |
| `LocationDesc_enc` | Label-encoded location description |
| `Domestic_enc` | 1 if domestic incident |

## Evaluation Metrics

**Classification:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

**Clustering:** Silhouette Score, Davies-Bouldin Index, visual inspection

## Expected Results

| Model | Expected Accuracy | Expected F1 |
|---|---|---|
| Logistic Regression | 68–72% | 0.55–0.65 |
| Random Forest | 78–85% | 0.70–0.78 |
| SVM | 72–78% | 0.60–0.70 |

## Notes

- `class_weight='balanced'` is set for all classifiers to handle the ~25–30% arrest-rate imbalance
- SVM training is limited to 30k rows due to O(n²) complexity
- All random operations use `random_state=42` for reproducibility
- DBSCAN uses the haversine metric on radians-converted coordinates
