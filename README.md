# San Francisco Data Mining Project

Data Mining course project using San Francisco open data (https://data.sfgov.org/) plus CA DMV vehicle registration aggregates.

## Aim (problem statement)

San Francisco’s EV adoption is not uniform: some areas are already high-adoption and fast-growing, while others lag behind despite having many vehicles. This project’s aim is to support **equitable EV infrastructure planning** by answering:

- Which SF areas (at ZIP-code resolution) cluster together by EV adoption level, vehicle scale, and recent growth?
- What multi-attribute patterns co-occur with higher EV adoption (with safeguards against false discoveries)?
- Can we forecast next-year ZIP-level EV adoption from prior-year features?
- Which ZIP codes look anomalous (possible data quality issues or “something new”)?

**Intended stakeholder / user:** city planners or advocacy groups prioritizing EV charging deployment, outreach, or incentives; residents comparing adoption trends where they live.

## Deployment architecture (batch analytics)

This project is designed as a reproducible batch pipeline:

1) **Ingest**: pull SF Open Data via Socrata API + download CA DMV fuel-type counts from CA Open Data (data.ca.gov / CKAN) and cache them as CSV.
2) **Store**: keep raw/cleaned/feature data locally as CSVs (lightweight “data-store”).
3) **Analyze**: clustering, association rules (with permutation + BH-FDR), predictive modeling, anomaly detection.
4) **Deliver**: write artifacts to `results/` (CSVs, figures, and saved models) for demo or for downstream dashboards.

```
SF Open Data + CA DMV CSVs
	|
	v
scripts/run_full_pipeline.py
	|
	+--> src/preprocessing/*  -> src/data_collection/data/processed/*.csv
	|
	+--> src/analysis/*       -> results/*.csv, results/figures/*.png, results/models/*.joblib
```

Operationally, you could run this as a scheduled job (e.g., weekly) and publish the `results/` artifacts to a shared drive or a simple dashboard.

This repo implements four core data mining techniques required by the assignment:
- **Clustering analysis**: cluster SF ZIP codes by EV adoption metrics.
- **Association rule discovery**: mine patterns across ZIP-level attributes derived from DMV data (with permutation testing + BH-FDR to reduce false discoveries).
- **Predictive modeling**: predict next-year EV adoption from prior-year features (regression + classification).
- **Anomaly detection**: flag unusual ZIP codes based on EV adoption and vehicle-mix concentration.

## Data sources
- **CA DMV vehicle fuel type / registration aggregates** (downloaded from CA Open Data and cached locally as CSV; filtered to SF ZIP codes).
- **SF Open Data** via Socrata API (example used: Parking Meters).

Downloaded + processed datasets live under:
- `src/data_collection/data/raw/`
- `src/data_collection/data/processed/`

SF Open Data used in this repo:
- Parking Meters (dataset ID `8vzz-qzz9`): https://data.sfgov.org/resource/8vzz-qzz9.json

CA DMV data used in this repo:
- Vehicle Fuel Type Count by Zip Code (CA Open Data / CKAN): https://data.ca.gov/dataset/vehicle-fuel-type-count-by-zip-code

## Quickstart

1) Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Run the end-to-end pipeline:

```bash
python scripts/run_full_pipeline.py
```

## How to use

### Run everything (recommended)

This runs: downloads → cleaning → feature engineering → clustering → association rules → predictive modeling → anomaly detection.

```bash
python scripts/run_full_pipeline.py
```

### Run individual analyses (faster for demo iteration)

If you already have the processed files under `src/data_collection/data/processed/`, you can run specific steps directly.

Clustering:
```bash
python -c "from src.analysis.clustering import perform_clustering_analysis; perform_clustering_analysis()"
```

Association rules:
```bash
python -c "from src.analysis.association_rules import run_association_rule_discovery; run_association_rule_discovery()"
```

Predictive modeling:
```bash
python -c "from src.analysis.predictive_modeling import run_predictive_modeling; run_predictive_modeling()"
```

Anomaly detection:
```bash
python -c "from src.analysis.anomaly_detection import run_anomaly_detection; run_anomaly_detection()"
```

### Customize parameters (optional)

Example: lower support/confidence thresholds for association rules.

```bash
python -c "from src.analysis.association_rules import run_association_rule_discovery, AssocRuleConfig; run_association_rule_discovery(config=AssocRuleConfig(min_support=0.10, min_confidence=0.50, min_lift=1.10))"
```

### Where inputs come from

- Raw CSVs are stored in `src/data_collection/data/raw/`.
- The pipeline downloads CA DMV CSVs from the CA Open Data URL above and caches them as `ca_dmv_vehicle_<year>.csv`.
- Cleaned/engineered datasets used by the analyses are stored in `src/data_collection/data/processed/`.
- All deliverables for the demo are written to `results/`.

### Generate a one-page findings summary

After running the pipeline, generate a demo-ready summary report from the produced CSVs:

```bash
python scripts/summarize_headlines.py
```

This writes `results/headline_summary.md`.

## Findings (headline results)

See `results/headline_summary.md` for an auto-generated table-based summary. Highlights from the latest run:

- **Top EV-adoption cluster (2024):** 94107 (Potrero Hill / Mission Bay / SoMa), 94105 (Financial District / South Beach / Mission Bay), 94158 (Mission Bay / Dogpatch), 94114 (Castro / Noe Valley), 94131 (Noe Valley / Glen Park), 94127 (West Portal / Twin Peaks).
- **Fastest EV growth (2024):** 94105, 94123 (Marina / Cow Hollow), 94158, 94127, 94114.
- **Flagged anomalies (2024):** 94105 (high EV% + high growth), 94104 (negative growth), 94112 (very high vehicle count but low EV%), 94130 (extreme make concentration), 94128 (SFO; low volume / boundary caveat).
- **Predictive modeling:** next-year EV% prediction achieves **R² ≈ 0.72** on the held-out year (see `results/predictive_modeling_metrics.csv`).

## What the pipeline produces

Key outputs (for the demo / grading):
- `results/clustering_results.csv`
- `results/figures/clustering_analysis.png`

- `results/association_rules.csv`
- `results/figures/association_rules_top.png`

- `results/predictive_modeling_metrics.csv`
- `results/predictive_modeling_predictions.csv`
- `results/figures/predictive_modeling_regression.png`
- `results/models/*.joblib`

- `results/anomaly_detection_results.csv`
- `results/figures/anomaly_detection.png`

## Notes for the assignment demo

Suggested 3–5 minute walkthrough:
1) Show the **data sources** and where the raw/processed files are.
2) Run `python scripts/run_full_pipeline.py` (or show recent outputs).
3) Explain **one key finding** from each technique (clusters, rules, model, anomalies).
4) Point to the generated CSVs/figures under `results/`.

## Limitations and validity notes

- **ZIP codes are not neighborhoods** and overlap multiple areas; ZIP→area labels are approximate.
- **Aggregated DMV counts** are not individual-level behavior; results should be interpreted as area-level patterns.
- **Association rules can produce false discoveries**; this project mitigates that via permutation p-values and BH-FDR q-values.
- **Data coverage caveat:** 94128 (SFO) is not part of most City & County SF boundary analyses and can behave like a low-volume outlier.

## Repo structure
- `src/data_collection/`: downloads (CA DMV CSV ingestion + SF Socrata API example)
- `src/preprocessing/`: cleaning + feature engineering
- `src/analysis/`: clustering, association rules, predictive modeling, anomaly detection
- `scripts/`: runnable pipeline entrypoints

## Documentation

- One-page summary output: `results/headline_summary.md` (generated)
