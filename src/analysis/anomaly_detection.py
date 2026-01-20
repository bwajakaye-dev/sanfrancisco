import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

from src.utils.config import DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR


@dataclass(frozen=True)
class AnomalyConfig:
    year: int = 2024
    contamination: float = 0.15
    random_state: int = 42


def _ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _make_concentration_features(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    df = cleaned_df.copy()
    if "vehicles" not in df.columns:
        df["vehicles"] = 1

    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

    by_zip_total = df.groupby("zip_code")["vehicles"].sum().rename("total_vehicles_clean")
    make_zip = df.groupby(["zip_code", "make"])["vehicles"].sum()

    # Make shares
    make_share = make_zip / make_zip.groupby(level=0).sum()

    # Herfindahl-Hirschman Index (concentration)
    hhi = (make_share ** 2).groupby(level=0).sum().rename("make_hhi")

    # Top make share
    top_share = make_share.groupby(level=0).max().rename("top_make_share")

    return pd.concat([by_zip_total, hhi, top_share], axis=1).reset_index()


def run_anomaly_detection(
    features_file: Optional[str] = None,
    cleaned_file: Optional[str] = None,
    output_csv: Optional[str] = None,
    config: Optional[AnomalyConfig] = None,
) -> pd.DataFrame:
    """Detect anomalous ZIP codes in EV adoption features using Isolation Forest."""
    _ensure_dirs()

    if config is None:
        config = AnomalyConfig()

    if features_file is None:
        features_file = os.path.join(DATA_PROCESSED, "sf_ev_features.csv")

    if cleaned_file is None:
        cleaned_file = os.path.join(DATA_PROCESSED, f"sf_vehicles_{config.year}_cleaned.csv")

    if output_csv is None:
        output_csv = os.path.join(RESULTS_DIR, "anomaly_detection_results.csv")

    print("=" * 60)
    print("Anomaly Detection")
    print("=" * 60)

    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Missing features file: {features_file}")

    features = pd.read_csv(features_file)
    features["zip_code"] = features["zip_code"].astype(str).str.zfill(5)

    df_year = features[features["year"] == config.year].copy()
    if df_year.empty:
        raise ValueError(f"No rows found for year={config.year} in {features_file}")

    # Add concentration features from cleaned file
    if os.path.exists(cleaned_file):
        cleaned = pd.read_csv(cleaned_file, low_memory=False)
        conc = _make_concentration_features(cleaned)
        df_year = df_year.merge(conc, on="zip_code", how="left")
    else:
        df_year["make_hhi"] = np.nan
        df_year["top_make_share"] = np.nan
        df_year["total_vehicles_clean"] = np.nan

    # Feature set for anomaly detection
    candidate_cols = [
        "ev_percentage",
        "ev_growth",
        "total_vehicles",
        "make_hhi",
        "top_make_share",
    ]
    feature_cols = [c for c in candidate_cols if c in df_year.columns]

    X = df_year[feature_cols].copy()

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=500,
                    contamination=config.contamination,
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    pipe.fit(X)

    # sklearn: higher score_samples => more normal; invert for readability
    normality = pipe.named_steps["model"].score_samples(pipe.named_steps["scaler"].transform(pipe.named_steps["imputer"].transform(X)))
    anomaly_score = -normality

    df_year["anomaly_score"] = anomaly_score
    df_year["is_anomaly"] = pipe.predict(X) == -1

    out = df_year.sort_values("anomaly_score", ascending=False)
    out.to_csv(output_csv, index=False)

    print(f"\n✅ Anomaly results saved to {output_csv}")
    print(f"📌 Flagged anomalies: {int(out['is_anomaly'].sum())} / {len(out)}")

    _plot_anomalies(out, config.year)

    return out


def _plot_anomalies(df: pd.DataFrame, year: int) -> None:
    _ensure_dirs()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="ev_growth",
        y="ev_percentage",
        hue="is_anomaly",
        size="anomaly_score",
        palette={False: "#4C78A8", True: "#E45756"},
        sizes=(60, 300),
        alpha=0.85,
    )

    # Annotate anomalies
    for _, r in df[df["is_anomaly"]].iterrows():
        plt.text(r["ev_growth"], r["ev_percentage"], str(r["zip_code"]), fontsize=9)

    plt.title(f"Anomaly detection in EV adoption ({year})")
    plt.xlabel("EV growth (percentage point change)")
    plt.ylabel("EV percentage")
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "anomaly_detection.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📈 Figure saved to {fig_path}")


if __name__ == "__main__":
    run_anomaly_detection()
