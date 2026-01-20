import os
from dataclasses import dataclass
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.utils.config import DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR, MODELS_DIR


@dataclass(frozen=True)
class PredictiveConfig:
    test_year: int = 2024
    random_state: int = 42


def _ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def _build_lag_features(features_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
    df = df.sort_values(["zip_code", "year"])

    # Next-year target for supervised learning (year t features -> year t+1 outcome)
    if "ev_percentage" in df.columns:
        df["target_next_ev_percentage"] = df.groupby("zip_code")["ev_percentage"].shift(-1)
        df["target_year"] = df["year"] + 1

    return df


def run_predictive_modeling(
    features_file: Optional[str] = None,
    output_metrics_csv: Optional[str] = None,
    config: Optional[PredictiveConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train simple models to predict EV adoption (regression + classification)."""
    _ensure_dirs()

    if config is None:
        config = PredictiveConfig()

    if features_file is None:
        features_file = os.path.join(DATA_PROCESSED, "sf_ev_features.csv")

    if output_metrics_csv is None:
        output_metrics_csv = os.path.join(RESULTS_DIR, "predictive_modeling_metrics.csv")

    print("=" * 60)
    print("Predictive Modeling")
    print("=" * 60)
    print(f"\n📄 Using features file: {features_file}")

    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Missing features file: {features_file}")

    features_df = pd.read_csv(features_file)
    df = _build_lag_features(features_df)

    # Predict next-year ev_percentage using current-year features
    candidate_features = [
        "ev_percentage",
        "ev_growth",
        "total_vehicles",
        "ev_count",
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]
    if not feature_cols:
        raise ValueError("No usable feature columns found in sf_ev_features.csv.")

    model_df = df.dropna(subset=["target_next_ev_percentage"]).copy()

    # Train on pairs where (year -> year+1) is strictly before the test year
    train_df = model_df[model_df["target_year"] < config.test_year].copy()
    test_df = model_df[model_df["target_year"] == config.test_year].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Need both train (<{config.test_year}) and test (=={config.test_year}) rows; "
            f"got train={len(train_df)} test={len(test_df)}"
        )

    # Drop any features that are completely missing in the training split
    usable_features = [c for c in feature_cols if not train_df[c].isna().all()]
    if not usable_features:
        raise ValueError("All candidate features are missing in the training split.")

    X_train = train_df[usable_features]
    y_train_reg = train_df["target_next_ev_percentage"].astype(float)

    X_test = test_df[usable_features]
    y_test_reg = test_df["target_next_ev_percentage"].astype(float)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, usable_features)],
        remainder="drop",
    )

    # Regression models
    ridge = Pipeline(steps=[("prep", preprocessor), ("model", Ridge(alpha=1.0))])
    rf_reg = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    random_state=config.random_state,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    reg_models = {"ridge": ridge, "random_forest": rf_reg}

    reg_rows = []
    best_reg_name = None
    best_reg_rmse = np.inf

    for name, model in reg_models.items():
        model.fit(X_train, y_train_reg)
        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test_reg, preds)))

        reg_rows.append(
            {
                "task": "regression",
                "model": name,
                "test_year": config.test_year,
                "rmse": rmse,
                "mae": float(mean_absolute_error(y_test_reg, preds)),
                "r2": float(r2_score(y_test_reg, preds)),
            }
        )

        if rmse < best_reg_rmse:
            best_reg_rmse = rmse
            best_reg_name = name

    reg_metrics = pd.DataFrame(reg_rows).sort_values("rmse")

    best_reg_model = reg_models[best_reg_name]
    best_reg_model.fit(X_train, y_train_reg)
    test_df = test_df.copy()
    test_df["pred_next_ev_percentage"] = best_reg_model.predict(X_test)

    model_path = os.path.join(MODELS_DIR, f"ev_percentage_regressor_{best_reg_name}.joblib")
    joblib.dump(best_reg_model, model_path)

    # Plot regression fit
    _plot_regression_predictions(test_df, config.test_year, best_reg_name)

    # Classification: high vs low EV adoption (based on median of TRAIN)
    threshold = float(np.median(y_train_reg))
    y_train_cls = (y_train_reg >= threshold).astype(int)
    y_test_cls = (y_test_reg >= threshold).astype(int)

    logreg = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    rf_cls = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=config.random_state,
                    min_samples_leaf=2,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    cls_models = {"logistic_regression": logreg, "random_forest": rf_cls}

    cls_rows = []
    best_cls_name = None
    best_cls_f1 = -np.inf

    for name, model in cls_models.items():
        model.fit(X_train, y_train_cls)
        preds = model.predict(X_test)

        # AUC (optional)
        auc = np.nan
        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test_cls, proba))
        except Exception:
            pass

        f1 = float(f1_score(y_test_cls, preds))
        cls_rows.append(
            {
                "task": "classification",
                "model": name,
                "test_year": config.test_year,
                "threshold_train_median_ev_pct": threshold,
                "accuracy": float(accuracy_score(y_test_cls, preds)),
                "f1": f1,
                "roc_auc": auc,
            }
        )

        if f1 > best_cls_f1:
            best_cls_f1 = f1
            best_cls_name = name

    # Cross-val sanity check on training data (small dataset)
    try:
        cv = StratifiedKFold(n_splits=min(5, int(y_train_cls.value_counts().min())), shuffle=True, random_state=config.random_state)
        cv_scores = cross_val_score(cls_models[best_cls_name], X_train, y_train_cls, cv=cv, scoring="f1")
        cv_f1 = float(np.mean(cv_scores))
    except Exception:
        cv_f1 = np.nan

    cls_metrics = pd.DataFrame(cls_rows).sort_values("f1", ascending=False)
    cls_metrics["cv_f1_train"] = cv_f1

    cls_model_path = os.path.join(MODELS_DIR, f"ev_high_classifier_{best_cls_name}.joblib")
    joblib.dump(cls_models[best_cls_name].fit(X_train, y_train_cls), cls_model_path)

    # Save metrics
    metrics = pd.concat([reg_metrics, cls_metrics], ignore_index=True)
    metrics.to_csv(output_metrics_csv, index=False)

    print(f"\n✅ Metrics saved to {output_metrics_csv}")
    print(f"✅ Regression model saved to {model_path}")
    print(f"✅ Classification model saved to {cls_model_path}")

    # Save per-zip predictions for demo
    pred_out = test_df[["zip_code", "year", "target_year", "target_next_ev_percentage", "pred_next_ev_percentage"]].copy()
    pred_path = os.path.join(RESULTS_DIR, "predictive_modeling_predictions.csv")
    pred_out.to_csv(pred_path, index=False)
    print(f"✅ Predictions saved to {pred_path}")

    return metrics, pred_out


def _plot_regression_predictions(test_df: pd.DataFrame, year: int, model_name: str) -> None:
    _ensure_dirs()

    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=test_df,
        x="target_next_ev_percentage",
        y="pred_next_ev_percentage",
        scatter_kws={"s": 80, "alpha": 0.7},
        line_kws={"color": "red"},
    )

    for _, r in test_df.iterrows():
        plt.text(
            r["target_next_ev_percentage"],
            r["pred_next_ev_percentage"],
            str(r["zip_code"]),
            fontsize=8,
            alpha=0.8,
        )

    plt.title(f"EV% prediction for {year} ({model_name})")
    plt.xlabel("Actual next-year EV%")
    plt.ylabel("Predicted next-year EV%")
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "predictive_modeling_regression.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📈 Figure saved to {fig_path}")


if __name__ == "__main__":
    run_predictive_modeling()
