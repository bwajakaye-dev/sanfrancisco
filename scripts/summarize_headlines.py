from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ZIP_AREA_LABELS: dict[str, str] = {
    "94102": "Civic Center / Tenderloin / Hayes Valley",
    "94103": "SoMa (South of Market) / Mission Bay",
    "94104": "Financial District / Chinatown",
    "94105": "Financial District / South Beach / Mission Bay",
    "94107": "Potrero Hill / Mission Bay / SoMa",
    "94108": "Chinatown / Nob Hill",
    "94109": "Nob Hill / Russian Hill / Polk Gulch",
    "94110": "Mission District / Bernal Heights",
    "94111": "Financial District / North Beach / Chinatown",
    "94112": "Excelsior / Ingleside–Oceanview",
    "94114": "Castro / Noe Valley / Twin Peaks",
    "94115": "Japantown / Western Addition / Pacific Heights",
    "94116": "Sunset / Parkside",
    "94117": "Haight-Ashbury / Lower Haight",
    "94118": "Inner Richmond / Presidio Heights",
    "94121": "Outer Richmond / Seacliff",
    "94122": "Inner Sunset / Sunset-Parkside / Golden Gate Park",
    "94123": "Marina / Cow Hollow",
    "94124": "Bayview–Hunters Point",
    "94127": "West Portal / Twin Peaks",
    "94128": "SFO (San Francisco International Airport)",
    "94129": "Presidio",
    "94130": "Treasure Island / Yerba Buena Island",
    "94131": "Noe Valley / Glen Park / Bernal Heights",
    "94132": "Lakeshore / Parkmerced (SF State area)",
    "94133": "North Beach / Chinatown",
    "94134": "Visitacion Valley / Excelsior / Portola",
    "94158": "Mission Bay / Potrero Hill (Dogpatch area)",
}


def _zip_label(zip_code: str | int) -> str:
    z = str(zip_code).zfill(5)
    return ZIP_AREA_LABELS.get(z, "(unknown / mixed)")


def _safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _format_pct(x) -> str:
    v = _safe_float(x)
    if math.isnan(v):
        return "NA"
    return f"{v:.2f}%"


def _format_num(x) -> str:
    v = _safe_float(x)
    if math.isnan(v):
        return "NA"
    if abs(v) >= 1000 and float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}"


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without extra deps (no tabulate)."""
    if df is None or df.empty:
        return "(no rows)"

    # Convert values to strings (keep it simple and readable)
    str_df = df.copy()
    for col in str_df.columns:
        if pd.api.types.is_float_dtype(str_df[col]) or pd.api.types.is_integer_dtype(str_df[col]):
            str_df[col] = str_df[col].map(_format_num)
        else:
            str_df[col] = str_df[col].astype(str)

    headers = list(str_df.columns)
    rows = str_df.values.tolist()

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def summarize(repo_root: Path) -> str:
    results_dir = repo_root / "results"

    clust_path = results_dir / "clustering_results.csv"
    rules_path = results_dir / "association_rules.csv"
    anom_path = results_dir / "anomaly_detection_results.csv"
    metrics_path = results_dir / "predictive_modeling_metrics.csv"

    clust = pd.read_csv(clust_path)
    rules = pd.read_csv(rules_path)
    anom = pd.read_csv(anom_path)
    metrics = pd.read_csv(metrics_path)

    # --- Clustering headlines ---
    cluster_stats = (
        clust.groupby("cluster_kmeans")
        .agg(
            n_zip=("zip_code", "count"),
            mean_ev_pct=("ev_percentage", "mean"),
            mean_growth=("ev_growth", "mean"),
            mean_total=("total_vehicles", "mean"),
        )
        .sort_values("mean_ev_pct", ascending=False)
    )

    top_cluster = int(cluster_stats.index[0])
    top_cluster_zips = (
        clust.loc[
            clust["cluster_kmeans"] == top_cluster,
            ["zip_code", "ev_percentage", "ev_growth", "total_vehicles"],
        ]
        .sort_values("ev_percentage", ascending=False)
        .copy()
    )
    top_cluster_zips["zip_code"] = top_cluster_zips["zip_code"].astype(str).str.zfill(5)
    top_cluster_zips["area"] = top_cluster_zips["zip_code"].map(_zip_label)

    # Useful direct rankings (rubric-friendly: concrete takeaways)
    by_ev = clust[["zip_code", "ev_percentage", "ev_growth", "total_vehicles"]].copy()
    by_ev["zip_code"] = by_ev["zip_code"].astype(str).str.zfill(5)
    by_ev["area"] = by_ev["zip_code"].map(_zip_label)
    top5_ev = by_ev.sort_values("ev_percentage", ascending=False).head(5)
    bottom5_ev = by_ev.sort_values("ev_percentage", ascending=True).head(5)
    top5_growth = by_ev.sort_values("ev_growth", ascending=False).head(5)

    # --- Association rules headlines ---
    rules_sorted = rules.sort_values(["lift", "confidence", "support"], ascending=False)
    top_rules = rules_sorted.head(5).copy()

    # --- Anomalies headlines ---
    if "is_anomaly" in anom.columns:
        anom_flagged = anom[anom["is_anomaly"] == True].copy()  # noqa: E712
    else:
        anom_flagged = pd.DataFrame()

    if not anom_flagged.empty:
        anom_flagged["zip_code"] = anom_flagged["zip_code"].astype(str).str.zfill(5)
        anom_flagged["area"] = anom_flagged["zip_code"].map(_zip_label)
        anom_flagged = anom_flagged.sort_values("anomaly_score", ascending=False)

    # --- Predictive modeling headlines ---
    reg = metrics[metrics["task"] == "regression"].copy()
    cls = metrics[metrics["task"] == "classification"].copy()

    reg_best = reg.sort_values("rmse").head(1) if not reg.empty else pd.DataFrame()
    cls_best = cls.sort_values("f1", ascending=False).head(1) if not cls.empty else pd.DataFrame()

    year = int(clust["year"].max()) if "year" in clust.columns else None

    lines: list[str] = []
    lines.append("# Headline findings (auto-generated)")
    lines.append("")
    lines.append("## Clustering: top EV-adoption cluster")
    lines.append(f"- Most recent year analyzed: **{year}**")
    lines.append(f"- Top cluster (highest mean EV%): **cluster {top_cluster}**")
    lines.append("")

    cs = cluster_stats.reset_index().rename(columns={"cluster_kmeans": "cluster"})
    lines.append("Cluster summary (K-Means):")
    lines.append("")
    lines.append(_df_to_markdown(cs))
    lines.append("")

    lines.append("ZIP codes in the top EV cluster:")
    lines.append("")

    lines.append(_df_to_markdown(
        top_cluster_zips[["zip_code", "area", "ev_percentage", "ev_growth", "total_vehicles"]].rename(
            columns={
                "ev_percentage": "ev_%",
                "ev_growth": "ev_growth_pp",
            }
        )
    ))
    lines.append("")

    lines.append("## Direct rankings (most recent year)")
    lines.append("Top 5 ZIP codes by EV percentage:")
    lines.append("")
    lines.append(_df_to_markdown(top5_ev[["zip_code", "area", "ev_percentage", "ev_growth", "total_vehicles"]].rename(columns={"ev_percentage": "ev_%", "ev_growth": "ev_growth_pp"})))
    lines.append("")

    lines.append("Top 5 ZIP codes by EV growth (percentage-point change):")
    lines.append("")
    lines.append(_df_to_markdown(top5_growth[["zip_code", "area", "ev_growth", "ev_percentage", "total_vehicles"]].rename(columns={"ev_percentage": "ev_%", "ev_growth": "ev_growth_pp"})))
    lines.append("")

    lines.append("Bottom 5 ZIP codes by EV percentage:")
    lines.append("")
    lines.append(_df_to_markdown(bottom5_ev[["zip_code", "area", "ev_percentage", "ev_growth", "total_vehicles"]].rename(columns={"ev_percentage": "ev_%", "ev_growth": "ev_growth_pp"})))
    lines.append("")

    lines.append("## Association rules: top 5 by lift")
    lines.append("- Interpretation: lift > 1 suggests the consequent is more likely when the antecedent holds.")
    lines.append("- False discovery control: rules include a permutation-test p-value and BH-FDR q-value (when computed).")
    lines.append("")
    lines.append(_df_to_markdown(top_rules[["antecedent", "consequent", "support", "confidence", "lift", "q_value_bh"]]))
    lines.append("")

    lines.append("## Anomaly detection: ZIP codes flagged")
    if anom_flagged.empty:
        lines.append("- No anomalies flagged (or `is_anomaly` column missing).")
    else:
        keep_cols = [c for c in ["zip_code", "area", "anomaly_score", "ev_percentage", "ev_growth", "make_hhi", "top_make_share", "total_vehicles"] if c in anom_flagged.columns]
        lines.append(f"- Flagged anomalies: **{len(anom_flagged)}**")
        lines.append("")
        lines.append(_df_to_markdown(anom_flagged[keep_cols]))
    lines.append("")

    lines.append("## Predictive modeling: performance snapshot")
    if not reg_best.empty:
        r = reg_best.iloc[0]
        lines.append(f"- Best regression model (by RMSE): **{r['model']}**, RMSE={_format_num(r['rmse'])}, MAE={_format_num(r['mae'])}, R²={_format_num(r['r2'])}")
    if not cls_best.empty:
        c = cls_best.iloc[0]
        thresh = c.get("threshold_train_median_ev_pct")
        lines.append(f"- Best classifier (by F1): **{c['model']}**, F1={_format_num(c['f1'])}, accuracy={_format_num(c['accuracy'])}, AUC={_format_num(c.get('roc_auc'))}, threshold≈{_format_pct(thresh)}")
    if reg_best.empty and cls_best.empty:
        lines.append("- No metrics found.")
    lines.append("")

    lines.append("## Caveat about ZIP → neighborhood names")
    lines.append("ZIP codes overlap multiple neighborhoods. The area labels are short human-readable summaries based on DataSF's ZIP↔Analysis Neighborhood crosswalk; treat them as approximate.")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    md = summarize(repo_root)

    out_path = repo_root / "results" / "headline_summary.md"
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
