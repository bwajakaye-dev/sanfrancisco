import os
import textwrap
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

from src.utils.config import DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR


@dataclass(frozen=True)
class AssocRuleConfig:
    year: int = 2024
    min_support: float = 0.2
    min_confidence: float = 0.6
    min_lift: float = 1.2
    top_makes: int = 12
    make_share_threshold: float = 0.12
    permutations: int = 2000
    random_state: int = 42
    fdr_alpha: float = 0.10


def _ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return BH-FDR adjusted q-values."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p

    order = np.argsort(p)
    ranked = p[order]

    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    out = np.empty_like(q)
    out[order] = q
    return out


def _discretize_ev_share(ev_share: pd.Series) -> pd.Series:
    # Robust quantile cut: fall back to fixed bins if quantiles collapse.
    try:
        return pd.qcut(ev_share, q=3, labels=["ev_share_low", "ev_share_mid", "ev_share_high"], duplicates="drop")
    except ValueError:
        return pd.cut(ev_share, bins=[-np.inf, 5, 10, np.inf], labels=["ev_share_low", "ev_share_mid", "ev_share_high"])


def _make_transaction_matrix(cleaned_df: pd.DataFrame, config: AssocRuleConfig) -> pd.DataFrame:
    df = cleaned_df.copy()

    if "vehicles" not in df.columns:
        df["vehicles"] = 1

    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

    # Aggregate by zip
    by_zip_total = df.groupby("zip_code")["vehicles"].sum().rename("total_vehicles")
    by_zip_ev = df.assign(ev_vehicles=df["vehicles"] * df.get("is_ev", 0)).groupby("zip_code")["ev_vehicles"].sum()
    zip_summary = pd.concat([by_zip_total, by_zip_ev], axis=1).fillna(0)
    zip_summary["ev_share"] = np.where(
        zip_summary["total_vehicles"] > 0,
        zip_summary["ev_vehicles"] / zip_summary["total_vehicles"],
        0.0,
    )

    # Item: discretized EV share
    zip_summary["ev_share_bucket"] = _discretize_ev_share(zip_summary["ev_share"])

    # Top makes overall (weighted by vehicles)
    make_totals = df.groupby("make")["vehicles"].sum().sort_values(ascending=False)
    top_makes = make_totals.head(config.top_makes).index.tolist()

    # Per-zip make share for top makes
    make_by_zip = (
        df[df["make"].isin(top_makes)]
        .groupby(["zip_code", "make"])["vehicles"].sum()
        .unstack(fill_value=0)
        .reindex(columns=top_makes, fill_value=0)
    )
    make_share = make_by_zip.div(by_zip_total, axis=0).fillna(0)

    # Duty distribution (optional)
    duty_items = pd.DataFrame(index=zip_summary.index)
    if "duty" in df.columns:
        duty_by_zip = df.groupby(["zip_code", "duty"])["vehicles"].sum().unstack(fill_value=0)
        if "Light" in duty_by_zip.columns:
            duty_items["duty_light_majority"] = (duty_by_zip["Light"] / by_zip_total).fillna(0) >= 0.70
        if "Heavy" in duty_by_zip.columns:
            duty_items["duty_heavy_present"] = duty_by_zip["Heavy"].fillna(0) > 0

    # Build boolean item matrix
    items = pd.DataFrame(index=zip_summary.index)

    # EV share bucket items
    for label in ["ev_share_low", "ev_share_mid", "ev_share_high"]:
        items[label] = zip_summary["ev_share_bucket"].astype(str).eq(label)

    # Make share items
    for make in top_makes:
        col = f"make_{str(make).strip().upper()}_high"
        items[col] = make_share[make] >= config.make_share_threshold

    # Add duty items if any
    for col in duty_items.columns:
        items[col] = duty_items[col].astype(bool)

    # Clean: ensure boolean dtype
    items = items.fillna(False).astype(bool)

    # Drop all-false columns to avoid apriori warnings
    items = items.loc[:, items.any(axis=0)]
    return items


def _rule_lift_permutation_pvalue(
    transactions: pd.DataFrame,
    antecedent_cols: Iterable[str],
    consequent_cols: Iterable[str],
    observed_lift: float,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    # For simplicity: permutation test over (single) consequent item by shuffling its column.
    # If multi-item consequent, approximate by shuffling the AND vector.
    ant_cols = list(antecedent_cols)
    con_cols = list(consequent_cols)

    ant_mask = transactions[ant_cols].all(axis=1).to_numpy(dtype=int)
    con_mask = transactions[con_cols].all(axis=1).to_numpy(dtype=int)

    n = ant_mask.size
    p_x = ant_mask.mean()
    p_y = con_mask.mean()

    if p_x == 0 or p_y == 0:
        return 1.0

    xy_obs = (ant_mask & con_mask).mean()
    lift_obs = xy_obs / (p_x * p_y)

    # If rounding/format mismatch, use computed observed (still close)
    if not np.isfinite(observed_lift):
        observed_lift = lift_obs

    # Permute Y while holding X fixed
    perm_lifts = np.empty(permutations, dtype=float)
    idx = np.arange(n)

    for i in range(permutations):
        rng.shuffle(idx)
        y_perm = con_mask[idx]
        xy = (ant_mask & y_perm).mean()
        perm_lifts[i] = xy / (p_x * p_y)

    # One-sided p-value: probability of lift >= observed
    p_val = (np.sum(perm_lifts >= observed_lift) + 1.0) / (permutations + 1.0)
    return float(p_val)


def run_association_rule_discovery(
    cleaned_file: Optional[str] = None,
    output_csv: Optional[str] = None,
    config: Optional[AssocRuleConfig] = None,
) -> pd.DataFrame:
    """Mine association rules on ZIP-level transactions derived from the cleaned DMV dataset."""
    _ensure_dirs()

    if config is None:
        config = AssocRuleConfig()

    if cleaned_file is None:
        cleaned_file = os.path.join(DATA_PROCESSED, f"sf_vehicles_{config.year}_cleaned.csv")

    if output_csv is None:
        output_csv = os.path.join(RESULTS_DIR, "association_rules.csv")

    print("=" * 60)
    print("Association Rule Discovery")
    print("=" * 60)
    print(f"\n📄 Using cleaned file: {cleaned_file}")

    if not os.path.exists(cleaned_file):
        raise FileNotFoundError(f"Missing cleaned DMV data: {cleaned_file}")

    cleaned_df = pd.read_csv(cleaned_file, low_memory=False)
    transactions = _make_transaction_matrix(cleaned_df, config)

    print(f"\n🧺 Transactions: {transactions.shape[0]} ZIP codes")
    print(f"   Items: {transactions.shape[1]}")

    if transactions.shape[0] < 5 or transactions.shape[1] < 2:
        raise ValueError("Not enough transactions/items to mine rules.")

    frequent = apriori(transactions, min_support=config.min_support, use_colnames=True)
    if frequent.empty:
        raise ValueError("No frequent itemsets found. Try lowering min_support.")

    rules = association_rules(frequent, metric="confidence", min_threshold=config.min_confidence)
    if rules.empty:
        raise ValueError("No association rules found. Try lowering min_confidence/min_support.")

    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)

    # Basic filtering
    rules = rules[(rules["lift"] >= config.min_lift)].copy()
    if rules.empty:
        raise ValueError("Rules exist, but none meet the min_lift filter.")

    # Permutation p-values and BH-FDR
    rng = np.random.default_rng(config.random_state)
    pvals = []

    # Compute p-values for top N rules to keep runtime tight.
    # Still outputs all rules, but only these get p-values.
    max_rules_for_test = min(len(rules), 200)

    for i, row in rules.head(max_rules_for_test).iterrows():
        p = _rule_lift_permutation_pvalue(
            transactions=transactions,
            antecedent_cols=list(row["antecedents"]),
            consequent_cols=list(row["consequents"]),
            observed_lift=float(row["lift"]),
            permutations=config.permutations,
            rng=rng,
        )
        pvals.append((i, p))

    rules["p_value_lift_perm"] = np.nan
    for idx, p in pvals:
        rules.loc[idx, "p_value_lift_perm"] = p

    tested = rules["p_value_lift_perm"].notna()
    rules.loc[tested, "q_value_bh"] = _benjamini_hochberg(rules.loc[tested, "p_value_lift_perm"].to_numpy())

    # Add pretty strings
    rules["antecedent"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequent"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))

    cols = [
        "antecedent",
        "consequent",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
        "p_value_lift_perm",
        "q_value_bh",
    ]
    out = rules[cols].sort_values(["lift", "confidence"], ascending=False)

    # Save CSV
    out.to_csv(output_csv, index=False)
    print(f"\n✅ Association rules saved to {output_csv}")

    # Plot top rules
    _plot_top_rules(out)

    # Optional: recommend demo-ready subset
    demo_subset = out.copy()
    if "q_value_bh" in demo_subset.columns:
        demo_subset = demo_subset[(demo_subset["q_value_bh"].isna()) | (demo_subset["q_value_bh"] <= config.fdr_alpha)]

    print(f"\n📌 Rules meeting lift>={config.min_lift}: {len(out)}")
    if "q_value_bh" in out.columns:
        print(f"📌 Rules meeting BH-FDR q<={config.fdr_alpha} (tested subset): {len(demo_subset)}")

    return out


def _plot_top_rules(rules_df: pd.DataFrame, top_n: int = 15) -> None:
    if rules_df.empty:
        return

    _ensure_dirs()

    top = rules_df.head(top_n).copy().reset_index(drop=True)
    top["rule"] = top["antecedent"] + " → " + top["consequent"]

    # Clean, readable chart: horizontal bars by lift with wrapped labels.
    wrap_width = 55
    top["rule_wrapped"] = top["rule"].apply(lambda s: textwrap.fill(str(s), width=wrap_width, break_long_words=False))

    # Plot highest lift at top
    top = top.sort_values("lift", ascending=True)

    fig_height = max(6.0, 0.55 * len(top))
    fig, ax = plt.subplots(figsize=(14, fig_height))

    sns.barplot(
        data=top,
        x="lift",
        y="rule_wrapped",
        color="#4C78A8",
        ax=ax,
    )

    ax.set_title("Top Association Rules (ranked by lift)")
    ax.set_xlabel("Lift")
    ax.set_ylabel("")

    # Add confidence/support as compact annotations at bar ends
    for i, r in enumerate(top.itertuples(index=False)):
        ax.text(
            float(r.lift) + 0.02,
            i,
            f"conf={float(r.confidence):.2f}, supp={float(r.support):.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    ax.grid(axis="x", linestyle=":", alpha=0.35)
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "association_rules_top.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📈 Figure saved to {fig_path}")


if __name__ == "__main__":
    run_association_rule_discovery()
