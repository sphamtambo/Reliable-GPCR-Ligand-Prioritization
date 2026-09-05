"""
Core Question 2 analysis: dataset properties versus model performance.

This script uses saved outputs only. It joins target/pool dataset descriptors
with combined-representation nested-CV performance from benchmark_results_long,
computes Spearman correlations, and creates a publication-ready scatter plot for
structural cleanliness versus regression performance.
"""
from pathlib import Path
import os

import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "figures"
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

TARGET_COLORS = {
    "adora2a": "#1f77b4",
    "cb2": "#2ca02c",
    "ccr5": "#d62728",
    "drd2": "#9467bd",
    "oprm1": "#ff7f0e",
}

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_combined_nested_cv_performance() -> pd.DataFrame:
    bench = pd.read_csv(ROOT / "ml/results/benchmark_results_long.csv")
    auc = (
        bench[
            (bench["feature_representation"] == "combined")
            & (bench["task"] == "classification")
            & (bench["metric_name"] == "roc_auc")
        ]
        .groupby(["target", "activity_pool"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "roc_auc"})
    )
    r2 = (
        bench[
            (bench["feature_representation"] == "combined")
            & (bench["task"] == "regression")
            & (bench["metric_name"] == "r2")
        ]
        .groupby(["target", "activity_pool"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "r2"})
    )
    return auc.merge(r2, on=["target", "activity_pool"], how="inner")


def load_dataset_predictors() -> pd.DataFrame:
    cols = [
        "target",
        "activity_pool",
        "final_unique_compounds",
        "active_pct",
        "scaffold_richness",
        "any_alert_free_pct",
        "pActivity_sd",
    ]
    predictors = pd.read_csv(ROOT / "data/processed/dataset_summary_all_targets_pools.csv")[cols]
    predictors["class_balance_deviation"] = (predictors["active_pct"] - 50).abs()
    return predictors


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        ("Dataset size (compound count)", "final_unique_compounds"),
        ("Class balance deviation", "class_balance_deviation"),
        ("Scaffold diversity", "scaffold_richness"),
        ("PAINS/Brenk-free %", "any_alert_free_pct"),
        ("pActivity spread", "pActivity_sd"),
    ]
    rows = []
    for label, col in predictors:
        row = {"predictor": label}
        for metric in ["roc_auc", "r2"]:
            sub = df[[col, metric]].dropna()
            rho, p_value = spearmanr(sub[col], sub[metric])
            row[f"rho_vs_{metric}"] = rho
            row[f"p_vs_{metric}"] = p_value
            row[f"n_vs_{metric}"] = len(sub)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_cleanliness_vs_r2(df: pd.DataFrame) -> None:
    rho, p_value = spearmanr(df["any_alert_free_pct"], df["r2"])
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for target, group in df.groupby("target"):
        ax.scatter(
            group["any_alert_free_pct"],
            group["r2"],
            s=70,
            color=TARGET_COLORS.get(target, "0.35"),
            edgecolors="white",
            linewidth=0.7,
            label=target.upper(),
            zorder=3,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["activity_pool"],
                (row["any_alert_free_pct"], row["r2"]),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=8,
                color="0.25",
            )

    ax.text(
        0.03,
        0.05,
        f"Spearman rho = {rho:.3f}\np = {p_value:.3g}",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.85", "alpha": 0.9},
    )
    ax.set_xlabel("Compounds free of PAINS/Brenk alerts (%)")
    ax.set_ylabel("Mean nested-CV R2 (combined representation)")
    ax.set_xlim(max(0, df["any_alert_free_pct"].min() - 5), min(100, df["any_alert_free_pct"].max() + 5))
    ax.set_ylim(max(0, df["r2"].min() - 0.08), min(0.8, df["r2"].max() + 0.08))
    ax.grid(True, color="0.9", linewidth=0.7, zorder=0)
    ax.legend(title="Target", frameon=True, loc="lower right")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"figure_structural_cleanliness_vs_r2.{ext}")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_combined_nested_cv_performance().merge(
        load_dataset_predictors(), on=["target", "activity_pool"], how="left"
    )
    df.to_csv(OUT / "figure_structural_cleanliness_vs_r2_data.csv", index=False)
    corr = correlation_table(df)
    corr.to_csv(ROOT / "data/processed/core_question_2_performance_predictor_correlations.csv", index=False)
    plot_cleanliness_vs_r2(df)
    print(corr.round(4).to_string(index=False))
    print("wrote figure_structural_cleanliness_vs_r2.png/.pdf")


if __name__ == "__main__":
    main()
