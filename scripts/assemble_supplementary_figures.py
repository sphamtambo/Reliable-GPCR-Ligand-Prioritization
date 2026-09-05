"""
Assemble supplementary-information (SI) figures S1-S10 from persisted source data.

This script does not rerun any analysis. It reads frozen CSV outputs (the same
outputs already used for the main-text figures and the SI tables) and writes
polished PNG/PDF figures to outputs/figures/supplementary/. Each figure gets a
JSON manifest with source paths and SHA-256 hashes, matching the convention in
scripts/assemble_main_text_figures.py. This script is intentionally
self-contained (its own copy of the small set of style constants/helpers)
rather than importing that module, so running it never triggers the six
main-text figures as a side effect.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / ".matplotlib-cache").mkdir(exist_ok=True)
(ROOT / ".cache").mkdir(exist_ok=True)

TARGET_COLORS = {
    "drd2": "#4c78a8",
    "cb2": "#59a14f",
    "adora2a": "#f28e2b",
    "oprm1": "#b07aa1",
    "ccr5": "#e15759",
}
TARGET_ORDER = ["drd2", "cb2", "adora2a", "oprm1", "ccr5"]
TARGET_LABELS = {"drd2": "DRD2", "cb2": "CB2", "adora2a": "ADORA2A", "oprm1": "OPRM1", "ccr5": "CCR5"}
GREY = "#9aa0a6"
DARK = "#2f3437"
PANEL_LETTER_SIZE = 18

mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelweight": "bold",
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def norm_target(x):
    return str(x).strip().lower()


def target_labels(keys):
    return [TARGET_LABELS[k] for k in keys]


def target_colors(keys):
    return [TARGET_COLORS[k] for k in keys]


def save(fig, stem):
    outputs = []
    for ext in ("png", "pdf"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    for path in outputs:
        print(f"wrote {path}")
    return outputs


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(stem, sources, outputs, notes):
    def records(paths):
        recs = []
        for p in paths:
            p = Path(p)
            recs.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)})
        return recs

    manifest = {
        "figure": stem,
        "deterministic": True,
        "analysis_rerun": False,
        "sources": records(sources),
        "outputs": records(outputs),
        "notes": notes,
    }
    path = OUT / f"{stem}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {path}")


def panel_label(ax, label):
    text_fn = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_fn(
        -0.12, 1.08, label, transform=ax.transAxes, fontsize=PANEL_LETTER_SIZE,
        fontweight="bold", va="bottom", ha="right", clip_on=False,
    )


def source(rel):
    return ROOT / rel


# --------------------------------------------------------------------------
# Figure S1: Dataset curation and endpoint-distribution diagnostics
# --------------------------------------------------------------------------
def figure_s1():
    ds_path = source("data/processed/dataset_summary_all_targets_pools.csv")
    ec_path = source("data/processed/endpoint_composition_summary.csv")
    pact_path = source("data/processed/pactivity_distribution_plot_data.csv")

    ds = pd.read_csv(ds_path)
    ds["target"] = ds["target"].map(norm_target)
    ds_full = ds[ds["activity_pool"] == "full"].set_index("target")

    ec = pd.read_csv(ec_path)
    ec["target"] = ec["target"].map(norm_target)
    ec_full = ec[ec["activity_pool"] == "full"].set_index("target")

    pact = pd.read_csv(pact_path)
    pact["target"] = pact["target"].map(norm_target)
    pact_full = pact[pact["activity_pool"] == "full"]

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6))
    xs = np.arange(len(TARGET_ORDER))

    ax = axes[0]
    w = 0.36
    raw = [ds_full.loc[t, "raw_records"] for t in TARGET_ORDER]
    final = [ds_full.loc[t, "final_unique_compounds"] for t in TARGET_ORDER]
    ax.bar(xs - w / 2, raw, width=w, color=GREY, label="Raw ChEMBL records")
    ax.bar(xs + w / 2, final, width=w, color=target_colors(TARGET_ORDER), label="Curated compounds")
    ax.set_xticks(xs)
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_ylabel("Count")
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    panel_label(ax, "A")

    ax = axes[1]
    bottoms = np.zeros(len(TARGET_ORDER))
    parts = [
        ("pct_retained_ki_measurements", "Ki", "#4c78a8"),
        ("pct_retained_ic50_measurements", "IC50", "#f28e2b"),
        ("pct_retained_ec50_measurements", "EC50", "#59a14f"),
    ]
    for col, label, color in parts:
        vals = np.array([ec_full.loc[t, col] for t in TARGET_ORDER], dtype=float)
        ax.bar(xs, vals, bottom=bottoms, width=0.62, color=color, label=label)
        bottoms += vals
    ax.set_xticks(xs)
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_ylabel("Retained measurements (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    panel_label(ax, "B")

    ax = axes[2]
    data = [pact_full.loc[pact_full["target"] == t, "pActivity"].values for t in TARGET_ORDER]
    parts_v = ax.violinplot(data, showmedians=True, showextrema=False)
    for i, body in enumerate(parts_v["bodies"]):
        body.set_facecolor(TARGET_COLORS[TARGET_ORDER[i]])
        body.set_alpha(0.75)
    parts_v["cmedians"].set_color(DARK)
    ax.set_xticks(np.arange(1, len(TARGET_ORDER) + 1))
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.axhline(6.0, color=DARK, linestyle="--", linewidth=0.8)
    ax.text(0.55, 6.05, "active/inactive threshold", fontsize=6.5, color=DARK)
    ax.set_ylabel("pActivity (full pool)")
    panel_label(ax, "C")

    fig.tight_layout()
    stem = "figure_s1_dataset_curation_endpoint_diagnostics"
    outs = save(fig, stem)
    write_manifest(
        stem,
        [ds_path, ec_path, pact_path],
        outs,
        [
            "Panel A compares raw retrieved ChEMBL records with final curated compound counts for the deployed full endpoint pool, per target.",
            "Panel B shows the percentage of retained measurements contributed by Ki, IC50, and EC50 assay types per target, full pool.",
            "Panel C shows the pActivity distribution per target for the full pool; the dashed line marks the pActivity=6.0 active/inactive threshold used for classification labels.",
        ],
    )


# --------------------------------------------------------------------------
# Figure S2: Chemical-space, scaffold, and structural-alert diagnostics
# --------------------------------------------------------------------------
def figure_s2():
    pca_path = source("data/processed/chemical_space_pca_coordinates.csv")
    var_path = source("data/processed/chemical_space_pca_variance.csv")
    scaf_path = source("data/processed/scaffold_diversity_summary.csv")
    ds_path = source("data/processed/dataset_summary_all_targets_pools.csv")

    pca = pd.read_csv(pca_path)
    pca["target"] = pca["target"].map(norm_target)
    var = pd.read_csv(var_path).set_index("component")["explained_variance_ratio"]

    scaf = pd.read_csv(scaf_path)
    scaf["target"] = scaf["target"].map(norm_target)
    scaf_full = scaf[scaf["activity_pool"] == "full"].set_index("target")

    ds = pd.read_csv(ds_path)
    ds["target"] = ds["target"].map(norm_target)
    ds_full = ds[ds["activity_pool"] == "full"].set_index("target")

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6))
    rng = np.random.default_rng(42)

    ax = axes[0]
    for t in TARGET_ORDER:
        sub = pca[pca["target"] == t]
        if len(sub) > 3000:
            idx = rng.choice(sub.index, size=3000, replace=False)
            sub = sub.loc[idx]
        ax.scatter(sub["PC1"], sub["PC2"], s=3, alpha=0.25, color=TARGET_COLORS[t], label=TARGET_LABELS[t], linewidths=0)
    ax.set_xlabel(f"PC1 ({var.get('PC1', np.nan) * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var.get('PC2', np.nan) * 100:.1f}% variance)")
    leg = ax.legend(frameon=False, loc="best", fontsize=7, markerscale=3)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    panel_label(ax, "A")

    ax = axes[1]
    xs = np.arange(len(TARGET_ORDER))
    richness = [scaf_full.loc[t, "scaffold_richness"] for t in TARGET_ORDER]
    ax.bar(xs, richness, width=0.55, color=target_colors(TARGET_ORDER))
    ax.set_xticks(xs)
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_ylabel("Scaffold richness (scaffolds / compounds)")
    ax2 = ax.twinx()
    singleton = [scaf_full.loc[t, "pct_singleton_scaffolds"] for t in TARGET_ORDER]
    ax2.plot(xs, singleton, marker="o", color=DARK, linewidth=1.2, markersize=5)
    ax2.set_ylabel("Singleton scaffolds (%)", color=DARK)
    ax2.spines["right"].set_visible(True)
    panel_label(ax, "B")

    ax = axes[2]
    w = 0.26
    cats = [("pains_free_pct", "PAINS-free", "#4c78a8"),
            ("brenk_free_pct", "Brenk-free", "#f28e2b"),
            ("any_alert_free_pct", "No alert", "#59a14f")]
    for i, (col, label, color) in enumerate(cats):
        vals = [ds_full.loc[t, col] for t in TARGET_ORDER]
        ax.bar(xs + (i - 1) * w, vals, width=w, color=color, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_ylabel("Compounds without alert (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    panel_label(ax, "C")

    fig.tight_layout()
    stem = "figure_s2_chemical_space_scaffold_alert_diagnostics"
    outs = save(fig, stem)
    write_manifest(
        stem,
        [pca_path, var_path, scaf_path, ds_path],
        outs,
        [
            "Panel A shows a PCA projection of physicochemical descriptor space (MW, LogP, TPSA, HBD, HBA, RotBonds, HeavyAtomCount, RingCount, AromaticRingCount, FractionCSP3), coloured by target; each target series is randomly subsampled to at most 3,000 compounds for rendering only, not for any reported statistic.",
            "Panel B shows Bemis-Murcko scaffold richness (bars, left axis) and the percentage of singleton scaffolds (line, right axis) per target, full pool.",
            "Panel C shows the percentage of compounds free of PAINS alerts, Brenk alerts, and any structural alert, per target, full pool.",
        ],
    )


# --------------------------------------------------------------------------
# Figure S3: Full internal benchmark grid across algorithms, endpoint pools,
# and molecular representations
# --------------------------------------------------------------------------
def figure_s3():
    path = source("ml/results/benchmark_results_long.csv")
    df = pd.read_csv(path)
    df["target"] = df["target"].map(norm_target)
    roc = df[(df["metric_name"] == "roc_auc") & (df["task"] == "classification")]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))
    xs = np.arange(len(TARGET_ORDER))

    ax = axes[0]
    algos = ["Random Forest", "XGBoost", "LightGBM"]
    algo_colors = {"Random Forest": "#4c78a8", "XGBoost": "#f28e2b", "LightGBM": "#59a14f"}
    sub = roc[(roc["activity_pool"] == "full") & (roc["feature_representation"] == "combined")]
    w = 0.26
    for i, algo in enumerate(algos):
        means = [sub[(sub["target"] == t) & (sub["algorithm"] == algo)]["metric_value"].mean() for t in TARGET_ORDER]
        ax.bar(xs + (i - 1) * w, means, width=w, color=algo_colors[algo], label=algo)
    ax.set_xticks(xs)
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_ylabel("Mean ROC-AUC (full pool, combined)")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    panel_label(ax, "A")

    ax = axes[1]
    pools = ["ki", "ki_ic50", "full"]
    pool_labels = {"ki": "Ki only", "ki_ic50": "Ki+IC50", "full": "Full"}
    sub = roc[roc["feature_representation"] == "combined"]
    for i, pool in enumerate(pools):
        means = [sub[(sub["target"] == t) & (sub["activity_pool"] == pool)]["metric_value"].mean() for t in TARGET_ORDER]
        ax.bar(xs + (i - 1) * w, means, width=w, color=["#a8c5e0", "#5b7c99", "#1f4e79"][i], label=pool_labels[pool])
    ax.set_xticks(xs)
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_ylabel("Mean ROC-AUC (combined, mean over algorithms)")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    panel_label(ax, "B")

    ax = axes[2]
    reps = ["morgan", "descriptors", "combined"]
    rep_labels = {"morgan": "Morgan", "descriptors": "Descriptors", "combined": "Combined"}
    sub = roc[roc["activity_pool"] == "full"]
    for i, rep in enumerate(reps):
        means = [sub[(sub["target"] == t) & (sub["feature_representation"] == rep)]["metric_value"].mean() for t in TARGET_ORDER]
        ax.bar(xs + (i - 1) * w, means, width=w, color=["#b07aa1", "#e15759", "#59a14f"][i], label=rep_labels[rep])
    ax.set_xticks(xs)
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_ylabel("Mean ROC-AUC (full pool, mean over algorithms)")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    panel_label(ax, "C")

    fig.tight_layout()
    stem = "figure_s3_full_internal_benchmark_grid"
    outs = save(fig, stem)
    write_manifest(
        stem,
        [path],
        outs,
        [
            "Panel A compares the three tested algorithm families at the deployed full pool and combined representation; the deployed algorithm for each target is one bar among the three shown, not a separately highlighted alternative.",
            "Panel B compares the three endpoint pools at the combined representation, averaged over all three algorithm families.",
            "Panel C compares the three feature representations at the full pool, averaged over all three algorithm families; this differs from the main-text representation comparison, which uses only the deployed algorithm per target.",
            "All values are outer-fold mean ROC-AUC from the nested scaffold-disjoint benchmark grid (ml/results/benchmark_results_long.csv).",
        ],
    )


# --------------------------------------------------------------------------
# Figure S4: Endpoint-pool sensitivity and model-selection diagnostics
# --------------------------------------------------------------------------
def figure_s4():
    sens_path = source("ml/results/activity_pool_sensitivity_summary.csv")
    conv_path = source("ml/results/optuna_convergence.csv")

    sens = pd.read_csv(sens_path)
    sens["target"] = sens["target"].map(norm_target)

    conv = pd.read_csv(conv_path)
    conv_cls = conv[conv["task"] == "classification"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    xs = np.arange(len(TARGET_ORDER))
    pools = ["ki", "ki_ic50", "full"]
    pool_labels = {"ki": "Ki only", "ki_ic50": "Ki+IC50", "full": "Full"}
    pool_colors = ["#a8c5e0", "#5b7c99", "#1f4e79"]
    w = 0.26

    ax = axes[0]
    for i, pool in enumerate(pools):
        sub = sens[sens["activity_pool"] == pool].set_index("target")
        means = [sub.loc[t, "mean_roc_auc"] if t in sub.index else np.nan for t in TARGET_ORDER]
        errs = [sub.loc[t, "std_roc_auc"] if t in sub.index else np.nan for t in TARGET_ORDER]
        ax.bar(xs + (i - 1) * w, means, width=w, yerr=errs, capsize=2, color=pool_colors[i], label=pool_labels[pool])
    ax.set_xticks(xs)
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_ylabel("Mean ROC-AUC (best algorithm per pool)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    panel_label(ax, "A")

    ax = axes[1]
    algo_colors = {"Random Forest": "#4c78a8", "XGBoost": "#f28e2b", "LightGBM": "#59a14f"}
    for algo, color in algo_colors.items():
        sub = conv_cls[conv_cls["algorithm"] == algo].sort_values("trial")
        ax.plot(sub["trial"], sub["mean"], color=color, label=algo, linewidth=1.3)
        ax.fill_between(sub["trial"], sub["mean"] - sub["std"], sub["mean"] + sub["std"], color=color, alpha=0.15)
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("Best-so-far objective (mean ± SD across combinations)")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    panel_label(ax, "B")

    fig.tight_layout()
    stem = "figure_s4_endpoint_pool_sensitivity_model_selection"
    outs = save(fig, stem)
    write_manifest(
        stem,
        [sens_path, conv_path],
        outs,
        [
            "Panel A shows mean classification ROC-AUC (with SD across outer folds) for the best algorithm at each endpoint pool, per target.",
            "Panel B shows Optuna hyperparameter-optimization convergence for the classification task, averaged across all target/pool combinations tuned per algorithm, with the 50-trial equal budget applied uniformly across targets.",
        ],
    )


# --------------------------------------------------------------------------
# Figure S5: DrugBank screening diagnostics
# --------------------------------------------------------------------------
def figure_s5():
    summary_path = source("ml/results/drugbank_screening/screening_summary_by_combination.csv")
    primary_path = source("ml/results/drugbank_screening/drugbank_primary_ranked_candidates.csv")

    summary = pd.read_csv(summary_path)
    summary["target"] = summary["target"].map(norm_target)
    summary_full = summary[
        (summary["activity_pool"] == "full") & (summary["feature_representation"] == "combined")
    ].set_index("target")

    primary = pd.read_csv(primary_path)
    primary["target"] = primary["target"].map(norm_target)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    xs = np.arange(len(TARGET_ORDER))
    w = 0.36

    ax = axes[0]
    hc = [summary_full.loc[t, "n_high_confidence"] if t in summary_full.index else 0 for t in TARGET_ORDER]
    nc = [summary_full.loc[t, "n_novel_candidates"] if t in summary_full.index else 0 for t in TARGET_ORDER]
    ax.bar(xs - w / 2, hc, width=w, color=GREY, label="High-confidence predictions")
    ax.bar(xs + w / 2, nc, width=w, color=target_colors(TARGET_ORDER), label="Target-novel predictions")
    ax.set_xticks(xs)
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_ylabel("Count (full pool, combined representation)")
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    panel_label(ax, "A")

    ax = axes[1]
    for t in TARGET_ORDER:
        sub = primary[primary["target"] == t]
        if sub.empty:
            continue
        inside = sub[sub["within_applicability_domain"].astype(bool)]
        outside = sub[~sub["within_applicability_domain"].astype(bool)]
        ax.scatter(inside["applicability_domain_distance"], inside["predicted_proba_calibrated"],
                   s=18, color=TARGET_COLORS[t], label=TARGET_LABELS[t], edgecolors="none", alpha=0.85)
        ax.scatter(outside["applicability_domain_distance"], outside["predicted_proba_calibrated"],
                   s=28, facecolors="none", edgecolors=TARGET_COLORS[t], linewidths=1.1)
    ax.set_xlabel("Applicability-domain distance")
    ax.set_ylabel("Calibrated active probability")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    panel_label(ax, "B")

    fig.tight_layout()
    stem = "figure_s5_drugbank_screening_diagnostics"
    outs = save(fig, stem)
    write_manifest(
        stem,
        [summary_path, primary_path],
        outs,
        [
            "Panel A compares the full high-confidence screening population with the smaller target-novel subset that enters docking-entry decisions, per target, full pool and combined representation; CCR5 had zero target-novel predictions.",
            "Panel B shows calibrated active probability against applicability-domain distance for the 92 primary (full pool, combined representation, deployed algorithm) target-novel candidates; open markers denote candidates outside the applicability domain.",
        ],
    )


# --------------------------------------------------------------------------
# Figure S6: Calibration and conformal reliability detail
# --------------------------------------------------------------------------
def figure_s6():
    calib_path = source("ml/results/meta_analysis/calibration_across_targets.csv")
    conf_path = source("ml/results/meta_analysis/conformal_coverage_across_targets.csv")

    calib = pd.read_csv(calib_path)
    calib["target_key"] = calib["target_key"].map(norm_target)
    calib = calib.set_index("target_key")

    conf = pd.read_csv(conf_path)
    conf["target"] = conf["target"].map(norm_target)
    conf_reg = conf[(conf["task"] == "regression") & (conf["nominal_level"] == 0.90)
                    & (conf["stratum"].isin(["within_ad", "outside_ad"]))]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))

    def dumbbell(ax, raw_col, calib_col, title):
        for i, t in enumerate(TARGET_ORDER):
            raw_v = calib.loc[t, raw_col]
            cal_v = calib.loc[t, calib_col]
            ax.plot([raw_v, cal_v], [i, i], color=GREY, linewidth=1.4, zorder=1)
            ax.scatter([raw_v], [i], color="#c9c9c9", edgecolors=DARK, s=45, zorder=2, label="Raw" if i == 0 else None)
            ax.scatter([cal_v], [i], color=TARGET_COLORS[t], edgecolors=DARK, s=45, zorder=2,
                       label="Venn-Abers calibrated" if i == 0 else None)
        ax.set_yticks(range(len(TARGET_ORDER)))
        ax.set_yticklabels(target_labels(TARGET_ORDER))
        ax.set_xlabel(title)
        ax.legend(frameon=False, loc="best", fontsize=6.5)

    dumbbell(axes[0], "brier_raw", "brier_calibrated", "Brier score")
    panel_label(axes[0], "A")
    dumbbell(axes[1], "ece_raw", "ece_calibrated", "Expected calibration error")
    panel_label(axes[1], "B")

    ax = axes[2]
    xs = np.arange(len(TARGET_ORDER))
    w = 0.32
    for i, stratum in enumerate(["within_ad", "outside_ad"]):
        vals = []
        for t in TARGET_ORDER:
            row = conf_reg[(conf_reg["target"] == t) & (conf_reg["stratum"] == stratum)]
            vals.append(row["empirical_coverage"].iloc[0] if len(row) else np.nan)
        color = "#a8c5e0" if stratum == "within_ad" else "#1f4e79"
        label = "Within AD" if stratum == "within_ad" else "Outside AD"
        ax.bar(xs + (i - 0.5) * w, vals, width=w, color=color, label=label)
    ax.axhline(0.90, color=DARK, linestyle="--", linewidth=0.9)
    ax.text(-0.45, 0.905, "nominal 90%", fontsize=6.5, color=DARK)
    ax.set_xticks(xs)
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_ylabel("Empirical regression conformal coverage")
    ax.set_ylim(0.6, 1.0)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    panel_label(ax, "C")

    fig.tight_layout()
    stem = "figure_s6_calibration_conformal_reliability_detail"
    outs = save(fig, stem)
    write_manifest(
        stem,
        [calib_path, conf_path],
        outs,
        [
            "Panels A and B show raw versus Venn-Abers-calibrated Brier score and expected calibration error per target; these are scalar summary metrics and are shown separately from the conformal-coverage and applicability-domain evidence that carries the main reliability claim.",
            "Panel C shows empirical regression conformal coverage at nominal 90% inside versus outside the applicability domain, per target; the dashed line marks the nominal level. This is the split-conformal regression axis reported in the main text (coverage degrades outside the domain for all five targets); interval width itself is fixed by construction and is not shown here.",
        ],
    )


# --------------------------------------------------------------------------
# Figure S7: Learning-curve and scaffold-stratified performance diagnostics
# --------------------------------------------------------------------------
def figure_s7():
    lc_path = source("ml/results/learning_curves.csv")
    scaf_path = source("ml/results/scaffold_performance_breakdown.csv")

    lc = pd.read_csv(lc_path)
    lc["target"] = lc["target"].map(norm_target)
    lc_full = lc[lc["activity_pool"] == "full"]

    scaf = pd.read_csv(scaf_path)
    scaf["target"] = scaf["target"].map(norm_target)
    scaf_sub = scaf[
        (scaf["activity_pool"] == "full") & (scaf["feature_representation"] == "combined")
        & (scaf["scaffold_stratum"].isin(["singleton_scaffold", "non_singleton_scaffold"]))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    ax = axes[0]
    for t in TARGET_ORDER:
        sub = lc_full[lc_full["target"] == t].sort_values("train_fraction")
        ax.plot(sub["train_fraction"], sub["test_roc_auc"], marker="o", markersize=3.5,
                color=TARGET_COLORS[t], label=TARGET_LABELS[t], linewidth=1.3)
    ax.set_xlabel("Training-set fraction used")
    ax.set_ylabel("Held-out test ROC-AUC (full pool)")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    panel_label(ax, "A")

    ax = axes[1]
    xs = np.arange(len(TARGET_ORDER))
    w = 0.34
    for i, stratum in enumerate(["singleton_scaffold", "non_singleton_scaffold"]):
        vals = []
        for t in TARGET_ORDER:
            row = scaf_sub[(scaf_sub["target"] == t) & (scaf_sub["scaffold_stratum"] == stratum)]
            vals.append(row["classification_roc_auc"].iloc[0] if len(row) else np.nan)
        color = "#a8c5e0" if stratum == "singleton_scaffold" else "#1f4e79"
        label = "Singleton scaffolds" if stratum == "singleton_scaffold" else "Non-singleton scaffolds"
        ax.bar(xs + (i - 0.5) * w, vals, width=w, color=color, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_ylabel("Classification ROC-AUC (full pool, combined)")
    ax.set_ylim(0.6, 1.0)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    panel_label(ax, "B")

    fig.tight_layout()
    stem = "figure_s7_learning_curve_scaffold_stratified_diagnostics"
    outs = save(fig, stem)
    write_manifest(
        stem,
        [lc_path, scaf_path],
        outs,
        [
            "Panel A shows learning curves (test ROC-AUC vs. training-set fraction) per target for the full pool at the deployed algorithm.",
            "Panel B compares classification ROC-AUC for singleton versus non-singleton scaffold strata, full pool, combined representation, per target.",
        ],
    )


# --------------------------------------------------------------------------
# Figure S8: Per-target explainability detail
# --------------------------------------------------------------------------
def figure_s8():
    heat_path = source("ml/results/meta_analysis/shap_descriptor_heatmap_combined.csv")
    shares_path = source("ml/results/meta_analysis/shap_feature_shares_long.csv")

    heat = pd.read_csv(heat_path).set_index("feature")
    heat = heat[[TARGET_LABELS[t] for t in TARGET_ORDER]]

    shares = pd.read_csv(shares_path)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    data = heat.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f2f2f2")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(len(TARGET_ORDER)))
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                        color="white" if data[i, j] > np.nanmax(data) * 0.55 else DARK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean |SHAP| (combined representation)")
    panel_label(ax, "A")

    ax = axes[1]
    order = ["descriptor", "Morgan bit"]
    box_data = [shares[shares["feature_type"] == ft]["share_of_top_features"].dropna().values for ft in order]
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5, showfliers=False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Descriptor", "Morgan bit"])
    for patch, color in zip(bp["boxes"], ["#4c78a8", "#e15759"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, vals in enumerate(box_data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=6, color=DARK, alpha=0.35, zorder=3)
    ax.set_ylabel("Share of top-20 SHAP importance")
    panel_label(ax, "B")

    fig.tight_layout()
    stem = "figure_s8_per_target_explainability_detail"
    outs = save(fig, stem)
    write_manifest(
        stem,
        [heat_path, shares_path],
        outs,
        [
            "Panel A shows mean |SHAP| for each physicochemical descriptor that reached any target's top-20 combined-representation feature list; blank cells indicate the descriptor did not reach the top-20 list for that target.",
            "Panel B compares the distribution of top-20 SHAP importance share contributed by descriptor features versus individual Morgan-fingerprint bits, pooled across all five targets; each point is one target/feature observation.",
        ],
    )


# --------------------------------------------------------------------------
# Figure S9: External-validation evidence hierarchy
# --------------------------------------------------------------------------
def figure_s9():
    deg_path = source("ml/results/meta_analysis/internal_vs_external_degradation.csv")
    ext_path = source("ml/results/external_validation/external_validation_summary.csv")

    deg = pd.read_csv(deg_path)
    deg["target_key"] = deg["target_key"].map(norm_target)

    ext = pd.read_csv(ext_path)
    ext["target"] = ext["target"].map(norm_target)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))

    ax = axes[0]
    xs = np.arange(len(TARGET_ORDER))
    w = 0.26
    axis_keys = {
        "temporal (never seen in training)": ("#1f4e79", "Temporal (never seen)"),
        "BindingDB (independent database)": ("#e15759", "BindingDB"),
    }
    for i, (axis, (color, label)) in enumerate(axis_keys.items()):
        vals = []
        for t in TARGET_ORDER:
            row = deg[(deg["target_key"] == t) & (deg["external_axis"] == axis)]
            vals.append(row["external_roc_auc"].iloc[0] if len(row) else np.nan)
        ax.bar(xs + (i - 0.5) * w, vals, width=w, color=color, label=label)
    internal_vals = []
    for t in TARGET_ORDER:
        row = deg[deg["target_key"] == t]
        internal_vals.append(row["internal_test_roc_auc"].iloc[0] if len(row) else np.nan)
    ax.plot(xs, internal_vals, marker="D", color=DARK, linestyle="none", markersize=6, label="Internal test")
    ax.set_xticks(xs)
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.05)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    panel_label(ax, "A")

    ax = axes[1]
    axis_marker = {"temporal": "o", "bindingdb": "^"}
    for axis in ["temporal", "bindingdb"]:
        sub = ext[ext["axis"] == axis]
        for t in TARGET_ORDER:
            rows = sub[sub["target"] == t]
            if rows.empty:
                continue
            sizes = np.clip(rows["n"].to_numpy(dtype=float), 10, None)
            ax.scatter(rows["n"], rows["roc_auc"], s=sizes * 0.9, color=TARGET_COLORS[t],
                       marker=axis_marker[axis], alpha=0.75, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("Independent evaluation n (log scale)")
    ax.set_ylabel("ROC-AUC")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=8, label="Temporal"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=GREY, markersize=8, label="BindingDB"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=7)
    panel_label(ax, "B")

    fig.tight_layout()
    stem = "figure_s9_external_validation_evidence_hierarchy"
    outs = save(fig, stem)
    write_manifest(
        stem,
        [deg_path, ext_path],
        outs,
        [
            "Panel A compares internal scaffold-split test ROC-AUC with the temporal never-seen and de-overlapped BindingDB external axes, per target.",
            "Panel B plots every external-validation stratum's ROC-AUC against its independent evaluation sample size on a log axis; marker size also scales with n, making the small-n BindingDB strata for DRD2, CB2, and CCR5 visually distinct from the larger, primary-evidence strata.",
        ],
    )


# --------------------------------------------------------------------------
# Figure S10: Zero-shot target-specificity control
# --------------------------------------------------------------------------
def figure_s10():
    matrix_path = source("ml/results/meta_analysis/zeroshot_transfer_matrix.csv")
    stratum_path = source("ml/results/meta_analysis/zeroshot_transfer_reported_stratum.csv")

    matrix = pd.read_csv(matrix_path).set_index("source_label")
    matrix = matrix[[TARGET_LABELS[t] for t in TARGET_ORDER]].loc[[TARGET_LABELS[t] for t in TARGET_ORDER]]

    stratum = pd.read_csv(stratum_path)
    stratum["pair"] = stratum["source_label"] + "→" + stratum["destination_label"]
    stratum = stratum.sort_values("roc_auc_gap_vs_destination_baseline")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})

    ax = axes[0]
    data = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("#f2f2f2")
    im = ax.imshow(masked, cmap=cmap, vmin=0.45, vmax=0.80)
    ax.set_xticks(range(len(TARGET_ORDER)))
    ax.set_xticklabels(target_labels(TARGET_ORDER))
    ax.set_yticks(range(len(TARGET_ORDER)))
    ax.set_yticklabels(target_labels(TARGET_ORDER))
    ax.set_xlabel("Destination target")
    ax.set_ylabel("Source target")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=DARK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Zero-shot transfer ROC-AUC")
    panel_label(ax, "A")

    ax = axes[1]
    ys = np.arange(len(stratum))
    colors = [TARGET_COLORS[norm_target(s)] for s in stratum["source_label"]]
    ax.barh(ys, stratum["roc_auc_gap_vs_destination_baseline"], color=colors)
    ax.axvline(0, color=DARK, linewidth=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels(stratum["pair"], fontsize=6.8)
    ax.set_xlabel("Transfer ROC-AUC − destination-specific baseline")
    panel_label(ax, "B")

    fig.tight_layout()
    stem = "figure_s10_zero_shot_target_specificity_control"
    outs = save(fig, stem)
    write_manifest(
        stem,
        [matrix_path, stratum_path],
        outs,
        [
            "Panel A is the full 5x5 directed zero-shot transfer matrix; the diagonal (self-transfer) is not applicable and is left blank.",
            "Panel B ranks all 20 directed source-to-destination pairs by the gap between zero-shot transfer ROC-AUC and the destination target's own baseline; negative values (transfer worse than the destination baseline) support treating each GPCR target as a distinct modelling problem.",
        ],
    )


def main():
    figure_s1()
    figure_s2()
    figure_s3()
    figure_s4()
    figure_s5()
    figure_s6()
    figure_s7()
    figure_s8()
    figure_s9()
    figure_s10()


if __name__ == "__main__":
    main()
