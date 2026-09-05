#!/usr/bin/env python3
"""Build compact manuscript-facing main tables from frozen analysis outputs."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


TARGET_ORDER = ["DRD2", "CB2", "ADORA2A", "OPRM1", "CCR5"]


def fmt_float(value, digits=3):
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_pct(value, digits=1):
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def write_table(df, name):
    csv_path = OUT / f"{name}.csv"
    md_path = OUT / f"{name}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(df) + "\n")
    print(f"wrote {csv_path} and {md_path}")


def to_markdown(df):
    text_df = df.fillna("").astype(str)
    headers = list(text_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in text_df.itertuples(index=False, name=None):
        cleaned = [cell.replace("\n", " ").replace("|", "/") for cell in row]
        lines.append("| " + " | ".join(cleaned) + " |")
    return "\n".join(lines)


def table1_dataset_characteristics():
    src = ROOT / "outputs" / "tables" / "table_dataset_characteristics.csv"
    df = pd.read_csv(src).set_index("target").reindex([t.lower() for t in TARGET_ORDER]).reset_index()
    out = pd.DataFrame(
        {
            "Target": TARGET_ORDER,
            "Final compounds": df["final_unique_compounds"].astype(int),
            "Raw records": df["raw_records"].astype(int),
            "Active (%)": df["active_pct"].map(fmt_pct),
            "Endpoint composition Ki/IC50/EC50 (%)": [
                f"{fmt_pct(r.pct_retained_ki_measurements)}/{fmt_pct(r.pct_retained_ic50_measurements)}/{fmt_pct(r.pct_retained_ec50_measurements)}"
                for r in df.itertuples(index=False)
            ],
            "Scaffold richness": df["scaffold_richness"].map(fmt_float),
            "Singleton scaffolds (%)": df["singleton_scaffold_pct"].map(fmt_pct),
            "Single-assay compounds (%)": df["single_assay_pct"].map(fmt_pct),
        }
    )
    write_table(out, "table1_dataset_characteristics_main")


def table2_model_performance():
    src = ROOT / "outputs" / "tables" / "table_model_performance_landscape.csv"
    df = pd.read_csv(src).set_index("target").reindex(TARGET_ORDER).reset_index()
    out = pd.DataFrame(
        {
            "Target": df["target"],
            "Classifier": df["deployed_algorithm_classification"],
            "Regressor": df["deployed_algorithm_regression"],
            "Internal ROC-AUC": df["internal_test_roc_auc"].map(fmt_float),
            "Internal R2": df["internal_test_r2"].map(fmt_float),
            "Brier": df["brier_calibrated"].map(fmt_float),
            "ECE": df["ece_calibrated"].map(fmt_float),
            "Class conformal coverage": df["conformal_coverage_classification"].map(fmt_float),
            "Test within AD (%)": df["pct_test_within_applicability_domain"].map(fmt_pct),
        }
    )
    write_table(out, "table2_model_performance_main")


def table3_external_validation():
    src = ROOT / "ml" / "results" / "external_validation" / "external_validation_summary.csv"
    df = pd.read_csv(src)
    keep = (
        (df["axis"].eq("bindingdb") & df["stratum"].eq("all"))
        | (df["axis"].eq("temporal") & df["stratum"].eq("post_cutoff_never_seen"))
    )
    df = df[keep].copy()
    df["Target"] = df["target"].str.upper()
    df["Validation axis"] = df["axis"].map({"bindingdb": "BindingDB de-overlapped", "temporal": "Temporal never-seen"})
    df["ROC-AUC (95% CI)"] = [
        f"{fmt_float(r.roc_auc)} ({fmt_float(r.roc_auc_ci_low)}-{fmt_float(r.roc_auc_ci_high)})"
        for r in df.itertuples(index=False)
    ]
    df["Evidence flag"] = df["flag"].str.replace("_", " ", regex=False)
    df["order"] = df["Target"].map({t: i for i, t in enumerate(TARGET_ORDER)})
    df["axis_order"] = df["axis"].map({"temporal": 0, "bindingdb": 1})
    df = df.sort_values(["order", "axis_order"])
    out = df[
        [
            "Target",
            "Validation axis",
            "n",
            "n_active",
            "n_inactive",
            "ROC-AUC (95% CI)",
            "pr_auc",
            "brier",
            "pct_within_ad",
            "conformal_empirical_coverage_90pct",
            "Evidence flag",
        ]
    ].rename(
        columns={
            "n": "n",
            "n_active": "Active n",
            "n_inactive": "Inactive n",
            "pr_auc": "PR-AUC",
            "brier": "Brier",
            "pct_within_ad": "Within AD (%)",
            "conformal_empirical_coverage_90pct": "Conformal coverage",
        }
    )
    for col in ["PR-AUC", "Brier", "Conformal coverage"]:
        out[col] = out[col].map(fmt_float)
    out["Within AD (%)"] = out["Within AD (%)"].map(fmt_pct)
    write_table(out, "table3_external_validation_main")


def table7_prioritization_hits():
    src = ROOT / "manuscript" / "table_docking_headline_hits_interpreted.csv"
    df = pd.read_csv(src)
    status_label = {
        "known_approved_target_pharmacology": "Known approved target pharmacology",
        "known_investigational_target_pharmacology": "Known investigational target pharmacology",
        "illicit_known_class_opioid_signal": "Hazard-associated known-class opioid signal",
        "investigational_cross_pharmacology_signal": "Investigational cross-pharmacology signal",
        "investigational_non_gpcr_chemical_matter": "Investigational non-GPCR chemical matter",
        "experimental_provisional_chemical_starting_point": "Experimental chemical matter",
        "experimental_chemical_matter": "Experimental chemical matter",
    }
    category_label = {
        "rediscovery_style_pharmacological_sanity_check": "Pharmacological positive control",
        "hazard_caveat_row_not_repurposing_lead": "Hazard-associated opioid-class signal",
        "strongest_genuine_repurposing_style_candidate": "Cautious cross-pharmacology candidate",
        "chemical_starting_point": "Experimental chemical matter",
    }
    claim_label = {
        "yes_cautious": "Cautious cross-pharmacology prioritization",
        "no_known_pharmacology": "Known target pharmacology",
        "no_hazard_known_class": "Hazard-associated known-class signal",
        "no_experimental_starting_point": "Experimental chemical matter",
    }
    category_order = {
        "strongest_genuine_repurposing_style_candidate": 0,
        "rediscovery_style_pharmacological_sanity_check": 1,
        "hazard_caveat_row_not_repurposing_lead": 2,
        "chemical_starting_point": 3,
    }
    df = df.assign(_category_order=df["evidence_category"].map(category_order)).sort_values(
        ["_category_order", "target", "drugbank_id"]
    )

    def display_compound(row):
        name = row["generic_name"]
        if pd.isna(name):
            return row["drugbank_id"]
        return name

    out = pd.DataFrame(
        {
            "Target": df["target"].str.upper(),
            "Compound": [display_compound(row) for _, row in df.iterrows()],
            "DrugBank ID": df["drugbank_id"],
            "Evidence category": df["evidence_category"].map(category_label),
            "Pharmacological status": df["pharmacology_status"].map(status_label),
            "Prioritization interpretation": df["candidate_claim_allowed"].map(claim_label),
            "Vina affinity (kcal/mol)": df["vina_affinity"].map(lambda v: fmt_float(v, 2)),
            "GNINA CNN score": df["gnina_cnn_score"].map(lambda v: fmt_float(v, 2)),
            "Lipinski/Muegge filters": [
                f"{'pass' if r.lipinski_pass else 'fail'}/{'pass' if r.muegge_pass else 'fail'}"
                for r in df.itertuples(index=False)
            ],
        }
    )
    write_table(out, "table7_prioritization_hits_main")


def table4_reliability_decision_readiness():
    cal = pd.read_csv(ROOT / "ml" / "results" / "meta_analysis" / "calibration_across_targets.csv")
    conf = pd.read_csv(ROOT / "ml" / "results" / "meta_analysis" / "confidence_accuracy_monotonicity.csv")
    ad = pd.read_csv(ROOT / "ml" / "results" / "meta_analysis" / "applicability_domain_error_analysis.csv")
    conformal = pd.read_csv(ROOT / "ml" / "results" / "meta_analysis" / "conformal_ad_stratified_effects.csv")
    reg_conformal = conformal[conformal["task"].eq("regression")].copy()

    cal = cal.set_index("target").reindex(TARGET_ORDER)
    conf = conf.set_index("target").reindex(TARGET_ORDER)
    ad = ad.set_index("target").reindex(TARGET_ORDER)
    reg_conformal = reg_conformal.set_index("target").reindex(TARGET_ORDER)

    out = pd.DataFrame(
        {
            "Target": TARGET_ORDER,
            "Raw Brier": cal["brier_raw"].map(fmt_float).tolist(),
            "Venn-Abers Brier": cal["brier_calibrated"].map(fmt_float).tolist(),
            "Raw ECE": cal["ece_raw"].map(fmt_float).tolist(),
            "Venn-Abers ECE": cal["ece_calibrated"].map(fmt_float).tolist(),
            "Within-AD error (%)": ad["error_rate_within_ad"].map(fmt_pct).tolist(),
            "Outside-AD error (%)": ad["error_rate_outside_ad"].map(fmt_pct).tolist(),
            "Error increase outside AD (%)": ad["error_rate_increase_outside_ad"].map(fmt_pct).tolist(),
            "Regression conformal coverage within/outside AD": [
                f"{fmt_float(r.empirical_coverage_within_ad)}/{fmt_float(r.empirical_coverage_outside_ad)}"
                for r in reg_conformal.itertuples(index=False)
            ],
            "Coverage drop outside AD": reg_conformal["coverage_drop_outside_ad"].map(fmt_float).tolist(),
            "Confidence monotonic": conf["accuracy_increases_monotonically"].map(lambda v: "yes" if bool(v) else "no").tolist(),
            "Low-to-high confidence accuracy (%)": [
                f"{fmt_pct(r.accuracy_lowest_bin)}->{fmt_pct(r.accuracy_highest_bin)}"
                for r in conf.itertuples(index=False)
            ],
            "AD-distance/error rho": ad["spearman_ad_distance_vs_abs_error"].map(fmt_float).tolist(),
            "AD-distance/error p": ad["spearman_ad_distance_vs_abs_error_p"].map(lambda v: "<0.001" if float(v) < 0.001 else fmt_float(v)).tolist(),
            "AD-distance/error n": ad["spearman_n"].astype(int).tolist(),
        }
    )
    write_table(out, "table4_reliability_decision_readiness_main")


def table5_explainability_feature_class():
    rep = pd.read_csv(ROOT / "ml" / "results" / "meta_analysis" / "feature_representation_comparison.csv")
    heatmap = pd.read_csv(ROOT / "ml" / "results" / "meta_analysis" / "shap_descriptor_heatmap_combined.csv")
    shares = pd.read_csv(ROOT / "ml" / "results" / "meta_analysis" / "shap_feature_shares_long.csv")
    transfer = pd.read_csv(ROOT / "ml" / "results" / "meta_analysis" / "feature_transferability_by_type.csv")

    rep_wide = rep.pivot_table(
        index=["target", "target_key", "deployed_algorithm_classification"],
        columns="feature_representation",
        values="test_roc_auc_deployed_algorithm",
        aggfunc="first",
    ).reset_index()
    rep_wide["Target"] = rep_wide["target"].astype(str)
    rep_wide = rep_wide.set_index("Target").reindex(TARGET_ORDER).reset_index()

    descriptor_cols = [c for c in heatmap.columns if c in TARGET_ORDER]
    top_desc = {}
    for target in descriptor_cols:
        ranked = heatmap[["feature", target]].dropna().sort_values(target, ascending=False)
        top_desc[target] = ", ".join(ranked["feature"].head(3).tolist())

    combined_shares = shares[shares["feature_representation"].eq("combined")].copy()
    descriptor_share = (
        combined_shares[combined_shares["feature_type"].eq("descriptor")]
        .groupby("target_label")["share_of_top_features"]
        .sum()
    )
    morgan_share = (
        combined_shares[combined_shares["feature_type"].eq("Morgan bit")]
        .groupby("target_label")["share_of_top_features"]
        .sum()
    )
    transfer_lookup = transfer.set_index("feature_type")["mean_targets_shared"]

    def representation_category(row, tie_band=0.01):
        combined = float(row["combined"])
        morgan = float(row["morgan"])
        descriptors = float(row["descriptors"])
        if abs(combined - morgan) < tie_band and combined > descriptors and morgan > descriptors:
            return "Combined ~= Morgan"
        if morgan > combined and morgan > descriptors:
            return "Morgan-best"
        if combined >= morgan and combined >= descriptors:
            return "Combined-best"
        return "Descriptor-best"

    out = pd.DataFrame(
        {
            "Target": TARGET_ORDER,
            "Deployed classifier": rep_wide["deployed_algorithm_classification"],
            "Combined ROC-AUC": rep_wide["combined"].map(fmt_float),
            "Morgan ROC-AUC": rep_wide["morgan"].map(fmt_float),
            "Descriptor ROC-AUC": rep_wide["descriptors"].map(fmt_float),
            "Combined - Morgan": (rep_wide["combined"] - rep_wide["morgan"]).map(fmt_float),
            "Combined - Descriptor": (rep_wide["combined"] - rep_wide["descriptors"]).map(fmt_float),
            "Top recurrent descriptors in combined model": [top_desc.get(t, "") for t in TARGET_ORDER],
            "Descriptor share of top-20 importance": [fmt_float(descriptor_share.get(t, pd.NA)) for t in TARGET_ORDER],
            "Morgan-bit share of top-20 importance": [fmt_float(morgan_share.get(t, pd.NA)) for t in TARGET_ORDER],
            "Representation category": [representation_category(row) for _, row in rep_wide.iterrows()],
        }
    )
    write_table(out, "table5_explainability_feature_class_main")
    footnote = (
        "Note: Mean targets per top feature were descriptors "
        f"{transfer_lookup['descriptor']:.2f} and Morgan bits {transfer_lookup['Morgan bit']:.2f}. "
        "Feature recurrence and feature-share values are computed from top-feature lists and should be interpreted "
        "as transferability of explanatory anchors, not exhaustive mechanistic attribution. "
        "Combined-minus-Morgan differences are small and should be interpreted as effect sizes."
    )
    with (OUT / "table5_explainability_feature_class_main.md").open("a") as fh:
        fh.write(f"\n{footnote}\n")


def table6_explainability_reliability_bridge():
    src = ROOT / "ml" / "results" / "meta_analysis" / "explainability_reliability_bridge.csv"
    df = pd.read_csv(src)
    # Notebook 7 now owns this analysis and names its columns slightly differently
    # from the retired standalone script. Map the old schema onto the new one so
    # the table builds from whichever producer wrote the file.
    aliases = {
        "target": "target_label",
        "shap_artifact_algorithm": "shap_model_algorithm",
        "descriptor_share_hc_correct": "descriptor_share_high_confidence_correct",
        "descriptor_share_hc_incorrect": "descriptor_share_high_confidence_incorrect",
        "n_inside_ad": "n_inside_applicability_domain",
        "n_outside_ad": "n_outside_applicability_domain",
        "recurrent_anchor_share_inside_ad": "recurrent_descriptor_share_inside_ad",
        "recurrent_anchor_share_outside_ad": "recurrent_descriptor_share_outside_ad",
    }
    for src_col, expected in aliases.items():
        if expected not in df.columns and src_col in df.columns:
            df[expected] = df[src_col]
    df = df.set_index("target_label").reindex(TARGET_ORDER).reset_index()

    def model_match(row):
        if bool(row["shap_matches_deployed_model"]):
            return "matched"
        return f"context only ({row['shap_model_algorithm']} SHAP)"

    out = pd.DataFrame(
        {
            "Target": df["target_label"],
            "Deployed classifier": df["deployed_algorithm"],
            "SHAP/deployed model status": [model_match(row) for _, row in df.iterrows()],
            "High-confidence correct/incorrect n": [
                f"{int(r.n_high_confidence_correct)}/{int(r.n_high_confidence_incorrect)}"
                for r in df.itertuples(index=False)
            ],
            "High-confidence incorrect n": df["n_high_confidence_incorrect"].astype(int),
            "Descriptor SHAP share, high-conf. correct/incorrect": [
                f"{fmt_float(r.descriptor_share_high_confidence_correct)}/{fmt_float(r.descriptor_share_high_confidence_incorrect)}"
                for r in df.itertuples(index=False)
            ],
            "Descriptor-share shift in high-conf. errors": [
                fmt_float(-r.descriptor_share_hc_correct_minus_incorrect)
                for r in df.itertuples(index=False)
            ],
            "Inside/outside AD n": [
                f"{int(r.n_inside_applicability_domain)}/{int(r.n_outside_applicability_domain)}"
                for r in df.itertuples(index=False)
            ],
            "Descriptor SHAP share, inside/outside AD": [
                f"{fmt_float(r.descriptor_share_inside_ad)}/{fmt_float(r.descriptor_share_outside_ad)}"
                for r in df.itertuples(index=False)
            ],
            "Recurrent-descriptor share, inside/outside AD": [
                f"{fmt_float(r.recurrent_descriptor_share_inside_ad)}/{fmt_float(r.recurrent_descriptor_share_outside_ad)}"
                for r in df.itertuples(index=False)
            ],
        }
    )
    write_table(out, "table6_explainability_reliability_bridge_main")
    footnote = (
        "Note: SHAP shares are computed per compound from the deployed full-pool combined classifier. "
        "For all five targets, the SHAP artifact corresponds to the deployed classifier. The recurrent descriptor anchors are "
        "FractionCSP3, MW, LogP, and HeavyAtomCount. Descriptor-share shifts in high-confidence errors are "
        "reported descriptively without bootstrap intervals; high-confidence incorrect counts are small, especially "
        "for CCR5."
    )
    with (OUT / "table6_explainability_reliability_bridge_main.md").open("a") as fh:
        fh.write(f"\n{footnote}\n")


def main():
    table1_dataset_characteristics()
    table2_model_performance()
    table3_external_validation()
    table4_reliability_decision_readiness()
    table5_explainability_feature_class()
    table6_explainability_reliability_bridge()
    table7_prioritization_hits()


if __name__ == "__main__":
    main()
