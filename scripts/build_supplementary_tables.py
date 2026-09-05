#!/usr/bin/env python3
"""Build supplementary-information tables from frozen analysis outputs."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "supplementary" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    return pd.read_csv(ROOT / path)


def fmt(value, digits=3):
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def clean_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def to_markdown(df):
    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(clean_value)
    headers = list(text_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in text_df.itertuples(index=False, name=None):
        cleaned = [str(cell).replace("\n", " ").replace("|", "/") for cell in row]
        lines.append("| " + " | ".join(cleaned) + " |")
    return "\n".join(lines)


def write_table(df, name, caption):
    csv_path = OUT / f"{name}.csv"
    md_path = OUT / f"{name}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(f"{caption}\n\n{to_markdown(df)}\n")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def table_s1():
    spread = read_csv("outputs/tables/table_cross_target_consistency.csv")
    drivers = read_csv("outputs/tables/table_performance_drivers.csv")

    spread_out = spread.rename(
        columns={
            "metric": "Variable",
            "n_targets": "Targets",
            "min": "Minimum",
            "max": "Maximum",
            "range": "Range",
            "mean": "Mean",
            "sd": "SD",
            "coefficient_of_variation": "Coefficient of variation",
            "best_target": "Highest target",
            "worst_target": "Lowest target",
        }
    )
    spread_out.insert(0, "Section", "Cross-target spread")

    drivers_out = drivers.rename(
        columns={
            "property": "Variable",
            "performance_metric": "Performance metric",
            "n_targets": "Targets",
            "spearman_rho": "Spearman rho",
            "spearman_p": "Spearman p",
            "abs_rho": "Absolute rho",
            "interpretation": "Interpretation",
            "n_properties_tied_at_this_rho": "Properties tied at this rho",
        }
    )
    drivers_out.insert(0, "Section", "Performance-driver screen")

    all_cols = [
        "Section",
        "Variable",
        "Performance metric",
        "Targets",
        "Minimum",
        "Maximum",
        "Range",
        "Mean",
        "SD",
        "Coefficient of variation",
        "Highest target",
        "Lowest target",
        "Spearman rho",
        "Spearman p",
        "Absolute rho",
        "Properties tied at this rho",
        "Interpretation",
    ]
    out = pd.concat(
        [spread_out.reindex(columns=all_cols), drivers_out.reindex(columns=all_cols)],
        ignore_index=True,
    )
    write_table(
        out,
        "table_s1_cross_target_consistency_and_drivers",
        "Table S1. Cross-target consistency and performance-driver statistics.",
    )


def table_s2():
    cal = read_csv("outputs/tables/table_calibration_reliability.csv")
    ad = read_csv("outputs/tables/table_applicability_domain_error.csv")
    conformal = read_csv("outputs/tables/table_conformal_coverage.csv")

    cal_out = cal.rename(
        columns={
            "target": "Target",
            "algorithm": "Algorithm",
            "brier_raw": "Raw Brier",
            "brier_calibrated": "Venn-Abers Brier",
            "brier_improvement": "Brier improvement",
            "brier_improvement_pct": "Brier improvement (%)",
            "ece_raw": "Raw ECE",
            "ece_calibrated": "Venn-Abers ECE",
            "ece_improvement": "ECE improvement",
            "calibration_slope_calibrated": "Calibration slope",
            "calibration_intercept_calibrated": "Calibration intercept",
            "venn_abers_mean_interval_width": "Mean Venn-Abers interval width",
            "venn_abers_std_interval_width": "SD Venn-Abers interval width",
        }
    )
    cal_out.insert(0, "Section", "Calibration")

    ad_out = ad.rename(
        columns={
            "target": "Target",
            "n_test": "Test n",
            "pct_within_ad": "Within AD (%)",
            "n_within_ad": "Within AD n",
            "error_rate_within_ad": "Within AD error (%)",
            "n_outside_ad": "Outside AD n",
            "error_rate_outside_ad": "Outside AD error (%)",
            "error_rate_increase_outside_ad": "Outside AD error increase (%)",
            "mean_abs_error_within_ad": "Mean absolute error within AD",
            "mean_abs_error_outside_ad": "Mean absolute error outside AD",
            "mean_abs_error_increase_outside_ad": "Mean absolute error increase outside AD",
            "spearman_ad_distance_vs_abs_error": "AD distance vs absolute error rho",
            "spearman_ad_distance_vs_abs_error_p": "AD distance vs absolute error p",
            "spearman_n": "AD-distance correlation n",
        }
    )
    ad_out.insert(0, "Section", "Applicability domain")

    conformal_out = conformal.rename(
        columns={
            "target": "Target",
            "task": "Task",
            "algorithm": "Algorithm",
            "nominal_level": "Nominal level",
            "stratum": "Stratum",
            "n": "n",
            "empirical_coverage": "Empirical coverage",
            "coverage_minus_nominal": "Coverage minus nominal",
            "covers_nominal": "Covers nominal",
            "mean_set_size": "Mean prediction-set size",
            "mean_interval_width": "Mean interval width",
        }
    )
    conformal_out.insert(0, "Section", "Conformal coverage")

    all_cols = [
        "Section",
        "Target",
        "Task",
        "Algorithm",
        "Nominal level",
        "Stratum",
        "n",
        "Raw Brier",
        "Venn-Abers Brier",
        "Brier improvement",
        "Brier improvement (%)",
        "Raw ECE",
        "Venn-Abers ECE",
        "ECE improvement",
        "Calibration slope",
        "Calibration intercept",
        "Mean Venn-Abers interval width",
        "SD Venn-Abers interval width",
        "Test n",
        "Within AD (%)",
        "Within AD n",
        "Within AD error (%)",
        "Outside AD n",
        "Outside AD error (%)",
        "Outside AD error increase (%)",
        "Mean absolute error within AD",
        "Mean absolute error outside AD",
        "Mean absolute error increase outside AD",
        "AD distance vs absolute error rho",
        "AD distance vs absolute error p",
        "AD-distance correlation n",
        "Empirical coverage",
        "Coverage minus nominal",
        "Covers nominal",
        "Mean prediction-set size",
        "Mean interval width",
    ]
    out = pd.concat(
        [
            cal_out.reindex(columns=all_cols),
            ad_out.reindex(columns=all_cols),
            conformal_out.reindex(columns=all_cols),
        ],
        ignore_index=True,
    )
    write_table(
        out,
        "table_s2_reliability_and_applicability_domain_strata",
        "Table S2. Full reliability and applicability-domain stratification.",
    )


def table_s3():
    redock = read_csv("docking/docking_results/redocking_validation.csv")
    discrim = read_csv("docking/docking_results/discriminative_validation.csv")
    rediscovery = read_csv("docking/docking_results/docking_side_rediscovery_check.csv")
    corr = read_csv("docking/docking_results/ml_docking_correlation.csv")

    redock_out = redock.rename(
        columns={
            "unit": "Receptor unit",
            "target": "Target",
            "state": "State",
            "pdb_id": "PDB ID",
            "n_attempts_used": "Attempts used",
            "best_rmsd": "Best redocking RMSD",
            "unit_passed": "Redocking passed",
            "winning_seed": "Winning seed",
            "winning_exhaustiveness": "Winning exhaustiveness",
        }
    )
    redock_out.insert(0, "Section", "Redocking validation")

    discrim_out = discrim.rename(
        columns={
            "unit": "Receptor unit",
            "score_type": "Score type",
            "auc": "ROC-AUC",
            "bedroc": "BEDROC",
            "ef_1pct": "EF 1%",
            "ef_5pct": "EF 5%",
            "ef_10pct": "EF 10%",
            "n_compounds": "Compounds n",
            "n_actives": "Actives n",
            "n_inactives": "Inactives n",
            "auc_ci_low": "ROC-AUC CI low",
            "auc_ci_high": "ROC-AUC CI high",
            "bedroc_ci_low": "BEDROC CI low",
            "bedroc_ci_high": "BEDROC CI high",
            "ef_5pct_ci_low": "EF 5% CI low",
            "ef_5pct_ci_high": "EF 5% CI high",
            "ef_10pct_ci_low": "EF 10% CI low",
            "ef_10pct_ci_high": "EF 10% CI high",
            "n_bootstrap_successful": "Bootstrap replicates",
        }
    )
    discrim_out.insert(0, "Section", "Discriminative enrichment")

    rediscovery_out = rediscovery.rename(
        columns={
            "unit": "Receptor unit",
            "drugbank_id": "DrugBank ID",
            "status": "Status",
            "vina_affinity": "Vina affinity",
            "vina_rmsd_vs_crystal_reference": "Vina RMSD to crystal ligand",
            "vina_n_protein_contacts": "Vina protein contacts",
            "gnina_cnn_score": "GNINA CNN score",
            "gnina_rmsd_vs_crystal_reference": "GNINA RMSD to crystal ligand",
            "gnina_n_protein_contacts": "GNINA protein contacts",
            "vina_rank_percentile_vs_candidates": "Vina rank percentile",
            "gnina_rank_percentile_vs_candidates": "GNINA rank percentile",
            "vina_passed_rmsd_threshold": "Vina passed RMSD threshold",
            "gnina_passed_rmsd_threshold": "GNINA passed RMSD threshold",
            "overall_rediscovery_pass": "Overall rediscovery pass",
        }
    )
    rediscovery_out.insert(0, "Section", "Rediscovery")

    corr_out = corr.rename(
        columns={
            "unit": "Receptor unit",
            "score_type": "Score type",
            "n_compounds": "Compounds n",
            "algorithm_used": "ML algorithm",
            "algorithm_provenance": "Algorithm provenance",
            "spearman_rho": "Spearman rho",
            "spearman_p": "Spearman p",
        }
    )
    corr_out.insert(0, "Section", "ML-docking correlation")

    all_cols = [
        "Section",
        "Receptor unit",
        "Target",
        "State",
        "PDB ID",
        "Score type",
        "DrugBank ID",
        "Status",
        "Attempts used",
        "Best redocking RMSD",
        "Redocking passed",
        "Winning seed",
        "Winning exhaustiveness",
        "ROC-AUC",
        "ROC-AUC CI low",
        "ROC-AUC CI high",
        "BEDROC",
        "BEDROC CI low",
        "BEDROC CI high",
        "EF 5%",
        "EF 5% CI low",
        "EF 5% CI high",
        "EF 10%",
        "EF 10% CI low",
        "EF 10% CI high",
        "Compounds n",
        "Actives n",
        "Inactives n",
        "Bootstrap replicates",
        "Vina affinity",
        "Vina RMSD to crystal ligand",
        "Vina protein contacts",
        "GNINA CNN score",
        "GNINA RMSD to crystal ligand",
        "GNINA protein contacts",
        "Vina rank percentile",
        "GNINA rank percentile",
        "Vina passed RMSD threshold",
        "GNINA passed RMSD threshold",
        "Overall rediscovery pass",
        "ML algorithm",
        "Algorithm provenance",
        "Spearman rho",
        "Spearman p",
    ]
    out = pd.concat(
        [
            redock_out.reindex(columns=all_cols),
            discrim_out.reindex(columns=all_cols),
            rediscovery_out.reindex(columns=all_cols),
            corr_out.reindex(columns=all_cols),
        ],
        ignore_index=True,
    )
    write_table(
        out,
        "table_s3_docking_validation_and_orthogonality",
        "Table S3. Docking validation, enrichment, rediscovery, and ML-docking orthogonality.",
    )


def table_s4():
    funnel = read_csv("outputs/tables/table_screening_docking_funnel.csv")
    selection = read_csv("docking/docking_results/docking_candidate_selection_summary.csv")
    audit = read_csv("ml/results/meta_analysis/meta_analysis_input_audit.csv")
    disagreements = read_csv("ml/results/meta_analysis/deployed_algorithm_disagreements.csv")

    funnel_out = funnel.rename(columns=lambda c: c.replace("_", " ").title())
    funnel_out.insert(0, "Section", "Screening-to-docking funnel")

    selection_out = selection.rename(columns=lambda c: c.replace("_", " ").title())
    selection_out.insert(0, "Section", "Docking candidate selection")

    audit_out = audit.rename(columns=lambda c: c.replace("_", " ").title())
    audit_out.insert(0, "Section", "Meta-analysis input audit")

    disagreements_out = disagreements.rename(columns=lambda c: c.replace("_", " ").title())
    disagreements_out.insert(0, "Section", "Deployed-algorithm disagreements")

    all_cols = list(dict.fromkeys(
        ["Section"] + list(funnel_out.columns) + list(selection_out.columns) + list(audit_out.columns) + list(disagreements_out.columns)
    ))
    out = pd.concat(
        [
            funnel_out.reindex(columns=all_cols),
            selection_out.reindex(columns=all_cols),
            audit_out.reindex(columns=all_cols),
            disagreements_out.reindex(columns=all_cols),
        ],
        ignore_index=True,
    )
    write_table(
        out,
        "table_s4_screening_and_docking_provenance",
        "Table S4. Screening, docking, and deployed-algorithm provenance.",
    )


def table_s5():
    matrix = read_csv("outputs/tables/table_zeroshot_transfer_matrix.csv")
    reported = read_csv("ml/results/meta_analysis/zeroshot_transfer_reported_stratum.csv")
    takeaway = read_csv("ml/results/meta_analysis/zeroshot_transfer_takeaway.csv")

    matrix_out = matrix.copy()
    matrix_out.insert(0, "Section", "Transfer matrix")

    reported_out = reported.rename(columns=lambda c: c.replace("_", " ").title())
    reported_out.insert(0, "Section", "Directed transfer metrics")

    takeaway_out = takeaway.rename(columns=lambda c: c.replace("_", " ").title())
    takeaway_out.insert(0, "Section", "Summary")

    all_cols = list(dict.fromkeys(["Section"] + list(matrix_out.columns) + list(reported_out.columns) + list(takeaway_out.columns)))
    out = pd.concat(
        [
            matrix_out.reindex(columns=all_cols),
            reported_out.reindex(columns=all_cols),
            takeaway_out.reindex(columns=all_cols),
        ],
        ignore_index=True,
    )
    write_table(
        out,
        "table_s5_zero_shot_transfer_metrics",
        "Table S5. Zero-shot source-target transfer metrics.",
    )


def table_s6():
    df = read_csv("manuscript/table_docking_headline_hits_interpreted.csv")
    keep = [
        "target",
        "unit",
        "drugbank_id",
        "generic_name",
        "pharmacology_status",
        "evidence_category",
        "candidate_claim_allowed",
        "status_note",
        "state",
        "global_compound_id",
        "vina_affinity",
        "gnina_cnn_score",
        "priority_score_from_notebook4",
        "is_unanimous_consensus_notebook4",
        "n_protein_contacts",
        "n_polar_contacts",
        "n_ionic_acid_contacts",
        "target_plif_rule",
        "lipinski_pass",
        "muegge_pass",
        "headline_hit_definition",
    ]
    out = df[[c for c in keep if c in df.columns]].rename(columns=lambda c: c.replace("_", " ").title())
    out = out.rename(
        columns={
            "Priority Score From Notebook4": "DrugBank screening priority score",
            "Is Unanimous Consensus Notebook4": "Unanimous screening consensus",
        }
    )
    write_table(
        out,
        "table_s6_prioritized_compound_pharmacological_context",
        "Table S6. Pharmacological context and evidence classification for prioritized compounds.",
    )


def table_s7():
    manifest = read_csv("manuscript/analysis_freeze_manifest_2026-08-30.csv")
    key_paths = [
        "manuscript/tables/table1_dataset_characteristics_main.csv",
        "manuscript/tables/table2_model_performance_main.csv",
        "manuscript/tables/table3_external_validation_main.csv",
        "manuscript/tables/table4_reliability_decision_readiness_main.csv",
        "manuscript/tables/table5_explainability_feature_class_main.csv",
        "manuscript/tables/table6_explainability_reliability_bridge_main.csv",
        "manuscript/tables/table7_prioritization_hits_main.csv",
        "outputs/figures/manuscript/figure1_framework_panel_manifest.json",
        "outputs/figures/manuscript/figure2_performance_scaling_manifest.json",
        "outputs/figures/manuscript/figure3_reliability_uncertainty_manifest.json",
        "outputs/figures/manuscript/figure4_representation_transferability_manifest.json",
        "outputs/figures/manuscript/figure5_screening_docking_funnel_manifest.json",
        "outputs/figures/manuscript/figure6_docking_validation_orthogonality_manifest.json",
        "ml/results/meta_analysis/model_performance_landscape.csv",
        "ml/results/meta_analysis/screening_docking_funnel.csv",
        "ml/results/meta_analysis/docking_headline_hits.csv",
        "ml/results/meta_analysis/ml_docking_agreement_summary.csv",
        "ml/results/meta_analysis/zeroshot_transfer_takeaway.csv",
        "ml/results/drugbank_screening/drugbank_primary_ranked_candidates.csv",
        "ml/results/drugbank_screening/screening_summary_primary_deployed.csv",
        "ml/results/external_validation/external_validation_summary.csv",
        "docking/docking_results/docking_results_final.csv",
        "docking/docking_results/redocking_validation.csv",
        "docking/docking_results/discriminative_validation.csv",
        "docking/docking_results/docking_side_rediscovery_check.csv",
        "docking/docking_results/ml_docking_correlation.csv",
        "docking/docking_results/manifest_05_docking.json",
    ]
    keep = manifest[manifest["path"].isin(key_paths)].copy()
    keep["Artifact class"] = keep["path"].map(
        lambda p: "main-table" if str(p).startswith("manuscript/tables/")
        else "analysis-table" if str(p).startswith("outputs/tables/")
        else "main-figure" if str(p).startswith("outputs/figures/manuscript/")
        else "meta-analysis" if str(p).startswith("ml/results/meta_analysis/")
        else "external-validation" if str(p).startswith("ml/results/external_validation/")
        else "drugbank-screening" if str(p).startswith("ml/results/drugbank_screening/")
        else "docking"
    )
    out = keep[["Artifact class", "path", "bytes", "sha256", "n_rows"]].rename(
        columns={"path": "Path", "bytes": "Bytes", "sha256": "SHA-256", "n_rows": "Rows"}
    )
    missing = sorted(set(key_paths) - set(keep["path"]))
    if missing:
        raise FileNotFoundError(f"Key manifest paths missing from freeze manifest: {missing}")
    write_table(
        out,
        "table_s7_artifact_and_manifest_inventory",
        "Table S7. Artifact and manifest inventory for manuscript-facing and supporting analyses.",
    )


def table_s8():
    """Summarize the read-only ChEMBL relation-status sensitivity diagnostic."""
    join = read_csv("data/quality/chembl_relation_join_report.csv")
    censoring = read_csv("data/quality/chembl_relation_censoring_summary.csv")
    intervals = read_csv("data/quality/chembl_relation_median_interval_classification.csv")
    labels = read_csv("data/quality/chembl_relation_label_change_summary.csv")

    out = (
        join.merge(censoring, on="target", how="inner", validate="one_to_one")
        .merge(intervals, on="target", how="inner", validate="one_to_one")
        .merge(labels, on="target", how="inner", validate="one_to_one", suffixes=("", "_labels"))
    )
    out = out.rename(
        columns={
            "target": "Target",
            "n_frozen_records": "Frozen records",
            "key_match_rate_pct": "Relation-query key match (%)",
            "relation_resolved_rate_pct": "Resolved relation metadata (%)",
            "n_exact": "Exact records",
            "pct_censored_of_all": "Censored records (%)",
            "pct_unresolved_of_all": "Unresolved records (%)",
            "n_all_compounds": "Compounds in median proxy",
            "n_zero_exact_compounds": "Compounds with no exact evidence",
            "n_indeterminate_by_median_interval": "Censored-only indeterminate compounds",
            "pct_indeterminate_of_all_compounds": "Censored-only indeterminate (%)",
            "n_compounds_evaluable": "Exact-only evaluable compounds",
            "n_compounds_label_would_change": "Exact-only label changes",
            "pct_compounds_label_would_change_of_evaluable": "Exact-only label-change rate (%)",
            "verdict": "Post hoc triage",
            "triggered_by": "Triage basis",
        }
    )
    keep = [
        "Target",
        "Frozen records",
        "Relation-query key match (%)",
        "Resolved relation metadata (%)",
        "Exact records",
        "Censored records (%)",
        "Unresolved records (%)",
        "Compounds in median proxy",
        "Compounds with no exact evidence",
        "Censored-only indeterminate compounds",
        "Censored-only indeterminate (%)",
        "Exact-only evaluable compounds",
        "Exact-only label changes",
        "Exact-only label-change rate (%)",
        "Post hoc triage",
        "Triage basis",
    ]
    out = out.reindex(columns=keep)
    for col in out.select_dtypes(include="number"):
        out[col] = out[col].round(3)
    out["Target"] = pd.Categorical(
        out["Target"], categories=["drd2", "cb2", "adora2a", "oprm1", "ccr5"], ordered=True
    )
    out = out.sort_values("Target")
    out["Target"] = out["Target"].str.upper()
    write_table(
        out,
        "table_s8_chembl_relation_status_sensitivity",
        "Table S8. Post hoc ChEMBL relation-status sensitivity diagnostic.",
    )


def main():
    table_s1()
    table_s2()
    table_s3()
    table_s4()
    table_s5()
    table_s6()
    table_s7()
    table_s8()


if __name__ == "__main__":
    main()
