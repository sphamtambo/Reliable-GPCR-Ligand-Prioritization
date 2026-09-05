"""
Generate publication-ready Markdown tables from the source CSVs.
Clean human-readable headers + units; raw CSVs untouched (kept as deposited source data).
Output: outputs/tables/formatted_tables.md  (paste into the manuscript).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
OUT = T / "formatted_tables.md"

# header rename maps (source col -> display header with units)
RENAME = {
    "table3_shap_morgan_bit_decode.csv": {
        "Bit_ID": "Fingerprint bit", "Mean_abs_SHAP": "Mean absolute SHAP",
        "Representative_SMARTS_Pattern": "Representative SMARTS", "Interpretation": "Interpretation",
        "Pct_actives_with_bit": "Actives with bit (%)", "Pct_inactives_with_bit": "Inactives with bit (%)",
        "Enrichment_ratio": "Enrichment ratio"},
    "table5_drugbank_threshold_sensitivity.csv": {
        "Probability_threshold": "Probability threshold", "Pool_size": "Pool size",
        "Known_CB2_associated_in_pool": "Known CB2-associated in pool", "Precision": "Precision",
        "Recall": "Recall", "Enrichment_factor": "Enrichment factor"},
    "table6_consensus_candidates_docking.csv": {
        "DrugBank_ID": "DrugBank ID", "Name": "Compound",
        "Vina_active": "Vina, active (kcal/mol)", "Vina_inactive": "Vina, inactive (kcal/mol)",
        "GNINA_CNN_active": "GNINA CNN, active (0-1)", "GNINA_CNN_inactive": "GNINA CNN, inactive (0-1)",
        "Consensus_state": "Consensus state"},
    "table7_final_candidates.csv": {
        "DrugBank_ID": "DrugBank ID", "Name": "Compound", "Drug_group": "Drug group",
        "Known_primary_target": "Known primary target", "Therapeutic_indication": "Therapeutic indication",
        "CB2_probability": "CB2 probability", "CB2_pChEMBL_value": "CB2 pChEMBL value",
        "CB1_probability": "CB1 probability", "Selectivity_index": "Selectivity index",
        "Within_CB2_AD": "Within CB2 AD", "Consensus_state": "Consensus state",
        "PAINS_Brenk": "PAINS/Brenk", "MW": "MW (g/mol)", "LogP": "LogP",
        "TPSA": "TPSA (Å²)", "Status": "Status"},
    "table8_external_validation.csv": {
        "Metric": "Metric", "Internal_test": "Internal test", "External_BindingDB": "External (BindingDB)"},
    "tableS2_redocking_validation.csv": {
        "Receptor_state": "Receptor state", "PDB_ID": "PDB ID", "Reference_ligand": "Reference ligand",
        "Engine": "Engine", "Heavy_atom_RMSD_A": "Heavy-atom RMSD (Å)",
        "Passed_2.0A_threshold": "Passed (≤ 2.0 Å)"},
    "tableS3_ml_vs_docking_correlation.csv": {
        "receptor_state": "Receptor state", "ml_output": "ML output", "docking_engine": "Docking engine",
        "spearman_r": "Spearman r", "spearman_p": "p-value", "n": "n"},
    "tableS4_all_docked_candidates.csv": {
        "DrugBank_ID": "DrugBank ID", "Name": "Compound",
        "Vina_active": "Vina, active (kcal/mol)", "Vina_inactive": "Vina, inactive (kcal/mol)",
        "GNINA_CNN_active": "GNINA CNN, active (0-1)", "GNINA_CNN_inactive": "GNINA CNN, inactive (0-1)",
        "Consensus_pass": "Consensus pass"},
    "tableS5_feature_ablation.csv": {"Single-split ensemble R2": "Single-split ensemble R²"},
}
TITLES = {
    "table1_dataset_composition.csv": "Table 1. Dataset composition and curation summary",
    "table2_repeated_scaffold_split_validation.csv": "Table 2. Repeated Bemis–Murcko scaffold-split validation",
    "table3_shap_morgan_bit_decode.csv": "Table 3. Top Morgan fingerprint features by SHAP",
    "table4_calibration_summary.csv": "Table 4. Calibration and applicability domain summary",
    "table5_drugbank_threshold_sensitivity.csv": "Table 5. DrugBank screening threshold sensitivity",
    "table6_consensus_candidates_docking.csv": "Table 6. Docking scores for the five consensus candidates",
    "table7_final_candidates.csv": "Table 7. Prioritized CB2-selective repurposing candidates",
    "table8_external_validation.csv": "Table 8. Internal vs external (BindingDB) validation",
    "tableS1_activity_pool_sensitivity.csv": "Table S1. Activity-pool sensitivity",
    "tableS2_redocking_validation.csv": "Table S2. Redocking validation",
    "tableS3_ml_vs_docking_correlation.csv": "Table S3. ML confidence vs docking-score correlations",
    "tableS4_all_docked_candidates.csv": "Table S4. All 50 docked candidates",
    "tableS5_feature_ablation.csv": "Table S5. Feature-set ablation",
}
ORDER = ["table1_dataset_composition.csv","table2_repeated_scaffold_split_validation.csv",
    "table3_shap_morgan_bit_decode.csv","table4_calibration_summary.csv",
    "table5_drugbank_threshold_sensitivity.csv","table6_consensus_candidates_docking.csv",
    "table7_final_candidates.csv","table8_external_validation.csv",
    "tableS1_activity_pool_sensitivity.csv","tableS2_redocking_validation.csv",
    "tableS3_ml_vs_docking_correlation.csv","tableS4_all_docked_candidates.csv",
    "tableS5_feature_ablation.csv"]

def md(fn):
    df = pd.read_csv(T / fn, dtype=str, keep_default_na=False, engine="python")
    # Table 3: merge modal/sampled into one "Decoding support" column
    if fn.startswith("table3") and "Modal_pattern_count" in df.columns:
        df["Decoding support"] = df["Modal_pattern_count"] + "/" + df["N_occurrences_sampled"]
        df = df.drop(columns=["Modal_pattern_count", "N_occurrences_sampled"])
        cols = list(df.columns); cols.insert(4, cols.pop(cols.index("Decoding support")))
        df = df[cols]
    df = df.rename(columns=RENAME.get(fn, {}))
    hdr = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(v).replace("|", "\\|") for v in r) + " |"
            for r in df.itertuples(index=False)]
    return f"**{TITLES.get(fn, fn)}**\n\n" + "\n".join([hdr, sep] + rows) + "\n"

with open(OUT, "w", encoding="utf-8") as f:
    f.write("# Publication-ready tables (paste into manuscript; raw CSVs remain the source data)\n\n")
    body=""
    for fn in ORDER:
        if (T / fn).exists():
            body+=md(fn)+"\n"
    f.write(body.replace(" R2"," R\u00b2").replace("+/-","\u00b1"))
print("wrote", OUT)
