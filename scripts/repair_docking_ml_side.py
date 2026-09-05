#!/usr/bin/env python3
"""Repair the ML-dependent columns of the docking layer after a deployed-model change.

Why this exists
---------------
The docking run scored its enrichment set with whichever classifier
`best_algorithm_by_combination.csv` named at the time. That table was later
corrected: ADORA2A and OPRM1 deploy XGBoost, not Random Forest. Two docking
artifacts therefore describe models the study does not deploy:

    enrichment_validation_set.csv   ->  algorithm_used, predicted_proba
    ml_docking_correlation.csv      ->  the ML side of every Spearman rho

Nothing structural is affected. Poses, redocking, consensus, interaction checks,
enrichment ROC-AUC/BEDROC/EF and the prioritized hits are either label-based or
geometry-based and stay valid, so this repair does NOT re-dock anything.

What it does
------------
1. Reads the corrected deployed algorithm per target.
2. Keeps the enrichment compound set EXACTLY as docked - it only swaps the ML
   prediction attached to each compound for the deployed model's prediction.
   Re-sampling would risk selecting compounds that were never docked.
3. Recomputes the ML-vs-docking Spearman correlations.
4. Optionally drops a duplicate headline-hit column.
5. Rewrites the affected manifest hashes so the manifest stays truthful.

Usage
-----
    python3 scripts/repair_docking_ml_side.py --dry-run
    python3 scripts/repair_docking_ml_side.py
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

POOL = "full"
REPRESENTATION = "combined"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", default=".")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    ap.add_argument("--keep-duplicate-headline-column", action="store_true",
                    help="do not drop the redundant 'headline_hit' column")
    args = ap.parse_args()

    root = Path(args.project_dir).resolve()
    dock = root / "docking" / "docking_results"
    results = root / "ml" / "results"

    # ---- corrected deployed algorithm -------------------------------------
    table = pd.read_csv(results / "best_algorithm_by_combination.csv")
    deployed = (table[table["activity_pool"].eq(POOL) & table["task"].eq("classification")]
                .set_index("target")["best_algorithm"].to_dict())
    print("Deployed classifier per target (leak-free selection):")
    for t, a in deployed.items():
        print(f"  {t:<9}{a}")

    # ---- enrichment set: keep compounds, replace the ML prediction ---------
    enrich_path = dock / "enrichment_validation_set.csv"
    enrich = pd.read_csv(enrich_path)
    preds = pd.read_csv(results / "deployed_model_test_predictions.csv")

    changed_targets, before_after = [], []
    for target, algo in deployed.items():
        mask = enrich["target"].eq(target)
        if not mask.any():
            continue
        current = str(enrich.loc[mask, "algorithm_used"].iloc[0])
        if current == algo:
            continue

        pool = preds[preds["target"].eq(target) & preds["activity_pool"].eq(POOL)
                     & preds["feature_representation"].eq(REPRESENTATION)
                     & preds["algorithm"].eq(algo)]
        lookup = pool.set_index("global_compound_id")

        ids = enrich.loc[mask, "global_compound_id"]
        missing = int((~ids.isin(lookup.index)).sum())
        if missing:
            print(f"\n  {target}: {missing} docked compound(s) absent from the {algo} predictions. "
                  f"Refusing to repair this target - the enrichment set and the deployed "
                  f"predictions do not describe the same compounds.")
            continue

        old_proba = enrich.loc[mask, "predicted_proba"].to_numpy(float)
        for col in ("predicted_proba", "predicted_class", "predicted_pactivity",
                    "within_applicability_domain", "applicability_domain_distance"):
            if col in lookup.columns:
                enrich.loc[mask, col] = ids.map(lookup[col]).to_numpy()
        enrich.loc[mask, "algorithm"] = algo
        enrich.loc[mask, "algorithm_used"] = algo

        new_proba = enrich.loc[mask, "predicted_proba"].to_numpy(float)
        changed_targets.append(target)
        before_after.append({
            "target": target, "was": current, "now": algo, "n_compounds": int(mask.sum()),
            "mean_abs_proba_shift": float(np.nanmean(np.abs(new_proba - old_proba))),
        })

    if before_after:
        print("\nEnrichment-set predictions replaced:")
        print(pd.DataFrame(before_after).to_string(index=False))
    else:
        print("\nEnrichment set already uses the deployed model for every target; nothing to replace.")

    # ---- recompute ML-vs-docking correlations -----------------------------
    vina = pd.read_csv(dock / "enrichment_vina_results.csv")
    gnina = pd.read_csv(dock / "enrichment_gnina_results.csv")
    ml = enrich[["target", "global_compound_id", "predicted_proba", "algorithm_used"]]

    rows = []
    for engine, frame, score_col in (("vina_affinity", vina, "vina_affinity"),
                                     ("gnina_cnn_score", gnina, "gnina_cnn_score")):
        merged = frame.merge(ml, on=["target", "global_compound_id"], how="left")
        for unit, sub in merged.groupby("unit"):
            sub = sub.dropna(subset=[score_col, "predicted_proba"])
            if len(sub) < 3:
                continue
            rho, p = stats.spearmanr(sub["predicted_proba"], sub[score_col])
            rows.append({
                "unit": unit, "score_type": engine, "n_compounds": int(len(sub)),
                "algorithm_used": str(sub["algorithm_used"].iloc[0]),
                "algorithm_provenance": "stamped from the leak-free deployed-model selection table",
                "spearman_rho": rho, "spearman_p": p,
            })

    corr = pd.DataFrame(rows).sort_values(["score_type", "unit"], ignore_index=True)
    old_corr = pd.read_csv(dock / "ml_docking_correlation.csv")
    compare = old_corr[["unit", "score_type", "spearman_rho"]].merge(
        corr[["unit", "score_type", "spearman_rho"]], on=["unit", "score_type"],
        suffixes=("_old", "_new"))
    compare["delta"] = compare["spearman_rho_new"] - compare["spearman_rho_old"]
    print("\nCorrelation changes (rows with |delta| > 1e-9):")
    moved = compare[compare["delta"].abs() > 1e-9]
    print(moved.round(4).to_string(index=False) if len(moved) else "  none")

    # ---- duplicate headline column ----------------------------------------
    final_path = dock / "docking_results_final.csv"
    final = pd.read_csv(final_path)
    drop_dup = ("headline_hit" in final.columns and "is_headline_hit" in final.columns
                and not args.keep_duplicate_headline_column)
    if drop_dup:
        a = final["is_headline_hit"].fillna(False).astype(bool)
        b = final["headline_hit"].fillna(False).astype(bool)
        if not (a == b).all():
            print("\n  'headline_hit' and 'is_headline_hit' DISAGREE; leaving both in place.")
            drop_dup = False
        else:
            print(f"\nDropping redundant 'headline_hit' column "
                  f"(identical to 'is_headline_hit', {int(a.sum())} hits).")

    if args.dry_run:
        print("\n[dry run] nothing written.")
        return 0

    # ---- write, keeping one-time backups ----------------------------------
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    written = []
    for path, frame in ((enrich_path, enrich), (dock / "ml_docking_correlation.csv", corr)):
        shutil.copy2(path, path.with_suffix(f".pre_repair_{stamp}.csv"))
        frame.to_csv(path, index=False)
        written.append(path)
    if drop_dup:
        shutil.copy2(final_path, final_path.with_suffix(f".pre_repair_{stamp}.csv"))
        final.drop(columns=["headline_hit"]).to_csv(final_path, index=False)
        written.append(final_path)

    # ---- keep the manifest truthful ---------------------------------------
    manifest_path = dock / "manifest_05_docking.json"
    manifest = json.load(open(manifest_path))
    outputs = manifest.get("outputs", {})
    refreshed = []
    for path in written:
        if path.name in outputs and isinstance(outputs[path.name], dict):
            outputs[path.name]["sha256"] = sha256_file(path)
            outputs[path.name]["bytes"] = path.stat().st_size
            refreshed.append(path.name)
    manifest["ml_side_repair"] = {
        "repaired_at": datetime.datetime.now().isoformat(),
        "reason": ("deployed classifier corrected for "
                   f"{', '.join(changed_targets) if changed_targets else 'no targets'}; "
                   "enrichment predictions and ML-vs-docking correlations rescored. "
                   "No docking was repeated."),
        "targets_rescored": changed_targets,
        "deployed_classifiers": deployed,
        "files_refreshed": refreshed,
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print("\nwrote:")
    for p in written:
        print(f"  {p}")
    print(f"  {manifest_path}  (hashes refreshed for: {', '.join(refreshed) or 'none'})")
    print(f"\nBackups saved with suffix .pre_repair_{stamp}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
