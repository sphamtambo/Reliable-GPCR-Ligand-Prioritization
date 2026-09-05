#!/usr/bin/env python3
"""Recompute SHAP attributions for a DEPLOYED classifier whose stored SHAP
artifact was generated with a different algorithm.

Motivation
----------
`ml/results/<target>_<pool>/<rep>/shap_summary.json` records the algorithm the
stored SHAP arrays describe. For OPRM1 that algorithm is XGBoost, while the
deployed classifier selected by leak-free inner-CV is Random Forest. Any
explainability or explainability/reliability-bridge claim built on those arrays
therefore describes a model the study does not deploy.

This script recomputes SHAP for the deployed model only. It does NOT retrain
anything: it loads the persisted classifier, rebuilds (or loads) the exact
feature matrix, restricts to the canonical held-out test rows, and writes new
artifacts under distinct filenames so the original arrays are never overwritten.

It also fixes a weakness in the original artifacts: the stored SHAP arrays carry
no row index, so downstream analyses had to infer row correspondence from
matching counts. This script writes an explicit index CSV alongside the arrays.

Usage
-----
    python scripts/recompute_deployed_rf_shap.py --sample-size 500
    python scripts/recompute_deployed_rf_shap.py --all
    python scripts/recompute_deployed_rf_shap.py --target oprm1 --all
    python scripts/recompute_deployed_rf_shap.py --target oprm1 --expected-algorithm "Random Forest" --all

Outputs (suffix keeps them separate from the existing XGBoost arrays):
    ml/results/<combo>/<rep>/shap_values_deployed_rf.npy
    ml/results/<combo>/<rep>/shap_X_explain_deployed_rf.npy
    ml/results/<combo>/<rep>/shap_index_deployed_rf.csv
    ml/results/<combo>/<rep>/shap_summary_deployed_rf.json
    ml/results/<combo>/<rep>/manifest_shap_deployed_rf.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Feature recipe: must match notebook 3 exactly. A mismatched radius, bit count
# or descriptor order would silently produce attributions for a feature layout
# the deployed model never saw.
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
DESCRIPTOR_COLS = [
    "MW", "LogP", "TPSA", "HBD", "HBA",
    "RotBonds", "HeavyAtomCount", "RingCount", "AromaticRingCount", "FractionCSP3",
]
TOP_N_FEATURES = 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_deployed_algorithm(root: Path, target: str, pool: str) -> str:
    """Deployed algorithm comes from the leak-free selection table, never from
    held-out test performance."""
    path = root / "ml" / "results" / "best_algorithm_by_combination.csv"
    table = pd.read_csv(path)
    row = table[
        table["target"].eq(target)
        & table["activity_pool"].eq(pool)
        & table["task"].eq("classification")
    ]
    if row.empty:
        raise SystemExit(f"No deployed classification algorithm recorded for {target}/{pool}")
    return str(row.iloc[0]["best_algorithm"])


ALGO_TO_MODEL_FILE = {
    "Random Forest": "rf_clf.joblib",
    "XGBoost": "xgb_clf.joblib",
    "LightGBM": "lgb_clf.joblib",
}


def build_features_from_cleaned(cleaned_path: Path, feature_names: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Rebuild the combined feature matrix when ml/features/ is not present.

    Rebuilt in the order given by the model's own feature_names.json, then
    verified against it, so a silently reordered matrix cannot slip through.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")

    frame = pd.read_csv(cleaned_path)
    missing = [c for c in DESCRIPTOR_COLS if c not in frame.columns]
    if missing:
        raise SystemExit(f"Cleaned data is missing descriptor columns: {missing}")

    fingerprints = []
    keep_rows = []
    for i, smiles in enumerate(frame["clean_smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        bits = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
        fingerprints.append(np.frombuffer(bits.ToBitString().encode(), "u1") - ord("0"))
        keep_rows.append(i)

    if len(keep_rows) != len(frame):
        raise SystemExit(
            f"{len(frame) - len(keep_rows)} structures failed to parse; the rebuilt matrix "
            f"would not align with the stored split indices. Provide ml/features/ instead."
        )

    morgan = np.vstack(fingerprints).astype(float)
    descriptors = frame[DESCRIPTOR_COLS].to_numpy(dtype=float)
    matrix = np.hstack([descriptors, morgan])

    expected = DESCRIPTOR_COLS + [f"Morgan_{i}" for i in range(MORGAN_NBITS)]
    if list(feature_names) != expected:
        raise SystemExit(
            "Stored feature_names.json does not match the rebuilt descriptor+Morgan layout. "
            "Refusing to compute SHAP against an unverified feature ordering."
        )
    return matrix, frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", default=".", help="repository root (default: current directory)")
    parser.add_argument("--target", default="oprm1")
    parser.add_argument("--pool", default="full")
    parser.add_argument("--representation", default="combined")
    parser.add_argument(
        "--expected-algorithm",
        default=None,
        help=(
            "optional safety check; exit if the leak-free selection table does not "
            "name this classifier. Use this when the output suffix encodes a model "
            "family, e.g. --expected-algorithm 'Random Forest' with --suffix deployed_rf."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sample-size", type=int, default=500,
                       help="deterministic subsample of held-out test rows (default: 500)")
    group.add_argument("--all", action="store_true", help="explain every held-out test row")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", default="deployed_rf", help="output filename suffix")
    args = parser.parse_args()

    root = Path(args.project_dir).resolve()
    combo = f"{args.target}_{args.pool}"
    results_dir = root / "ml" / "results" / combo
    rep_dir = results_dir / args.representation
    models_dir = root / "ml" / "models" / combo / args.representation
    features_dir = root / "ml" / "features" / combo
    cleaned_path = root / "data" / "processed" / f"cleaned_data_{args.target}_{args.pool}.csv"

    print(f"Project: {root}")
    print(f"Combination: {combo} / {args.representation}")

    deployed = resolve_deployed_algorithm(root, args.target, args.pool)
    print(f"Deployed classifier (leak-free selection): {deployed}")
    if args.expected_algorithm and deployed != args.expected_algorithm:
        raise SystemExit(
            f"Selection table reports deployed algorithm {deployed!r}, but "
            f"--expected-algorithm is {args.expected_algorithm!r}. Refusing to write "
            f"{args.suffix!r} SHAP artifacts from the wrong model. Resolve which "
            "best_algorithm_by_combination.csv is authoritative before rerunning."
        )

    stored_summary = rep_dir / "shap_summary.json"
    if stored_summary.exists():
        stored_algo = json.load(open(stored_summary)).get("algorithm")
        print(f"Existing SHAP artifact algorithm: {stored_algo}")
        if stored_algo == deployed:
            print("  Existing SHAP already matches the deployed model; recomputation is optional.")

    model_file = ALGO_TO_MODEL_FILE.get(deployed)
    if model_file is None:
        raise SystemExit(f"No persisted model filename known for algorithm {deployed!r}")
    model_path = models_dir / model_file
    if not model_path.exists():
        raise SystemExit(f"Deployed model not found: {model_path}")

    import joblib
    model = joblib.load(model_path)
    feature_names = json.load(open(models_dir / "feature_names.json"))
    print(f"Loaded model: {model_path.name}  ({len(feature_names)} features)")

    # ---- feature matrix: prefer the persisted matrix, rebuild only if absent ----
    combined_npy = features_dir / f"{args.representation}.npy"
    if combined_npy.exists():
        X_all = np.load(combined_npy)
        frame = pd.read_csv(cleaned_path)
        print(f"Loaded persisted features: {combined_npy}  {X_all.shape}")
    else:
        print(f"Persisted features not found at {combined_npy}; rebuilding from cleaned data.")
        X_all, frame = build_features_from_cleaned(cleaned_path, feature_names)
        print(f"Rebuilt features: {X_all.shape}")

    if X_all.shape[1] != len(feature_names):
        raise SystemExit(f"Feature-count mismatch: matrix has {X_all.shape[1]}, model expects {len(feature_names)}")
    n_expected = getattr(model, "n_features_in_", X_all.shape[1])
    if X_all.shape[1] != n_expected:
        raise SystemExit(f"Model expects {n_expected} features, matrix has {X_all.shape[1]}")

    # ---- canonical held-out test rows ----
    test_idx = np.load(results_dir / "test_indices.npy")
    print(f"Canonical test rows: {len(test_idx)}")

    rng = np.random.default_rng(args.seed)
    if args.all:
        chosen = np.sort(test_idx)
    else:
        size = min(args.sample_size, len(test_idx))
        chosen = np.sort(rng.choice(test_idx, size=size, replace=False))
    print(f"Explaining {len(chosen)} rows (seed={args.seed}, mode={'all' if args.all else 'sample'})")

    X_explain = X_all[chosen]

    # ---- SHAP ----
    import shap
    explainer = shap.TreeExplainer(model)
    print("Computing SHAP values (tree_path_dependent)...")
    raw = explainer.shap_values(X_explain, check_additivity=False)

    # Binary classifiers return either a 2-element list or an (n, f, 2) array
    # depending on the shap/sklearn versions in use. Take the positive class.
    if isinstance(raw, list):
        shap_values = np.asarray(raw[1] if len(raw) > 1 else raw[0])
    elif getattr(raw, "ndim", 2) == 3:
        shap_values = np.asarray(raw[:, :, 1])
    else:
        shap_values = np.asarray(raw)
    print(f"SHAP array: {shap_values.shape}")

    # ---- row index, so downstream joins never infer order from row counts ----
    id_cols = [c for c in ("global_compound_id", "molecule_chembl_id") if c in frame.columns]
    index_frame = pd.DataFrame({"row_position": np.arange(len(chosen)), "feature_matrix_index": chosen})
    for col in id_cols:
        index_frame[col] = frame.iloc[chosen][col].to_numpy()

    # ---- self-check: descriptor column 0 must equal the source MW ----
    mw_diff = float("nan")
    if "MW" in frame.columns and feature_names[0] == "MW":
        mw_diff = float(np.nanmax(np.abs(frame.iloc[chosen]["MW"].to_numpy(float) - X_explain[:, 0])))
        if not mw_diff < 1e-6:
            raise SystemExit(f"Row-alignment check failed: max |MW difference| = {mw_diff}")
        print(f"Row-alignment check passed (max |MW difference| = {mw_diff:.2e})")

    # ---- summary in the same shape as the original shap_summary.json ----
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:TOP_N_FEATURES]
    summary = {
        "target": args.target,
        "activity_pool": args.pool,
        "feature_representation": args.representation,
        "algorithm": deployed,
        "explained_rows": int(len(chosen)),
        "explained_mode": "all" if args.all else f"sample_{len(chosen)}",
        "seed": args.seed,
        "top_features": [feature_names[i] for i in order],
        "top_features_indices": [int(i) for i in order],
        "mean_abs_shap": [float(mean_abs[i]) for i in order],
    }

    rep_dir.mkdir(parents=True, exist_ok=True)
    out_values = rep_dir / f"shap_values_{args.suffix}.npy"
    out_x = rep_dir / f"shap_X_explain_{args.suffix}.npy"
    out_index = rep_dir / f"shap_index_{args.suffix}.csv"
    out_summary = rep_dir / f"shap_summary_{args.suffix}.json"

    np.save(out_values, shap_values)
    np.save(out_x, X_explain)
    index_frame.to_csv(out_index, index=False)
    with open(out_summary, "w") as fh:
        json.dump(summary, fh, indent=2)

    manifest = {
        "analysis": "deployed-model SHAP recomputation",
        "created_at": datetime.datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "target": args.target,
        "activity_pool": args.pool,
        "feature_representation": args.representation,
        "deployed_algorithm": deployed,
        "deployed_algorithm_source": "ml/results/best_algorithm_by_combination.csv (leak-free inner-CV selection)",
        "superseded_artifact_algorithm": (json.load(open(stored_summary)).get("algorithm")
                                          if stored_summary.exists() else None),
        "model_file": str(model_path),
        "feature_source": str(combined_npy) if combined_npy.exists() else str(cleaned_path),
        "n_canonical_test_rows": int(len(test_idx)),
        "n_explained_rows": int(len(chosen)),
        "explained_mode": "all" if args.all else f"sample_{len(chosen)}",
        "seed": args.seed,
        "row_alignment_check": "passed" if mw_diff == mw_diff else "not applicable",
        "mw_alignment_max_abs_diff": mw_diff,
        "note": ("Original shap_values.npy is left untouched. These artifacts describe the "
                 "deployed classifier and carry an explicit row index, which the original "
                 "arrays do not."),
        "outputs": {p.name: {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                    for p in (out_values, out_x, out_index, out_summary)},
    }
    manifest_path = rep_dir / f"manifest_shap_{args.suffix}.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print()
    for p in (out_values, out_x, out_index, out_summary, manifest_path):
        print(f"wrote {p}")
    print()
    print("Top features (deployed model):")
    for name, value in zip(summary["top_features"][:10], summary["mean_abs_shap"][:10]):
        print(f"  {name:<22} {value:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
