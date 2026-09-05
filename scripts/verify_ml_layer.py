#!/usr/bin/env python3
"""Verify the notebook-3/4 machine-learning layer is internally coherent.

Run this ON THE CLUSTER before syncing anything down, and again locally after
the sync. It answers one question: does `best_algorithm_by_combination.csv`
agree with the deployment tuning history it claims to be derived from?

Background
----------
Two copies of that table disagreed for ADORA2A and OPRM1. The table records
`selected_by = "deployment inner-CV best_value (leak-free)"`, and that value is
stored per model in each combination's `optuna_tuning_history.json`. So the
table is checkable: recompute the argmax and compare. This script does that for
every (target, pool, task), plus the artifact checks that make a layer usable.

Checks
------
1. selection table matches the deployment tuning-history argmax
2. the selected algorithm's model file exists for the deployed pool
3. feature_names.json length matches the model's n_features_in_
4. canonical split artifacts exist
5. final_model_test_performance.csv covers every combination
6. selection is NOT simply the held-out test argmax everywhere (leak smell test)

Exit code is non-zero if any hard check fails, so it can gate an rsync.

Usage
-----
    python3 scripts/verify_ml_layer.py
    python3 scripts/verify_ml_layer.py --project-dir /mnt/lustre/users/smtambo/gpcr_benchmark
    python3 scripts/verify_ml_layer.py --write-manifest
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd

TARGETS = ["drd2", "cb2", "adora2a", "oprm1", "ccr5"]
POOLS = ["ki", "ki_ic50", "full"]
TASKS = ["classification", "regression"]
DEPLOY_POOL = "full"
DEPLOY_REP = "combined"

ALGO_KEY_TO_LABEL = {"rf": "Random Forest", "xgb": "XGBoost", "lgb": "LightGBM"}
ALGO_LABEL_TO_FILE = {"Random Forest": "rf", "XGBoost": "xgb", "LightGBM": "lgb"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deployment_argmax(history: dict, task: str) -> tuple[str | None, dict]:
    """Recompute the deployment selection the table claims to record."""
    values = {}
    for key, entry in history.items():
        if not key.endswith(task):
            continue
        algo_key = key.split("_")[0]
        label = ALGO_KEY_TO_LABEL.get(algo_key)
        if label and isinstance(entry, dict) and "best_value" in entry:
            values[label] = float(entry["best_value"])
    if not values:
        return None, {}
    return max(values, key=values.get), values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--write-manifest", action="store_true",
                        help="write ml_layer_verification.json next to the selection table")
    args = parser.parse_args()

    root = Path(args.project_dir).resolve()
    results = root / "ml" / "results"
    models = root / "ml" / "models"
    table_path = results / "best_algorithm_by_combination.csv"

    print(f"Project: {root}")
    if not table_path.exists():
        print(f"FAIL: selection table not found at {table_path}")
        return 2

    table = pd.read_csv(table_path)
    print(f"Selection table: {table_path}  sha256={sha256_file(table_path)[:16]}  rows={len(table)}")
    print()

    hard_failures: list[str] = []
    soft_warnings: list[str] = []
    rows = []

    # ---- check 1: table vs deployment tuning history -------------------------
    for target in TARGETS:
        for pool in POOLS:
            history_path = results / f"{target}_{pool}" / "optuna_tuning_history.json"
            if not history_path.exists():
                soft_warnings.append(f"{target}/{pool}: no optuna_tuning_history.json (combination not tuned?)")
                continue
            history = json.load(open(history_path))
            for task in TASKS:
                recomputed, values = deployment_argmax(history, task)
                if recomputed is None:
                    soft_warnings.append(f"{target}/{pool}/{task}: no best_value entries in tuning history")
                    continue
                row = table[table["target"].eq(target) & table["activity_pool"].eq(pool) & table["task"].eq(task)]
                recorded = str(row.iloc[0]["best_algorithm"]) if len(row) else None
                agree = recorded == recomputed
                if not agree:
                    hard_failures.append(
                        f"{target}/{pool}/{task}: table says {recorded!r}, tuning history argmax is "
                        f"{recomputed!r}  ({', '.join(f'{k}={v:.5f}' for k, v in sorted(values.items()))})"
                    )
                rows.append({
                    "target": target, "activity_pool": pool, "task": task,
                    "recorded": recorded, "recomputed_from_tuning_history": recomputed,
                    "agree": agree,
                    **{f"best_value_{k}": v for k, v in values.items()},
                })

    check = pd.DataFrame(rows)
    if len(check):
        agree_n = int(check["agree"].sum())
        print(f"Check 1  selection table vs deployment tuning history: {agree_n}/{len(check)} agree")
        for _, r in check[~check["agree"]].iterrows():
            print(f"  MISMATCH {r['target']}/{r['activity_pool']}/{r['task']}: "
                  f"table={r['recorded']} history={r['recomputed_from_tuning_history']}")
    print()

    # ---- checks 2-4: deployed artifacts exist and are consistent -------------
    for target in TARGETS:
        row = table[table["target"].eq(target) & table["activity_pool"].eq(DEPLOY_POOL)
                    & table["task"].eq("classification")]
        if not len(row):
            hard_failures.append(f"{target}: no deployed classification row for pool {DEPLOY_POOL}")
            continue
        algo = str(row.iloc[0]["best_algorithm"])
        stem = ALGO_LABEL_TO_FILE.get(algo)
        model_dir = models / f"{target}_{DEPLOY_POOL}" / DEPLOY_REP
        model_file = model_dir / f"{stem}_clf.joblib"
        if not model_file.exists():
            hard_failures.append(f"{target}: deployed model file missing: {model_file}")
            continue
        names_path = model_dir / "feature_names.json"
        if names_path.exists():
            n_names = len(json.load(open(names_path)))
            try:
                import joblib
                n_model = getattr(joblib.load(model_file), "n_features_in_", None)
                if n_model is not None and n_model != n_names:
                    hard_failures.append(
                        f"{target}: feature_names.json has {n_names} names, model expects {n_model}")
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                soft_warnings.append(f"{target}: could not load model to check feature count ({exc})")
        else:
            soft_warnings.append(f"{target}: no feature_names.json in {model_dir}")

        for artifact in ("test_indices.npy", "train_test_split.csv"):
            path = results / f"{target}_{DEPLOY_POOL}" / artifact
            if not path.exists():
                hard_failures.append(f"{target}: missing canonical split artifact {artifact}")

    print(f"Checks 2-4  deployed model, feature names, canonical splits: "
          f"{'issues found' if hard_failures else 'all present'}")
    print()

    # ---- check 5: test-performance coverage ---------------------------------
    perf_path = results / "final_model_test_performance.csv"
    if perf_path.exists():
        perf = pd.read_csv(perf_path)
        missing = [t for t in TARGETS
                   if perf[perf["target"].eq(t) & perf["activity_pool"].eq(DEPLOY_POOL)
                           & perf["feature_representation"].eq(DEPLOY_REP)].empty]
        if missing:
            hard_failures.append(f"final_model_test_performance.csv missing deployed rows for: {missing}")
        print(f"Check 5  test-performance coverage: {len(TARGETS) - len(missing)}/{len(TARGETS)} targets")
    else:
        hard_failures.append("final_model_test_performance.csv not found")
        perf = None
    print()

    # ---- check 6: leak smell test -------------------------------------------
    # Selection legitimately agrees with the test argmax sometimes. Agreeing on
    # EVERY target is not proof of a leak, but it is worth surfacing, because a
    # selection computed on the test split would look exactly like this.
    if perf is not None:
        matches = 0
        for target in TARGETS:
            sub = perf[perf["target"].eq(target) & perf["activity_pool"].eq(DEPLOY_POOL)
                       & perf["feature_representation"].eq(DEPLOY_REP)]
            if sub.empty or "test_roc_auc" not in sub.columns:
                continue
            test_best = sub.loc[sub["test_roc_auc"].idxmax(), "algorithm"]
            row = table[table["target"].eq(target) & table["activity_pool"].eq(DEPLOY_POOL)
                        & table["task"].eq("classification")]
            if len(row) and str(row.iloc[0]["best_algorithm"]) == str(test_best):
                matches += 1
        print(f"Check 6  selection equals held-out test argmax for {matches}/{len(TARGETS)} targets")
        if matches == len(TARGETS):
            soft_warnings.append(
                "selection matches the test argmax for every target; confirm via check 1 that it "
                "was derived from the tuning history and not from test performance")
    print()

    # ---- report --------------------------------------------------------------
    if soft_warnings:
        print("Warnings:")
        for w in soft_warnings:
            print(f"  - {w}")
        print()
    if hard_failures:
        print("FAILURES:")
        for f in hard_failures:
            print(f"  - {f}")
        print()
        print("RESULT: FAIL — do not sync or publish from this layer until resolved.")
    else:
        print("RESULT: PASS — selection table is reproducible from the tuning history and "
              "deployed artifacts are present.")

    if args.write_manifest:
        out = {
            "verified_at": datetime.datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "project_dir": str(root),
            "selection_table": {"path": str(table_path), "sha256": sha256_file(table_path)},
            "n_combinations_checked": int(len(check)),
            "n_agree": int(check["agree"].sum()) if len(check) else 0,
            "hard_failures": hard_failures,
            "warnings": soft_warnings,
            "result": "FAIL" if hard_failures else "PASS",
            "deployed_classifiers": {
                t: str(table[table["target"].eq(t) & table["activity_pool"].eq(DEPLOY_POOL)
                             & table["task"].eq("classification")].iloc[0]["best_algorithm"])
                for t in TARGETS
                if len(table[table["target"].eq(t) & table["activity_pool"].eq(DEPLOY_POOL)
                             & table["task"].eq("classification")])
            },
        }
        manifest_path = results / "ml_layer_verification.json"
        with open(manifest_path, "w") as fh:
            json.dump(out, fh, indent=2)
        if len(check):
            check.to_csv(results / "ml_layer_selection_check.csv", index=False)
        print(f"\nwrote {manifest_path}")

    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
