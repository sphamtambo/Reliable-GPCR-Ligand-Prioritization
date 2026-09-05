#!/usr/bin/env python3
"""Create an analysis freeze manifest for manuscript-facing artifacts.

The manifest is intentionally file-level and hash-based: it records the exact
notebooks, scripts, result tables, docking outputs, figures, and manuscript
sources used for the current manuscript vintage. It does not include trained
model binaries by default; those are already captured by notebook-level
manifests and are too large for a lightweight manuscript freeze.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd


DEFAULT_ROOTS = [
    "01_data_collection.ipynb",
    "02_data_processing.ipynb",
    "03_ml_benchmark.ipynb",
    "03b_zero_shot_transfer.ipynb",
    "04_drugbank_screening.ipynb",
    "05_docking.ipynb",
    "06_external_validation.ipynb",
    "07_meta_analysis.ipynb",
    "scripts",
    "ml/results",
    "docking/docking_results",
    "outputs",
    "manuscript/sections",
    "manuscript/tables",
    "manuscript/figure_captions.md",
    "manuscript/table_captions.md",
    "manuscript/figure_table_plan.md",
    "manuscript/manuscript_readthrough_draft.md",
    "manuscript/CHANGELOG_2026-08-30_deployed_model_correction.md",
]

INCLUDE_SUFFIXES = {
    ".csv", ".tsv", ".json", ".md", ".txt", ".ipynb", ".py",
    ".png", ".pdf", ".sdf", ".pdbqt",
}

EXCLUDE_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
    "archive",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_count(path: Path) -> int | None:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return None
    try:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return len(pd.read_csv(path, sep=sep))
    except Exception:
        return None


def iter_files(root: Path, entries: list[str]):
    seen: set[Path] = set()
    for entry in entries:
        p = root / entry
        if not p.exists():
            continue
        candidates = [p] if p.is_file() else p.rglob("*")
        for file_path in candidates:
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(root)
            if any(part in EXCLUDE_PARTS for part in rel.parts):
                continue
            if file_path.suffix.lower() not in INCLUDE_SUFFIXES:
                continue
            if rel in seen:
                continue
            seen.add(rel)
            yield rel, file_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--name", default="analysis_freeze_2026-08-30_corrected_vintage")
    parser.add_argument(
        "--note",
        default=("Corrected post-deployed-model, DrugBank screening, docking, "
                 "external-validation, meta-analysis, figure, table, and manuscript vintage."),
    )
    args = parser.parse_args()

    root = Path(args.project_dir).resolve()
    out_base = root / "manuscript" / "analysis_freeze_manifest_2026-08-30"

    records = []
    for rel, file_path in sorted(iter_files(root, DEFAULT_ROOTS), key=lambda x: str(x[0])):
        records.append({
            "path": str(rel),
            "bytes": file_path.stat().st_size,
            "sha256": sha256_file(file_path),
            "n_rows": row_count(file_path),
        })

    manifest = {
        "freeze_name": args.name,
        "created_at": dt.datetime.now().isoformat(),
        "project_dir": str(root),
        "revision_note": args.note,
        "n_files": len(records),
        "selection_table": next(
            (r for r in records if r["path"] == "ml/results/best_algorithm_by_combination.csv"),
            None,
        ),
        "critical_counts": {},
        "files": records,
    }

    funnel = root / "ml/results/meta_analysis/screening_docking_funnel.csv"
    if funnel.exists():
        df = pd.read_csv(funnel)
        manifest["critical_counts"]["drugbank_high_confidence_target_novel_rows"] = int(
            df["n_high_confidence_novel_rows"].sum()
        )
        manifest["critical_counts"]["docking_headline_hits"] = int(df["n_headline_hits"].sum())

    json_path = out_base.with_suffix(".json")
    csv_path = out_base.with_suffix(".csv")
    json_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "bytes", "sha256", "n_rows"])
        writer.writeheader()
        writer.writerows(records)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"files: {len(records)}")
    if manifest["critical_counts"]:
        print("critical counts:", manifest["critical_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
