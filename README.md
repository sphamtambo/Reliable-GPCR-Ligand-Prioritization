# Reliable GPCR Ligand Prioritization

Code for a reliability-aware, uncertainty-calibrated machine learning and structure-based prioritization workflow across five pharmacologically distinct human GPCR targets: DRD2, CB2, ADORA2A, OPRM1, CCR5.

The pipeline curates ChEMBL bioactivity data per target, benchmarks Random Forest, XGBoost, and LightGBM under scaffold-disjoint validation, and deploys the pre-specified best algorithm for each target. The deployed models use Venn-Abers probability calibration, conformal prediction, and an applicability-domain reference before screening the DrugBank library. Candidates are evaluated by redocking-validated consensus docking with AutoDock Vina and GNINA, alongside external validation against BindingDB and a temporal never-seen holdout.

## Pipeline

Notebooks run in order:

| Notebook | Stage |
|---|---|
| `01_data_collection.ipynb` | ChEMBL bioactivity retrieval for the three net-new targets; reuses frozen DRD2 and CB2 inputs |
| `02_data_cleaning.ipynb` | Structure standardization, dedup, scaffold assignment, curation |
| `03_ml_benchmark.ipynb` | Model training, calibration, conformal prediction, applicability domain |
| `03b_zeroshot_transfer.ipynb` | Cross-target transfer, target-specificity control |
| `04_drugbank_screening.ipynb` | DrugBank virtual screening |
| `05_docking.ipynb` | Redocking validation, candidate docking, interaction checks |
| `06_external_validation.ipynb` | BindingDB and temporal never-seen validation |
| `07_meta_analysis.ipynb` | Cross-target synthesis, manuscript figures/tables |

## Directory structure

- `01`–`07` notebooks — the analysis pipeline (repository root)

The repository contains the analysis notebooks. Large generated directories (`data/`, `ml/`, `docking/`, and rendered figures) are distributed separately through the archived release described below. Running the notebooks in order against the corresponding archived inputs reproduces the downstream results.

## Reproducing the analysis

The notebooks are designed for a Python/Colab-style environment. Open them in the numbered order shown above, provide the archived project data at the path configured in each notebook, and run `01` through `07`; `03b` is the bounded zero-shot analysis and can be run after `03`. The notebooks install or verify their principal dependencies as needed. For a consolidated environment specification, see [`requirements.txt`](requirements.txt); [`requirements-lock.txt`](requirements-lock.txt) records the frozen package set used for the analysis.

The workflow is data-dependent and is not a one-command rerun: the archived ChEMBL/BindingDB-derived files, trained objects, docking outputs, and the licensed DrugBank export must be available before downstream notebooks can run. The notebooks record seeds, configuration, inputs, outputs, and hashes in their manifests where applicable.

## Data Availability

Raw/processed datasets, trained models, calibration and conformal objects, applicability-domain references, and docking results are archived on Zenodo: [DOI: 10.5281/zenodo.22336445](https://doi.org/10.5281/zenodo.22336445). The release currently includes the archive bundles `curated_datasets.zip`, `trained_models.zip`, `docking_data.zip`, and `benchmark_results.zip`; consult the Zenodo record for the definitive file list and version.

The raw DrugBank structure library is not redistributed due to licensing and must be obtained independently from [DrugBank](https://go.drugbank.com/releases/latest). Because ChEMBL, BindingDB, and DrugBank are versioned external resources, exact reruns require the archived inputs or the corresponding source releases used for the frozen analysis.

## Scope and limitations

This repository is an analysis release, not a packaged software library. It does not provide a command-line interface, guarantee identical results across software environments, or claim that docking confirms binding. The reported workflow is intended to support reproducible benchmarking and cautious ligand prioritization; computational candidates require independent experimental assessment.

## Citation and license

Please cite the associated publication and the Zenodo record when reusing this work. The repository currently does not declare a standalone software license; licensing and reuse conditions should therefore be taken from the Zenodo record and the licenses of the underlying data sources until a repository license file is added.
