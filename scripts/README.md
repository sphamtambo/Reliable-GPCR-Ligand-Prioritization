# Analysis Scripts

The notebooks are the primary entry points for the analysis. These scripts provide
reproducible output assembly, validation, and selected supplementary analyses used
by the manuscript.

## Output assembly

- `assemble_main_text_figures.py` generates the main manuscript figures.
- `assemble_supplementary_figures.py` generates supplementary figures.
- `build_main_text_tables.py` generates the main manuscript tables.
- `build_supplementary_tables.py` generates supplementary tables.
- `build_interpreted_docking_hits_table.py` assembles the interpreted docking-hit table.

## Validation and provenance

- `create_analysis_freeze_manifest.py` records released artifact hashes and metadata.
- `check_docking_environment.py` checks docking dependencies and executable availability.
- `verify_ml_layer.py` checks key machine-learning artifacts and relationships.

## Supporting analyses

- `core_question_2_analysis.py` assembles the cross-target performance-driver analysis.
- `figure_recompute.py` recomputes selected figure inputs from saved model artifacts.
- `recompute_deployed_rf_shap.py` recomputes the deployed Random Forest SHAP analysis.
- `format_tables.py` applies the repository's table-formatting conventions.

These utilities expect the archived data and result artifacts described in the root
README. They do not replace the numbered notebook workflow and do not contain the
licensed DrugBank structure library.
