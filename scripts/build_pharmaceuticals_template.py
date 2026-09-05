from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "pharmaceuticals_template_filled.md"

def read(name):
    return (ROOT / name).read_text()

title = read("manuscript/sections/title.md")
abstract = read("manuscript/sections/abstract.md")
intro = read("manuscript/sections/introduction.md")
methods = read("manuscript/sections/methods.md")
results = read("manuscript/sections/results.md")
discussion = read("manuscript/sections/discussion.md")
conclusion = read("manuscript/sections/conclusion.md")
backmatter = read("manuscript/sections/backmatter.md")

figures = {
    1: "outputs/figures/manuscript/figure1_framework_panel.png",
    2: "outputs/figures/manuscript/figure2_performance_scaling.png",
    3: "outputs/figures/manuscript/figure3_reliability_uncertainty.png",
    4: "outputs/figures/manuscript/figure4_representation_transferability.png",
    5: "outputs/figures/manuscript/figure5_screening_docking_funnel.png",
    6: "outputs/figures/manuscript/figure6_docking_validation_orthogonality.png",
}
tables = {
    1: "manuscript/tables/table1_dataset_characteristics_main.md",
    2: "manuscript/tables/table2_model_performance_main.md",
    3: "manuscript/tables/table3_external_validation_main.md",
    4: "manuscript/tables/table4_reliability_decision_readiness_main.md",
    5: "manuscript/tables/table5_explainability_feature_class_main.md",
    6: "manuscript/tables/table6_explainability_reliability_bridge_main.md",
    7: "manuscript/tables/table7_prioritization_hits_main.md",
}
captions = read("manuscript/table_captions.md")

def display_block(kind, n):
    if kind == "Figure":
        return f"\n![Figure {n}]({figures[n]})\n\n"
    path = tables[n]
    body = read(path)
    # Table source files carry their own caption/header; retain the table itself.
    return f"\n{body}\n\n"

for n in range(1, 7):
    results = results.replace(f"*[Insert Figure {n} here]*", display_block("Figure", n), 1)
for n in range(1, 8):
    marker = re.compile(rf"(\*\*Table {n}\.[\s\S]*?\*\*[^\n]*\n)")
    match = marker.search(results)
    if match:
        results = results[:match.end()] + display_block("Table", n) + results[match.end():]

# Remove internal Markdown title markers so the template controls numbering.
def strip_h1(text):
    return re.sub(r"^# (.+)$", r"\1", text, flags=re.MULTILINE)

parts = [
    title,
    strip_h1(abstract),
    strip_h1(intro),
    strip_h1(results),
    strip_h1(discussion),
    strip_h1(methods),
    strip_h1(conclusion),
    "## Patents\nNot applicable; no patents arise from the work reported here.",
    "## Supplementary Materials\nSupplementary Figures S1-S10 and Tables S1-S9 accompany this manuscript.",
    strip_h1(backmatter),
    "## Abbreviations\nAD, applicability domain; AUC, area under the receiver-operating-characteristic curve; GPCR, G protein-coupled receptor; ML, machine learning; SHAP, SHapley Additive exPlanations; RMSD, root-mean-square deviation; SI, supplementary information.",
    "## References\n[References are retained in the project reference file and will be inserted in the journal-required numbered format after bibliographic normalization.]",
]

OUT.write_text("\n\n".join(parts))
print(OUT)
