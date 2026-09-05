"""
Assemble manuscript-facing five-target GPCR figures from persisted source data.

This script does not rerun analysis notebooks. It reads frozen CSV outputs and
writes polished PNG/PDF figures to outputs/figures/manuscript/. Each assembled
figure gets a JSON manifest with source paths and SHA-256 hashes.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / ".matplotlib-cache").mkdir(exist_ok=True)
(ROOT / ".cache").mkdir(exist_ok=True)

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
GREY = "#9aa0a6"
DARK = "#2f3437"
LIGHT_GREY = "#e5e7eb"
RED = "#d62728"

# Colour semantics are fixed once, globally, so the same hue never means two
# different things across panels of the same figure:
#   TARGET_COLORS  -> one colour per receptor, used only for per-target series
#   STRATUM_LIGHT/DARK -> the two conditions of one quantity (raw vs calibrated,
#                     inside vs outside the applicability domain)
TARGET_COLORS = {
    "drd2": "#4c78a8",
    "cb2": "#59a14f",
    "adora2a": "#f28e2b",
    "oprm1": "#b07aa1",
    "ccr5": "#e15759",
}
PANEL_LETTER_SIZE = 18
STRATUM_LIGHT = "#a8c5e0"
STRATUM_DARK = "#1f4e79"
NEUTRAL_BAR = "#5b7c99"

mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, stem):
    outputs = []
    for ext in ("png", "pdf"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path)
        outputs.append(path)
    plt.close(fig)
    for path in outputs:
        print(f"wrote {path}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(stem, sources, outputs, notes, deterministic=True):
    records = []
    for path in sources:
        p = Path(path)
        records.append(
            {
                "path": str(p.relative_to(ROOT)),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
        )
    out_records = []
    for path in outputs:
        p = Path(path)
        out_records.append(
            {
                "path": str(p.relative_to(ROOT)),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
        )
    manifest = {
        "figure": stem,
        "deterministic": deterministic,
        "analysis_rerun": False,
        "sources": records,
        "outputs": out_records,
        "notes": notes,
    }
    path = OUT / f"{stem}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {path}")


def panel_label(ax, label):
    # Axes3D.text() has the signature (x, y, z, s); only text2D takes (x, y, s)
    # with an axes transform, so 3D panels must use it or the call raises.
    text_fn = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_fn(
        -0.08,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )


def draw_box(ax, xy, text, color, width=0.17, height=0.18, fontsize=8.5):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=DARK,
        facecolor=color,
        alpha=0.95,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="white" if color != LIGHT_GREY else DARK,
        transform=ax.transAxes,
    )
    return (x, y, width, height)


def draw_arrow(ax, start_box, end_box):
    sx, sy, sw, sh = start_box
    ex, ey, ew, eh = end_box
    if ex >= sx:
        start = (sx + sw, sy + sh / 2)
        end = (ex, ey + eh / 2)
    else:
        start = (sx, sy + sh / 2)
        end = (ex + ew, ey + eh / 2)
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.0,
        color=DARK,
        transform=ax.transAxes,
    )
    ax.add_patch(arr)


def stage_counts():
    """Per-stage counts for the Figure 1A schematic, read from frozen artifacts so
    every box carries a number a reader can check against a file."""
    ds = pd.read_csv(ROOT / "data" / "processed" / "dataset_summary_all_targets_pools.csv")
    full = ds[ds["activity_pool"].eq("full")]
    ext = pd.read_csv(ROOT / "ml" / "results" / "external_validation" / "external_validation_summary.csv")
    never_seen = ext[ext["axis"].eq("temporal") & ext["stratum"].str.contains("never_seen", na=False)]
    bindingdb = ext[ext["axis"].eq("bindingdb")]
    algos = pd.read_csv(ROOT / "ml" / "results" / "best_algorithm_by_combination.csv")
    deployed = algos[algos["activity_pool"].eq("full") & algos["task"].eq("classification")]
    funnel = pd.read_csv(ROOT / "ml" / "results" / "meta_analysis" / "screening_docking_funnel.csv")
    redock = pd.read_csv(ROOT / "docking" / "docking_results" / "redocking_validation.csv")
    library = pd.read_csv(
        ROOT / "ml" / "results" / "drugbank_screening" / "drugbank_library_standardised.csv",
        usecols=[0],
    )
    return {
        "targets": "5 receptor families",
        "raw_records": f"{int(full['raw_records'].sum()):,} activity records",
        "pools": "3 endpoint pools",
        "representations": "3 representations",
        # Two lines for the longer stages: on one line the row-2 counts ran into
        # each other between adjacent boxes.
        "models": f"{len(deployed)} deployed models,\n3 algorithms",
        "uncertainty": "Venn-Abers, conformal,\napplicability domain",
        "external": f"{int(never_seen['n'].sum()):,} never-seen,\n{int(bindingdb['n'].sum())} independent",
        "docking": f"{len(library):,} screened,\n{int(funnel['n_headline_hits'].sum())} prioritized",
    }


def figure1_framework_panel():
    """Figure 1: workflow schematic and benchmark-panel construction."""
    endpoint_src = ROOT / "data" / "processed" / "endpoint_composition_summary.csv"
    balance_src = ROOT / "data" / "processed" / "figure_class_balance_data.csv"
    pca_src = ROOT / "data" / "processed" / "chemical_space_pca_coordinates.csv"
    variance_src = ROOT / "data" / "processed" / "chemical_space_pca_variance.csv"

    endpoint = pd.read_csv(endpoint_src)
    balance = pd.read_csv(balance_src)
    pca = pd.read_csv(pca_src)
    variance = pd.read_csv(variance_src)

    targets = ["DRD2", "CB2", "ADORA2A", "OPRM1", "CCR5"]
    target_keys = [t.lower() for t in targets]
    color_map = TARGET_COLORS

    fig = plt.figure(figsize=(9.8, 6.9))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.70, 1.30],
        width_ratios=[1.0, 1.0],
        hspace=0.16,
        wspace=0.30,
    )

    ax = fig.add_subplot(gs[0, :])
    ax_schematic = ax
    ax.axis("off")
    # Stage counts are read from the frozen artifacts rather than typed in, so
    # every box in the schematic is auditable against a file.
    counts = stage_counts()
    boxes = []
    labels = [
        ("Five GPCR\ntargets", BLUE, counts["targets"]),
        ("ChEMBL\ncuration", DARK, counts["raw_records"]),
        ("Endpoint\npools", GREY, counts["pools"]),
        ("Feature\nsets", ORANGE, counts["representations"]),
        ("Scaffold-split\nbenchmark", BLUE, counts["models"]),
        ("Uncertainty and\napplicability domain", GREEN, counts["uncertainty"]),
        ("External\nvalidation", ORANGE, counts["external"]),
        ("DrugBank screening\nand docking", RED, counts["docking"]),
    ]
    # Both rows read left to right; the wrap between them is an explicit elbow,
    # so the eye never has to track the workflow backwards.
    # Block sits higher in its axes (there was dead space above row 1), with a
    # wider gap between the two rows, and shifted slightly left toward the panel
    # letter.
    positions = [
        (0.0, 0.72), (0.255, 0.72), (0.510, 0.72), (0.765, 0.72),
        (0.0, 0.22), (0.255, 0.22), (0.510, 0.22), (0.765, 0.22),
    ]
    for xy, (text, color, note) in zip(positions, labels):
        box = draw_box(ax, xy, text, color, width=0.212, height=0.26, fontsize=9.2)
        boxes.append(box)
        ax.text(
            xy[0] + 0.19 / 2,
            xy[1] - 0.090,
            note,
            ha="center",
            va="top",
            fontsize=9.0,
            color=DARK,
            transform=ax.transAxes,
        )
    for left, right in zip(boxes[:3], boxes[1:4]):
        draw_arrow(ax, left, right)
    for left, right in zip(boxes[4:7], boxes[5:8]):
        draw_arrow(ax, left, right)
    # No wrap connector. The row-1-to-row-2 return line had to run the full width
    # of the panel backwards, which is the one direction a reader does not scan.
    # Numbering each stage carries the ordering without the doubling back.
    for idx, (box, (text, color, note)) in enumerate(zip(boxes, labels), start=1):
        bx, by, bw, bh = box
        ax.text(
            bx + 0.010,
            by + bh - 0.04,
            str(idx),
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color=DARK if color in (GREY, LIGHT_GREY) else "white",
            transform=ax.transAxes,
        )

    ax = fig.add_subplot(gs[1, 0])
    ax_bar = ax
    full_endpoint = endpoint[endpoint["activity_pool"].eq("full")].set_index("target").reindex(target_keys)
    full_balance = balance[balance["activity_pool"].eq("full")].set_index("target").reindex(target_keys)
    x = np.arange(len(targets))
    bottom = np.zeros(len(targets))
    measures = [
        ("pct_retained_ki_measurements", "Ki", BLUE),
        ("pct_retained_ic50_measurements", "IC50", ORANGE),
        ("pct_retained_ec50_measurements", "EC50", GREEN),
    ]
    for col, label, color in measures:
        vals = full_endpoint[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.8, label=label)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Retained measurements (%)")
    ax.set_ylim(0, 100)
    # Active-class prevalence is a different quantity from endpoint composition,
    # so it gets its own axis instead of sharing the stacked 0-100% scale.
    ax_active = ax.twinx()
    ax_active.spines["right"].set_visible(True)
    line, = ax_active.plot(
        x, full_balance["active_pct"], color=DARK, marker="o", linewidth=2.0,
        markersize=5, label="Active compounds",
    )
    ax_active.set_ylim(50, 90)
    ax_active.set_ylabel("Active compounds (%)")
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [line],
        labels=labels_ + ["Active compounds"],
        frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13),
    )

    # One mini-panel per target against the pooled cloud in grey. A single
    # overlaid contour plot showed five nearly coincident densities, which reads
    # as "these targets are identical" rather than showing where each one sits.
    inner = gs[1, 1].subgridspec(2, 3, wspace=0.10, hspace=0.34)
    pc1 = variance.loc[variance["component"].eq("PC1"), "explained_variance_ratio"].iloc[0] * 100
    pc2 = variance.loc[variance["component"].eq("PC2"), "explained_variance_ratio"].iloc[0] * 100
    xlim = tuple(np.nanpercentile(pca["PC1"], [1, 99]))
    ylim = tuple(np.nanpercentile(pca["PC2"], [1, 99]))
    pooled = pca.sample(n=min(6000, len(pca)), random_state=42)
    first_ax = None
    for idx, (target, label) in enumerate(zip(target_keys, targets)):
        sub_ax = fig.add_subplot(inner[idx // 3, idx % 3])
        if first_ax is None:
            first_ax = sub_ax
        sub_ax.scatter(pooled["PC1"], pooled["PC2"], s=2, color="#dcdfe4", edgecolors="none", rasterized=True)
        sub = pca[pca["target"].eq(target)]
        sub_sample = sub.sample(n=min(2500, len(sub)), random_state=42)
        sub_ax.scatter(
            sub_sample["PC1"], sub_sample["PC2"], s=2.5, alpha=0.55,
            color=TARGET_COLORS[target], edgecolors="none", rasterized=True,
        )
        sub_ax.set_xlim(*xlim)
        sub_ax.set_ylim(*ylim)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        sub_ax.set_title(label, fontsize=9.5, pad=3)
        for spine in sub_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("#c9ced6")
    axis_ax = fig.add_subplot(inner[1, 2])
    axis_ax.axis("off")
    axis_ax.text(
        0.02, 0.72,
        f"PC1: {pc1:.1f}% of variance\nPC2: {pc2:.1f}% of variance",
        fontsize=9, va="top", transform=axis_ax.transAxes, linespacing=1.5,
    )
    axis_ax.text(
        0.02, 0.34,
        "Grey: pooled benchmark\nColour: that target",
        fontsize=9, color=DARK, va="top", transform=axis_ax.transAxes, linespacing=1.5,
    )
    ax = first_ax
    fig.tight_layout()
    # Nudge the schematic left as a whole rather than moving its contents past
    # the axes edge, which clips the boxes.
    pos_a = ax_schematic.get_position()
    ax_schematic.set_position([pos_a.x0 - 0.048, pos_a.y0 + 0.058, pos_a.width, pos_a.height])
    pos_bar = ax_bar.get_position()
    pos_tile = first_ax.get_position()
    letter_y = max(pos_bar.y1, pos_tile.y1) + 0.055
    for letter, lx, ly in (
        ("A", 0.012, 0.985),
        ("B", 0.012, letter_y),
        ("C", pos_tile.x0 - 0.045, letter_y),
    ):
        fig.text(lx, ly, letter, fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="top", ha="left")
    stem = "figure1_framework_panel"
    save(fig, stem)
    write_manifest(
        stem,
        [endpoint_src, balance_src, pca_src, variance_src],
        [OUT / f"{stem}.png", OUT / f"{stem}.pdf"],
        [
            "Panel A is a schematic drawn from the verified Figure 1A text spec, not a new analysis; stage counts are read from the frozen artifacts at build time.",
            "Figure canvas is sized near final print width so point sizes are close to what prints; check any resize against a 5 pt floor.",
            "Panel B shows full-pool retained endpoint composition with active-class percentage overlaid.",
            "Panel C shows one deterministic downsample per target against the pooled benchmark cloud; overlaid contours were replaced because five near-coincident densities obscured per-target position.",
        ],
    )


def figure2_performance_scaling():
    """Figure 2: performance, learning curves, and scaffold-stratified ROC-AUC."""
    perf_src = ROOT / "outputs" / "tables" / "table_model_performance_landscape.csv"
    learn_src = ROOT / "ml" / "results" / "learning_curves.csv"
    scaffold_src = ROOT / "ml" / "results" / "scaffold_performance_breakdown.csv"

    df = pd.read_csv(perf_src)
    learn = pd.read_csv(learn_src)
    scaffold = pd.read_csv(scaffold_src)
    targets = df["target"].tolist()
    x = np.arange(len(df))
    width = 0.23

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.9, 4.4),
        gridspec_kw={"width_ratios": [1.32, 1.12, 1.2], "wspace": 0.22},
    )

    # One tick size for all three panels: at the default 10 pt the five target
    # names in panel A overlap at this width, and shrinking only that panel would
    # leave three different tick sizes in one figure.
    for panel_ax in axes:
        panel_ax.tick_params(labelsize=10.5)

    ax = axes[0]
    series = [
        ("internal_test_roc_auc", None, "Internal, scaffold-split", BLUE),
        ("temporal_roc_auc", None, "Temporal never-seen", ORANGE),
        ("bindingdb_roc_auc", "bindingdb_n", "BindingDB", GREEN),
    ]
    for i, (col, n_col, label, color) in enumerate(series):
        offset = (i - 1) * width
        values = pd.to_numeric(df[col], errors="coerce")
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        if col == "bindingdb_roc_auc":
            weak = df["bindingdb_evidence_strength"].eq("weak (small n)")
            for bar, is_weak in zip(bars, weak):
                if is_weak:
                    bar.set_facecolor("#b7d9b0")
                    bar.set_edgecolor(DARK)
                    bar.set_linewidth(0.7)
                    bar.set_hatch("////")
        if n_col:
            for xi, value, n in zip(x, values, pd.to_numeric(df[n_col], errors="coerce")):
                if pd.notna(value) and pd.notna(n):
                    ax.text(
                        xi + offset,
                        value + 0.012,
                        f"n={int(n)}",
                        ha="center",
                        va="bottom",
                        fontsize=9.5,
                        fontweight="bold",
                        rotation=90,
                    )

    ax.axhline(0.5, color=GREY, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.04)
    # Handles are built explicitly. Reading them back from the axes returns the
    # BindingDB bar container, whose first bar is a hatched small-n bar, so the
    # legend swatch for the de-overlapped series came out hatched and identical
    # to the small-n entry, defeating the point of hatching at all.
    handles = [
        Patch(facecolor=BLUE, edgecolor="white", label="Internal, scaffold-split"),
        Patch(facecolor=ORANGE, edgecolor="white", label="Temporal never-seen"),
        Patch(facecolor=GREEN, edgecolor="white", label="BindingDB"),
        Patch(facecolor="#b7d9b0", edgecolor=DARK, hatch="////", label="BindingDB, small n"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        handlelength=1.6,
        fontsize=9.5,
    )

    ax = axes[1]
    full_learn = learn[learn["activity_pool"].eq("full")].copy()
    color_map = {t.upper(): c for t, c in TARGET_COLORS.items()}
    for target in targets:
        sub = full_learn[full_learn["target"].str.upper().eq(target)].sort_values("train_fraction")
        ax.plot(
            sub["train_fraction"] * 100,
            sub["test_roc_auc"],
            marker="o",
            linewidth=1.6,
            markersize=4,
            color=color_map.get(target, BLUE),
            label=target,
        )
    ax.axhline(0.5, color=GREY, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("Training data used (%)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.72, 0.97)
    ax.set_xticks([10, 25, 50, 75, 100])
    # Legend below the axes, matching panel A. Parked at the right it reserved a
    # column of empty space between panels B and C.
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        handlelength=1.6,
        columnspacing=1.4,
        fontsize=9.5,
    )

    ax = axes[2]
    sc = scaffold[
        scaffold["activity_pool"].eq("full")
        & scaffold["feature_representation"].eq("combined")
        & scaffold["classification_roc_auc"].notna()
        & scaffold["scaffold_stratum"].isin(["singleton_scaffold", "non_singleton_scaffold"])
    ].copy()
    sc["target_upper"] = sc["target"].str.upper()
    singleton = (
        sc[sc["scaffold_stratum"].eq("singleton_scaffold")]
        .set_index("target_upper")
        .reindex(targets)["classification_roc_auc"]
    )
    nonsingleton = (
        sc[sc["scaffold_stratum"].eq("non_singleton_scaffold")]
        .set_index("target_upper")
        .reindex(targets)["classification_roc_auc"]
    )
    # Plotting the difference rather than two bars on a 0.82-0.95 axis: the claim
    # is directional ("singletons are not uniformly harder"), and a truncated
    # absolute axis turns gaps of a few thousandths into visual chasms.
    delta = (singleton - nonsingleton).to_numpy()
    ax.barh(x, delta, height=0.52, color=NEUTRAL_BAR, edgecolor="white", linewidth=0.8)
    ax.axvline(0.0, color=DARK, linewidth=1.0)
    for xi, (d, s_val, n_val) in enumerate(zip(delta, singleton.to_numpy(), nonsingleton.to_numpy())):
        if not np.isfinite(d):
            continue
        offset = 0.005 if d >= 0 else -0.005
        ax.text(
            d + offset, xi, f"{s_val:.3f} vs {n_val:.3f}",
            va="center", ha="left" if d >= 0 else "right", fontsize=9,
            fontweight="bold", color=DARK,
        )
    ax.set_yticks(x)
    ax.set_yticklabels(targets)
    # Kept short: a two-line label on the rightmost panel overran the figure edge.
    ax.set_xlabel("Singleton minus non-singleton ROC-AUC\n(singleton vs non-singleton values shown)", fontsize=9.5)
    span = float(np.nanmax(np.abs(delta))) * 3.0
    ax.set_xlim(-span, span)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.subplots_adjust(left=0.055, right=0.995)
    # Letters in figure coordinates: placed per-axes they get pushed right by the
    # width of each y-axis label, so the three no longer line up.
    letter_y = max(a.get_position().y1 for a in axes) + 0.085
    for letter, target_ax in zip("ABC", axes):
        lx = 0.012 if target_ax is axes[0] else target_ax.get_position().x0 - 0.030
        fig.text(lx, letter_y, letter, fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="top", ha="left")

    stem = "figure2_performance_scaling"
    save(fig, stem)
    write_manifest(
        stem,
        [perf_src, learn_src, scaffold_src],
        [OUT / f"{stem}.png", OUT / f"{stem}.pdf"],
        [
            "Panel A hatches BindingDB bars flagged as weak small-n evidence in the frozen performance source.",
            "Panel B uses full-pool learning curves only.",
            "Panel C shows the singleton minus non-singleton ROC-AUC difference per target; the absolute stratum values are DRD2 0.922/0.859, CB2 0.896/0.918, ADORA2A 0.914/0.929, OPRM1 0.920/0.918, CCR5 0.910/0.854 and belong in the caption; positive values mean singleton scaffolds scored higher. A difference encoding avoids the truncated absolute axis a two-bar version would need.",
        ],
    )


def figure4_representation_transferability():
    """Figure 4: representation performance and descriptor transferability."""
    rep_src = ROOT / "ml" / "results" / "meta_analysis" / "feature_representation_comparison.csv"
    heat_src = ROOT / "ml" / "results" / "meta_analysis" / "shap_descriptor_heatmap_combined.csv"
    transfer_src = ROOT / "ml" / "results" / "meta_analysis" / "feature_transferability_by_type.csv"

    rep = pd.read_csv(rep_src)
    heat = pd.read_csv(heat_src).set_index("feature")
    transfer = pd.read_csv(transfer_src)

    targets = ["DRD2", "CB2", "ADORA2A", "OPRM1", "CCR5"]
    reps = [
        ("morgan", "Morgan", GREY),
        ("descriptors", "Descriptors", ORANGE),
        ("combined", "Combined", BLUE),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.8, 4.6),
        gridspec_kw={"width_ratios": [1.0, 0.88], "wspace": 0.42},
    )

    for panel_ax in axes:
        panel_ax.tick_params(labelsize=10.5)

    ax = axes[0]
    x = np.arange(len(targets))
    width = 0.25
    for i, (rep_key, label, color) in enumerate(reps):
        sub = (
            rep[rep["feature_representation"].eq(rep_key)]
            .set_index("target")
            .reindex(targets)
        )
        ax.bar(
            x + (i - 1) * width,
            sub["test_roc_auc_deployed_algorithm"],
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
    ax.axhline(0.5, color=GREY, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Held-out ROC-AUC")
    ax.set_ylim(0.68, 0.995)
    ax.legend(
        frameon=False,
        loc="upper center",
        ncol=3,
        handlelength=1.6,
        fontsize=9.5,
        borderaxespad=0.4,
        columnspacing=1.6,
    )

    ax = axes[1]
    heat = heat.reindex(
        [
            "FractionCSP3",
            "MW",
            "LogP",
            "HeavyAtomCount",
            "TPSA",
            "HBD",
            "HBA",
            "RingCount",
            "AromaticRingCount",
            "RotBonds",
        ]
    )
    heat = heat[targets]
    data = heat.astype(float).to_numpy()
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f2f3f5")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=np.nanmax(data))
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white" if value > np.nanmax(data) * 0.58 else DARK,
                )
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cb.set_label("Share of top-feature importance", fontweight="bold")
    ax.set_xlabel("Target")

    fig.tight_layout()
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.125)
    for letter, target_ax in zip("AB", axes):
        pos = target_ax.get_position()
        # Panel B's letter sits just outside the heatmap's own left edge, not out
        # beyond its row labels.
        lx = 0.008 if pos.x0 < 0.4 else pos.x0 - 0.022
        fig.text(lx, pos.y1 + 0.012, letter, fontsize=PANEL_LETTER_SIZE,
                 fontweight="bold", va="bottom", ha="left")

    stem = "figure4_representation_transferability"
    save(fig, stem)
    write_manifest(
        stem,
        [rep_src, heat_src, transfer_src],
        [OUT / f"{stem}.png", OUT / f"{stem}.pdf"],
        [
            "Panel A uses deployed-algorithm held-out ROC-AUC by feature representation.",
            "Panel B uses combined-representation descriptor SHAP top-feature shares.",
            "Mean targets shared per feature type remains in the manifest source table and should be reported in the caption or Table 5, not inside the figure body.",
        ],
    )


def figure3_reliability_uncertainty():
    """Figure 3: calibration, conformal coverage, confidence, and AD error."""
    cal_src = ROOT / "ml" / "results" / "meta_analysis" / "calibration_across_targets.csv"
    conformal_src = ROOT / "ml" / "results" / "meta_analysis" / "conformal_ad_stratified_effects.csv"
    conf_src = ROOT / "ml" / "results" / "meta_analysis" / "confidence_stratified_accuracy.csv"
    mono_src = ROOT / "ml" / "results" / "meta_analysis" / "confidence_accuracy_monotonicity.csv"
    ad_src = ROOT / "ml" / "results" / "meta_analysis" / "applicability_domain_error_analysis.csv"

    cal = pd.read_csv(cal_src)
    conformal = pd.read_csv(conformal_src)
    conf = pd.read_csv(conf_src)
    mono = pd.read_csv(mono_src)
    ad = pd.read_csv(ad_src)

    targets = ["DRD2", "CB2", "ADORA2A", "OPRM1", "CCR5"]
    x = np.arange(len(targets))
    color_map = {t.upper(): c for t, c in TARGET_COLORS.items()}

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.6))
    for panel_ax in axes.ravel():
        panel_ax.tick_params(labelsize=10.5)

    ax = axes[0, 0]
    cal = cal.set_index("target").reindex(targets).reset_index()
    width = 0.36
    ax.bar(x - width / 2, cal["ece_raw"], width=width, color=STRATUM_LIGHT, edgecolor="white", label="Raw")
    ax.bar(x + width / 2, cal["ece_calibrated"], width=width, color=STRATUM_DARK, edgecolor="white", label="Venn-Abers")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Expected calibration error")
    ax.set_ylim(0, max(cal["ece_calibrated"].max(), cal["ece_raw"].max()) * 1.25)
    ax.legend(frameon=False, fontsize=9.5)

    ax = axes[0, 1]
    reg = conformal[conformal["task"].eq("regression")].set_index("target").reindex(targets).reset_index()
    ax.bar(x - width / 2, reg["empirical_coverage_within_ad"], width=width, color=STRATUM_LIGHT, edgecolor="white", label="Inside applicability domain")
    ax.bar(x + width / 2, reg["empirical_coverage_outside_ad"], width=width, color=STRATUM_DARK, edgecolor="white", label="Outside applicability domain")
    ax.axhline(0.90, color=RED, linestyle="--", linewidth=1.2, label="Nominal 90%")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Regression conformal coverage")
    ax.set_ylim(0.78, 1.005)
    ax.legend(
        handles=[
            Line2D([0], [0], color=RED, linestyle="--", linewidth=1.2, label="Nominal 90% coverage"),
            Patch(facecolor=STRATUM_LIGHT, edgecolor="white", label="Inside applicability domain"),
            Patch(facecolor=STRATUM_DARK, edgecolor="white", label="Outside applicability domain"),
        ],
        frameon=False, fontsize=8.5, loc="upper left", handlelength=1.6,
        borderaxespad=0.3, labelspacing=0.35,
    )

    ax = axes[1, 0]
    for target in targets:
        sub = conf[conf["target"].eq(target)].sort_values("bin_lower")
        ax.plot(
            sub["bin_lower"] + 0.05,
            sub["accuracy"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=color_map[target],
            label=target,
        )
    ax.set_xlabel("Calibrated confidence bin midpoint")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0.52, 0.98)
    ax.set_ylim(48, 100)
    ax.legend(
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        handlelength=1.5,
        fontsize=9.5,
        columnspacing=1.2,
    )

    ax = axes[1, 1]
    ad = ad.set_index("target").reindex(targets).reset_index()
    ax.bar(x - width / 2, ad["error_rate_within_ad"], width=width, color=STRATUM_LIGHT, edgecolor="white", label="Inside applicability domain")
    ax.bar(x + width / 2, ad["error_rate_outside_ad"], width=width, color=STRATUM_DARK, edgecolor="white", label="Outside applicability domain")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Misclassification rate (%)")
    ax.set_ylim(0, max(ad["error_rate_outside_ad"].max(), ad["error_rate_within_ad"].max()) * 1.32)
    ax.legend(
        frameon=False, fontsize=8.5, loc="upper right", handlelength=1.6,
        borderaxespad=0.3, labelspacing=0.35,
    )

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.115, hspace=0.34)
    # Letters in figure coordinates. Placed per-axes, panel B's letter lands on
    # top of its own y-axis label, which is long enough to reach the offset.
    for letter, target_ax in zip("ABCD", axes.ravel()):
        pos = target_ax.get_position()
        # The right-hand column needs a wider offset: its y-axis labels are long
        # enough to reach into a narrower one.
        lx = 0.008 if pos.x0 < 0.4 else pos.x0 - 0.088
        # Anchored by its baseline, not its top: with va="top" the glyph hangs
        # down into the y-axis label of its own panel.
        fig.text(lx, pos.y1 + 0.012, letter, fontsize=PANEL_LETTER_SIZE,
                 fontweight="bold", va="bottom", ha="left")
    stem = "figure3_reliability_uncertainty"
    save(fig, stem)
    write_manifest(
        stem,
        [cal_src, conformal_src, conf_src, mono_src, ad_src],
        [OUT / f"{stem}.png", OUT / f"{stem}.pdf"],
        [
            "Panel A shows raw versus Venn-Abers ECE; lower is better, with mixed scalar-calibration changes across targets.",
            "Panel B shows regression conformal coverage inside and outside the applicability domain.",
            "Panel C shows accuracy by calibrated-confidence bin; CCR5 non-monotonicity is described in the caption/text rather than annotated inside the panel.",
            "Panel D shows classification error inside and outside the applicability domain; outside-domain group sizes are DRD2 135, CB2 161, ADORA2A 164, OPRM1 152, CCR5 45 and belong in the caption rather than as in-panel labels.",
        ],
    )


def figure5_screening_docking_funnel():
    """Figure 5: DrugBank screening, docking funnel, and representative headline pose."""
    funnel_src = ROOT / "ml" / "results" / "meta_analysis" / "screening_docking_funnel.csv"
    uncertainty_src = (
        ROOT
        / "ml"
        / "results"
        / "drugbank_screening"
        / "figures"
        / "figure_uncertainty_vs_applicability_domain_data.csv"
    )
    hits_src = ROOT / "manuscript" / "table_docking_headline_hits_interpreted.csv"
    pymol_panel_c_src = ROOT / "outputs" / "figures" / "manuscript" / "figure5C_ocaperidone_DRD2_pymol.png"
    pymol_session_src = ROOT / "outputs" / "figures" / "manuscript" / "figure5C_ocaperidone_DRD2_pymol.pse"

    funnel = pd.read_csv(funnel_src)
    uncertainty = pd.read_csv(uncertainty_src)
    hits = pd.read_csv(hits_src)

    targets = ["DRD2", "CB2", "ADORA2A", "OPRM1", "CCR5"]
    target_order = {target: i for i, target in enumerate(targets)}
    funnel["target"] = funnel["target"].str.upper()
    funnel = funnel.sort_values("target", key=lambda s: s.map(target_order))

    fig = plt.figure(figsize=(9.9, 6.4))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.92, 1.08],
        height_ratios=[0.80, 1.20],
        wspace=0.46,
        hspace=0.34,
    )

    ax = fig.add_subplot(gs[:, 0])
    ax_a = ax
    ax.tick_params(labelsize=10.5)
    stage_cols = [
        ("High-confidence\nnovel", "n_high_confidence_novel_unique_drugs"),
        ("Docking\ngates", "n_after_docking_hard_gates"),
        ("Selected", "n_selected_for_docking"),
        ("Consensus\npose", "n_consensus_pass"),
        ("Interaction\ncheck", "n_passes_interaction_check"),
        ("Prioritized\noutputs", "n_headline_hits"),
    ]
    matrix = funnel.set_index("target")[[col for _, col in stage_cols]].reindex(targets).fillna(0).to_numpy(float)
    shown = matrix.copy()
    shown[shown <= 0] = np.nan
    im = ax.imshow(shown, cmap="YlGnBu", norm=LogNorm(vmin=1, vmax=max(1, np.nanmax(shown))), aspect="auto")
    ax.set_xticks(np.arange(len(stage_cols)))
    # Six two-line stage names cannot sit horizontally in this panel width; angled
    # labels keep them readable without shrinking them into the noise.
    ax.set_xticklabels(
        [label.replace("\n", " ") for label, _ in stage_cols],
        rotation=28, ha="right", fontsize=9.5,
    )
    ax.set_yticks(np.arange(len(targets)))
    ax.set_yticklabels(targets)
    ax.set_xlabel("Prioritization stage")
    ax.set_ylabel("Target")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            text_color = "white" if value >= 20 else DARK
            ax.text(j, i, str(value), ha="center", va="center", fontsize=10.5, fontweight="bold", color=text_color)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Compound count, log scale", fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    uncertainty["target_upper"] = uncertainty["target"].str.upper()
    uncertainty["high_confidence"] = pd.to_numeric(
        uncertainty["predicted_proba_calibrated"], errors="coerce"
    ).ge(0.90)
    hc = (
        uncertainty[uncertainty["high_confidence"]]
        .groupby(["target_upper", "within_applicability_domain"])
        .size()
        .unstack(fill_value=0)
        .reindex(targets, fill_value=0)
    )
    within = hc.get(True, pd.Series(0, index=targets))
    outside = hc.get(False, pd.Series(0, index=targets))
    x = np.arange(len(targets))
    ax.bar(x, within, color=STRATUM_LIGHT, edgecolor="white", label="Inside applicability domain")
    ax.bar(x, outside, bottom=within, color=STRATUM_DARK, edgecolor="white", label="Outside applicability domain")
    for xi, total in zip(x, within + outside):
        ax.text(xi, total + max(2, (within + outside).max() * 0.02), f"{int(total)}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.tick_params(labelsize=10.5)
    ax.set_ylabel("All high-confidence\nDrugBank compounds", fontsize=11)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, fontsize=9.5)
    ax_b = ax

    ax = fig.add_subplot(gs[1, 1])
    ax_c = ax
    # Panel C is a persisted external PyMOL render (Vina Ocaperidone pose only,
    # not GNINA), not a live 3D reconstruction from local coordinates. Swapped
    # in 2026-09-02 to replace the earlier matplotlib scatter placeholder; the
    # PyMOL session (.pse) is retained alongside the PNG for audit/reproduction.
    panel_c_img = plt.imread(pymol_panel_c_src)
    ax.imshow(panel_c_img)
    # Default imshow aspect is 'equal', which re-centres the drawn image inside
    # whatever box is assigned at render time -- that silently offsets this
    # panel's letter from panel B's even when their boxes are set identically
    # below. 'auto' makes the assigned box the actual visual box, so the two
    # letters line up exactly.
    ax.set_aspect("auto")
    ax.set_axis_off()

    fig.tight_layout()
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.11, top=0.93)
    # Align panel C's left edge and width to panel B so the right-hand column
    # reads as one stack rather than two independently centred panels.
    fig.canvas.draw()
    pos_b = ax_b.get_position()
    pos_c = ax_c.get_position()
    # Panel C is a plain imshow axes now (a pre-rendered PyMOL PNG, not a live
    # 3D reconstruction), so its box can align directly to panel B's left edge
    # and width without the inset-compensation offset the old 3D axes needed.
    ax_c.set_position([pos_b.x0, pos_c.y0 - 0.015,
                       pos_b.width, pos_c.height])
    # Panel A spans both rows, so it inherits the full column height and ends up
    # far taller than the six-column grid needs. Trim it and centre it vertically.
    pos_a = ax_a.get_position()
    trimmed_height = pos_a.height * 0.78
    # Top-aligned with panel B rather than centred in the column, so the two read
    # as starting from the same line.
    ax_a.set_position([pos_a.x0, pos_b.y1 - trimmed_height,
                       pos_a.width, trimmed_height])
    # cb.ax was sized against panel A's pre-trim (taller) box at creation time
    # and never moves on its own when ax_a is resized above, so it ends up
    # mismatched against the heatmap it's supposed to describe. Re-fit it to
    # ax_a's actual final box so the scale spans exactly the heatmap's height.
    pos_a_final = ax_a.get_position()
    cb_width = 0.014
    cb_pad = 0.012
    cb.ax.set_position([pos_a_final.x1 + cb_pad, pos_a_final.y0,
                        cb_width, pos_a_final.height])

    for letter, target_ax in (("A", ax_a), ("B", ax_b), ("C", ax_c)):
        pos = target_ax.get_position()
        lx = 0.008 if pos.x0 < 0.4 else pos.x0 - 0.055
        fig.text(lx, pos.y1 + 0.012, letter, fontsize=PANEL_LETTER_SIZE,
                 fontweight="bold", va="bottom", ha="left")

    stem = "figure5_screening_docking_funnel"
    funnel.to_csv(OUT / f"{stem}_source.csv", index=False)
    save(fig, stem)
    write_manifest(
        stem,
        [funnel_src, uncertainty_src, hits_src, pymol_panel_c_src, pymol_session_src],
        [OUT / f"{stem}.png", OUT / f"{stem}.pdf"],
        [
            "Panel A uses the frozen screening-to-docking funnel counts; white cells are zero compounds, which a logarithmic colour scale cannot encode, and this belongs in the caption.",
            "Panel B counts DrugBank compounds with calibrated active probability >= 0.90, stratified by applicability-domain status.",
            "Panel C embeds a persisted external PyMOL render of the Vina Ocaperidone pose (not GNINA) in the DRD2 active-state pocket; Table 4 carries the full headline-hit interpretation. The rendered anchor residue (chain R, residue 114, Asp) was identified by exact CA-coordinate matching between the prepared docking receptor (chain A, residue 83 in the renumbered docking-internal numbering) and the repaired receptor file's real numbering, and verified against the PLIF check's recorded nearest-acidic-residue distance (3.34 A), which matches the Vina pose only -- the GNINA pose sits 4.03 A from the same residue and was excluded from this panel for that reason. The PyMOL session (.pse, same stem) is retained alongside the PNG for audit and to allow re-rendering without repeating the residue-identification steps.",
        ],
        deterministic=False,
    )


def figure6_docking_validation_orthogonality():
    """Candidate Figure 6: docking validation and ML-docking orthogonality."""
    redock_src = ROOT / "docking" / "docking_results" / "redocking_validation.csv"
    disc_src = ROOT / "docking" / "docking_results" / "discriminative_validation.csv"
    rediscovery_src = ROOT / "docking" / "docking_results" / "docking_side_rediscovery_check.csv"
    corr_src = ROOT / "docking" / "docking_results" / "ml_docking_correlation.csv"

    redock = pd.read_csv(redock_src)
    disc = pd.read_csv(disc_src)
    rediscovery = pd.read_csv(rediscovery_src)
    corr = pd.read_csv(corr_src)

    unit_order = [
        "drd2_active",
        "drd2_inactive",
        "cb2_active",
        "cb2_inactive",
        "adora2a_active",
        "adora2a_inactive",
        "oprm1_active",
        "oprm1_inactive",
        "ccr5_inactive_allosteric_primary",
        "ccr5_inactive_allosteric_comparator",
    ]
    unit_labels = {
        "drd2_active": "DRD2\nactive",
        "drd2_inactive": "DRD2\ninactive",
        "cb2_active": "CB2\nactive",
        "cb2_inactive": "CB2\ninactive",
        "adora2a_active": "ADORA2A\nactive",
        "adora2a_inactive": "ADORA2A\ninactive",
        "oprm1_active": "OPRM1\nactive",
        "oprm1_inactive": "OPRM1\ninactive",
        "ccr5_inactive_allosteric_primary": "CCR5\n4MBS",
        "ccr5_inactive_allosteric_comparator": "CCR5\n6AKY",
    }
    unit_labels_short = {
        "drd2_active": "DRD2 act",
        "drd2_inactive": "DRD2 inact",
        "cb2_active": "CB2 act",
        "cb2_inactive": "CB2 inact",
        "adora2a_active": "A2A act",
        "adora2a_inactive": "A2A inact",
        "oprm1_active": "OPRM1 act",
        "oprm1_inactive": "OPRM1 inact",
        "ccr5_inactive_allosteric_primary": "CCR5 4MBS",
        "ccr5_inactive_allosteric_comparator": "CCR5 6AKY",
    }
    target_for_unit = {
        "drd2_active": "drd2",
        "drd2_inactive": "drd2",
        "cb2_active": "cb2",
        "cb2_inactive": "cb2",
        "adora2a_active": "adora2a",
        "adora2a_inactive": "adora2a",
        "oprm1_active": "oprm1",
        "oprm1_inactive": "oprm1",
        "ccr5_inactive_allosteric_primary": "ccr5",
        "ccr5_inactive_allosteric_comparator": "ccr5",
    }
    engine_labels = {
        "vina_affinity": "Vina",
        "gnina_cnn_score": "GNINA",
    }
    rediscovery_ligand_names = {
        "DB01200": "Bromocriptine",
        "DB03719": "NECA",
        "DB14030": "PZM21",
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 7.4),
        gridspec_kw={"height_ratios": [1.0, 1.02], "wspace": 0.38, "hspace": 0.50},
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    for ax in axes.ravel():
        ax.tick_params(labelsize=10.5)

    # A: redocking RMSD. The validation question is whether every unit clears the
    # 2 A pose-recovery threshold, so encode the threshold directly.
    rd = redock.set_index("unit").reindex(unit_order).reset_index()
    x = np.arange(len(rd))
    colors = [TARGET_COLORS[target_for_unit[u]] for u in rd["unit"]]
    y = np.arange(len(rd))
    ax_a.barh(y, rd["best_rmsd"], color=colors, edgecolor="white", linewidth=0.8)
    ax_a.axvline(2.0, color=RED, linestyle="--", linewidth=1.2)
    for yi, rmsd, attempts in zip(y, rd["best_rmsd"], rd["n_attempts_used"]):
        ax_a.text(
            rmsd + 0.035,
            yi,
            f"{rmsd:.2f}",
            ha="left",
            va="center",
            fontsize=9.0,
            fontweight="bold",
        )
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([unit_labels_short[u] for u in rd["unit"]])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Best redocking RMSD")
    ax_a.set_xlim(0, 2.18)

    # B: enrichment AUCs with bootstrap intervals. EF@1% is intentionally absent:
    # with approximately 100 compounds per unit, top 1% is a one-compound metric.
    disc_plot = disc.copy()
    disc_plot["engine"] = disc_plot["score_type"].map(engine_labels)
    disc_plot["unit"] = pd.Categorical(disc_plot["unit"], categories=unit_order, ordered=True)
    disc_plot = disc_plot.sort_values(["unit", "engine"])
    width = 0.36
    for offset, (score_type, label, color) in zip(
        [-width / 2, width / 2],
        [("vina_affinity", "Vina", BLUE), ("gnina_cnn_score", "GNINA", ORANGE)],
    ):
        sub = disc_plot[disc_plot["score_type"].eq(score_type)].set_index("unit").reindex(unit_order)
        vals = sub["auc"].to_numpy()
        lo = sub["auc_ci_low"].to_numpy()
        hi = sub["auc_ci_high"].to_numpy()
        err = np.vstack([vals - lo, hi - vals])
        ax_b.bar(
            x + offset,
            vals,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=label,
            yerr=err,
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0},
        )
    ax_b.axhline(0.5, color=RED, linestyle="--", linewidth=1.1)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([unit_labels_short[u] for u in unit_order], rotation=35, ha="right")
    ax_b.set_ylabel("Enrichment ROC-AUC")
    ax_b.set_ylim(0.30, 0.86)
    ax_b.legend(frameon=False, loc="upper left", ncol=2, fontsize=9.5)
    mean_vina = disc.loc[disc["score_type"].eq("vina_affinity"), "auc"].mean()
    mean_gnina = disc.loc[disc["score_type"].eq("gnina_cnn_score"), "auc"].mean()
    ax_b.text(
        0.98,
        0.985,
        f"Mean AUC: Vina {mean_vina:.3f}; GNINA {mean_gnina:.3f}",
        transform=ax_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        clip_on=False,
    )

    # C: docking-side co-crystal rediscovery. This is deliberately separated
    # from candidate ranking: pose recovery and score rank are not equivalent.
    rediscovery = rediscovery.copy()
    rediscovery["unit_label"] = rediscovery["unit"].map(unit_labels)
    rediscovery["ligand_label"] = rediscovery["drugbank_id"].map(rediscovery_ligand_names).fillna(rediscovery["drugbank_id"])
    xr = np.arange(len(rediscovery))
    w = 0.34
    ax_c.bar(
        xr - w / 2,
        rediscovery["vina_rmsd_vs_crystal_reference"],
        width=w,
        color=BLUE,
        edgecolor="white",
        linewidth=0.8,
        label="Vina RMSD",
    )
    ax_c.bar(
        xr + w / 2,
        rediscovery["gnina_rmsd_vs_crystal_reference"],
        width=w,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.8,
        label="GNINA RMSD",
    )
    ax_c.axhline(2.0, color=RED, linestyle="--", linewidth=1.2)
    for xi, row in rediscovery.iterrows():
        for dx, rmsd_col, pctl_col in [
            (-w / 2, "vina_rmsd_vs_crystal_reference", "vina_rank_percentile_vs_candidates"),
            (w / 2, "gnina_rmsd_vs_crystal_reference", "gnina_rank_percentile_vs_candidates"),
        ]:
            ax_c.text(
                xi + dx,
                row[rmsd_col] + 0.08,
                f"PCTL {row[pctl_col]:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                rotation=0,
                color=DARK,
                fontweight="bold",
            )
    ax_c.set_xticks(xr)
    ax_c.set_xticklabels(
        [f"{u}\n{name}" for u, name in zip(rediscovery["unit_label"], rediscovery["ligand_label"])],
        rotation=0,
        ha="center",
    )
    ax_c.set_ylabel("RMSD to crystal ligand")
    ax_c.set_ylim(0, 3.05)
    ax_c.legend(frameon=False, loc="upper left", ncol=2, fontsize=9.5)

    # D: ML-docking rank correlation. A two-column heatmap is more legible than
    # ten paired bars at manuscript width and keeps significance tied to cells.
    # Vina affinities are stored as negative energies; negative rho means stronger
    # docking scores tend to align with higher ML probabilities.
    corr_plot = corr.copy()
    corr_plot["unit"] = pd.Categorical(corr_plot["unit"], categories=unit_order, ordered=True)
    corr_plot = corr_plot.sort_values(["unit", "score_type"])
    corr_matrix = np.column_stack([
        corr_plot[corr_plot["score_type"].eq("vina_affinity")]
        .set_index("unit")
        .reindex(unit_order)["spearman_rho"]
        .to_numpy(),
        corr_plot[corr_plot["score_type"].eq("gnina_cnn_score")]
        .set_index("unit")
        .reindex(unit_order)["spearman_rho"]
        .to_numpy(),
    ])
    p_matrix = np.column_stack([
        corr_plot[corr_plot["score_type"].eq("vina_affinity")]
        .set_index("unit")
        .reindex(unit_order)["spearman_p"]
        .to_numpy(),
        corr_plot[corr_plot["score_type"].eq("gnina_cnn_score")]
        .set_index("unit")
        .reindex(unit_order)["spearman_p"]
        .to_numpy(),
    ])
    norm = TwoSlopeNorm(vmin=-0.55, vcenter=0.0, vmax=0.35)
    im_d = ax_d.imshow(corr_matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["Vina\naffinity", "GNINA\nCNN"])
    ax_d.set_yticks(np.arange(len(unit_order)))
    ax_d.set_yticklabels([unit_labels_short[u] for u in unit_order])
    ax_d.tick_params(axis="x", bottom=False, top=False, labelbottom=True)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            rho = corr_matrix[i, j]
            pval = p_matrix[i, j]
            star = "*" if pval < 0.05 else ""
            ax_d.text(
                j,
                i,
                f"{rho:.2f}{star}",
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight="bold",
                color="white" if abs(rho) > 0.30 else DARK,
            )
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    ax_d.set_xlabel("Docking score")
    cb_d = fig.colorbar(im_d, ax=ax_d, fraction=0.045, pad=0.025)
    cb_d.set_label("Spearman rho", fontweight="bold")
    ax_d.text(
        1.0,
        1.03,
        "* p < 0.05",
        transform=ax_d.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=DARK,
        clip_on=False,
    )

    fig.tight_layout(rect=[0.02, 0.045, 0.995, 0.955])
    for letter, target_ax in zip("ABCD", [ax_a, ax_b, ax_c, ax_d]):
        pos = target_ax.get_position()
        fig.text(
            pos.x0 - 0.045,
            pos.y1 + 0.012,
            letter,
            fontsize=PANEL_LETTER_SIZE,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    stem = "figure6_docking_validation_orthogonality"
    save(fig, stem)
    write_manifest(
        stem,
        [redock_src, disc_src, rediscovery_src, corr_src],
        [OUT / f"{stem}.png", OUT / f"{stem}.pdf"],
        [
            "Candidate main-text Figure 6 assembled from frozen docking validation artifacts; no docking or scoring is rerun.",
            "Panel A shows all 10 receptor units passing the 2.0 A redocking threshold; OPRM1 active passed on the second attempt.",
            "Panel B reports enrichment ROC-AUC with bootstrap intervals; EF@1% is intentionally omitted because about 100 compounds per unit make it a one-compound statistic.",
            "Panel C separates pose rediscovery from rank percentile among candidates; NECA and PZM21 illustrate that accurate pose recovery and rank prioritization can be decoupled.",
            "Panel D reports ML-docking Spearman correlations using deployed-model probabilities stamped in the correlation source file.",
        ],
    )


if __name__ == "__main__":
    figure1_framework_panel()
    figure2_performance_scaling()
    figure3_reliability_uncertainty()
    figure4_representation_transferability()
    figure5_screening_docking_funnel()
    figure6_docking_validation_orthogonality()
