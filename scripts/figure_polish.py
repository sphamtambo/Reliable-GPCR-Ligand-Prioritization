"""
Publication figure polish — standalone, regenerates figures from SAVED DATA only
(no re-run of the analysis notebooks). Covers the figures whose plot-source data
is persisted locally: Figure 2 (CB2/CB1 potency correlation) and Figure 7 (CB1
selectivity scatter). Figures 3-6 and S1-S4 cannot be regenerated here because
their plot-source arrays (permutation AUCs, SHAP values, calibration bins,
per-compound conformal intervals, t-SNE coords) were never persisted — only the
rendered images were. Fix those in their source notebooks (03/04) instead.

Output: outputs/figures/polished/  (PNG at 300 dpi + vector PDF). Originals untouched.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "polished"
OUT.mkdir(parents=True, exist_ok=True)

# ---- shared publication style (reusable next project) ----
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 15,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.titlesize": 12,
})
BLUE, GREY, RED, ORANGE = "#1f77b4", "#c9c9c9", "#d62728", "#ff7f0e"


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf")


def figure2():
    pools = [("ki", "Ki"), ("ki_ic50", "Ki+IC50"), ("full", "Full (Ki+IC50+EC50)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, (key, label), letter in zip(axes, pools, "abc"):
        df = pd.read_csv(ROOT / f"data/processed/cb1_cb2_shared_activity_{key}.csv")
        x, y = df["cb2_pActivity"].values, df["cb1_pActivity"].values
        r, _ = pearsonr(x, y)
        ax.scatter(x, y, s=10, alpha=0.25, color=BLUE, edgecolors="none", rasterized=True)
        ax.set_title(f"{label}\n(n = {len(df):,}, r = {r:.3f})")
        ax.set_xlabel("CB2 pChEMBL value")
        if letter == "a":
            ax.set_ylabel("CB1 pChEMBL value")
        ax.text(-0.04, 1.07, f"({letter})", transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="bottom", ha="right")
    fig.tight_layout()
    save(fig, "figure2_cb2_cb1_potency_correlation")


def figure7():
    m = pd.read_csv(ROOT / "ml/results/cb1_selectivity/selectivity_full_merge.csv")
    xc, yc = "ensemble_proba_cb2", "ensemble_proba_cb1"
    shortlist = m[(m[xc] >= 0.90) & (m[yc] < 0.50)]
    novel = {"DB13796": "Dibunate", "DB12833": "Tandospirone",
             "DB06471": "Naxifylline", "DB15186": "Paxalisib"}
    control = {"DB11903": "GW842166"}

    fig, ax = plt.subplots(figsize=(7, 6.2))
    ax.scatter(m[xc], m[yc], s=6, alpha=0.35, color=GREY, edgecolors="none",
               rasterized=True, label="DrugBank library (not retained)")
    ax.scatter(shortlist[xc], shortlist[yc], s=14, alpha=0.8, color=BLUE,
               edgecolors="none", rasterized=True,
               label=f"Selectivity shortlist (n = {len(shortlist)})")
    ax.axvspan(0.90, 1.0, ymin=0, ymax=0.50, color="#2ca02c", alpha=0.08)
    ax.axvline(0.90, ls="--", lw=0.8, color="0.4")
    ax.axhline(0.50, ls="--", lw=0.8, color="0.4")

    def coords(did):
        row = m[m["DRUGBANK_ID"] == did].iloc[0]
        return row[xc], row[yc]
    # per-compound label side to avoid collisions (Dibunate & Tandospirone share CB1=0.236)
    label_side = {"DB12833": "left"}   # Tandospirone -> label to the left
    for did, name in novel.items():
        x, y = coords(did)
        ax.scatter(x, y, marker="D", s=90, color=RED, edgecolors="k", lw=0.6, zorder=5)
        if label_side.get(did) == "left":
            ax.annotate(name, (x, y), xytext=(-9, 0), textcoords="offset points",
                        fontsize=9, fontweight="bold", va="center", ha="right")
        else:
            ax.annotate(name, (x, y), xytext=(9, 0), textcoords="offset points",
                        fontsize=9, fontweight="bold", va="center", ha="left")
    for did, name in control.items():
        x, y = coords(did)
        ax.scatter(x, y, marker="*", s=220, color=ORANGE, edgecolors="k", lw=0.6, zorder=5)
        ax.annotate(name, (x, y), xytext=(11, 0), textcoords="offset points",
                    fontsize=9, fontweight="bold", va="center", ha="left")
    ax.text(0.955, 0.47, "retention region", color="#2ca02c", style="italic",
            fontsize=9, fontweight="bold", ha="right")

    ax.set_xlabel("CB2 activity probability")
    ax.set_ylabel("CB1 activity probability")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    handles = [
        Line2D([], [], marker="o", ls="", color=GREY, label="DrugBank library (not retained)"),
        Line2D([], [], marker="o", ls="", color=BLUE, label=f"Selectivity shortlist (n = {len(shortlist)})"),
        Line2D([], [], marker="D", ls="", color=RED, mec="k", label="Consensus candidate (novel)"),
        Line2D([], [], marker="*", ls="", color=ORANGE, mec="k", markersize=13, label="Positive control (GW842166)"),
    ]
    ax.legend(handles=handles, loc="upper center", frameon=True, ncol=1)
    fig.tight_layout()
    save(fig, "figure7_cb1_selectivity_scatter")


if __name__ == "__main__":
    figure2()
    figure7()
    print("done ->", OUT)
