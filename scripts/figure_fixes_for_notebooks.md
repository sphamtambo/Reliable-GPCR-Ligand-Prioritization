# Figure fixes — to apply in notebooks 03 / 04 (Colab or HPC)

Figs 2, 7 and the Fig 1 filename are already fixed locally (`scripts/figure_polish.py`,
output in `outputs/figures/polished/`). The figures below **must** be fixed in the
notebook environment because their models (XGBoost/LightGBM) and libraries (shap) do
not load locally, and their plot-source arrays were never saved.

For each: (A) apply the plotting fix and re-run that cell; (B) add the one-line
persistence so future polish is standalone.

First, at the top of the plotting cell, apply the shared publication style:

```python
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
})
# save BOTH: fig.savefig(base+".png"); fig.savefig(base+".pdf")
```

## Universal fixes (apply to every figure below)
- **Remove the on-figure title** (`ax.set_title(...)` / `plt.suptitle(...)`) — the caption carries it.
- **American spelling** everywhere in figure text: "Randomised" -> "Randomized" (legend AND title), "colour"->"color", etc.
- **No code identifiers** in labels/titles/legends: "LGB" -> "LightGBM", "ki/ki_ic50/full" -> "Ki / Ki+IC50 / Full".
- Save PNG (300 dpi) **and** vector PDF.

## Figure 3 — Y-randomization (notebook 03)
- Remove title (has an em dash and "Test ... — p<0.001").
- Legend: "Y-Randomised" -> "Y-Randomized", "Randomised Mean" -> "Randomized Mean".
- (B) Persist the array so it is re-plottable later:
  ```python
  np.save(save_dir/"y_randomization_scores.npy", permuted_aucs)  # the 100 permuted AUCs
  ```

## Figure 4 / S4 — SHAP summary (notebook 03)
- Title "SHAP Summary – LGB" -> remove (or if kept anywhere, "LightGBM").
- (B) Persist the SHAP matrix used for the beeswarm:
  ```python
  np.save(save_dir/"shap_values_explain.npy", shap_values)     # (n_explain, n_features)
  np.save(save_dir/"shap_feature_matrix.npy", X_explain)       # the explained instances
  # feature names already in a saved list/CSV
  ```

## Figure 5 / S2 — Reliability diagram (notebook 04)
- Remove title (em dash "— CB2 (full variant, test set)").
- Report ECE at 4 dp in the legend to match the text: `ECE=0.0588`, `ECE=0.0257`.
- (B) Persist the binned curve so it re-plots without the models:
  ```python
  pd.DataFrame({"mean_pred_raw":mpr,"obs_frac_raw":ofr,
                "mean_pred_cal":mpc,"obs_frac_cal":ofc}).to_csv(
                save_dir/"reliability_curve.csv", index=False)
  ```

## Figure 6 — Interval width vs. error (notebook 04)  [most substantive]
- Remove title (em dash).
- **Add the correlation the figure is meant to show**: Spearman r/p annotation + a trend line.
  ```python
  from scipy.stats import spearmanr
  r, p = spearmanr(halfwidth, abs_error)
  ax.annotate(f"Spearman r = {r:.3f}\np = {p:.1e}  (n = {len(abs_error)})",
              xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=10)
  # trend line
  b, a = np.polyfit(halfwidth, abs_error, 1)
  xs = np.linspace(halfwidth.min(), halfwidth.max(), 100)
  ax.plot(xs, a + b*xs, color="0.25", lw=1.5)
  ```
- (B) Persist the per-compound arrays (this is the one the audit flagged as lost):
  ```python
  pd.DataFrame({"conformal_halfwidth":halfwidth,
                "abs_error":abs_error}).to_csv(
                save_dir/"interval_width_vs_error.csv", index=False)
  ```

## Figure S1 — Chemical-space t-SNE (notebook 03)
- Remove title; label axes "t-SNE 1" / "t-SNE 2".
- (B) Persist coords + group label so it re-plots without recomputing t-SNE:
  ```python
  pd.DataFrame({"tsne1":emb[:,0],"tsne2":emb[:,1],"set":group}).to_csv(
                save_dir/"tsne_coords.csv", index=False)
  ```

## After re-running: promote to outputs/figures/
Copy each regenerated PNG+PDF to `outputs/figures/` under the manuscript's expected
name (figure3_*, figure4_*, figure5_*, figure6_*, figureS1_* ... figureS4_*).

## S3 / S4 (CB1 analogs of 3 / 4)
Same fixes as Figure 3 / Figure 4, run on the CB1 target's cell.
```
