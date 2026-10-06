"""Phase 9, part 2: draw the charts.

Run:  python src/make_figures.py        (after python src/evaluate.py)

Reads   results/phase7_rf_results.csv, phase8_xgb_results.csv,
        results/phase9_predictions.csv, phase9_importance.csv, phase9_worst.csv
Writes  results/fig1_model_comparison.png   test MAE, Random Forest vs XGBoost, both splits
        results/fig2_parity.png             predicted vs true XLogP3, scaffold test set
        results/fig3_error_by_size.png      error by molecule size
        results/fig4_importance.png         what XGBoost relies on
        results/fig5_worst_molecules.png    the 12 biggest misses, drawn as structures

Colour rules: XGBoost = blue, Random Forest = orange, always. Text stays in grey/black.
Charts that mix inputs of two kinds (descriptors vs fingerprint slots) reuse the same two hues.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # XGBoost, Random Forest
MODEL_COLOR = {"XGB": BLUE, "RF": ORANGE}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.dpi": 100, "savefig.dpi": 160,
})


def style_axes(ax, y_grid=True):
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)          # grid lines sit BEHIND the marks
    if y_grid:
        ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.tick_params(length=0)


def legend_row(fig, entries, y=0.93):
    """Swatch + grey label, one row, centred. Text is never coloured by the series."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in entries]
    fig.legend(handles, [n for n, _ in entries], loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=len(entries), frameon=False, handlelength=1.0, handleheight=1.0, columnspacing=2.0,
               labelcolor=INK2)


def save(fig, name):
    path = RES / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"saved results/{name}")


# ----------------------------------------------------------------------
# Figure 1: model comparison
# ----------------------------------------------------------------------
def fig_comparison(rf, xgb):
    sets = ["Morgan counts", "Descriptors", "Morgan + descriptors"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    fig.suptitle("XGBoost cuts the test error by about a third versus Random Forest",
                 x=0.06, ha="left", fontsize=14, fontweight="bold", y=0.985)
    legend_row(fig, [("XGBoost (tuned)", BLUE), ("Random Forest", ORANGE)], y=0.925)
    for ax, split in zip(axes, ["random", "scaffold"]):
        base = rf[(rf["split"] == split) & (rf["model"] == "Baseline: always the mean")]["MAE"].iloc[0]
        for i, fs in enumerate(sets):
            r = rf[(rf["split"] == split) & (rf["model"] == f"RF: {fs}")]["MAE"].iloc[0]
            x = xgb[(xgb["split"] == split) & (xgb["model"] == f"XGB tuned: {fs}")]["MAE"].iloc[0]
            for dx, v, c in [(-0.2, r, ORANGE), (0.2, x, BLUE)]:
                ax.bar(i + dx, v, width=0.36, color=c, edgecolor=SURFACE, linewidth=1)
                ax.text(i + dx, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
        ax.axhline(base, color=INK2, linewidth=1)
        ax.text(2.45, base + 0.025, f"always guess the average: {base:.2f}", ha="right", va="bottom",
                fontsize=10, color=INK2)
        ax.set_xticks(range(3))
        ax.set_xticklabels([s.replace(" + ", "\n+ ") for s in sets])
        ax.set_title("Random split (easier)" if split == "random" else "Scaffold split (harder, new ring skeletons)",
                     loc="left", fontsize=11, color=INK2)
        ax.set_ylim(0, 1.6)
        style_axes(ax)
    axes[0].set_ylabel("Test MAE, XLogP3 units (lower is better)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "fig1_model_comparison.png")


# ----------------------------------------------------------------------
# Figure 2: parity plots
# ----------------------------------------------------------------------
def fig_parity(pred):
    part = pred[pred["split"] == "scaffold"]
    models = [("RF: Descriptors", ORANGE, "Random Forest, descriptors"),
              ("XGB: Morgan + descriptors", BLUE, "XGBoost, Morgan + descriptors")]
    lo, hi = np.floor(part["y_true"].min()) - 0.5, np.ceil(part["y_true"].max()) + 0.5
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharex=True, sharey=True)
    fig.suptitle("Predicted vs true XLogP3 on unseen ring skeletons: the closer to the line, the better",
                 x=0.06, ha="left", fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.06, 0.915, "Each dot is one test molecule. The grey line is a perfect prediction.",
             ha="left", fontsize=10.5, color=INK2)
    for ax, (m, color, title) in zip(axes, models):
        d = part[part["model"] == m]
        ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1, zorder=1)
        ax.scatter(d["y_true"], d["y_pred"], s=16, color=color, alpha=0.38, linewidths=0, zorder=2)
        mae = np.abs(d["y_true"] - d["y_pred"]).mean()
        r2 = 1 - ((d["y_true"] - d["y_pred"]) ** 2).sum() / ((d["y_true"] - d["y_true"].mean()) ** 2).sum()
        ax.text(0.04, 0.95, f"MAE {mae:.2f}   R² {r2:.2f}", transform=ax.transAxes, va="top", fontsize=11,
                color=INK, bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID))
        ax.set_title(title, loc="left", fontsize=11, color=INK2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel("True XLogP3 (PubChem)")
        style_axes(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=1)
    axes[0].set_ylabel("Predicted XLogP3")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig2_parity.png")


# ----------------------------------------------------------------------
# Figure 3: error by size
# ----------------------------------------------------------------------
def fig_size(pred):
    bins = [(0, 15, "up to 15"), (16, 22, "16 to 22"), (23, 29, "23 to 29"), (30, 36, "30 to 36"), (37, 999, "37 and up")]
    part = pred[pred["split"] == "scaffold"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.suptitle("Error is flat for typical drug-sized molecules and grows for the largest ones",
                 x=0.06, ha="left", fontsize=14, fontweight="bold", y=0.985)
    legend_row(fig, [("XGBoost (Morgan + descriptors)", BLUE), ("Random Forest (descriptors)", ORANGE)], y=0.925)
    labels = []
    for i, (lo, hi, name) in enumerate(bins):
        sel = part[(part["n_heavy"] >= lo) & (part["n_heavy"] <= hi)]
        n = sel["cid"].nunique()
        labels.append(f"{name}\nn = {n}")
        for dx, m, c in [(-0.2, "RF: Descriptors", ORANGE), (0.2, "XGB: Morgan + descriptors", BLUE)]:
            d = sel[sel["model"] == m]
            v = np.abs(d["y_true"] - d["y_pred"]).mean()
            ax.bar(i + dx, v, width=0.36, color=c, edgecolor=SURFACE, linewidth=1)
            ax.text(i + dx, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(labels)
    ax.set_xlabel("Heavy atoms per molecule (every atom except hydrogen)")
    ax.set_ylabel("Test MAE, XLogP3 units (lower is better)")
    ax.set_ylim(0, 1.45)
    style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "fig3_error_by_size.png")


# ----------------------------------------------------------------------
# Figure 4: importance
# ----------------------------------------------------------------------
def fig_importance(imp, top=15):
    d = imp.sort_values("share", ascending=False).head(top).iloc[::-1]
    kinds = imp.groupby("kind")["share"].sum()
    fig, ax = plt.subplots(figsize=(10, 6.2))
    desc_share = kinds.get("descriptor", 0)
    fig.suptitle(f"Descriptors carry about {desc_share:.0%} of what XGBoost uses; fingerprint slots the rest",
                 x=0.06, ha="left", fontsize=14, fontweight="bold", y=0.985)
    legend_row(fig, [("Descriptor (named number)", BLUE), ("Fingerprint slot (substructure)", ORANGE)], y=0.93)
    colors = [BLUE if k == "descriptor" else ORANGE for k in d["kind"]]
    ax.barh(range(len(d)), d["share"] * 100, color=colors, height=0.62, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["label"], fontsize=10)
    for i, v in enumerate(d["share"] * 100):
        ax.text(v + 0.08, i, f"{v:.1f}%", va="center", fontsize=10, color=INK)
    ax.set_xlabel("Share of the model's total splitting gain (%)")
    ax.set_xlim(0, d["share"].max() * 100 * 1.15)
    style_axes(ax, y_grid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig4_importance.png")


# ----------------------------------------------------------------------
# Figure 5: worst molecules
# ----------------------------------------------------------------------
def fig_worst(worst, n=12):
    d = worst.head(n)
    mols = [Chem.MolFromSmiles(s) for s in d["smiles"]]
    legends = [f"true {t:+.1f} | pred {p:+.1f}" for t, p in zip(d["y_true"], d["y_pred"])]
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(330, 260), legends=legends)
    path = RES / "fig5_worst_molecules.png"
    if hasattr(img, "save"):
        img.save(path)
    else:
        path.write_bytes(img.data)
    print("saved results/fig5_worst_molecules.png")


def main():
    need = ["phase7_rf_results.csv", "phase8_xgb_results.csv", "phase9_predictions.csv",
            "phase9_importance.csv", "phase9_worst.csv"]
    missing = [f for f in need if not (RES / f).exists()]
    if missing:
        raise SystemExit(f"missing {missing}. Run train_rf.py, train_xgb.py and evaluate.py first.")
    rf, xgb = pd.read_csv(RES / need[0]), pd.read_csv(RES / need[1])
    pred, imp, worst = pd.read_csv(RES / need[2]), pd.read_csv(RES / need[3]), pd.read_csv(RES / need[4])
    fig_comparison(rf, xgb)
    fig_parity(pred)
    fig_size(pred)
    fig_importance(imp)
    fig_worst(worst)


if __name__ == "__main__":
    main()
