"""Phase 13: the final benchmark. Every model, the same test molecules, one honest table.

Run:  python src/benchmark.py        (about 1 min, no training)

Reads   results/phase9_predictions.csv      RF descriptors, XGB descriptors, XGB Morgan + descriptors
        results/gnn_gcn_predictions.csv     GCN, 3 seeds
        results/gnn_mpnn_predictions.csv    MPNN, 3 seeds             (skipped if missing)
        results/gnn_mpnn_nobond_predictions.csv                       (skipped if missing)
        results/phase7_rf_results.csv       the "always predict the median" baseline
Writes  results/phase13_benchmark.csv       one row per model and split: MAE, interval, RMSE, R2
        results/phase13_pairs.csv           paired comparisons: is the gap real or noise?
        results/phase13_bins.csv            error by molecule size and by true XLogP3 range
        results/phase13_readme_table.md     the table, ready to paste into the README
        results/fig6_final_benchmark.png    the headline chart

Three ground rules, so the table can be trusted:
  1. Same test molecules. Every file is aligned by PubChem CID and the true values are checked.
  2. GNN rows are the MEAN of the per-seed errors (one network, like one XGBoost fit). The
     3-seed AVERAGE of predictions is an ensemble, so it is reported on its own labelled row
     and is never the headline.
  3. Gaps are judged with a paired bootstrap on the same resampled molecules (2000 resamples).
     If the interval for a difference contains 0, the gap is NOT distinguishable from noise.

One caution that no code can fix: the models were compared on this test set many times over the
phases, so the winner's margin is a little optimistic. A fresh hold-out set would be the cure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

SEED = 42
N_BOOT = 2000
SPLITS = ["random", "scaffold"]

GNN_FILES = {"gcn": "GCN", "mpnn": "MPNN", "mpnn_nobond": "MPNN, no bond features"}

# Colour by model family. Blue/orange match the earlier figures; violet is the GNN family.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
FAMILY_COLOR = {"rf": "#eb6834", "xgb": "#2a78d6", "gnn": "#4a3aa7"}
FAMILY_NAME = {"rf": "Random Forest", "xgb": "XGBoost", "gnn": "Graph neural network"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.dpi": 100, "savefig.dpi": 160,
})


# ----------------------------------------------------------------------
# Loading: one dict of models per split, all aligned on the same molecules
# ----------------------------------------------------------------------
def load_split(split):
    """Return (y, meta, entries). entries[name] = dict(P=(k, n) predictions, family, kind, chart)."""
    p9 = pd.read_csv(RES / "phase9_predictions.csv")
    p9 = p9[p9["split"] == split]
    entries, y, meta = {}, None, None

    for model, g in p9.groupby("model", sort=False):
        g = g.sort_values("cid").reset_index(drop=True)
        if y is None:
            y, meta = g["y_true"].to_numpy(), g[["cid", "smiles", "n_heavy"]].copy()
        assert (g["cid"].to_numpy() == meta["cid"].to_numpy()).all(), f"{model}: different test molecules"
        assert np.allclose(g["y_true"].to_numpy(), y), f"{model}: y_true differs"
        fam = "rf" if model.startswith("RF") else "xgb"
        entries[model] = dict(P=g["y_pred"].to_numpy()[None, :], family=fam, kind="single fit", chart=True)

    for key, label in GNN_FILES.items():
        path = RES / f"gnn_{key}_predictions.csv"
        if not path.exists():
            if split == SPLITS[0]:
                print(f"note: {path.name} not found, {label} is left out (run train_gnn.py --model {key})")
            continue
        d = pd.read_csv(path)
        d = d[d["split"] == split]
        rows = []
        for seed in sorted(d["seed"].unique()):
            s = d[d["seed"] == seed].sort_values("cid").reset_index(drop=True)
            assert (s["cid"].to_numpy() == meta["cid"].to_numpy()).all(), f"{label} seed {seed}: different molecules"
            assert np.allclose(s["y_true"].to_numpy(), y), f"{label} seed {seed}: y_true differs"
            rows.append(s["y_pred"].to_numpy())
        P = np.vstack(rows)
        k = len(P)
        entries[label] = dict(P=P, family="gnn", kind=f"mean of {k} seeds", chart=True)
        entries[f"{label} (average of {k} seeds)"] = dict(
            P=P.mean(axis=0, keepdims=True), family="gnn", kind="ensemble", chart=False)
    return y, meta, entries


def baseline_row(split):
    b = pd.read_csv(RES / "phase7_rf_results.csv")
    b = b[(b["split"] == split) & (b["model"] == "Baseline: always the median")].iloc[0]
    return dict(split=split, model="Baseline: always the median", kind="constant", MAE=b["MAE"],
                MAE_lo=np.nan, MAE_hi=np.nan, RMSE=b["RMSE"], R2=b["R2"], seed_std=np.nan)


# ----------------------------------------------------------------------
# Scores. Per-molecule error e_i (averaged over seeds) makes every MAE a simple mean,
# so one resampling of molecule indices serves every model and every pair.
# ----------------------------------------------------------------------
def per_molecule_error(P, y):
    return np.abs(P - y).mean(axis=0)


def score_all(P, y):
    maes = np.abs(P - y).mean(axis=1)
    sq = (P - y) ** 2
    rmses = np.sqrt(sq.mean(axis=1))
    r2s = 1 - sq.sum(axis=1) / ((y - y.mean()) ** 2).sum()
    return maes, rmses, r2s


def bootstrap_indices(n):
    return np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))


def interval(v, idx):
    return np.percentile(v[idx].mean(axis=1), [2.5, 97.5])


def build_table(split, y, entries, idx):
    rows = [baseline_row(split)]
    for name, e in entries.items():
        maes, rmses, r2s = score_all(e["P"], y)
        lo, hi = interval(per_molecule_error(e["P"], y), idx)
        rows.append(dict(split=split, model=name, kind=e["kind"], MAE=maes.mean(), MAE_lo=lo, MAE_hi=hi,
                         RMSE=rmses.mean(), R2=r2s.mean(),
                         seed_std=maes.std(ddof=1) if len(maes) > 1 else np.nan))
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def compare(split, a, b, ea, eb, idx):
    d = ea - eb
    lo, hi = interval(d, idx)
    if hi < 0:
        verdict = f"{a} is better (real)"
    elif lo > 0:
        verdict = f"{b} is better (real)"
    else:
        verdict = "NOT distinguishable from noise"
    return dict(split=split, A=a, B=b, MAE_A=ea.mean(), MAE_B=eb.mean(), diff_A_minus_B=d.mean(),
                lo=lo, hi=hi, verdict=verdict)


def best_of(entries, y, family):
    cands = {n: per_molecule_error(e["P"], y).mean() for n, e in entries.items()
             if e["family"] in family and e["chart"]}
    return min(cands, key=cands.get) if cands else None


def build_pairs(split, y, entries, idx):
    err = {n: per_molecule_error(e["P"], y) for n, e in entries.items()}
    pairs = []
    tree, gnn = best_of(entries, y, {"rf", "xgb"}), best_of(entries, y, {"gnn"})
    if tree and gnn:
        pairs.append((gnn, tree))                                  # the headline question
    pairs.append(("XGB: Morgan + descriptors", "RF: Descriptors"))  # boosting vs forest
    pairs.append(("XGB: Morgan + descriptors", "XGB: Descriptors"))  # do fingerprints add anything?
    pairs.append(("MPNN", "GCN"))                                  # does message passing beat plain GCN?
    pairs.append(("MPNN", "MPNN, no bond features"))               # the clean bond-feature test
    out = []
    for a, b in pairs:
        if a in err and b in err:
            out.append(compare(split, a, b, err[a], err[b], idx))

    # Blend: best GNN ensemble and XGB Morgan + descriptors, weight 0.5 chosen in advance (not tuned).
    ens = [n for n, e in entries.items() if e["kind"] == "ensemble"]
    xgb = "XGB: Morgan + descriptors"
    blend = None
    if ens and xgb in entries:
        g = min(ens, key=lambda n: err[n].mean())
        P = 0.5 * entries[g]["P"] + 0.5 * entries[xgb]["P"]
        blend = f"Blend: 50% {g.split(' (')[0]} ensemble + 50% XGB"
        entries[blend] = dict(P=P, family="gnn", kind="blend (weight fixed at 0.5)", chart=False)
        err[blend] = per_molecule_error(P, y)
        better_parent = g if err[g].mean() < err[xgb].mean() else xgb
        out.append(compare(split, blend, better_parent, err[blend], err[better_parent], idx))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# Where does each model win or lose? Error by size and by true XLogP3 range.
# ----------------------------------------------------------------------
SIZE_BINS = [(0, 15, "up to 15"), (16, 25, "16 to 25"), (26, 35, "26 to 35"), (36, 999, "36+")]
RANGE_BINS = [(-99, 0, "below 0"), (0, 2, "0 to 2"), (2, 4, "2 to 4"), (4, 99, "above 4")]


def build_bins(split, y, meta, entries):
    n_heavy = meta["n_heavy"].to_numpy()
    rows = []
    for kind, bins, values in [("heavy atoms", SIZE_BINS, n_heavy), ("true XLogP3", RANGE_BINS, y)]:
        for lo, hi, label in bins:
            m = (values >= lo) & (values <= hi) if kind == "heavy atoms" else (values >= lo) & (values < hi)
            if m.sum() == 0:
                continue
            for name, e in entries.items():
                if not e["chart"]:
                    continue
                rows.append(dict(split=split, grouping=kind, bin=label, n=int(m.sum()), model=name,
                                 MAE=per_molecule_error(e["P"], y)[m].mean()))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Printing, README table, figure
# ----------------------------------------------------------------------
def fmt_table(t):
    t = t.copy()
    t["MAE (95% interval)"] = [
        f"{m:.3f}" if np.isnan(lo) else f"{m:.3f}  [{lo:.3f}, {hi:.3f}]"
        for m, lo, hi in zip(t["MAE"], t["MAE_lo"], t["MAE_hi"])]
    t["seed std"] = t["seed_std"].map(lambda v: "" if np.isnan(v) else f"{v:.3f}")
    t["RMSE"] = t["RMSE"].map("{:.3f}".format)
    t["R2"] = t["R2"].map("{:.3f}".format)
    return t[["model", "kind", "MAE (95% interval)", "seed std", "RMSE", "R2"]]


def readme_table(tables):
    lines = ["| Model | Random split MAE | Scaffold split MAE |", "|---|---|---|"]
    names = [n for n in tables["scaffold"]["model"]]
    for name in names:
        cells = []
        for s in SPLITS:
            r = tables[s][tables[s]["model"] == name]
            cells.append("n/a" if r.empty else f"{r['MAE'].iloc[0]:.3f}")
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    lines.append("")
    lines.append("MAE of predicted XLogP3 on held-out molecules (lower is better). GNN rows are the mean of 3 seeds.")
    return "\n".join(lines)


def fig_benchmark(tables):
    chart = {s: tables[s][~tables[s]["kind"].isin(["constant", "ensemble"]) &
                          ~tables[s]["kind"].str.startswith("blend")] for s in SPLITS}
    order = list(chart["scaffold"].sort_values("MAE")["model"])      # best on top, same rows in both panels
    def family(name):
        if name.startswith("RF"):
            return "rf"
        if name.startswith("XGB"):
            return "xgb"
        return "gnn"

    best = {s: chart[s].sort_values("MAE")["model"].iloc[0] for s in SPLITS}
    if best["random"] == best["scaffold"]:
        title = f"{best['scaffold']} has the lowest test error on both splits"
    else:
        title = "The best model depends on the split"
    base = {s: tables[s][tables[s]["kind"] == "constant"]["MAE"].iloc[0] for s in SPLITS}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.76, bottom=0.12, wspace=0.08)
    fig.text(0.01, 0.975, title, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.925,
             f"Test MAE of predicted XLogP3, dot = value, line = 95% bootstrap interval. "
             f"Always predicting the median scores {base['random']:.2f} (random) and {base['scaffold']:.2f} (scaffold).",
             ha="left", fontsize=10, color=INK2)
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=FAMILY_COLOR[f],
                          markeredgecolor=SURFACE, markersize=9) for f in ["rf", "xgb", "gnn"]]
    fig.legend(handles, [FAMILY_NAME[f] for f in ["rf", "xgb", "gnn"]], loc="upper left",
               bbox_to_anchor=(0.01, 0.905), bbox_transform=fig.transFigure, ncol=3, frameon=False, labelcolor=INK2,
               columnspacing=2.0, borderaxespad=0)

    top = max(chart[s]["MAE_hi"].max() for s in SPLITS)
    for ax, split, title_s in zip(axes, SPLITS, ["Random split", "Scaffold split (new chemical skeletons)"]):
        t = chart[split].set_index("model")
        ax.set_title(title_s, loc="left", fontsize=11, color=INK2, pad=8)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)
        for i, name in enumerate(order):
            y = len(order) - 1 - i
            r = t.loc[name]
            c = FAMILY_COLOR[family(name)]
            ax.plot([r["MAE_lo"], r["MAE_hi"]], [y, y], color=c, linewidth=2, zorder=2)
            ax.plot([r["MAE_lo"], r["MAE_hi"]], [y, y], "|", color=c, markersize=9, markeredgewidth=2, zorder=2)
            ax.plot(r["MAE"], y, "o", color=c, markersize=8, markeredgewidth=0, zorder=3)
            ax.text(r["MAE_hi"] + top * 0.03, y, f"{r['MAE']:.3f}", va="center", fontsize=10, color=INK2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order[::-1])
        ax.set_xlim(0, top * 1.28)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel("test MAE (lower is better)")
    path = RES / "fig6_final_benchmark.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"saved results/{path.name}")


def cross_checks(tables):
    """Compare against numbers the earlier phases already wrote, so a silent misalignment cannot hide."""
    g = RES / "gnn_gcn_results.csv"
    if g.exists():
        r = pd.read_csv(g)
        for s in SPLITS:
            # per-seed rows only: the file also holds a seed = -1 row for the seed-average ensemble
            theirs = r[(r["split"] == s) & (r["model"] == "GCN") & (r["seed"] >= 0)]["MAE"].mean()
            mine = tables[s][tables[s]["model"] == "GCN"]["MAE"].iloc[0]
            assert abs(theirs - mine) < 1e-6, f"GCN {s}: results file says {theirs:.4f}, predictions give {mine:.4f}"
    x = RES / "phase8_xgb_results.csv"
    if x.exists():
        r = pd.read_csv(x)
        for s in SPLITS:
            theirs = r[(r["split"] == s) & (r["model"] == "XGB tuned: Morgan + descriptors")]["MAE"]
            mine = tables[s][tables[s]["model"] == "XGB: Morgan + descriptors"]["MAE"].iloc[0]
            if len(theirs) and abs(theirs.iloc[0] - mine) > 0.01:
                print(f"warning: XGB {s} differs from phase 8 by {abs(theirs.iloc[0] - mine):.3f}, check the refit")
    print("cross-checks passed (GCN matches its results file)")


def main():
    tables, pair_frames, bin_frames = {}, [], []
    for split in SPLITS:
        y, meta, entries = load_split(split)
        idx = bootstrap_indices(len(y))
        pairs = build_pairs(split, y, entries, idx)   # also adds the blend entry
        tables[split] = build_table(split, y, entries, idx)
        pair_frames.append(pairs)
        bin_frames.append(build_bins(split, y, meta, entries))
        print(f"\n=== {split} split, {len(y)} test molecules ===")
        print(fmt_table(tables[split]).to_string(index=False))
        print("\nPaired comparisons (difference in MAE, A minus B; negative means A is better):")
        for _, r in pairs.iterrows():
            print(f"  {r['A']}  vs  {r['B']}:  {r['diff_A_minus_B']:+.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]  -> {r['verdict']}")

    cross_checks(tables)
    pd.concat(tables.values()).to_csv(RES / "phase13_benchmark.csv", index=False)
    pd.concat(pair_frames).to_csv(RES / "phase13_pairs.csv", index=False)
    bins = pd.concat(bin_frames)
    bins.to_csv(RES / "phase13_bins.csv", index=False)
    (RES / "phase13_readme_table.md").write_text(readme_table(tables) + "\n")

    print("\n=== Error by molecule size and by true XLogP3 (scaffold split, MAE) ===")
    for grouping in ["heavy atoms", "true XLogP3"]:
        b = bins[(bins["split"] == "scaffold") & (bins["grouping"] == grouping)]
        wide = b.pivot_table(index="model", columns="bin", values="MAE", sort=False)
        counts = b.drop_duplicates("bin").set_index("bin")["n"]
        order = [lbl for _, _, lbl in (SIZE_BINS if grouping == "heavy atoms" else RANGE_BINS) if lbl in wide.columns]
        wide = wide[order]
        wide.columns = [f"{c} (n={counts[c]})" for c in order]
        print(f"\nBy {grouping}:")
        print(wide.round(3).to_string())
        small = [c for c in order if counts[c] < 30]
        if small:
            print(f"  careful: {', '.join(small)} has fewer than 30 molecules, so those numbers are shaky")

    fig_benchmark(tables)
    print("\nsaved results/phase13_benchmark.csv, phase13_pairs.csv, phase13_bins.csv, phase13_readme_table.md")
    print("Reminder: the winner was picked on this test set, so its margin is slightly optimistic.")


if __name__ == "__main__":
    main()
