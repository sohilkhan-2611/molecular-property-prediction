"""Phase 9, part 1: look INSIDE the models. Where do they predict well, and where badly?

Run:  python src/evaluate.py
Then: python src/make_figures.py

Reads   data/processed/clean.csv + the feature files
        results/phase8_xgb_results.csv   (to reuse the tuned XGBoost settings, no re-tuning)
Writes  results/phase9_predictions.csv   one row per test molecule per model (reused in Phase 13)
        results/phase9_importance.csv    which inputs XGBoost relied on
        results/phase9_worst.csv         the 15 biggest misses of the best model (scaffold split)

Three models are refit on the Phase 7 splits, same seeds, same train/test molecules:
  RF: Descriptors          the Phase 7 winner
  XGB: Descriptors         tuned settings from Phase 8
  XGB: Morgan + descriptors  tuned settings from Phase 8 (our best honest model)

Then it answers four questions in plain numbers:
  1. Are differences between models bigger than test-set noise?   (bootstrap)
  2. Does the error depend on molecule size?                      (size bins)
  3. Which inputs does the model use?                             (importance)
  4. Which molecules does it get most wrong?                      (worst list)
"""

import ast
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from features import slot_contents
from splits import get_splits, self_checks
from train_rf import SEED, build_feature_sets, scores
from train_xgb import make_model

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "processed" / "clean.csv"
XGB_CSV = ROOT / "results" / "phase8_xgb_results.csv"
PRED_CSV = ROOT / "results" / "phase9_predictions.csv"
IMP_CSV = ROOT / "results" / "phase9_importance.csv"
WORST_CSV = ROOT / "results" / "phase9_worst.csv"

SIZE_BINS = [(0, 15), (16, 22), (23, 29), (30, 36), (37, 999)]   # heavy atoms per molecule
N_BOOT = 2000


def tuned_settings(results, split_name, feature_set):
    """Read back the settings Phase 8 chose: (params dict, number of trees)."""
    row = results[(results["split"] == split_name) & (results["model"] == f"XGB tuned: {feature_set}")]
    if row.empty:
        raise SystemExit(f"no tuned row for {feature_set!r} / {split_name} in {XGB_CSV.name}. Run Phase 8 first.")
    text = row.iloc[0]["params"]
    params = ast.literal_eval(text[: text.index("}") + 1])
    trees = int(re.search(r"(\d+) trees", text).group(1))
    return params, trees


def fit_models(clean, splits, feature_sets, results):
    """Return (predictions DataFrame, importance DataFrame)."""
    y = clean["xlogp"].values
    rows, importance = [], None
    t0 = time.time()
    for split_name, (tr, te) in splits.items():
        plan = [("RF: Descriptors", "Descriptors", None),
                ("XGB: Descriptors", "Descriptors", "xgb"),
                ("XGB: Morgan + descriptors", "Morgan + descriptors", "xgb")]
        for label, fs_name, kind in plan:
            X, names = feature_sets[fs_name]
            if kind is None:
                model = RandomForestRegressor(n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=SEED)
            else:
                params, trees = tuned_settings(results, split_name, fs_name)
                model = make_model(params, trees)
                model.set_params(importance_type="gain")
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            rows.append(pd.DataFrame({
                "split": split_name, "model": label, "cid": clean["cid"].values[te],
                "smiles": clean["smiles"].values[te], "n_heavy": clean["n_heavy"].values[te],
                "mol_wt": clean["mol_wt"].values[te], "y_true": y[te], "y_pred": pred}))
            print(f"  fitted [{split_name:<8}] {label:<28} test MAE {mean_absolute_error(y[te], pred):.3f}"
                  f"  ({time.time() - t0:.0f}s so far)")
            if split_name == "scaffold" and label == "XGB: Morgan + descriptors":
                importance = pd.DataFrame({"feature": names, "gain": model.feature_importances_})
    return pd.concat(rows, ignore_index=True), importance


def describe_importance(importance, clean, top=15):
    """Add a plain-language name to fingerprint slots, e.g. 'bit_1380' -> 'bit 1380: c (aromatic carbon)'."""
    importance = importance.sort_values("gain", ascending=False).reset_index(drop=True)
    importance["kind"] = np.where(importance["feature"].str.startswith("bit_"), "fingerprint slot", "descriptor")
    slots = [int(f.split("_")[1]) for f in importance.head(top)["feature"] if f.startswith("bit_")]
    found = slot_contents(clean["smiles"].tolist(), slots) if slots else {}
    labels = []
    for f in importance["feature"]:
        if f.startswith("bit_") and int(f.split("_")[1]) in found and found[int(f.split("_")[1])]:
            frag = found[int(f.split("_")[1])].most_common(1)[0][0]
            labels.append(f"slot {f.split('_')[1]}: {frag}")
        else:
            labels.append(f)
    importance["label"] = labels
    importance["share"] = importance["gain"] / importance["gain"].sum()
    return importance


def bootstrap(pred, split_name, seed=SEED, n_boot=N_BOOT):
    """95% intervals for each model's MAE and for the differences between models."""
    part = pred[pred["split"] == split_name]
    wide = part.pivot(index="cid", columns="model", values="y_pred")
    truth = part.drop_duplicates("cid").set_index("cid").loc[wide.index, "y_true"].values
    errs = {m: np.abs(wide[m].values - truth) for m in wide.columns}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(truth), size=(n_boot, len(truth)))
    boots = {m: e[idx].mean(axis=1) for m, e in errs.items()}
    ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
    out = {"mae": {m: (errs[m].mean(), *ci(boots[m])) for m in errs}, "diff": {}}
    for a, b in [("XGB: Descriptors", "RF: Descriptors"),
                 ("XGB: Morgan + descriptors", "XGB: Descriptors")]:
        d = boots[a] - boots[b]
        out["diff"][(a, b)] = (errs[a].mean() - errs[b].mean(), *ci(d))
    return out


def report(pred, importance, splits):
    line = "=" * 92
    print(f"\n{line}\nMODEL CHECK-UP (Phase 9)  units = XLogP3, test sets of {len(splits['random'][1])} molecules\n{line}")

    print("\n1) IS THE DIFFERENCE REAL OR NOISE?  95% intervals from resampling the test set 2000 times")
    for split_name in splits:
        b = bootstrap(pred, split_name)
        print(f"\n  {split_name.upper()} split")
        for m, (v, lo, hi) in b["mae"].items():
            print(f"    {m:<28} MAE {v:.3f}   95% interval [{lo:.3f}, {hi:.3f}]")
        for (a, c), (v, lo, hi) in b["diff"].items():
            verdict = "real (interval excludes 0)" if (lo > 0 or hi < 0) else "NOT distinguishable from noise"
            print(f"    {a} minus {c}: {v:+.3f} [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")

    print("\n2) DOES ERROR DEPEND ON MOLECULE SIZE?  (scaffold test split, best model vs RF)")
    part = pred[(pred["split"] == "scaffold")]
    print(f"    {'heavy atoms':<14}{'n':>5}{'XGB MAE':>10}{'RF MAE':>9}   mean true XLogP3")
    for lo, hi in SIZE_BINS:
        sel = part[(part["n_heavy"] >= lo) & (part["n_heavy"] <= hi)]
        x = sel[sel["model"] == "XGB: Morgan + descriptors"]
        r = sel[sel["model"] == "RF: Descriptors"]
        name = f"<= {hi}" if lo == 0 else (f">= {lo}" if hi == 999 else f"{lo} to {hi}")
        if len(x):
            print(f"    {name:<14}{len(x):>5}{np.abs(x.y_true - x.y_pred).mean():>10.3f}"
                  f"{np.abs(r.y_true - r.y_pred).mean():>9.3f}   {x.y_true.mean():.2f}")
    print(f"\n    error by TRUE value (does it struggle at the extremes?)")
    print(f"    {'true XLogP3':<14}{'n':>5}{'XGB MAE':>10}{'avg miss direction':>22}")
    for lo, hi, name in [(-99, 0, "< 0"), (0, 2, "0 to 2"), (2, 4, "2 to 4"), (4, 6, "4 to 6"), (6, 99, ">= 6")]:
        x = part[(part["model"] == "XGB: Morgan + descriptors") & (part["y_true"] >= lo) & (part["y_true"] < hi)]
        if len(x):
            signed = (x.y_pred - x.y_true).mean()
            way = "too high" if signed > 0.05 else ("too low" if signed < -0.05 else "balanced")
            print(f"    {name:<14}{len(x):>5}{np.abs(x.y_true - x.y_pred).mean():>10.3f}{signed:>+14.2f} ({way})")
    small = pred[(pred["split"] == "random") & (pred["model"] == "XGB: Morgan + descriptors") & (pred["n_heavy"] <= 8)]
    print(f"    tiny molecules (<= 8 heavy atoms) in the random test set: {len(small)}"
          + (f", their MAE {np.abs(small.y_true - small.y_pred).mean():.3f}" if len(small) else ""))

    print("\n3) WHAT DOES XGBOOST RELY ON?  (share of total 'gain', scaffold-split model)")
    by_kind = importance.groupby("kind")["share"].sum()
    for kind, share in by_kind.items():
        n = int((importance["kind"] == kind).sum())
        print(f"    {kind:<18}{share:>6.1%} of the signal from {n} inputs")
    for _, r in importance.head(10).iterrows():
        print(f"    {r['label']:<46}{r['share']:>6.1%}")

    print("\n4) BIGGEST MISSES  (best model, scaffold split)")
    best = pred[(pred["split"] == "scaffold") & (pred["model"] == "XGB: Morgan + descriptors")].copy()
    best["error"] = best["y_pred"] - best["y_true"]
    worst = best.reindex(best["error"].abs().sort_values(ascending=False).index).head(15)
    for _, r in worst.head(8).iterrows():
        print(f"    true {r['y_true']:+6.2f}  pred {r['y_pred']:+6.2f}  err {r['error']:+6.2f}  heavy {int(r['n_heavy']):>3}  {r['smiles'][:60]}")
    over = (best["error"].abs() > 1.0).mean()
    print(f"    share of test molecules off by more than 1.0: {over:.1%};  worst single miss {worst['error'].abs().iloc[0]:.2f}")
    bias = best["error"].mean()
    print(f"    average signed error (bias): {bias:+.3f}   (near 0 = not systematically high or low)")
    print(line)
    return worst


def main():
    clean = pd.read_csv(CLEAN_CSV)
    smiles = clean["smiles"].tolist()
    splits = get_splits(smiles)
    self_checks(smiles, splits)
    if not XGB_CSV.exists():
        raise SystemExit("results/phase8_xgb_results.csv not found. Run: python src/train_xgb.py")
    results = pd.read_csv(XGB_CSV)
    feature_sets = build_feature_sets(clean)

    print("refitting three models with the settings Phase 7 and 8 chose ...")
    pred, importance = fit_models(clean, splits, feature_sets, results)
    importance = describe_importance(importance, clean)

    PRED_CSV.parent.mkdir(exist_ok=True)
    pred.to_csv(PRED_CSV, index=False)
    importance.to_csv(IMP_CSV, index=False)
    worst = report(pred, importance, splits)
    worst.to_csv(WORST_CSV, index=False)
    for p in (PRED_CSV, IMP_CSV, WORST_CSV):
        print(f"saved {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
