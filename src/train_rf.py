"""Phase 7: first real model. Random Forest on fingerprints, descriptors, and both.

Run:  python src/train_rf.py
      python src/train_rf.py --trees 500

Reads   data/processed/clean.csv, morgan_r2_2048_counts.npz, descriptors.npz
Writes  results/phase7_rf_results.csv

What it answers
  1. Does a Random Forest beat "always guess the average"? (It must, by a lot.)
  2. Which features work better: fingerprints, descriptors, or both?
  3. How big is the gap between a random split and a scaffold split?
  4. How much does the leaky MolLogP column inflate the score? (Shown once, on purpose.)

Target is PubChem XLogP3, so every number below is "error in XLogP3 units".
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from descriptors import CRIPPEN_FAMILY, load_descriptors
from features import load_morgan
from splits import get_splits, self_checks

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "processed" / "clean.csv"
OUT_CSV = ROOT / "results" / "phase7_rf_results.csv"
SEED = 42


def scores(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def build_feature_sets(clean):
    """name -> (matrix, column names). Every matrix is row-aligned with clean.csv."""
    X_fp, cid_fp = load_morgan()
    X_desc, names_desc, cid_d = load_descriptors()                      # MolLogP/MolMR excluded
    X_leaky, names_leaky, _ = load_descriptors(keep_crippen=True)       # only for the leakage demo
    for cid in (cid_fp, cid_d):
        assert (cid == clean["cid"].values).all(), "feature rows are not aligned with clean.csv"

    fp_names = [f"bit_{i}" for i in range(X_fp.shape[1])]
    return {
        "Morgan counts": (X_fp.astype(np.float32), fp_names),
        "Descriptors": (X_desc, names_desc),
        "Morgan + descriptors": (np.hstack([X_fp.astype(np.float64), X_desc]), fp_names + names_desc),
        "Descriptors + MolLogP (LEAKY)": (X_leaky, names_leaky),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()

    clean = pd.read_csv(CLEAN_CSV)
    y = clean["xlogp"].values
    smiles = clean["smiles"].tolist()
    print(f"dataset: {len(clean)} molecules, target = XLogP3 (mean {y.mean():.2f}, std {y.std():.2f})")

    splits = get_splits(smiles)
    self_checks(smiles, splits)
    feature_sets = build_feature_sets(clean)

    rows = []
    importances = None
    for split_name, (tr, te) in splits.items():
        y_tr, y_te = y[tr], y[te]

        # Baselines: what a model with zero chemistry knowledge scores.
        for label, guess in [("Baseline: always the median", np.median(y_tr)),
                             ("Baseline: always the mean", y_tr.mean())]:
            s = scores(y_te, np.full(len(te), guess))
            rows.append({"split": split_name, "model": label, "train_MAE": np.nan, **s})

        for fs_name, (X, names) in feature_sets.items():
            t0 = time.time()
            rf = RandomForestRegressor(n_estimators=args.trees, max_features="sqrt",
                                       n_jobs=-1, random_state=SEED)
            rf.fit(X[tr], y_tr)
            s = scores(y_te, rf.predict(X[te]))
            train_mae = mean_absolute_error(y_tr, rf.predict(X[tr]))
            rows.append({"split": split_name, "model": f"RF: {fs_name}", "train_MAE": train_mae, **s})
            print(f"  [{split_name:<8}] {fs_name:<30} test MAE {s['MAE']:.3f}  ({time.time() - t0:.0f}s)")
            if split_name == "random" and fs_name == "Descriptors":
                importances = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)

    results = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    results.to_csv(OUT_CSV, index=False)

    line = "=" * 92
    print(f"\n{line}\nRANDOM FOREST RESULTS  ({args.trees} trees, test = 20% held out, units = XLogP3)\n{line}")
    for split_name in splits:
        part = results[results["split"] == split_name]
        print(f"\n{split_name.upper()} split   train {len(splits[split_name][0])} / test {len(splits[split_name][1])}")
        print(f"  {'model':<36}{'test MAE':>9}{'test RMSE':>11}{'test R2':>9}{'train MAE':>11}")
        for _, r in part.iterrows():
            tm = "" if np.isnan(r["train_MAE"]) else f"{r['train_MAE']:.3f}"
            print(f"  {r['model']:<36}{r['MAE']:>9.3f}{r['RMSE']:>11.3f}{r['R2']:>9.3f}{tm:>11}")

    # Honest reading aids
    base = results[(results["split"] == "random") & (results["model"] == "Baseline: always the median")].iloc[0]
    clean_rows = results[(results["split"] == "random") & results["model"].str.startswith("RF:")
                         & ~results["model"].str.contains("LEAKY")]
    best = clean_rows.sort_values("MAE").iloc[0]
    leaky = results[(results["split"] == "random") & results["model"].str.contains("LEAKY")].iloc[0]
    desc = results[(results["split"] == "random") & (results["model"] == "RF: Descriptors")].iloc[0]
    print(f"\nbest honest model (random split): {best['model']}  MAE {best['MAE']:.3f} "
          f"vs median baseline {base['MAE']:.3f}  ({1 - best['MAE'] / base['MAE']:.0%} lower error)")
    print(f"leakage demo: adding {' + '.join(CRIPPEN_FAMILY)} moves Descriptors MAE "
          f"{desc['MAE']:.3f} -> {leaky['MAE']:.3f}  (R2 {desc['R2']:.3f} -> {leaky['R2']:.3f})")
    print("   that gain comes from copying another logP calculator, not from learning chemistry.")

    print("\ntop 10 descriptors by Random Forest importance (random split):")
    for name, v in importances.head(10).items():
        print(f"  {name:<28}{v:.3f}")
    print(f"\nsaved {OUT_CSV.relative_to(ROOT)}")
    print(line)


if __name__ == "__main__":
    main()
