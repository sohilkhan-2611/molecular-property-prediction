"""Phase 8: XGBoost, tuned honestly.

Run:  python src/train_xgb.py
      python src/train_xgb.py --quick        (tiny grid, for a smoke test)

Reads   data/processed/clean.csv, morgan_r2_2048_counts.npz, descriptors.npz
        results/phase7_rf_results.csv        (optional, for the side-by-side table)
Writes  results/phase8_xgb_results.csv

How the tuning stays honest
  * Settings are chosen by 5-fold cross-validation INSIDE the training set only.
  * The test set is used once per model, after the settings are frozen.
  * Random split  -> ordinary shuffled folds.
    Scaffold split -> group folds, so a ring skeleton never sits in both the
    fit part and the validation part. Otherwise we would tune for "familiar
    skeletons" and then get a bad surprise on the scaffold test.
  * Each fold stops adding trees when its validation error stops improving
    (early stopping). The average stopping point becomes the final tree count.
    This makes the CV number slightly optimistic, which is why we compare it
    with the real test number at the end.

Target is PubChem XLogP3; every error is in XLogP3 units.
"""

import argparse
import itertools
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

from splits import get_splits, murcko_scaffold, self_checks
from train_rf import SEED, build_feature_sets, scores

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "processed" / "clean.csv"
RF_CSV = ROOT / "results" / "phase7_rf_results.csv"
OUT_CSV = ROOT / "results" / "phase8_xgb_results.csv"

LEARNING_RATE = 0.1
MAX_TREES = 800
EARLY_STOP = 30
N_FOLDS = 5

GRID = {
    "max_depth": [4, 6, 8],           # how many questions deep each tree may go
    "min_child_weight": [3],          # a leaf needs at least this much evidence
    "colsample_bytree": [0.3, 0.7],   # share of columns each tree may look at
    "subsample": [0.8],               # share of rows each tree may look at
}
GRID_QUICK = {"max_depth": [4, 6], "min_child_weight": [3], "colsample_bytree": [0.5], "subsample": [0.8]}

DEFAULTS = {"max_depth": 6, "min_child_weight": 1, "colsample_bytree": 1.0, "subsample": 1.0}


def make_model(params, n_estimators, early_stopping=False):
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=LEARNING_RATE,
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
        early_stopping_rounds=EARLY_STOP if early_stopping else None,
        **params,
    )


def configs(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def cv_score(X, y, params, folds):
    """Mean validation MAE and mean best tree count across folds, for one setting."""
    maes, trees = [], []
    for fit_idx, val_idx in folds:
        model = make_model(params, MAX_TREES, early_stopping=True)
        model.fit(X[fit_idx], y[fit_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        maes.append(mean_absolute_error(y[val_idx], model.predict(X[val_idx])))
        trees.append(model.best_iteration + 1)
    return float(np.mean(maes)), int(round(np.mean(trees)))


def tune(X_tr, y_tr, groups_tr, split_name, grid):
    if split_name == "scaffold":
        folds = list(GroupKFold(n_splits=N_FOLDS).split(X_tr, y_tr, groups_tr))
    else:
        folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X_tr))
    best = None
    for params in configs(grid):
        mae, trees = cv_score(X_tr, y_tr, params, folds)
        if best is None or mae < best["cv_MAE"]:
            best = {"params": params, "n_estimators": trees, "cv_MAE": mae}
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="tiny grid for a fast smoke test")
    args = parser.parse_args()
    grid = GRID_QUICK if args.quick else GRID

    clean = pd.read_csv(CLEAN_CSV)
    y = clean["xlogp"].values
    smiles = clean["smiles"].tolist()
    splits = get_splits(smiles)
    self_checks(smiles, splits)
    scaffolds = np.array([murcko_scaffold(s) for s in smiles])
    feature_sets = build_feature_sets(clean)
    n_cfg = len(configs(grid))
    print(f"dataset: {len(clean)} molecules | grid: {n_cfg} settings x {N_FOLDS} folds per model | learning rate {LEARNING_RATE}")

    rows = []
    t_start = time.time()
    for split_name, (tr, te) in splits.items():
        y_tr, y_te = y[tr], y[te]
        for fs_name, (X, _names) in feature_sets.items():
            X_tr, X_te = X[tr], X[te]
            t0 = time.time()

            # 1) Untuned reference: library defaults (depth 6, all columns, all rows), fixed 300 trees at our learning rate.
            ref = make_model(DEFAULTS, 300).fit(X_tr, y_tr)
            s = scores(y_te, ref.predict(X_te))
            rows.append({"split": split_name, "model": f"XGB default: {fs_name}",
                         "train_MAE": mean_absolute_error(y_tr, ref.predict(X_tr)), **s,
                         "cv_MAE": np.nan, "params": "defaults, 300 trees"})

            # 2) Tuned: choose by CV inside train, then fit once on all of train and score on test once.
            best = tune(X_tr, y_tr, scaffolds[tr], split_name, grid)
            final = make_model(best["params"], best["n_estimators"]).fit(X_tr, y_tr)
            s = scores(y_te, final.predict(X_te))
            rows.append({"split": split_name, "model": f"XGB tuned: {fs_name}",
                         "train_MAE": mean_absolute_error(y_tr, final.predict(X_tr)), **s,
                         "cv_MAE": best["cv_MAE"],
                         "params": f"{best['params']}, {best['n_estimators']} trees"})
            print(f"  [{split_name:<8}] {fs_name:<30} default {rows[-2]['MAE']:.3f} -> tuned {s['MAE']:.3f}"
                  f"  (cv {best['cv_MAE']:.3f}, {time.time() - t0:.0f}s)")

    results = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    report(results, splits, time.time() - t_start)


def report(results, splits, seconds):
    line = "=" * 98
    print(f"\n{line}\nXGBOOST RESULTS  (test = 20% held out, units = XLogP3, total {seconds / 60:.1f} min)\n{line}")
    for split_name in splits:
        part = results[results["split"] == split_name]
        print(f"\n{split_name.upper()} split   train {len(splits[split_name][0])} / test {len(splits[split_name][1])}")
        print(f"  {'model':<40}{'test MAE':>9}{'test RMSE':>11}{'test R2':>9}{'cv MAE':>8}{'train MAE':>11}")
        for _, r in part.iterrows():
            cv = "" if np.isnan(r["cv_MAE"]) else f"{r['cv_MAE']:.3f}"
            print(f"  {r['model']:<40}{r['MAE']:>9.3f}{r['RMSE']:>11.3f}{r['R2']:>9.3f}{cv:>8}{r['train_MAE']:>11.3f}")

    if RF_CSV.exists():
        rf = pd.read_csv(RF_CSV)
        print("\nRANDOM FOREST (Phase 7) vs XGBOOST tuned, test MAE (lower is better)")
        print(f"  {'features':<32}{'RF random':>10}{'XGB random':>12}{'RF scaffold':>13}{'XGB scaffold':>14}")
        for fs in ["Morgan counts", "Descriptors", "Morgan + descriptors", "Descriptors + MolLogP (LEAKY)"]:
            cells = []
            for split_name, prefix in [("random", "RF: "), ("random", "XGB tuned: "),
                                       ("scaffold", "RF: "), ("scaffold", "XGB tuned: ")]:
                src = rf if prefix == "RF: " else results
                v = src[(src["split"] == split_name) & (src["model"] == prefix + fs)]["MAE"]
                cells.append(f"{v.iloc[0]:.3f}" if len(v) else "n/a")
            print(f"  {fs:<32}{cells[0]:>10}{cells[1]:>12}{cells[2]:>13}{cells[3]:>14}")

    honest = results[~results["model"].str.contains("LEAKY") & results["model"].str.startswith("XGB tuned")]
    print("\nchecks")
    for split_name in splits:
        part = honest[honest["split"] == split_name].sort_values("MAE")
        best = part.iloc[0]
        gap = best["MAE"] - best["cv_MAE"]
        print(f"  [{split_name}] best honest model: {best['model']}  test MAE {best['MAE']:.3f}  "
              f"(cv said {best['cv_MAE']:.3f}, gap {gap:+.3f})")
        # Real tuning overfit = the tuned model does WORSE than the untuned one on test.
        for _, r in part.iterrows():
            twin = results[(results["split"] == split_name) & (results["model"] == r["model"].replace("tuned", "default"))]
            if len(twin) and r["MAE"] > twin.iloc[0]["MAE"]:
                print(f"    WARNING: {r['model']} is worse than its untuned twin on test. Tuning overfitted.")
        if gap > 0.03 and split_name == "scaffold":
            print("    note: on the scaffold split the test set holds the rarest ring skeletons, while the CV folds\n"
                  "    contain big families (benzene, pyridine ...). A gap here is expected: CV is the easier exam.\n"
                  "    It is not tuning overfit as long as tuned still beats default on test (checked above).")
        elif gap > 0.05:
            print("    WARNING: test is clearly worse than CV promised.")
    # Is the grid too small? Winners sitting on its edge mean "try further out".
    tuned = results[results["model"].str.startswith("XGB tuned")]
    n_trees = tuned["params"].str.extract(r"(\d+) trees")[0].astype(int)
    depth = tuned["params"].str.extract(r"'max_depth': (\d+)")[0].astype(int)
    at_cap = int((n_trees >= 0.9 * MAX_TREES).sum())
    at_depth_edge = int((depth == min(GRID["max_depth"])).sum())
    print(f"\n  tuning headroom: {at_cap} of {len(tuned)} tuned models used >= 90% of the {MAX_TREES}-tree cap; "
          f"{at_depth_edge} of {len(tuned)} chose the shallowest depth in the grid ({min(GRID['max_depth'])}).")
    if at_cap or at_depth_edge:
        print("  => the grid edge is binding. More trees and shallower trees may still help a little.")
    print(f"\nsaved {OUT_CSV.relative_to(ROOT)}")
    print(line)


if __name__ == "__main__":
    main()
