"""Phase 6: RDKit descriptors (SMILES -> ~200 named, human-readable numbers).

Run:  python src/descriptors.py

Reads   data/processed/clean.csv
Writes  data/processed/descriptors.npz   (X, names, medians, cid; git-ignored)

Other scripts reuse:
  compute_descriptors(smiles_list)   SMILES -> DataFrame, one column per descriptor
  load_descriptors()                 (X, names, cid), cleaned and row-aligned with clean.csv
  apply_schema(df, names, medians)   make a fresh DataFrame match the saved columns
                                     (this is what predict.py will call)

Input SMILES must already be standardized (data_processing.standardize).

LEAKAGE WARNING
  Our target is PubChem XLogP3. RDKit ships its own logP estimate, MolLogP
  (Crippen). Both are algorithms, and on our data they correlate at ~0.90.
  Feeding MolLogP to a model is like giving it a second opinion of the answer,
  so the model would look brilliant and learn very little chemistry.
  MolMR comes from the same Crippen atom typing, so it is excluded too.
  load_descriptors() drops both unless you pass keep_crippen=True (which is
  how Phase 7 shows the size of the effect).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "processed" / "clean.csv"
OUT_PATH = ROOT / "data" / "processed" / "descriptors.npz"

CRIPPEN_FAMILY = ["MolLogP", "MolMR"]   # same method as our target's cousin: leakage
MAX_NAN_FRACTION = 0.01                 # drop a column if more than 1% of molecules fail on it


# ----------------------------------------------------------------------
# Core: SMILES -> descriptors
# ----------------------------------------------------------------------
def all_descriptor_names():
    return [name for name, _ in Descriptors._descList]


def compute_descriptors(smiles_list):
    """List of standardized SMILES -> DataFrame (n molecules x all RDKit descriptors).

    A descriptor that fails on a molecule becomes NaN (not a crash). Infinite
    values are turned into NaN as well, so "missing" has one meaning.
    """
    names = all_descriptor_names()
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"cannot parse SMILES: {smiles!r}")
        row = {}
        for name, fn in Descriptors._descList:
            try:
                row[name] = float(fn(mol))
            except Exception:
                row[name] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows, columns=names)
    return df.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------
# Cleaning: decide which columns are usable, and how to fill the gaps
# ----------------------------------------------------------------------
def choose_schema(df):
    """Return (kept_names, medians, dropped) learned from the training data.

    dropped is a dict {name: reason}. Medians are learned here and saved, so a
    new molecule at prediction time is filled with the SAME numbers.
    """
    dropped = {}
    kept = []
    for name in df.columns:
        col = df[name]
        if col.isna().mean() > MAX_NAN_FRACTION:
            dropped[name] = f"{col.isna().mean():.1%} missing"
        elif col.nunique(dropna=True) <= 1:
            dropped[name] = "constant (same value for every molecule)"
        else:
            kept.append(name)
    medians = df[kept].median()
    return kept, medians, dropped


def apply_schema(df, names, medians):
    """Select the saved columns, in the saved order, and fill gaps with saved medians."""
    out = df.reindex(columns=list(names)).replace([np.inf, -np.inf], np.nan)
    return out.fillna(pd.Series(dict(zip(names, medians))))


def load_descriptors(keep_crippen=False):
    """(X float64 matrix, names list, cid array), row-aligned with clean.csv."""
    z = np.load(OUT_PATH, allow_pickle=False)
    X, names = z["X"], [str(n) for n in z["names"]]
    if not keep_crippen:
        keep = [i for i, n in enumerate(names) if n not in CRIPPEN_FAMILY]
        X, names = X[:, keep], [names[i] for i in keep]
    return X, names, z["cid"]


# ----------------------------------------------------------------------
# Build + report
# ----------------------------------------------------------------------
def build():
    clean = pd.read_csv(CLEAN_CSV)
    t0 = time.time()
    raw = compute_descriptors(clean["smiles"].tolist())
    seconds = time.time() - t0

    kept, medians, dropped = choose_schema(raw)
    filled = apply_schema(raw, kept, medians.values)

    np.savez_compressed(
        OUT_PATH,
        X=filled.values.astype(np.float64),
        names=np.array(kept),
        medians=medians.values.astype(np.float64),
        cid=clean["cid"].values,
    )
    return clean, raw, filled, dropped, seconds


def report(clean, raw, filled, dropped, seconds):
    y = clean["xlogp"]
    line = "=" * 74
    print(f"\n{line}\nRDKIT DESCRIPTORS\n{line}")
    print(f"molecules                : {len(raw)}   (rows match clean.csv: {len(raw) == len(clean)})")
    print(f"descriptors RDKit offers : {raw.shape[1]}")
    print(f"built in                 : {seconds:.1f} s")

    print("\n-- which columns did we drop, and why? --")
    if dropped:
        for name, why in dropped.items():
            print(f"  {name:<28} {why}")
    else:
        print("  none")
    n_gaps = int(raw[filled.columns].isna().sum().sum())
    print(f"kept {filled.shape[1]} columns; filled {n_gaps} individual missing cells with the column median")

    print("\n-- sanity: do the numbers look like chemistry? --")
    for name in ["MolWt", "HeavyAtomCount", "TPSA", "NumHDonors", "NumHAcceptors", "RingCount"]:
        c = filled[name]
        print(f"  {name:<16} min {c.min():8.1f}   median {c.median():8.1f}   max {c.max():8.1f}")

    print("\n-- LEAKAGE CHECK: correlation with the target (XLogP3) --")
    corr = filled.corrwith(y).dropna()
    for name in CRIPPEN_FAMILY:
        print(f"  {name:<12} r = {raw[name].corr(y):+.3f}   <- excluded from the model inputs")
    print("  strongest remaining descriptors (by |r|):")
    for name in corr.drop(CRIPPEN_FAMILY, errors="ignore").abs().sort_values(ascending=False).index[:10]:
        print(f"    {name:<26} r = {corr[name]:+.3f}")

    print("\n-- redundancy: many descriptors measure nearly the same thing --")
    usable = filled.drop(columns=CRIPPEN_FAMILY, errors="ignore")
    c = usable.corr().abs().to_numpy(copy=True)
    np.fill_diagonal(c, 0)
    pairs = int((np.triu(c) > 0.95).sum())
    print(f"  pairs of descriptors with |r| > 0.95 : {pairs}  (harmless for trees, matters for linear models)")

    print("\n-- scale: descriptors live on very different ranges --")
    spans = usable.max() - usable.min()
    top = spans.sort_values(ascending=False).index[:3]
    print("  widest ranges: " + ", ".join(f"{n} ({spans[n]:.3g})" for n in top))
    print("  trees do not care about scale; a linear model or a neural net would (Phase 11 onward)")
    print(line)


def self_checks(clean, filled):
    assert len(filled) == len(clean), "row count changed"
    assert not filled.isna().any().any(), "NaN left after filling"
    assert np.isfinite(filled.values).all(), "inf left after filling"
    X, names, cid = load_descriptors()
    assert (cid == clean["cid"].values).all(), "saved rows are not aligned with clean.csv"
    assert not any(n in names for n in CRIPPEN_FAMILY), "Crippen leaked into default inputs"
    Xk, nk, _ = load_descriptors(keep_crippen=True)
    assert "MolLogP" in nk and Xk.shape[1] == X.shape[1] + 2
    ethanol = apply_schema(compute_descriptors(["CCO"]), nk, np.median(Xk, axis=0))
    assert ethanol.shape == (1, len(nk)) and not ethanol.isna().any().any()
    print("self-checks passed")


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    clean, raw, filled, dropped, seconds = build()
    print(f"saved {OUT_PATH.relative_to(ROOT)}")
    report(clean, raw, filled, dropped, seconds)
    self_checks(clean, filled)


if __name__ == "__main__":
    main()
