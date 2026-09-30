"""Phase 4: clean the raw PubChem pull into a training set.

Run:  python src/data_processing.py
      python src/data_processing.py --max-mw 800      (try a different weight cap)

Reads   data/raw/pubchem_xlogp_raw.csv
Writes  data/processed/clean.csv
        data/processed/cleaning_log.json   (row counts after every step)

Other scripts reuse two things from here:
  standardize(smiles)   one SMILES -> canonical form, or the reason it was rejected
  load_clean()          the cleaned DataFrame

Every rule below is applied to a NEW molecule at prediction time too
(predict.py in Phase 14), so training data and user input match.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")  # we handle bad SMILES ourselves, no console spam

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "pubchem_xlogp_raw.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "clean.csv"
LOG_JSON = ROOT / "data" / "processed" / "cleaning_log.json"

MAX_MW = 1000.0        # your decision: keep molecules with weight <= 1000
CONFLICT_TOL = 0.5     # duplicates whose labels differ by more than this are dropped
ALLOWED_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}

# Order matters: a row is counted under the FIRST rule that rejects it.
REASONS = [
    ("invalid_smiles", "SMILES missing, unparseable, or zero atoms"),
    ("multi_component", "several separate pieces (salts, solvates)"),
    ("bad_element", "element outside C H N O F P S Cl Br I"),
    ("too_heavy", "molecular weight above the cap"),
]


# ----------------------------------------------------------------------
# One molecule
# ----------------------------------------------------------------------
def standardize(smiles, max_mw=MAX_MW):
    """Check one SMILES and put it in canonical form.

    Returns a dict: smiles (canonical, no stereo) or None, reason (why it was
    rejected) or None, plus mol_wt, n_heavy, net_charge when computable.
    """
    out = {"smiles": None, "reason": None,
           "mol_wt": None, "n_heavy": None, "net_charge": None}

    if not isinstance(smiles, str) or not smiles.strip():
        out["reason"] = "invalid_smiles"
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        out["reason"] = "invalid_smiles"
        return out

    out["mol_wt"] = Descriptors.MolWt(mol)
    out["n_heavy"] = mol.GetNumHeavyAtoms()
    out["net_charge"] = Chem.GetFormalCharge(mol)

    if len(Chem.GetMolFrags(mol)) > 1:
        out["reason"] = "multi_component"
    elif {a.GetSymbol() for a in mol.GetAtoms()} - ALLOWED_ELEMENTS:
        out["reason"] = "bad_element"
    elif out["mol_wt"] > max_mw:
        out["reason"] = "too_heavy"
    else:
        # Reduce to the plain 2D structure, so the same molecule always gets
        # the same string:
        #  1. drop stereo (@, @@, /, \): mirror images and E/Z variants merge
        #  2. drop isotope labels: a tritium-labelled compound ([3H]) is the
        #     same molecule for logP purposes (one such row is in the data)
        #  3. remove the explicit [H] atoms that are left over
        Chem.RemoveStereochemistry(mol)
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        mol = Chem.RemoveHs(mol)
        out["smiles"] = Chem.MolToSmiles(mol, isomericSmiles=False)
    return out


# ----------------------------------------------------------------------
# Whole dataset
# ----------------------------------------------------------------------
def resolve_duplicates(df, tol):
    """Merge rows that are the same molecule after standardising.

    Same molecule, labels within tol  -> keep one row, average the label.
    Same molecule, labels differ > tol -> drop them all: we cannot know which
    label is right, and keeping either would teach the model a coin flip.
    """
    grouped = df.groupby("smiles").agg(
        cid=("cid", "min"),
        xlogp=("xlogp", "mean"),
        spread=("xlogp", lambda s: s.max() - s.min()),
        n_records=("xlogp", "size"),
        mol_wt=("mol_wt", "first"),
        n_heavy=("n_heavy", "first"),
        net_charge=("net_charge", "first"),
    ).reset_index()
    conflicts = grouped[(grouped["n_records"] > 1) & (grouped["spread"] > tol)]
    kept = grouped.drop(conflicts.index).drop(columns="spread")
    n_merged = int(kept["n_records"].gt(1).sum())
    return kept, len(conflicts), int(conflicts["n_records"].sum()), n_merged


def clean(raw, max_mw=MAX_MW, tol=CONFLICT_TOL):
    """Raw PubChem DataFrame -> (clean DataFrame, list of (step, rows_left))."""
    steps = [("raw rows from PubChem", len(raw))]

    df = raw.rename(columns={"CID": "cid", "XLogP": "xlogp"})
    df = df[df["xlogp"].notna()].copy()
    steps.append(("has an XLogP value", len(df)))

    info = pd.DataFrame([standardize(s, max_mw) for s in df["SMILES"]], index=df.index)
    df = pd.concat([df[["cid", "xlogp"]], info], axis=1)

    for reason, label in REASONS:
        df = df[df["reason"] != reason]
        steps.append((f"drop {reason}", len(df)))

    df = df.drop(columns="reason")
    df, n_conf_groups, n_conf_rows, n_merged = resolve_duplicates(df, tol)
    steps.append(("merge duplicates + drop label conflicts", len(df)))

    df = df.sort_values("cid").reset_index(drop=True)
    df = df[["cid", "smiles", "xlogp", "mol_wt", "n_heavy", "net_charge", "n_records"]]
    extra = {"conflict_groups_dropped": n_conf_groups,
             "conflict_rows_dropped": n_conf_rows,
             "duplicate_groups_merged": n_merged}
    return df, steps, extra


def load_clean():
    """The cleaned dataset, for every later phase."""
    if not CLEAN_CSV.exists():
        sys.exit(f"{CLEAN_CSV} not found. Run: python src/data_processing.py")
    return pd.read_csv(CLEAN_CSV)


# ----------------------------------------------------------------------
# Checks and report
# ----------------------------------------------------------------------
def self_checks(df):
    """Things that must be true, or every later result is suspect."""
    results = []

    results.append(("no duplicate SMILES (leakage guard)", df["smiles"].is_unique))
    results.append(("no missing values in core columns",
                    not df[["cid", "smiles", "xlogp", "mol_wt"]].isna().any().any()))

    again = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in df["smiles"]]
    results.append(("every SMILES is already canonical", again == df["smiles"].tolist()))
    results.append(("no stereo marks left",
                    not df["smiles"].str.contains(r"[@/\\]", regex=True).any()))
    results.append(("no multi-component SMILES", not df["smiles"].str.contains(".", regex=False).any()))
    return results


def report(df, steps, extra, max_mw):
    print("=" * 74)
    print(f"CLEANING STEPS  (molecular weight cap: {max_mw:g})")
    print("=" * 74)
    print(f"{'step':<44} {'rows left':>10} {'removed':>9}")
    print("-" * 74)
    prev = None
    for label, n in steps:
        rem = "" if prev is None else str(prev - n)
        print(f"{label:<44} {n:>10} {rem:>9}")
        prev = n
    print("-" * 74)
    print(f"  duplicate groups merged (same label)   : {extra['duplicate_groups_merged']}")
    print(f"  conflicting-label groups dropped       : {extra['conflict_groups_dropped']} "
          f"({extra['conflict_rows_dropped']} rows)")

    print("\n" + "=" * 74)
    print("SELF-CHECKS")
    print("=" * 74)
    ok = True
    for name, passed in self_checks(df):
        print(f"  [{'ok ' if passed else 'FAIL'}] {name}")
        ok = ok and passed

    print("\n" + "=" * 74)
    print("CLEAN DATASET  (paste this block back to Claude)")
    print("=" * 74)
    print(f"shape: {df.shape}")
    print("\nfirst 5 rows:")
    print(df.head(5).to_string(max_colwidth=45))
    print("\nmissing values per column:")
    print(df.isna().sum().to_string())
    print("\nxlogp (the target):")
    print(df["xlogp"].describe().round(3).to_string())
    print("\nquantiles:")
    print(df["xlogp"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(2).to_string())
    print("\nmol_wt:")
    print(df["mol_wt"].describe().round(1).to_string())
    print("\nheavy atoms (non-hydrogen):")
    print(df["n_heavy"].describe().round(1).to_string())
    print(f"\nmolecules with a net charge : {(df['net_charge'] != 0).sum()}")
    print(f"target above 10 or below -5 : {((df['xlogp'] > 10) | (df['xlogp'] < -5)).sum()}")
    print(f"molecules with <= 8 heavy atoms (ethanol has 3): {(df['n_heavy'] <= 8).sum()}")
    if not ok:
        sys.exit("A self-check failed. Do not continue until this is fixed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-mw", type=float, default=MAX_MW, help="molecular weight cap (default 1000)")
    ap.add_argument("--tol", type=float, default=CONFLICT_TOL, help="duplicate label conflict tolerance (default 0.5)")
    args = ap.parse_args()

    if not RAW_CSV.exists():
        sys.exit(f"{RAW_CSV} not found. Run: python src/pubchem_pull.py")
    raw = pd.read_csv(RAW_CSV)

    df, steps, extra = clean(raw, args.max_mw, args.tol)

    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    LOG_JSON.write_text(json.dumps(
        {"max_mw": args.max_mw, "tol": args.tol,
         "steps": [{"step": s, "rows_left": n} for s, n in steps], **extra}, indent=2))
    print(f"saved {len(df)} rows to {CLEAN_CSV.relative_to(ROOT)}\n")

    report(df, steps, extra, args.max_mw)


if __name__ == "__main__":
    main()
