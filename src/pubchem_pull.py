"""Phase 3: pull molecules and XLogP3 values from PubChem.

Modes
  python src/pubchem_pull.py --smoke              6 known molecules, proves the API works
  python src/pubchem_pull.py --n 6000 --seed 42   the real pull
  python src/pubchem_pull.py --report-only        re-print the data report for the saved CSV

Output (git-ignored, regenerate with the same seed):
  data/raw/pubchem_xlogp_raw.csv
  data/raw/pubchem_pull_meta.json

IMPORTANT: XLogP3 is a value PubChem COMPUTES with an algorithm. It is not a
lab measurement. Models trained on it learn to imitate that algorithm.
"""

import argparse
import io
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_CSV = RAW_DIR / "pubchem_xlogp_raw.csv"
OUT_META = RAW_DIR / "pubchem_pull_meta.json"

BATCH_SIZE = 100
PAUSE_SECONDS = 0.25  # PubChem asks for at most 5 requests per second
PROBE_CID = 2244      # aspirin, used to test which property names work

# PubChem renamed two property tags. We are not certain which spelling the
# API accepts today, so try both and normalise whatever comes back.
PROPERTY_SETS = [
    "XLogP,MolecularWeight,MolecularFormula,IsomericSMILES,CanonicalSMILES",
    "XLogP,MolecularWeight,MolecularFormula,SMILES,ConnectivitySMILES",
]
COLUMN_MAP = {
    "IsomericSMILES": "SMILES",
    "CanonicalSMILES": "ConnectivitySMILES",
}
FINAL_COLUMNS = [
    "CID", "SMILES", "ConnectivitySMILES",
    "MolecularFormula", "MolecularWeight", "XLogP",
]

SMOKE_MOLECULES = {
    702: "ethanol",
    241: "benzene",
    2244: "aspirin",
    2519: "caffeine",
    3672: "ibuprofen",
    1983: "paracetamol",
}


class PubChemError(Exception):
    """PubChem rejected the request (bad property name, unknown CID).
    Retrying the same request will not help."""


# ----------------------------------------------------------------------
# HTTP layer
# ----------------------------------------------------------------------
def make_session():
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "molecular-property-prediction/0.1 (learning project)"}
    )
    return session


def fetch_csv(session, cids, props, max_retries=5):
    """GET one batch as CSV text. Retries transient failures with backoff."""
    url = f"{BASE_URL}/{','.join(str(c) for c in cids)}/property/{props}/CSV"
    last = "no attempt made"
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = type(exc).__name__
        else:
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {resp.status_code}"
            else:
                raise PubChemError(
                    f"HTTP {resp.status_code}: {resp.text[:160].strip()}"
                )
        wait = 2 ** attempt
        print(f"    retry {attempt + 1}/{max_retries} in {wait}s ({last})")
        time.sleep(wait)
    raise RuntimeError(f"Gave up after {max_retries} attempts ({last}). "
                       "Check your internet connection and try again.")


def parse_csv(text):
    """CSV text -> DataFrame with the same columns every time."""
    if not text.strip():
        return pd.DataFrame(columns=FINAL_COLUMNS)
    df = pd.read_csv(io.StringIO(text)).rename(columns=COLUMN_MAP)
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[FINAL_COLUMNS]


def fetch_many(session, cids, props, bad_cids):
    """Fetch a batch. If PubChem rejects it, split in half until the
    offending CID is isolated, so one bad ID cannot sink 99 good ones."""
    time.sleep(PAUSE_SECONDS)
    try:
        text = fetch_csv(session, cids, props)
    except PubChemError:
        if len(cids) == 1:
            bad_cids.append(cids[0])
            return []
        mid = len(cids) // 2
        return (fetch_many(session, cids[:mid], props, bad_cids)
                + fetch_many(session, cids[mid:], props, bad_cids))
    df = parse_csv(text)
    return [] if df.empty else [df]


def choose_property_set(session):
    """Find which property spelling PubChem accepts, using aspirin as a probe."""
    for props in PROPERTY_SETS:
        time.sleep(PAUSE_SECONDS)
        try:
            text = fetch_csv(session, [PROBE_CID], props)
        except PubChemError as exc:
            print(f"  rejected : {props}\n             {exc}")
            continue
        df = parse_csv(text)
        if df["SMILES"].notna().any() and df["XLogP"].notna().any():
            print(f"  accepted : {props}")
            return props
        print(f"  no SMILES/XLogP for aspirin with: {props}")
    sys.exit("No working property set. Paste this whole output back to Claude.")


# ----------------------------------------------------------------------
# Pulling
# ----------------------------------------------------------------------
def sample_cids(n, max_cid, seed):
    if n > max_cid:
        sys.exit(f"--n ({n}) cannot exceed --max-cid ({max_cid})")
    return sorted(random.Random(seed).sample(range(1, max_cid + 1), n))


def pull(session, cids, props):
    frames, bad = [], []
    batches = [cids[i:i + BATCH_SIZE] for i in range(0, len(cids), BATCH_SIZE)]
    for i, batch in enumerate(batches, 1):
        frames += fetch_many(session, batch, props, bad)
        if i % 10 == 0 or i == len(batches):
            rows = sum(len(f) for f in frames)
            print(f"  batch {i}/{len(batches)}   rows so far: {rows}   "
                  f"rejected CIDs: {len(bad)}")
    if not frames:
        sys.exit("PubChem returned no rows at all.")
    df = pd.concat(frames, ignore_index=True)
    df["CID"] = pd.to_numeric(df["CID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["CID"]).drop_duplicates(subset="CID")
    df = df.sort_values("CID").reset_index(drop=True)
    for col in ["XLogP", "MolecularWeight"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, bad


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def report(df):
    print()
    print("=" * 70)
    print("DATA REPORT  (paste this whole block back to Claude)")
    print("=" * 70)
    print(f"shape: {df.shape}")

    print("\nfirst 5 rows:")
    print(df.head(5).to_string(max_colwidth=38))

    print("\nmissing values per column:")
    print(df.isna().sum().to_string())

    xl = df["XLogP"].dropna()
    print(f"\nXLogP (the target), {len(xl)} non-missing values:")
    if len(xl):
        print(xl.describe().round(3).to_string())
        qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        print("\nquantiles:")
        print(xl.quantile(qs).round(3).to_string())

    smiles = df["SMILES"].dropna().astype(str)
    n_no_xlogp = int(df["XLogP"].isna().sum())
    n_no_smiles = int(df["SMILES"].isna().sum())
    n_fragments = int(smiles.str.contains(".", regex=False).sum())
    n_stereo = int(smiles.str.contains(r"[@/\\]", regex=True).sum())
    n_dupes = int(df["ConnectivitySMILES"].dropna().duplicated().sum())
    n_heavy = int((df["MolecularWeight"] > 1000).sum())
    print("\nquick flags (Phase 4 will handle these):")
    print(f"  rows with no XLogP                  : {n_no_xlogp}")
    print(f"  rows with no SMILES                 : {n_no_smiles}")
    print(f"  multi-fragment SMILES (contain '.') : {n_fragments}")
    print(f"  SMILES with stereo marks            : {n_stereo}")
    print(f"  duplicate ConnectivitySMILES        : {n_dupes}")
    print(f"  MolecularWeight above 1000          : {n_heavy}")
    print("=" * 70)


def run_smoke(session, props):
    from rdkit import Chem
    from rdkit.Chem import Crippen

    df, bad = pull(session, list(SMOKE_MOLECULES), props)
    print()
    print(f"{'CID':>6} {'name':<12} {'XLogP3':>7} {'Crippen':>8} {'gap':>6}  SMILES")
    print("-" * 78)
    for _, row in df.iterrows():
        name = SMOKE_MOLECULES.get(int(row["CID"]), "?")
        mol = Chem.MolFromSmiles(row["SMILES"]) if isinstance(row["SMILES"], str) else None
        crip = Crippen.MolLogP(mol) if mol is not None else float("nan")
        print(f"{int(row['CID']):>6} {name:<12} {row['XLogP']:>7.1f} {crip:>8.2f} "
              f"{row['XLogP'] - crip:>6.2f}  {row['SMILES']}")
    print()
    print(f"rows returned: {len(df)} of {len(SMOKE_MOLECULES)}   rejected CIDs: {bad}")
    print("Two different algorithms, two different numbers for the same molecule.")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="test the API on 6 known molecules")
    ap.add_argument("--report-only", action="store_true", help="re-print the report for the saved CSV")
    ap.add_argument("--n", type=int, default=6000, help="how many CIDs to request (default 6000)")
    ap.add_argument("--max-cid", type=int, default=3_000_000, help="sample CIDs from 1..this (default 3,000,000)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    args = ap.parse_args()

    if args.report_only:
        if not OUT_CSV.exists():
            sys.exit(f"{OUT_CSV} does not exist yet. Run the pull first.")
        report(pd.read_csv(OUT_CSV))
        return

    session = make_session()
    print("Checking which PubChem property names work...")
    props = choose_property_set(session)

    if args.smoke:
        run_smoke(session, props)
        return

    cids = sample_cids(args.n, args.max_cid, args.seed)
    print(f"\nRequesting {len(cids)} CIDs from 1..{args.max_cid} "
          f"(seed {args.seed}), {BATCH_SIZE} per request")
    start = time.time()
    df, bad = pull(session, cids, props)
    elapsed = time.time() - start

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    OUT_META.write_text(json.dumps({
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_requested": len(cids),
        "max_cid": args.max_cid,
        "seed": args.seed,
        "property_set": props,
        "rows_returned": int(len(df)),
        "rows_with_xlogp": int(df["XLogP"].notna().sum()),
        "n_rejected_cids": len(bad),
        "rejected_cids_sample": bad[:50],
        "seconds": round(elapsed, 1),
    }, indent=2))

    print(f"\nsaved {len(df)} rows to {OUT_CSV.relative_to(RAW_DIR.parent.parent)}  "
          f"({elapsed:.0f}s)")
    report(df)


if __name__ == "__main__":
    main()
