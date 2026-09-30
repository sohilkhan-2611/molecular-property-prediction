"""Phase 5: Morgan fingerprints (SMILES -> fixed-length numeric vector).

Run:  python src/features.py
      python src/features.py --radius 3 --n-bits 4096
      python src/features.py --binary          (on/off flags instead of counts)

Reads   data/processed/clean.csv
Writes  data/processed/morgan_r{radius}_{n_bits}.npz   (X and cid, git-ignored)

Other scripts reuse:
  featurize(smiles_list)   list of SMILES -> matrix, one row per molecule
  load_morgan()            (X, cid) from disk, row-aligned with clean.csv
  environment_smiles(...)  turn a fingerprint slot back into a substructure

Input SMILES must already be standardized (data_processing.standardize).
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "processed" / "clean.csv"

RADIUS = 2      # look this many bonds out from every atom ("ECFP4")
N_BITS = 2048   # length of the vector


def feature_path(radius=RADIUS, n_bits=N_BITS, binary=False):
    tag = "bits" if binary else "counts"
    return ROOT / "data" / "processed" / f"morgan_r{radius}_{n_bits}_{tag}.npz"


# ----------------------------------------------------------------------
# Core: SMILES -> vector
# ----------------------------------------------------------------------
def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"cannot parse SMILES: {smiles!r}")
    return mol


def featurize(smiles_list, radius=RADIUS, n_bits=N_BITS, binary=False):
    """List of standardized SMILES -> uint16 matrix of shape (n, n_bits).

    counts: slot value = how many times the substructure occurs in the molecule
    binary: slot value = 1 if it occurs at all, else 0
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles_list), n_bits), dtype=np.uint16)
    cap = np.iinfo(np.uint16).max
    for i, smiles in enumerate(smiles_list):
        counts = gen.GetCountFingerprintAsNumPy(smiles_to_mol(smiles))
        X[i] = np.minimum(counts, cap)
    if binary:
        X = (X > 0).astype(np.uint16)
    return X


def load_morgan(radius=RADIUS, n_bits=N_BITS, binary=False):
    """Return (X, cid). X rows line up with the rows of clean.csv."""
    path = feature_path(radius, n_bits, binary)
    if not path.exists():
        sys.exit(f"{path.name} not found. Run: python src/features.py")
    data = np.load(path)
    return data["X"], data["cid"]


# ----------------------------------------------------------------------
# Looking inside the fingerprint
# ----------------------------------------------------------------------
def environment_smiles(mol, atom_idx, radius):
    """The little piece of molecule that one (atom, radius) pair describes,
    written as a SMILES fragment rooted at that atom."""
    if radius == 0:
        atoms, bonds = {atom_idx}, []
    else:
        bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx))
        atoms = {atom_idx}
        for b in bonds:
            bond = mol.GetBondWithIdx(b)
            atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), bondsToUse=bonds,
                                    rootedAtAtom=atom_idx, canonical=True)


def slot_contents(smiles_list, slots, radius=RADIUS, n_bits=N_BITS, limit=1500):
    """For each slot in `slots`, which substructures land there (from the first
    `limit` molecules)? Returns {slot: Counter of fragment SMILES}."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    wanted = set(slots)
    found = {s: Counter() for s in slots}
    for smiles in smiles_list[:limit]:
        mol = smiles_to_mol(smiles)
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetCountFingerprint(mol, additionalOutput=ao)
        for slot, envs in ao.GetBitInfoMap().items():
            if slot in wanted:
                for atom_idx, rad in envs:
                    found[slot][environment_smiles(mol, atom_idx, rad)] += 1
    return found


def count_tanimoto(a, b):
    """Similarity of two count vectors: 1 = identical, 0 = nothing in common."""
    a, b = a.astype(np.int64), b.astype(np.int64)
    denom = np.maximum(a, b).sum()
    return float(np.minimum(a, b).sum() / denom) if denom else 0.0


def nearest_neighbour_similarity(X, chunk=500):
    """For every molecule, the on/off Tanimoto similarity to its closest OTHER
    molecule in the dataset."""
    B = (X > 0).astype(np.float32)
    size = B.sum(axis=1)
    best = np.zeros(len(B), dtype=np.float32)
    for start in range(0, len(B), chunk):
        block = B[start:start + chunk]
        inter = block @ B.T
        union = size[start:start + chunk, None] + size[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        for k in range(block.shape[0]):
            sim[k, start + k] = -1.0          # ignore the molecule itself
        best[start:start + block.shape[0]] = sim.max(axis=1)
    return best


def unique_environments(smiles_list, radius=RADIUS, common_min=5):
    """How many different substructures exist in the data, before folding them
    into a fixed number of slots? Returns (all, common), where common means
    "appears in at least common_min different molecules"."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    in_molecules = Counter()
    for smiles in smiles_list:
        in_molecules.update(gen.GetSparseCountFingerprint(smiles_to_mol(smiles)).GetNonzeroElements())
    common = sum(1 for n in in_molecules.values() if n >= common_min)
    return len(in_molecules), common


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def report(df, X, radius, n_bits, binary, seconds):
    smiles = df["smiles"].tolist()
    kind = "on/off flags" if binary else "counts"
    print("=" * 74)
    print(f"MORGAN FINGERPRINTS  (radius {radius}, {n_bits} slots, {kind})")
    print("=" * 74)
    print(f"matrix shape        : {X.shape}   dtype {X.dtype}   {X.nbytes / 1e6:.0f} MB")
    print(f"built in            : {seconds:.1f} s")
    print(f"rows match clean.csv: {X.shape[0] == len(df)}")

    active = (X > 0).sum(axis=1)
    print("\n-- how full is each molecule's vector? --")
    print(f"non-zero slots per molecule: min {active.min()}, median {int(np.median(active))}, "
          f"max {active.max()}  (of {n_bits})")
    print(f"molecules with an all-zero vector : {(active == 0).sum()}   (must be 0)")
    used = ((X > 0).sum(axis=0) > 0).sum()
    print(f"slots used by at least one molecule: {used} of {n_bits}")
    if not binary:
        print(f"largest single count anywhere      : {X.max()}")

    n_env, n_common = unique_environments(smiles, radius)
    print("\n-- collisions (different substructures forced into the same slot) --")
    print(f"different substructures in this dataset  : {n_env}")
    print(f"  of which seen in 5 or more molecules   : {n_common}")
    print(f"slots available                          : {n_bits}")
    print(f"=> on average {n_env / n_bits:.1f} substructures share each slot "
          f"({n_common / n_bits:.1f} if we count only the common ones)")

    B = (X > 0)
    rows_as_bytes = [r.tobytes() for r in B]
    dup_rows = len(rows_as_bytes) - len(set(rows_as_bytes))
    print(f"\nmolecules whose vector is identical to another molecule's: {dup_rows}")

    print("\n-- does similarity make chemical sense? (count Tanimoto, 1 = identical) --")
    pairs = [("CCO", "CCCO", "ethanol vs propanol"),
             ("CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)NC1=CC=C(O)C=C1", "aspirin vs paracetamol"),
             ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "CC(=O)OC1=CC=CC=C1C(=O)O", "ibuprofen vs aspirin"),
             ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "CCO", "ibuprofen vs ethanol"),
             ("c1ccccc1", "CCCCCCCC", "benzene vs octane")]
    for a, b, label in pairs:
        fa, fb = featurize([a], radius, n_bits)[0], featurize([b], radius, n_bits)[0]
        print(f"  {label:<26} {count_tanimoto(fa, fb):.2f}")
    same = [featurize([s], radius, n_bits)[0] for s in ("CCO", "OCC")]
    print(f"  'CCO' vs 'OCC' (same molecule, different spelling): identical = "
          f"{np.array_equal(same[0], same[1])}")

    print("\n-- nearest neighbour: how close is each molecule to its closest twin? --")
    nn = nearest_neighbour_similarity(X)
    q = np.percentile(nn, [5, 25, 50, 75, 95])
    print(f"on/off Tanimoto to nearest other molecule: 5% {q[0]:.2f}  25% {q[1]:.2f}  "
          f"median {q[2]:.2f}  75% {q[3]:.2f}  95% {q[4]:.2f}")
    print(f"molecules with a near-twin (similarity >= 0.7): {(nn >= 0.7).sum()} "
          f"({(nn >= 0.7).mean():.1%})")

    print("\n-- what does the busiest slot mean? --")
    freq = (X > 0).sum(axis=0)
    top = np.argsort(freq)[::-1][:6]
    found = slot_contents(smiles, [int(s) for s in top], radius, n_bits)
    for s in top:
        s = int(s)
        parts = ", ".join(f"{frag} (x{n})" for frag, n in found[s].most_common(2)) or "n/a"
        print(f"  slot {s:>4}  in {freq[s] / len(df):.0%} of molecules   e.g. {parts}")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=int, default=RADIUS)
    ap.add_argument("--n-bits", type=int, default=N_BITS)
    ap.add_argument("--binary", action="store_true", help="on/off flags instead of counts")
    args = ap.parse_args()

    if not CLEAN_CSV.exists():
        sys.exit(f"{CLEAN_CSV} not found. Run: python src/data_processing.py")
    df = pd.read_csv(CLEAN_CSV)

    start = time.time()
    X = featurize(df["smiles"].tolist(), args.radius, args.n_bits, args.binary)
    seconds = time.time() - start

    out = feature_path(args.radius, args.n_bits, args.binary)
    np.savez_compressed(out, X=X, cid=df["cid"].to_numpy())
    print(f"saved {out.relative_to(ROOT)}\n")

    report(df, X, args.radius, args.n_bits, args.binary, seconds)


if __name__ == "__main__":
    main()
