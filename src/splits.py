"""Shared train/test splits, so every model in every phase is scored on the SAME molecules.

Two splits:
  random    shuffle molecules, 80% train / 20% test (seed 42).
  scaffold  group molecules by their Bemis-Murcko scaffold (the ring skeleton with
            side chains removed). Whole groups go to train or test, never both.
            The test set then holds ring skeletons the model has never seen, which
            is closer to "predict a brand-new kind of molecule".

Why two? A random split lets close cousins of a test molecule sit in train, so the
score is optimistic. The scaffold split is stricter. The gap between the two scores
tells you how much the model memorises vs. how much it generalises.

Use:
    from splits import get_splits
    splits = get_splits(clean["smiles"].tolist())
    train_idx, test_idx = splits["scaffold"]
"""

from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

TEST_FRAC = 0.2 # Fraction of molecules to include in the test set
SEED = 42


def murcko_scaffold(smiles):
    """Ring skeleton of a molecule as a SMILES string ('' if it has no rings)."""
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def random_split(n, test_frac=TEST_FRAC, seed=SEED):
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    n_test = int(round(n * test_frac))
    return np.sort(order[n_test:]), np.sort(order[:n_test])


def scaffold_split(smiles_list, test_frac=TEST_FRAC):
    """Deterministic: biggest scaffold groups fill train first, the rest become test.

    Consequence: the test set is made of the rarer scaffolds. That is the hard,
    honest version of "new chemistry". Molecules with no ring share the '' group.
    """
    groups = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        groups[murcko_scaffold(smiles)].append(i)

    n = len(smiles_list)
    n_train = n - int(round(n * test_frac))
    train, test = [], []
    for scaffold in sorted(groups, key=lambda s: (-len(groups[s]), s)):
        members = groups[scaffold]
        if len(train) + len(members) <= n_train:
            train.extend(members)
        else:
            test.extend(members)
    return np.sort(np.array(train)), np.sort(np.array(test))


def get_splits(smiles_list):
    return {
        "random": random_split(len(smiles_list)),
        "scaffold": scaffold_split(smiles_list),
    }


def self_checks(smiles_list, splits):
    n = len(smiles_list)
    for name, (tr, te) in splits.items():
        assert len(set(tr) & set(te)) == 0, f"{name}: a molecule is in both train and test"
        assert len(tr) + len(te) == n, f"{name}: some molecules are in neither"
    tr, te = splits["scaffold"]
    tr_scaf = {murcko_scaffold(smiles_list[i]) for i in tr}
    te_scaf = {murcko_scaffold(smiles_list[i]) for i in te}
    assert not (tr_scaf & te_scaf), "scaffold split: a scaffold appears on both sides"


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    clean = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "processed" / "clean.csv")
    smiles = clean["smiles"].tolist()
    splits = get_splits(smiles)
    self_checks(smiles, splits)
    for name, (tr, te) in splits.items():
        print(f"{name:<9} train {len(tr):>5}   test {len(te):>5}   "
              f"target mean train {clean['xlogp'].iloc[tr].mean():+.2f} / test {clean['xlogp'].iloc[te].mean():+.2f}")
    tr, te = splits["scaffold"]
    n_scaf = {murcko_scaffold(smiles[i]) for i in range(len(smiles))}
    acyclic = sum(1 for s in smiles if murcko_scaffold(s) == "")
    print(f"distinct scaffolds: {len(n_scaf)}   molecules with no ring: {acyclic}")
    print("self-checks passed")
