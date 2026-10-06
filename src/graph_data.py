"""Phase 10: SMILES -> molecular graph (the input format for GNNs in Phases 11 and 12).

Run:  python src/graph_data.py            builds data/processed/graphs.pt and prints a report
      python src/graph_data.py "CCO"      also prints the graph of one molecule

Reads   data/processed/clean.csv
Writes  data/processed/graphs.pt          (git-ignored)

A molecule becomes a graph:
  node  = one heavy atom (every atom except hydrogen)         -> x          [n_atoms, 33]
  edge  = one bond, stored in BOTH directions                 -> edge_index [2, 2 * n_bonds]
  edge attributes = what kind of bond                         -> edge_attr  [2 * n_bonds, 6]
  label = PubChem XLogP3                                      -> y          [1]

Other scripts reuse:
  smiles_to_arrays(smiles)   numpy only, works without torch (used for testing and predict.py)
  smiles_to_graph(smiles)    torch_geometric Data object
  load_graphs()              (list of Data, cid array), row-aligned with clean.csv
  NODE_DIM, EDGE_DIM

Input SMILES must already be standardized (data_processing.standardize).
Hydrogens are NOT nodes. Each atom instead carries its hydrogen count as a feature.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "processed" / "clean.csv"
GRAPH_PATH = ROOT / "data" / "processed" / "graphs.pt"

# Same element whitelist that data_processing.py enforces (H never appears as a node).
ELEMENTS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
HYBRIDIZATIONS = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
                  rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
                  rdchem.HybridizationType.SP3D2]
BOND_TYPES = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
              rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]


def one_hot(value, choices, overflow=True):
    """1.0 at the position of `value`; if it is not in `choices` the extra last slot lights up."""
    vec = [0.0] * (len(choices) + (1 if overflow else 0))
    vec[choices.index(value) if value in choices else -1] = 1.0
    return vec


def atom_features(atom):
    return (
        one_hot(atom.GetSymbol(), ELEMENTS)                              # 10: which element
        + one_hot(min(atom.GetDegree(), 5), [0, 1, 2, 3, 4], True)        # 6 : number of heavy neighbours (5 = "5 or more")
        + one_hot(atom.GetFormalCharge(), [-1, 0, 1])                     # 4 : charge
        + one_hot(min(atom.GetTotalNumHs(), 4), [0, 1, 2, 3], True)       # 5 : attached hydrogens (4 = "4 or more")
        + one_hot(atom.GetHybridization(), HYBRIDIZATIONS)                # 6 : sp, sp2, sp3 ... shape around the atom
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]           # 2
    )


def bond_features(bond):
    return (
        one_hot(bond.GetBondType(), BOND_TYPES, overflow=False)           # 4 : single / double / triple / aromatic
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]         # 2
    )


NODE_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
EDGE_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def smiles_to_arrays(smiles):
    """SMILES -> (x, edge_index, edge_attr) as plain numpy arrays. No torch needed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"cannot parse SMILES: {smiles!r}")
    x = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    src, dst, attr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = bond_features(bond)
        src += [i, j]           # a bond is undirected, so we store it as two arrows: i->j and j->i
        dst += [j, i]
        attr += [f, f]
    edge_index = np.array([src, dst], dtype=np.int64).reshape(2, -1)
    edge_attr = np.array(attr, dtype=np.float32).reshape(-1, EDGE_DIM)
    return x, edge_index, edge_attr


def smiles_to_graph(smiles, y=None):
    import torch
    from torch_geometric.data import Data

    x, edge_index, edge_attr = smiles_to_arrays(smiles)
    data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_attr))
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    data.smiles = smiles
    return data


def build_graphs(clean):
    return [smiles_to_graph(s, y) for s, y in zip(clean["smiles"], clean["xlogp"])]


def load_graphs():
    """(list of Data, cid array). Row i is molecule i of clean.csv."""
    import torch

    if not GRAPH_PATH.exists():
        sys.exit(f"{GRAPH_PATH.name} not found. Run: python src/graph_data.py")
    blob = torch.load(GRAPH_PATH, weights_only=False)
    return blob["graphs"], blob["cid"]


# ----------------------------------------------------------------------
# Checks and report
# ----------------------------------------------------------------------
def self_checks(clean, graphs):
    assert len(graphs) == len(clean), "one graph per molecule"
    for i, (g, smiles) in enumerate(zip(graphs, clean["smiles"])):
        mol = Chem.MolFromSmiles(smiles)
        assert g.num_nodes == mol.GetNumAtoms(), f"row {i}: node count != atom count"
        assert g.edge_index.shape[1] == 2 * mol.GetNumBonds(), f"row {i}: edge count != 2 x bonds"
        assert g.x.shape[1] == NODE_DIM and g.edge_attr.shape[1] == EDGE_DIM
        if g.edge_index.numel():
            assert int(g.edge_index.max()) < g.num_nodes and int(g.edge_index.min()) >= 0, f"row {i}: bad index"
        assert not bool(g.x.isnan().any()), f"row {i}: NaN feature"
        assert float(g.x[:, :len(ELEMENTS) + 1].sum()) == g.num_nodes, f"row {i}: element one-hot broken"
        fwd = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
        assert all((b, a) in fwd for a, b in fwd), f"row {i}: an edge has no reverse arrow"
    print("self-checks passed")


def report(clean, graphs):
    import torch

    n = np.array([g.num_nodes for g in graphs])
    e = np.array([g.edge_index.shape[1] // 2 for g in graphs])
    line = "=" * 78
    print(f"\n{line}\nMOLECULAR GRAPHS\n{line}")
    print(f"graphs built            : {len(graphs)}  (rows match clean.csv: {len(graphs) == len(clean)})")
    print(f"node features per atom  : {NODE_DIM}      edge features per bond: {EDGE_DIM}")
    print(f"atoms per molecule      : min {n.min()}  median {int(np.median(n))}  max {n.max()}")
    print(f"bonds per molecule      : min {e.min()}  median {int(np.median(e))}  max {e.max()}")
    print(f"graphs with no bonds    : {int((e == 0).sum())}   (single atoms; they still work, just no messages)")
    rings = (e - n + 1)
    print(f"rings per molecule      : median {int(np.median(rings))}  max {rings.max()}   (bonds - atoms + 1, valid for connected graphs)")

    allx = torch.cat([g.x for g in graphs])
    sym = ELEMENTS + ["other"]
    counts = allx[:, :len(sym)].sum(0).int().tolist()
    print("\natoms by element (all molecules together):")
    print("  " + "   ".join(f"{s} {c}" for s, c in zip(sym, counts)))
    alle = torch.cat([g.edge_attr for g in graphs]) if any(g.edge_attr.numel() for g in graphs) else None
    if alle is not None:
        bt = (alle[:, :4].sum(0) / 2).int().tolist()
        print("bonds by type            : " + "   ".join(f"{s} {c}" for s, c in zip(["single", "double", "triple", "aromatic"], bt)))
    if int(allx[:, 9].sum()) > 0:
        print("  WARNING: some atoms fell into 'other' element. The element whitelist and Phase 4 disagree.")

    batch_size = 64
    mem_mb = sum(g.x.numel() * 4 + g.edge_index.numel() * 8 + g.edge_attr.numel() * 4 for g in graphs) / 1e6
    print(f"\nin memory               : about {mem_mb:.0f} MB for all {len(graphs)} graphs")
    print(f"a batch of {batch_size} molecules will hold on average {int(n.mean() * batch_size)} atoms in one big disconnected graph")
    print(line)


def show_one(smiles):
    x, ei, ea = smiles_to_arrays(smiles)
    print(f"\n{smiles}: {x.shape[0]} atoms, {ei.shape[1] // 2} bonds")
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        print(f"  node {a.GetIdx():>2}  {a.GetSymbol():<2} degree {a.GetDegree()}  Hs {a.GetTotalNumHs()}  "
              f"aromatic {int(a.GetIsAromatic())}  neighbours {[n.GetIdx() for n in a.GetNeighbors()]}")
    print(f"  edge_index (arrows from row 0 to row 1):\n  {ei.tolist()}")


def main():
    import torch

    clean = pd.read_csv(CLEAN_CSV)
    graphs = build_graphs(clean)
    GRAPH_PATH.parent.mkdir(exist_ok=True)
    torch.save({"graphs": graphs, "cid": clean["cid"].values}, GRAPH_PATH)
    print(f"saved {GRAPH_PATH.relative_to(ROOT)}")
    self_checks(clean, graphs)
    report(clean, graphs)
    show_one(sys.argv[1] if len(sys.argv) > 1 else "CCO")


if __name__ == "__main__":
    main()
