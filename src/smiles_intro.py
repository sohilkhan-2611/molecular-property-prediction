"""Phase 2: SMILES and RDKit fundamentals.

A guided tour of what a SMILES string encodes and what RDKit hands back.
Everything demonstrated here gets used in Phases 3 to 12.

Run:  python src/smiles_intro.py
"""

from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, rdMolDescriptors

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------
# 1. What a SMILES string actually encodes
# ----------------------------------------------------------------------
def section_parsing():
    banner("1. PARSING: SMILES text -> molecule object")

    examples = [
        ("CCO", "ethanol", "3 atoms in a chain: C-C-O"),
        ("CC(C)C", "isobutane", "parentheses = a branch"),
        ("C1CCCCC1", "cyclohexane", "matching digits = ring closure"),
        ("c1ccccc1", "benzene", "lowercase = aromatic ring"),
        ("CC(=O)O", "acetic acid", "= is a double bond"),
        ("C#N", "hydrogen cyanide", "# is a triple bond"),
        ("[Na+].[Cl-]", "salt", "a dot = two separate fragments"),
    ]

    print(f"{'smiles':<14} {'name':<18} {'heavy':>5} {'H':>4} {'bonds':>6}  note")
    print("-" * 78)
    for smiles, name, note in examples:
        mol = Chem.MolFromSmiles(smiles)
        heavy = mol.GetNumAtoms()                 # hydrogens are implicit
        with_h = Chem.AddHs(mol).GetNumAtoms()    # make them explicit
        bonds = mol.GetNumBonds()
        print(f"{smiles:<14} {name:<18} {heavy:>5} {with_h - heavy:>4} {bonds:>6}  {note}")

    print()
    print("Takeaway: RDKit counts HEAVY atoms (non-hydrogen) by default.")
    print("It works the hydrogens out from valence rules, so you never type them.")


# ----------------------------------------------------------------------
# 2. Canonicalisation
# ----------------------------------------------------------------------
def section_canonical():
    banner("2. CANONICALISATION: many strings, one molecule")

    groups = [
        ("ethanol", ["CCO", "OCC", "C(C)O", "[CH3][CH2][OH]"]),
        ("benzene", ["c1ccccc1", "C1=CC=CC=C1", "[cH]1[cH][cH][cH][cH][cH]1"]),
        ("acetic acid", ["CC(=O)O", "OC(C)=O", "OC(=O)C"]),
    ]

    for name, variants in groups:
        print(f"\n{name}:")
        canonicals = set()
        for smiles in variants:
            mol = Chem.MolFromSmiles(smiles)
            canon = Chem.MolToSmiles(mol)
            canonicals.add(canon)
            print(f"  {smiles:<22} -> {canon}")
        status = "ALL IDENTICAL" if len(canonicals) == 1 else f"{len(canonicals)} DIFFERENT"
        print(f"  => {status}")

    print()
    print("Why this matters: PubChem will hand us the same molecule written")
    print("different ways. If we dedupe on the raw string we keep duplicates,")
    print("and the same molecule lands in both train and test. That is data")
    print("leakage. We canonicalise first, in Phase 4.")


# ----------------------------------------------------------------------
# 3. Invalid input
# ----------------------------------------------------------------------
def section_invalid():
    banner("3. INVALID SMILES: RDKit returns None, it does not raise")

    RDLogger.DisableLog("rdApp.*")
    cases = [
        ("CCO", "valid"),
        ("CCQ", "Q is not an element"),
        ("C1CCC", "ring opened, never closed"),
        ("c1cccc1", "5 atoms written as a 6-ring, cannot be aromatic"),
        ("N(C)(C)(C)(C)C", "nitrogen with 5 bonds breaks valence"),
        ("FC(F)(F)(F)F", "carbon with 5 bonds breaks valence"),
        ("CCO)", "unbalanced parenthesis"),
        ("", "empty string: WATCH OUT, see below"),
    ]
    for smiles, why in cases:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            verdict = "None"
        else:
            verdict = f"mol({mol.GetNumAtoms()})"
        print(f"  {smiles!r:<18} -> {verdict:<9}  ({why})")
    RDLogger.EnableLog("rdApp.*")

    print()
    print("Two separate failure modes, and only one of them is None:")
    print()
    print("  1. Unparseable    -> MolFromSmiles returns None.")
    print("     Guard with:    if mol is None: skip")
    print()
    print("  2. Parses to an EMPTY molecule. An empty string gives you a real")
    print("     Mol object with zero atoms. It passes an `is None` check and")
    print("     then quietly produces an all-zero fingerprint later.")
    print("     Guard with:    if mol is None or mol.GetNumAtoms() == 0: skip")
    print()
    print("Real PubChem downloads contain both kinds. Phase 4 filters on both.")


# ----------------------------------------------------------------------
# 4. Descriptors
# ----------------------------------------------------------------------
def section_descriptors():
    banner("4. DESCRIPTORS: numbers computed from structure")

    drugs = [
        ("CCO", "ethanol"),
        ("c1ccccc1", "benzene"),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
        ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "caffeine"),
        ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "ibuprofen"),
        ("CC(=O)NC1=CC=C(O)C=C1", "paracetamol"),
    ]

    print(f"{'name':<13} {'MW':>7} {'logP':>7} {'TPSA':>7} {'HBD':>4} {'HBA':>4} {'rot':>4} {'ring':>5}")
    print("-" * 62)
    for smiles, name in drugs:
        mol = Chem.MolFromSmiles(smiles)
        print(
            f"{name:<13} "
            f"{Descriptors.MolWt(mol):>7.2f} "
            f"{Crippen.MolLogP(mol):>7.3f} "
            f"{Descriptors.TPSA(mol):>7.2f} "
            f"{Lipinski.NumHDonors(mol):>4} "
            f"{Lipinski.NumHAcceptors(mol):>4} "
            f"{Lipinski.NumRotatableBonds(mol):>4} "
            f"{rdMolDescriptors.CalcNumRings(mol):>5}"
        )

    print()
    print("Glossary:")
    print("  MW    molecular weight in g/mol")
    print("  logP  how much the molecule prefers oil over water. Our TARGET.")
    print("        Positive = greasy/lipophilic. Negative = water-loving.")
    print("  TPSA  topological polar surface area, the area made up of polar")
    print("        atoms (N, O). Predicts membrane crossing.")
    print("  HBD   hydrogen bond donors (roughly: N-H and O-H groups)")
    print("  HBA   hydrogen bond acceptors (roughly: N and O atoms)")
    print("  rot   rotatable bonds, a proxy for molecular floppiness")
    print()
    print(f"RDKit ships {len(Descriptors._descList)} descriptors in total.")
    print("In Phase 6 we compute a chosen subset as model features.")


# ----------------------------------------------------------------------
# 5. A molecule is a graph
# ----------------------------------------------------------------------
def section_graph():
    banner("5. A MOLECULE IS A GRAPH (this is what the GNN eats in Phase 10)")

    smiles = "CC(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    print(f"molecule: {smiles} (acetic acid)\n")

    print("NODES (atoms):")
    print(f"  {'idx':>3} {'sym':>4} {'degree':>7} {'formalQ':>8} {'aromatic':>9} {'implicitH':>10}")
    for atom in mol.GetAtoms():
        print(
            f"  {atom.GetIdx():>3} {atom.GetSymbol():>4} {atom.GetDegree():>7} "
            f"{atom.GetFormalCharge():>8} {str(atom.GetIsAromatic()):>9} "
            f"{atom.GetTotalNumHs():>10}"
        )

    print("\nEDGES (bonds):")
    print(f"  {'begin':>6} {'end':>4} {'type':>9} {'inRing':>7}")
    for bond in mol.GetBonds():
        print(
            f"  {bond.GetBeginAtomIdx():>6} {bond.GetEndAtomIdx():>4} "
            f"{str(bond.GetBondType()):>9} {str(bond.IsInRing()):>7}"
        )

    print("\nADJACENCY MATRIX:")
    adj = Chem.GetAdjacencyMatrix(mol)
    print("     " + " ".join(f"{i:>2}" for i in range(len(adj))))
    for i, row in enumerate(adj):
        print(f"  {i:>2} " + " ".join(f"{v:>2}" for v in row))

    print()
    print("Atom features -> node vectors. Bonds -> edge index. That pair is")
    print("exactly what PyTorch Geometric wants. Phase 10 is this conversion,")
    print("done properly and in bulk.")


# ----------------------------------------------------------------------
# 6. Draw something
# ----------------------------------------------------------------------
def section_draw():
    banner("6. DRAWING: sanity-check a structure with your eyes")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "phase2_molecules.svg"

    names = ["aspirin", "caffeine", "ibuprofen", "paracetamol"]
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "CC(=O)NC1=CC=C(O)C=C1",
    ]
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    svg = Draw.MolsToGridImage(
        mols, molsPerRow=2, subImgSize=(300, 250), legends=names, useSVG=True
    )
    svg_text = svg.data if hasattr(svg, "data") else str(svg)
    out.write_text(svg_text)

    print(f"  wrote {out.relative_to(RESULTS_DIR.parent)}  ({len(svg_text)} bytes)")
    print("  open it in a browser to see the four structures.")


def main():
    section_parsing()
    section_canonical()
    section_invalid()
    section_descriptors()
    section_graph()
    section_draw()

    banner("PHASE 2 COMPLETE. Ready for Phase 3 (PubChem data pull).")


if __name__ == "__main__":
    main()
