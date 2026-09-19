"""Phase 1 sanity check: prove the environment can do cheminformatics.

Run:  python src/check_env.py
"""

import sys


def check_imports():
    """Import every library the project needs and print its version."""
    results = {}
    libs = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("rdkit", "rdkit"),
        ("matplotlib", "matplotlib"),
        ("requests", "requests"),
    ]
    for module_name, display_name in libs:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            results[display_name] = version
        except ImportError as exc:
            results[display_name] = f"MISSING ({exc})"
    return results


def check_rdkit():
    """Parse a few SMILES and compute basic properties."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors

    test_molecules = [
        ("CCO", "ethanol"),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
        ("c1ccccc1", "benzene"),
    ]

    rows = []
    for smiles, name in test_molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rows.append((name, smiles, None, None, None))
            continue
        rows.append(
            (
                name,
                smiles,
                mol.GetNumAtoms(),                 # heavy atoms only, H is implicit
                round(Descriptors.MolWt(mol), 2),  # molecular weight, g/mol
                round(Crippen.MolLogP(mol), 4),    # RDKit's own computed logP
            )
        )
    return rows


def check_bad_smiles():
    """RDKit should return None for garbage, not crash."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")  # silence the expected parse warnings
    bad = Chem.MolFromSmiles("this-is-not-a-molecule")
    RDLogger.EnableLog("rdApp.*")
    return bad is None


def main():
    print("=" * 62)
    print("PHASE 1 ENVIRONMENT CHECK")
    print("=" * 62)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print("-" * 62)

    print("LIBRARIES")
    versions = check_imports()
    missing = []
    for name, version in versions.items():
        flag = "ok " if not version.startswith("MISSING") else "FAIL"
        print(f"  [{flag}] {name:<16} {version}")
        if version.startswith("MISSING"):
            missing.append(name)

    if missing:
        print("-" * 62)
        print(f"STOP: missing libraries -> {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    print("-" * 62)
    print("RDKIT SMILES PARSING")
    print(f"  {'name':<10} {'smiles':<26} {'atoms':>5} {'MW':>8} {'logP':>9}")
    for name, smiles, atoms, mw, logp in check_rdkit():
        if atoms is None:
            print(f"  {name:<10} {smiles:<26}  PARSE FAILED")
            sys.exit(1)
        print(f"  {name:<10} {smiles:<26} {atoms:>5} {mw:>8} {logp:>9}")

    print("-" * 62)
    print("ERROR HANDLING")
    if check_bad_smiles():
        print("  [ok ] invalid SMILES returns None instead of crashing")
    else:
        print("  [FAIL] invalid SMILES did not return None")
        sys.exit(1)

    print("=" * 62)
    print("ENVIRONMENT OK. Ready for Phase 2.")
    print("=" * 62)


if __name__ == "__main__":
    main()