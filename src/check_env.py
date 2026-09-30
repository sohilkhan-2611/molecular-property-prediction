"""Phase 1 sanity check: prove the environment can do cheminformatics.

Run:  python src/check_env.py
"""

import sys


def _first_meaningful_line(exc):
    """Some libraries raise multi-line errors. Grab the first useful line."""
    for line in str(exc).splitlines():
        line = line.strip()
        if line:
            return line[:110]
    return type(exc).__name__


def check_imports():
    """Import every library the project needs and report its version.

    Catches Exception, not just ImportError: some libraries import fine but
    fail at load time on a missing system library (xgboost needs libomp on
    macOS). We want that reported as a row, not as a traceback.
    """
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
            results[display_name] = getattr(mod, "__version__", "unknown")
        except ImportError as exc:
            results[display_name] = f"MISSING ({_first_meaningful_line(exc)})"
        except Exception as exc:
            results[display_name] = (
                f"MISSING [{type(exc).__name__}] {_first_meaningful_line(exc)}"
            )
    return results


def check_rdkit():
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
        rows.append((name, smiles, mol.GetNumAtoms(),
                     round(Descriptors.MolWt(mol), 2),
                     round(Crippen.MolLogP(mol), 4)))
    return rows


def check_bad_smiles():
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
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
    broken = []
    for name, version in versions.items():
        ok = not version.startswith("MISSING")
        print(f"  [{'ok ' if ok else 'FAIL'}] {name:<16} {version}")
        if not ok:
            broken.append(name)

    if broken:
        print("-" * 62)
        print(f"STOP: these libraries did not load -> {', '.join(broken)}")
        print("  not installed      -> pip install -r requirements.txt")
        print("  xgboost + libomp   -> brew install libomp   (macOS only)")
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
