"""Phase 14: SMILES in, predicted XLogP3 out.

Run:  python predict.py "CCO"
      python predict.py "CCO" "c1ccccc1" "CC(=O)Oc1ccccc1C(=O)O"
      python predict.py --file my_smiles.txt          (one SMILES per line)
      python predict.py "CCO" --quiet                 (prints only the number, handy in scripts)

Needs   artifacts/final_model.pt        create it once with:  python src/train_final.py

What happens to your SMILES
  1. standardize() from data_processing.py, the SAME cleaning the training data went through
     (one molecule only, allowed elements, weight cap, canonical form, no stereo).
  2. smiles_to_graph() from graph_data.py turns the clean molecule into atoms and bonds.
  3. Every network in the saved ensemble predicts, and the answer is their average.

Read the output honestly
  * The number is PubChem's COMPUTED XLogP3 as the model imitates it. It is NOT a lab-measured logP.
  * "spread" is how much the networks disagree. A big spread means "do not trust this one".
  * The typical error printed at the end is the test MAE measured in Phase 13, not a promise for
    this particular molecule.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
CHECKPOINT = ROOT / "artifacts" / "final_model.pt"

REJECTION_TEXT = {
    "invalid_smiles": "that is not a valid SMILES string",
    "multi_component": "it has several separate pieces (a salt or a mixture), give one molecule",
    "bad_element": "it has an element outside C H N O F P S Cl Br I, which the model never saw",
    "too_heavy": "it is heavier than 1000 g/mol, outside the range the model was trained on",
    "no_heavy_atoms": "it has no heavy atoms (only hydrogens)",
}
SPREAD_WARNING = 0.30       # networks disagreeing by more than this (XLogP3 units) is worth a warning


def load_checkpoint(path=CHECKPOINT):
    """Rebuild every saved network. Returns (checkpoint dict, list of (network, y_mean, y_std))."""
    import torch
    from graph_data import EDGE_DIM, NODE_DIM
    from models.factory import build_model

    if not Path(path).exists():
        sys.exit(f"No trained model at {Path(path).relative_to(ROOT)}.\nTrain it once with:  python src/train_final.py")
    ck = torch.load(path, weights_only=True)
    if ck["node_dim"] != NODE_DIM or ck["edge_dim"] != EDGE_DIM:
        sys.exit("The saved model was built with different atom/bond features than graph_data.py now produces. "
                 "Retrain with:  python src/train_final.py")
    nets = []
    for member in ck["members"]:
        net = build_model(ck["model"], ck["node_dim"], ck["edge_dim"], **ck["hparams"])
        net.load_state_dict(member["state"])
        net.eval()
        nets.append((net, member["y_mean"], member["y_std"]))
    return ck, nets


def predict_smiles(smiles, nets, ck=None):
    """One SMILES -> dict. ok=False carries a plain-English reason. Never raises on bad chemistry."""
    import numpy as np
    import torch
    from data_processing import standardize
    from graph_data import smiles_to_graph
    from torch_geometric.data import Batch

    std = standardize(smiles)
    reason = std["reason"]
    if reason is None and std["n_heavy"] == 0:
        reason = "no_heavy_atoms"
    if reason is not None:
        return {"ok": False, "input": smiles, "reason": reason, "message": REJECTION_TEXT[reason]}

    batch = Batch.from_data_list([smiles_to_graph(std["smiles"])])
    with torch.no_grad():
        preds = np.array([float(net(batch).item()) * y_std + y_mean for net, y_mean, y_std in nets])
    spread = float(preds.std(ddof=1)) if len(preds) > 1 else 0.0

    notes = []
    dom = (ck or {}).get("domain")
    if dom:
        if std["n_heavy"] < dom["n_heavy_p1"]:
            notes.append(f"very small molecule ({std['n_heavy']} heavy atom{'s' if std['n_heavy'] != 1 else ''}): only {dom['n_small']} of "
                         f"{dom['n_train']} training molecules have 8 or fewer, so treat the number loosely")
        elif std["n_heavy"] > dom["n_heavy_p99"]:
            notes.append(f"larger than 99% of the training molecules ({std['n_heavy']} heavy atoms), "
                         f"treat the number loosely")
        if std["net_charge"] != 0:
            notes.append(f"charged molecule (net charge {std['net_charge']:+d}): only {dom['n_charged']} of "
                         f"{dom['n_train']} training molecules are charged")
    if spread > SPREAD_WARNING:
        notes.append(f"the {len(preds)} networks disagree by {spread:.2f}, so this prediction is shaky")
    return {"ok": True, "input": smiles, "canonical": std["smiles"], "prediction": float(preds.mean()),
            "spread": spread, "n_heavy": std["n_heavy"], "notes": notes}


def score_many(smiles_list, nets, ck=None, on_progress=None):
    """Many SMILES -> a pandas table, one row each. Bad molecules get status "rejected" and a reason."""
    import pandas as pd

    rows, n = [], len(smiles_list)
    for i, s in enumerate(smiles_list):
        r = predict_smiles(s, nets, ck)
        if r["ok"]:
            rows.append({"input": s, "standardised": r["canonical"],
                         "predicted_XLogP3": round(r["prediction"], 3), "spread": round(r["spread"], 3),
                         "heavy_atoms": r["n_heavy"], "status": "ok", "notes": "; ".join(r["notes"])})
        else:
            rows.append({"input": s, "standardised": None, "predicted_XLogP3": None, "spread": None,
                         "heavy_atoms": None, "status": "rejected", "notes": r["message"]})
        if on_progress and (i % 25 == 0 or i == n - 1):
            on_progress((i + 1) / n)
    return pd.DataFrame(rows)


def typical_error_line(ck):
    b = (ck or {}).get("benchmark") or {}
    if "scaffold" in b and "random" in b:
        return (f"typical error of this model (Phase 13 test MAE): about {b['scaffold']:.2f} on new chemical "
                f"skeletons, {b['random']:.2f} on familiar ones")
    return None


def main():
    p = argparse.ArgumentParser(description="Predict PubChem XLogP3 from a SMILES string.")
    p.add_argument("smiles", nargs="*", help="one or more SMILES strings")
    p.add_argument("--file", help="text file with one SMILES per line")
    p.add_argument("-q", "--quiet", action="store_true", help="print only the predicted number(s)")
    args = p.parse_args()

    todo = list(args.smiles)
    if args.file:
        todo += [ln.strip() for ln in Path(args.file).read_text().splitlines() if ln.strip()]
    if not todo:
        p.error('give at least one SMILES, for example:  python predict.py "CCO"')

    ck, nets = load_checkpoint()
    failed = 0
    results = [predict_smiles(s, nets, ck) for s in todo]
    for r in results:
        failed += not r["ok"]
        if args.quiet:
            if r["ok"]:
                print(f"{r['prediction']:.2f}")
            else:
                print("nan")
                print(f"{r['input']!r}: {r['message']}", file=sys.stderr)
            continue
        if not r["ok"]:
            print(f"{r['input'] or '(empty input)'}\n  could not predict: {r['message']}\n")
            continue
        same = "" if r["canonical"] == r["input"] else f"   (standardised to {r['canonical']})"
        print(f"SMILES            {r['input']}{same}")
        print(f"predicted XLogP3  {r['prediction']:.2f}")
        if len(nets) > 1:
            print(f"  spread across the {len(nets)} networks: {r['spread']:.2f}")
        for note in r["notes"]:
            print(f"  note: {note}")
        print()
    if not args.quiet:
        line = typical_error_line(ck)
        print("This is PubChem's computed XLogP3 as the model imitates it, not a lab-measured logP.")
        if line:
            print(line + ".")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
