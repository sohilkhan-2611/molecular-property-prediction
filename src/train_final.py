"""Phase 14, part 1: train the model that predict.py will use, and save it.

Run:  python src/train_final.py                 picks the best GNN from your Phase 13 table (about 10 to 20 min)
      python src/train_final.py --model gcn     force one (gcn, mpnn, mpnn_nobond)
      python src/train_final.py --quick         1 seed, 15 epochs: checks the plumbing, NOT a usable model

Reads   data/processed/clean.csv, graphs.pt, results/phase13_benchmark.csv
Writes  artifacts/final_model.pt        the networks, their target scaling, and a few facts predict.py prints

Why this differs from the benchmark runs
  * The benchmark models never saw their test molecules. That is what made the scores honest.
  * The deployed model trains on ALL 5753 molecules (minus a 10% validation slice that picks the best
    epoch), because more chemistry means better predictions on new molecules.
  * That means this final model has no test set of its own. The honest error numbers stay the Phase 13
    ones. Do not quote a score for this file from its validation MAE.
  * Choosing the architecture: the lowest SCAFFOLD test MAE among the GNN rows of Phase 13, because
    a user will mostly type molecules the model has not seen before. If you have not run the MPNN
    yet, only the GCN is in that table, so the GCN is used.
  * 3 networks with different seeds are saved and averaged. Averaging helped in Phase 11/13.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                    # so we can import predict.py from the repo root

from graph_data import EDGE_DIM, NODE_DIM, load_graphs
from models.factory import MODEL_KEYS, build_model
from splits import SEED
from train_gnn import sanity_overfit, train_one

CLEAN_CSV = ROOT / "data" / "processed" / "clean.csv"
BENCH_CSV = ROOT / "results" / "phase13_benchmark.csv"
OUT = ROOT / "artifacts" / "final_model.pt"
LABEL = {"gcn": "GCN", "mpnn": "MPNN", "mpnn_nobond": "MPNN, no bond features"}


def pick_model():
    """Best GNN by scaffold test MAE in the Phase 13 table. Falls back to the GCN."""
    if not BENCH_CSV.exists():
        print("results/phase13_benchmark.csv not found, using gcn (run src/benchmark.py to choose properly)")
        return "gcn"
    b = pd.read_csv(BENCH_CSV)
    b = b[(b["split"] == "scaffold") & b["kind"].str.startswith("mean of")]
    cands = {k: b[b["model"] == LABEL[k]]["MAE"].iloc[0] for k in MODEL_KEYS if (b["model"] == LABEL[k]).any()}
    if not cands:
        print("no GNN rows in the Phase 13 table, using gcn")
        return "gcn"
    print("scaffold test MAE per GNN in your Phase 13 table:")
    for k, v in sorted(cands.items(), key=lambda kv: kv[1]):
        print(f"  {LABEL[k]:<26}{v:.3f}")
    best = min(cands, key=cands.get)
    print(f"-> using {best}")
    return best


def benchmark_numbers(key):
    """The honest test MAEs of this architecture, copied into the file so predict.py can show them."""
    if not BENCH_CSV.exists():
        return None
    b = pd.read_csv(BENCH_CSV)
    b = b[b["kind"].str.startswith("mean of") & (b["model"] == LABEL[key])]
    out = {r["split"]: float(r["MAE"]) for _, r in b.iterrows()}
    return out if {"random", "scaffold"} <= set(out) else None


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="auto", choices=["auto"] + MODEL_KEYS)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--quick", action="store_true", help="1 seed, 15 epochs: plumbing check only")
    args = p.parse_args()
    if args.quick:
        args.seeds, args.epochs, args.patience = 1, 15, 15

    key = pick_model() if args.model == "auto" else args.model
    clean = pd.read_csv(CLEAN_CSV)
    graphs, cid = load_graphs()
    assert (cid == clean["cid"].values).all(), "graphs.pt is not aligned with clean.csv. Rerun graph_data.py"
    y = clean["xlogp"].values.astype(np.float64)
    make_model = lambda: build_model(key, NODE_DIM, EDGE_DIM, args.hidden, args.layers, args.dropout)

    n_params = sum(q.numel() for q in make_model().parameters())
    print(f"\nfinal model: {key}, {n_params:,} parameters, {args.seeds} seeds, trained on {len(y)} molecules")
    if not sanity_overfit(make_model, graphs, y, args):
        raise SystemExit(1)

    fit_idx, val_idx = next(ShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED).split(np.arange(len(y))))
    print(f"fit {len(fit_idx)} / validation {len(val_idx)} (the validation slice only picks the best epoch)")
    members = []
    t0 = time.time()
    for seed in range(args.seeds):
        print(f"\n  seed {seed}")
        _, info = train_one(make_model, graphs, y, fit_idx, val_idx, [], args, seed, verbose=(seed == 0))
        print(f"    -> best epoch {info['best_epoch']}, validation MAE {info['val_MAE']:.3f}  ({info['seconds']:.0f}s)")
        members.append({"state": {k: v.detach().clone() for k, v in info["state"].items()},
                        "y_mean": info["y_mean"], "y_std": info["y_std"],
                        "val_MAE": float(info["val_MAE"]), "best_epoch": int(info["best_epoch"])})

    n_heavy = clean["n_heavy"].values
    ck = {
        "model": key,
        "hparams": {"hidden": args.hidden, "layers": args.layers, "dropout": args.dropout},
        "node_dim": int(NODE_DIM), "edge_dim": int(EDGE_DIM),
        "members": members,
        "target": "PubChem computed XLogP3",
        "n_train": int(len(y)),
        "benchmark": benchmark_numbers(key),
        "domain": {"n_train": int(len(y)),
                   "n_heavy_p1": int(np.percentile(n_heavy, 1)), "n_heavy_p99": int(np.percentile(n_heavy, 99)),
                   "n_small": int((n_heavy <= 8).sum()), "n_charged": int((clean["net_charge"] != 0).sum())},
        "quick_run": bool(args.quick),
    }
    OUT.parent.mkdir(exist_ok=True)
    torch.save(ck, OUT)
    print(f"\nsaved {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")

    round_trip_check(ck, clean, graphs)
    if args.quick:
        print("\nThis was --quick: the plumbing works, but the model is rough. Run without --quick for the real one.")
    print("Next:  python predict.py \"CCO\"")


def round_trip_check(ck, clean, graphs, n=25):
    """Reload the file exactly as predict.py does and compare with the in-memory networks.

    Also proves standardize() leaves the training SMILES unchanged, so predict.py feeds the
    model the same kind of string it was trained on.
    """
    from torch_geometric.data import Batch
    from predict import load_checkpoint, predict_smiles

    _, nets = load_checkpoint(OUT)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(clean), size=n, replace=False)
    worst = 0.0
    for i in idx:
        smiles = clean["smiles"].iloc[i]
        batch = Batch.from_data_list([graphs[i]])
        with torch.no_grad():
            direct = np.mean([float(net(batch).item()) * ys + ym for net, ym, ys in nets])
        r = predict_smiles(smiles, nets, ck)
        assert r["ok"] and r["canonical"] == smiles, f"standardize changed a training SMILES: {smiles} -> {r}"
        worst = max(worst, abs(r["prediction"] - direct))
    assert worst < 1e-4, f"predict.py disagrees with the training pipeline by {worst:.5f}"
    print(f"round trip check: {n} training molecules, predict.py matches the training pipeline "
          f"(largest difference {worst:.1e}), standardize() left every SMILES unchanged")


if __name__ == "__main__":
    main()
