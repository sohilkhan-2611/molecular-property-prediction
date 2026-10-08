"""Phase 11 (and 12): train a graph neural network on the molecular graphs.

Run:  python src/train_gnn.py                    GCN, 3 seeds, both splits (Phase 11)
      python src/train_gnn.py --model mpnn       MPNN with bond features (Phase 12)
      python src/train_gnn.py --model mpnn_nobond   the same MPNN with bond features switched off (ablation)
      python src/train_gnn.py --quick            1 seed, 15 epochs, to check the plumbing (about 1 min)
      python src/train_gnn.py --model gcn --seeds 5

Reads   data/processed/graphs.pt, clean.csv, results/phase7_rf_results.csv, phase8_xgb_results.csv
Writes  results/gnn_<model>_results.csv       one row per seed, plus mean and seed-average rows
        results/gnn_<model>_predictions.csv   test predictions of every seed (reused in Phase 13)

How this stays honest
  * Same train/test molecules as Random Forest and XGBoost (src/splits.py).
  * From the training part, 10% is held back as a VALIDATION set. It picks the best epoch and
    lowers the learning rate. The test set is never used for any decision.
    On the scaffold split the validation set is also scaffold-grouped, so it is a fair mini-test.
  * Several seeds, because a neural network's score moves with its random start.
    We report mean and spread, and also the average prediction of all seeds (an ensemble).
  * Targets are standardised for training (mean 0, std 1) and converted back, so every error
    below is in XLogP3 units like the earlier phases.
  * Before the real run, a sanity check: can the model memorise 64 molecules? If not, there is a bug.
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from torch import nn
from torch_geometric.loader import DataLoader

from graph_data import EDGE_DIM, NODE_DIM, load_graphs
from models.gcn import GCN
from models.mpnn import MPNN
from splits import SEED, get_splits, murcko_scaffold, self_checks
from train_rf import scores

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "processed" / "clean.csv"
RF_CSV = ROOT / "results" / "phase7_rf_results.csv"
XGB_CSV = ROOT / "results" / "phase8_xgb_results.csv"

MODELS = {
    "gcn": lambda a: GCN(NODE_DIM, hidden=a.hidden, layers=a.layers, dropout=a.dropout),
    "mpnn": lambda a: MPNN(NODE_DIM, EDGE_DIM, hidden=a.hidden, steps=a.layers, dropout=a.dropout),
    # same MPNN, bond features zeroed out: tells you what the bond information is actually worth
    "mpnn_nobond": lambda a: MPNN(NODE_DIM, EDGE_DIM, hidden=a.hidden, steps=a.layers, dropout=a.dropout,
                                  use_bond_features=False),
}
# which earlier GNN a new one is compared against in the "is the difference real?" check
REFERENCE = {"mpnn": "gcn", "mpnn_nobond": "mpnn"}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def split_validation(train_idx, split_name, scaffolds):
    """Hold back 10% of the training molecules. Fixed seed, so only the model's own randomness varies."""
    if split_name == "scaffold":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        fit, val = next(splitter.split(train_idx, groups=scaffolds[train_idx]))
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        fit, val = next(splitter.split(train_idx))
    return train_idx[fit], train_idx[val]


@torch.no_grad()
def predict(model, loader, y_mean, y_std):
    model.eval()
    out = [model(b).numpy() * y_std + y_mean for b in loader]
    return np.concatenate(out)


def train_one(make_model, graphs, y, fit_idx, val_idx, test_idx, args, seed, verbose):
    """Train until validation MAE stops improving. Returns (predictions on test, info dict)."""
    set_seed(seed)
    y_mean, y_std = float(y[fit_idx].mean()), float(y[fit_idx].std())
    gen = torch.Generator().manual_seed(seed)
    fit_loader = DataLoader([graphs[i] for i in fit_idx], batch_size=args.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=256)
    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=256)

    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    loss_fn = nn.MSELoss()

    best = {"val": np.inf, "epoch": 0, "state": None}
    history, t0 = [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in fit_loader:
            opt.zero_grad()
            target = (batch.y - y_mean) / y_std
            loss = loss_fn(model(batch), target)
            loss.backward()
            opt.step()
            total += float(loss) * batch.num_graphs
        train_rmse = (total / len(fit_idx)) ** 0.5 * y_std            # in XLogP3 units
        val_mae = float(np.abs(predict(model, val_loader, y_mean, y_std) - y[val_idx]).mean())
        sched.step(val_mae)
        history.append((epoch, train_rmse, val_mae))
        if val_mae < best["val"] - 1e-4:
            best = {"val": val_mae, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"      epoch {epoch:>3}  train RMSE {train_rmse:.3f}  val MAE {val_mae:.3f}  "
                  f"lr {opt.param_groups[0]['lr']:.0e}  ({time.time() - t0:.0f}s)")
        if epoch - best["epoch"] >= args.patience:
            break

    model.load_state_dict(best["state"])
    pred = predict(model, test_loader, y_mean, y_std) if len(test_idx) else None
    info = {"best_epoch": best["epoch"], "val_MAE": best["val"], "epochs_run": epoch,
            "seconds": time.time() - t0, "history": history,
            # kept so train_final.py can save the trained network (main() ignores these)
            "state": best["state"], "y_mean": y_mean, "y_std": y_std}
    return pred, info


def sanity_overfit(make_model, graphs, y, args):
    """The classic bug detector: a healthy model can memorise 64 molecules."""
    set_seed(0)
    idx = np.arange(64)
    batch = next(iter(DataLoader([graphs[i] for i in idx], batch_size=64)))
    y_mean, y_std = float(y[idx].mean()), float(y[idx].std())
    model = make_model()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(400):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(batch), (batch.y - y_mean) / y_std)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        mae = float(np.abs(model(batch).numpy() * y_std + y_mean - y[idx]).mean())
    ok = mae < 0.15
    print(f"sanity check: memorise 64 molecules -> MAE {mae:.3f}  {'OK' if ok else 'PROBLEM: it cannot even memorise 64 molecules. Stop and tell me.'}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="gcn", choices=list(MODELS))
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
    p.add_argument("--no-test", action="store_true", help="development: skip the test set entirely")
    p.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    args = p.parse_args()
    if args.quick:
        args.seeds, args.epochs, args.patience = 1, 15, 15

    clean = pd.read_csv(CLEAN_CSV)
    graphs, cid = load_graphs()
    assert (cid == clean["cid"].values).all(), "graphs.pt is not aligned with clean.csv. Rerun graph_data.py"
    y = clean["xlogp"].values.astype(np.float64)
    smiles = clean["smiles"].tolist()
    splits = get_splits(smiles)
    self_checks(smiles, splits)
    scaffolds = np.array([murcko_scaffold(s) for s in smiles])
    make_model = lambda: MODELS[args.model](args)

    n_params = sum(q.numel() for q in make_model().parameters())
    print(f"model {args.model}: {n_params:,} parameters | hidden {args.hidden}, layers {args.layers}, "
          f"dropout {args.dropout}, lr {args.lr}, batch {args.batch_size} | device cpu")
    if not sanity_overfit(make_model, graphs, y, args):
        raise SystemExit(1)

    rows, all_preds = [], []
    t_start = time.time()
    for split_name in args.splits:
        tr, te = splits[split_name]
        fit_idx, val_idx = split_validation(tr, split_name, scaffolds)
        print(f"\n[{split_name}] fit {len(fit_idx)} / validation {len(val_idx)} / test {len(te)}")
        preds = {}
        for seed in range(args.seeds):
            print(f"  seed {seed}")
            pred, info = train_one(make_model, graphs, y, fit_idx, val_idx, [] if args.no_test else te,
                                   args, seed, verbose=(seed == 0))
            row = {"split": split_name, "model": args.model.upper(), "seed": seed,
                   "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                   "val_MAE": info["val_MAE"], "seconds": info["seconds"]}
            if pred is not None:
                preds[seed] = pred
                row.update(scores(y[te], pred))
                all_preds.append(pd.DataFrame({"split": split_name, "seed": seed, "cid": clean["cid"].values[te],
                                               "y_true": y[te], "y_pred": pred}))
            print(f"    -> best epoch {info['best_epoch']}, val MAE {info['val_MAE']:.3f}"
                  + ("" if pred is None else f", TEST MAE {row['MAE']:.3f}") + f"  ({info['seconds']:.0f}s)")
            rows.append(row)
        if preds:
            ens = scores(y[te], np.mean(list(preds.values()), axis=0))
            rows.append({"split": split_name, "model": f"{args.model.upper()} (seed average)", "seed": -1, **ens})

    results = pd.DataFrame(rows)
    out = ROOT / "results" / f"gnn_{args.model}_results.csv"
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)
    if all_preds:
        pd.concat(all_preds).to_csv(ROOT / "results" / f"gnn_{args.model}_predictions.csv", index=False)
    report(results, args, time.time() - t_start)


def compare_to_reference(model, ref, n_boot=2000):
    """Is the seed-averaged MAE of `model` really different from `ref`? Resample the test molecules."""
    a_path = ROOT / "results" / f"gnn_{model}_predictions.csv"
    b_path = ROOT / "results" / f"gnn_{ref}_predictions.csv"
    if not (a_path.exists() and b_path.exists()):
        return
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    print(f"\nIS THE DIFFERENCE REAL?  {model.upper()} minus {ref.upper()}, seed-averaged predictions, "
          f"{n_boot} resamples of the test set (negative = {model.upper()} better)")
    rng = np.random.RandomState(SEED)
    for split_name in a["split"].unique():
        pa = a[a["split"] == split_name].groupby("cid").agg(y=("y_true", "first"), p=("y_pred", "mean"))
        pb = b[b["split"] == split_name].groupby("cid")["y_pred"].mean().reindex(pa.index)
        if pb.isna().any():
            continue
        ea, eb = np.abs(pa["p"].values - pa["y"].values), np.abs(pb.values - pa["y"].values)
        idx = rng.randint(0, len(ea), size=(n_boot, len(ea)))
        diff = ea[idx].mean(axis=1) - eb[idx].mean(axis=1)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        verdict = "real" if (hi < 0 or lo > 0) else "NOT distinguishable from noise"
        print(f"  {split_name:<9} {ea.mean() - eb.mean():+.3f}  95% interval [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")


def report(results, args, seconds):
    name = args.model.upper()
    line = "=" * 92
    print(f"\n{line}\n{name} RESULTS  (test = same 20% as RF and XGBoost, units = XLogP3, total {seconds / 60:.1f} min)\n{line}")
    if "MAE" not in results.columns:
        print("development run, test set not used.")
        print(results.groupby("split")[["val_MAE", "best_epoch"]].mean().round(3))
        return
    rf = pd.read_csv(RF_CSV) if RF_CSV.exists() else None
    xgb = pd.read_csv(XGB_CSV) if XGB_CSV.exists() else None
    for split_name in results["split"].unique():
        part = results[(results["split"] == split_name) & (results["seed"] >= 0)]
        ens = results[(results["split"] == split_name) & (results["seed"] == -1)]
        print(f"\n{split_name.upper()} split")
        print(f"  {'model':<38}{'test MAE':>9}{'test RMSE':>11}{'test R2':>9}")
        if rf is not None:
            for m in ["Baseline: always the median", "RF: Descriptors"]:
                r = rf[(rf["split"] == split_name) & (rf["model"] == m)].iloc[0]
                print(f"  {m:<38}{r['MAE']:>9.3f}{r['RMSE']:>11.3f}{r['R2']:>9.3f}")
        if xgb is not None:
            for m in ["XGB tuned: Descriptors", "XGB tuned: Morgan + descriptors"]:
                r = xgb[(xgb["split"] == split_name) & (xgb["model"] == m)].iloc[0]
                print(f"  {m:<38}{r['MAE']:>9.3f}{r['RMSE']:>11.3f}{r['R2']:>9.3f}")
        for other in ["gcn", "mpnn", "mpnn_nobond"]:
            f = ROOT / "results" / f"gnn_{other}_results.csv"
            if other != args.model and f.exists():
                o = pd.read_csv(f)
                om = o[(o["split"] == split_name) & (o["seed"] >= 0)]
                if len(om):
                    print(f"  {other.upper() + ' (mean of ' + str(len(om)) + ' seeds)':<38}{om['MAE'].mean():>9.3f}{om['RMSE'].mean():>11.3f}{om['R2'].mean():>9.3f}")
        print(f"  {name + ' (mean of ' + str(len(part)) + ' seeds)':<38}{part['MAE'].mean():>9.3f}{part['RMSE'].mean():>11.3f}{part['R2'].mean():>9.3f}")
        print(f"  {'   spread across seeds (std of MAE)':<38}{part['MAE'].std(ddof=0):>9.3f}")
        if len(ens):
            e = ens.iloc[0]
            print(f"  {name + ' (average of seeds, an ensemble)':<38}{e['MAE']:>9.3f}{e['RMSE']:>11.3f}{e['R2']:>9.3f}")
        print(f"  best epochs {part['best_epoch'].tolist()}  | mean seconds per seed {part['seconds'].mean():.0f}")
    ref = REFERENCE.get(args.model)
    if ref:
        compare_to_reference(args.model, ref)
    print(f"\nsaved results/gnn_{args.model}_results.csv and gnn_{args.model}_predictions.csv")
    print(line)


if __name__ == "__main__":
    main()
