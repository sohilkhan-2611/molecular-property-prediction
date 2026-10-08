"""One place that knows how to build each network, so training and prediction cannot drift apart."""

from models.gcn import GCN
from models.mpnn import MPNN

MODEL_KEYS = ["gcn", "mpnn", "mpnn_nobond"]


def build_model(key, node_dim, edge_dim, hidden=128, layers=3, dropout=0.1):
    """Same settings as train_gnn.py. `layers` is the number of message passing steps."""
    if key == "gcn":
        return GCN(node_dim, hidden=hidden, layers=layers, dropout=dropout)
    if key in ("mpnn", "mpnn_nobond"):
        return MPNN(node_dim, edge_dim, hidden=hidden, steps=layers, dropout=dropout,
                    use_bond_features=(key == "mpnn"))
    raise ValueError(f"unknown model {key!r}, choose from {MODEL_KEYS}")
