"""Phase 12: a message passing neural network (MPNN) that uses bond features.

How it differs from the GCN of Phase 11 (three things change at once, not one)
  1. BOND FEATURES. Every message an atom receives from a neighbour is built from the neighbour's
     vector PLUS a learned transformation of the 6 numbers describing that bond (single/double/
     triple/aromatic, conjugated, in a ring), passed through a ReLU. A double bond and a single
     bond therefore send different messages. (This is the GINE layer from PyTorch Geometric.)
  2. GRU UPDATE. Instead of replacing an atom's vector with the new aggregate, a GRU (a small
     memory cell) decides how much of the old state to keep and how much of the message to take.
  3. SHARED STEPS. The same layer is applied `steps` times with the same weights (the GCN used
     three separate layers). No BatchNorm here; dropout only.
  Because of 2 and 3 a gain over the GCN cannot be credited to bond features alone. The clean
  test is the same MPNN with bond features switched off: python src/train_gnn.py --model mpnn_nobond

Design note: the textbook MPNN multiplies each neighbour by a bond-specific weight MATRIX
(PyG's NNConv). I measured that on this data: about 13 times slower per epoch than the GCN on a
2-core machine. The additive GINE version keeps the idea (messages depend on the bond) at about
the GCN's cost, so a full 3-seed run stays practical.

Readout is the same as the GCN: sum pooling plus mean pooling, then a small head.
"""

import torch
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool


class MPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=128, steps=3, dropout=0.1, use_bond_features=True):
        super().__init__()
        self.use_bond_features = use_bond_features
        self.steps = steps
        self.embed = nn.Linear(node_dim, hidden)
        message_net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.conv = GINEConv(message_net, edge_dim=edge_dim)
        self.gru = nn.GRU(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if self.use_bond_features else torch.zeros_like(data.edge_attr)
        out = torch.relu(self.embed(x))
        h = out.unsqueeze(0)                                    # GRU memory, one vector per atom
        for _ in range(self.steps):
            message = torch.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(message.unsqueeze(0), h)
            out = self.dropout(out.squeeze(0))
        pooled = torch.cat([global_add_pool(out, batch), global_mean_pool(out, batch)], dim=1)
        return self.head(pooled).squeeze(-1)
