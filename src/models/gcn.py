"""Phase 11: a plain Graph Convolutional Network (GCN) for molecules.

What one GCN layer does, in plain words
  Every atom looks at its bonded neighbours, averages their feature vectors with its own
  (a degree-normalised average), and passes the result through a small learned layer.
  After 1 layer an atom "knows" its neighbours. After 3 layers it knows everything within
  3 bonds, which is about the same reach as a Morgan fingerprint with radius 3.

What it does NOT do
  GCNConv has no input for bond features. A single and a double bond look identical to it.
  Only the connection pattern and the atom features reach the model. Phase 12 (MPNN) fixes this.

From atoms to one molecule-level number
  We pool the atoms of each molecule with BOTH a sum and a mean and feed both to a small
  network. The sum matters here: XLogP3 is built by adding up atom contributions, so a molecule
  with twice the atoms should be able to produce a bigger number. A mean alone cannot tell
  ethane from hexadecane.
"""

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool


class GCN(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_dim if i == 0 else hidden, hidden) for i in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = self.dropout(torch.relu(norm(conv(x, edge_index))))
        pooled = torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        return self.head(pooled).squeeze(-1)
