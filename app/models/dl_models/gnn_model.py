"""
Graph Neural Network model for traffic prediction on road networks.

Architecture:
  Node features (n_nodes, 15)
  → GCNConv (15 → 64) + BatchNorm + ReLU
  → GCNConv (64 → 32) + BatchNorm + ReLU
  → GCNConv (32 → 16) + ReLU
  → FC head → scalar travel_time_factor per node
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool


class TrafficGNN(nn.Module):
    """
    3-layer GCN for per-node travel-time factor prediction.

    Parameters
    ----------
    in_channels  : number of node features (default 15)
    hidden_channels : hidden layer width (default 64)
    dropout : dropout probability (default 0.2)
    """

    def __init__(
        self,
        in_channels: int = 15,
        hidden_channels: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.bn2 = BatchNorm(hidden_channels // 2)

        self.conv3 = GCNConv(hidden_channels // 2, hidden_channels // 4)

        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_channels // 4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x          : (n_nodes, in_channels)
        edge_index : (2, n_edges)  COO format
        batch      : optional batch assignment for multiple graphs

        Returns (n_nodes, 1) travel_time_factor predictions.
        """
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        return self.head(x)

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return intermediate node embeddings (n_nodes, hidden//4) for the hybrid model."""
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.conv3(x, edge_index))
        return x
