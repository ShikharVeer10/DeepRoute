"""
Hybrid GNN-LSTM fusion model.

Combines spatial graph embeddings from GNN with temporal sequence
embeddings from LSTM via a gated fusion mechanism.

Architecture:
  GNN embeddings (n_nodes, 16) ─┐
                                 ├─ Gated Fusion → FC head → travel_time_factor
  LSTM embeddings (batch, 128) ─┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.models.dl_models.lstm_model import TrafficLSTM
from app.models.dl_models.gnn_model import TrafficGNN


class GatedFusion(nn.Module):
    """Learnable gated fusion of two embedding streams."""

    def __init__(self, dim_a: int, dim_b: int, fused_dim: int):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, fused_dim)
        self.proj_b = nn.Linear(dim_b, fused_dim)
        self.gate = nn.Linear(fused_dim * 2, fused_dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_proj = self.proj_a(a)
        b_proj = self.proj_b(b)
        gate_input = torch.cat([a_proj, b_proj], dim=-1)
        g = torch.sigmoid(self.gate(gate_input))
        return g * a_proj + (1 - g) * b_proj


class HybridGNNLSTM(nn.Module):
    """
    Hybrid model fusing GNN spatial embeddings with LSTM temporal embeddings.

    Parameters
    ----------
    gnn_embed_dim  : dimension of GNN node embeddings (default 16)
    lstm_embed_dim : dimension of LSTM output (bidirectional hidden * 2, default 128)
    fused_dim      : fused representation dimension (default 64)
    input_size     : number of input features (default 15)
    """

    def __init__(
        self,
        gnn_embed_dim: int = 16,
        lstm_embed_dim: int = 128,
        fused_dim: int = 64,
        input_size: int = 15,
    ):
        super().__init__()
        self.gnn = TrafficGNN(in_channels=input_size, hidden_channels=64)
        self.lstm = TrafficLSTM(input_size=input_size, hidden_size=64)

        self.fusion = GatedFusion(gnn_embed_dim, lstm_embed_dim, fused_dim)

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        sequences: torch.Tensor,
        node_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_features : (n_nodes, input_size)
        edge_index    : (2, n_edges)
        sequences     : (batch, seq_len, input_size)
        node_indices  : (batch,) indices into node_features; if None use first N nodes

        Returns
        -------
        (batch, 1) travel_time_factor predictions
        """
        gnn_embeds = self.gnn.get_embeddings(node_features, edge_index)

        if node_indices is not None:
            gnn_embeds = gnn_embeds[node_indices]
        else:
            gnn_embeds = gnn_embeds[: sequences.size(0)]

        lstm_out = self.lstm(sequences)

        lstm_intermediate = self.lstm.lstm(
            self.lstm.input_bn(
                sequences.reshape(-1, sequences.size(-1))
            ).reshape(sequences.shape)
        )[0]
        lstm_embeds = self.lstm.attention(lstm_intermediate)

        fused = self.fusion(gnn_embeds, lstm_embeds)
        return self.head(fused)
