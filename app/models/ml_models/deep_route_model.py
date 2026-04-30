"""
DeepRoute: Hybrid Temporal-Graph Neural Network for Traffic-Aware Route Optimization.

Architecture Overview:
    1. TemporalEncoder    -- Multi-Head Self-Attention Transformer over time-series traffic features
    2. GraphAttentionNet  -- GAT propagation over the road network graph
    3. RoutePredictionHead -- Attention-based scoring for next-node probability
    4. DeepRouteModel     -- End-to-end composition of all modules

Paper-ready implementation. All tensor shapes annotated inline.

Author : DeepRoute Research Team
License: MIT
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. TEMPORAL ENCODER  (Transformer-based traffic time-series encoder)
# ===========================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                       # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float() # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                      # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional information added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """
    Transformer encoder that maps a per-edge traffic time-series into
    a fixed-length latent representation for each edge.

    Input:  (batch, seq_len, num_edge_features)
    Output: (batch, num_edges)  -- predicted travel-time weights per edge
    """

    def __init__(
        self,
        num_edge_features: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        num_edges: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_edges = num_edges

        # Project raw features to model dimension
        self.input_proj = nn.Linear(num_edge_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Temporal aggregation: learnable query that attends to all time steps
        self.agg_query = nn.Parameter(torch.randn(1, 1, d_model))

        # Output projection: latent -> edge travel-time weights
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_edges),
            nn.Softplus(),  # ensure positive travel times
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_edge_features) -- traffic time-series
        Returns:
            edge_weights: (batch, num_edges) -- predicted travel-time per edge
        """
        B = x.size(0)

        # (batch, seq_len, d_model)
        h = self.input_proj(x)
        h = self.pos_enc(h)

        # (batch, seq_len, d_model)
        h = self.transformer(h)

        # Learnable aggregation via cross-attention with a single query
        query = self.agg_query.expand(B, -1, -1)            # (batch, 1, d_model)
        attn_weights = torch.bmm(query, h.transpose(1, 2))  # (batch, 1, seq_len)
        attn_weights = F.softmax(attn_weights / math.sqrt(h.size(-1)), dim=-1)
        context = torch.bmm(attn_weights, h).squeeze(1)     # (batch, d_model)

        # Project to edge weights
        edge_weights = self.output_head(context)             # (batch, num_edges)
        return edge_weights


# ===========================================================================
# 2. GRAPH ATTENTION NETWORK  (GAT for road-network message passing)
# ===========================================================================

class GATLayer(nn.Module):
    """
    Single Graph Attention layer (Velickovic et al., 2018).

    Implements multi-head attention over graph neighbours with optional
    edge-weight modulation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Per-head linear transforms
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)

        # Attention mechanism parameters (source + target)
        self.a_src = nn.Parameter(torch.randn(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.randn(num_heads, out_features))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            (num_nodes, in_features) -- node feature matrix
            edge_index:   (2, num_edges)           -- COO adjacency
            edge_weights: (num_edges,) optional    -- scalar weight per edge

        Returns:
            out: (num_nodes, out_features * num_heads) if concat
                 (num_nodes, out_features)              if not concat (mean)
        """
        N = x.size(0)
        H = self.num_heads
        D = self.out_features

        # Linear transform + reshape to multi-head
        h = self.W(x).view(N, H, D)                         # (N, H, D)

        src, dst = edge_index[0], edge_index[1]              # each (E,)

        # Attention coefficients
        e_src = (h[src] * self.a_src.unsqueeze(0)).sum(-1)   # (E, H)
        e_dst = (h[dst] * self.a_dst.unsqueeze(0)).sum(-1)   # (E, H)
        e = self.leaky_relu(e_src + e_dst)                   # (E, H)

        # Modulate attention by predicted edge weight (temporal signal)
        if edge_weights is not None:
            # Normalise edge weights to [0.5, 2.0] range to avoid vanishing
            w = edge_weights.clamp(min=0.1)
            w = w / (w.mean() + 1e-8)
            e = e * w.unsqueeze(-1)                          # (E, H)

        # Sparse softmax over neighbours
        e_max = torch.zeros(N, H, device=x.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(e), e, reduce="amax",
                              include_self=True)
        e = torch.exp(e - e_max[dst])

        denom = torch.zeros(N, H, device=x.device)
        denom.scatter_add_(0, dst.unsqueeze(-1).expand_as(e), e)
        alpha = e / (denom[dst] + 1e-8)                      # (E, H)
        alpha = self.dropout(alpha)

        # Weighted message aggregation
        msg = alpha.unsqueeze(-1) * h[src]                    # (E, H, D)
        out = torch.zeros(N, H, D, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(msg), msg)

        if self.concat:
            return out.view(N, H * D)                         # (N, H*D)
        else:
            return out.mean(dim=1)                            # (N, D)


class GraphAttentionNetwork(nn.Module):
    """
    Multi-layer GAT encoder for the road network.

    Architecture:
        GATLayer (concat) -> ELU -> Dropout -> GATLayer (mean) -> ELU

    Input:  node features + edge index + temporal edge weights
    Output: node embeddings
    """

    def __init__(
        self,
        node_feature_dim: int = 16,
        hidden_dim: int = 32,
        output_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # First layer: concat heads
        self.layers.append(
            GATLayer(node_feature_dim, hidden_dim, num_heads, dropout, concat=True)
        )
        self.norms.append(nn.LayerNorm(hidden_dim * num_heads))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GATLayer(hidden_dim * num_heads, hidden_dim, num_heads, dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))

        # Final layer: mean over heads -> output_dim
        if num_layers >= 2:
            self.layers.append(
                GATLayer(hidden_dim * num_heads, output_dim, num_heads, dropout, concat=False)
            )
            self.norms.append(nn.LayerNorm(output_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            (num_nodes, node_feature_dim)
            edge_index:   (2, num_edges)
            edge_weights: (num_edges,) optional

        Returns:
            node_embeddings: (num_nodes, output_dim)
        """
        h = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h = layer(h, edge_index, edge_weights)
            h = norm(h)
            h = F.elu(h)
            h = self.dropout(h)
        return h  # (num_nodes, output_dim)


# ===========================================================================
# 3. ROUTE PREDICTION HEAD  (Attention-based node scoring)
# ===========================================================================

class RoutePredictionHead(nn.Module):
    """
    Given source/destination node embeddings and all node embeddings,
    produce a probability distribution over candidate next nodes.

    Uses a bilinear attention mechanism conditioned on the query context
    (source + destination concatenation).
    """

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        # Context encoder: combines source + destination embeddings
        self.context_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Bilinear attention: context x node_embeddings -> scores
        self.attn_bilinear = nn.Bilinear(embed_dim, embed_dim, 1, bias=True)

        # Optional MLP refinement
        self.score_mlp = nn.Sequential(
            nn.Linear(embed_dim + embed_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_embeddings: (num_nodes, embed_dim)
            src_idx:         (batch,)  -- source node indices
            dst_idx:         (batch,)  -- destination node indices

        Returns:
            probs: (batch, num_nodes) -- probability distribution over next nodes
        """
        N, D = node_embeddings.shape
        B = src_idx.size(0)

        src_emb = node_embeddings[src_idx]   # (batch, embed_dim)
        dst_emb = node_embeddings[dst_idx]   # (batch, embed_dim)

        # Build query context from source + destination
        context = self.context_mlp(
            torch.cat([src_emb, dst_emb], dim=-1)
        )  # (batch, embed_dim)

        # Expand for scoring against all nodes
        context_exp = context.unsqueeze(1).expand(B, N, D)           # (B, N, D)
        nodes_exp = node_embeddings.unsqueeze(0).expand(B, N, D)     # (B, N, D)

        # Bilinear attention scores
        bilinear_scores = self.attn_bilinear(
            context_exp.reshape(B * N, D),
            nodes_exp.reshape(B * N, D),
        ).view(B, N, 1)                                              # (B, N, 1)

        # MLP refinement with concatenated features
        combined = torch.cat([
            context_exp, nodes_exp, bilinear_scores
        ], dim=-1)                                                    # (B, N, 2D+1)
        scores = self.score_mlp(combined).squeeze(-1)                 # (B, N)

        # Softmax to get probability distribution
        probs = F.softmax(scores, dim=-1)                             # (B, N)
        return probs


# ===========================================================================
# 4. DeepRouteModel  (End-to-End Hybrid Model)
# ===========================================================================

class DeepRouteModel(nn.Module):
    """
    DeepRoute: Hybrid Temporal-Graph Neural Network for Route Optimization.

    Pipeline:
        traffic_series  -->  TemporalEncoder  -->  edge_weights
                                                       |
        node_features + edge_index + edge_weights -->  GAT  -->  node_embeddings
                                                                      |
        src_idx + dst_idx + node_embeddings -->  RoutePredictionHead  -->  P(next_node)

    This model is fully differentiable and can be trained end-to-end with
    cross-entropy loss on ground-truth next-node labels.
    """

    def __init__(
        self,
        # Temporal encoder params
        num_edge_features: int = 8,
        temporal_d_model: int = 64,
        temporal_nhead: int = 4,
        temporal_layers: int = 2,
        num_edges: int = 200,
        # GAT params
        node_feature_dim: int = 16,
        gat_hidden_dim: int = 32,
        gat_output_dim: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 2,
        # Prediction head params
        pred_hidden_dim: int = 128,
        # Shared
        dropout: float = 0.1,
    ):
        super().__init__()

        self.temporal_encoder = TemporalEncoder(
            num_edge_features=num_edge_features,
            d_model=temporal_d_model,
            nhead=temporal_nhead,
            num_layers=temporal_layers,
            dim_feedforward=temporal_d_model * 2,
            num_edges=num_edges,
            dropout=dropout,
        )

        self.gat = GraphAttentionNetwork(
            node_feature_dim=node_feature_dim,
            hidden_dim=gat_hidden_dim,
            output_dim=gat_output_dim,
            num_heads=gat_heads,
            num_layers=gat_layers,
            dropout=dropout,
        )

        self.prediction_head = RoutePredictionHead(
            embed_dim=gat_output_dim,
            hidden_dim=pred_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        traffic_series: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass of the DeepRoute hybrid model.

        Args:
            traffic_series: (batch, seq_len, num_edge_features)
                            Historical traffic time-series data.
            node_features:  (num_nodes, node_feature_dim)
                            Static + dynamic node features.
            edge_index:     (2, num_edges)
                            Graph connectivity in COO format.
            src_idx:        (batch,)
                            Source node index per sample.
            dst_idx:        (batch,)
                            Destination node index per sample.

        Returns:
            next_node_probs: (batch, num_nodes)
                             Probability distribution over candidate next nodes.
            edge_weights:    (batch, num_edges)
                             Predicted travel-time weights from temporal encoder.
            node_embeddings: (num_nodes, gat_output_dim)
                             Learned node representations from GAT.
        """
        # Stage 1: Temporal encoding -> predicted edge travel times
        edge_weights = self.temporal_encoder(traffic_series)  # (batch, num_edges)

        # Use mean edge weights across batch for graph-level propagation
        mean_edge_weights = edge_weights.mean(dim=0)          # (num_edges,)

        # Stage 2: Graph attention network -> node embeddings
        node_embeddings = self.gat(
            node_features, edge_index, mean_edge_weights
        )  # (num_nodes, gat_output_dim)

        # Stage 3: Route prediction -> next-node probabilities
        next_node_probs = self.prediction_head(
            node_embeddings, src_idx, dst_idx
        )  # (batch, num_nodes)

        return next_node_probs, edge_weights, node_embeddings


# ===========================================================================
# 5. UTILITY: Model summary & parameter count
# ===========================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: DeepRouteModel) -> str:
    """Pretty-print model architecture summary."""
    lines = [
        "=" * 65,
        "  DeepRoute Hybrid Model Architecture",
        "=" * 65,
        "",
        f"  Total parameters : {count_parameters(model):,}",
        f"  Temporal Encoder : {count_parameters(model.temporal_encoder):,} params",
        f"  GAT Network      : {count_parameters(model.gat):,} params",
        f"  Prediction Head  : {count_parameters(model.prediction_head):,} params",
        "",
        "  Pipeline:",
        "    traffic_series -> TemporalEncoder -> edge_weights",
        "    node_features + edge_index + edge_weights -> GAT -> node_embeddings",
        "    src/dst + node_embeddings -> RoutePredictionHead -> P(next_node)",
        "",
        "=" * 65,
    ]
    return "\n".join(lines)
