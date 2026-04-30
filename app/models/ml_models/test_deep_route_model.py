"""
DeepRoute Model -- Forward Pass Verification & Demo.

This script instantiates the full DeepRouteModel with dummy data and
verifies that every tensor shape is correct through the pipeline.

Run:
    python -m app.models.ml_models.test_deep_route_model
"""

import torch
from app.models.ml_models.deep_route_model import (
    DeepRouteModel,
    TemporalEncoder,
    GraphAttentionNetwork,
    RoutePredictionHead,
    model_summary,
    count_parameters,
)


def _build_dummy_graph(num_nodes: int = 50, avg_degree: int = 4, seed: int = 42):
    """Build a random road-network-like graph in COO format."""
    torch.manual_seed(seed)
    edges_src, edges_dst = [], []
    for n in range(num_nodes):
        n_nbrs = torch.randint(1, avg_degree + 1, (1,)).item()
        nbrs = torch.randint(0, num_nodes, (n_nbrs,))
        for nb in nbrs:
            if nb.item() != n:
                edges_src.append(n)
                edges_dst.append(nb.item())
                edges_src.append(nb.item())
                edges_dst.append(n)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def test_temporal_encoder():
    """Test the TemporalEncoder independently."""
    print("\n[1/4] Testing TemporalEncoder...")
    B, T, F_e, E = 4, 12, 8, 200

    model = TemporalEncoder(num_edge_features=F_e, num_edges=E)
    x = torch.randn(B, T, F_e)

    out = model(x)
    assert out.shape == (B, E), f"Expected ({B}, {E}), got {out.shape}"
    assert (out >= 0).all(), "Softplus should produce non-negative outputs"
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Params: {count_parameters(model):,}")
    print("  PASSED")


def test_gat():
    """Test the GraphAttentionNetwork independently."""
    print("\n[2/4] Testing GraphAttentionNetwork...")
    N, F_n, D_out = 50, 16, 64

    edge_index = _build_dummy_graph(N)
    num_edges = edge_index.size(1)

    model = GraphAttentionNetwork(
        node_feature_dim=F_n, output_dim=D_out, num_heads=4, num_layers=2
    )
    x = torch.randn(N, F_n)
    ew = torch.rand(num_edges)

    out = model(x, edge_index, ew)
    assert out.shape == (N, D_out), f"Expected ({N}, {D_out}), got {out.shape}"
    print(f"  Nodes:  {N}  |  Edges: {num_edges}")
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Params: {count_parameters(model):,}")
    print("  PASSED")


def test_prediction_head():
    """Test the RoutePredictionHead independently."""
    print("\n[3/4] Testing RoutePredictionHead...")
    N, D, B = 50, 64, 4

    model = RoutePredictionHead(embed_dim=D, hidden_dim=128)
    node_emb = torch.randn(N, D)
    src = torch.randint(0, N, (B,))
    dst = torch.randint(0, N, (B,))

    probs = model(node_emb, src, dst)
    assert probs.shape == (B, N), f"Expected ({B}, {N}), got {probs.shape}"
    # Check that probabilities sum to 1
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), f"Probabilities should sum to 1, got {sums}"
    print(f"  Input:  node_emb={tuple(node_emb.shape)}, src={tuple(src.shape)}, dst={tuple(dst.shape)}")
    print(f"  Output: {tuple(probs.shape)}")
    print(f"  Prob sums: {sums.tolist()}")
    print(f"  Params: {count_parameters(model):,}")
    print("  PASSED")


def test_full_model():
    """End-to-end forward pass of DeepRouteModel."""
    print("\n[4/4] Testing DeepRouteModel (full pipeline)...")

    # ---- Hyperparameters ----
    BATCH       = 4
    SEQ_LEN     = 12
    EDGE_FEATS  = 8
    NUM_NODES   = 50
    NODE_FEATS  = 16
    GAT_OUT     = 64

    edge_index = _build_dummy_graph(NUM_NODES)
    NUM_EDGES = edge_index.size(1)

    model = DeepRouteModel(
        num_edge_features=EDGE_FEATS,
        temporal_d_model=64,
        temporal_nhead=4,
        temporal_layers=2,
        num_edges=NUM_EDGES,
        node_feature_dim=NODE_FEATS,
        gat_hidden_dim=32,
        gat_output_dim=GAT_OUT,
        gat_heads=4,
        gat_layers=2,
        pred_hidden_dim=128,
        dropout=0.1,
    )

    # ---- Dummy inputs ----
    traffic_series = torch.randn(BATCH, SEQ_LEN, EDGE_FEATS)
    node_features  = torch.randn(NUM_NODES, NODE_FEATS)
    src_idx        = torch.randint(0, NUM_NODES, (BATCH,))
    dst_idx        = torch.randint(0, NUM_NODES, (BATCH,))

    # ---- Forward pass ----
    model.eval()
    with torch.no_grad():
        next_node_probs, edge_weights, node_embeddings = model(
            traffic_series, node_features, edge_index, src_idx, dst_idx
        )

    # ---- Shape assertions ----
    assert next_node_probs.shape == (BATCH, NUM_NODES)
    assert edge_weights.shape == (BATCH, NUM_EDGES)
    assert node_embeddings.shape == (NUM_NODES, GAT_OUT)
    assert torch.allclose(next_node_probs.sum(-1), torch.ones(BATCH), atol=1e-5)

    print(f"  traffic_series:  {tuple(traffic_series.shape)}")
    print(f"  node_features:   {tuple(node_features.shape)}")
    print(f"  edge_index:      {tuple(edge_index.shape)}")
    print(f"  src_idx:         {tuple(src_idx.shape)}")
    print(f"  dst_idx:         {tuple(dst_idx.shape)}")
    print(f"  ---")
    print(f"  next_node_probs: {tuple(next_node_probs.shape)}")
    print(f"  edge_weights:    {tuple(edge_weights.shape)}")
    print(f"  node_embeddings: {tuple(node_embeddings.shape)}")
    print(f"  ---")
    print(f"  Top-3 predicted nodes for sample 0: {next_node_probs[0].topk(3).indices.tolist()}")
    print(f"  Prob sums (should be ~1.0): {next_node_probs.sum(-1).tolist()}")
    print("  PASSED")

    # ---- Model summary ----
    print("\n" + model_summary(model))

    # ---- Loss demo ----
    print("\n  [Loss demo] Computing cross-entropy on dummy targets...")
    target_nodes = torch.randint(0, NUM_NODES, (BATCH,))
    model.train()
    probs, _, _ = model(traffic_series, node_features, edge_index, src_idx, dst_idx)
    loss = torch.nn.functional.cross_entropy(probs.log(), target_nodes)
    loss.backward()
    print(f"  Target nodes: {target_nodes.tolist()}")
    print(f"  Loss value:   {loss.item():.4f}")
    print(f"  Gradients OK: {all(p.grad is not None for p in model.parameters() if p.requires_grad)}")


if __name__ == "__main__":
    print("=" * 65)
    print("  DeepRoute Model -- Forward Pass Verification")
    print("=" * 65)

    test_temporal_encoder()
    test_gat()
    test_prediction_head()
    test_full_model()

    print("\n" + "=" * 65)
    print("  ALL TESTS PASSED")
    print("=" * 65)
