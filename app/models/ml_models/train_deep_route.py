
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from app.models.ml_models.deep_route_model import DeepRouteModel, model_summary
from app.models.model_registry import register_model
from app.schemas import ModelType

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using device: {device}")

# Seed
torch.manual_seed(42)
np.random.seed(42)

def train_deep_route(epochs: int = 15, batch_size: int = 64, lr: float = 0.001):
    data_dir = Path("data")
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Datasets
    print("\n[1/4] Loading generated datasets...")
    if not (data_dir / "X_seq.npy").exists():
        raise FileNotFoundError("Sequence datasets not found. Run synthetic data generation first.")

    X_seq = np.load(data_dir / "X_seq.npy")  # (2000, 12, 27)
    y_seq = np.load(data_dir / "y_seq.npy")  # (2000, 1)
    node_feat = np.load(data_dir / "node_features.npy")  # (200, 27)
    edge_index = np.load(data_dir / "edge_index.npy")  # (2, n_edges)

    num_samples = X_seq.shape[0]
    seq_len = X_seq.shape[1]
    n_features = X_seq.shape[2]
    num_nodes = node_feat.shape[0]
    num_edges = edge_index.shape[1]

    print(f"  Samples: {num_samples} | Sequence Length: {seq_len} | Features: {n_features}")
    print(f"  Nodes: {num_nodes} | Edges in Adjacency: {num_edges}")

    # Convert static data to tensors
    node_features_t = torch.tensor(node_feat, dtype=torch.float32, device=device)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)

    # 2. Instantiate Model
    print("\n[2/4] Instantiating DeepRoute PyTorch Model...")
    model = DeepRouteModel(
        num_edge_features=n_features,
        temporal_d_model=64,
        temporal_nhead=4,
        temporal_layers=2,
        num_edges=num_edges,
        node_feature_dim=n_features,
        gat_hidden_dim=32,
        gat_output_dim=64,
        gat_heads=4,
        gat_layers=2,
        pred_hidden_dim=128,
        dropout=0.1,
    ).to(device)

    print(model_summary(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Training Loop
    print(f"\n[3/4] Training deep learning model for {epochs} epochs...")
    model.train()

    # Pre-generate target route next-nodes for classification task
    # To simulate routing towards destinations:
    # We select random source/dest nodes, and target next-node is a valid neighbor of src
    adj_list = {i: [] for i in range(num_nodes)}
    for u, v in edge_index.T:
        adj_list[u].append(v)

    for epoch in range(epochs):
        t0 = time.time()
        permutation = np.random.permutation(num_samples)
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_acc = 0.0
        batches = 0

        for i in range(0, num_samples, batch_size):
            indices = permutation[i : i + batch_size]
            current_batch_size = len(indices)
            if current_batch_size < 4:
                continue

            batches += 1

            # Get batch sequences
            batch_x = torch.tensor(X_seq[indices], dtype=torch.float32, device=device)
            batch_y = torch.tensor(y_seq[indices], dtype=torch.float32, device=device) # (batch, 1)

            # Generate random src, dst, and target next nodes for classification
            src_idx = torch.randint(0, num_nodes, (current_batch_size,), device=device)
            dst_idx = torch.randint(0, num_nodes, (current_batch_size,), device=device)
            
            # Select target next node (ensure it's a neighbor if possible, else random)
            target_next = []
            for src_node in src_idx.tolist():
                neighbors = adj_list[src_node]
                if neighbors:
                    target_next.append(np.random.choice(neighbors))
                else:
                    target_next.append(np.random.randint(0, num_nodes))
            target_next_t = torch.tensor(target_next, dtype=torch.long, device=device)

            # Forward pass
            optimizer.zero_grad()
            next_node_probs, edge_weights, _ = model(
                batch_x, node_features_t, edge_index_t, src_idx, dst_idx
            )

            # Loss 1: Travel time factor regression on edge weights (MSE)
            # Expand batch_y (predicted average factor) to all edges as a baseline
            target_edge_weights = batch_y.expand(-1, num_edges)  # (batch, num_edges)
            loss_mse = F.mse_loss(edge_weights, target_edge_weights)

            # Loss 2: Next node probability classification (Cross Entropy)
            loss_ce = F.cross_entropy(torch.log(next_node_probs.clamp(min=1e-8)), target_next_t)

            # Joint objective
            loss = loss_ce + 15.0 * loss_mse

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()

            # Calculate batch next-node prediction accuracy
            preds = next_node_probs.argmax(dim=-1)
            correct = (preds == target_next_t).sum().item()
            epoch_acc += correct / current_batch_size

        scheduler.step()
        elapsed = time.time() - t0
        avg_loss = epoch_loss / batches
        avg_mse = epoch_mse / batches
        avg_acc = epoch_acc / batches

        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | MSE (Factor): {avg_mse:.6f} | NextNode Acc: {avg_acc*100:.1f}% | Time: {elapsed:.2f}s")

    # 4. Evaluation and Registry
    print("\n[4/4] Finalizing and saving model...")
    model.eval()

    model_path = model_dir / "deep_route_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  Model weights successfully saved to: {model_path}")

    # Register the model in registry.json
    final_mae = float(np.sqrt(avg_mse))
    register_model(
        name="deep_route",
        model_type=ModelType.DEEP_ROUTE,
        version="1.0.0",
        metrics={"mae": round(final_mae, 6), "next_node_acc": round(avg_acc, 4)},
        file_path=str(model_path.resolve())
    )

    print("\n[SUCCESS] DeepRoute Model trained and registered successfully!")

if __name__ == "__main__":
    train_deep_route()
