"""
Hybrid GNN-LSTM training pipeline.

Trains the gated-fusion model that combines spatial (GNN) and temporal (LSTM)
embeddings for travel-time factor prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from app.models.dl_models.hybrid_gnn_lstm import HybridGNNLSTM
from app.data_pipeline.synthetic_data import generate_sequence_data, generate_graph_data


def train_hybrid(
    n_nodes: int = 500,
    n_sequences: int = 2000,
    seq_length: int = 12,
    batch_size: int = 64,
    epochs: int = 80,
    lr: float = 8e-4,
    patience: int = 12,
    output_dir: str = "data/models",
    seed: int = 42,
) -> dict:
    """
    Train the HybridGNNLSTM model.

    The model receives:
      - Static graph data (node features + edge index) from generate_graph_data
      - Temporal sequences from generate_sequence_data
    And learns a gated fusion of GNN spatial embeddings with LSTM temporal embeddings.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Generate data ─────────────────────────────────────────────────────────
    node_feat, edge_idx, _ = generate_graph_data(n_nodes=n_nodes, seed=seed)
    X_seq, y_seq = generate_sequence_data(n_sequences=n_sequences, seq_length=seq_length, seed=seed)

    node_features = torch.tensor(node_feat, dtype=torch.float32).to(device)
    edge_index = torch.tensor(edge_idx, dtype=torch.long).to(device)

    # Assign each sequence sample to a random graph node
    node_indices = torch.randint(0, n_nodes, (len(X_seq),), dtype=torch.long)

    split = int(0.8 * len(X_seq))
    X_train = torch.tensor(X_seq[:split])
    y_train = torch.tensor(y_seq[:split])
    idx_train = node_indices[:split]

    X_val = torch.tensor(X_seq[split:])
    y_val = torch.tensor(y_seq[split:])
    idx_val = node_indices[split:]

    train_ds = TensorDataset(X_train, y_train, idx_train)
    val_ds = TensorDataset(X_val, y_val, idx_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    n_features = X_seq.shape[2]
    model = HybridGNNLSTM(input_size=n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    patience_counter = 0
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb, nidx in train_loader:
            xb, yb, nidx = xb.to(device), yb.to(device), nidx.to(device)
            pred = model(node_features, edge_index, xb, nidx)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        scheduler.step()
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, nidx in val_loader:
                xb, yb, nidx = xb.to(device), yb.to(device), nidx.to(device)
                pred = model(node_features, edge_index, xb, nidx)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), out_path / "gnn_lstm.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Hybrid] Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"[Hybrid] Epoch {epoch + 1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    metrics = {
        "model": "HybridGNN-LSTM",
        "best_val_loss": float(best_val_loss),
        "epochs_trained": epoch + 1,
        "model_path": str(out_path / "gnn_lstm.pth"),
    }

    print(f"[Hybrid] Best val loss: {best_val_loss:.4f}")
    return metrics


if __name__ == "__main__":
    train_hybrid()
