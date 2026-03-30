"""
GNN training pipeline.

Trains the 3-layer GCN on graph-structured road-network data.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.data import Data

from app.models.dl_models.gnn_model import TrafficGNN
from app.data_pipeline.synthetic_data import generate_graph_data


def train_gnn(
    n_nodes: int = 500,
    avg_degree: int = 4,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    output_dir: str = "data/models",
    seed: int = 42,
) -> dict:
    """
    Train the TrafficGNN model on synthetic graph data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_feat, edge_idx, node_tgt = generate_graph_data(n_nodes=n_nodes, avg_degree=avg_degree, seed=seed)

    x = torch.tensor(node_feat, dtype=torch.float32)
    edge_index = torch.tensor(edge_idx, dtype=torch.long)
    y = torch.tensor(node_tgt, dtype=torch.float32)

    n_train = int(0.8 * n_nodes)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask = ~train_mask

    data = Data(x=x, edge_index=edge_index, y=y)
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    model = TrafficGNN(in_channels=x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    patience_counter = 0
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        pred = model(data.x, data.edge_index)

        train_loss = criterion(pred[train_mask], data.y[train_mask])
        optimizer.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(data.x, data.edge_index)
            val_loss = criterion(val_pred[val_mask], data.y[val_mask]).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), out_path / "gnn.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[GNN] Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"[GNN] Epoch {epoch + 1:3d} | Train: {train_loss.item():.4f} | Val: {val_loss:.4f}")

    metrics = {
        "model": "TrafficGNN",
        "best_val_loss": float(best_val_loss),
        "epochs_trained": epoch + 1,
        "model_path": str(out_path / "gnn.pth"),
    }

    print(f"[GNN] Best val loss: {best_val_loss:.4f}")
    return metrics


if __name__ == "__main__":
    train_gnn()
