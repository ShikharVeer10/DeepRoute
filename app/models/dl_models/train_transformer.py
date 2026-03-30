"""
Transformer training pipeline.

Trains the TrafficTransformer on sequential data.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from app.models.dl_models.transformer_model import TrafficTransformer
from app.data_pipeline.synthetic_data import generate_sequence_data


def train_transformer(
    n_sequences: int = 3000,
    seq_length: int = 12,
    batch_size: int = 64,
    epochs: int = 80,
    lr: float = 5e-4,
    patience: int = 12,
    output_dir: str = "data/models",
    seed: int = 42,
) -> dict:
    """
    Train the TrafficTransformer model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_sequence_data(n_sequences=n_sequences, seq_length=seq_length, seed=seed)

    split = int(0.8 * len(X))
    X_train, X_val = torch.tensor(X[:split]), torch.tensor(X[split:])
    y_train, y_val = torch.tensor(y[:split]), torch.tensor(y[split:])

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TrafficTransformer(input_size=X.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    patience_counter = 0
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
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
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), out_path / "transformer.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Transformer] Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"[Transformer] Epoch {epoch + 1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    metrics = {
        "model": "TrafficTransformer",
        "best_val_loss": float(best_val_loss),
        "epochs_trained": epoch + 1,
        "model_path": str(out_path / "transformer.pth"),
    }

    print(f"[Transformer] Best val loss: {best_val_loss:.4f}")
    return metrics


if __name__ == "__main__":
    train_transformer()
