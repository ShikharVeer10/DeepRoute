"""
LSTM model definition for traffic / travel-time prediction.

Architecture:
  Input (seq_len, 15 features)
  → LSTM (hidden=64, 2 layers, dropout)
  → Self-Attention over time steps
  → Fully-connected head → scalar travel_time_factor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Scaled dot-product self-attention over the time dimension."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, hidden)  →  (batch, hidden)"""
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn = F.softmax(attn, dim=-1)
        context = torch.bmm(attn, V)
        return context.mean(dim=1)


class TrafficLSTM(nn.Module):
    """
    Bi-directional LSTM with temporal attention for travel-time factor prediction.

    Parameters
    ----------
    input_size  : number of input features per time step (default 15)
    hidden_size : LSTM hidden dimension (default 64)
    num_layers  : stacked LSTM layers (default 2)
    dropout     : recurrent dropout (default 0.2)
    """

    def __init__(
        self,
        input_size: int = 15,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = TemporalAttention(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)  →  (batch, 1)
        """
        batch, seq_len, feat = x.shape
        x_flat = x.reshape(batch * seq_len, feat)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(batch, seq_len, feat)

        lstm_out, _ = self.lstm(x)
        attended = self.attention(lstm_out)
        return self.head(attended)
