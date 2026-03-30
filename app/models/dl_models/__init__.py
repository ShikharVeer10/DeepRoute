"""DL models package."""

from .lstm_model import TrafficLSTM, TemporalAttention
from .transformer_model import TrafficTransformer, PositionalEncoding
from .train_lstm import train_lstm
from .train_transformer import train_transformer

try:
    from .gnn_model import TrafficGNN
    from .hybrid_gnn_lstm import HybridGNNLSTM, GatedFusion
    from .train_gnn import train_gnn
    from .train_hybrid import train_hybrid
except ImportError:
    pass  # torch_geometric not installed
