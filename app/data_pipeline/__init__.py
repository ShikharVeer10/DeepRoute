"""Data pipeline package."""

from .traffic_loader import get_traffic
from .weather_loader import get_weather
from .synthetic_data import (
    generate_training_data,
    generate_sequence_data,
    generate_graph_data,
    save_datasets,
)

try:
    from .osm_loader import load_graph, get_nearest_node
except ImportError:
    pass  # osmnx is optional — graph loading unavailable
