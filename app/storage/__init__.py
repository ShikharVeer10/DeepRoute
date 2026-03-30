"""
DeepRoute Storage Layer
SQLite-based time-series storage for traffic snapshots, user trips, and weather history.
"""

from app.storage.database import (
    init_db,
    record_traffic_snapshot,
    record_weather_snapshot,
    record_trip,
    record_prediction,
    get_historical_speed_profile,
    get_edge_traffic_history,
    get_prediction_accuracy,
    get_recent_trips,
    get_congestion_for_edges,
)
