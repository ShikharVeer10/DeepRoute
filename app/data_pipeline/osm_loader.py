"""
OpenStreetMap graph loader using OSMnx.
Loads a road network, enriches edges with metadata, and caches locally.
"""

try:
    import osmnx as ox
    _HAS_OSMNX = True
except ImportError:
    _HAS_OSMNX = False

import networkx as nx
from pathlib import Path


_CACHE_DIR = Path("data/graphs")


def load_graph(
    city: str = "Vijayawada, India",
    network_type: str = "drive",
    use_cache: bool = True,
) -> nx.MultiDiGraph:
    
    if not _HAS_OSMNX:
        raise ImportError("osmnx is required for graph loading. Install with: pip install osmnx")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = city.replace(",", "").replace(" ", "_").lower()
    cache_path = _CACHE_DIR / f"{safe_name}.graphml"

    if use_cache and cache_path.exists():
        return ox.load_graphml(cache_path)

    graph = ox.graph_from_place(city, network_type=network_type)

    graph = ox.routing.add_edge_speeds(graph)
    graph = ox.routing.add_edge_travel_times(graph)

    ox.save_graphml(graph, cache_path)
    return graph


def get_nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Return the nearest OSM node id to the given coordinates."""
    if not _HAS_OSMNX:
        raise ImportError("osmnx is required. Install with: pip install osmnx")
    return ox.nearest_nodes(graph, lon, lat)