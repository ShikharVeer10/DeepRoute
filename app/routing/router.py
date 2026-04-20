"""
Dynamic multi-objective routing engine.

Computes optimal and alternative routes with:
  - ML/DL predicted edge weights
  - A* pathfinding with custom heuristics
  - K-shortest-paths for alternatives
  - Monte Carlo uncertainty simulation
  - Route ranking by composite score
"""

import random
import math
import uuid
import heapq
import numpy as np
import networkx as nx
from datetime import datetime

from app.schemas import (
    CombinedFeatureVector,
    RouteSegment,
    RouteResult,
    RiskLevel,
    OptimizationObjective,
)
from app.routing.edge_weight_builder import (
    compute_edge_weight,
    estimate_emissions,
    estimate_fuel_cost,
)


# ─── Core pathfinding ────────────────────────────────────────────────────────


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km between two lat/lon points."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compute_route(
    graph: nx.MultiDiGraph,
    start_node: int,
    end_node: int,
    weight_attr: str = "smart_weight",
) -> list[int]:
    """
    A* shortest path on the graph using the precomputed `weight_attr`.
    Falls back to Dijkstra if A* fails.
    """
    try:
        return nx.astar_path(graph, start_node, end_node, weight=weight_attr)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        try:
            return nx.shortest_path(graph, start_node, end_node, weight=weight_attr)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return nx.shortest_path(graph, start_node, end_node, weight="length")


def compute_k_shortest(
    graph: nx.MultiDiGraph,
    start_node: int,
    end_node: int,
    k: int = 3,
    weight_attr: str = "smart_weight",
) -> list[list[int]]:
    """
    Yen's K-shortest loopless paths algorithm.
    Returns up to k routes as lists of node IDs.
    """
    try:
        paths = list(nx.shortest_simple_paths(graph, start_node, end_node, weight=weight_attr))
        return [p for _, p in zip(range(k), paths)]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        route = compute_route(graph, start_node, end_node, weight_attr)
        return [route]


# ─── Edge weight assignment ──────────────────────────────────────────────────


def assign_smart_weights(
    graph: nx.MultiDiGraph,
    predicted_factor: float,
    features: CombinedFeatureVector,
    objective: OptimizationObjective = OptimizationObjective.BALANCED,
) -> nx.MultiDiGraph:
    """
    Assign ML-predicted 'smart_weight' to every edge in the graph.
    """
    for u, v, data in graph.edges(data=True):
        distance = data.get("length", 100)
        speed = data.get("speed_kph", features.spatial.speed_limit_kph)
        w = compute_edge_weight(distance, predicted_factor, features, objective, speed)
        data["smart_weight"] = w

    return graph


# ─── Monte Carlo uncertainty ─────────────────────────────────────────────────


def monte_carlo_travel_time(
    base_time_s: float,
    predicted_factor: float,
    n_simulations: int = 1000,
    noise_std: float = 0.05,
) -> tuple[float, float, float]:
    """
    Run Monte Carlo simulation for travel-time uncertainty.
    Uses tight noise to produce realistic confidence intervals (~±10%).

    Returns (mean_time_s, lower_95_s, upper_95_s).
    """
    rng = np.random.RandomState()
    
    # Small per-simulation noise (±5% typical)
    factors = predicted_factor + rng.normal(0, noise_std, n_simulations)
    factors = np.clip(factors, 0.85, 1.60)
    
    simulated_times = base_time_s * factors
    
    mean_t = float(np.mean(simulated_times))
    lower = float(np.percentile(simulated_times, 5))
    upper = float(np.percentile(simulated_times, 95))
    
    return mean_t, lower, upper


# ─── Route result builder ────────────────────────────────────────────────────


def _format_duration(seconds: float) -> str:
    """Convert seconds to human-readable duration string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} min"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}min"


def _classify_risk(score: float) -> RiskLevel:
    if score < 0.25:
        return RiskLevel.LOW
    elif score < 0.50:
        return RiskLevel.MEDIUM
    elif score < 0.75:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


def build_route_result(
    graph: nx.MultiDiGraph,
    node_path: list[int],
    predicted_factor: float,
    features: CombinedFeatureVector,
    rank: int = 1,
) -> RouteResult:
    """
    Construct a full RouteResult from a path of node IDs.
    """
    segments: list[RouteSegment] = []
    total_distance = 0.0
    total_time = 0.0
    risk_scores = []

    for i in range(len(node_path) - 1):
        u, v = node_path[i], node_path[i + 1]

        edge_data = {}
        if graph.has_edge(u, v):
            edge_data = list(graph[u][v].values())[0]

        dist = edge_data.get("length", 100)
        speed = edge_data.get("speed_kph", 40)
        free_flow_time = dist / (speed * 1000 / 3600) if speed > 0 else dist / 11.1
        seg_time = free_flow_time * predicted_factor

        seg_congestion = features.context.congestion_index + random.gauss(0, 0.05)
        seg_congestion = max(0, min(1, seg_congestion))

        seg_risk = features.context.road_risk_score + random.gauss(0, 0.05)
        seg_risk = max(0, min(1, seg_risk))

        segments.append(RouteSegment(
            segment_id=f"seg_{u}_{v}",
            start_node=u,
            end_node=v,
            start_lat=graph.nodes[u].get('y'),
            start_lon=graph.nodes[u].get('x'),
            end_lat=graph.nodes[v].get('y'),
            end_lon=graph.nodes[v].get('x'),
            distance_m=round(dist, 1),
            predicted_travel_time_s=round(seg_time, 1),
            speed_kph=round(speed, 1),
            congestion_index=round(seg_congestion, 3),
            risk_score=round(seg_risk, 3),
        ))

        total_distance += dist
        total_time += seg_time
        risk_scores.append(seg_risk)

    mean_time, ci_lower, ci_upper = monte_carlo_travel_time(total_time, predicted_factor)
    avg_risk = float(np.mean(risk_scores)) if risk_scores else 0.0
    reliability = max(0, min(1, 1 - avg_risk))

    total_emissions = estimate_emissions(total_distance, predicted_factor)
    total_fuel_cost = estimate_fuel_cost(total_distance, predicted_factor)

    return RouteResult(
        route_id=str(uuid.uuid4())[:8],
        segments=segments,
        total_distance_m=round(total_distance, 1),
        total_travel_time_s=round(mean_time, 1),
        total_travel_time_display=_format_duration(mean_time),
        confidence_interval_lower_s=round(ci_lower, 1),
        confidence_interval_upper_s=round(ci_upper, 1),
        risk_level=_classify_risk(avg_risk),
        reliability_score=round(reliability, 3),
        emissions_g_co2=round(total_emissions, 1),
        fuel_cost_estimate=round(total_fuel_cost, 2),
        rank=rank,
    )