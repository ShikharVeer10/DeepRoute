"""Compatibility layer for route planning helpers."""
from __future__ import annotations

import math
from itertools import islice
from typing import Any

import networkx as nx

from app.routing.route_planner import (
    _classify_risk,
    _format_duration,
    fetch_diverse_routes,
    monte_carlo_cvar_travel_time,
    plan_intelligent_routes,
    plan_routes,
)
from app.schemas import RouteResult


def monte_carlo_travel_time(
    base_time_s: float,
    predicted_factor: float,
    n_simulations: int = 1000,
    noise_std: float = 0.05,
) -> tuple[float, float, float]:
    mean_t, lower, upper, _ = monte_carlo_cvar_travel_time(
        base_time_s=base_time_s,
        predicted_factor=predicted_factor,
        n_simulations=n_simulations,
        noise_std=noise_std,
    )
    return mean_t, lower, upper


def compute_route(
    graph: nx.MultiDiGraph,
    start_node: int,
    end_node: int,
    weight_attr: str = "smart_weight",
) -> list[int]:
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
    try:
        generator = nx.shortest_simple_paths(graph, start_node, end_node, weight=weight_attr)
        return [list(path) for path in islice(generator, k)]
    except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
        return []


def assign_smart_weights(
    graph: nx.MultiDiGraph,
    congestion_factor: float = 1.0,
    weather_factor: float = 1.0,
) -> nx.MultiDiGraph:
    weighted = graph.copy()
    for _, _, data in weighted.edges(data=True):
        length = float(data.get("length", 1.0))
        speed = max(1.0, float(data.get("speed_kph", data.get("maxspeed", 40.0))))
        road_type = str(data.get("highway", data.get("road_type", "road"))).lower()
        road_penalty = 1.0
        if "motorway" in road_type or "trunk" in road_type:
            road_penalty = 0.9
        elif "primary" in road_type:
            road_penalty = 1.0
        elif "secondary" in road_type:
            road_penalty = 1.08
        else:
            road_penalty = 1.15
        travel_time = length / max(speed, 1.0)
        smart = travel_time * road_penalty * congestion_factor * weather_factor
        data["smart_weight"] = smart
    return weighted


def build_route_result(route: dict[str, Any]) -> RouteResult:
    return RouteResult(**route)
