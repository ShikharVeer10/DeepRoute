"""
Forecast-aware edge weight builder.

Computes dynamic edge weights that incorporate ML/DL predictions,
traffic congestion, weather, risk scores, and multi-objective costs.
"""

import math
from app.schemas import (
    CombinedFeatureVector,
    OptimizationObjective,
)


# ── Objective weight presets ──────────────────────────────────────────────────

_OBJECTIVE_WEIGHTS: dict[OptimizationObjective, dict[str, float]] = {
    OptimizationObjective.FASTEST: {
        "travel_time": 0.80, "distance": 0.05, "emissions": 0.05, "risk": 0.05, "reliability": 0.05,
    },
    OptimizationObjective.SHORTEST: {
        "travel_time": 0.10, "distance": 0.75, "emissions": 0.05, "risk": 0.05, "reliability": 0.05,
    },
    OptimizationObjective.SAFEST: {
        "travel_time": 0.10, "distance": 0.05, "emissions": 0.05, "risk": 0.65, "reliability": 0.15,
    },
    OptimizationObjective.ECO: {
        "travel_time": 0.15, "distance": 0.10, "emissions": 0.55, "risk": 0.05, "reliability": 0.15,
    },
    OptimizationObjective.BALANCED: {
        "travel_time": 0.35, "distance": 0.20, "emissions": 0.15, "risk": 0.15, "reliability": 0.15,
    },
}


def compute_edge_weight(
    distance_m: float,
    predicted_factor: float,
    features: CombinedFeatureVector,
    objective: OptimizationObjective = OptimizationObjective.BALANCED,
    base_speed_kph: float = 50.0,
) -> float:
    """
    Compute a multi-objective edge weight for route optimization.

    Parameters
    ----------
    distance_m       : road segment length in metres
    predicted_factor : ML/DL predicted travel-time multiplier
    features         : the combined feature vector for this segment
    objective        : optimization goal
    base_speed_kph   : free-flow speed

    Returns
    -------
    Composite edge weight (lower = better).
    """
    w = _OBJECTIVE_WEIGHTS[objective]

    # Travel time component (hours)
    free_flow_time_h = (distance_m / 1000) / max(base_speed_kph, 1)
    travel_time = free_flow_time_h * predicted_factor

    # Distance component (km, normalised)
    distance_km = distance_m / 1000

    # Emissions estimate (g CO2, simplified)
    emissions_g = distance_km * 120 * (0.5 + 0.5 * predicted_factor)

    # Risk component
    risk = features.context.road_risk_score

    # Reliability (inverse of prediction variance proxy)
    reliability_penalty = abs(predicted_factor - 1.5) / 2.0

    cost = (
        w["travel_time"] * travel_time
        + w["distance"] * distance_km
        + w["emissions"] * (emissions_g / 1000)
        + w["risk"] * risk
        + w["reliability"] * reliability_penalty
    )

    return max(cost, 1e-6)


def estimate_emissions(distance_m: float, travel_time_factor: float) -> float:
    """Estimate CO2 emissions in grams for a segment."""
    return (distance_m / 1000) * 120 * (0.5 + 0.5 * travel_time_factor)


def estimate_fuel_cost(distance_m: float, travel_time_factor: float, price_per_litre: float = 100.0) -> float:
    """Estimate fuel cost in local currency for a segment."""
    litres = (distance_m / 1000) * 0.08 * travel_time_factor
    return round(litres * price_per_litre, 2)