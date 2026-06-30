"""
Multi-objective optimization engine for routing.

Evaluates candidate routes against 11 objectives and ranks them:
- Minimize travel time
- Minimize distance
- Minimize congestion
- Minimize accidents/risk
- Minimize road closures
- Minimize weather impact
- Minimize fuel consumption
- Minimize EV energy
- Maximize road safety
- Maximize reliability
- Maximize prediction confidence
- Maximize driving comfort
"""

from typing import List, Dict, Any, Optional
from app.schemas import OptimizationObjective

# ── Objective Weights Map ─────────────────────────────────────────────────────
# Map each main OptimizationObjective to weights for our 11 detailed objectives.
# Note: The sum of weights in each category is normalized.
OBJECTIVE_WEIGHT_PRESETS = {
    OptimizationObjective.FASTEST: {
        "travel_time": 0.45,
        "cvar": 0.05,
        "distance": 0.05,
        "congestion": 0.15,
        "accidents": 0.05,
        "road_closures": 0.10,
        "weather": 0.05,
        "fuel": 0.04,
        "ev_energy": 0.02,
        "safety": 0.02,
        "reliability": 0.01,
        "confidence": 0.01,
        "comfort": 0.01,
    },
    OptimizationObjective.SHORTEST: {
        "travel_time": 0.10,
        "cvar": 0.05,
        "distance": 0.40,
        "congestion": 0.05,
        "accidents": 0.05,
        "road_closures": 0.10,
        "weather": 0.05,
        "fuel": 0.10,
        "ev_energy": 0.05,
        "safety": 0.02,
        "reliability": 0.01,
        "confidence": 0.01,
        "comfort": 0.01,
    },
    OptimizationObjective.SAFEST: {
        "travel_time": 0.05,
        "cvar": 0.10,
        "distance": 0.05,
        "congestion": 0.05,
        "accidents": 0.20,
        "road_closures": 0.10,
        "weather": 0.05,
        "fuel": 0.02,
        "ev_energy": 0.01,
        "safety": 0.25,
        "reliability": 0.08,
        "confidence": 0.04,
        "comfort": 0.05,
    },
    OptimizationObjective.ECO: {
        "travel_time": 0.10,
        "cvar": 0.05,
        "distance": 0.10,
        "congestion": 0.08,
        "accidents": 0.02,
        "road_closures": 0.05,
        "weather": 0.02,
        "fuel": 0.30,
        "ev_energy": 0.20,
        "safety": 0.02,
        "reliability": 0.03,
        "confidence": 0.01,
        "comfort": 0.02,
    },
    OptimizationObjective.BALANCED: {
        "travel_time": 0.15,
        "cvar": 0.05,
        "distance": 0.10,
        "congestion": 0.10,
        "accidents": 0.10,
        "road_closures": 0.10,
        "weather": 0.05,
        "fuel": 0.08,
        "ev_energy": 0.05,
        "safety": 0.10,
        "reliability": 0.05,
        "confidence": 0.03,
        "comfort": 0.04,
    },
    OptimizationObjective.RISK_AVERSE: {
        "travel_time": 0.10,
        "cvar": 0.45,
        "distance": 0.05,
        "congestion": 0.10,
        "accidents": 0.05,
        "road_closures": 0.10,
        "weather": 0.05,
        "fuel": 0.02,
        "ev_energy": 0.02,
        "safety": 0.02,
        "reliability": 0.02,
        "confidence": 0.01,
        "comfort": 0.01,
    },
}


class BaseRouteOptimizer:
    """Base interface for all route optimization algorithms."""

    def optimize(
        self,
        routes: List[Dict[str, Any]],
        objective: OptimizationObjective,
        discard_critical: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Optimize, score, and rank a list of candidate routes.

        Parameters
        ----------
        routes: List of routes, each containing route metrics and attributes.
        objective: The OptimizationObjective (e.g., fastest, safest, balanced).
        discard_critical: If True, discard routes with critical hazards (like complete road closure).

        Returns
        -------
        Ranked and scored routes.
        """
        raise NotImplementedError("Subclasses must implement optimize")


class WeightedSumRouteOptimizer(BaseRouteOptimizer):
    """
    Standard Weighted Sum Model (WSM) multi-objective optimizer.
    Normalizes all costs/benefits across candidates and computes composite score.
    """

    def optimize(
        self,
        routes: List[Dict[str, Any]],
        objective: OptimizationObjective,
        discard_critical: bool = True,
    ) -> List[Dict[str, Any]]:
        if not routes:
            return []

        weights = OBJECTIVE_WEIGHT_PRESETS.get(objective, OBJECTIVE_WEIGHT_PRESETS[OptimizationObjective.BALANCED])

        # Step 1: Discard blocked routes (e.g., major accident or road closure active) if we have alternatives
        viable_routes = []
        blocked_routes = []

        for r in routes:
            has_closure = r.get("road_closure_active", False) or r.get("external_event") == "⛔ Road Closed"
            # Major accidents block the road
            has_major_accident = r.get("accident_active", False) or "Major Accident" in r.get("external_event", "")
            
            if has_closure or has_major_accident:
                blocked_routes.append(r)
            else:
                viable_routes.append(r)

        # If all routes are blocked, we keep all of them so we have at least something to show,
        # but if we have viable alternatives, we discard the blocked ones.
        if viable_routes:
            final_candidates = viable_routes
        else:
            final_candidates = blocked_routes

        # Step 2: Extract raw objective metrics for normalization
        n = len(final_candidates)
        if n == 0:
            return []

        # Helper to safely extract lists for min/max
        def get_metric_list(key: str) -> List[float]:
            return [float(r.get(key, 0.0)) for r in final_candidates]

        raw_metrics = {
            "travel_time": get_metric_list("total_travel_time_s"),
            "cvar": get_metric_list("total_cvar_s"),
            "distance": get_metric_list("total_distance_m"),
            "congestion": get_metric_list("route_congestion"),
            "accidents": get_metric_list("risk_score"),
            "road_closures": [1.0 if r.get("road_closure_active", False) else 0.0 for r in final_candidates],
            "weather": get_metric_list("weather_severity"),
            "fuel": get_metric_list("fuel_cost_estimate"),
            "ev_energy": get_metric_list("ev_energy_kwh"),
            "safety": get_metric_list("safety_score"),
            "reliability": get_metric_list("reliability_score"),
            "confidence": get_metric_list("confidence_score"),
            "comfort": get_metric_list("driving_comfort_score"),
        }

        # Step 3: Compute min & max for each metric to perform Min-Max normalization
        bounds = {}
        for key, vals in raw_metrics.items():
            bounds[key] = {"min": min(vals), "max": max(vals)}

        # Step 4: Calculate composite cost and user-friendly optimization score
        # For cost-minimization criteria (travel_time, distance, congestion, accidents, closures, weather, fuel, ev_energy):
        #   normalized_cost = (val - min) / (max - min)  (if max > min, else 0)
        # For benefit-maximization criteria (safety, reliability, confidence, comfort):
        #   normalized_cost = (max - val) / (max - min)  (if max > min, else 0)
        scored_routes = []
        for idx, r in enumerate(final_candidates):
            composite_cost = 0.0
            
            for key, weight in weights.items():
                val = raw_metrics[key][idx]
                b_min = bounds[key]["min"]
                b_max = bounds[key]["max"]
                diff = b_max - b_min
                
                # Check if it's a benefit to maximize or a cost to minimize
                is_benefit = key in {"safety", "reliability", "confidence", "comfort"}
                
                if diff < 1e-5:
                    normalized_cost = 0.0
                else:
                    if is_benefit:
                        normalized_cost = (b_max - val) / diff
                    else:
                        normalized_cost = (val - b_min) / diff
                
                composite_cost += weight * normalized_cost

            # Convert cost to optimization score (0% to 100%, higher is better)
            opt_score = (1.0 - composite_cost) * 100
            
            # If the route is blocked, penalize it heavily in case it wasn't discarded
            is_blocked = r.get("road_closure_active", False) or r.get("external_event") == "⛔ Road Closed" or "Major Accident" in r.get("external_event", "")
            if is_blocked:
                opt_score = max(0.0, opt_score - 50.0)  # Huge penalty

            # Make a copy of route dict and update scores
            new_r = dict(r)
            new_r["optimization_score"] = round(opt_score, 1)
            scored_routes.append(new_r)

        # Step 5: Rank routes by optimization score descending (higher is better)
        ranked_routes = sorted(scored_routes, key=lambda x: x["optimization_score"], reverse=True)
        for rank_idx, r in enumerate(ranked_routes):
            r["rank"] = rank_idx + 1

        return ranked_routes


# ── Future Research-Grade Optimizers Plugs ────────────────────────────────────
# These placeholders showcase the system's modularity for Plug-and-Play algorithms.

class NSGAIIROptimizer(BaseRouteOptimizer):
    """
    Placeholder for Non-dominated Sorting Genetic Algorithm II.
    Used for finding Pareto-optimal front across conflicting objectives.
    """
    def optimize(self, routes: List[Dict[str, Any]], objective: OptimizationObjective, discard_critical: bool = True) -> List[Dict[str, Any]]:
        # Future implementation would sort routes into non-dominated fronts.
        # Fallback to standard weighted sum.
        return WeightedSumRouteOptimizer().optimize(routes, objective, discard_critical)


class AntColonyRouteOptimizer(BaseRouteOptimizer):
    """
    Placeholder for Ant Colony Optimization.
    Simulates chemical pheromone trails to explore grid-based path alternatives.
    """
    def optimize(self, routes: List[Dict[str, Any]], objective: OptimizationObjective, discard_critical: bool = True) -> List[Dict[str, Any]]:
        return WeightedSumRouteOptimizer().optimize(routes, objective, discard_critical)


class RLRouteOptimizer(BaseRouteOptimizer):
    """
    Placeholder for Q-Learning / Deep Reinforcement Learning optimizer.
    Finds optimal routing policy under dynamic Markov Decision Processes (MDP).
    """
    def optimize(self, routes: List[Dict[str, Any]], objective: OptimizationObjective, discard_critical: bool = True) -> List[Dict[str, Any]]:
        return WeightedSumRouteOptimizer().optimize(routes, objective, discard_critical)
