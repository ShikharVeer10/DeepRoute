"""
DeepRoute — Route Planning API Layer
Endpoints:
  POST /api/route           — Plan a route with ML/DL predictions
  POST /api/alternatives    — Generate alternative routes explicitly
  POST /api/forecast        — Forecast travel times for future windows
  POST /api/risk            — Risk assessment for a route
  GET  /api/models          — List registered models
  POST /api/recommend       — Get AI-powered route recommendation
  POST /api/travel_data/collect — Actual travel data collection (closed loop)
  GET  /api/health          — Health check
"""

import uuid
import random
import math
import numpy as np
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.schemas import (
    RouteRequest,
    ForecastRequest,
    RiskAssessmentRequest,
    TripCompletionRequest,
    RouteResponse,
    ForecastResponse,
    ForecastResult,
    RiskAssessmentResponse,
    RiskSegment,
    TrafficCondition,
    WeatherCondition,
    ModelRegistryResponse,
    RouteRecommendation,
    RiskLevel,
    Coordinate,
    RouteSegment,
    RouteResult,
)
from app.features.feature_builder import build_features, build_features_array
from app.models.inference import predict
from app.models.model_registry import list_models
from app.data_pipeline.traffic_loader import get_traffic
from app.data_pipeline.weather_loader import get_weather
from app.agents.route_agent import RouteContext, build_recommendation_from_data
from app.routing.router import monte_carlo_travel_time, _classify_risk, build_route_result, _format_duration
from app.routing.edge_weight_builder import estimate_emissions, estimate_fuel_cost
from app.features.historical_profiles import get_congestion_color


router = APIRouter(prefix="/api", tags=["routing"])


def _route_traffic_level(congestion_index: float) -> str:
    if congestion_index < 0.30:
        return "low"
    if congestion_index < 0.45:
        return "moderate"
    if congestion_index < 0.60:
        return "high"
    return "severe"


def _build_traffic_reasoning(
    congestion_index: float,
    historical_congestion: float,
    speed_reliability: float,
    historical_speed_kph: float,
    current_speed_kph: float,
    is_peak_hour: bool,
    weather_severity: float,
) -> str:
    trend = "similar to" if abs(congestion_index - historical_congestion) < 0.08 else (
        "higher than" if congestion_index > historical_congestion else "lower than"
    )
    reliability_note = (
        "movement is highly repeatable" if speed_reliability >= 0.7
        else "movement varies day-to-day"
    )
    speed_note = (
        "current speed is below historical typical speed"
        if current_speed_kph < historical_speed_kph else
        "current speed is near/above historical typical speed"
    )
    peak_note = "peak-hour load is active" if is_peak_hour else "off-peak timing helps flow"
    weather_note = "weather adds extra delay risk" if weather_severity >= 0.35 else "weather impact is minor"
    return (
        f"Predicted congestion is {trend} the usual pattern for this time/day; "
        f"{reliability_note}; {speed_note}; {peak_note}; {weather_note}."
    )


# ─── Health ───────────────────────────────────────────────────────────────────


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DeepRoute",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


# ─── Route Planning ──────────────────────────────────────────────────────────


@router.post("/route", response_model=RouteResponse)
def plan_route(request: RouteRequest):
    """
    Plan a route using ML/DL predicted travel-time factors with coordinate-based paths.

    The response includes multiple alternative routes ranked by composite cost,
    along with traffic/weather snapshots and prediction metadata.
    """
    from app.routing.router import plan_intelligent_routes
    
    result = plan_intelligent_routes(
        origin_lat=request.origin.latitude,
        origin_lon=request.origin.longitude,
        dest_lat=request.destination.latitude,
        dest_lon=request.destination.longitude,
        model_type=request.model_type,
        objective=request.objective,
        num_alternatives=request.num_alternatives,
        departure_time=request.departure_time,
        consider_weather=request.consider_weather,
        consider_incidents=request.consider_incidents,
    )

    return RouteResponse(
        request_id=str(uuid.uuid4()),
        routes=result["routes"],
        selected_route_index=0,
        traffic=result["traffic"],
        weather=result["weather"],
        prediction_meta=result["prediction_meta"],
        computed_at=datetime.now().isoformat(),
    )

@router.post("/forecast", response_model=ForecastResponse)
def forecast_travel_time(request: ForecastRequest):
    """
    Forecast travel times for multiple future time windows.
    """
    features = build_features(
        origin_lat=request.origin.latitude,
        origin_lon=request.origin.longitude,
    )
    predicted_factor, pred_meta = predict(features, request.model_type)

    forecasts = []
    window_offsets = {"15min": 0.05, "30min": 0.10, "1h": 0.18, "2h": 0.30}

    for window in request.forecast_windows:
        offset = window_offsets.get(window, 0.10)

        dist_km = _haversine(
            request.origin.latitude, request.origin.longitude,
            request.destination.latitude, request.destination.longitude,
        )
        base_time = (dist_km / 65) * 3600
        factor = predicted_factor + offset
        travel_time = base_time * factor

        _, ci_low, ci_up = monte_carlo_travel_time(base_time, factor)

        forecasts.append(ForecastResult(
            window=window,
            predicted_travel_time_s=round(travel_time, 1),
            confidence_lower_s=round(ci_low, 1),
            confidence_upper_s=round(ci_up, 1),
            expected_congestion=round(min(1.0, features.context.congestion_index + offset), 3),
        ))

    return ForecastResponse(
        request_id=str(uuid.uuid4()),
        origin=request.origin,
        destination=request.destination,
        forecasts=forecasts,
        model_used=pred_meta.model_used,
        computed_at=datetime.now().isoformat(),
    )

@router.post("/risk", response_model=RiskAssessmentResponse)
def assess_risk(request: RiskAssessmentRequest):
    """
    Assess route risk based on current traffic, weather, and model predictions.
    """
    features = build_features(
        departure_time=request.departure_time,
        origin_lat=request.origin.latitude,
        origin_lon=request.origin.longitude,
    )
    risk_score = features.context.road_risk_score

    risk_factors = []
    if features.context.congestion_index > 0.6:
        risk_factors.append("High congestion")
    if features.context.weather_severity > 0.4:
        risk_factors.append("Adverse weather conditions")
    if features.context.incident_proximity < 3:
        risk_factors.append("Nearby incident reported")
    if features.temporal.is_peak_hour:
        risk_factors.append("Peak hour traffic")

    if risk_score < 0.25:
        overall = RiskLevel.LOW
    elif risk_score < 0.50:
        overall = RiskLevel.MEDIUM
    elif risk_score < 0.75:
        overall = RiskLevel.HIGH
    else:
        overall = RiskLevel.CRITICAL

    tolerance_map = {
        RiskLevel.LOW: 0.25,
        RiskLevel.MEDIUM: 0.50,
        RiskLevel.HIGH: 0.75,
        RiskLevel.CRITICAL: 1.0,
    }
    requested_tolerance = tolerance_map.get(request.risk_tolerance, 0.5)
    safer_available = risk_score > requested_tolerance

    recommendations = []
    if risk_score > 0.5:
        recommendations.append("Consider delaying departure by 30-60 minutes")
    if features.context.weather_severity > 0.4:
        recommendations.append("Enable headlights and reduce speed")
    if features.context.congestion_index > 0.6:
        recommendations.append("Check alternative routes for less congestion")
    if not recommendations:
        recommendations.append("Conditions are favorable — proceed as planned")

    segments = [
        RiskSegment(
            segment_id="route_overview",
            risk_level=overall,
            risk_score=round(risk_score, 3),
            risk_factors=risk_factors if risk_factors else ["None identified"],
        )
    ]

    return RiskAssessmentResponse(
        request_id=str(uuid.uuid4()),
        overall_risk=overall,
        overall_risk_score=round(risk_score, 3),
        segments=segments,
        recommendations=recommendations,
        safer_alternative_available=safer_available,
    )


# ─── Models Registry ─────────────────────────────────────────────────────────


@router.get("/models", response_model=ModelRegistryResponse)
def get_models():
    """List all registered models."""
    return list_models()

@router.post("/recommend", response_model=RouteRecommendation)
def get_recommendation(request: RouteRequest):
    """
    Get an AI-powered route recommendation based on current conditions.
    Uses pydantic-ai structured output with rule-based fallback.
    """
    features = build_features(
        departure_time=request.departure_time,
        origin_lat=request.origin.latitude,
        origin_lon=request.origin.longitude,
    )
    predicted_factor, pred_meta = predict(features, request.model_type)

    traffic_data = get_traffic()
    weather_data = get_weather(
        lat=request.origin.latitude,
        lon=request.origin.longitude,
    )

    dist_km = _haversine(
        request.origin.latitude, request.origin.longitude,
        request.destination.latitude, request.destination.longitude,
    )
    base_time_min = (dist_km / 65) * 60
    travel_time_min = base_time_min * predicted_factor
    _, ci_low, ci_up = monte_carlo_travel_time(base_time_min * 60, predicted_factor)

    context = RouteContext(
        origin_name=f"({request.origin.latitude:.4f}, {request.origin.longitude:.4f})",
        destination_name=f"({request.destination.latitude:.4f}, {request.destination.longitude:.4f})",
        total_distance_km=round(dist_km, 1),
        predicted_travel_time_min=round(travel_time_min, 1),
        confidence_lower_min=round(ci_low / 60, 1),
        confidence_upper_min=round(ci_up / 60, 1),
        congestion_index=features.context.congestion_index,
        weather_condition=weather_data["condition"],
        weather_severity=weather_data["severity"],
        risk_level=_classify_risk(features.context.road_risk_score).value,
        reliability_score=max(0, 1 - features.context.road_risk_score),
        num_alternatives=request.num_alternatives,
        departure_time=request.departure_time or datetime.now().isoformat(),
        model_used=pred_meta.model_used,
    )

    return build_recommendation_from_data(context)


# ─── Alternatives ────────────────────────────────────────────────────────────


@router.post("/alternatives", response_model=RouteResponse)
def get_alternative_routes(request: RouteRequest):
    """
    Explicit endpoint for generating alternative routes.
    Forces num_alternatives to be at least 3 to ensure options are provided.
    """
    if request.num_alternatives < 3:
        request.num_alternatives = 3
    return plan_route(request)


# ─── Data Collection ─────────────────────────────────────────────────────────


@router.post("/travel_data/collect")
def collect_actual_travel_data(request: TripCompletionRequest):
    """
    Actual Travel Data Collection endpoint.
    Clients (Mobile, Web, Fleet, Logistics API) use this to report actual 
    travel times, closing the continuous learning loop for the ML models.
    """
    try:
        from app.storage.database import complete_trip
        complete_trip(request.trip_id, request.actual_travel_time_s)
        return {"status": "success", "message": "Travel data collected for continuous learning."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    import math
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
