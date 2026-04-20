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

    traffic = TrafficCondition(
        segment_id="global",
        congestion_index=traffic_data.get("congestion_index", 0.5),
        avg_speed_kph=traffic_data.get("avg_speed_kph", 65),
        incident_active=traffic_data.get("incident_active", False),
    )
    weather = WeatherCondition(
        condition=weather_data.get("condition", "clear"),
        severity=weather_data.get("severity", 0),
        temperature_c=weather_data.get("temperature_c", 20),
        visibility_km=weather_data.get("visibility_km", 10),
    )

    # Generate routes with real coordinate segments
    routes = []
    for i in range(request.num_alternatives):
        jitter = 1.0 + (i * 0.15)
        factor = predicted_factor * jitter

        dist_km = _haversine(
            request.origin.latitude, request.origin.longitude,
            request.destination.latitude, request.destination.longitude,
        )
        dist_m = dist_km * 1000

        base_speed = traffic_data.get("avg_speed_kph", 65)
        if base_speed <= 0:
            base_speed = 65
        base_time_s = (dist_m / 1000) / base_speed * 3600
        travel_time_s = base_time_s * factor

        mean_t, ci_low, ci_up = monte_carlo_travel_time(base_time_s, factor)

        route_congestion = features.context.congestion_index + i * 0.08
        route_congestion += max(0.0, features.context.historical_congestion - features.context.congestion_index) * 0.15
        route_congestion += (1.0 - features.context.speed_reliability) * 0.10
        route_congestion = max(0.0, min(1.0, route_congestion))

        risk_score = features.context.road_risk_score + i * 0.05
        risk_score = min(1.0, risk_score)
        n_segments = max(5, min(50, int(dist_km / 2)))  # 1 segment per 2km
        seg_dist = dist_m / n_segments
        segments = []
        
        # Add slight lateral offset for alternative routes
        lat_offset_factor = (i - 1) * 0.002  # offset for alternatives
        
        for s in range(n_segments):
            seg_time = travel_time_s / n_segments
            t_start = s / n_segments
            t_end = (s + 1) / n_segments
            t_mid = (t_start + t_end) / 2
            curve_offset = lat_offset_factor * math.sin(math.pi * t_mid)
            
            start_lat = request.origin.latitude + (request.destination.latitude - request.origin.latitude) * t_start
            start_lon = request.origin.longitude + (request.destination.longitude - request.origin.longitude) * t_start
            end_lat = request.origin.latitude + (request.destination.latitude - request.origin.latitude) * t_end
            end_lon = request.origin.longitude + (request.destination.longitude - request.origin.longitude) * t_end
            
            # Apply offset to create different paths
            start_lat += curve_offset
            start_lon -= curve_offset * 0.3
            end_lat += curve_offset
            end_lon -= curve_offset * 0.3
            
            segments.append(RouteSegment(
                segment_id=f"seg_{i}_{s}",
                start_node=s,
                end_node=s + 1,
                start_lat=start_lat,
                start_lon=start_lon,
                end_lat=end_lat,
                end_lon=end_lon,
                distance_m=round(seg_dist, 1),
                predicted_travel_time_s=round(seg_time, 1),
                speed_kph=round(base_speed * (1 - 0.5 * route_congestion), 1),
                congestion_index=round(
                    max(0.0, min(1.0, route_congestion + random.gauss(0, 0.03))),
                    3,
                ),
                risk_score=round(
                    max(0.0, min(1.0, risk_score + random.gauss(0, 0.02))),
                    3,
                ),
            ))

        reliability = max(0, min(1, 1 - risk_score))
        emissions = estimate_emissions(dist_m, factor)
        fuel = estimate_fuel_cost(dist_m, factor)
        traffic_color = get_congestion_color(route_congestion)
        traffic_level = _route_traffic_level(route_congestion)
        traffic_reasoning = _build_traffic_reasoning(
            congestion_index=route_congestion,
            historical_congestion=features.context.historical_congestion,
            speed_reliability=features.context.speed_reliability,
            historical_speed_kph=features.context.historical_speed_kph,
            current_speed_kph=base_speed * (1 - 0.5 * route_congestion),
            is_peak_hour=features.temporal.is_peak_hour,
            weather_severity=features.context.weather_severity,
        )

        routes.append(RouteResult(
            route_id=str(uuid.uuid4())[:8],
            segments=segments,
            total_distance_m=round(dist_m, 1),
            total_travel_time_s=round(mean_t, 1),
            total_travel_time_display=_format_duration(mean_t),
            confidence_interval_lower_s=round(ci_low, 1),
            confidence_interval_upper_s=round(ci_up, 1),
            risk_level=_classify_risk(risk_score),
            reliability_score=round(reliability, 3),
            emissions_g_co2=round(emissions, 1),
            fuel_cost_estimate=round(fuel, 2),
            traffic_color=traffic_color,
            traffic_level=traffic_level,
            traffic_reasoning=traffic_reasoning,
            rank=i + 1,
        ))

    return RouteResponse(
        request_id=str(uuid.uuid4()),
        routes=routes,
        selected_route_index=0,
        traffic=traffic,
        weather=weather,
        prediction_meta=pred_meta,
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
