"""
Feature builder — orchestrates temporal, spatial, context, Indian calendar,
and historical profile feature extraction, producing a CombinedFeatureVector
ready for model input.

Enhanced with:
  - Indian calendar features (festivals, monsoon, school hours)
  - Historical speed profiles from time-series DB
  - Real-time traffic integration
"""

import numpy as np
from datetime import datetime

from app.schemas import CombinedFeatureVector
from app.data_pipeline.traffic_loader import get_traffic
from app.data_pipeline.weather_loader import get_weather
from app.features.temporal_features import temporal_features
from app.features.spatial_features import spatial_features
from app.features.context_features import context_features
from app.features.indian_calendar import compute_indian_calendar_features

from loguru import logger


def build_features(
    departure_time: str | None = None,
    segment_length_m: float = 500.0,
    speed_limit_kph: float = 50.0,
    num_lanes: int = 2,
    road_type: str = "residential",
    elevation_change_m: float = 0.0,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
    edge_id: str | None = None,
) -> CombinedFeatureVector:
    """
    Build a complete CombinedFeatureVector from request parameters + live data.

    Parameters
    ----------
    departure_time : ISO-format datetime string; defaults to now
    segment_length_m : road segment length
    speed_limit_kph : posted speed limit
    num_lanes : number of lanes
    road_type : OSM road type tag
    elevation_change_m : elevation delta
    origin_lat : origin latitude (for weather API)
    origin_lon : origin longitude (for weather API)
    edge_id : optional road edge ID for historical profile lookup
    """
    if departure_time:
        dt = datetime.fromisoformat(departure_time)
    else:
        dt = datetime.now()

    day_name = dt.strftime("%A")
    time_str = dt.strftime("%H:%M")

    # ── Temporal features ─────────────────────────────────────────────────
    temporal = temporal_features(day_name, time_str)

    # ── Indian calendar features ──────────────────────────────────────────
    indian = compute_indian_calendar_features(dt)

    # Merge Indian features into temporal
    temporal.is_festival = indian.is_festival
    temporal.festival_severity = indian.festival_severity
    temporal.is_monsoon_season = indian.is_monsoon_season
    temporal.monsoon_severity = indian.monsoon_severity
    temporal.is_school_hours = indian.is_school_hours
    temporal.is_market_day = indian.is_market_day or indian.is_weekend_market

    # ── Spatial features ──────────────────────────────────────────────────
    spatial = spatial_features(
        length_m=segment_length_m,
        speed_limit_kph=speed_limit_kph,
        num_lanes=num_lanes,
        road_type=road_type,
        elevation_change_m=elevation_change_m,
    )

    # ── Context features (live traffic + weather) ─────────────────────────
    traffic_data = get_traffic(hour=dt.hour)
    weather_data = get_weather(lat=origin_lat, lon=origin_lon)

    ctx = context_features(traffic_data, weather_data)

    # ── Historical context from time-series DB ────────────────────────────
    try:
        from app.features.historical_profiles import get_historical_context
        hist_ctx = get_historical_context(
            edge_id=edge_id or "default",
            hour=dt.hour,
            day_of_week=dt.weekday(),
        )
        if hist_ctx.get("has_data"):
            ctx.historical_speed_kph = hist_ctx["historical_speed_kph"] or 40.0
            ctx.historical_congestion = hist_ctx["historical_congestion"] or 0.3
            ctx.speed_reliability = hist_ctx["reliability"] or 0.5
    except Exception as e:
        logger.debug(f"Historical context lookup skipped: {e}")

    # ── Apply Indian calendar to risk score ────────────────────────────────
    from app.features.indian_calendar import get_traffic_multiplier
    calendar_multiplier = get_traffic_multiplier(indian)

    if calendar_multiplier > 1.0:
        # Festival/monsoon/school increases risk
        risk_boost = (calendar_multiplier - 1.0) * 0.3
        ctx.road_risk_score = min(1.0, ctx.road_risk_score + risk_boost)

    return CombinedFeatureVector(
        temporal=temporal,
        spatial=spatial,
        context=ctx,
    )


def build_features_array(features: CombinedFeatureVector) -> np.ndarray:
    """Convert a CombinedFeatureVector to a numpy array shaped (1, n_features)."""
    return np.array(features.to_flat_list(), dtype=np.float32).reshape(1, -1)