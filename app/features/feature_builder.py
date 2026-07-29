
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
    if departure_time:
        dt = datetime.fromisoformat(departure_time)
    else:
        dt = datetime.now()

    day_name = dt.strftime("%A")
    time_str = dt.strftime("%H:%M")
    temporal = temporal_features(day_name, time_str)

    indian = compute_indian_calendar_features(dt)

    temporal.is_festival = indian.is_festival
    temporal.festival_severity = indian.festival_severity
    temporal.is_monsoon_season = indian.is_monsoon_season
    temporal.monsoon_severity = indian.monsoon_severity
    temporal.is_school_hours = indian.is_school_hours
    temporal.is_market_day = indian.is_market_day or indian.is_weekend_market

    spatial = spatial_features(
        length_m=segment_length_m,
        speed_limit_kph=speed_limit_kph,
        num_lanes=num_lanes,
        road_type=road_type,
        elevation_change_m=elevation_change_m,
    )

    traffic_data = get_traffic(hour=dt.hour)
    weather_data = get_weather(lat=origin_lat, lon=origin_lon)

    ctx = context_features(traffic_data, weather_data)

    road_type_map = {
        "motorway": 0.0,
        "trunk": 0.5,
        "primary": 1.0,
        "secondary": 2.0,
        "tertiary": 2.5,
        "residential": 3.0,
        "service": 3.5,
    }
    rt = (road_type or "residential").lower()
    ctx.road_type_encoded = road_type_map.get(rt, 2.0)
    ctx.highway_percentage = max(0.0, min(1.0, 1.0 - (ctx.road_type_encoded / 4.0)))
    ctx.route_curvature = round(max(0.01, abs(elevation_change_m) / max(segment_length_m, 1.0)), 4)
    ctx.intersection_count = round(max(1.0, segment_length_m / 120.0), 2)
    ctx.toll_roads = 1.0 if (ctx.highway_percentage > 0.7 and segment_length_m > 20000) else 0.0
    ctx.urban_density = max(0.0, min(1.0, 0.2 + (ctx.road_type_encoded / 4.0)))
    if segment_length_m < 5_000:
        ctx.distance_category = 0.0
    elif segment_length_m < 30_000:
        ctx.distance_category = 1.0
    elif segment_length_m < 120_000:
        ctx.distance_category = 2.0
    else:
        ctx.distance_category = 3.0

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

    from app.features.indian_calendar import get_traffic_multiplier
    calendar_multiplier = get_traffic_multiplier(indian)

    if calendar_multiplier > 1.0:
        risk_boost = (calendar_multiplier - 1.0) * 0.3
        ctx.road_risk_score = min(1.0, ctx.road_risk_score + risk_boost)

    return CombinedFeatureVector(
        temporal=temporal,
        spatial=spatial,
        context=ctx,
    )


def build_features_array(features: CombinedFeatureVector) -> np.ndarray:
    return np.array(features.to_flat_list(), dtype=np.float32).reshape(1, -1)