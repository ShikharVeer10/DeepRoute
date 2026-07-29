"""Route candidate generation and route-specific ETA evaluation."""
from __future__ import annotations

import math
import os
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import requests

try:
    from dotenv import load_dotenv
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

from app.data_pipeline.tomtom_traffic import traffic_for_route
from app.data_pipeline.weather_loader import get_weather
from app.features.feature_builder import build_features
from app.models.inference import predict
from app.routing.edge_weight_builder import estimate_emissions, estimate_fuel_cost
from app.routing.optimizer import WeightedSumRouteOptimizer
from app.schemas import (
    CombinedFeatureVector,
    ContextFeatures,
    ModelType,
    OptimizationObjective,
    RouteResult,
    RouteSegment,
    RiskLevel,
    SpatialFeatures,
    TrafficCondition,
    WeatherCondition,
)


_TOMTOM_ROUTE_URL = (
    "https://api.tomtom.com/routing/1/calculateRoute/"
    "{origin_lat},{origin_lon}:{dest_lat},{dest_lon}/json"
)
_OSRM_ROUTE_URL = (
    "https://router.project-osrm.org/route/v1/driving/"
    "{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
)


def _haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return 6371000 * 2 * math.asin(
        math.sqrt(
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
    )


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    if hours:
        return f"{hours}h {minutes:02d}min"
    return f"{minutes}min"


def _classify_risk(score: float) -> RiskLevel:
    if score < 0.25:
        return RiskLevel.LOW
    if score < 0.5:
        return RiskLevel.MEDIUM
    if score < 0.75:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


def monte_carlo_cvar_travel_time(
    base_time_s: float,
    predicted_factor: float,
    n_simulations: int = 1000,
    noise_std: float = 0.05,
    alpha: float = 0.95,
) -> tuple[float, float, float, float]:
    """Deterministic travel-time envelope used for display and ranking."""
    mean_t = float(max(0.0, base_time_s * predicted_factor))
    spread = max(0.03, min(0.18, noise_std * 1.4 + abs(predicted_factor - 1.0) * 0.05))
    lower = mean_t * (1.0 - spread)
    upper = mean_t * (1.0 + spread * 1.1)
    tail = mean_t * (1.0 + spread * (1.0 + (1.0 - alpha)))
    return mean_t, lower, upper, tail


def _sample(points: list[list[float]], count: int = 100) -> list[tuple[int, int]]:
    if not points:
        return []
    indexes = np.linspace(0, len(points) - 1, min(count, len(points))).astype(int)
    return [(round(points[i][0] * 1e5), round(points[i][1] * 1e5)) for i in indexes]


def geometry_similarity(a: list[list[float]], b: list[list[float]]) -> float:
    sa, sb = set(_sample(a)), set(_sample(b))
    if not sa or not sb:
        return 1.0
    matched = sum(any(abs(x - u) <= 1 and abs(y - v) <= 1 for u, v in sb) for x, y in sa)
    return matched / max(1, min(len(sa), len(sb)))


def _polyline_decode(encoded: str, precision: int = 5) -> list[list[float]]:
    index = 0
    lat = 0
    lon = 0
    coordinates: list[list[float]] = []
    factor = 10**precision

    while index < len(encoded):
        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if result & 1 else (result >> 1)
        lat += dlat

        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlon = ~(result >> 1) if result & 1 else (result >> 1)
        lon += dlon

        coordinates.append([lat / factor, lon / factor])

    return coordinates


def _extract_coords(route: dict[str, Any]) -> list[list[float]]:
    coords: list[list[float]] = []

    for leg in route.get("legs", []) or []:
        for point in leg.get("points", []) or []:
            lat = point.get("latitude", point.get("lat"))
            lon = point.get("longitude", point.get("lon"))
            if lat is not None and lon is not None:
                coords.append([float(lat), float(lon)])

    if not coords:
        points = route.get("points") or route.get("geometry", {}).get("coordinates", [])
        for point in points or []:
            if isinstance(point, dict):
                lat = point.get("latitude", point.get("lat"))
                lon = point.get("longitude", point.get("lon"))
                if lat is not None and lon is not None:
                    coords.append([float(lat), float(lon)])
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                # GeoJSON is [lon, lat]
                if abs(point[0]) <= 180 and abs(point[1]) <= 90:
                    coords.append([float(point[1]), float(point[0])])
                else:
                    coords.append([float(point[0]), float(point[1])])

    if not coords:
        encoded = route.get("encodedPolyline") or route.get("route", {}).get("encodedPolyline")
        if isinstance(encoded, str) and encoded:
            coords = _polyline_decode(encoded)

    deduped: list[list[float]] = []
    for lat, lon in coords:
        if not deduped or abs(deduped[-1][0] - lat) > 1e-7 or abs(deduped[-1][1] - lon) > 1e-7:
            deduped.append([lat, lon])
    return deduped


def _route_bbox(coords: list[list[float]]) -> tuple[float, float, float, float]:
    lats = [p[0] for p in coords]
    lons = [p[1] for p in coords]
    return min(lats), min(lons), max(lats), max(lons)


def _turn_count(coords: list[list[float]]) -> int:
    headings = []
    for a, b in zip(coords, coords[1:]):
        dy, dx = b[0] - a[0], (b[1] - a[1]) * math.cos(math.radians(a[0]))
        if dx or dy:
            headings.append(math.atan2(dy, dx))
    return sum(
        abs((b - a + math.pi) % (2 * math.pi) - math.pi) > math.radians(25)
        for a, b in zip(headings, headings[1:])
    )


def _avg_frc_score(flow: dict[str, Any]) -> float:
    return float(flow.get("road_class_score", 0.5))


def _route_flow_fallback(route: dict[str, Any]) -> dict[str, Any]:
    summary = route.get("summary", {}) or {}
    duration = float(summary.get("travelTimeInSeconds") or route.get("duration") or 1.0)
    freeflow = float(summary.get("noTrafficTravelTimeInSeconds") or duration)
    distance = float(summary.get("lengthInMeters") or route.get("distance") or 1.0)
    avg_speed = max(5.0, distance / max(duration, 1.0) * 3.6)
    free_speed = max(avg_speed, distance / max(freeflow, 1.0) * 3.6)
    congestion = max(0.0, min(1.0, 1.0 - avg_speed / max(free_speed, 1.0)))
    return {
        "congestion_index": congestion,
        "avg_speed_kph": avg_speed,
        "free_flow_speed_kph": free_speed,
        "current_travel_time_s": duration,
        "free_flow_travel_time_s": freeflow,
        "road_class_score": 0.5,
        "incident_count": 0,
        "construction_count": 0,
        "accident_count": 0,
        "road_closure_count": 0,
        "incident_active": False,
        "incident_markers": [],
        "source": "route_summary",
        "live_coverage": 0.0,
    }


def _tomtom_routes(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    limit: int,
    departure_time: str | None,
) -> list[dict[str, Any]]:
    key = os.getenv("TOMTOM_API_KEY")
    if not key:
        return []

    profiles = [
        {"routeType": "fastest", "avoid": None},
        {"routeType": "eco", "avoid": None},
        {"routeType": "shortest", "avoid": None},
        {"routeType": "fastest", "avoid": "tollRoads"},
        {"routeType": "fastest", "avoid": "motorways"},
    ]

    accepted: list[dict[str, Any]] = []
    max_alts = max(0, min(4, limit - 1))

    for profile in profiles:
        params = {
            "key": key,
            "routeType": profile["routeType"],
            "travelMode": "car",
            "traffic": "true",
            "alternativeType": "anyRoute",
            "maxAlternatives": max_alts,
            "routeRepresentation": "polyline",
            "computeTravelTimeFor": "all",
            "sectionType": "traffic",
        }
        if departure_time:
            params["departAt"] = departure_time
        if profile["avoid"]:
            params["avoid"] = profile["avoid"]

        try:
            response = requests.get(
                _TOMTOM_ROUTE_URL.format(
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    dest_lat=dest_lat,
                    dest_lon=dest_lon,
                ),
                params=params,
                timeout=12,
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            continue

        for route in data.get("routes", []) or []:
            coords = _extract_coords(route)
            if len(coords) < 2:
                continue
            summary = route.get("summary", {}) or {}
            distance = float(summary.get("lengthInMeters") or route.get("distance") or 0.0)
            duration = float(summary.get("travelTimeInSeconds") or route.get("duration") or 0.0)
            if distance <= 0 or duration <= 0:
                continue
            normalized = dict(route)
            normalized["_latlngs"] = coords
            normalized["distance"] = distance
            normalized["duration"] = duration
            normalized["source"] = "tomtom"
            normalized["route_type"] = profile["routeType"]
            normalized["avoid"] = profile["avoid"]
            normalized["summary"] = summary
            if all(geometry_similarity(coords, existing["_latlngs"]) < 0.90 for existing in accepted):
                accepted.append(normalized)
            if len(accepted) >= limit:
                return accepted

    return accepted


def _osrm_routes(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    limit: int,
) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            _OSRM_ROUTE_URL.format(
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                dest_lat=dest_lat,
                dest_lon=dest_lon,
            )
            + "?overview=full&geometries=geojson&steps=true&annotations=true&alternatives=true",
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    accepted: list[dict[str, Any]] = []
    for route in data.get("routes", []) or []:
        coords = _extract_coords(route)
        distance = float(route.get("distance", 0.0))
        duration = float(route.get("duration", 0.0))
        if len(coords) < 2 or distance <= 0 or duration <= 0:
            continue
        normalized = dict(route)
        normalized["_latlngs"] = coords
        normalized["source"] = "osrm"
        normalized["route_type"] = "fastest"
        if all(geometry_similarity(coords, existing["_latlngs"]) < 0.90 for existing in accepted):
            accepted.append(normalized)
        if len(accepted) >= limit:
            break
    return accepted


def fetch_diverse_routes(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    limit: int = 3,
    departure_time: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch real route alternatives and reject near-identical geometries."""
    accepted = _tomtom_routes(origin_lat, origin_lon, dest_lat, dest_lon, limit, departure_time)
    if len(accepted) < limit:
        accepted.extend(
            route
            for route in _osrm_routes(origin_lat, origin_lon, dest_lat, dest_lon, limit)
            if all(geometry_similarity(route["_latlngs"], existing["_latlngs"]) < 0.90 for existing in accepted)
        )
    return accepted[:limit]


def _route_features(
    route: dict[str, Any],
    departure: str,
    traffic: dict[str, Any],
    weather: dict[str, Any],
    origin_lat: float,
    origin_lon: float,
) -> tuple[CombinedFeatureVector, dict[str, float], list[dict[str, Any]]]:
    coords = route["_latlngs"]
    summary = route.get("summary", {}) or {}
    distance_m = float(summary.get("lengthInMeters") or route.get("distance") or 0.0)
    live_duration_s = float(summary.get("travelTimeInSeconds") or route.get("duration") or 0.0)
    no_traffic_duration_s = float(summary.get("noTrafficTravelTimeInSeconds") or live_duration_s)
    traffic_delay_s = float(summary.get("trafficDelayInSeconds") or max(0.0, live_duration_s - no_traffic_duration_s))

    flow = traffic_for_route(coords, _route_flow_fallback(route))
    turns = _turn_count(coords)
    curvature = turns / max(distance_m / 1000.0, 1.0)
    avg_speed = float(flow.get("avg_speed_kph", max(20.0, distance_m / max(live_duration_s, 1.0) * 3.6)))
    free_speed = float(flow.get("free_flow_speed_kph", max(avg_speed, distance_m / max(no_traffic_duration_s, 1.0) * 3.6)))
    congestion = float(flow.get("congestion_index", 0.3))
    road_class_score = _avg_frc_score(flow)
    highway_percentage = max(0.0, min(1.0, road_class_score))
    road_type = "motorway" if road_class_score >= 0.8 else "primary" if road_class_score >= 0.6 else "secondary" if road_class_score >= 0.4 else "tertiary"

    base = build_features(
        departure_time=departure,
        segment_length_m=distance_m,
        speed_limit_kph=max(20.0, free_speed),
        road_type=road_type,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        edge_id=f"tomtom:{route.get('route_type', 'fastest')}:{round(coords[len(coords)//2][0], 3)}:{round(coords[len(coords)//2][1], 3)}",
    )

    context = ContextFeatures(
        congestion_index=congestion,
        weather_severity=float(weather.get("severity", 0.0)),
        incident_proximity=float(flow.get("incident_proximity_km", 99.0)),
        event_proximity=99.0,
        road_risk_score=min(
            1.0,
            0.28 * congestion
            + 0.22 * float(weather.get("severity", 0.0))
            + 0.18 * (1.0 - road_class_score)
            + 0.10 * min(1.0, turns / 25.0),
        ),
        road_closure_active=bool(flow.get("road_closure_count", 0)),
        roadworks_active=bool(flow.get("construction_count", 0)),
        accident_active=bool(flow.get("accident_count", 0)),
        historical_speed_kph=max(20.0, free_speed),
        historical_congestion=max(0.0, min(1.0, congestion * 0.9)),
        speed_reliability=max(0.2, 1.0 - congestion * 0.45 - curvature * 0.05),
        road_type_encoded={"motorway": 0.0, "primary": 1.0, "secondary": 2.0, "tertiary": 2.5}.get(road_type, 2.0),
        highway_percentage=highway_percentage,
        route_curvature=curvature,
        intersection_count=float(turns),
        toll_roads=float(route.get("avoid") == "tollRoads"),
        urban_density=min(1.0, max(0.05, 0.15 + (1.0 - road_class_score) * 0.65 + min(0.2, turns / 60.0))),
        distance_category=min(3.0, distance_m / 120000.0),
    )

    features = CombinedFeatureVector(
        temporal=base.temporal,
        spatial=SpatialFeatures(
            length_m=distance_m,
            speed_limit_kph=max(20.0, free_speed),
            road_type=road_type,
            elevation_change_m=0.0,
        ),
        context=context,
    )

    road_quality_score = max(
        0.0,
        min(
            1.0,
            0.40 * road_class_score
            + 0.25 * highway_percentage
            + 0.20 * context.speed_reliability
            + 0.15 * max(0.0, 1.0 - congestion),
        ),
    )

    step_sections = route.get("sections", []) or []
    toll_count = sum(1 for s in step_sections if "toll" in str(s).lower())

    metrics = {
        "distance_m": distance_m,
        "live_duration_s": live_duration_s,
        "no_traffic_duration_s": no_traffic_duration_s,
        "traffic_delay_s": traffic_delay_s,
        "turns": float(turns),
        "highway_pct": highway_percentage,
        "road_class_score": road_class_score,
        "road_quality_score": road_quality_score,
        "congestion": congestion,
        "avg_speed_kph": avg_speed,
        "free_flow_speed_kph": free_speed,
        "incident_count": float(flow.get("incident_count", 0)),
        "construction_count": float(flow.get("construction_count", 0)),
        "accident_count": float(flow.get("accident_count", 0)),
        "road_closure_count": float(flow.get("road_closure_count", 0)),
        "toll_count": float(toll_count),
    }
    return features, metrics, []


def plan_routes(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    model_type: ModelType,
    objective: OptimizationObjective,
    num_alternatives: int,
    departure_time: str | None,
    supplied_routes: list[dict[str, Any]] | None = None,
    consider_weather: bool = True,
    consider_incidents: bool = True,
) -> dict[str, Any]:
    departure = departure_time or datetime.now().astimezone().isoformat()
    weather = get_weather(lat=origin_lat, lon=origin_lon) if consider_weather else {"condition": "clear", "severity": 0.0, "temperature_c": 25.0, "visibility_km": 10.0, "source": "disabled"}
    global_traffic = {"congestion_index": 0.3, "avg_speed_kph": 40.0, "incident_active": False, "incident_proximity_km": 99.0}

    raw_routes = supplied_routes or fetch_diverse_routes(
        origin_lat,
        origin_lon,
        dest_lat,
        dest_lon,
        limit=max(1, num_alternatives),
        departure_time=departure,
    )

    candidates: list[dict[str, Any]] = []
    for route in raw_routes:
        coords = route.get("_latlngs") or _extract_coords(route)
        if len(coords) < 2:
            continue
        if all(geometry_similarity(coords, existing["_latlngs"]) < 0.90 for existing in candidates):
            copy = dict(route)
            copy["_latlngs"] = coords
            candidates.append(copy)

    results: list[dict[str, Any]] = []
    for route in candidates[:num_alternatives]:
        route_traffic = traffic_for_route(route["_latlngs"], global_traffic)
        if not consider_incidents:
            route_traffic = dict(route_traffic)
            route_traffic.update(
                {
                    "incident_count": 0,
                    "construction_count": 0,
                    "accident_count": 0,
                    "road_closure_count": 0,
                    "incident_active": False,
                    "incident_markers": [],
                }
            )

        features, metrics, steps = _route_features(route, departure, route_traffic, weather, origin_lat, origin_lon)
        predicted_factor, pred_meta = predict(features, model_type)
        live_duration_s = float(metrics["live_duration_s"])
        base_factor = max(0.75, min(1.35, 0.88 + 0.08 * predicted_factor + 0.03 * metrics["congestion"] + 0.02 * float(weather.get("severity", 0.0))))
        eta_s = live_duration_s * base_factor

        total_distance = metrics["distance_m"]
        coords = route["_latlngs"]
        segments: list[RouteSegment] = []
        segment_time_scale = eta_s / max(total_distance, 1.0)
        for idx, (a, b) in enumerate(zip(coords, coords[1:])):
            seg_dist = _haversine(tuple(a), tuple(b))
            seg_time = seg_dist * segment_time_scale
            segments.append(
                RouteSegment(
                    segment_id=f"{uuid.uuid4().hex[:8]}:{idx}",
                    start_node=idx,
                    end_node=idx + 1,
                    start_lat=a[0],
                    start_lon=a[1],
                    end_lat=b[0],
                    end_lon=b[1],
                    distance_m=seg_dist,
                    predicted_travel_time_s=seg_time,
                    speed_kph=max(5.0, seg_dist / max(seg_time, 1.0) * 3.6),
                    congestion_index=metrics["congestion"],
                    risk_score=min(1.0, features.context.road_risk_score),
                )
            )

        risk = float(features.context.road_risk_score)
        route_closure = bool(route_traffic.get("road_closure_count", 0))
        roadworks = bool(route_traffic.get("construction_count", 0))
        accidents = bool(route_traffic.get("accident_count", 0))
        external_event = "⛔ Road Closed" if route_closure else "🚧 Construction" if roadworks else "🚨 Incident" if accidents else "Clear Route"
        incident_markers = route_traffic.get("incident_markers", [])

        cvar_mean, cvar_low, cvar_high, cvar_95 = monte_carlo_cvar_travel_time(live_duration_s, eta_s / max(live_duration_s, 1.0))

        results.append(
            {
                "route_id": uuid.uuid4().hex[:8],
                "route_source": route.get("source", "tomtom"),
                "route_type": route.get("route_type", "fastest"),
                "segments": segments,
                "total_distance_m": total_distance,
                "total_travel_time_s": eta_s,
                "total_travel_time_display": _format_duration(eta_s),
                "confidence_interval_lower_s": cvar_low,
                "confidence_interval_upper_s": cvar_high,
                "risk_level": _classify_risk(risk),
                "reliability_score": max(0.0, min(1.0, 1.0 - risk)),
                "emissions_g_co2": estimate_emissions(total_distance, base_factor),
                "fuel_cost_estimate": estimate_fuel_cost(total_distance, base_factor),
                "traffic_color": "#D50000" if metrics["congestion"] >= 0.6 else "#FF9100" if metrics["congestion"] >= 0.45 else "#FFD600" if metrics["congestion"] >= 0.3 else "#00C853",
                "traffic_level": "severe" if metrics["congestion"] >= 0.6 else "high" if metrics["congestion"] >= 0.45 else "moderate" if metrics["congestion"] >= 0.3 else "low",
                "traffic_reasoning": (
                    f"{route.get('route_type', 'fastest').title()} route; "
                    f"{metrics['road_class_score']:.2f} road-class score; "
                    f"{metrics['turns']:.0f} turns; "
                    f"{metrics['incident_count']:.0f} incidents; "
                    f"live congestion {metrics['congestion']:.2f}."
                ),
                "has_road_geometry": True,
                "external_event": external_event,
                "incident_markers": incident_markers,
                "road_closure_active": route_closure,
                "accident_active": accidents,
                "roadworks_active": roadworks,
                "weather_severity": float(weather.get("severity", 0.0)),
                "coords": coords,
                "steps": steps,
                "osrm_duration_s": live_duration_s,
                "osrm_duration_display": _format_duration(live_duration_s),
                "route_congestion": metrics["congestion"],
                "optimization_score": 0.0,
                "ev_energy_kwh": total_distance / 1000.0 * 0.16 * (1.0 + 0.45 * metrics["congestion"]),
                "driving_comfort_score": max(0.0, 1.0 - 0.45 * metrics["congestion"] - 0.20 * float(weather.get("severity", 0.0)) - 0.01 * metrics["turns"]),
                "safety_score": max(0.0, 1.0 - risk),
                "risk_score": risk,
                "total_cvar_s": cvar_95,
                "total_cvar_display": _format_duration(cvar_95),
                "confidence_score": pred_meta.confidence_score,
                "pred_meta": pred_meta,
                "predicted_factor": predicted_factor,
                "features": features,
                "road_class_score": metrics["road_class_score"],
                "road_quality_score": metrics["road_quality_score"],
                "incident_count": metrics["incident_count"],
                "construction_count": metrics["construction_count"],
                "accident_count": metrics["accident_count"],
                "road_closure_count": metrics["road_closure_count"],
                "toll_count": metrics["toll_count"],
                "turns_count": metrics["turns"],
                "traffic_delay_s": metrics["traffic_delay_s"],
                "avg_speed_kph": metrics["avg_speed_kph"],
                "free_flow_speed_kph": metrics["free_flow_speed_kph"],
                "historical_speed_kph": features.context.historical_speed_kph,
            }
        )

    ranked = WeightedSumRouteOptimizer().optimize(results, objective)

    formatted: list[RouteResult] = []
    for index, route in enumerate(ranked):
        route["rank"] = index + 1
        formatted.append(
            RouteResult(
                **{
                    key: value
                    for key, value in route.items()
                    if key
                    not in {
                        "pred_meta",
                        "features",
                        "predicted_factor",
                        "confidence_score",
                    }
                }
            )
        )

    best = ranked[0] if ranked else None
    return {
        "routes": formatted,
        "routes_raw": ranked,
        "traffic": TrafficCondition(
            segment_id="global",
            congestion_index=global_traffic.get("congestion_index", 0.3),
            avg_speed_kph=global_traffic.get("avg_speed_kph", 0.0),
            incident_active=bool(global_traffic.get("incident_active", False)),
        ),
        "weather": WeatherCondition(
            condition=weather.get("condition", "clear"),
            severity=float(weather.get("severity", 0.0)),
            temperature_c=float(weather.get("temperature_c", 20.0)),
            visibility_km=float(weather.get("visibility_km", 10.0)),
        ),
        "prediction_meta": best["pred_meta"] if best else None,
        "predicted_factor": best["predicted_factor"] if best else 1.0,
        "features": best["features"] if best else None,
    }


def plan_intelligent_routes(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    model_type: ModelType = ModelType.XGBOOST,
    objective: OptimizationObjective = OptimizationObjective.BALANCED,
    num_alternatives: int = 3,
    departure_time: str | None = None,
    consider_weather: bool = True,
    consider_incidents: bool = True,
) -> dict[str, Any]:
    return plan_routes(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
        model_type=model_type,
        objective=objective,
        num_alternatives=num_alternatives,
        departure_time=departure_time,
        consider_weather=consider_weather,
        consider_incidents=consider_incidents,
    )
