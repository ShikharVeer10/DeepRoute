"""TomTom live traffic adapter used for route-specific traffic features."""
from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import urlencode

import requests

_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
_INCIDENT_URL = "https://api.tomtom.com/traffic/services/5/incidentDetails"

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL_SECONDS = 90
_CACHE_MAX_ITEMS = 256

try:
    from dotenv import load_dotenv
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(_ROOT / ".env")
except Exception:
    pass


def _cache_get(key: str) -> dict[str, Any] | None:
    payload = _CACHE.get(key)
    if not payload:
        return None
    ts, data = payload
    if time.time() - ts > _CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return data


def _cache_set(key: str, data: dict[str, Any]) -> None:
    if len(_CACHE) >= _CACHE_MAX_ITEMS:
        oldest = min(_CACHE.items(), key=lambda item: item[1][0])[0]
        _CACHE.pop(oldest, None)
    _CACHE[key] = (time.time(), data)


def _cached_get_json(url: str, params: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    key = f"{url}?{urlencode(sorted((k, str(v)) for k, v in params.items()))}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    response = requests.get(url, params=params, headers=headers, timeout=5)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        _cache_set(key, data)
    return data


def _route_bbox(coords: list[list[float]]) -> tuple[float, float, float, float]:
    lats = [p[0] for p in coords]
    lons = [p[1] for p in coords]
    return min(lons), min(lats), max(lons), max(lats)


def _frc_score(frc: str | None) -> float:
    mapping = {
        "FRC0": 1.0,
        "FRC1": 0.92,
        "FRC2": 0.84,
        "FRC3": 0.72,
        "FRC4": 0.55,
        "FRC5": 0.38,
        "FRC6": 0.22,
    }
    return mapping.get((frc or "").upper(), 0.5)


def _incident_markers(incidents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for item in incidents[:8]:
        props = item.get("properties", {}) if isinstance(item, dict) else {}
        icon = str(props.get("iconCategory", "incident"))
        label = icon.replace("_", " ").title()
        desc = props.get("description") or props.get("roadNumbers") or "Live traffic incident"
        text = str(desc[0]) if isinstance(desc, list) and desc else str(desc)
        lower = f"{icon} {text}".lower()
        if "construction" in lower or "roadwork" in lower:
            kind = "construction"
        elif "accident" in lower or "collision" in lower:
            kind = "accident"
        elif "closure" in lower:
            kind = "closure"
        else:
            kind = "incident"
        markers.append(
            {
                "icon": "🚧" if kind == "construction" else "⚠️" if kind == "closure" else "🚨",
                "label": label,
                "desc": text,
                "type": kind,
            }
        )
    return markers


def incidents_for_route(coords: list[list[float]]) -> dict[str, Any]:
    """Fetch live incident details for the route bounding box."""
    key = os.getenv("TOMTOM_API_KEY")
    if not key or len(coords) < 2:
        return {
            "incident_count": 0,
            "construction_count": 0,
            "accident_count": 0,
            "road_closure_count": 0,
            "incident_active": False,
            "incident_markers": [],
            "source": "historical_fallback",
        }

    min_lon, min_lat, max_lon, max_lat = _route_bbox(coords)
    fields = "{incidents{type,geometry{type,coordinates},properties{iconCategory,description}}}"
    params = {
        "key": key,
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "fields": fields,
        "language": "en-GB",
        "timeValidityFilter": "present",
    }

    # TomTom may require a traffic model reference for some incident responses.
    # Try the current epoch as a freshness hint, then retry without it.
    try:
        params["t"] = int(time.time())
        data = _cached_get_json(_INCIDENT_URL, params)
    except Exception:
        params.pop("t", None)
        try:
            data = _cached_get_json(_INCIDENT_URL, params)
        except Exception:
            return {
                "incident_count": 0,
                "construction_count": 0,
                "accident_count": 0,
                "road_closure_count": 0,
                "incident_active": False,
                "incident_markers": [],
                "source": "historical_fallback",
            }

    incidents = data.get("incidents", []) if isinstance(data, dict) else []
    markers = _incident_markers(incidents)
    counts = {"construction": 0, "accident": 0, "closure": 0}
    for marker in markers:
        counts[marker["type"]] = counts.get(marker["type"], 0) + 1

    return {
        "incident_count": len(incidents),
        "construction_count": counts.get("construction", 0),
        "accident_count": counts.get("accident", 0),
        "road_closure_count": counts.get("closure", 0),
        "incident_active": bool(incidents),
        "incident_markers": markers,
        "source": "tomtom_incidents",
    }


def traffic_for_route(coords: list[list[float]], fallback: dict[str, Any]) -> dict[str, Any]:
    """Sample live TomTom flow along a route and return route-specific traffic."""
    key = os.getenv("TOMTOM_API_KEY")
    if not key or len(coords) < 2:
        return dict(fallback, source="historical_fallback", live_coverage=0.0)

    sample_indexes = sorted({int(round(i * (len(coords) - 1) / 6)) for i in range(7)})
    speeds: list[float] = []
    freeflows: list[float] = []
    travel_times: list[float] = []
    freeflow_times: list[float] = []
    confidences: list[float] = []
    frc_scores: list[float] = []
    closures = 0

    for index in sample_indexes:
        lat, lon = coords[index]
        try:
            response = requests.get(
                _FLOW_URL,
                params={
                    "key": key,
                    "point": f"{lat},{lon}",
                    "unit": "KMPH",
                },
                timeout=5,
            )
            response.raise_for_status()
            segment = response.json().get("flowSegmentData", {})
        except Exception:
            continue

        speed = float(segment.get("currentSpeed", 0) or 0)
        freeflow = float(segment.get("freeFlowSpeed", 0) or 0)
        current_tt = float(segment.get("currentTravelTime", 0) or 0)
        freeflow_tt = float(segment.get("freeFlowTravelTime", 0) or 0)
        confidence = float(segment.get("confidence", 0) or 0)
        frc = str(segment.get("frc", "") or "")

        if speed > 0 and freeflow > 0:
            speeds.append(speed)
            freeflows.append(freeflow)
            travel_times.append(current_tt or (1.0 / max(speed, 1e-6)))
            freeflow_times.append(freeflow_tt or (1.0 / max(freeflow, 1e-6)))
            confidences.append(confidence)
            frc_scores.append(_frc_score(frc))
            if bool(segment.get("roadClosure", False)):
                closures += 1

    incident_data = incidents_for_route(coords)
    if not speeds:
        return dict(fallback, **incident_data, source="historical_fallback", live_coverage=0.0)

    current_speed = sum(speeds) / len(speeds)
    freeflow_speed = sum(freeflows) / len(freeflows)
    current_time = sum(travel_times) / len(travel_times)
    freeflow_time = sum(freeflow_times) / len(freeflow_times)
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    frc_score = sum(frc_scores) / len(frc_scores) if frc_scores else 0.5

    congestion = 1.0 - current_speed / max(freeflow_speed, 1.0)
    delay_ratio = current_time / max(freeflow_time, 1.0)
    congestion = max(0.0, min(1.0, 0.65 * congestion + 0.35 * (delay_ratio - 1.0)))

    return {
        "congestion_index": round(congestion, 4),
        "avg_speed_kph": round(current_speed, 2),
        "free_flow_speed_kph": round(freeflow_speed, 2),
        "current_travel_time_s": round(current_time, 2),
        "free_flow_travel_time_s": round(freeflow_time, 2),
        "confidence": round(confidence, 4),
        "road_class_score": round(frc_score, 4),
        "incident_active": bool(closures or incident_data.get("incident_active")),
        "incident_proximity_km": 0.0 if incident_data.get("incident_active") else 99.0,
        "road_closure_count": closures + int(incident_data.get("road_closure_count", 0)),
        "source": "tomtom_flow",
        "live_coverage": round(len(speeds) / len(sample_indexes), 3),
        **incident_data,
    }
