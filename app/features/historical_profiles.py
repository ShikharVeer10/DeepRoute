
from datetime import datetime
from typing import Optional

from loguru import logger


def get_historical_context(
    edge_id: str,
    hour: int | None = None,
    day_of_week: int | None = None,
) -> dict:

    from app.storage.database import get_historical_speed_profile

    h = hour if hour is not None else datetime.now().hour
    dow = day_of_week if day_of_week is not None else datetime.now().weekday()

    profiles = get_historical_speed_profile(edge_id, h, dow)

    if not profiles:
        return {
            "historical_speed_kph": None,
            "historical_congestion": None,
            "reliability": 0.5,
            "sample_count": 0,
            "has_data": False,
        }

    p = profiles[0]
    return {
        "historical_speed_kph": p["avg_speed_kph"],
        "historical_congestion": p["avg_congestion"],
        "reliability": p["reliability_score"],
        "sample_count": p["sample_count"],
        "std_dev_speed": p.get("std_dev_speed", 0),
        "has_data": True,
    }


def get_congestion_color(congestion_index: float) -> str:
    if congestion_index < 0.15:
        return "#00C853"   # Green: free flow
    elif congestion_index < 0.30:
        return "#64DD17"   # Light green: light traffic
    elif congestion_index < 0.45:
        return "#FFD600"   # Yellow: moderate
    elif congestion_index < 0.60:
        return "#FF9100"   # Orange: heavy
    elif congestion_index < 0.75:
        return "#FF3D00"   # Red-orange: very heavy
    else:
        return "#D50000"   # Dark red: severe / gridlock


def get_speed_color(speed_kph: float, speed_limit_kph: float = 50) -> str:
    """
    Map actual speed vs speed limit to traffic color.
    """
    ratio = speed_kph / max(speed_limit_kph, 1)

    if ratio >= 0.85:
        return "#00C853"
    elif ratio >= 0.65:
        return "#64DD17"
    elif ratio >= 0.45:
        return "#FFD600"
    elif ratio >= 0.30:
        return "#FF9100"
    elif ratio >= 0.15:
        return "#FF3D00"
    else:
        return "#D50000"


def get_24h_pattern(edge_id: str, day_of_week: int | None = None) -> list[dict]:
    """
    Get the full 24-hour congestion pattern for an edge.
    Returns a list of 24 items (one per hour) with speed/congestion.
    Used for the "Typical Traffic" slider.
    """
    from app.storage.database import get_hourly_congestion_pattern

    dow = day_of_week if day_of_week is not None else datetime.now().weekday()
    pattern = get_hourly_congestion_pattern(edge_id, dow)

    # Fill gaps with interpolation
    hour_map = {p["hour_bucket"]: p for p in pattern}
    result = []

    for h in range(24):
        if h in hour_map:
            p = hour_map[h]
            result.append({
                "hour": h,
                "speed_kph": p["avg_speed_kph"],
                "congestion": p["avg_congestion"],
                "reliability": p["reliability_score"],
                "color": get_congestion_color(p["avg_congestion"]),
                "has_data": True,
            })
        else:
            # Interpolate from neighbors
            result.append({
                "hour": h,
                "speed_kph": 40,
                "congestion": 0.2,
                "reliability": 0.5,
                "color": "#64DD17",
                "has_data": False,
            })

    return result


def compute_congestion_trend(edge_id: str, hours_back: int = 6) -> dict:
    """
    Analyze if traffic is getting better or worse on this edge.
    Returns trend direction and magnitude.
    """
    from app.storage.database import get_edge_traffic_history

    history = get_edge_traffic_history(edge_id, hours_back)

    if len(history) < 2:
        return {"trend": "stable", "magnitude": 0, "data_points": len(history)}

    congestions = [h["congestion_index"] for h in history]
    recent_avg = sum(congestions[:len(congestions)//2]) / max(1, len(congestions)//2)
    older_avg = sum(congestions[len(congestions)//2:]) / max(1, len(congestions) - len(congestions)//2)

    diff = recent_avg - older_avg

    if diff > 0.1:
        trend = "worsening"
    elif diff < -0.1:
        trend = "improving"
    else:
        trend = "stable"

    return {
        "trend": trend,
        "magnitude": round(abs(diff), 3),
        "recent_congestion": round(recent_avg, 3),
        "older_congestion": round(older_avg, 3),
        "data_points": len(history),
    }


def build_traffic_heatmap_data(
    route_segments: list[dict],
    hour: int | None = None,
    day_of_week: int | None = None,
) -> list[dict]:
    """
    Build traffic heatmap data for a list of route segments.
    Each segment gets colored based on congestion level.

    Parameters
    ----------
    route_segments : list of dicts with at least edge_id, start_lat, start_lon, end_lat, end_lon
    hour, day_of_week : optional time context

    Returns
    -------
    List of segments with color and congestion info added
    """
    from app.storage.database import get_congestion_for_edges

    edge_ids = [s.get("edge_id", s.get("segment_id", "")) for s in route_segments]
    congestion_data = get_congestion_for_edges(edge_ids, hour, day_of_week)

    colored_segments = []
    for seg in route_segments:
        edge_id = seg.get("edge_id", seg.get("segment_id", ""))
        if edge_id in congestion_data:
            c_data = congestion_data[edge_id]
            color = get_congestion_color(c_data["congestion_index"])
            seg_out = {**seg, **c_data, "color": color}
        else:
            # Default: moderate traffic
            seg_out = {
                **seg,
                "congestion_index": 0.3,
                "speed_kph": 35,
                "color": get_congestion_color(0.3),
                "reliability": 0.5,
            }
        colored_segments.append(seg_out)

    return colored_segments
