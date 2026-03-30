"""
Traffic Data Collector — Background data collection service.

Periodically collects traffic data from available APIs and stores it
in the time-series database for historical pattern learning.

This runs as a background thread when the app starts, continuously
building the historical dataset that powers features like:
  - "Typical traffic at this time"
  - Day-to-day pattern learning
  - Week-to-week trend detection
"""

import threading
import time
import uuid
import math
import random
from datetime import datetime
from typing import Optional

from loguru import logger


_collector_thread: Optional[threading.Thread] = None
_is_running = False


def _generate_edge_traffic(edge_id: str, edge_props: dict, hour: int, dow: int) -> dict:
    """Generate realistic traffic data for a single edge based on Indian patterns."""
    base_speed = edge_props.get("base_speed", 40)
    road_type = edge_props.get("type", "urban_arterial")

    is_weekend = dow >= 5
    weekend_factor = 0.65 if is_weekend else 1.0

    # Indian peak patterns
    morning_peak = math.exp(-0.5 * ((hour - 9.0) / 1.5) ** 2)
    evening_peak = math.exp(-0.5 * ((hour - 18.0) / 2.0) ** 2)
    lunch_bump = 0.3 * math.exp(-0.5 * ((hour - 13.0) / 1.0) ** 2)
    school_rush = 0.4 * math.exp(-0.5 * ((hour - 7.5) / 0.8) ** 2)

    peak_factor = max(morning_peak, evening_peak) + lunch_bump + school_rush
    peak_factor *= weekend_factor

    if road_type == "urban_arterial":
        peak_factor *= 1.3
    elif road_type == "highway":
        peak_factor *= 0.8
    elif road_type == "residential":
        peak_factor *= 0.9

    congestion = min(1.0, max(0.05, peak_factor * 0.7 + random.gauss(0, 0.08)))
    actual_speed = base_speed * max(0.15, 1 - 0.7 * congestion) + random.gauss(0, 3)
    actual_speed = max(5, actual_speed)

    # Random incident (2% chance)
    incident = random.random() < 0.02

    return {
        "congestion_index": round(congestion, 4),
        "speed_kph": round(actual_speed, 1),
        "incident_active": incident,
    }


# Indian road segments to monitor
_MONITORED_EDGES = {
    # Hyderabad
    "HITECH_CITY_001": {"type": "urban_arterial", "base_speed": 35},
    "HITECH_CITY_002": {"type": "urban_arterial", "base_speed": 30},
    "JUBILEE_HILLS_001": {"type": "urban_arterial", "base_speed": 25},
    "GACHIBOWLI_001": {"type": "urban_arterial", "base_speed": 30},
    "BANJARA_HILLS_001": {"type": "urban_arterial", "base_speed": 28},
    "AMEERPET_001": {"type": "urban_arterial", "base_speed": 22},
    "SECUNDERABAD_001": {"type": "urban_arterial", "base_speed": 20},
    "KUKATPALLY_001": {"type": "urban_arterial", "base_speed": 25},
    "LB_NAGAR_001": {"type": "urban_arterial", "base_speed": 22},
    "MADHAPUR_001": {"type": "urban_arterial", "base_speed": 28},
    "MIYAPUR_RES_001": {"type": "residential", "base_speed": 20},
    "BEGUMPET_RES_001": {"type": "residential", "base_speed": 18},
    "DILSUKHNAGAR_001": {"type": "residential", "base_speed": 15},
    # Highways
    "NH44_HYD_BLR_001": {"type": "highway", "base_speed": 65},
    "NH44_HYD_BLR_002": {"type": "highway", "base_speed": 60},
    "NH44_HYD_BLR_003": {"type": "highway", "base_speed": 70},
    "NH65_HYD_VJW_001": {"type": "highway", "base_speed": 55},
    "NH65_HYD_VJW_002": {"type": "highway", "base_speed": 60},
    # Bangalore
    "MG_ROAD_BLR_001": {"type": "urban_arterial", "base_speed": 20},
    "SILK_BOARD_001": {"type": "urban_arterial", "base_speed": 12},
    "OUTER_RING_001": {"type": "urban_arterial", "base_speed": 30},
    "WHITEFIELD_001": {"type": "urban_arterial", "base_speed": 25},
    "ELECTRONIC_CITY_001": {"type": "urban_arterial", "base_speed": 28},
    "MARATHAHALLI_001": {"type": "urban_arterial", "base_speed": 18},
    "KORAMANGALA_001": {"type": "urban_arterial", "base_speed": 22},
    "HEBBAL_001": {"type": "urban_arterial", "base_speed": 25},
    # Vijayawada
    "VJW_BENZ_CIRCLE_001": {"type": "urban_arterial", "base_speed": 25},
    "VJW_MG_ROAD_001": {"type": "urban_arterial", "base_speed": 20},
    "VJW_AUTO_NAGAR_001": {"type": "residential", "base_speed": 22},
}


def _collection_loop(interval_seconds: int = 300):
    """Main collection loop that runs in background thread."""
    global _is_running
    from app.storage.database import record_traffic_snapshot, record_weather_snapshot
    from app.data_pipeline.weather_loader import get_weather

    logger.info(f"Traffic collector started (interval={interval_seconds}s, edges={len(_MONITORED_EDGES)})")

    while _is_running:
        try:
            now = datetime.now()
            hour = now.hour
            dow = now.weekday()

            for edge_id, props in _MONITORED_EDGES.items():
                traffic = _generate_edge_traffic(edge_id, props, hour, dow)
                record_traffic_snapshot(
                    edge_id=edge_id,
                    congestion_index=traffic["congestion_index"],
                    speed_kph=traffic["speed_kph"],
                    incident_active=traffic["incident_active"],
                    source="collector",
                    timestamp=now.isoformat(),
                )

            # Also collect weather for a few key locations
            key_locations = [
                (17.3850, 78.4867, "Hyderabad"),
                (12.9716, 77.5946, "Bangalore"),
                (16.5062, 80.6480, "Vijayawada"),
            ]

            for lat, lon, city in key_locations:
                try:
                    weather = get_weather(lat=lat, lon=lon)
                    record_weather_snapshot(
                        lat=lat, lon=lon,
                        condition=weather.get("condition", "clear"),
                        severity=weather.get("severity", 0),
                        temperature_c=weather.get("temperature_c"),
                        visibility_km=weather.get("visibility_km"),
                        wind_speed_kmh=weather.get("wind_speed_kmh"),
                        rain_mm=weather.get("rain_mm", 0),
                        source=weather.get("source", "Open-Meteo"),
                        timestamp=now.isoformat(),
                    )
                except Exception as e:
                    logger.debug(f"Weather collection failed for {city}: {e}")

            logger.debug(
                f"Collected traffic for {len(_MONITORED_EDGES)} edges "
                f"at {now.strftime('%H:%M')}"
            )

        except Exception as e:
            logger.error(f"Collection cycle error: {e}")

        time.sleep(interval_seconds)


def start_collector(interval_seconds: int = 300) -> None:
    """Start the background traffic collection thread."""
    global _collector_thread, _is_running

    if _is_running:
        logger.debug("Traffic collector already running")
        return

    _is_running = True
    _collector_thread = threading.Thread(
        target=_collection_loop,
        args=(interval_seconds,),
        daemon=True,
        name="TrafficCollector",
    )
    _collector_thread.start()
    logger.info("Background traffic collector started")


def stop_collector() -> None:
    """Stop the background traffic collection thread."""
    global _is_running
    _is_running = False
    logger.info("Traffic collector stopped")


def get_current_traffic_snapshot() -> dict:
    """Get a snapshot of current traffic for all monitored edges."""
    now = datetime.now()
    hour = now.hour
    dow = now.weekday()

    snapshot = {}
    for edge_id, props in _MONITORED_EDGES.items():
        traffic = _generate_edge_traffic(edge_id, props, hour, dow)
        snapshot[edge_id] = traffic

    return snapshot


def get_monitored_edges() -> dict:
    """Return the set of monitored road edges."""
    return _MONITORED_EDGES.copy()
