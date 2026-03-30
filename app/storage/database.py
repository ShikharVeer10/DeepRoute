"""
DeepRoute — SQLite Time-Series Database

Stores:
  - Traffic snapshots per road edge (congestion, speed) every 5-15 min
  - Weather history (condition, severity, temp) per location
  - User trip records (route taken, actual travel time)
  - Prediction logs (predicted vs actual for model evaluation)

This enables:
  - Historical speed profiles ("Monday 9 AM on Ring Road → 12 km/h")
  - Day-to-day and week-to-week pattern learning
  - Prediction accuracy tracking
  - Continuous model improvement
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from loguru import logger

_DB_DIR = Path("data")
_DB_PATH = _DB_DIR / "deeproute.db"


@contextmanager
def _get_conn():
    """Thread-safe connection context manager."""
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they don't exist."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)

    with _get_conn() as conn:
        conn.executescript("""
        -- Traffic snapshots per road edge
        CREATE TABLE IF NOT EXISTS traffic_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            edge_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            hour INTEGER NOT NULL,
            day_of_week INTEGER NOT NULL,
            congestion_index REAL NOT NULL,
            speed_kph REAL,
            incident_active INTEGER DEFAULT 0,
            source TEXT DEFAULT 'simulated',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_traffic_edge_time
            ON traffic_snapshots(edge_id, hour, day_of_week);
        CREATE INDEX IF NOT EXISTS idx_traffic_timestamp
            ON traffic_snapshots(timestamp);

        -- Weather history
        CREATE TABLE IF NOT EXISTS weather_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            timestamp TEXT NOT NULL,
            condition TEXT NOT NULL,
            severity REAL NOT NULL,
            temperature_c REAL,
            visibility_km REAL,
            wind_speed_kmh REAL,
            rain_mm REAL DEFAULT 0,
            source TEXT DEFAULT 'Open-Meteo',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_weather_loc_time
            ON weather_history(latitude, longitude, timestamp);

        -- User trip records
        CREATE TABLE IF NOT EXISTS trips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id TEXT UNIQUE NOT NULL,
            origin_lat REAL NOT NULL,
            origin_lon REAL NOT NULL,
            dest_lat REAL NOT NULL,
            dest_lon REAL NOT NULL,
            origin_name TEXT,
            dest_name TEXT,
            departure_time TEXT NOT NULL,
            arrival_time TEXT,
            predicted_travel_time_s REAL,
            actual_travel_time_s REAL,
            distance_m REAL,
            route_geometry TEXT,
            model_used TEXT,
            congestion_at_departure REAL,
            weather_at_departure TEXT,
            day_of_week INTEGER,
            hour INTEGER,
            status TEXT DEFAULT 'predicted',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_trips_time
            ON trips(departure_time);
        CREATE INDEX IF NOT EXISTS idx_trips_dow_hour
            ON trips(day_of_week, hour);

        -- Prediction logs (predicted vs actual for model evaluation)
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id TEXT UNIQUE NOT NULL,
            trip_id TEXT,
            model_used TEXT NOT NULL,
            predicted_factor REAL NOT NULL,
            predicted_travel_time_s REAL,
            actual_travel_time_s REAL,
            error_percent REAL,
            origin_lat REAL,
            origin_lon REAL,
            dest_lat REAL,
            dest_lon REAL,
            features_json TEXT,
            congestion_index REAL,
            weather_condition TEXT,
            weather_severity REAL,
            hour INTEGER,
            day_of_week INTEGER,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_pred_model
            ON prediction_logs(model_used, timestamp);
        CREATE INDEX IF NOT EXISTS idx_pred_time
            ON prediction_logs(timestamp);

        -- Historical speed profiles (aggregated per edge per time bucket)
        CREATE TABLE IF NOT EXISTS speed_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            edge_id TEXT NOT NULL,
            hour_bucket INTEGER NOT NULL,
            day_of_week INTEGER NOT NULL,
            avg_speed_kph REAL NOT NULL,
            avg_congestion REAL NOT NULL,
            sample_count INTEGER DEFAULT 1,
            std_dev_speed REAL DEFAULT 0,
            reliability_score REAL DEFAULT 1.0,
            last_updated TEXT DEFAULT (datetime('now')),
            UNIQUE(edge_id, hour_bucket, day_of_week)
        );

        CREATE INDEX IF NOT EXISTS idx_speed_edge
            ON speed_profiles(edge_id, hour_bucket, day_of_week);

        -- Indian events calendar
        CREATE TABLE IF NOT EXISTS indian_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT NOT NULL,
            event_date TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity_multiplier REAL DEFAULT 1.0,
            affected_cities TEXT,
            description TEXT,
            recurring INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_events_date
            ON indian_events(event_date);
        """)

    logger.info(f"Database initialized at {_DB_PATH}")


# ─── Traffic Snapshots ────────────────────────────────────────────────────────


def record_traffic_snapshot(
    edge_id: str,
    congestion_index: float,
    speed_kph: float | None = None,
    incident_active: bool = False,
    source: str = "simulated",
    timestamp: str | None = None,
) -> None:
    """Record a traffic snapshot for a road edge."""
    now = timestamp or datetime.now().isoformat()
    dt = datetime.fromisoformat(now) if isinstance(now, str) else now
    hour = dt.hour
    dow = dt.weekday()

    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO traffic_snapshots
               (edge_id, timestamp, hour, day_of_week, congestion_index,
                speed_kph, incident_active, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (edge_id, now, hour, dow, congestion_index,
             speed_kph, int(incident_active), source),
        )

    # Update aggregated speed profile
    _update_speed_profile(edge_id, hour, dow, speed_kph, congestion_index)


def _update_speed_profile(
    edge_id: str, hour: int, dow: int,
    speed_kph: float | None, congestion: float,
) -> None:
    """Incrementally update the aggregated speed profile."""
    if speed_kph is None:
        return

    with _get_conn() as conn:
        existing = conn.execute(
            """SELECT avg_speed_kph, avg_congestion, sample_count, std_dev_speed
               FROM speed_profiles
               WHERE edge_id = ? AND hour_bucket = ? AND day_of_week = ?""",
            (edge_id, hour, dow),
        ).fetchone()

        if existing:
            n = existing["sample_count"]
            new_n = n + 1
            new_avg_speed = (existing["avg_speed_kph"] * n + speed_kph) / new_n
            new_avg_cong = (existing["avg_congestion"] * n + congestion) / new_n
            # Running variance (Welford's method approximation)
            old_std = existing["std_dev_speed"]
            new_std = ((old_std ** 2 * n + (speed_kph - new_avg_speed) ** 2) / new_n) ** 0.5
            reliability = max(0, min(1, 1.0 - new_std / max(new_avg_speed, 1)))

            conn.execute(
                """UPDATE speed_profiles
                   SET avg_speed_kph = ?, avg_congestion = ?,
                       sample_count = ?, std_dev_speed = ?,
                       reliability_score = ?, last_updated = datetime('now')
                   WHERE edge_id = ? AND hour_bucket = ? AND day_of_week = ?""",
                (new_avg_speed, new_avg_cong, new_n, new_std, reliability,
                 edge_id, hour, dow),
            )
        else:
            conn.execute(
                """INSERT INTO speed_profiles
                   (edge_id, hour_bucket, day_of_week, avg_speed_kph,
                    avg_congestion, sample_count, std_dev_speed, reliability_score)
                   VALUES (?, ?, ?, ?, ?, 1, 0, 1.0)""",
                (edge_id, hour, dow, speed_kph, congestion),
            )


# ─── Weather History ──────────────────────────────────────────────────────────


def record_weather_snapshot(
    lat: float, lon: float,
    condition: str, severity: float,
    temperature_c: float | None = None,
    visibility_km: float | None = None,
    wind_speed_kmh: float | None = None,
    rain_mm: float = 0,
    source: str = "Open-Meteo",
    timestamp: str | None = None,
) -> None:
    """Record a weather snapshot."""
    now = timestamp or datetime.now().isoformat()
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO weather_history
               (latitude, longitude, timestamp, condition, severity,
                temperature_c, visibility_km, wind_speed_kmh, rain_mm, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (round(lat, 4), round(lon, 4), now, condition, severity,
             temperature_c, visibility_km, wind_speed_kmh, rain_mm, source),
        )


# ─── Trip Recording ──────────────────────────────────────────────────────────


def record_trip(
    trip_id: str,
    origin_lat: float, origin_lon: float,
    dest_lat: float, dest_lon: float,
    departure_time: str,
    predicted_travel_time_s: float,
    distance_m: float,
    model_used: str,
    congestion_at_departure: float = 0,
    weather_at_departure: str = "clear",
    origin_name: str | None = None,
    dest_name: str | None = None,
    route_geometry: str | None = None,
) -> None:
    """Record a predicted trip (before user actually travels)."""
    dt = datetime.fromisoformat(departure_time)
    with _get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO trips
               (trip_id, origin_lat, origin_lon, dest_lat, dest_lon,
                origin_name, dest_name, departure_time,
                predicted_travel_time_s, distance_m, route_geometry,
                model_used, congestion_at_departure, weather_at_departure,
                day_of_week, hour, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'predicted')""",
            (trip_id, origin_lat, origin_lon, dest_lat, dest_lon,
             origin_name, dest_name, departure_time,
             predicted_travel_time_s, distance_m, route_geometry,
             model_used, congestion_at_departure, weather_at_departure,
             dt.weekday(), dt.hour),
        )


def complete_trip(trip_id: str, actual_travel_time_s: float) -> None:
    """Mark a trip as completed with actual travel time."""
    arrival = datetime.now().isoformat()
    with _get_conn() as conn:
        conn.execute(
            """UPDATE trips
               SET actual_travel_time_s = ?, arrival_time = ?, status = 'completed'
               WHERE trip_id = ?""",
            (actual_travel_time_s, arrival, trip_id),
        )
        
        pred = conn.execute(
            "SELECT predicted_travel_time_s FROM prediction_logs WHERE trip_id = ?",
            (trip_id,)
        ).fetchone()
        
        if pred:
            pred_s = pred['predicted_travel_time_s']
            error_pct = ((actual_travel_time_s - pred_s) / max(pred_s, 1)) * 100
            
            conn.execute(
                """UPDATE prediction_logs
                   SET actual_travel_time_s = ?, error_percent = ?
                   WHERE trip_id = ?""",
                (actual_travel_time_s, error_pct, trip_id)
            )


# ─── Prediction Logging ──────────────────────────────────────────────────────


def record_prediction(
    prediction_id: str,
    model_used: str,
    predicted_factor: float,
    predicted_travel_time_s: float,
    origin_lat: float, origin_lon: float,
    dest_lat: float, dest_lon: float,
    features_json: str | None = None,
    congestion_index: float = 0,
    weather_condition: str = "clear",
    weather_severity: float = 0,
    trip_id: str | None = None,
) -> None:
    """Log a prediction for accuracy tracking."""
    now = datetime.now()
    with _get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO prediction_logs
               (prediction_id, trip_id, model_used, predicted_factor,
                predicted_travel_time_s, origin_lat, origin_lon,
                dest_lat, dest_lon, features_json,
                congestion_index, weather_condition, weather_severity,
                hour, day_of_week, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (prediction_id, trip_id, model_used, predicted_factor,
             predicted_travel_time_s, origin_lat, origin_lon,
             dest_lat, dest_lon, features_json,
             congestion_index, weather_condition, weather_severity,
             now.hour, now.weekday(), now.isoformat()),
        )


# ─── Query Functions ──────────────────────────────────────────────────────────


def get_historical_speed_profile(
    edge_id: str,
    hour: int | None = None,
    day_of_week: int | None = None,
) -> list[dict]:
    """
    Get historical speed profile for an edge.
    If hour and day_of_week provided, returns specific profile.
    Otherwise returns all profiles for the edge.
    """
    with _get_conn() as conn:
        if hour is not None and day_of_week is not None:
            rows = conn.execute(
                """SELECT * FROM speed_profiles
                   WHERE edge_id = ? AND hour_bucket = ? AND day_of_week = ?""",
                (edge_id, hour, day_of_week),
            ).fetchall()
        elif hour is not None:
            rows = conn.execute(
                """SELECT * FROM speed_profiles
                   WHERE edge_id = ? AND hour_bucket = ?
                   ORDER BY day_of_week""",
                (edge_id, hour),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM speed_profiles
                   WHERE edge_id = ? ORDER BY day_of_week, hour_bucket""",
                (edge_id,),
            ).fetchall()

        return [dict(r) for r in rows]


def get_edge_traffic_history(
    edge_id: str,
    hours_back: int = 24,
) -> list[dict]:
    """Get recent traffic snapshots for an edge."""
    cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM traffic_snapshots
               WHERE edge_id = ? AND timestamp >= ?
               ORDER BY timestamp DESC""",
            (edge_id, cutoff),
        ).fetchall()
        return [dict(r) for r in rows]


def get_congestion_for_edges(
    edge_ids: list[str] | None = None,
    hour: int | None = None,
    day_of_week: int | None = None,
) -> dict[str, dict]:
    """
    Get current/historical congestion for multiple edges.
    Returns {edge_id: {congestion_index, speed_kph, reliability}}.
    Used for traffic heatmap coloring.
    """
    h = hour if hour is not None else datetime.now().hour
    dow = day_of_week if day_of_week is not None else datetime.now().weekday()

    with _get_conn() as conn:
        if edge_ids:
            placeholders = ",".join("?" for _ in edge_ids)
            rows = conn.execute(
                f"""SELECT edge_id, avg_speed_kph, avg_congestion,
                           sample_count, reliability_score
                    FROM speed_profiles
                    WHERE edge_id IN ({placeholders})
                      AND hour_bucket = ? AND day_of_week = ?""",
                (*edge_ids, h, dow),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT edge_id, avg_speed_kph, avg_congestion,
                          sample_count, reliability_score
                   FROM speed_profiles
                   WHERE hour_bucket = ? AND day_of_week = ?
                   LIMIT 5000""",
                (h, dow),
            ).fetchall()

    result = {}
    for r in rows:
        result[r["edge_id"]] = {
            "congestion_index": r["avg_congestion"],
            "speed_kph": r["avg_speed_kph"],
            "sample_count": r["sample_count"],
            "reliability": r["reliability_score"],
        }
    return result


def get_prediction_accuracy(
    model_name: str | None = None,
    days_back: int = 30,
) -> dict:
    """Get prediction accuracy metrics."""
    cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()

    with _get_conn() as conn:
        if model_name:
            rows = conn.execute(
                """SELECT model_used,
                          COUNT(*) as total,
                          AVG(error_percent) as avg_error,
                          AVG(ABS(error_percent)) as mae,
                          AVG(predicted_travel_time_s) as avg_predicted,
                          AVG(actual_travel_time_s) as avg_actual
                   FROM prediction_logs
                   WHERE model_used = ? AND timestamp >= ?
                     AND actual_travel_time_s IS NOT NULL
                   GROUP BY model_used""",
                (model_name, cutoff),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT model_used,
                          COUNT(*) as total,
                          AVG(error_percent) as avg_error,
                          AVG(ABS(error_percent)) as mae,
                          AVG(predicted_travel_time_s) as avg_predicted,
                          AVG(actual_travel_time_s) as avg_actual
                   FROM prediction_logs
                   WHERE timestamp >= ?
                     AND actual_travel_time_s IS NOT NULL
                   GROUP BY model_used""",
                (cutoff,),
            ).fetchall()

    return {r["model_used"]: dict(r) for r in rows}


def get_recent_trips(limit: int = 20) -> list[dict]:
    """Get recent trip records."""
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM trips
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_traffic_heatmap_data(
    hour: int | None = None,
    day_of_week: int | None = None,
) -> list[dict]:
    """
    Get all edge congestion data for building a traffic heatmap.
    Returns list of {edge_id, congestion, speed, reliability}.
    """
    h = hour if hour is not None else datetime.now().hour
    dow = day_of_week if day_of_week is not None else datetime.now().weekday()

    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT edge_id, avg_speed_kph, avg_congestion,
                      sample_count, reliability_score, std_dev_speed
               FROM speed_profiles
               WHERE hour_bucket = ? AND day_of_week = ?
               ORDER BY avg_congestion DESC""",
            (h, dow),
        ).fetchall()
        return [dict(r) for r in rows]


def get_hourly_congestion_pattern(edge_id: str, day_of_week: int) -> list[dict]:
    """Get 24-hour congestion pattern for an edge on a specific day."""
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT hour_bucket, avg_speed_kph, avg_congestion,
                      sample_count, reliability_score
               FROM speed_profiles
               WHERE edge_id = ? AND day_of_week = ?
               ORDER BY hour_bucket""",
            (edge_id, day_of_week),
        ).fetchall()
        return [dict(r) for r in rows]


def seed_historical_data() -> None:
    """
    Seed the database with realistic Indian traffic patterns
    so the system has historical data from day one.
    Based on real Indian traffic behavior.
    """
    import numpy as np
    import math

    logger.info("Seeding historical traffic data for Indian roads...")

    rng = np.random.RandomState(42)

    # Major Indian road segments (representative edges)
    road_segments = {
        # Highways
        "NH44_HYD_BLR_001": {"type": "highway", "base_speed": 65},
        "NH44_HYD_BLR_002": {"type": "highway", "base_speed": 60},
        "NH44_HYD_BLR_003": {"type": "highway", "base_speed": 70},
        "NH65_HYD_VJW_001": {"type": "highway", "base_speed": 55},
        "NH65_HYD_VJW_002": {"type": "highway", "base_speed": 60},
        # Urban arterials
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
        # Residential
        "MIYAPUR_RES_001": {"type": "residential", "base_speed": 20},
        "BEGUMPET_RES_001": {"type": "residential", "base_speed": 18},
        "DILSUKHNAGAR_001": {"type": "residential", "base_speed": 15},
        # Bangalore roads
        "MG_ROAD_BLR_001": {"type": "urban_arterial", "base_speed": 20},
        "SILK_BOARD_001": {"type": "urban_arterial", "base_speed": 12},  # Famous bottleneck!
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

    count = 0
    with _get_conn() as conn:
        for edge_id, props in road_segments.items():
            base_speed = props["base_speed"]
            road_type = props["type"]

            for dow in range(7):
                is_weekend = dow >= 5
                weekend_factor = 0.65 if is_weekend else 1.0

                for hour in range(24):
                    # Indian peak patterns
                    morning_peak = math.exp(-0.5 * ((hour - 9.0) / 1.5) ** 2)
                    evening_peak = math.exp(-0.5 * ((hour - 18.0) / 2.0) ** 2)
                    lunch_bump = 0.3 * math.exp(-0.5 * ((hour - 13.0) / 1.0) ** 2)
                    school_rush = 0.4 * math.exp(-0.5 * ((hour - 7.5) / 0.8) ** 2)

                    peak_factor = max(morning_peak, evening_peak) + lunch_bump + school_rush
                    peak_factor *= weekend_factor

                    # Road type affects intensity
                    if road_type == "urban_arterial":
                        peak_factor *= 1.3
                    elif road_type == "residential":
                        peak_factor *= 0.8

                    congestion = min(1.0, max(0.05, peak_factor * 0.7 + rng.normal(0, 0.05)))
                    actual_speed = base_speed * max(0.15, 1 - 0.7 * congestion) + rng.normal(0, 2)
                    actual_speed = max(5, actual_speed)
                    std_dev = max(1, abs(rng.normal(3, 2)))
                    reliability = max(0.1, min(1.0, 1 - std_dev / max(actual_speed, 1)))
                    # Simulate ~4 weeks of data
                    sample_count = rng.randint(15, 60)

                    conn.execute(
                        """INSERT OR REPLACE INTO speed_profiles
                           (edge_id, hour_bucket, day_of_week, avg_speed_kph,
                            avg_congestion, sample_count, std_dev_speed,
                            reliability_score)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (edge_id, hour, dow, round(actual_speed, 1),
                         round(congestion, 4), sample_count,
                         round(std_dev, 2), round(reliability, 3)),
                    )
                    count += 1

    logger.info(f"Seeded {count} speed profile records for {len(road_segments)} road segments")


def seed_indian_events() -> None:
    """Seed Indian calendar events that affect traffic."""
    events = [
        # National holidays
        ("Republic Day", "2026-01-26", "national_holiday", 1.5, "all", "Major parade in Delhi, all roads affected"),
        ("Holi", "2026-03-17", "festival", 1.4, "all", "Festival of colors, morning peak shifts"),
        ("Good Friday", "2026-04-03", "holiday", 1.2, "all", "Public holiday, reduced traffic"),
        ("Eid al-Fitr", "2026-03-31", "festival", 1.5, "all", "End of Ramadan, old city heavy traffic"),
        ("Independence Day", "2026-08-15", "national_holiday", 1.5, "all", "National holiday with processions"),
        ("Ganesh Chaturthi", "2026-08-26", "festival", 1.7, "hyderabad,mumbai,pune", "Immersion processions block major roads"),
        ("Dussehra", "2026-10-02", "festival", 1.4, "all", "Festival traffic + rally/processions"),
        ("Diwali", "2026-10-21", "festival", 1.6, "all", "Market rush days before, quiet on day"),
        ("Christmas", "2026-12-25", "holiday", 1.2, "all", "Holiday traffic"),
        # Monsoon periods (sustained traffic impact)
        ("Monsoon Onset SW", "2026-06-01", "weather_event", 1.35, "all", "Southwest monsoon arrives, flooding risk begins"),
        ("Monsoon Peak", "2026-07-15", "weather_event", 1.4, "all", "Heavy rainfall, waterlogging, major delays"),
        ("Monsoon Retreat", "2026-09-15", "weather_event", 1.2, "all", "Monsoon withdrawing, intermittent rain"),
        # Recurring weekly events
        ("Weekend Market Rush", "2026-01-03", "weekly", 1.15, "all", "Saturday market traffic"),
        # Local events
        ("IPL Season Start", "2026-03-22", "sports", 1.3, "hyderabad,bangalore,mumbai,delhi,chennai", "Stadium vicinity heavy traffic 4-11 PM"),
        ("School Reopening", "2026-06-15", "education", 1.25, "all", "Morning/afternoon peaks intensify near schools"),
    ]

    with _get_conn() as conn:
        for event in events:
            conn.execute(
                """INSERT OR IGNORE INTO indian_events
                   (event_name, event_date, event_type, severity_multiplier,
                    affected_cities, description, recurring)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (*event, 1 if event[2] == "weekly" else 0),
            )

    logger.info(f"Seeded {len(events)} Indian calendar events")


def get_events_near_date(
    date: str | None = None,
    window_days: int = 3,
) -> list[dict]:
    """Get events near a given date."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    dt = datetime.strptime(date, "%Y-%m-%d")
    start = (dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=window_days)).strftime("%Y-%m-%d")

    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM indian_events
               WHERE event_date BETWEEN ? AND ?
               ORDER BY event_date""",
            (start, end),
        ).fetchall()
        return [dict(r) for r in rows]


def get_db_stats() -> dict:
    """Get database statistics."""
    with _get_conn() as conn:
        stats = {}
        for table in ["traffic_snapshots", "weather_history", "trips",
                       "prediction_logs", "speed_profiles", "indian_events"]:
            row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
            stats[table] = row["cnt"]
        return stats
