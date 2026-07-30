"""
Context feature extraction — combines traffic, weather, and event signals.
"""

from app.schemas import ContextFeatures


def context_features(
    traffic: dict,
    weather: dict,
    event_proximity_km: float | None = None,
) -> ContextFeatures:
    """
    Build context features from live traffic and weather data.

    Parameters
    ----------
    traffic : dict from traffic_loader.get_traffic()
    weather : dict from weather_loader.get_weather()
    event_proximity_km : optional explicit event proximity
    """
    congestion = traffic.get("congestion_index", 0.3)
    weather_sev = weather.get("severity", 0.1)
    incident_prox = traffic.get("incident_proximity_km", 99.0)

    if event_proximity_km is None:
        event_proximity_km = 99.0

    road_risk = min(1.0, max(0.0,
        0.35 * congestion
        + 0.30 * weather_sev
        + 0.20 * (1.0 / (1.0 + incident_prox))
        + 0.15 * (1.0 / (1.0 + event_proximity_km))
    ))

    return ContextFeatures(
        congestion_index=round(congestion, 4),
        weather_severity=round(weather_sev, 4),
        incident_proximity=round(incident_prox, 2),
        event_proximity=round(event_proximity_km, 2),
        road_risk_score=round(road_risk, 4),
    )
