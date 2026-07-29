"""
Weather data loader using Open-Meteo API (FREE, no API key needed).
Falls back to deterministic data only if the API call fails.

Open-Meteo: https://open-meteo.com/
- No API key required
- No sign-up required
- Free for non-commercial / open-source use
"""
import requests
from loguru import logger


# WMO Weather interpretation codes → our internal conditions
_WMO_CODE_MAP = {
    0: "clear",           # Clear sky
    1: "clear",           # Mainly clear
    2: "cloudy",          # Partly cloudy
    3: "cloudy",          # Overcast
    45: "fog",            # Fog
    48: "fog",            # Depositing rime fog
    51: "rain",           # Drizzle: light
    53: "rain",           # Drizzle: moderate
    55: "rain",           # Drizzle: dense
    56: "rain",           # Freezing drizzle: light
    57: "rain",           # Freezing drizzle: dense
    61: "rain",           # Rain: slight
    63: "rain",           # Rain: moderate
    65: "heavy_rain",     # Rain: heavy
    66: "rain",           # Freezing rain: light
    67: "heavy_rain",     # Freezing rain: heavy
    71: "snow",           # Snow fall: slight
    73: "snow",           # Snow fall: moderate
    75: "snow",           # Snow fall: heavy
    77: "snow",           # Snow grains
    80: "rain",           # Rain showers: slight
    81: "rain",           # Rain showers: moderate
    82: "heavy_rain",     # Rain showers: violent
    85: "snow",           # Snow showers: slight
    86: "snow",           # Snow showers: heavy
    95: "thunderstorm",   # Thunderstorm
    96: "thunderstorm",   # Thunderstorm with slight hail
    99: "thunderstorm",   # Thunderstorm with heavy hail
}

_SEVERITY_MAP = {
    "clear": (0.0, 0.05),
    "cloudy": (0.05, 0.15),
    "rain": (0.2, 0.45),
    "heavy_rain": (0.45, 0.7),
    "fog": (0.3, 0.6),
    "snow": (0.5, 0.8),
    "thunderstorm": (0.7, 0.95),
}


def get_weather(lat: float | None = None, lon: float | None = None) -> dict:
    """
    Get weather data using Open-Meteo API (FREE, no key needed).
    Falls back to simulated data if API call fails.

    Parameters
    ----------
    lat : latitude (optional, needed for API calls)
    lon : longitude (optional, needed for API calls)

    Returns
    -------
    dict with keys: condition, severity, temperature_c, visibility_km, source
    """
    if lat is not None and lon is not None:
        try:
            return _fetch_open_meteo(lat, lon)
        except Exception as e:
            logger.warning(f"Open-Meteo API failed, using simulated data: {e}")

    return _simulate_weather()


def _fetch_open_meteo(lat: float, lon: float) -> dict:
    """
    Fetch real weather data from Open-Meteo API.
    No API key required!

    API Docs: https://open-meteo.com/en/docs
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "rain",
            "snowfall",
            "weather_code",
            "wind_speed_10m",
            "cloud_cover",
        ]),
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=8)
    response.raise_for_status()
    data = response.json()

    current = data.get("current", {})

    # Map WMO weather code to our condition
    weather_code = current.get("weather_code", 0)
    condition = _WMO_CODE_MAP.get(weather_code, "clear")

    # Calculate severity from weather factors
    rain_mm = current.get("rain", 0) or 0
    snowfall = current.get("snowfall", 0) or 0
    wind_speed = current.get("wind_speed_10m", 0) or 0
    cloud_cover = current.get("cloud_cover", 0) or 0

    # Severity based on precipitation + wind + clouds
    precip_severity = min(1.0, (rain_mm + snowfall * 3) / 10.0)
    wind_severity = min(1.0, wind_speed / 50.0)
    cloud_severity = cloud_cover / 100.0 * 0.15

    severity = round(
        0.50 * precip_severity + 0.30 * wind_severity + 0.20 * cloud_severity,
        4,
    )

    # Clamp to condition's known range
    sev_lo, sev_hi = _SEVERITY_MAP.get(condition, (0.0, 1.0))
    severity = max(sev_lo, min(sev_hi, severity))

    temperature = current.get("temperature_2m", 25.0)

    # Deterministic visibility estimate derived from condition severity.
    if condition == "fog":
        visibility_km = max(0.5, 3.0 - severity * 2.5)
    elif condition in ("heavy_rain", "thunderstorm", "snow"):
        visibility_km = max(1.0, 6.0 - severity * 4.0)
    elif condition == "rain":
        visibility_km = max(2.5, 8.0 - severity * 3.0)
    else:
        visibility_km = max(5.0, 10.0 - severity * 5.0)

    weather_data = {
        "condition": condition,
        "severity": severity,
        "temperature_c": round(temperature, 1),
        "visibility_km": max(0.1, visibility_km),
        "wind_speed_kmh": round(wind_speed, 1),
        "humidity_percent": current.get("relative_humidity_2m", 50),
        "cloud_cover_percent": cloud_cover,
        "rain_mm": rain_mm,
        "weather_code": weather_code,
        "source": "Open-Meteo (Live)",
    }

    logger.info(
        f"Weather (Open-Meteo): code={weather_code}, condition={condition}, "
        f"severity={severity:.3f}, temp={temperature}°C, "
        f"wind={wind_speed}km/h, rain={rain_mm}mm"
    )

    return weather_data


def _simulate_weather() -> dict:
    """Generate deterministic fallback weather data."""
    condition = "clear"
    severity = 0.05
    temperature = 28.0
    visibility = 10.0

    weather_data = {
        "condition": condition,
        "severity": severity,
        "temperature_c": temperature,
        "visibility_km": visibility,
        "source": "Deterministic fallback",
    }

    logger.debug(
        f"Weather (Fallback): condition={condition}, severity={severity:.3f}, "
        f"temp={temperature}°C, visibility={visibility}km"
    )

    return weather_data
