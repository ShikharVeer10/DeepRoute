"""
Temporal feature extraction with cyclical encoding.
"""

import math
from app.schemas import TemporalFeatures


_PEAK_HOURS_MORNING = range(7, 10)
_PEAK_HOURS_EVENING = range(16, 19)
_DAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def temporal_features(day: str, time: str) -> TemporalFeatures:
    """
    Extract temporal features from a day name and HH:MM time string.

    Returns a validated TemporalFeatures Pydantic model with:
      - raw hour / day_of_week
      - cyclical sin/cos encodings
      - boolean peak-hour / weekend flags
    """
    hour = int(time.split(":")[0])
    day_index = _DAY_MAP.get(day.lower(), hash(day) % 7)

    is_peak = hour in _PEAK_HOURS_MORNING or hour in _PEAK_HOURS_EVENING
    is_weekend = day_index >= 5

    return TemporalFeatures(
        hour=hour,
        day_of_week=day_index,
        is_peak_hour=is_peak,
        hour_sin=round(math.sin(2 * math.pi * hour / 24), 6),
        hour_cos=round(math.cos(2 * math.pi * hour / 24), 6),
        day_sin=round(math.sin(2 * math.pi * day_index / 7), 6),
        day_cos=round(math.cos(2 * math.pi * day_index / 7), 6),
        is_weekend=is_weekend,
    )