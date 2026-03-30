
from datetime import datetime, date, timedelta
from typing import Optional
from pydantic import BaseModel, Field

from loguru import logger


class IndianCalendarFeatures(BaseModel):
    """India-specific calendar features that affect traffic."""
    is_national_holiday: bool = False
    is_festival: bool = False
    is_monsoon_season: bool = False
    is_school_hours: bool = False
    is_market_day: bool = False
    is_weekend_market: bool = False
    festival_severity: float = Field(default=0.0, ge=0.0, le=1.0)
    monsoon_severity: float = Field(default=0.0, ge=0.0, le=1.0)
    event_name: str = ""
    days_to_nearest_festival: int = 365
    is_dry_day: bool = False  # Alcohol ban days = less late-night traffic
    is_election_day: bool = False
    is_bandh_likely: bool = False
    school_proximity_factor: float = Field(default=0.0, ge=0.0, le=1.0)



_NATIONAL_HOLIDAYS_2026 = {
    date(2026, 1, 26): ("Republic Day", 0.8),
    date(2026, 5, 1): ("May Day", 0.4),
    date(2026, 8, 15): ("Independence Day", 0.8),
    date(2026, 10, 2): ("Gandhi Jayanti", 0.6),
    date(2026, 11, 14): ("Children's Day", 0.3),
}


_FESTIVALS_2026 = {
    date(2026, 1, 14): ("Makar Sankranti / Pongal", 0.7),
    date(2026, 2, 19): ("Maha Shivaratri", 0.5),
    date(2026, 3, 14): ("Holi (Holika Dahan)", 0.6),
    date(2026, 3, 15): ("Holi", 0.8),
    date(2026, 3, 30): ("Eid al-Fitr (approx)", 0.7),
    date(2026, 4, 2): ("Ram Navami", 0.5),
    date(2026, 4, 6): ("Ugadi / Gudi Padwa", 0.6),
    date(2026, 4, 10): ("Good Friday", 0.4),
    date(2026, 6, 6): ("Eid al-Adha (approx)", 0.6),
    date(2026, 8, 11): ("Raksha Bandhan", 0.5),
    date(2026, 8, 14): ("Janmashtami (approx)", 0.6),
    date(2026, 8, 26): ("Ganesh Chaturthi", 0.9),  # Massive processions
    date(2026, 9, 5): ("Ganesh Visarjan", 0.95),   # Road closures
    date(2026, 10, 2): ("Dussehra / Vijayadashami", 0.7),
    date(2026, 10, 19): ("Diwali (Lakshmi Puja)", 0.8),
    date(2026, 10, 20): ("Diwali (Main Day)", 0.9),
    date(2026, 10, 21): ("Govardhan Puja", 0.6),
    date(2026, 10, 22): ("Bhai Dooj", 0.5),
    date(2026, 11, 2): ("Guru Nanak Jayanti", 0.5),
    date(2026, 12, 25): ("Christmas", 0.5),
}

_PRE_FESTIVAL_HEAVY = {
    "Diwali": (date(2026, 10, 16), date(2026, 10, 19), 0.85),
    "Ganesh Chaturthi": (date(2026, 8, 24), date(2026, 8, 26), 0.7),
    "Holi": (date(2026, 3, 13), date(2026, 3, 15), 0.5),
    "Eid": (date(2026, 3, 28), date(2026, 3, 30), 0.6),
}

_MONSOON_PERIODS = [
    (date(2026, 6, 10), date(2026, 6, 30), 0.4),   # Onset, intermittent
    (date(2026, 7, 1), date(2026, 8, 15), 0.7),     # Peak monsoon
    (date(2026, 8, 16), date(2026, 9, 15), 0.5),    # Moderate
    (date(2026, 9, 16), date(2026, 10, 5), 0.3),    # Retreating
]

# ── School timing patterns ────────────────────────────────────────────────

_SCHOOL_MORNING = range(7, 10)   # 7:00 - 9:30 AM
_SCHOOL_AFTERNOON = range(13, 16)  # 1:00 - 3:30 PM (shift schools)
_SCHOOL_VACATION_PERIODS = [
    (date(2026, 4, 15), date(2026, 6, 14)),  # Summer vacation
    (date(2026, 10, 15), date(2026, 10, 30)),  # Diwali vacation
    (date(2026, 12, 20), date(2027, 1, 5)),    # Winter vacation
]


def compute_indian_calendar_features(
    dt: Optional[datetime] = None,
) -> IndianCalendarFeatures:

    if dt is None:
        dt = datetime.now()

    d = dt.date()
    hour = dt.hour
    dow = dt.weekday()  # 0=Monday

    # ── National Holiday ──
    is_holiday = d in _NATIONAL_HOLIDAYS_2026
    holiday_name = _NATIONAL_HOLIDAYS_2026.get(d, ("", 0))[0]

    # ── Festival ──
    is_festival = d in _FESTIVALS_2026
    festival_info = _FESTIVALS_2026.get(d, ("", 0.0))
    festival_name = festival_info[0]
    festival_severity = festival_info[1] if is_festival else 0.0

    # Check pre-festival shopping rush
    for fest_name, (start, end, sev) in _PRE_FESTIVAL_HEAVY.items():
        if start <= d <= end:
            if not is_festival:
                is_festival = True
                festival_name = f"Pre-{fest_name} Rush"
                festival_severity = sev

    event_name = festival_name or holiday_name

    # Days to nearest festival
    days_to_festival = 365
    for fd in list(_FESTIVALS_2026.keys()) + list(_NATIONAL_HOLIDAYS_2026.keys()):
        diff = (fd - d).days
        if diff >= 0:
            days_to_festival = min(days_to_festival, diff)

    # ── Monsoon ──
    is_monsoon = False
    monsoon_sev = 0.0
    for start, end, sev in _MONSOON_PERIODS:
        if start <= d <= end:
            is_monsoon = True
            monsoon_sev = sev
            break

    # ── School Hours ──
    is_school_vacation = any(start <= d <= end for start, end in _SCHOOL_VACATION_PERIODS)
    is_school_hours = False
    school_factor = 0.0

    if not is_school_vacation and dow < 6:  # Not vacation and not Sunday
        if hour in _SCHOOL_MORNING:
            is_school_hours = True
            school_factor = 0.7 if 7 <= hour <= 8 else 0.4
        elif hour in _SCHOOL_AFTERNOON:
            is_school_hours = True
            school_factor = 0.5 if 14 <= hour <= 15 else 0.3

    # Saturday half-day school
    if not is_school_vacation and dow == 5 and hour in _SCHOOL_MORNING:
        is_school_hours = True
        school_factor = 0.4


    is_market = dow == 5 and 9 <= hour <= 20  # Saturday shopping
    is_weekend_market = dow == 6 and 7 <= hour <= 13  # Sunday morning markets

    # ── Special flags ──
    # Dry days (before major festivals, Republic Day, Independence Day, Gandhi Jayanti)
    dry_day_dates = [date(2026, 1, 26), date(2026, 8, 15), date(2026, 10, 2)]
    is_dry = d in dry_day_dates

    return IndianCalendarFeatures(
        is_national_holiday=is_holiday,
        is_festival=is_festival,
        is_monsoon_season=is_monsoon,
        is_school_hours=is_school_hours,
        is_market_day=is_market,
        is_weekend_market=is_weekend_market,
        festival_severity=round(festival_severity, 3),
        monsoon_severity=round(monsoon_sev, 3),
        event_name=event_name,
        days_to_nearest_festival=days_to_festival,
        is_dry_day=is_dry,
        is_election_day=False,
        is_bandh_likely=False,
        school_proximity_factor=round(school_factor, 3),
    )


def get_traffic_multiplier(features: IndianCalendarFeatures) -> float:
    multiplier = 1.0

    if features.is_national_holiday:
        multiplier *= 0.7   # Less traffic on holidays (opposite of festival)

    if features.is_festival:
        multiplier *= (1.0 + features.festival_severity * 0.5)

    if features.is_monsoon_season:
        multiplier *= (1.0 + features.monsoon_severity * 0.3)

    if features.is_school_hours:
        multiplier *= (1.0 + features.school_proximity_factor * 0.2)

    if features.is_market_day:
        multiplier *= 1.15

    if features.is_weekend_market:
        multiplier *= 1.10

    return round(multiplier, 3)
