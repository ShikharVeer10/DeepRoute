import random
import math
from datetime import datetime
from loguru import logger

def _peak_factor(hour: int) -> float:
    morning = math.exp(-0.5 * ((hour - 8.5) / 1.5) ** 2)
    evening = math.exp(-0.5 * ((hour - 17.5) / 1.5) ** 2)
    return max(morning, evening)


def get_traffic(hour: int | None = None) -> dict:
    if hour is None:
        hour = datetime.now().hour
        
    day_of_week = datetime.now().weekday()
    weekend_factor = 0.6 if day_of_week >= 5 else 1.0

    base_congestion = _peak_factor(hour) * weekend_factor
    noise = random.gauss(0, 0.08)
    congestion = max(0.0, min(1.0, base_congestion + noise))

    free_flow_speed = random.uniform(60, 100)
    avg_speed = free_flow_speed * (1 - 0.7 * congestion)

    incident = random.random() < 0.05
    incident_proximity = random.expovariate(0.2) if incident else 99.0

    traffic_data = {
        "congestion_index": round(congestion, 4),
        "avg_speed_kph": round(avg_speed, 2),
        "incident_active": incident,
        "incident_proximity_km": round(incident_proximity, 2),
    }

    logger.debug(f"Traffic data: congestion={traffic_data['congestion_index']:.3f}, "
                 f"speed={traffic_data['avg_speed_kph']:.1f}kph, incident={incident}")

    return traffic_data