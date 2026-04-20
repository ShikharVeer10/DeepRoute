
from app.schemas import SpatialFeatures


_ROAD_TYPE_MAP = {
    "motorway": 120, "trunk": 100, "primary": 80, "secondary": 60,
    "tertiary": 50, "residential": 30, "service": 20, "unclassified": 40,
}


def spatial_features(
    length_m: float = 500.0,
    speed_limit_kph: float | None = None,
    num_lanes: int = 2,
    road_type: str = "residential",
    elevation_change_m: float = 0.0,
) -> SpatialFeatures:
    """
    Extract spatial features for a road segment.

    If speed_limit_kph is not provided, it is inferred from road_type.
    """
    if speed_limit_kph is None:
        speed_limit_kph = float(_ROAD_TYPE_MAP.get(road_type, 40))

    return SpatialFeatures(
        length_m=length_m,
        speed_limit_kph=speed_limit_kph,
        num_lanes=max(1, num_lanes),
        road_type=road_type,
        elevation_change_m=elevation_change_m,
    )