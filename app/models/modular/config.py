
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.pkl"
LIGHTGBM_MODEL_PATH = MODELS_DIR / "lightgbm.pkl"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost.pkl"
DEEP_MODEL_PATH = MODELS_DIR / "deep_route_model.pth"

N_SAMPLES = 10000
TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURE_NAMES = [
    "hour_sin", "hour_cos", "day_sin", "day_cos", "is_peak_hour", "is_weekend",
    "length_m", "speed_limit_kph", "num_lanes", "road_type_encoded", "elevation_change_m",
    "highway_percentage", "route_curvature", "urban_density", "intersection_count", "toll_booth_count",
    "congestion_index", "weather_severity", "incident_proximity", "event_proximity",
    "road_risk_score", "road_closure_active", "roadworks_active", "accident_active",
    "historical_speed_kph", "historical_congestion", "speed_reliability",
    "temp_c", "wind_kph", "precip_mm", "visibility_km", "is_night", "osrm_base_duration_s",
]

OPTUNA_N_TRIALS = 30
CV_FOLDS = 5
