"""
Model selection utility with optional Google Maps alignment scoring.

This script compares tabular regressors on DeepRoute synthetic data and,
when a Google Maps API key is available, adds ETA alignment scoring against
Google Directions (traffic-aware duration).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.data_pipeline.synthetic_data import generate_training_data
from app.features.feature_builder import build_features

try:
    from xgboost import XGBRegressor

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


INDIA_ROAD_CORRECTION = 1.27
REPORT_PATH = Path("data/models/google_alignment_report.json")


@dataclass(frozen=True)
class ODPair:
    name: str
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float


OD_PAIRS = [
    ODPair("Hyderabad-Bengaluru", 17.3850, 78.4867, 12.9716, 77.5946),
    ODPair("Delhi-Gurugram", 28.6139, 77.2090, 28.4595, 77.0266),
    ODPair("Mumbai-Pune", 19.0760, 72.8777, 18.5204, 73.8567),
    ODPair("Chennai-Vellore", 13.0827, 80.2707, 12.9165, 79.1325),
    ODPair("Kolkata-Durgapur", 22.5726, 88.3639, 23.5204, 87.3119),
]


def _decode_polyline(polyline_str: str) -> list[tuple[float, float]]:
    """Decode Google encoded polyline into (lat, lon) points."""
    coords: list[tuple[float, float]] = []
    lat = 0
    lon = 0
    i = 0

    while i < len(polyline_str):
        shift = 0
        result = 0
        while True:
            b = ord(polyline_str[i]) - 63
            i += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift = 0
        result = 0
        while True:
            b = ord(polyline_str[i]) - 63
            i += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlon = ~(result >> 1) if (result & 1) else (result >> 1)
        lon += dlon

        coords.append((lat / 1e5, lon / 1e5))

    return coords


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def _overlap_score(
    path_a: list[tuple[float, float]],
    path_b: list[tuple[float, float]],
    tolerance_m: float = 300.0,
) -> float:
    """Bidirectional point-overlap proxy score in [0, 1]."""
    if not path_a or not path_b:
        return 0.0

    def _ratio(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> float:
        hit = 0
        for la, loa in src:
            nearest = min(_haversine_m(la, loa, lb, lob) for lb, lob in dst)
            if nearest <= tolerance_m:
                hit += 1
        return hit / len(src)

    return round(float((_ratio(path_a, path_b) + _ratio(path_b, path_a)) / 2), 4)


def _fetch_osrm_route(od: ODPair) -> dict | None:
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{od.origin_lon},{od.origin_lat};{od.dest_lon},{od.dest_lat}"
    )
    params = {"overview": "full", "geometries": "geojson", "alternatives": "false"}
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("routes"):
            return None
        route = data["routes"][0]
        coords = route.get("geometry", {}).get("coordinates", [])
        return {
            "duration_s": float(route.get("duration", 0.0)),
            "distance_m": float(route.get("distance", 0.0)),
            "path": [(float(lat), float(lon)) for lon, lat in coords],
        }
    except Exception:
        return None


def _fetch_google_route(od: ODPair, api_key: str) -> dict | None:
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{od.origin_lat},{od.origin_lon}",
        "destination": f"{od.dest_lat},{od.dest_lon}",
        "departure_time": "now",
        "traffic_model": "best_guess",
        "alternatives": "false",
        "key": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK" or not data.get("routes"):
            return None

        route = data["routes"][0]
        leg = route.get("legs", [{}])[0]
        dur_traffic = float(leg.get("duration_in_traffic", {}).get("value", 0.0))
        dur_normal = float(leg.get("duration", {}).get("value", 0.0))
        dist_m = float(leg.get("distance", {}).get("value", 0.0))

        poly = route.get("overview_polyline", {}).get("points", "")
        path = _decode_polyline(poly) if poly else []

        return {
            "duration_traffic_s": dur_traffic,
            "duration_normal_s": dur_normal,
            "distance_m": dist_m,
            "path": path,
        }
    except Exception:
        return None


def _build_candidates(seed: int = 42) -> dict[str, object]:
    candidates: dict[str, object] = {
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "gbm": GradientBoostingRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            random_state=seed,
        ),
    }

    if _HAS_XGB:
        candidates["xgboost"] = XGBRegressor(
            n_estimators=700,
            max_depth=5,
            learning_rate=0.025,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.02,
            reg_lambda=1.5,
            random_state=seed,
            tree_method="hist",
            verbosity=0,
            eval_metric="mae",
        )
    return candidates


def select_best_model(
    n_samples: int = 10000,
    seed: int = 42,
    google_maps_api_key: str | None = None,
) -> dict:
    """
    Compare candidate models and return a decision report.

    Score components:
      - predictive accuracy (MAE/RMSE/R2 on holdout synthetic set)
      - ETA alignment vs Google Directions (if API key is provided)
      - route overlap baseline (OSRM vs Google, model-independent)
    """
    google_key = google_maps_api_key or os.getenv("GOOGLE_MAPS_API_KEY", "")

    df = generate_training_data(n_samples=n_samples, seed=seed)
    feature_cols = [c for c in df.columns if c != "travel_time_factor"]
    X = df[feature_cols].values
    y = df["travel_time_factor"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    models = _build_candidates(seed=seed)
    tabular: dict[str, dict] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))
        mape = float(np.mean(np.abs((y_test - pred) / y_test)) * 100.0)
        tabular[name] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape_percent": mape,
        }

    google_eval_enabled = bool(google_key)
    google_coverage = 0
    overlap_values: list[float] = []
    google_metrics: dict[str, list[float]] = {k: [] for k in models.keys()}

    if google_eval_enabled:
        for od in OD_PAIRS:
            osrm_ref = _fetch_osrm_route(od)
            google_ref = _fetch_google_route(od, google_key)
            if not osrm_ref or not google_ref or google_ref["duration_traffic_s"] <= 0:
                continue

            google_coverage += 1
            overlap_values.append(_overlap_score(osrm_ref["path"], google_ref["path"]))

            feat = build_features(
                origin_lat=od.origin_lat,
                origin_lon=od.origin_lon,
                departure_time=datetime.now().isoformat(),
            )
            x_row = np.array(feat.to_flat_list(), dtype=np.float32).reshape(1, -1)
            for name, model in models.items():
                factor = float(model.predict(x_row)[0])
                factor = max(0.80, min(1.80, factor))

                eta_pred_s = osrm_ref["duration_s"] * INDIA_ROAD_CORRECTION * factor
                eta_google_s = google_ref["duration_traffic_s"]

                abs_pct = abs(eta_pred_s - eta_google_s) / eta_google_s * 100.0
                google_metrics[name].append(float(abs_pct))

    model_names = list(models.keys())
    mae_vals = np.array([tabular[m]["mae"] for m in model_names], dtype=np.float64)
    mae_min = float(mae_vals.min())
    mae_max = float(mae_vals.max())

    eta_vals: dict[str, float | None] = {}
    if google_eval_enabled and google_coverage > 0:
        for m in model_names:
            vals = google_metrics.get(m, [])
            eta_vals[m] = float(np.mean(vals)) if vals else None
        valid_eta = [v for v in eta_vals.values() if v is not None]
        eta_min = min(valid_eta) if valid_eta else None
        eta_max = max(valid_eta) if valid_eta else None
    else:
        eta_min = None
        eta_max = None
        for m in model_names:
            eta_vals[m] = None

    combined_scores: dict[str, float] = {}
    for m in model_names:
        mae = tabular[m]["mae"]
        if mae_max > mae_min:
            mae_norm = (mae - mae_min) / (mae_max - mae_min)
        else:
            mae_norm = 0.0
        acc_score = 1.0 - mae_norm

        eta = eta_vals[m]
        if (
            eta is not None
            and eta_min is not None
            and eta_max is not None
            and eta_max > eta_min
        ):
            eta_norm = (eta - eta_min) / (eta_max - eta_min)
            eta_score = 1.0 - eta_norm
            combined = 0.65 * acc_score + 0.35 * eta_score
        elif eta is not None and eta_min is not None and eta_max is not None:
            combined = acc_score
        else:
            combined = acc_score

        combined_scores[m] = float(round(combined, 6))

    winner = max(model_names, key=lambda n: combined_scores[n])

    report = {
        "generated_at": datetime.now().isoformat(),
        "decision": {
            "selected_model": winner,
            "scoring": {
                "accuracy_weight": 0.65,
                "google_eta_weight": 0.35
                if google_eval_enabled and google_coverage > 0
                else 0.0,
            },
        },
        "tabular_metrics": tabular,
        "google_alignment": {
            "enabled": google_eval_enabled,
            "sampled_od_pairs": len(OD_PAIRS),
            "successful_pairs": google_coverage,
            "mean_osrm_google_path_overlap": (
                float(np.mean(overlap_values)) if overlap_values else None
            ),
            "eta_mape_percent": {
                k: (float(np.mean(v)) if v else None) for k, v in google_metrics.items()
            },
        },
        "combined_scores": combined_scores,
        "notes": [
            "Path overlap measures OSRM vs Google baseline geometry and is model-independent.",
            "Model-specific Google alignment uses ETA closeness to traffic-aware Google duration.",
            "If GOOGLE_MAPS_API_KEY is missing, selection is based on predictive accuracy only.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    out = select_best_model()
    sel = out["decision"]["selected_model"]
    print(f"[ModelSelection] Selected model: {sel}")
    print(f"[ModelSelection] Report: {REPORT_PATH}")
