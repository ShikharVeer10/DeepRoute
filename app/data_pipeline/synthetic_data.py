import numpy as np
import pandas as pd
from pathlib import Path


def _peak_hour_factor(hour: int) -> float:
    """Return a congestion multiplier based on typical Indian peak patterns.
    
    Indian peak hours:
      - Morning: 8:00-10:30 AM (office/school rush)
      - Evening: 5:00-8:00 PM (return + market rush)
    """
    morning_peak = np.exp(-0.5 * ((hour - 9.0) / 1.5) ** 2)
    evening_peak = np.exp(-0.5 * ((hour - 18.0) / 2.0) ** 2)
    return max(morning_peak, evening_peak)


def _weekend_factor(day_of_week: int) -> float:
    """Weekends have lighter traffic on highways."""
    return 0.5 if day_of_week >= 5 else 1.0


def generate_training_data(
    n_samples: int = 10000,
    n_segments: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic training data for XGBoost travel-time factor prediction.
    
    The travel_time_factor represents how much slower traffic moves compared to
    OSRM free-flow estimates, AFTER the India road correction is applied.
    
    Calibration target (Hyderabad → Bangalore):
      - OSRM free-flow: ~7h 37min
      - India correction (×1.27): ~9h 41min  ← matches Google Maps
      - ML factor adds traffic/weather adjustments on top:
        * Clear conditions: factor ≈ 0.97-1.03 (marginal adjustment)
        * Moderate traffic: factor ≈ 1.03-1.10
        * Heavy traffic + peak: factor ≈ 1.10-1.25
        * Extreme (monsoon + accident): factor ≈ 1.25-1.50
    """
    rng = np.random.RandomState(seed)

    hours = rng.randint(0, 24, size=n_samples)
    days = rng.randint(0, 7, size=n_samples)

    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    day_sin = np.sin(2 * np.pi * days / 7)
    day_cos = np.cos(2 * np.pi * days / 7)

    is_peak = np.array([_peak_hour_factor(h) > 0.4 for h in hours]).astype(float)
    is_weekend = (days >= 5).astype(float)

    length_m = rng.uniform(50, 5000, size=n_samples)
    speed_limit = rng.choice([30, 40, 50, 60, 80, 100, 120], size=n_samples).astype(float)
    num_lanes = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.1, 0.5, 0.3, 0.1]).astype(float)
    elevation = rng.uniform(-50, 50, size=n_samples)

    # Congestion index (0-1) based on time of day & day of week
    base_congestion = np.array([_peak_hour_factor(h) * _weekend_factor(d)
                                for h, d in zip(hours, days)])
    congestion = np.clip(base_congestion + rng.normal(0, 0.08, n_samples), 0, 1)

    weather_sev = np.clip(rng.beta(2, 8, size=n_samples), 0, 1)  # mostly clear
    incident_prox = rng.exponential(8, size=n_samples)
    event_prox = rng.exponential(15, size=n_samples)
    
    # External events (rare)
    road_closure_active = (rng.random(n_samples) < 0.03).astype(float)   # 3% chance
    roadworks_active = (rng.random(n_samples) < 0.10).astype(float)      # 10% chance
    accident_active = (rng.random(n_samples) < 0.06).astype(float)       # 6% chance
    
    # Indian calendar features
    is_festival = (rng.random(n_samples) < 0.05).astype(float)
    festival_severity = rng.uniform(0.5, 1.0, n_samples) * is_festival
    is_monsoon = (rng.random(n_samples) < 0.25).astype(float)
    monsoon_severity = rng.uniform(0.3, 0.9, n_samples) * is_monsoon
    is_school_hours = (rng.random(n_samples) < 0.20).astype(float)
    is_market_day = (rng.random(n_samples) < 0.30).astype(float)
    
    # Historical context
    hist_speed = speed_limit * rng.uniform(0.7, 1.1, n_samples)
    hist_cong = np.clip(base_congestion + rng.normal(0, 0.15, n_samples), 0, 1)
    speed_rel = rng.uniform(0.4, 0.9, n_samples)
    
    # Additional features from literature
    road_type_enc = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.1, 0.3, 0.4, 0.2]).astype(float)  # motorway=0, primary=1, secondary=2, residential=3
    highway_pct = np.clip(1.0 - road_type_enc * 0.25 + rng.normal(0, 0.1, n_samples), 0, 1)
    route_curvature = rng.exponential(0.15, n_samples)  # Higher = more curves
    intersection_count = rng.poisson(5, n_samples).astype(float) * (1 + road_type_enc * 2)  # More in urban
    toll_roads = (rng.random(n_samples) < (0.3 * highway_pct)).astype(float)
    urban_density = np.clip(road_type_enc * 0.25 + rng.normal(0, 0.1, n_samples), 0, 1)
    distance_category = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.25, 0.35, 0.25, 0.15]).astype(float)  # short=0, medium=1, long=2, very_long=3
    
    risk_score = np.clip(
        0.3 * congestion + 0.2 * weather_sev + 0.2 * (1 / (1 + incident_prox))
        + 0.3 * accident_active + 0.05 * urban_density + rng.normal(0, 0.03, n_samples),
        0, 1
    )

    # ── Target: travel_time_factor ────────────────────────────────────────────
    # This factor represents the ADDITIONAL adjustment on top of the base
    # OSRM + India correction time. It should be centered around 1.0.
    #
    # The formula produces values in the range [0.92, 1.50]:
    #   - Base: 1.0 (no adjustment)
    #   - Congestion adds up to +15% (peak hour, heavy traffic)
    #   - Weather adds up to +8% (severe weather)
    #   - Incidents add up to +5% if nearby
    #   - Road closures/accidents add significant delays
    #   - Weekends/off-peak can reduce by up to 5%
    
    # Congestion component: 0 to +0.08
    congestion_adj = congestion * 0.08
    
    # Weather component: 0 to +0.05
    weather_adj = weather_sev * 0.05
    
    # Time-of-day component: -0.02 to +0.03
    peak_factors = np.array([_peak_hour_factor(h) for h in hours])
    time_adj = peak_factors * 0.03 - (1 - peak_factors) * 0.02
    
    # Weekend discount: 0 to -0.03
    weekend_adj = -is_weekend * 0.03
    
    # Incident proximity: 0 to +0.03
    incident_adj = 0.03 * np.exp(-incident_prox / 3)
    
    # External events (rare but significant)
    event_adj = (
        road_closure_active * 0.20     # Road closure: +20%
        + roadworks_active * 0.06      # Roadworks: +6%
        + accident_active * 0.12       # Active accident: +12%
        + festival_severity * 0.04     # Festival: up to +4%
        + monsoon_severity * 0.04      # Monsoon: up to +4%
    )
    
    # Lane capacity effect (more lanes = slightly faster)
    lane_adj = (2 - num_lanes) * 0.02  # 1 lane: +0.02, 4 lanes: -0.04
    
    # Road type / urbanization effects
    urban_adj = urban_density * 0.03  # Urban areas are slower
    curvature_adj = route_curvature * 0.02  # Curves slow you down
    intersection_adj = np.clip(intersection_count / 50.0, 0, 0.04)  # Many intersections slow
    highway_adj = -highway_pct * 0.02  # Highways are faster
    toll_adj = -toll_roads * 0.01  # Toll roads are maintained better
    
    travel_time_factor = (
        1.0
        + congestion_adj
        + weather_adj
        + time_adj
        + weekend_adj
        + incident_adj
        + event_adj
        + lane_adj
        + urban_adj
        + curvature_adj
        + intersection_adj
        + highway_adj
        + toll_adj
        + rng.normal(0, 0.012, n_samples)  # reduced noise for better signal
    )
    travel_time_factor = np.clip(travel_time_factor, 0.92, 1.50)

    df = pd.DataFrame({
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "is_peak_hour": is_peak,
        "is_weekend": is_weekend,
        "is_festival": is_festival,
        "festival_severity": festival_severity,
        "is_monsoon_season": is_monsoon,
        "monsoon_severity": monsoon_severity,
        "is_school_hours": is_school_hours,
        "is_market_day": is_market_day,
        "length_m": length_m,
        "speed_limit_kph": speed_limit,
        "num_lanes": num_lanes,
        "elevation_change_m": elevation,
        "congestion_index": congestion,
        "weather_severity": weather_sev,
        "incident_proximity": incident_prox,
        "event_proximity": event_prox,
        "road_risk_score": risk_score,
        "road_closure_active": road_closure_active,
        "roadworks_active": roadworks_active,
        "accident_active": accident_active,
        "historical_speed_kph": hist_speed,
        "historical_congestion": hist_cong,
        "speed_reliability": speed_rel,
        "road_type_encoded": road_type_enc,
        "highway_percentage": highway_pct,
        "route_curvature": route_curvature,
        "intersection_count": intersection_count,
        "toll_roads": toll_roads,
        "urban_density": urban_density,
        "distance_category": distance_category,
        # ── target ────────────────────────────
        "travel_time_factor": travel_time_factor,
    })

    return df


def generate_sequence_data(
    n_sequences: int = 2000,
    seq_length: int = 12,
    n_features: int = 15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    df = generate_training_data(n_samples=n_sequences + seq_length, seed=seed)
    feature_cols = [c for c in df.columns if c != "travel_time_factor"]
    values = df[feature_cols].values
    targets = df["travel_time_factor"].values

    X, y = [], []
    for i in range(len(values) - seq_length):
        X.append(values[i : i + seq_length])
        y.append(targets[i + seq_length])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)


def generate_graph_data(
    n_nodes: int = 200,
    avg_degree: int = 4,
    n_features: int = 15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)

    df = generate_training_data(n_samples=n_nodes, seed=seed)
    feature_cols = [c for c in df.columns if c != "travel_time_factor"]
    node_features = df[feature_cols].values.astype(np.float32)
    node_targets = df["travel_time_factor"].values.astype(np.float32).reshape(-1, 1)

    # Build random edges (undirected road network)
    edges = set()
    for node in range(n_nodes):
        n_neighbors = rng.randint(1, avg_degree + 1)
        neighbors = rng.choice(n_nodes, size=n_neighbors, replace=False)
        for nb in neighbors:
            if nb != node:
                edges.add((node, nb))
                edges.add((nb, node))

    edge_list = list(edges)
    edge_index = np.array(edge_list, dtype=np.int64).T  # shape (2, n_edges)

    return node_features, edge_index, node_targets


def save_datasets(output_dir: str = "data") -> None:
    """Generate and persist all training datasets."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Tabular
    df = generate_training_data()
    df.to_csv(out / "training_data.csv", index=False)

    # Sequence (kept for compatibility)
    X_seq, y_seq = generate_sequence_data()
    np.save(out / "X_seq.npy", X_seq)
    np.save(out / "y_seq.npy", y_seq)

    # Graph (kept for compatibility)
    node_feat, edge_idx, node_tgt = generate_graph_data()
    np.save(out / "node_features.npy", node_feat)
    np.save(out / "edge_index.npy", edge_idx)
    np.save(out / "node_targets.npy", node_tgt)

    print(f"[DataGen] Saved datasets to {out.resolve()}")
    print(f"  - training_data.csv     : {df.shape}")
    print(f"  - travel_time_factor range: [{df['travel_time_factor'].min():.3f}, {df['travel_time_factor'].max():.3f}]")
    print(f"  - travel_time_factor mean:  {df['travel_time_factor'].mean():.3f}")


if __name__ == "__main__":
    save_datasets()
