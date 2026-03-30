import numpy as np
import pandas as pd
from pathlib import Path


def _peak_hour_factor(hour: int) -> float:
    """Return a congestion multiplier based on typical peak patterns."""
    morning_peak = np.exp(-0.5 * ((hour - 8.5) / 1.5) ** 2)
    evening_peak = np.exp(-0.5 * ((hour - 17.5) / 1.5) ** 2)
    return 0.2 + 0.8 * max(morning_peak, evening_peak)


def _weekend_factor(day_of_week: int) -> float:
    """Weekends have lighter traffic."""
    return 0.6 if day_of_week >= 5 else 1.0


def _weather_impact(severity: float) -> float:
    """Map weather severity to travel time multiplier (1.0 = no impact)."""
    return 1.0 + 0.5 * severity


def generate_training_data(
    n_samples: int = 5000,
    n_segments: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    
    rng = np.random.RandomState(seed)

    hours = rng.randint(0, 24, size=n_samples)
    days = rng.randint(0, 7, size=n_samples)

    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    day_sin = np.sin(2 * np.pi * days / 7)
    day_cos = np.cos(2 * np.pi * days / 7)

    is_peak = np.array([_peak_hour_factor(h) > 0.5 for h in hours]).astype(float)
    is_weekend = (days >= 5).astype(float)

    length_m = rng.uniform(50, 5000, size=n_samples)
    speed_limit = rng.choice([30, 40, 50, 60, 80, 100, 120], size=n_samples).astype(float)
    num_lanes = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.1, 0.5, 0.3, 0.1]).astype(float)
    elevation = rng.uniform(-50, 50, size=n_samples)

    base_congestion = np.array([_peak_hour_factor(h) * _weekend_factor(d)
                                for h, d in zip(hours, days)])
    congestion = np.clip(base_congestion + rng.normal(0, 0.1, n_samples), 0, 1)

    weather_sev = np.clip(rng.beta(2, 5, size=n_samples), 0, 1)
    incident_prox = rng.exponential(5, size=n_samples)
    event_prox = rng.exponential(10, size=n_samples)
    risk_score = np.clip(
        0.3 * congestion + 0.3 * weather_sev + 0.2 * (1 / (1 + incident_prox)) + rng.normal(0, 0.05, n_samples),
        0, 1
    )

    # ── Target: travel_time_factor ────────────────────────────────────────────
    base_time = length_m / (speed_limit * 1000 / 3600)
    congestion_multiplier = 1.0 + 1.5 * congestion
    weather_multiplier = np.array([_weather_impact(w) for w in weather_sev])
    lane_factor = 1.0 / np.sqrt(num_lanes)
    incident_factor = 1.0 + 0.3 * np.exp(-incident_prox / 2)

    travel_time_factor = (
        congestion_multiplier * weather_multiplier * lane_factor * incident_factor
        + rng.normal(0, 0.05, n_samples)
    )
    travel_time_factor = np.clip(travel_time_factor, 0.8, 5.0)

    df = pd.DataFrame({
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "is_peak_hour": is_peak,
        "is_weekend": is_weekend,
        "length_m": length_m,
        "speed_limit_kph": speed_limit,
        "num_lanes": num_lanes,
        "elevation_change_m": elevation,
        "congestion_index": congestion,
        "weather_severity": weather_sev,
        "incident_proximity": incident_prox,
        "event_proximity": event_prox,
        "road_risk_score": risk_score,
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

    # Sequence
    X_seq, y_seq = generate_sequence_data()
    np.save(out / "X_seq.npy", X_seq)
    np.save(out / "y_seq.npy", y_seq)

    # Graph
    node_feat, edge_idx, node_tgt = generate_graph_data()
    np.save(out / "node_features.npy", node_feat)
    np.save(out / "edge_index.npy", edge_idx)
    np.save(out / "node_targets.npy", node_tgt)

    print(f"[DataGen] Saved datasets to {out.resolve()}")
    print(f"  - training_data.csv     : {df.shape}")
    print(f"  - X_seq.npy / y_seq.npy : {X_seq.shape} / {y_seq.shape}")
    print(f"  - graph data            : nodes={node_feat.shape}, edges={edge_idx.shape}")


if __name__ == "__main__":
    save_datasets()
