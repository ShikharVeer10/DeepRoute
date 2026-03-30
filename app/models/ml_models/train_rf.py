"""
Random Forest training pipeline for travel-time factor prediction.

Trains on the synthetic dataset, performs cross-validation, persists model.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.data_pipeline.synthetic_data import generate_training_data

# Suppress sklearn warnings about joblib versions
warnings.filterwarnings('ignore', category=UserWarning)


def train_random_forest(
    n_samples: int = 5000,
    n_estimators: int = 200,
    max_depth: int | None = 15,
    min_samples_split: int = 5,
    output_dir: str = "data/models",
    seed: int = 42,
) -> dict:
    """
    Train a Random Forest regressor on synthetic traffic data.
    
    Parameters
    ----------
    n_samples : int
        Number of training samples to generate
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum tree depth
    min_samples_split : int
        Minimum samples required to split
    output_dir : str
        Directory to save the model
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict with training metrics and model performance.
    """
    print("[RF] Generating training data...")
    df = generate_training_data(n_samples=n_samples, seed=seed)
    feature_cols = [c for c in df.columns if c != "travel_time_factor"]
    X = df[feature_cols].values
    y = df["travel_time_factor"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    print(f"[RF] Training Random Forest ({n_estimators} estimators)...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    print(f"[RF] Fitting model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_path = out_path / "random_forest.pkl"
    joblib.dump(model, model_path, compress=3)

    metrics = {
        "model": "RandomForest",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "cv_mae_mean": float(-np.mean(cv_scores)),
        "cv_mae_std": float(np.std(cv_scores)),
        "test_mae": float(mae),
        "test_rmse": rmse,
        "test_r2": float(r2),
        "feature_importances": importances,
        "model_path": str(model_path),
    }

    print(f"[RF] ✅ Test MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f}")
    return metrics


if __name__ == "__main__":
    train_random_forest()