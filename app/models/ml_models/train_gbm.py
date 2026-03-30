"""
Gradient Boosting training pipeline for travel-time factor prediction.
Enhanced with better hyperparameters and logging.
"""

import numpy as np
import joblib
import warnings
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.data_pipeline.synthetic_data import generate_training_data

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


def train_gbm(
    n_samples: int = 5000,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    output_dir: str = "data/models",
    seed: int = 42,
) -> dict:
    """
    Train a Gradient Boosting regressor on synthetic traffic data.
    
    Parameters
    ----------
    n_samples : int
        Number of training samples
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth
    learning_rate : float
        Learning rate (shrinkage)
    subsample : float
        Subsample ratio for training set
    output_dir : str
        Directory to save the model
    seed : int
        Random seed

    Returns
    -------
    dict with training metrics and model performance.
    """
    print("[GBM] Generating training data...")
    df = generate_training_data(n_samples=n_samples, seed=seed)
    feature_cols = [c for c in df.columns if c != "travel_time_factor"]
    X = df[feature_cols].values
    y = df["travel_time_factor"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    print(f"[GBM] Training Gradient Boosting ({n_estimators} estimators)...")
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=seed,
        verbose=0,
    )

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")

    print(f"[GBM] Fitting model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_path = out_path / "gbm.pkl"
    joblib.dump(model, model_path, compress=3)

    metrics = {
        "model": "GradientBoosting",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "cv_mae_mean": float(-np.mean(cv_scores)),
        "cv_mae_std": float(np.std(cv_scores)),
        "test_mae": float(mae),
        "test_rmse": rmse,
        "test_r2": float(r2),
        "feature_importances": importances,
        "model_path": str(model_path),
    }

    print(f"[GBM] ✅ Test MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f}")
    return metrics


if __name__ == "__main__":
    train_gbm()