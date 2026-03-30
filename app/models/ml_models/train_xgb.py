"""
XGBoost training pipeline for travel-time factor prediction.
Enhanced with better hyperparameters and error handling.
"""

import numpy as np
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from app.data_pipeline.synthetic_data import generate_training_data

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


def train_xgboost(
    n_samples: int = 5000,
    n_estimators: int = 400,
    max_depth: int = 7,
    learning_rate: float = 0.03,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    output_dir: str = "data/models",
    seed: int = 42,
) -> dict:
    """
    Train an XGBoost regressor on synthetic traffic data.
    Falls back to sklearn GBM if xgboost is not installed.
    
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
    colsample_bytree : float
        Column subsample ratio
    reg_alpha : float
        L1 regularization term
    reg_lambda : float
        L2 regularization term
    output_dir : str
        Directory to save the model
    seed : int
        Random seed

    Returns
    -------
    dict with training metrics and model performance.
    """
    if not _HAS_XGB:
        from sklearn.ensemble import GradientBoostingRegressor
        print("[XGB] xgboost not installed — falling back to sklearn GBM")

    print("[XGB] Generating training data...")
    df = generate_training_data(n_samples=n_samples, seed=seed)
    feature_cols = [c for c in df.columns if c != "travel_time_factor"]
    X = df[feature_cols].values
    y = df["travel_time_factor"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    print(f"[XGB] Training XGBoost ({n_estimators} estimators)...")
    if _HAS_XGB:
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=seed,
            tree_method="hist",
            verbosity=0,
            eval_metric='mae',
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=min(n_estimators, 300),
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=seed,
        )

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")

    print(f"[XGB] Fitting model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_path = out_path / "xgboost.pkl"
    joblib.dump(model, model_path, compress=3)

    metrics = {
        "model": "XGBoost" if _HAS_XGB else "GBM_fallback",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "cv_mae_mean": float(-np.mean(cv_scores)),
        "cv_mae_std": float(np.std(cv_scores)),
        "test_mae": float(mae),
        "test_rmse": rmse,
        "test_r2": float(r2),
        "model_path": str(model_path),
    }

    print(f"[XGB] ✅ Test MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f}")
    return metrics


if __name__ == "__main__":
    train_xgboost()
