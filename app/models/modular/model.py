
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, Any


def create_lightgbm_model(params: Dict[str, Any] = None) -> lgb.LGBMRegressor:
    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default_params.update(params)
    return lgb.LGBMRegressor(**default_params)


def create_xgboost_model(params: Dict[str, Any] = None) -> xgb.XGBRegressor:
    """Create an XGBoost Regressor instance with default or custom params."""
    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if params:
        default_params.update(params)
    return xgb.XGBRegressor(**default_params)
