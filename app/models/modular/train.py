"""
Modular Training & Optimization Engine for DeepRoute ETA Models.
"""

import joblib
import numpy as np
import optuna
from sklearn.model_selection import KFold, train_test_split
from app.models.modular.config import DEFAULT_MODEL_PATH, LIGHTGBM_MODEL_PATH, XGBOOST_MODEL_PATH, OPTUNA_N_TRIALS
from app.models.modular.dataset import load_dataset
from app.models.modular.evaluate import evaluate_predictions
from app.models.modular.model import create_lightgbm_model, create_xgboost_model

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_training_pipeline(n_trials: int = OPTUNA_N_TRIALS):
    """
    Executes dataset loading, 5-fold cross validation, Optuna Bayesian optimization,
    model comparison (LightGBM vs XGBoost), metric evaluation, and exports best_model.pkl.
    """
    print("[DeepRoute Modular ML] Loading dataset & features...")
    X, y, feature_names = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Optuna for LightGBM
    print(f"[Optuna] Running Bayesian Optimization on LightGBM ({n_trials} trials)...")
    def objective_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in kf.split(X_train):
            model = create_lightgbm_model(params)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = model.predict(X_train.iloc[va_idx])
            scores.append(np.mean(np.abs(y_train.iloc[va_idx] - preds)))
        return np.mean(scores)

    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(objective_lgb, n_trials=n_trials)
    
    best_lgb_model = create_lightgbm_model(study_lgb.best_params)
    best_lgb_model.fit(X_train, y_train)
    lgb_preds = best_lgb_model.predict(X_test)
    lgb_metrics = evaluate_predictions(y_test.values, lgb_preds)

    # 2. Optuna for XGBoost
    print(f"[Optuna] Running Bayesian Optimization on XGBoost ({n_trials} trials)...")
    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in kf.split(X_train):
            model = create_xgboost_model(params)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = model.predict(X_train.iloc[va_idx])
            scores.append(np.mean(np.abs(y_train.iloc[va_idx] - preds)))
        return np.mean(scores)

    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(objective_xgb, n_trials=n_trials)
    
    best_xgb_model = create_xgboost_model(study_xgb.best_params)
    best_xgb_model.fit(X_train, y_train)
    xgb_preds = best_xgb_model.predict(X_test)
    xgb_metrics = evaluate_predictions(y_test.values, xgb_preds)

    print("\n--- BENCHMARK RESULTS ---")
    print(f"LightGBM: MAE={lgb_metrics['MAE']}, RMSE={lgb_metrics['RMSE']}, MAPE={lgb_metrics['MAPE_pct']}%, R2={lgb_metrics['R2']}")
    print(f"XGBoost:  MAE={xgb_metrics['MAE']}, RMSE={xgb_metrics['RMSE']}, MAPE={xgb_metrics['MAPE_pct']}%, R2={xgb_metrics['R2']}")

    # Save models
    joblib.dump(best_lgb_model, LIGHTGBM_MODEL_PATH)
    joblib.dump(best_xgb_model, XGBOOST_MODEL_PATH)

    if lgb_metrics["MAE"] <= xgb_metrics["MAE"]:
        winner = "LightGBM"
        joblib.dump(best_lgb_model, DEFAULT_MODEL_PATH)
    else:
        winner = "XGBoost"
        joblib.dump(best_xgb_model, DEFAULT_MODEL_PATH)

    print(f"[Winner] {winner} saved to {DEFAULT_MODEL_PATH}")
    return {"LightGBM": lgb_metrics, "XGBoost": xgb_metrics, "winner": winner}


if __name__ == "__main__":
    run_training_pipeline()
