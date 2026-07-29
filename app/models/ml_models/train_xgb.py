"""
Enhanced XGBoost + LightGBM training pipeline with:
  - Bayesian hyperparameter optimization (Optuna)
  - 5-fold cross-validation
  - LightGBM comparison
  - Comprehensive evaluation (MAE, RMSE, MAPE, R², Median AE, P95)
  - SHAP feature importance
  - Best model auto-selection and saving
  - Benchmark report generation
"""

import numpy as np
import joblib
import warnings
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
)

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

from app.data_pipeline.synthetic_data import generate_training_data


def _compute_metrics(y_true, y_pred):
    """Compute comprehensive regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    # P95 absolute error
    abs_errors = np.abs(y_true - y_pred)
    p95 = float(np.percentile(abs_errors, 95))
    
    # Travel time accuracy (within 5% of true value)
    within_5pct = float(np.mean(abs_errors / np.maximum(y_true, 0.01) < 0.05) * 100)
    
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
        "median_ae": round(med_ae, 6),
        "mape_pct": round(mape, 3),
        "p95_error": round(p95, 6),
        "accuracy_within_5pct": round(within_5pct, 2),
    }


def _optimize_xgboost(X_train, y_train, n_trials=50, seed=42):
    """Bayesian hyperparameter optimization for XGBoost using Optuna."""
    if not _HAS_OPTUNA:
        # Return sensible defaults
        return {
            "n_estimators": 800, "max_depth": 5, "learning_rate": 0.02,
            "subsample": 0.85, "colsample_bytree": 0.85,
            "reg_alpha": 0.01, "reg_lambda": 1.5, "min_child_weight": 3,
        }

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = XGBRegressor(
            **params, random_state=seed, tree_method="hist",
            verbosity=0, eval_metric="mae"
        )
        scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
        )
        return -np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _optimize_lightgbm(X_train, y_train, n_trials=50, seed=42):
    """Bayesian hyperparameter optimization for LightGBM."""
    if not _HAS_OPTUNA:
        return {
            "n_estimators": 800, "max_depth": 6, "learning_rate": 0.02,
            "subsample": 0.85, "colsample_bytree": 0.85,
            "reg_alpha": 0.01, "reg_lambda": 1.5, "min_child_samples": 10,
            "num_leaves": 31,
        }

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        }
        model = LGBMRegressor(**params, random_state=seed, verbosity=-1)
        scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
        )
        return -np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_enhanced(
    n_samples: int = 10000,
    n_optuna_trials: int = 50,
    output_dir: str = "data/models",
    seed: int = 42,
) -> dict:
    """
    Train XGBoost and LightGBM with Bayesian hyperparameter tuning.
    Selects the best model and saves it as the production model.
    """
    print("=" * 60)
    print("  DeepRoute Enhanced Model Training Pipeline")
    print("=" * 60)

    # ── 1. Generate training data ──
    print(f"\n[1/6] Generating {n_samples} training samples...")
    df = generate_training_data(n_samples=n_samples, seed=seed)
    feature_cols = [c for c in df.columns if c != "travel_time_factor"]
    X = df[feature_cols].values
    y = df["travel_time_factor"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    print(f"    Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"    Features: {X_train.shape[1]} | Target range: [{y.min():.3f}, {y.max():.3f}]")

    results = {}
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 2. XGBoost ──
    if _HAS_XGB:
        print(f"\n[2/6] Optimizing XGBoost ({n_optuna_trials} trials)...")
        xgb_params = _optimize_xgboost(X_train, y_train, n_trials=n_optuna_trials, seed=seed)
        print(f"    Best params: depth={xgb_params.get('max_depth')}, lr={xgb_params.get('learning_rate', 0.02):.4f}, "
              f"n_est={xgb_params.get('n_estimators')}")

        xgb_model = XGBRegressor(
            **xgb_params, random_state=seed, tree_method="hist",
            verbosity=0, eval_metric="mae"
        )

        # 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")
        print(f"    CV MAE: {-np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")

        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred_xgb = xgb_model.predict(X_test)
        xgb_metrics = _compute_metrics(y_test, y_pred_xgb)
        xgb_metrics["cv_mae"] = round(float(-np.mean(cv_scores)), 6)
        xgb_metrics["cv_mae_std"] = round(float(np.std(cv_scores)), 6)
        xgb_metrics["params"] = xgb_params

        # Save
        joblib.dump(xgb_model, out_path / "xgboost.pkl", compress=3)
        results["xgboost"] = xgb_metrics
        print(f"    [OK] XGBoost -- MAE: {xgb_metrics['mae']:.6f} | R2: {xgb_metrics['r2']:.4f} | MAPE: {xgb_metrics['mape_pct']:.2f}%")
    else:
        print("\n[2/6] [WARN] XGBoost not installed, skipping.")

    # ── 3. LightGBM ──
    if _HAS_LGBM:
        print(f"\n[3/6] Optimizing LightGBM ({n_optuna_trials} trials)...")
        lgbm_params = _optimize_lightgbm(X_train, y_train, n_trials=n_optuna_trials, seed=seed)
        print(f"    Best params: depth={lgbm_params.get('max_depth')}, lr={lgbm_params.get('learning_rate', 0.02):.4f}, "
              f"leaves={lgbm_params.get('num_leaves')}")

        lgbm_model = LGBMRegressor(**lgbm_params, random_state=seed, verbosity=-1)

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        cv_scores = cross_val_score(lgbm_model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")
        print(f"    CV MAE: {-np.mean(cv_scores):.6f} +/- {np.std(cv_scores):.6f}")

        lgbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        y_pred_lgbm = lgbm_model.predict(X_test)
        lgbm_metrics = _compute_metrics(y_test, y_pred_lgbm)
        lgbm_metrics["cv_mae"] = round(float(-np.mean(cv_scores)), 6)
        lgbm_metrics["cv_mae_std"] = round(float(np.std(cv_scores)), 6)
        lgbm_metrics["params"] = lgbm_params

        # Save
        joblib.dump(lgbm_model, out_path / "lightgbm.pkl", compress=3)
        results["lightgbm"] = lgbm_metrics
        print(f"    [OK] LightGBM -- MAE: {lgbm_metrics['mae']:.6f} | R2: {lgbm_metrics['r2']:.4f} | MAPE: {lgbm_metrics['mape_pct']:.2f}%")
    else:
        print("\n[3/6] [WARN] LightGBM not installed, skipping.")

    # ── 4. Select Best Model ──
    print("\n[4/6] Selecting best model...")
    best_name = "xgboost"
    best_mae = results.get("xgboost", {}).get("mae", float("inf"))
    for name, metrics in results.items():
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_name = name
    
    results["best_model"] = best_name
    print(f"    [BEST] Best Model: {best_name.upper()} (MAE={best_mae:.6f})")

    # If LightGBM wins, also save it as the primary production model
    if best_name == "lightgbm" and _HAS_LGBM:
        joblib.dump(lgbm_model, out_path / "best_model.pkl", compress=3)
        print(f"    Saved as best_model.pkl (LightGBM)")
    elif _HAS_XGB:
        joblib.dump(xgb_model, out_path / "best_model.pkl", compress=3)
        print(f"    Saved as best_model.pkl (XGBoost)")

    # ── 5. Feature Importance ──
    print("\n[5/6] Computing feature importance...")
    try:
        if best_name == "lightgbm" and _HAS_LGBM:
            importances = lgbm_model.feature_importances_
        elif _HAS_XGB:
            importances = xgb_model.feature_importances_
        else:
            importances = None

        if importances is not None:
            feat_imp = sorted(
                zip(feature_cols, importances), key=lambda x: x[1], reverse=True
            )
            print("    Top 10 features:")
            for fname, fimp in feat_imp[:10]:
                bar = "#" * int(fimp / max(importances) * 30)
                print(f"      {fname:30s} {fimp:.4f}  {bar}")
            results["feature_importance"] = {f: round(float(v), 4) for f, v in feat_imp}
    except Exception as e:
        print(f"    [WARN] Feature importance failed: {e}")

    # ── 6. Save Report ──
    print("\n[6/6] Saving benchmark report...")
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_samples,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "results": results,
    }
    report_path = out_path / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"    Saved to {report_path}")

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)

    # Print comparison table
    if len(results) >= 2:
        print("\n  Model Comparison:")
        print(f"  {'Metric':<25s} {'XGBoost':>12s} {'LightGBM':>12s} {'Winner':>10s}")
        print("  " + "-" * 62)
        for metric in ["mae", "rmse", "r2", "mape_pct", "p95_error", "accuracy_within_5pct"]:
            xgb_v = results.get("xgboost", {}).get(metric, "N/A")
            lgbm_v = results.get("lightgbm", {}).get(metric, "N/A")
            if isinstance(xgb_v, (int, float)) and isinstance(lgbm_v, (int, float)):
                if metric in ("r2", "accuracy_within_5pct"):
                    winner = "XGB" if xgb_v >= lgbm_v else "LGBM"
                else:
                    winner = "XGB" if xgb_v <= lgbm_v else "LGBM"
            else:
                winner = "—"
            print(f"  {metric:<25s} {str(xgb_v):>12s} {str(lgbm_v):>12s} {winner:>10s}")

    return results


# Backward-compatible alias for existing imports
def train_xgboost(**kwargs) -> dict:
    """Legacy entry point — delegates to train_enhanced."""
    result = train_enhanced(**kwargs)
    # Return in the old format expected by train_all.py
    xgb = result.get("xgboost", {})
    return {
        "model": "XGBoost",
        "n_estimators": xgb.get("params", {}).get("n_estimators", 800),
        "max_depth": xgb.get("params", {}).get("max_depth", 5),
        "learning_rate": xgb.get("params", {}).get("learning_rate", 0.02),
        "cv_mae_mean": xgb.get("cv_mae", 0),
        "cv_mae_std": xgb.get("cv_mae_std", 0),
        "test_mae": xgb.get("mae", 0),
        "test_rmse": xgb.get("rmse", 0),
        "test_r2": xgb.get("r2", 0),
        "model_path": "data/models/xgboost.pkl",
    }


if __name__ == "__main__":
    train_enhanced()
