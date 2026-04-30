"""
Comprehensive Model Benchmarking — DeepRoute

Evaluates multiple ML regressors on DeepRoute synthetic travel-time data
and selects the best model based on MAE, RMSE, R², and MAPE.

Candidate models:
  1. XGBoost (current production model)
  2. sklearn GradientBoostingRegressor
  3. RandomForestRegressor
  4. ExtraTreesRegressor
  5. LightGBM (if available)
  6. Ridge (linear baseline)
  7. HistGradientBoostingRegressor (sklearn native histogram boosting)

Output: data/models/benchmark_report.json + console summary
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from app.data_pipeline.synthetic_data import generate_training_data

warnings.filterwarnings("ignore", category=UserWarning)

# ── Optional imports ──────────────────────────────────────────────────────────
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


REPORT_PATH = Path("data/models/benchmark_report.json")
SEED = 42


def _build_all_candidates() -> dict[str, object]:
    """Instantiate every candidate regressor."""
    candidates = {}

    # 1. Current production model
    if _HAS_XGB:
        candidates["xgboost"] = XGBRegressor(
            n_estimators=700,
            max_depth=5,
            learning_rate=0.025,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.02,
            reg_lambda=1.5,
            random_state=SEED,
            tree_method="hist",
            verbosity=0,
            eval_metric="mae",
        )

    # 2. sklearn GBM
    candidates["gbm"] = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        random_state=SEED,
    )

    # 3. Random Forest
    candidates["random_forest"] = RandomForestRegressor(
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )

    # 4. Extra Trees (often outperforms RF on structured data)
    candidates["extra_trees"] = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )

    # 5. LightGBM (typically fastest + very accurate)
    if _HAS_LGBM:
        candidates["lightgbm"] = LGBMRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_alpha=0.02,
            reg_lambda=1.5,
            random_state=SEED,
            verbose=-1,
            n_jobs=-1,
        )

    # 6. HistGradientBoosting (sklearn native histogram boosting)
    candidates["hist_gbm"] = HistGradientBoostingRegressor(
        max_iter=600,
        max_depth=6,
        learning_rate=0.03,
        l2_regularization=1.5,
        random_state=SEED,
    )

    # 7. Ridge (linear baseline to compare against)
    candidates["ridge"] = Ridge(alpha=1.0)

    return candidates


def benchmark(n_samples: int = 10000) -> dict:
    """
    Run full benchmark and return a structured report.

    Steps:
      1. Generate synthetic data
      2. Train/test split (80/20)
      3. For each model: 5-fold CV + holdout evaluation
      4. Rank by combined score (0.5 × MAE_rank + 0.3 × R²_rank + 0.2 × speed_rank)
      5. Select winner
    """
    print("=" * 70)
    print("  DeepRoute — Comprehensive Model Benchmarking")
    print("=" * 70)

    print(f"\n[1/4] Generating {n_samples} synthetic samples...")
    df = generate_training_data(n_samples=n_samples, seed=SEED)
    feature_cols = [c for c in df.columns if c != "travel_time_factor"]
    X = df[feature_cols].values
    y = df["travel_time_factor"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")

    candidates = _build_all_candidates()
    results: dict[str, dict] = {}

    print(f"\n[2/4] Training {len(candidates)} candidate models...\n")
    print(f"  {'Model':<20} {'MAE':>10} {'RMSE':>10} {'R2':>10} {'MAPE%':>10} {'CV MAE':>10} {'Time(s)':>10}")
    print("  " + "-" * 80)

    for name, model in candidates.items():
        t0 = time.perf_counter()

        # 5-fold cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
        )
        cv_mae = float(-np.mean(cv_scores))

        # Fit on full train set
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        # Holdout evaluation
        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100.0)

        results[name] = {
            "mae": round(mae, 8),
            "rmse": round(rmse, 8),
            "r2": round(r2, 8),
            "mape_percent": round(mape, 6),
            "cv_mae_mean": round(cv_mae, 8),
            "cv_mae_std": round(float(np.std(cv_scores)), 8),
            "train_time_s": round(elapsed, 3),
        }

        print(f"  {name:<20} {mae:>10.6f} {rmse:>10.6f} {r2:>10.6f} {mape:>10.4f} {cv_mae:>10.6f} {elapsed:>10.2f}")

    # ── Ranking ───────────────────────────────────────────────────────────────
    print(f"\n[3/4] Ranking models...")

    model_names = list(results.keys())

    # Lower MAE is better — rank ascending
    mae_sorted = sorted(model_names, key=lambda m: results[m]["mae"])
    mae_ranks = {m: (i + 1) for i, m in enumerate(mae_sorted)}

    # Higher R2 is better -- rank descending
    r2_sorted = sorted(model_names, key=lambda m: results[m]["r2"], reverse=True)
    r2_ranks = {m: (i + 1) for i, m in enumerate(r2_sorted)}

    # Lower train time is better -- rank ascending
    time_sorted = sorted(model_names, key=lambda m: results[m]["train_time_s"])
    time_ranks = {m: (i + 1) for i, m in enumerate(time_sorted)}

    # Combined rank (lower is better): 50% accuracy + 30% R2 + 20% speed
    combined = {}
    for m in model_names:
        score = 0.50 * mae_ranks[m] + 0.30 * r2_ranks[m] + 0.20 * time_ranks[m]
        combined[m] = round(score, 3)
        results[m]["mae_rank"] = mae_ranks[m]
        results[m]["r2_rank"] = r2_ranks[m]
        results[m]["speed_rank"] = time_ranks[m]
        results[m]["combined_rank_score"] = combined[m]

    winner = min(model_names, key=lambda m: combined[m])

    print(f"\n  {'Model':<20} {'MAE Rank':>10} {'R2 Rank':>10} {'Speed Rank':>10} {'Combined':>10}")
    print("  " + "-" * 60)
    for m in sorted(model_names, key=lambda n: combined[n]):
        marker = " ** BEST" if m == winner else ""
        print(f"  {m:<20} {mae_ranks[m]:>10} {r2_ranks[m]:>10} {time_ranks[m]:>10} {combined[m]:>10.3f}{marker}")

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving report to {REPORT_PATH}...")

    current_production = "xgboost"
    improvement = None
    if winner != current_production and current_production in results:
        old_mae = results[current_production]["mae"]
        new_mae = results[winner]["mae"]
        improvement = {
            "mae_reduction_percent": round((old_mae - new_mae) / old_mae * 100, 4),
            "r2_improvement": round(results[winner]["r2"] - results[current_production]["r2"], 8),
        }

    report = {
        "generated_at": datetime.now().isoformat(),
        "n_samples": n_samples,
        "n_features": X_train.shape[1],
        "decision": {
            "selected_model": winner,
            "current_production": current_production,
            "switch_recommended": winner != current_production,
            "improvement_over_current": improvement,
        },
        "ranking_weights": {
            "mae_weight": 0.50,
            "r2_weight": 0.30,
            "speed_weight": 0.20,
        },
        "model_results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 70}")
    print(f"  Winner: {winner.upper()}")
    print(f"     MAE={results[winner]['mae']:.6f}  R2={results[winner]['r2']:.6f}  MAPE={results[winner]['mape_percent']:.4f}%")
    if improvement:
        print(f"     MAE reduction vs XGBoost: {improvement['mae_reduction_percent']:.2f}%")
        print(f"     R2 improvement vs XGBoost: {improvement['r2_improvement']:.6f}")
    print(f"{'=' * 70}")

    return report


if __name__ == "__main__":
    benchmark()
