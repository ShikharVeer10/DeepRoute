
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-5, y_true))) * 100.0
    
    abs_pct_err = np.abs((y_true - y_pred) / np.maximum(1e-5, y_true))
    acc_5pct = float(np.mean(abs_pct_err <= 0.05) * 100.0)
    acc_10pct = float(np.mean(abs_pct_err <= 0.10) * 100.0)
    
    return {
        "MAE": round(float(mae), 6),
        "RMSE": round(float(rmse), 6),
        "R2": round(float(r2), 4),
        "MAPE_pct": round(float(mape), 3),
        "Accuracy_within_5pct": round(acc_5pct, 2),
        "Accuracy_within_10pct": round(acc_10pct, 2),
    }
