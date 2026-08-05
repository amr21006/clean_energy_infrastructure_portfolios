from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-8, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    value = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def mape_nonzero(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 1.0) -> float:
    mask = np.abs(y_true) >= threshold
    if not np.any(mask):
        return 0.0
    return mape(y_true[mask], y_pred[mask])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mae = float(mean_absolute_error(y_true, y_pred))
    mean_true = float(np.mean(y_true))
    denom = mean_true if abs(mean_true) > 1e-8 else 1.0
    absolute_load = float(np.sum(np.abs(y_true)))
    wape_denom = absolute_load if absolute_load > 1e-8 else 1.0
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": mae,
        "rmse": rmse,
        "cvrmse_pct": float(rmse / denom * 100.0),
        "nmbe_pct": float(np.mean(y_pred - y_true) / denom * 100.0),
        "mape_pct": mape(y_true, y_pred),
        "mape_nonzero_pct": mape_nonzero(y_true, y_pred),
        "wape_pct": float(np.sum(np.abs(y_true - y_pred)) / wape_denom * 100.0),
        "smape_pct": smape(y_true, y_pred),
        "zero_or_near_zero_fraction": float(np.mean(np.abs(y_true) < 1.0)),
        "n_samples": int(len(y_true)),
    }


def metrics_frame(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    order = [
        "model",
        "split",
        "r2",
        "mae",
        "rmse",
        "cvrmse_pct",
        "nmbe_pct",
        "mape_pct",
        "mape_nonzero_pct",
        "wape_pct",
        "smape_pct",
        "zero_or_near_zero_fraction",
        "n_samples",
    ]
    df = pd.DataFrame(rows)
    return df[[col for col in order if col in df.columns]]
