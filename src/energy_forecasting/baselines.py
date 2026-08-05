from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from energy_forecasting.features import flatten_windows, flatten_windows_full_lags
from energy_forecasting.metrics import regression_metrics


def _make_model(name: str, seed: int):
    base_name = name.removesuffix("_full_lags").removesuffix("_summary")
    if base_name == "ridge":
        return Ridge(alpha=1.0)
    if base_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06, random_state=seed)
    if base_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LightGBM is not available") from exc
        return LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=seed, verbosity=-1)
    if base_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("XGBoost is not available") from exc
        return XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
        )
    if base_name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("CatBoost is not available") from exc
        return CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6, random_seed=seed, verbose=False)
    raise ValueError(f"Unsupported baseline: {name}")


def run_baselines(
    arrays: dict[str, dict[str, np.ndarray]],
    enabled: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []

    if "persistence" in enabled:
        for split in ["validation", "test"]:
            y_true = arrays[split]["y"]
            y_pred = arrays[split]["last_target"]
            metrics = regression_metrics(y_true, y_pred)
            metrics.update({"model": "persistence", "split": split})
            rows.append(metrics)
            pred_frames.append(_prediction_frame(arrays[split], y_pred, "persistence", split))

    train_y = arrays["train"]["y"]

    for name in enabled:
        if name == "persistence":
            continue
        flatten = (
            flatten_windows_full_lags
            if name.endswith("_full_lags")
            else flatten_windows
        )
        train_X = flatten(arrays["train"]["X"])
        try:
            model = _make_model(name, seed)
            model.fit(train_X, train_y)
        except Exception as exc:
            rows.append({"model": name, "split": "error", "error": str(exc)})
            continue
        for split in ["validation", "test"]:
            X_split = flatten(arrays[split]["X"])
            y_true = arrays[split]["y"]
            y_pred = model.predict(X_split)
            metrics = regression_metrics(y_true, y_pred)
            metrics.update({"model": name, "split": split})
            rows.append(metrics)
            pred_frames.append(_prediction_frame(arrays[split], y_pred, name, split))

    return pd.DataFrame(rows), pd.concat(pred_frames, ignore_index=True)


def _prediction_frame(split_arrays: dict[str, np.ndarray], y_pred: np.ndarray, model: str, split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model,
            "split": split,
            "timestamp": split_arrays["timestamp"],
            "window_start_timestamp": split_arrays["window_start_timestamp"],
            "window_end_timestamp": split_arrays["window_end_timestamp"],
            "building_id": split_arrays["building_id"],
            "site_id": split_arrays.get(
                "site_id",
                np.full(len(y_pred), "", dtype=str),
            ),
            "y_true": split_arrays["y"],
            "y_pred": y_pred,
            "residual": y_pred - split_arrays["y"],
        }
    )
