from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class StandardScalerState:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


@dataclass
class TargetScalerState:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    month = ts.dt.month.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return out


def chronological_split(df: pd.DataFrame, split_cfg: dict[str, float]) -> pd.DataFrame:
    train_frac = float(split_cfg["train"])
    validation_frac = float(split_cfg["validation"])
    if train_frac <= 0 or validation_frac <= 0 or train_frac + validation_frac >= 1:
        raise ValueError("Invalid split fractions.")

    frames: list[pd.DataFrame] = []
    for _, group in df.sort_values(["building_id", "timestamp"]).groupby("building_id"):
        n = len(group)
        train_end = max(1, int(n * train_frac))
        val_end = max(train_end + 1, int(n * (train_frac + validation_frac)))
        labels = np.full(n, "test", dtype=object)
        labels[:train_end] = "train"
        labels[train_end:val_end] = "validation"
        part = group.copy()
        part["split"] = labels
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def fit_feature_scaler(df: pd.DataFrame, feature_cols: list[str]) -> StandardScalerState:
    train = df[df["split"] == "train"]
    values = train[feature_cols].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return StandardScalerState(mean=mean, std=std)


def fit_target_scaler(df: pd.DataFrame, target_col: str) -> TargetScalerState:
    train = df[df["split"] == "train"]
    values = train[target_col].to_numpy(dtype=np.float32)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        std = 1.0
    return TargetScalerState(mean=mean, std=std)


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int,
    horizon: int,
    feature_scaler: StandardScalerState,
    target_scaler: TargetScalerState,
    target_col: str = "meter_reading",
    require_contiguous_hourly: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, list[Any]]] = {
        "train": {
            "X": [],
            "y": [],
            "y_scaled": [],
            "building_id": [],
            "timestamp": [],
            "window_start_timestamp": [],
            "window_end_timestamp": [],
            "last_target": [],
            "site_id": [],
        },
        "validation": {
            "X": [],
            "y": [],
            "y_scaled": [],
            "building_id": [],
            "timestamp": [],
            "window_start_timestamp": [],
            "window_end_timestamp": [],
            "last_target": [],
            "site_id": [],
        },
        "test": {
            "X": [],
            "y": [],
            "y_scaled": [],
            "building_id": [],
            "timestamp": [],
            "window_start_timestamp": [],
            "window_end_timestamp": [],
            "last_target": [],
            "site_id": [],
        },
    }

    for building_id, group in df.sort_values(["building_id", "timestamp"]).groupby("building_id"):
        group = group.reset_index(drop=True)
        features = group[feature_cols].to_numpy(dtype=np.float32)
        features_scaled = feature_scaler.transform(features).astype(np.float32)
        target = group[target_col].to_numpy(dtype=np.float32)
        splits = group["split"].to_numpy()
        timestamp_values = pd.to_datetime(group["timestamp"]).to_numpy()
        timestamps = pd.Series(timestamp_values).astype(str).to_numpy()
        site_values = (
            group["site_id"].astype(str).to_numpy()
            if "site_id" in group.columns
            else np.full(len(group), "", dtype=object)
        )
        if len(group) > 1:
            hourly_edges = np.diff(timestamp_values) == np.timedelta64(1, "h")
            nonhourly_prefix = np.concatenate(
                [[0], np.cumsum(~hourly_edges, dtype=np.int64)]
            )
        else:
            nonhourly_prefix = np.zeros(1, dtype=np.int64)

        for end_idx in range(sequence_length - 1, len(group) - horizon):
            target_idx = end_idx + horizon
            if require_contiguous_hourly:
                start_idx = end_idx - sequence_length + 1
                if nonhourly_prefix[target_idx] - nonhourly_prefix[start_idx] != 0:
                    continue
            split = str(splits[target_idx])
            if split not in arrays:
                continue
            window = features_scaled[end_idx - sequence_length + 1 : end_idx + 1]
            y_value = float(target[target_idx])
            arrays[split]["X"].append(window)
            arrays[split]["y"].append(y_value)
            arrays[split]["y_scaled"].append(float(target_scaler.transform(np.array([y_value]))[0]))
            arrays[split]["building_id"].append(str(building_id))
            arrays[split]["timestamp"].append(str(timestamps[target_idx]))
            arrays[split]["window_start_timestamp"].append(str(timestamps[end_idx - sequence_length + 1]))
            arrays[split]["window_end_timestamp"].append(str(timestamps[end_idx]))
            arrays[split]["last_target"].append(float(target[end_idx]))
            arrays[split]["site_id"].append(str(site_values[target_idx]))

    out: dict[str, dict[str, np.ndarray]] = {}
    for split, values in arrays.items():
        if values["X"]:
            out[split] = {
                "X": np.stack(values["X"]).astype(np.float32),
                "y": np.asarray(values["y"], dtype=np.float32),
                "y_scaled": np.asarray(values["y_scaled"], dtype=np.float32),
                "building_id": np.asarray(values["building_id"]),
                "timestamp": np.asarray(values["timestamp"]),
                "window_start_timestamp": np.asarray(values["window_start_timestamp"]),
                "window_end_timestamp": np.asarray(values["window_end_timestamp"]),
                "last_target": np.asarray(values["last_target"], dtype=np.float32),
                "site_id": np.asarray(values["site_id"]),
            }
        else:
            out[split] = {
                "X": np.empty((0, sequence_length, len(feature_cols)), dtype=np.float32),
                "y": np.empty((0,), dtype=np.float32),
                "y_scaled": np.empty((0,), dtype=np.float32),
                "building_id": np.empty((0,), dtype=str),
                "timestamp": np.empty((0,), dtype=str),
                "window_start_timestamp": np.empty((0,), dtype=str),
                "window_end_timestamp": np.empty((0,), dtype=str),
                "last_target": np.empty((0,), dtype=np.float32),
                "site_id": np.empty((0,), dtype=str),
            }
    return out


def flatten_windows(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError("Expected window array with shape [samples, sequence, features].")
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    trend = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([last, mean, std, trend], axis=1)


def flatten_windows_full_lags(X: np.ndarray) -> np.ndarray:
    """Return every lag-feature cell as a two-dimensional baseline matrix."""
    if X.ndim != 3:
        raise ValueError("Expected window array with shape [samples, sequence, features].")
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def split_summary(arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    return {
        split: {
            "samples": int(values["X"].shape[0]),
            "sequence_length": int(values["X"].shape[1]) if values["X"].ndim == 3 else 0,
            "n_features": int(values["X"].shape[2]) if values["X"].ndim == 3 else 0,
            "n_buildings": int(len(set(values["building_id"].tolist()))),
        }
        for split, values in arrays.items()
    }
