from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_LONG_COLUMNS = {"timestamp", "building_id", "meter_reading"}


def generate_synthetic_meter_data(
    n_buildings: int,
    periods: int,
    freq: str,
    start: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=periods, freq=freq)
    frames: list[pd.DataFrame] = []

    for idx in range(n_buildings):
        building_id = f"synthetic_bldg_{idx:03d}"
        area_factor = rng.uniform(0.7, 1.8)
        base = rng.uniform(45, 140) * area_factor
        temp_phase = rng.uniform(-0.5, 0.5)
        hour = timestamps.hour.to_numpy()
        day = np.arange(periods)
        air_temperature = (
            16
            + 9 * np.sin(2 * np.pi * (day / 365.0 + temp_phase))
            + 5 * np.sin(2 * np.pi * hour / 24.0)
            + rng.normal(0, 1.5, periods)
        )
        occupancy = np.where((hour >= 7) & (hour <= 19), 1.0, 0.45)
        weekday = np.where(pd.Series(timestamps).dt.dayofweek.to_numpy() < 5, 1.0, 0.72)
        hvac = np.maximum(air_temperature - 18, 0) * rng.uniform(2.5, 5.0)
        heating = np.maximum(15 - air_temperature, 0) * rng.uniform(1.0, 2.5)
        trend = np.linspace(0, rng.uniform(-5, 8), periods)
        noise = rng.normal(0, base * 0.04, periods)
        meter = base * occupancy * weekday + hvac + heating + trend + noise
        meter = np.clip(meter, 0.1, None)

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "building_id": building_id,
                    "meter_reading": meter,
                    "air_temperature": air_temperature,
                    "site_id": f"synthetic_site_{idx % 3}",
                    "primary_use": "Office" if idx % 2 == 0 else "Education",
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def load_long_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_LONG_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Long CSV missing required columns: {sorted(missing)}")
    return df


def _read_wide_meter_csv(path: Path, max_buildings: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).rename(columns={timestamp_col: "timestamp"})

    value_cols = [col for col in df.columns if col != "timestamp"]
    if max_buildings is not None:
        value_cols = value_cols[:max_buildings]
    long_df = df.melt(
        id_vars=["timestamp"],
        value_vars=value_cols,
        var_name="building_id",
        value_name="meter_reading",
    )
    long_df["meter_file"] = path.name
    return long_df


def _read_selected_wide_meter_csv(
    path: Path,
    value_cols: list[str],
) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    timestamp_col = "timestamp" if "timestamp" in header.columns else header.columns[0]
    usecols = [timestamp_col] + value_cols
    df = pd.read_csv(path, usecols=usecols)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).rename(columns={timestamp_col: "timestamp"})
    long_df = df.melt(
        id_vars=["timestamp"],
        value_vars=value_cols,
        var_name="building_id",
        value_name="meter_reading",
    )
    long_df["meter_file"] = path.name
    return long_df


def load_bdg2_meter_csv(
    meter_path: str | Path,
    metadata_path: str | Path | None,
    weather_path: str | Path | None,
    max_buildings: int | None,
    selection_strategy: str,
    selected_buildings_path: str | Path | None = None,
) -> pd.DataFrame:
    meter_file = Path(meter_path)
    if not meter_file.exists():
        raise FileNotFoundError(f"BDG2 meter CSV not found: {meter_file}")

    header = pd.read_csv(meter_file, nrows=0)
    timestamp_col = "timestamp" if "timestamp" in header.columns else header.columns[0]
    meter_cols = [col for col in header.columns if col != timestamp_col]

    metadata = _load_bdg2_metadata(metadata_path)
    if selected_buildings_path is not None:
        selection_path = Path(selected_buildings_path)
        if not selection_path.exists():
            raise FileNotFoundError(
                f"Selected-building manifest not found: {selection_path}"
            )
        selection = pd.read_csv(selection_path)
        if "building_id" not in selection.columns:
            raise ValueError(
                f"Selected-building manifest requires a building_id column: {selection_path}"
            )
        if selection["building_id"].duplicated().any():
            duplicates = sorted(
                selection.loc[
                    selection["building_id"].duplicated(keep=False),
                    "building_id",
                ]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Selected-building manifest contains duplicate identifiers: {duplicates}"
            )
        selected_cols = selection["building_id"].astype(str).tolist()
        missing_selected = sorted(set(selected_cols) - set(meter_cols))
        if missing_selected:
            raise ValueError(
                "Selected-building identifiers are absent from the electricity meter file: "
                f"{missing_selected}"
            )
        if max_buildings is not None and len(selected_cols) > int(max_buildings):
            raise ValueError(
                "Selected-building manifest exceeds data.max_buildings: "
                f"{len(selected_cols)} > {max_buildings}"
            )
    else:
        selected_cols = _select_meter_columns(
            meter_cols,
            metadata=metadata,
            max_buildings=max_buildings,
            strategy=selection_strategy,
        )
    long_df = _read_selected_wide_meter_csv(meter_file, selected_cols)

    if metadata is not None:
        meta_cols = [
            col
            for col in ["building_id", "site_id", "primaryspaceusage", "sqm", "sqft"]
            if col in metadata.columns
        ]
        long_df = long_df.merge(metadata[meta_cols], on="building_id", how="left")
        if "primaryspaceusage" in long_df.columns:
            long_df = long_df.rename(columns={"primaryspaceusage": "primary_use"})
    if "site_id" not in long_df.columns:
        long_df["site_id"] = long_df["building_id"].astype(str).str.split("_").str[0]

    if weather_path is not None:
        long_df = _merge_bdg2_weather(long_df, weather_path)
    return long_df


def load_bdg2_dir(root: str | Path, max_buildings: int | None = None) -> pd.DataFrame:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"BDG2 root not found: {root_path}")

    candidates = list(root_path.rglob("*.csv"))
    meter_candidates = [
        p
        for p in candidates
        if "meter" in str(p).lower() and "weather" not in str(p).lower()
    ]
    if not meter_candidates:
        raise FileNotFoundError(f"No meter CSV files found under {root_path}")

    frames: list[pd.DataFrame] = []
    remaining = max_buildings
    for csv_path in sorted(meter_candidates):
        frame = _read_wide_meter_csv(csv_path, remaining)
        frames.append(frame)
        if remaining is not None:
            used = frame["building_id"].nunique()
            remaining -= used
            if remaining <= 0:
                break
    return pd.concat(frames, ignore_index=True)


def _load_bdg2_metadata(metadata_path: str | Path | None) -> pd.DataFrame | None:
    if metadata_path is None:
        return None
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"BDG2 metadata CSV not found: {path}")
    metadata = pd.read_csv(path)
    if "building_id" not in metadata.columns:
        raise ValueError(f"BDG2 metadata missing building_id column: {path}")
    return metadata


def _select_meter_columns(
    meter_cols: list[str],
    metadata: pd.DataFrame | None,
    max_buildings: int | None,
    strategy: str,
) -> list[str]:
    if max_buildings is None or max_buildings >= len(meter_cols):
        return meter_cols
    if strategy == "site_balanced" and metadata is not None and "site_id" in metadata.columns:
        meta = metadata[metadata["building_id"].isin(meter_cols)].copy()
        if not meta.empty:
            meta["site_id"] = meta["site_id"].fillna(
                meta["building_id"].astype(str).str.split("_").str[0]
            )
            grouped = {
                str(site_id): sorted(group["building_id"].astype(str).tolist())
                for site_id, group in meta.groupby("site_id")
            }
            selected: list[str] = []
            while len(selected) < max_buildings and grouped:
                for site_id in sorted(list(grouped)):
                    if grouped[site_id]:
                        selected.append(grouped[site_id].pop(0))
                        if len(selected) == max_buildings:
                            break
                    if not grouped[site_id]:
                        grouped.pop(site_id, None)
            return selected
    return meter_cols[:max_buildings]


def _merge_bdg2_weather(df: pd.DataFrame, weather_path: str | Path) -> pd.DataFrame:
    path = Path(weather_path)
    if not path.exists():
        raise FileNotFoundError(f"BDG2 weather CSV not found: {path}")
    weather = pd.read_csv(path)
    required = {"timestamp", "site_id"}
    missing = required - set(weather.columns)
    if missing:
        raise ValueError(f"BDG2 weather CSV missing required columns: {sorted(missing)}")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce")
    rename_map = {
        "airTemperature": "air_temperature",
        "dewTemperature": "dew_temperature",
        "windSpeed": "wind_speed",
        "cloudCoverage": "cloud_coverage",
    }
    weather = weather.rename(columns={k: v for k, v in rename_map.items() if k in weather.columns})
    keep_cols = [
        col
        for col in [
            "timestamp",
            "site_id",
            "air_temperature",
            "dew_temperature",
            "wind_speed",
            "cloud_coverage",
        ]
        if col in weather.columns
    ]
    weather = weather[keep_cols].dropna(subset=["timestamp", "site_id"])
    out = df.merge(weather, on=["timestamp", "site_id"], how="left")
    return out


def add_weather_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "air_temperature" in out.columns:
        return out
    timestamps = pd.to_datetime(out["timestamp"])
    hour = timestamps.dt.hour.to_numpy()
    day = timestamps.dt.dayofyear.to_numpy()
    out["air_temperature"] = 16 + 10 * np.sin(2 * np.pi * day / 365.0) + 4 * np.sin(
        2 * np.pi * hour / 24.0
    )
    return out


def clean_long_data(
    df: pd.DataFrame,
    max_buildings: int | None,
    min_history: int,
    clip_negative: bool,
    min_nonzero_fraction: float = 0.0,
    min_target_mean: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["meter_reading"] = pd.to_numeric(out["meter_reading"], errors="coerce")
    out = out.dropna(subset=["timestamp", "building_id", "meter_reading"])
    if clip_negative:
        out = out[out["meter_reading"] >= 0]
    out = add_weather_if_missing(out)
    out["air_temperature"] = pd.to_numeric(out["air_temperature"], errors="coerce")
    out = out.dropna(subset=["air_temperature"])
    out = out.sort_values(["building_id", "timestamp"])
    out = out.drop_duplicates(subset=["building_id", "timestamp"], keep="last")

    quality = (
        out.groupby("building_id")["meter_reading"]
        .agg(
            history_count="size",
            target_mean="mean",
            nonzero_fraction=lambda values: float((values > 0).mean()),
        )
        .sort_values(["history_count", "nonzero_fraction"], ascending=False)
    )
    keep = quality[
        (quality["history_count"] >= min_history)
        & (quality["nonzero_fraction"] >= min_nonzero_fraction)
        & (quality["target_mean"] >= min_target_mean)
    ]
    keep_buildings = keep.index.tolist()
    if max_buildings is not None:
        keep_buildings = keep_buildings[:max_buildings]
    out = out[out["building_id"].isin(keep_buildings)].reset_index(drop=True)
    if out.empty:
        raise ValueError("No building time series remained after cleaning filters.")
    return out


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    source = data_cfg["source"]
    source_type = source["type"]

    if source_type == "synthetic":
        syn = data_cfg["synthetic"]
        raw = generate_synthetic_meter_data(
            n_buildings=int(syn["n_buildings"]),
            periods=int(syn["periods"]),
            freq=str(syn["freq"]),
            start=str(syn["start"]),
            seed=int(config["experiment"].get("seed", 42)),
        )
    elif source_type == "long_csv":
        raw = load_long_csv(source["path"])
    elif source_type == "bdg2_dir":
        raw = load_bdg2_dir(source["root"], data_cfg.get("max_buildings"))
    elif source_type == "bdg2_meter_csv":
        raw = load_bdg2_meter_csv(
            meter_path=source["meter_path"],
            metadata_path=source.get("metadata_path"),
            weather_path=source.get("weather_path"),
            max_buildings=data_cfg.get("max_buildings"),
            selection_strategy=str(source.get("selection_strategy", "site_balanced")),
            selected_buildings_path=source.get("selected_buildings_path"),
        )
    else:
        raise ValueError(f"Unsupported data source type: {source_type}")

    return clean_long_data(
        raw,
        max_buildings=data_cfg.get("max_buildings"),
        min_history=int(data_cfg.get("min_history", 1)),
        clip_negative=bool(data_cfg.get("clip_negative", True)),
        min_nonzero_fraction=float(data_cfg.get("min_nonzero_fraction", 0.0)),
        min_target_mean=float(data_cfg.get("min_target_mean", 0.0)),
    )


def data_profile(df: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(df["timestamp"])
    profile: dict[str, Any] = {
        "rows": int(len(df)),
        "n_buildings": int(df["building_id"].nunique()),
        "start": timestamps.min().isoformat(),
        "end": timestamps.max().isoformat(),
        "target_min": float(df["meter_reading"].min()),
        "target_mean": float(df["meter_reading"].mean()),
        "target_max": float(df["meter_reading"].max()),
        "missing_fraction_by_column": {
            col: float(df[col].isna().mean()) for col in df.columns
        },
        "rows_by_building": {
            str(k): int(v) for k, v in df.groupby("building_id").size().to_dict().items()
        },
    }
    return profile


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
