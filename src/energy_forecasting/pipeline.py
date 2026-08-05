from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from energy_forecasting.baselines import run_baselines
from energy_forecasting.config import save_config
from energy_forecasting.data import data_profile, load_dataset, save_json
from energy_forecasting.features import (
    add_time_features,
    chronological_split,
    fit_feature_scaler,
    fit_target_scaler,
    make_windows,
    split_summary,
)
from energy_forecasting.metrics import metrics_frame
from energy_forecasting.neural_baselines import train_neural_baselines
from energy_forecasting.train import train_pg_sstt
from energy_forecasting.validation import leakage_audit, paired_error_tests, reviewer_validation_checklist


def run_pipeline(config: dict[str, Any]) -> Path:
    pipeline_start = time.perf_counter()
    run_dir = _make_run_dir(config)
    _log_stage(f"Run directory: {run_dir}")
    save_config(config, run_dir / "00_config_used.yaml")

    seed = int(config["experiment"].get("seed", 42))
    _set_seed(seed)

    _log_stage("Loading and cleaning data")
    raw_clean = load_dataset(config)
    save_json(data_profile(raw_clean), run_dir / "01_raw_data_profile.json")
    raw_clean.to_parquet(run_dir / "01_cleaned_long.parquet", index=False)

    _log_stage("Building time features and supervised windows")
    feature_cfg = config["features"]
    featured = add_time_features(raw_clean)
    split_df = chronological_split(featured, feature_cfg["time_split"])
    feature_cols = list(feature_cfg["feature_cols"])
    feature_scaler = fit_feature_scaler(split_df, feature_cols)
    target_scaler = fit_target_scaler(split_df, config["data"].get("target_col", "meter_reading"))
    arrays = make_windows(
        split_df,
        feature_cols=feature_cols,
        sequence_length=int(feature_cfg["sequence_length"]),
        horizon=int(feature_cfg["horizon"]),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        target_col=config["data"].get("target_col", "meter_reading"),
        require_contiguous_hourly=bool(
            feature_cfg.get("require_contiguous_hourly", True)
        ),
    )
    save_json(split_summary(arrays), run_dir / "02_split_summary.json")
    _save_window_arrays(arrays, run_dir / "02_window_arrays.npz")
    leakage_detail, leakage_summary = leakage_audit(arrays)
    leakage_detail.to_csv(run_dir / "09_data_leakage_audit.csv", index=False)
    save_json(leakage_summary, run_dir / "09_data_leakage_summary.json")

    _log_stage("Training classical/statistical baselines")
    baseline_metrics, baseline_predictions = run_baselines(
        arrays,
        enabled=list(config["baselines"]["enabled"]),
        seed=seed,
    )
    baseline_metrics.to_csv(run_dir / "03_baseline_metrics.csv", index=False)
    baseline_predictions.to_parquet(run_dir / "03_baseline_predictions.parquet", index=False)

    _log_stage("Training neural sequence baselines")
    neural_result = train_neural_baselines(
        arrays,
        baseline_cfg=config.get("neural_baselines", {"enabled": []}),
        target_scaler=target_scaler,
        seed=seed,
        device_name=str(config["experiment"].get("device", "cuda")),
    )
    neural_result.metric_frame.to_csv(run_dir / "03b_neural_baseline_metrics.csv", index=False)
    neural_result.prediction_frames.to_parquet(run_dir / "03b_neural_baseline_predictions.parquet", index=False)
    neural_result.history.to_csv(run_dir / "03b_neural_baseline_training_history.csv", index=False)
    save_json(neural_result.diagnostics, run_dir / "03b_neural_baseline_diagnostics.json")
    for model_name, model in neural_result.models.items():
        torch.save(model.state_dict(), run_dir / f"03b_{model_name}_model.pt")

    proposed_name = str(config["model"].get("reporting_name", "pa_stt"))
    _log_stage(f"Training proposed {proposed_name} model")
    train_result = train_pg_sstt(
        arrays,
        model_cfg=config["model"],
        target_scaler=target_scaler,
        seed=seed,
        device_name=str(config["experiment"].get("device", "auto")),
    )
    train_result.history.to_csv(
        run_dir / f"04_{proposed_name}_training_history.csv",
        index=False,
    )
    torch.save(train_result.model.state_dict(), run_dir / f"04_{proposed_name}_model.pt")
    train_result.metric_frame.to_csv(
        run_dir / f"05_{proposed_name}_metrics.csv",
        index=False,
    )
    train_result.prediction_frames.to_parquet(
        run_dir / f"05_{proposed_name}_predictions.parquet",
        index=False,
    )
    model_diagnostics = {
        **train_result.diagnostics,
        "pipeline_seconds_until_model_complete": float(time.perf_counter() - pipeline_start),
    }
    save_json(model_diagnostics, run_dir / "09_model_diagnostics.json")

    all_metrics = pd.concat(
        [baseline_metrics, neural_result.metric_frame, train_result.metric_frame],
        ignore_index=True,
        sort=False,
    )
    valid_metrics = all_metrics[all_metrics["split"].isin(["validation", "test"])].copy()
    leaderboard = metrics_frame(valid_metrics.to_dict("records"))
    leaderboard = leaderboard.sort_values(["split", "rmse", "mae"], ascending=[True, True, True])
    leaderboard.to_csv(run_dir / "06_leaderboard.csv", index=False)

    summary = _metric_summary(leaderboard)
    save_json(summary, run_dir / "06_metric_summary.json")
    checklist = reviewer_validation_checklist(
        config=config,
        leakage_summary=leakage_summary,
        model_diagnostics=model_diagnostics,
        leaderboard=leaderboard,
        proposed_model=proposed_name,
    )
    save_json(checklist, run_dir / "09_reviewer_validation_checklist.json")

    all_predictions = pd.concat(
        [baseline_predictions, neural_result.prediction_frames, train_result.prediction_frames],
        ignore_index=True,
    )
    figure_predictions = all_predictions[all_predictions["split"] == "test"].copy()
    figure_predictions.to_csv(run_dir / "07_figure_data_predictions.csv", index=False)
    residual_summary = (
        figure_predictions.groupby(["model", "building_id"])["residual"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    residual_summary.to_csv(run_dir / "07_figure_data_residuals.csv", index=False)
    paired_error_tests(
        figure_predictions,
        proposed_model=proposed_name,
    ).to_csv(run_dir / "09_paired_error_tests.csv", index=False)

    _log_stage("Writing manifest and reviewer validation checklist")
    manifest = {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_name": config["experiment"]["name"],
        "seed": seed,
        "n_rows_cleaned": int(len(raw_clean)),
        "n_buildings": int(raw_clean["building_id"].nunique()),
        "best_test_model": summary.get("best_test_model"),
        "best_test_rmse": summary.get("best_test_rmse"),
        "pipeline_seconds": float(time.perf_counter() - pipeline_start),
        "device": model_diagnostics.get("device"),
        "cuda_device_name": model_diagnostics.get("cuda_device_name"),
        "saved_files": sorted(p.name for p in run_dir.iterdir() if p.is_file()),
    }
    save_json(manifest, run_dir / "08_run_manifest.json")
    _log_stage("Run complete")
    return run_dir


def _make_run_dir(config: dict[str, Any]) -> Path:
    base = Path(config["experiment"].get("output_dir", "results"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config['experiment']['name']}_{timestamp}"
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_window_arrays(arrays: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    payload = {}
    for split, values in arrays.items():
        for name, arr in values.items():
            payload[f"{split}_{name}"] = arr
    np.savez_compressed(path, **payload)


def _metric_summary(leaderboard: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in ["validation", "test"]:
        split_df = leaderboard[leaderboard["split"] == split]
        if split_df.empty:
            continue
        best = split_df.sort_values(["rmse", "mae"], ascending=True).iloc[0]
        summary[f"best_{split}_model"] = str(best["model"])
        summary[f"best_{split}_rmse"] = float(best["rmse"])
        summary[f"best_{split}_r2"] = float(best["r2"])
    return summary


def _log_stage(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)
