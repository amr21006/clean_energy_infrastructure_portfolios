"""Run the evidence requested by the Energy & Buildings reviewers.

The script is resumable at the experiment-unit level and writes every metric,
prediction set, configuration, diagnostic, and timing record to
``revision_results/extended_validation``. It never edits the reviewed results.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_forecasting.baselines import run_baselines
from energy_forecasting.config import load_config, save_config
from energy_forecasting.features import (
    TargetScalerState,
    add_time_features,
    chronological_split,
    fit_feature_scaler,
    fit_target_scaler,
    make_windows,
    split_summary,
)
from energy_forecasting.metrics import regression_metrics
from energy_forecasting.models import AnchoredTemporalTransformer
from energy_forecasting.train import (
    TrainResult,
    resolve_device,
    train_anchored_transformer,
)


FEATURE_COLUMNS = [
    "meter_reading",
    "air_temperature",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_npz_arrays(path: Path) -> dict[str, dict[str, np.ndarray]]:
    archive = np.load(path, allow_pickle=True)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in ["train", "validation", "test"]:
        prefix = f"{split}_"
        arrays[split] = {
            key.removeprefix(prefix): archive[key]
            for key in archive.files
            if key.startswith(prefix)
        }
    return arrays


def build_chronological_arrays(
    clean: pd.DataFrame,
    horizon: int,
) -> tuple[dict[str, dict[str, np.ndarray]], TargetScalerState]:
    featured = add_time_features(clean)
    split_frame = chronological_split(
        featured,
        {"train": 0.70, "validation": 0.15, "test": 0.15},
    )
    feature_scaler = fit_feature_scaler(split_frame, FEATURE_COLUMNS)
    target_scaler = fit_target_scaler(split_frame, "meter_reading")
    arrays = make_windows(
        split_frame,
        feature_cols=FEATURE_COLUMNS,
        sequence_length=48,
        horizon=horizon,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        target_col="meter_reading",
        require_contiguous_hourly=True,
    )
    return arrays, target_scaler


def internal_train_site_holdout_split(
    featured: pd.DataFrame,
    held_out_sites: set[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in (
        featured.sort_values(["building_id", "timestamp"]).groupby("building_id")
    ):
        group = group.copy()
        site = str(group["site_id"].iloc[0])
        if site in held_out_sites:
            group["split"] = "test"
        else:
            boundary = max(1, int(len(group) * 0.85))
            labels = np.full(len(group), "validation", dtype=object)
            labels[:boundary] = "train"
            group["split"] = labels
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def build_cross_site_arrays(
    clean: pd.DataFrame,
    held_out_sites: set[str],
) -> tuple[dict[str, dict[str, np.ndarray]], TargetScalerState]:
    featured = add_time_features(clean)
    split_frame = internal_train_site_holdout_split(featured, held_out_sites)
    feature_scaler = fit_feature_scaler(split_frame, FEATURE_COLUMNS)
    target_scaler = fit_target_scaler(split_frame, "meter_reading")
    arrays = make_windows(
        split_frame,
        feature_cols=FEATURE_COLUMNS,
        sequence_length=48,
        horizon=1,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        target_col="meter_reading",
        require_contiguous_hourly=True,
    )
    return arrays, target_scaler


def save_train_result(result: TrainResult, directory: Path, tag: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(directory / f"{tag}_history.csv", index=False)
    result.metric_frame.to_csv(directory / f"{tag}_metrics.csv", index=False)
    result.prediction_frames.to_parquet(
        directory / f"{tag}_predictions.parquet",
        index=False,
    )
    torch.save(result.model.state_dict(), directory / f"{tag}_model.pt")
    save_json(result.diagnostics, directory / f"{tag}_diagnostics.json")


def batch_predict_scaled(
    model: AnchoredTemporalTransformer,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(
                X[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            predictions.append(model(batch).detach().cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=float)


def ramp_diagnostic_rows(
    model: AnchoredTemporalTransformer,
    arrays: dict[str, dict[str, np.ndarray]],
    target_scaler: TargetScalerState,
    device: torch.device,
    model_label: str,
    tau: float,
    kappa: float,
) -> list[dict[str, Any]]:
    train_temp_delta = np.abs(
        arrays["train"]["X"][:, -1, 1] - arrays["train"]["X"][:, 0, 1]
    )
    high_temperature_threshold = float(np.quantile(train_temp_delta, 0.90))
    rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "test"]:
        values = arrays[split]
        pred_scaled = batch_predict_scaled(model, values["X"], device)
        last_scaled = (
            values["last_target"] - target_scaler.mean
        ) / target_scaler.std
        true_scaled = (values["y"] - target_scaler.mean) / target_scaler.std
        temp_delta = np.abs(values["X"][:, -1, 1] - values["X"][:, 0, 1])
        tolerance = tau + kappa * temp_delta
        predicted_excess = np.maximum(
            np.abs(pred_scaled - last_scaled) - tolerance,
            0.0,
        )
        observed_excess = np.maximum(
            np.abs(true_scaled - last_scaled) - tolerance,
            0.0,
        )
        y_pred = target_scaler.inverse_transform(pred_scaled)
        high_mask = temp_delta >= high_temperature_threshold
        all_metrics = regression_metrics(values["y"], y_pred)
        high_metrics = (
            regression_metrics(values["y"][high_mask], y_pred[high_mask])
            if high_mask.any()
            else {}
        )
        rows.append(
            {
                "model": model_label,
                "split": split,
                "tau_0": tau,
                "kappa": kappa,
                "n_samples": int(len(values["y"])),
                "predicted_activation_rate_pct": float(
                    100.0 * np.mean(predicted_excess > 0)
                ),
                "observed_activation_rate_pct": float(
                    100.0 * np.mean(observed_excess > 0)
                ),
                "predicted_mean_excess_scaled": float(predicted_excess.mean()),
                "observed_mean_excess_scaled": float(observed_excess.mean()),
                "predicted_mean_excess_kwh": float(
                    predicted_excess.mean() * target_scaler.std
                ),
                "observed_mean_excess_kwh": float(
                    observed_excess.mean() * target_scaler.std
                ),
                "temperature_delta_train_p90_scaled": high_temperature_threshold,
                "high_temperature_change_fraction_pct": float(
                    100.0 * high_mask.mean()
                ),
                "rmse_all_kwh": all_metrics["rmse"],
                "mae_all_kwh": all_metrics["mae"],
                "rmse_high_temperature_change_kwh": high_metrics.get("rmse"),
                "mae_high_temperature_change_kwh": high_metrics.get("mae"),
            }
        )
    return rows


def instantiate_model(
    model_config: dict[str, Any],
    n_features: int,
    state_path: Path,
    device: torch.device,
) -> AnchoredTemporalTransformer:
    model = AnchoredTemporalTransformer(
        n_features=n_features,
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        n_layers=int(model_config["n_layers"]),
        dim_feedforward=int(model_config["dim_feedforward"]),
        dropout=float(model_config["dropout"]),
        residual_feature_index=model_config.get("residual_feature_index", 0),
    ).to(device)
    state = torch.load(state_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def run_multihorizon(
    clean: pd.DataFrame,
    main_run: Path,
    base_config: dict[str, Any],
    output: Path,
) -> None:
    stage = output / "multihorizon"
    stage.mkdir(parents=True, exist_ok=True)
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []

    main_metrics = pd.read_csv(main_run / "06_leaderboard.csv")
    keep_models = {
        "pa_stt",
        "persistence",
        "lightgbm_summary",
        "lightgbm_full_lags",
    }
    main_metrics = main_metrics[main_metrics["model"].isin(keep_models)].copy()
    main_metrics.insert(0, "horizon_hours", 1)
    metric_frames.append(main_metrics)

    main_proposed = pd.read_parquet(main_run / "05_pa_stt_predictions.parquet")
    main_baselines = pd.read_parquet(main_run / "03_baseline_predictions.parquet")
    main_predictions = pd.concat(
        [
            main_proposed,
            main_baselines[main_baselines["model"].isin(keep_models)],
        ],
        ignore_index=True,
    )
    main_predictions.insert(0, "horizon_hours", 1)
    prediction_frames.append(main_predictions)

    for horizon in [3, 6, 12, 24]:
        unit = stage / f"horizon_{horizon:02d}h"
        completed = unit / "completed.json"
        if completed.exists():
            metric_frames.append(pd.read_csv(unit / "all_metrics.csv"))
            prediction_frames.append(
                pd.read_parquet(unit / "all_predictions.parquet")
            )
            diagnostic_rows.append(
                json.loads((unit / "model_diagnostics.json").read_text(encoding="utf-8"))
            )
            print(f"[multihorizon] reuse horizon={horizon}", flush=True)
            continue

        print(f"[multihorizon] build horizon={horizon}", flush=True)
        started = time.perf_counter()
        arrays, target_scaler = build_chronological_arrays(clean, horizon)
        save_json(split_summary(arrays), unit / "split_summary.json")

        baseline_metrics, baseline_predictions = run_baselines(
            arrays,
            enabled=[
                "persistence",
                "lightgbm_summary",
                "lightgbm_full_lags",
            ],
            seed=42,
        )
        model_config = copy.deepcopy(base_config["model"])
        model_config["reporting_name"] = "pa_stt"
        result = train_anchored_transformer(
            arrays=arrays,
            model_cfg=model_config,
            target_scaler=target_scaler,
            seed=42,
            device_name=str(base_config["experiment"]["device"]),
        )
        save_train_result(result, unit, "pa_stt")

        all_metrics = pd.concat(
            [baseline_metrics, result.metric_frame],
            ignore_index=True,
        )
        all_metrics.insert(0, "horizon_hours", horizon)
        all_metrics.to_csv(unit / "all_metrics.csv", index=False)
        all_predictions = pd.concat(
            [baseline_predictions, result.prediction_frames],
            ignore_index=True,
        )
        all_predictions.insert(0, "horizon_hours", horizon)
        all_predictions.to_parquet(unit / "all_predictions.parquet", index=False)
        diagnostics = {
            **result.diagnostics,
            "horizon_hours": horizon,
            "wall_seconds": float(time.perf_counter() - started),
        }
        save_json(diagnostics, unit / "model_diagnostics.json")
        save_json({"completed_at": datetime.now(timezone.utc).isoformat()}, completed)
        metric_frames.append(all_metrics)
        prediction_frames.append(all_predictions)
        diagnostic_rows.append(diagnostics)
        del arrays, baseline_predictions, result, all_predictions
        gc.collect()
        torch.cuda.empty_cache()

    pd.concat(metric_frames, ignore_index=True).to_csv(
        output / "multihorizon_metrics.csv",
        index=False,
    )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        output / "multihorizon_predictions.parquet",
        index=False,
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        output / "multihorizon_diagnostics.csv",
        index=False,
    )


def run_component_ablation(
    clean: pd.DataFrame,
    main_run: Path,
    base_config: dict[str, Any],
    output: Path,
) -> None:
    stage = output / "component_ablation"
    stage.mkdir(parents=True, exist_ok=True)
    arrays = load_npz_arrays(main_run / "02_window_arrays.npz")
    featured = add_time_features(clean)
    chronological = chronological_split(
        featured,
        {"train": 0.70, "validation": 0.15, "test": 0.15},
    )
    target_scaler = fit_target_scaler(chronological, "meter_reading")

    metric_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    ramp_rows: list[dict[str, Any]] = []
    device = resolve_device(str(base_config["experiment"]["device"]))

    main_metrics = pd.read_csv(main_run / "06_leaderboard.csv")
    main_metrics = main_metrics[
        main_metrics["model"].isin(
            [
                "pa_stt",
                "tcn",
                "residual_tcn",
                "persistence",
                "lightgbm_summary",
                "lightgbm_full_lags",
            ]
        )
    ].copy()
    main_metrics.insert(0, "variant", main_metrics["model"])
    metric_frames.append(main_metrics)

    main_model = instantiate_model(
        base_config["model"],
        n_features=arrays["train"]["X"].shape[2],
        state_path=main_run / "04_pa_stt_model.pt",
        device=device,
    )
    ramp_rows.extend(
        ramp_diagnostic_rows(
            main_model,
            arrays,
            target_scaler,
            device,
            model_label="pa_stt_no_ramp",
            tau=0.25,
            kappa=3.0,
        )
    )
    del main_model
    torch.cuda.empty_cache()

    variants = {
        "plain_transformer": {
            "residual_feature_index": None,
            "ramp_weight": 0.0,
            "reporting_name": "plain_transformer",
        },
        "anchored_with_ramp": {
            "residual_feature_index": 0,
            "ramp_weight": 0.02,
            "ramp_base_tolerance": 0.25,
            "ramp_temperature_coefficient": 3.0,
            "reporting_name": "anchored_with_ramp",
        },
    }
    for name, overrides in variants.items():
        unit = stage / name
        completed = unit / "completed.json"
        if completed.exists():
            metric_frames.append(pd.read_csv(unit / f"{name}_metrics.csv"))
            diagnostics.append(
                json.loads(
                    (unit / f"{name}_diagnostics.json").read_text(encoding="utf-8")
                )
            )
            model = instantiate_model(
                {**base_config["model"], **overrides},
                n_features=arrays["train"]["X"].shape[2],
                state_path=unit / f"{name}_model.pt",
                device=device,
            )
            if name == "anchored_with_ramp":
                ramp_rows.extend(
                    ramp_diagnostic_rows(
                        model,
                        arrays,
                        target_scaler,
                        device,
                        model_label=name,
                        tau=0.25,
                        kappa=3.0,
                    )
                )
            del model
            torch.cuda.empty_cache()
            print(f"[component] reuse {name}", flush=True)
            continue

        config = copy.deepcopy(base_config["model"])
        config.update(overrides)
        result = train_anchored_transformer(
            arrays=arrays,
            model_cfg=config,
            target_scaler=target_scaler,
            seed=42,
            device_name=str(base_config["experiment"]["device"]),
        )
        result.metric_frame.insert(0, "variant", name)
        save_train_result(result, unit, name)
        metric_frames.append(result.metric_frame)
        diagnostics.append({**result.diagnostics, "variant": name})
        if name == "anchored_with_ramp":
            ramp_rows.extend(
                ramp_diagnostic_rows(
                    result.model,
                    arrays,
                    target_scaler,
                    device,
                    model_label=name,
                    tau=0.25,
                    kappa=3.0,
                )
            )
        save_json({"completed_at": datetime.now(timezone.utc).isoformat()}, completed)
        del result
        gc.collect()
        torch.cuda.empty_cache()

    pd.concat(metric_frames, ignore_index=True).to_csv(
        output / "component_ablation_metrics.csv",
        index=False,
    )
    pd.DataFrame(diagnostics).to_csv(
        output / "component_ablation_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(ramp_rows).to_csv(
        output / "ramp_diagnostics.csv",
        index=False,
    )
    del arrays
    gc.collect()


def run_ramp_sensitivity(
    clean: pd.DataFrame,
    main_run: Path,
    base_config: dict[str, Any],
    output: Path,
) -> None:
    stage = output / "ramp_sensitivity"
    stage.mkdir(parents=True, exist_ok=True)
    arrays = load_npz_arrays(main_run / "02_window_arrays.npz")
    featured = add_time_features(clean)
    chronological = chronological_split(
        featured,
        {"train": 0.70, "validation": 0.15, "test": 0.15},
    )
    target_scaler = fit_target_scaler(chronological, "meter_reading")
    device = resolve_device(str(base_config["experiment"]["device"]))

    settings = [
        ("lambda_0p005", 0.005, 0.25, 3.0),
        ("lambda_0p020", 0.020, 0.25, 3.0),
        ("lambda_0p050", 0.050, 0.25, 3.0),
        ("tau_0p100", 0.020, 0.10, 3.0),
        ("tau_0p500", 0.020, 0.50, 3.0),
        ("kappa_1p000", 0.020, 0.25, 1.0),
        ("kappa_6p000", 0.020, 0.25, 6.0),
    ]
    metric_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []

    for name, weight, tau, kappa in settings:
        unit = stage / name
        completed = unit / "completed.json"
        config = copy.deepcopy(base_config["model"])
        config.update(
            {
                "reporting_name": name,
                "residual_feature_index": 0,
                "ramp_weight": weight,
                "ramp_base_tolerance": tau,
                "ramp_temperature_coefficient": kappa,
                "max_epochs": 6,
                "patience": 2,
            }
        )
        if completed.exists():
            metrics = pd.read_csv(unit / f"{name}_metrics.csv")
            metric_frames.append(metrics)
            diagnostic_rows.append(
                json.loads(
                    (unit / f"{name}_diagnostics.json").read_text(encoding="utf-8")
                )
            )
            model = instantiate_model(
                config,
                n_features=arrays["train"]["X"].shape[2],
                state_path=unit / f"{name}_model.pt",
                device=device,
            )
            activation_rows.extend(
                ramp_diagnostic_rows(
                    model,
                    arrays,
                    target_scaler,
                    device,
                    model_label=name,
                    tau=tau,
                    kappa=kappa,
                )
            )
            del model
            torch.cuda.empty_cache()
            print(f"[sensitivity] reuse {name}", flush=True)
            continue

        result = train_anchored_transformer(
            arrays=arrays,
            model_cfg=config,
            target_scaler=target_scaler,
            seed=42,
            device_name=str(base_config["experiment"]["device"]),
        )
        result.metric_frame.insert(0, "kappa", kappa)
        result.metric_frame.insert(0, "tau_0", tau)
        result.metric_frame.insert(0, "lambda", weight)
        result.metric_frame.insert(0, "variant", name)
        save_train_result(result, unit, name)
        metric_frames.append(result.metric_frame)
        diagnostic_rows.append(
            {
                **result.diagnostics,
                "variant": name,
                "lambda": weight,
                "tau_0": tau,
                "kappa": kappa,
            }
        )
        activation_rows.extend(
            ramp_diagnostic_rows(
                result.model,
                arrays,
                target_scaler,
                device,
                model_label=name,
                tau=tau,
                kappa=kappa,
            )
        )
        save_json({"completed_at": datetime.now(timezone.utc).isoformat()}, completed)
        del result
        gc.collect()
        torch.cuda.empty_cache()

    pd.concat(metric_frames, ignore_index=True).to_csv(
        output / "ramp_sensitivity_metrics.csv",
        index=False,
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        output / "ramp_sensitivity_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(activation_rows).to_csv(
        output / "ramp_sensitivity_activation.csv",
        index=False,
    )
    del arrays
    gc.collect()


def run_cross_site(
    clean: pd.DataFrame,
    base_config: dict[str, Any],
    output: Path,
) -> None:
    stage = output / "cross_site"
    stage.mkdir(parents=True, exist_ok=True)
    sites = sorted(clean["site_id"].astype(str).unique().tolist())
    folds = {
        fold: [site for index, site in enumerate(sites) if index % 5 == fold]
        for fold in range(5)
    }
    save_json(
        {
            "assignment_rule": "Sort site identifiers and assign index modulo five.",
            "folds": folds,
        },
        stage / "site_fold_assignment.json",
    )

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for fold, held_out in folds.items():
        unit = stage / f"fold_{fold}"
        completed = unit / "completed.json"
        if completed.exists():
            metric_frames.append(pd.read_csv(unit / "all_metrics.csv"))
            prediction_frames.append(
                pd.read_parquet(unit / "test_predictions.parquet")
            )
            diagnostics.append(
                json.loads((unit / "model_diagnostics.json").read_text(encoding="utf-8"))
            )
            print(f"[cross-site] reuse fold={fold}", flush=True)
            continue

        started = time.perf_counter()
        arrays, target_scaler = build_cross_site_arrays(clean, set(held_out))
        save_json(split_summary(arrays), unit / "split_summary.json")
        baseline_metrics, baseline_predictions = run_baselines(
            arrays,
            enabled=["persistence", "lightgbm_summary"],
            seed=42,
        )
        config = copy.deepcopy(base_config["model"])
        config.update(
            {
                "reporting_name": "pa_stt",
                "ramp_weight": 0.0,
                "max_epochs": 6,
                "patience": 2,
            }
        )
        result = train_anchored_transformer(
            arrays=arrays,
            model_cfg=config,
            target_scaler=target_scaler,
            seed=42,
            device_name=str(base_config["experiment"]["device"]),
        )
        save_train_result(result, unit, "pa_stt")
        all_metrics = pd.concat(
            [baseline_metrics, result.metric_frame],
            ignore_index=True,
        )
        all_metrics.insert(0, "held_out_sites", "|".join(held_out))
        all_metrics.insert(0, "fold", fold)
        all_metrics.to_csv(unit / "all_metrics.csv", index=False)

        predictions = pd.concat(
            [baseline_predictions, result.prediction_frames],
            ignore_index=True,
        )
        predictions = predictions[predictions["split"] == "test"].copy()
        predictions.insert(0, "fold", fold)
        predictions.to_parquet(unit / "test_predictions.parquet", index=False)
        fold_diagnostics = {
            **result.diagnostics,
            "fold": fold,
            "held_out_sites": held_out,
            "wall_seconds": float(time.perf_counter() - started),
        }
        save_json(fold_diagnostics, unit / "model_diagnostics.json")
        save_json({"completed_at": datetime.now(timezone.utc).isoformat()}, completed)
        metric_frames.append(all_metrics)
        prediction_frames.append(predictions)
        diagnostics.append(fold_diagnostics)
        del arrays, baseline_predictions, result, predictions
        gc.collect()
        torch.cuda.empty_cache()

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_metrics.to_csv(output / "cross_site_fold_metrics.csv", index=False)
    all_predictions.to_parquet(
        output / "cross_site_predictions.parquet",
        index=False,
    )
    pooled_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    for model, group in all_predictions.groupby("model"):
        row = regression_metrics(
            group["y_true"].to_numpy(),
            group["y_pred"].to_numpy(),
        )
        row["model"] = model
        row["analysis"] = "five_fold_site_held_out_pooled"
        pooled_rows.append(row)
        for site, site_group in group.groupby("site_id"):
            site_row = regression_metrics(
                site_group["y_true"].to_numpy(),
                site_group["y_pred"].to_numpy(),
            )
            site_row.update({"model": model, "site_id": site})
            site_rows.append(site_row)
    pd.DataFrame(pooled_rows).to_csv(
        output / "cross_site_metrics.csv",
        index=False,
    )
    pd.DataFrame(site_rows).to_csv(
        output / "cross_site_per_site_metrics.csv",
        index=False,
    )
    pd.DataFrame(diagnostics).to_csv(
        output / "cross_site_diagnostics.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--main-run", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision_results/extended_validation"),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["multihorizon", "component", "sensitivity", "crosssite"],
        default=["multihorizon", "component", "sensitivity", "crosssite"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    if not args.main_run.exists():
        raise FileNotFoundError(args.main_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    save_config(config, args.output_dir / "base_config.yaml")
    clean = pd.read_parquet(args.main_run / "01_cleaned_long.parquet")

    manifest = {
        "schema_version": "1.0",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": {"path": str(args.config), "sha256": sha256(args.config)},
        "main_run": str(args.main_run),
        "main_window_arrays_sha256": sha256(args.main_run / "02_window_arrays.npz"),
        "cleaned_data_sha256": sha256(args.main_run / "01_cleaned_long.parquet"),
        "stages": args.stages,
        "seed": 42,
        "determinism_notes": (
            "Seeds are fixed, but CUDA kernels and mixed-precision training may be "
            "nondeterministic across driver or hardware versions."
        ),
    }
    save_json(manifest, args.output_dir / "run_manifest.json")

    if "multihorizon" in args.stages:
        run_multihorizon(clean, args.main_run, config, args.output_dir)
    if "component" in args.stages:
        run_component_ablation(clean, args.main_run, config, args.output_dir)
    if "sensitivity" in args.stages:
        run_ramp_sensitivity(clean, args.main_run, config, args.output_dir)
    if "crosssite" in args.stages:
        run_cross_site(clean, config, args.output_dir)

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["saved_files"] = sorted(
        str(path.relative_to(args.output_dir))
        for path in args.output_dir.rglob("*")
        if path.is_file()
    )
    save_json(manifest, args.output_dir / "run_manifest.json")
    print("REVIEWER_EXPERIMENTS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
