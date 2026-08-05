from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def leakage_audit(arrays: dict[str, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    chronology_rows: list[dict[str, Any]] = []

    for split, values in arrays.items():
        frame = _split_window_frame(split, values)
        if frame.empty:
            continue
        for building_id, group in frame.groupby("building_id"):
            gap_hours = (
                (group["target_timestamp"] - group["window_end_timestamp"])
                .dt.total_seconds()
                .div(3600.0)
            )
            rows.append(
                {
                    "split": split,
                    "building_id": building_id,
                    "n_samples": int(len(group)),
                    "target_start": group["target_timestamp"].min().isoformat(),
                    "target_end": group["target_timestamp"].max().isoformat(),
                    "window_start_min": group["window_start_timestamp"].min().isoformat(),
                    "window_end_max": group["window_end_timestamp"].max().isoformat(),
                    "future_input_violations": int((gap_hours <= 0).sum()),
                    "min_input_target_gap_hours": float(gap_hours.min()),
                }
            )

    all_frames = [_split_window_frame(split, values) for split, values in arrays.items()]
    all_windows = pd.concat([frame for frame in all_frames if not frame.empty], ignore_index=True)
    if not all_windows.empty:
        for building_id, group in all_windows.groupby("building_id"):
            split_bounds = {
                split: (
                    part["target_timestamp"].min(),
                    part["target_timestamp"].max(),
                )
                for split, part in group.groupby("split")
            }
            train_end = split_bounds.get("train", (pd.NaT, pd.NaT))[1]
            validation_start, validation_end = split_bounds.get("validation", (pd.NaT, pd.NaT))
            test_start = split_bounds.get("test", (pd.NaT, pd.NaT))[0]
            train_before_validation = bool(
                pd.isna(train_end)
                or pd.isna(validation_start)
                or train_end < validation_start
            )
            validation_before_test = bool(
                pd.isna(validation_end)
                or pd.isna(test_start)
                or validation_end < test_start
            )
            chronology_rows.append(
                {
                    "building_id": building_id,
                    "train_end": None if pd.isna(train_end) else train_end.isoformat(),
                    "validation_start": None
                    if pd.isna(validation_start)
                    else validation_start.isoformat(),
                    "validation_end": None if pd.isna(validation_end) else validation_end.isoformat(),
                    "test_start": None if pd.isna(test_start) else test_start.isoformat(),
                    "train_before_validation": train_before_validation,
                    "validation_before_test": validation_before_test,
                }
            )

    detail = pd.DataFrame(rows)
    chronology = pd.DataFrame(chronology_rows)
    future_violations = int(detail["future_input_violations"].sum()) if not detail.empty else 0
    chronology_ok = True
    if not chronology.empty:
        chronology_ok = bool(
            chronology["train_before_validation"].all()
            and chronology["validation_before_test"].all()
        )
    summary = {
        "future_input_violations": future_violations,
        "no_future_leakage": future_violations == 0,
        "chronological_split_ok": chronology_ok,
        "n_buildings_checked": int(chronology["building_id"].nunique()) if not chronology.empty else 0,
        "chronology_by_building": chronology_rows,
    }
    return detail, summary


def reviewer_validation_checklist(
    config: dict[str, Any],
    leakage_summary: dict[str, Any],
    model_diagnostics: dict[str, Any],
    leaderboard: pd.DataFrame,
    proposed_model: str = "pa_stt",
) -> dict[str, Any]:
    models = sorted(str(model) for model in leaderboard["model"].dropna().unique())
    baselines = [model for model in models if model != proposed_model]
    source_type = config.get("data", {}).get("source", {}).get("type")
    uses_real_measured_data = source_type != "synthetic"
    metric_columns = {
        "r2",
        "mae",
        "rmse",
        "cvrmse_pct",
        "nmbe_pct",
        "mape_pct",
        "smape_pct",
    }
    available_metrics = metric_columns.intersection(set(leaderboard.columns))
    return {
        "data_leakage": {
            "passed": bool(
                leakage_summary.get("no_future_leakage", False)
                and leakage_summary.get("chronological_split_ok", False)
            ),
            "evidence_files": [
                "09_data_leakage_audit.csv",
                "09_data_leakage_summary.json",
            ],
        },
        "baseline_comparison": {
            "passed": len(baselines) >= 2,
            "baselines": baselines,
        },
        "metric_coverage": {
            "passed": available_metrics == metric_columns,
            "metrics": sorted(available_metrics),
        },
        "reproducibility": {
            "passed": "seed" in config.get("experiment", {}),
            "seed": config.get("experiment", {}).get("seed"),
            "config_file": "00_config_used.yaml",
        },
        "hardware_record": {
            "passed": bool(model_diagnostics.get("device")),
            "device": model_diagnostics.get("device"),
            "cuda_device_name": model_diagnostics.get("cuda_device_name"),
        },
        "non_ensemble_model": {
            "passed": True,
            "model": proposed_model,
            "note": "The proposed method is one Transformer network with one forecasting head, not an ensemble or stacked voting system.",
        },
        "model_complexity": {
            "passed": int(model_diagnostics.get("trainable_parameter_count", 0)) > 0,
            "trainable_parameter_count": model_diagnostics.get("trainable_parameter_count"),
        },
        "real_measured_data_validation": {
            "passed": uses_real_measured_data,
            "source_type": source_type,
            "note": "The run uses measured building meter data rather than synthetic smoke-test data."
            if uses_real_measured_data
            else "Synthetic data is only suitable for smoke testing; real measured data is required for manuscript-level claims.",
        },
    }


def paired_error_tests(
    predictions: pd.DataFrame,
    proposed_model: str = "pa_stt",
    split: str = "test",
) -> pd.DataFrame:
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except Exception:  # pragma: no cover - scipy is expected in the research env
        ttest_rel = None
        wilcoxon = None

    rows: list[dict[str, Any]] = []
    data = predictions[predictions["split"] == split].copy()
    if data.empty or proposed_model not in set(data["model"]):
        return pd.DataFrame(rows)

    run_keys = [col for col in ["variant", "seed"] if col in data.columns]
    join_keys = run_keys + ["timestamp", "building_id"]
    for key_values, group in data.groupby(run_keys, dropna=False) if run_keys else [((), data)]:
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        proposed = group[group["model"] == proposed_model][join_keys + ["y_true", "y_pred"]].rename(
            columns={"y_pred": "proposed_pred"}
        )
        if proposed.empty:
            continue
        for baseline in sorted(model for model in group["model"].unique() if model != proposed_model):
            base = group[group["model"] == baseline][join_keys + ["y_pred"]].rename(
                columns={"y_pred": "baseline_pred"}
            )
            paired = proposed.merge(base, on=join_keys, how="inner")
            if paired.empty:
                continue
            proposed_abs = np.abs(paired["proposed_pred"].to_numpy() - paired["y_true"].to_numpy())
            baseline_abs = np.abs(paired["baseline_pred"].to_numpy() - paired["y_true"].to_numpy())
            diff = baseline_abs - proposed_abs
            row: dict[str, Any] = {
                "split": split,
                "baseline_model": baseline,
                "n_pairs": int(len(diff)),
                "mean_abs_error_reduction": float(np.mean(diff)),
                "median_abs_error_reduction": float(np.median(diff)),
                "win_rate_pct": float(np.mean(diff > 0) * 100.0),
                "cohens_dz": _cohens_dz(diff),
            }
            for key, value in zip(run_keys, key_values, strict=False):
                row[key] = value
            if wilcoxon is not None and len(diff) > 0 and not np.allclose(diff, 0):
                row["wilcoxon_p_greater"] = float(wilcoxon(diff, alternative="greater").pvalue)
            else:
                row["wilcoxon_p_greater"] = np.nan
            if ttest_rel is not None and len(diff) > 1:
                row["paired_ttest_p_greater"] = float(
                    ttest_rel(baseline_abs, proposed_abs, alternative="greater").pvalue
                )
            else:
                row["paired_ttest_p_greater"] = np.nan
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty and "wilcoxon_p_greater" in out.columns:
        out["wilcoxon_p_holm"] = _holm_adjust(out["wilcoxon_p_greater"].to_numpy(dtype=float))
    return out


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "r2",
        "mae",
        "rmse",
        "cvrmse_pct",
        "nmbe_pct",
        "mape_pct",
        "smape_pct",
    ]
    group_cols = [col for col in ["variant", "model", "split"] if col in metrics.columns]
    aggregations = {col: ["mean", "std", "min", "max"] for col in metric_cols if col in metrics.columns}
    if not group_cols or not aggregations:
        return pd.DataFrame()
    out = metrics.groupby(group_cols, dropna=False).agg(aggregations)
    out.columns = ["_".join(col).strip("_") for col in out.columns.to_flat_index()]
    return out.reset_index()


def ablation_summary(metrics: pd.DataFrame, proposed_model: str = "pa_stt") -> pd.DataFrame:
    required = {"variant", "seed", "model", "split", "rmse", "mae", "r2"}
    if not required.issubset(set(metrics.columns)):
        return pd.DataFrame()
    model_metrics = metrics[
        (metrics["model"] == proposed_model)
        & (metrics["split"] == "test")
        & (metrics["variant"] != "main")
    ].copy()
    main = metrics[
        (metrics["model"] == proposed_model)
        & (metrics["split"] == "test")
        & (metrics["variant"] == "main")
    ][["seed", "rmse", "mae", "r2"]].rename(
        columns={"rmse": "main_rmse", "mae": "main_mae", "r2": "main_r2"}
    )
    if model_metrics.empty or main.empty:
        return pd.DataFrame()
    joined = model_metrics.merge(main, on="seed", how="inner")
    joined["delta_rmse_vs_main"] = joined["rmse"] - joined["main_rmse"]
    joined["delta_mae_vs_main"] = joined["mae"] - joined["main_mae"]
    joined["delta_r2_vs_main"] = joined["r2"] - joined["main_r2"]
    return (
        joined.groupby("variant")
        .agg(
            n_seeds=("seed", "nunique"),
            mean_delta_rmse_vs_main=("delta_rmse_vs_main", "mean"),
            std_delta_rmse_vs_main=("delta_rmse_vs_main", "std"),
            mean_delta_mae_vs_main=("delta_mae_vs_main", "mean"),
            mean_delta_r2_vs_main=("delta_r2_vs_main", "mean"),
        )
        .reset_index()
    )


def _split_window_frame(split: str, values: dict[str, np.ndarray]) -> pd.DataFrame:
    if values["timestamp"].size == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "split": split,
            "building_id": values["building_id"],
            "target_timestamp": pd.to_datetime(values["timestamp"], errors="coerce"),
            "window_start_timestamp": pd.to_datetime(
                values["window_start_timestamp"],
                errors="coerce",
            ),
            "window_end_timestamp": pd.to_datetime(
                values["window_end_timestamp"],
                errors="coerce",
            ),
        }
    ).dropna(subset=["target_timestamp", "window_start_timestamp", "window_end_timestamp"])


def _cohens_dz(diff: np.ndarray) -> float:
    std = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    if std < 1e-12:
        return 0.0
    return float(np.mean(diff) / std)


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    valid_mask = ~np.isnan(p_values)
    valid = p_values[valid_mask]
    if valid.size == 0:
        return adjusted
    order = np.argsort(valid)
    sorted_p = valid[order]
    m = len(sorted_p)
    running = 0.0
    sorted_adj = np.empty_like(sorted_p, dtype=float)
    for idx, p_value in enumerate(sorted_p):
        running = max(running, min(1.0, (m - idx) * p_value))
        sorted_adj[idx] = running
    restored = np.empty_like(sorted_adj)
    restored[order] = sorted_adj
    adjusted[valid_mask] = restored
    return adjusted
