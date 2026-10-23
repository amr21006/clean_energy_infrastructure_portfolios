from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig


def _check(name: str, passed: bool, evidence: Any, severity: str = "error") -> dict[str, Any]:
    return {
        "check": name,
        "passed": bool(passed),
        "severity": severity,
        "evidence": evidence,
    }


def validate_analysis(
    config: AnalysisConfig,
    projects: pd.DataFrame,
    panel: pd.DataFrame,
    transitions: pd.DataFrame,
    landmarks: pd.DataFrame,
    data_audit: dict[str, Any],
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    continuity = data_audit.get("continuity", {})
    checks.append(
        _check(
            "continuous monthly archive",
            continuity.get("continuous_monthly_archive") is True,
            continuity,
        )
    )
    duplicate_panel = int(panel.duplicated(["unit_id", "report_ordinal"]).sum())
    checks.append(_check("unique unit-month rows", duplicate_panel == 0, duplicate_panel))
    event_order_failures = int(
        (
            projects["event_ordinal"].notna()
            & (projects["event_ordinal"] <= projects["first_report_ordinal"])
            & ~projects["operation_before_or_at_entry"]
        ).sum()
    )
    checks.append(
        _check("terminal events follow cohort entry", event_order_failures == 0, event_order_failures)
    )
    valid_states = {"P", "L", "T", "U", "V", "TS", "O", "C", "OTHER"}
    invalid_current = sorted(set(panel["current_state"]) - valid_states)
    invalid_next = sorted(set(transitions["next_state"]) - valid_states)
    checks.append(
        _check(
            "all states are harmonized",
            not invalid_current and not invalid_next,
            {"invalid_current": invalid_current, "invalid_next": invalid_next},
        )
    )
    bad_transition_gaps = int(
        (~transitions["transition_gap_months"].isin([0, 1])).sum()
    )
    checks.append(
        _check("transition gaps restricted to immediate states", bad_transition_gaps == 0, bad_transition_gaps)
    )
    split_overlap: dict[str, int] = {}
    split_sets = {
        split: set(projects.loc[projects["analysis_split"] == split, "unit_id"])
        for split in ("train", "validation", "test")
    }
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        split_overlap[f"{first}|{second}"] = len(split_sets[first] & split_sets[second])
    checks.append(
        _check(
            "temporal entry-cohort splits are project-disjoint",
            all(value == 0 for value in split_overlap.values()),
            split_overlap,
        )
    )
    observable_12_errors = int(
        (
            landmarks["operation_within_12_months"].eq(1)
            & ~landmarks["observable_12_month_outcome"]
        ).sum()
    )
    observable_24_errors = int(
        (
            landmarks["cancellation_within_24_months"].eq(1)
            & ~landmarks["observable_24_month_outcome"]
        ).sum()
    )
    checks.append(
        _check(
            "landmark event labels are observable",
            observable_12_errors == 0 and observable_24_errors == 0,
            {"operation_12m": observable_12_errors, "cancellation_24m": observable_24_errors},
        )
    )
    clean_primary_failures = int(
        (
            projects["eligible_primary_cohort"]
            & ~projects["technology_group"].isin(config.clean_groups)
        ).sum()
    )
    checks.append(
        _check("primary cohort contains only prespecified clean technologies", clean_primary_failures == 0, clean_primary_failures)
    )
    weight_failures = int(
        (~np.isfinite(projects["cluster_equalizing_weight"]))
        .fillna(True)
        .sum()
    )
    checks.append(_check("cluster weights are finite", weight_failures == 0, weight_failures))

    required_files = [
        config.processed_dir / "projects.parquet",
        config.processed_dir / "unit_month_panel.parquet",
        config.processed_dir / "monthly_transitions.parquet",
        config.processed_dir / "landmarks.parquet",
        config.tables_dir / "cohort_flow.csv",
        config.tables_dir / "transition_probabilities.csv",
        config.tables_dir / "next_state_model_performance.csv",
        config.tables_dir / "operation_12m_model_performance.csv",
        config.tables_dir / "cancellation_24m_model_performance.csv",
        config.tables_dir / "next_state_proposed_paired_comparisons.csv",
        config.tables_dir / "next_state_custom_deep_paired_comparisons.csv",
        config.tables_dir / "next_state_delivery_sparse_moe_paired_comparisons.csv",
        config.tables_dir / "operation_12m_proposed_paired_comparisons.csv",
        config.tables_dir / "operation_12m_custom_deep_paired_comparisons.csv",
        config.tables_dir / "operation_12m_delivery_sparse_moe_paired_comparisons.csv",
        config.tables_dir / "operation_12m_delivery_path_moe_paired_comparisons.csv",
        config.tables_dir / "cancellation_24m_proposed_paired_comparisons.csv",
        config.tables_dir / "cancellation_24m_custom_deep_paired_comparisons.csv",
        config.tables_dir / "cancellation_24m_delivery_sparse_moe_paired_comparisons.csv",
        config.tables_dir / "cancellation_24m_delivery_path_moe_paired_comparisons.csv",
        config.tables_dir / "landmark_custom_deep_validation.csv",
        config.tables_dir / "landmark_delivery_sparse_moe_validation.csv",
        config.tables_dir / "landmark_delivery_path_moe_rollout_audit.csv",
        config.tables_dir / "adjusted_association_model_audit.csv",
        config.predictions_dir / "next_state_test_predictions.parquet",
        config.predictions_dir / "operation_12m_test_predictions.parquet",
        config.predictions_dir / "cancellation_24m_test_predictions.parquet",
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    checks.append(_check("all required analysis artifacts exist", not missing_files, missing_files))

    expected_next_models = {
        "empirical_markov",
        "multinomial_logistic",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "lightgbm_cpu",
        "catboost_gpu",
        "xgboost_gpu",
        "tabm_gpu",
        "delivery_sparse_moe_gpu",
        "delivery_state_residual_net_gpu",
        "delivery_net_no_markov_ablation",
        "proposed_delivery_ensemble",
        "equal_weight_pool_ablation",
    }
    expected_binary_models = {
        "status_schedule_logistic",
        "full_logistic",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "lightgbm_cpu",
        "catboost_gpu",
        "xgboost_gpu",
        "tabm_gpu",
        "delivery_sparse_moe_gpu",
        "delivery_path_moe_gpu",
        "delivery_competing_risk_net_gpu",
        "proposed_delivery_ensemble",
        "equal_weight_pool_ablation",
    }
    performance_evidence: dict[str, Any] = {}
    for outcome, expected in (
        ("next_state", expected_next_models),
        ("operation_12m", expected_binary_models),
        ("cancellation_24m", expected_binary_models),
    ):
        path = config.tables_dir / f"{outcome}_model_performance.csv"
        if not path.exists():
            continue
        performance = pd.read_csv(path)
        observed = set(performance["model"])
        missing = sorted(expected - observed)
        nonfinite = int(
            (~np.isfinite(performance.select_dtypes(include=[np.number]))).sum().sum()
        )
        backend = performance.set_index("model")["backend"].to_dict()
        backend_ok = (
            backend.get("catboost_gpu") == "cuda_gpu"
            and backend.get("xgboost_gpu") == "cuda_gpu"
            and backend.get("tabm_gpu") == "cuda_gpu"
            and backend.get("lightgbm_cpu") == "cpu"
            and backend.get("delivery_sparse_moe_gpu") == "cuda_gpu"
            and (
                True
                if outcome == "next_state"
                else backend.get("delivery_path_moe_gpu") == "cuda_gpu"
            )
            and (
                backend.get("delivery_state_residual_net_gpu") == "cuda_gpu"
                if outcome == "next_state"
                else backend.get("delivery_competing_risk_net_gpu") == "cuda_gpu"
            )
        )
        performance_evidence[outcome] = {
            "missing_models": missing,
            "nonfinite_numeric_cells": nonfinite,
            "backend_ok": backend_ok,
        }
    checks.append(
        _check(
            "all prespecified benchmark families and backends are present",
            bool(performance_evidence)
            and all(
                not evidence["missing_models"]
                and evidence["nonfinite_numeric_cells"] == 0
                and evidence["backend_ok"]
                for evidence in performance_evidence.values()
            ),
            performance_evidence,
        )
    )

    rollout_path = config.tables_dir / "landmark_delivery_path_moe_rollout_audit.csv"
    rollout_evidence: dict[str, Any] = {"missing": not rollout_path.exists()}
    rollout_ok = False
    if rollout_path.exists():
        rollout = pd.read_csv(rollout_path)
        maximum_error = float(rollout["maximum_probability_mass_error"].max())
        rollout_evidence = {
            "splits": sorted(rollout["split"].astype(str).unique()),
            "maximum_probability_mass_error": maximum_error,
            "observations": int(rollout["observations"].sum()),
        }
        rollout_ok = (
            set(rollout_evidence["splits"]) == {"test", "validation"}
            and maximum_error <= 1e-5
            and rollout_evidence["observations"] > 0
        )
    checks.append(
        _check(
            "Delivery-PathMoE conserves probability mass",
            rollout_ok,
            rollout_evidence,
        )
    )

    routing_evidence: dict[str, Any] = {}
    routing_ok = True
    for name in (
        "next_state_boosting_validation.csv",
        "landmark_delivery_sparse_moe_validation.csv",
    ):
        path = config.tables_dir / name
        if not path.exists():
            routing_ok = False
            routing_evidence[name] = {"missing": True}
            continue
        tuning = pd.read_csv(path)
        selected = tuning.loc[
            tuning.get("selected", pd.Series(False, index=tuning.index)).astype(str).str.lower().eq("true")
        ]
        if name.startswith("next_state"):
            selected = selected.loc[selected["algorithm"] == "delivery_sparse_moe_gpu"]
        minimum = (
            float(selected["minimum_expert_utilization"].min())
            if len(selected) and "minimum_expert_utilization" in selected
            else math.nan
        )
        routing_evidence[name] = {
            "selected_rows": len(selected),
            "minimum_expert_utilization": minimum,
        }
        routing_ok = routing_ok and len(selected) == 1 and np.isfinite(minimum) and minimum >= 0.05
    checks.append(
        _check(
            "Delivery sparse-MoE routing uses every expert",
            routing_ok,
            routing_evidence,
        )
    )

    weight_evidence: dict[str, Any] = {}
    for outcome in ("next_state", "operation_12m", "cancellation_24m"):
        path = config.tables_dir / f"{outcome}_proposed_ensemble_weights.csv"
        if not path.exists():
            continue
        weights = pd.read_csv(path)
        metadata = {"state", "validation_n", "reliability", "validation_log_loss"}
        weight_columns = [column for column in weights if column not in metadata]
        maximum_sum_error = float(
            np.max(np.abs(weights[weight_columns].sum(axis=1) - 1.0))
        )
        negative_weights = int((weights[weight_columns] < -1e-10).sum().sum())
        weight_evidence[outcome] = {
            "learners": weight_columns,
            "maximum_sum_error": maximum_sum_error,
            "negative_weights": negative_weights,
        }
    checks.append(
        _check(
            "proposed ensemble weights satisfy simplex constraints",
            len(weight_evidence) == 3
            and all(
                evidence["maximum_sum_error"] < 1e-6
                and evidence["negative_weights"] == 0
                for evidence in weight_evidence.values()
            ),
            weight_evidence,
        )
    )

    association_path = config.tables_dir / "adjusted_association_model_audit.csv"
    if association_path.exists():
        association = pd.read_csv(association_path)
        association_ok = bool(
            association["status"].eq("fitted").all()
            and association["converged"].fillna(False).all()
        )
        checks.append(
            _check(
                "adjusted association models converged with finite inference",
                association_ok,
                association.to_dict(orient="records"),
            )
        )

    for outcome in ("operation_12m", "cancellation_24m"):
        prediction_path = config.predictions_dir / f"{outcome}_test_predictions.parquet"
        if not prediction_path.exists():
            continue
        predictions = pd.read_parquet(prediction_path)
        probability_columns = [
            column for column in predictions.columns if column.endswith("_probability")
        ]
        invalid = {
            column: int((~predictions[column].between(0, 1)).sum())
            for column in probability_columns
        }
        checks.append(
            _check(
                f"{outcome} probabilities are bounded",
                all(value == 0 for value in invalid.values()),
                invalid,
            )
        )

    next_path = config.predictions_dir / "next_state_test_predictions.parquet"
    if next_path.exists():
        predictions = pd.read_parquet(next_path)
        model_prefixes = {
            column.split("_prob_")[0]
            for column in predictions.columns
            if "_prob_" in column
        }
        deviations = {}
        for prefix in model_prefixes:
            columns = [
                column for column in predictions.columns if column.startswith(prefix + "_prob_")
            ]
            deviations[prefix] = float(
                np.max(np.abs(predictions[columns].sum(axis=1) - 1.0))
            )
        checks.append(
            _check(
                "next-state probability vectors sum to one",
                all(value < 1e-6 for value in deviations.values()),
                deviations,
            )
        )

    result = pd.DataFrame(checks)
    result.to_csv(
        config.results_dir / "validation_checks.csv", index=False, encoding="utf-8-sig"
    )
    with (config.results_dir / "validation_checks.json").open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2, default=str)
    failures = result.loc[(~result["passed"]) & (result["severity"] == "error")]
    if not failures.empty:
        names = ", ".join(failures["check"])
        raise AssertionError(f"Analysis validation failed: {names}")
    return result
