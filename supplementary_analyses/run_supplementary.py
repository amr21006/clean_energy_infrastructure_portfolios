"""Supplementary analyses of the reported study.

All analyses read the frozen primary outputs in outputs/delivery_twin and write to
outputs/supplementary. Primary models, splits and reported numbers are not modified.

Run from the repository root:
    python supplementary_analyses/run_supplementary.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eia_delivery_twin.config import AnalysisConfig  # noqa: E402
from eia_delivery_twin.modeling import (  # noqa: E402
    _logistic_pipeline,
    _prepare_features,
    _simple_features,
    _tree_pipeline,
    apply_platt,
    calibration_intercept_slope,
    feature_columns,
    fit_platt,
)

PRIMARY = ROOT / "outputs" / "delivery_twin"
OUT = ROOT / "outputs" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261016
REPS = 500
CONFIG = AnalysisConfig(raw_dir=ROOT / "data" / "raw", output_dir=PRIMARY)

ENDPOINTS = {
    "operation_12m": ("operation_within_12_months", "observable_12_month_outcome", 12),
    "cancellation_24m": ("cancellation_within_24_months", "observable_24_month_outcome", 24),
}
LABELS = {
    "status_schedule_logistic": "Status-schedule logistic",
    "full_logistic": "Full logistic",
    "random_forest": "Random forest",
    "extra_trees": "Extremely randomized trees",
    "hist_gradient_boosting": "Histogram gradient boosting",
    "lightgbm_cpu": "LightGBM",
    "catboost_gpu": "CatBoost",
    "xgboost_gpu": "XGBoost",
    "tabm_gpu": "TabM",
    "delivery_competing_risk_net_gpu": "Competing-risk network",
    "delivery_sparse_moe_gpu": "Delivery-PathMoE (transition)",
    "delivery_path_moe_gpu": "Delivery-PathMoE (rollout)",
    "equal_weight_pool_ablation": "Equal-weight pool (ablation)",
    "proposed_delivery_ensemble": "Delivery-state pooled ensemble",
    "empirical_markov": "Empirical Markov",
    "multinomial_logistic": "Multinomial logistic",
    "delivery_state_residual_net_gpu": "Delivery residual network",
    "delivery_net_no_markov_ablation": "Residual network, no Markov prior (ablation)",
}


def save(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame.to_csv(OUT / f"{name}.csv", index=False)
    return frame


def cluster_indices(clusters: pd.Series, rng: np.random.Generator) -> np.ndarray:
    codes, uniques = pd.factorize(clusters)
    members = [np.flatnonzero(codes == k) for k in range(len(uniques))]
    draw = rng.integers(0, len(uniques), len(uniques))
    return np.concatenate([members[k] for k in draw])


def load_predictions(endpoint: str) -> pd.DataFrame:
    return pd.read_parquet(PRIMARY / "predictions" / f"{endpoint}_test_predictions.parquet")


def model_names(frame: pd.DataFrame) -> list[str]:
    return [c[: -len("_probability")] for c in frame.columns if c.endswith("_probability")]


# ---------------------------------------------------------------------------
# 1. Cancellation class in next-state prediction (one-vs-rest ranking)
# ---------------------------------------------------------------------------
def next_state_class_c() -> pd.DataFrame:
    frame = pd.read_parquet(PRIMARY / "predictions" / "next_state_test_predictions.parquet")
    y = (frame["next_state"] == "C").astype(int).to_numpy()
    pre = frame["current_state"].isin(["P", "L", "T"]).to_numpy()
    models = sorted({c.rsplit("_prob_", 1)[0] for c in frame.columns if "_prob_" in c})
    rows = []
    for model in models:
        p = frame[f"{model}_prob_C"].to_numpy()
        prob_cols = [c for c in frame.columns if c.startswith(f"{model}_prob_")]
        argmax = frame[prob_cols].to_numpy().argmax(axis=1)
        predicted_c = np.array([prob_cols[i].endswith("_prob_C") for i in argmax])
        order = np.argsort(-p)
        top1 = order[: int(np.ceil(0.01 * len(p)))]
        top5 = order[: int(np.ceil(0.05 * len(p)))]
        rows.append(
            {
                "model": LABELS.get(model, model),
                "transitions": len(y),
                "cancellation_transitions": int(y.sum()),
                "prevalence": y.mean(),
                "roc_auc": roc_auc_score(y, p),
                "average_precision": average_precision_score(y, p),
                "roc_auc_preconstruction_states": roc_auc_score(y[pre], p[pre]),
                "average_precision_preconstruction_states": average_precision_score(y[pre], p[pre]),
                "max_predicted_probability": float(p.max()),
                "argmax_predictions_of_C": int(predicted_c.sum()),
                "recall_top_1pct": y[top1].sum() / y.sum(),
                "recall_top_5pct": y[top5].sum() / y.sum(),
            }
        )
    return save(pd.DataFrame(rows), "s_next_state_cancellation_ranking")


# ---------------------------------------------------------------------------
# 2. Threshold-based and imbalance-aware metrics
# ---------------------------------------------------------------------------
def threshold_metrics() -> pd.DataFrame:
    rows = []
    for endpoint, (outcome, _, _) in ENDPOINTS.items():
        frame = load_predictions(endpoint)
        perf = pd.read_csv(PRIMARY / "tables" / f"{endpoint}_model_performance.csv")
        thresholds = dict(zip(perf["model"], perf["threshold"]))
        y = frame[outcome].to_numpy()
        prevalence = y.mean()
        reference_brier = prevalence * (1 - prevalence)
        for model in model_names(frame):
            p = frame[f"{model}_probability"].to_numpy()
            threshold = thresholds[model]
            flag = p >= threshold
            tp = int((flag & (y == 1)).sum())
            fp = int((flag & (y == 0)).sum())
            fn = int((~flag & (y == 1)).sum())
            tn = int((~flag & (y == 0)).sum())
            brier = brier_score_loss(y, p)
            rows.append(
                {
                    "endpoint": endpoint,
                    "model": LABELS.get(model, model),
                    "n": len(y),
                    "events": int(y.sum()),
                    "prevalence": prevalence,
                    "roc_auc": roc_auc_score(y, p),
                    "pr_auc_average_precision": average_precision_score(y, p),
                    "brier": brier,
                    "reference_brier_prevalence_only": reference_brier,
                    "scaled_brier": 1 - brier / reference_brier,
                    "threshold_validation_f1": threshold,
                    "flagged": int(flag.sum()),
                    "true_positives": tp,
                    "false_positives": fp,
                    "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                    "specificity": tn / (tn + fp) if tn + fp else np.nan,
                    "ppv": tp / (tp + fp) if tp + fp else np.nan,
                    "npv": tn / (tn + fn) if tn + fn else np.nan,
                }
            )
    return save(pd.DataFrame(rows), "s_threshold_metrics")


# ---------------------------------------------------------------------------
# 3. Equivalence against the full logistic
# ---------------------------------------------------------------------------
def equivalence(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for endpoint, (outcome, _, _) in ENDPOINTS.items():
        frame = load_predictions(endpoint)
        y = frame[outcome].to_numpy()
        models = model_names(frame)
        draws = [cluster_indices(frame["development_cluster_id"], rng) for _ in range(REPS)]
        reference = frame["full_logistic_probability"].to_numpy()

        def scores(index: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
            return (
                roc_auc_score(y[index], p[index]),
                average_precision_score(y[index], p[index]),
                brier_score_loss(y[index], p[index]),
            )

        reference_scores = np.array([scores(ix, reference) for ix in draws])
        reference_point = np.array(scores(np.arange(len(y)), reference))
        for model in models:
            if model == "full_logistic":
                continue
            p = frame[f"{model}_probability"].to_numpy()
            point = np.array(scores(np.arange(len(y)), p)) - reference_point
            boot = np.array([scores(ix, p) for ix in draws]) - reference_scores
            for k, metric in enumerate(("roc_auc", "average_precision", "brier")):
                lo90, hi90 = np.percentile(boot[:, k], [5, 95])
                lo95, hi95 = np.percentile(boot[:, k], [2.5, 97.5])
                rows.append(
                    {
                        "endpoint": endpoint,
                        "model": LABELS.get(model, model),
                        "metric": metric,
                        "difference_vs_full_logistic": point[k],
                        "ci90_low": lo90,
                        "ci90_high": hi90,
                        "ci95_low": lo95,
                        "ci95_high": hi95,
                        "smallest_margin_for_equivalence": max(abs(lo90), abs(hi90)),
                    }
                )
    table = pd.DataFrame(rows)
    margins = {"roc_auc": 0.02, "average_precision": 0.05, "brier": 0.005}
    table["declared_margin"] = table["metric"].map(margins)
    table["within_margin_90"] = (table["ci90_low"] > -table["declared_margin"]) & (
        table["ci90_high"] < table["declared_margin"]
    )
    table["ci95_excludes_zero"] = (table["ci95_low"] > 0) | (table["ci95_high"] < 0)
    return save(table, "s_equivalence_vs_full_logistic")


# ---------------------------------------------------------------------------
# 4. Decision curves with treat-all and treat-none, and bootstrap intervals
# ---------------------------------------------------------------------------
def net_benefit(y: np.ndarray, p: np.ndarray, t: float) -> float:
    n = len(y)
    flag = p >= t
    return ((flag & (y == 1)).sum() - (flag & (y == 0)).sum() * t / (1 - t)) / n


def treat_all(y: np.ndarray, t: float) -> float:
    prevalence = y.mean()
    return prevalence - (1 - prevalence) * t / (1 - t)


def decision_curves(rng: np.random.Generator) -> pd.DataFrame:
    grid = {
        "cancellation_24m": [0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
        "operation_12m": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
    }
    rows = []
    band_rows = []
    for endpoint, (outcome, _, _) in ENDPOINTS.items():
        frame = load_predictions(endpoint)
        y = frame[outcome].to_numpy()
        draws = [cluster_indices(frame["development_cluster_id"], rng) for _ in range(REPS)]
        for model in model_names(frame):
            p = frame[f"{model}_probability"].to_numpy()
            for t in grid[endpoint]:
                nb = net_benefit(y, p, t)
                ta = treat_all(y, t)
                best_default = max(ta, 0.0)
                boot = np.array(
                    [net_benefit(y[ix], p[ix], t) - max(treat_all(y[ix], t), 0.0) for ix in draws]
                )
                rows.append(
                    {
                        "endpoint": endpoint,
                        "model": LABELS.get(model, model),
                        "threshold": t,
                        "net_benefit": nb,
                        "treat_all_net_benefit": ta,
                        "treat_none_net_benefit": 0.0,
                        "gain_over_best_default": nb - best_default,
                        "gain_ci95_low": np.percentile(boot, 2.5),
                        "gain_ci95_high": np.percentile(boot, 97.5),
                        "net_true_positives_per_100_units": 100 * nb,
                        "gain_per_100_units": 100 * (nb - best_default),
                    }
                )
            fine = np.round(np.arange(0.01, 0.51, 0.01), 2)
            beats = [t for t in fine if net_benefit(y, p, t) > max(treat_all(y, t), 0.0)]
            band_rows.append(
                {
                    "endpoint": endpoint,
                    "model": LABELS.get(model, model),
                    "thresholds_beating_both_defaults": ";".join(f"{t:.2f}" for t in beats),
                    "lowest": min(beats) if beats else np.nan,
                    "highest_contiguous": _highest_contiguous(beats),
                    "treat_all_crosses_zero_at": y.mean(),
                }
            )
    save(pd.DataFrame(band_rows), "s_decision_curve_bands")
    return save(pd.DataFrame(rows), "s_decision_curve_points")


def _highest_contiguous(values: list[float]) -> float:
    if not values:
        return np.nan
    run_end = values[0]
    for value in values[1:]:
        if round(value - run_end, 2) == 0.01:
            run_end = value
        else:
            break
    return run_end


# ---------------------------------------------------------------------------
# 5. Partition audit, timing and plant-overlap sensitivity
# ---------------------------------------------------------------------------
def partition_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    projects = pd.read_parquet(PRIMARY / "processed" / "projects.parquet")
    landmarks = pd.read_parquet(PRIMARY / "processed" / "landmarks.parquet")
    cohort = projects.loc[
        projects["eligible_primary_cohort"]
        & projects["analysis_split"].isin(["train", "validation", "test"])
    ].copy()
    spans = cohort.groupby("development_cluster_id")["analysis_split"].nunique()
    spanning = spans[spans > 1].index
    plant_spans = cohort.groupby("plant_id")["analysis_split"].nunique()
    earlier = {
        "validation": set(cohort.loc[cohort["analysis_split"] == "train", "plant_id"]),
        "test": set(
            cohort.loc[cohort["analysis_split"].isin(["train", "validation"]), "plant_id"]
        ),
    }
    earlier_clusters = set(
        cohort.loc[cohort["analysis_split"].isin(["train", "validation"]), "development_cluster_id"]
    )
    train_entities = set(cohort.loc[cohort["analysis_split"] == "train", "entity_id_initial"])
    rows = []
    for split in ("train", "validation", "test"):
        part = cohort.loc[cohort["analysis_split"] == split]
        rows.append(
            {
                "partition": split,
                "units": len(part),
                "development_clusters": part["development_cluster_id"].nunique(),
                "plants": part["plant_id"].nunique(),
                "entities": part["entity_id_initial"].nunique(),
                "first_entry_month": part["first_report_month"].min(),
                "last_entry_month": part["first_report_month"].max(),
                "units_in_clusters_spanning_partitions": int(
                    part["development_cluster_id"].isin(spanning).sum()
                ),
                "units_sharing_plant_with_earlier_partition": int(
                    part["plant_id"].isin(earlier.get(split, set())).sum()
                ),
                "units_sharing_cluster_with_earlier_partition": int(
                    part["development_cluster_id"].isin(earlier_clusters).sum()
                )
                if split == "test"
                else np.nan,
                "units_with_entity_seen_in_training": int(
                    part["entity_id_initial"].isin(train_entities).sum()
                )
                if split != "train"
                else np.nan,
            }
        )
    audit = pd.DataFrame(rows)
    audit.attrs["spanning_clusters"] = int(len(spanning))
    audit.attrs["plants_spanning"] = int((plant_spans > 1).sum())
    spanning_detail = cohort.loc[
        cohort["development_cluster_id"].isin(spanning),
        ["development_cluster_id", "unit_id", "first_report_month", "analysis_split", "event_type"],
    ]
    save(spanning_detail, "s_partition_spanning_clusters")
    save(audit, "s_partition_audit")

    # Outcome timing: last calendar month whose outcome enters each partition's labels.
    base = landmarks.loc[landmarks["landmark_age_months"] == 0]
    timing = []
    for endpoint, (outcome, observable, horizon) in ENDPOINTS.items():
        for split in ("train", "validation", "test"):
            part = base.loc[(base["analysis_split"] == split) & base[observable]]
            window_end = (part["report_ordinal"] + horizon).max()
            positives = part.loc[part[outcome] == 1]
            event_months = positives["report_ordinal"] + positives["months_to_event"]
            test_start = pd.Period(CONFIG.test_entry_start, freq="M").ordinal
            timing.append(
                {
                    "endpoint": endpoint,
                    "partition": split,
                    "n": len(part),
                    "events": int(part[outcome].sum()),
                    "prevalence": part[outcome].mean(),
                    "entry_first": str(pd.Period(ordinal=int(part["report_ordinal"].min()), freq="M")),
                    "entry_last": str(pd.Period(ordinal=int(part["report_ordinal"].max()), freq="M")),
                    "outcome_window_closes_by": str(pd.Period(ordinal=int(window_end), freq="M")),
                    "latest_event_month": str(pd.Period(ordinal=int(event_months.max()), freq="M")),
                    "events_resolved_on_or_after_test_entry_start": int((event_months >= test_start).sum()),
                    "units_whose_window_closes_after_test_entry_start": int(
                        ((part["report_ordinal"] + horizon) >= test_start).sum()
                    ),
                }
            )
    timing_table = save(pd.DataFrame(timing), "s_partition_outcome_timing")

    # Sensitivity: drop test units that share a plant (or entity) with earlier partitions.
    sens = []
    for endpoint, (outcome, _, _) in ENDPOINTS.items():
        frame = load_predictions(endpoint).merge(
            cohort[["unit_id", "plant_id", "entity_id_initial", "terminal_category"]],
            on="unit_id", how="left"
        )
        subsets = {
            "all test units": np.ones(len(frame), dtype=bool),
            "excluding units sharing a plant with earlier cohorts": ~frame["plant_id"].isin(
                earlier["test"]
            ).to_numpy(),
            "units of reporting entities unseen in training": ~frame["entity_id_initial"].isin(
                train_entities
            ).to_numpy(),
            "excluding units without a terminal record": (
                frame["terminal_category"] != "unresolved"
            ).to_numpy(),
        }
        for subset, mask in subsets.items():
            y = frame.loc[mask, outcome].to_numpy()
            for model in ("full_logistic", "proposed_delivery_ensemble", "catboost_gpu",
                          "xgboost_gpu", "tabm_gpu", "delivery_competing_risk_net_gpu"):
                p = frame.loc[mask, f"{model}_probability"].to_numpy()
                sens.append(
                    {
                        "endpoint": endpoint,
                        "subset": subset,
                        "model": LABELS[model],
                        "n": int(mask.sum()),
                        "events": int(y.sum()),
                        "roc_auc": roc_auc_score(y, p),
                        "average_precision": average_precision_score(y, p),
                        "brier": brier_score_loss(y, p),
                    }
                )
    sensitivity = save(pd.DataFrame(sens), "s_partition_overlap_sensitivity")
    return audit, timing_table, sensitivity


# ---------------------------------------------------------------------------
# 6. Refits: primary-protocol reproduction, deployment-date emulation,
#    alternative state definitions and learning curves
# ---------------------------------------------------------------------------
PHASE = {"P": "pre_construction", "L": "pre_construction", "T": "pre_construction",
         "U": "construction", "V": "construction", "TS": "commissioning"}


def _frames(outcome: str, observable: str) -> dict[str, pd.DataFrame]:
    landmarks = pd.read_parquet(PRIMARY / "processed" / "landmarks.parquet")
    frame = landmarks.loc[(landmarks["landmark_age_months"] == 0) & landmarks[observable]]
    return {s: frame.loc[frame["analysis_split"] == s].copy() for s in ("train", "validation", "test")}


def _fit_score(train, validation, test, outcome, categorical, numeric, model="logistic",
               calibrate=True, seed=SEED):
    if model == "logistic":
        pipe = _logistic_pipeline(categorical, numeric)
    else:
        pipe = _tree_pipeline(categorical, numeric, "hist_gradient_boosting", seed)
    pipe.fit(
        _prepare_features(train, categorical, numeric),
        train[outcome],
        classifier__sample_weight=train["cluster_equalizing_weight"],
    )
    p_test = pipe.predict_proba(_prepare_features(test, categorical, numeric))[:, 1]
    if calibrate:
        p_val = pipe.predict_proba(_prepare_features(validation, categorical, numeric))[:, 1]
        p_test = apply_platt(fit_platt(validation[outcome], p_val), p_test)
    y = test[outcome].to_numpy()
    intercept, slope = calibration_intercept_slope(test[outcome], p_test)
    return {
        "train_n": len(train),
        "train_events": int(train[outcome].sum()),
        "roc_auc": roc_auc_score(y, p_test),
        "average_precision": average_precision_score(y, p_test),
        "brier": brier_score_loss(y, p_test),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def refits() -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical, numeric = feature_columns(CONFIG)
    simple_categorical, simple_numeric = _simple_features()
    deployment_cutoff = pd.Period("2022-12", freq="M").ordinal
    rows = []
    curve = []
    rng = np.random.default_rng(SEED)
    for endpoint, (outcome, observable, horizon) in ENDPOINTS.items():
        f = _frames(outcome, observable)
        train, validation, test = f["train"], f["validation"], f["test"]

        def record(analysis, specification, result):
            rows.append({"endpoint": endpoint, "analysis": analysis,
                         "specification": specification, **result})

        record("primary protocol (reproduction)", "Full logistic",
               _fit_score(train, validation, test, outcome, categorical, numeric))
        record("primary protocol (reproduction)", "Histogram gradient boosting",
               _fit_score(train, validation, test, outcome, categorical, numeric, model="hgb"))
        # Deployment-date emulation: only labels whose outcome window closed by Dec 2022;
        # no validation recalibration because 2022 entrants are unresolved at that date.
        closed = train.loc[train["report_ordinal"] + horizon <= deployment_cutoff]
        record("labels resolved by Dec 2022, no recalibration", "Full logistic",
               _fit_score(closed, validation, test, outcome, categorical, numeric, calibrate=False))
        record("labels resolved by Dec 2022, no recalibration", "Histogram gradient boosting",
               _fit_score(closed, validation, test, outcome, categorical, numeric, model="hgb",
                          calibrate=False))
        record("labels resolved by Dec 2022, no recalibration", "Status-schedule logistic",
               _fit_score(closed, validation, test, outcome, simple_categorical, simple_numeric,
                          calibrate=False))
        # Alternative state definitions.
        alt = {}
        for name, frame in (("train", train), ("validation", validation), ("test", test)):
            g = frame.copy()
            g["delivery_phase"] = g["current_state"].map(PHASE).fillna("other")
            alt[name] = g
        phase_categorical = [c for c in categorical if c != "current_state"] + ["delivery_phase"]
        record("alternative state definition", "Full logistic, three delivery phases",
               _fit_score(alt["train"], alt["validation"], alt["test"], outcome,
                          phase_categorical, numeric))
        no_state = [c for c in categorical if c != "current_state"]
        record("alternative state definition", "Full logistic, reported state removed",
               _fit_score(train, validation, test, outcome, no_state, numeric))
        # Learning curve on whole development clusters of the training cohort.
        clusters = train["development_cluster_id"].unique()
        for fraction in (0.25, 0.5, 0.75, 1.0):
            repeats = 1 if fraction == 1.0 else 20
            for r in range(repeats):
                chosen = clusters if fraction == 1.0 else rng.choice(
                    clusters, int(round(fraction * len(clusters))), replace=False)
                sub = train.loc[train["development_cluster_id"].isin(chosen)]
                result = _fit_score(sub, validation, test, outcome, categorical, numeric)
                curve.append({"endpoint": endpoint, "training_fraction": fraction,
                              "repeat": r, **result})
    refit_table = save(pd.DataFrame(rows), "s_refit_sensitivity")
    curve_table = pd.DataFrame(curve)
    summary = (
        curve_table.groupby(["endpoint", "training_fraction"])
        .agg(train_events_mean=("train_events", "mean"),
             roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
             average_precision_mean=("average_precision", "mean"),
             average_precision_sd=("average_precision", "std"),
             brier_mean=("brier", "mean"))
        .reset_index()
    )
    save(curve_table, "s_learning_curve_runs")
    return refit_table, save(summary, "s_learning_curve")


# ---------------------------------------------------------------------------
# 7. Identity and record-handling audit
# ---------------------------------------------------------------------------
def identity_audit() -> pd.DataFrame:
    projects = pd.read_parquet(PRIMARY / "processed" / "projects.parquet")
    audit = json.loads((PRIMARY / "results" / "data_quality_audit.json").read_text(encoding="utf-8"))
    primary = projects.loc[projects["eligible_primary_cohort"]]
    key = projects["generator_id"].astype(str).str.upper().map(lambda s: re.sub(r"[^A-Z0-9]", "", s))
    projects = projects.assign(normalised_generator=key)
    groups = projects.groupby(["plant_id", "normalised_generator"])["unit_id"].nunique()
    recoded = groups[groups > 1]
    recoded_units = projects.set_index(["plant_id", "normalised_generator"]).index.isin(recoded.index)
    rec = projects.loc[recoded_units]
    overlap = 0
    for _, g in rec.groupby(["plant_id", "normalised_generator"]):
        spans = g[["first_report_ordinal", "last_planned_report_ordinal"]].to_numpy()
        spans = spans[np.argsort(spans[:, 0])]
        overlap += int(any(spans[i + 1, 0] <= spans[i, 1] for i in range(len(spans) - 1)))
    rows = [
        ("Monthly files parsed", audit["continuity"]["monthly_file_count"], "September 2015 to May 2026, no missing or duplicated month"),
        ("Planned-inventory rows parsed", audit["observations"]["monthly_planned_rows"], "Planned sheet of every monthly file"),
        ("Unit-month rows with a duplicated plant and generator key", audit["observations"]["duplicate_unit_month_rows"], "first row retained if any occurred"),
        ("Units ever reported as planned", audit["observations"]["unique_ever_planned_units"], "unit key: EIA plant code and generator code"),
        ("Generator codes equal after removing punctuation and case (groups)", int(len(recoded)), "candidate re-coded identifiers within a plant"),
        ("  of which with overlapping reporting months", overlap, "simultaneously reported, so distinct generators"),
        ("  units involved", int(rec["unit_id"].nunique()), "kept as separate units"),
        ("  units involved in the primary cohort", int(rec.loc[rec["eligible_primary_cohort"], "unit_id"].nunique()), ""),
        ("Internal gaps in a unit's monthly history", audit["transitions"]["internal_nonconsecutive_observation_gaps"], "transitions spanning a gap excluded"),
        ("Units whose reporting entity changed", audit["projects"]["units_with_entity_changes"], "entity at first report used for entity features"),
        ("Units whose technology group changed", audit["projects"]["units_with_technology_group_changes"], "technology at first report used"),
        ("Units whose reported state code changed", audit["projects"]["units_with_state_changes"], "state at first report used"),
        ("Units without a terminal identity in the May 2026 inventory", audit["projects"]["unresolved_terminal_identity"], "censored"),
        ("  of which in the primary cohort", int((primary["terminal_category"] == "unresolved").sum()), "censored"),
        ("Primary-cohort units cancelled with no Canceled-sheet month", int(((primary["event_type"] == "C") & primary["first_canceled_ordinal"].isna()).sum()), "event month set to the month after the last planned report"),
        ("Missing capacity (share of panel rows)", audit["primary_missingness_fraction"]["capacity_mw"], "median imputation within the training cohort"),
        ("Missing planned operation month (share of panel rows)", audit["primary_missingness_fraction"]["planned_operation_ordinal"], "median imputation within the training cohort"),
    ]
    return save(pd.DataFrame(rows, columns=["item", "value", "treatment"]), "s_identity_audit")


# ---------------------------------------------------------------------------
# 8. Denominators and stratified attrition
# ---------------------------------------------------------------------------
REGION = {
    **dict.fromkeys(["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"], "Northeast"),
    **dict.fromkeys(["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"], "Midwest"),
    **dict.fromkeys(["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"], "South"),
    **dict.fromkeys(["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"], "West"),
}


def denominators() -> tuple[pd.DataFrame, pd.DataFrame]:
    projects = pd.read_parquet(PRIMARY / "processed" / "projects.parquet")
    p = projects.loc[projects["eligible_primary_cohort"]].copy()
    p["outcome"] = p["terminal_category"].replace({"operating": "operated", "retired": "operated"})
    rows = []
    for state, g in p.groupby("initial_state"):
        if state not in PHASE:
            continue
        c = (g["outcome"] == "canceled").sum()
        o = (g["outcome"] == "operated").sum()
        rows.append({
            "entry_state": state, "units": len(g), "cancelled": int(c), "operated": int(o),
            "still_planned": int((g["outcome"] == "planned").sum()),
            "unresolved": int((g["outcome"] == "unresolved").sum()),
            "cancelled_share_all_units": c / len(g),
            "cancelled_share_resolved_units": c / (c + o),
            "capacity_mw_all": g["capacity_mw_initial"].sum(),
            "capacity_mw_cancelled": g.loc[g["outcome"] == "canceled", "capacity_mw_initial"].sum(),
            "capacity_mw_operated": g.loc[g["outcome"] == "operated", "capacity_mw_initial"].sum(),
        })
    by_state = pd.DataFrame(rows)
    total = {
        "entry_state": "All", "units": len(p),
        "cancelled": int((p["outcome"] == "canceled").sum()),
        "operated": int((p["outcome"] == "operated").sum()),
        "still_planned": int((p["outcome"] == "planned").sum()),
        "unresolved": int((p["outcome"] == "unresolved").sum()),
        "capacity_mw_all": p["capacity_mw_initial"].sum(),
        "capacity_mw_cancelled": p.loc[p["outcome"] == "canceled", "capacity_mw_initial"].sum(),
        "capacity_mw_operated": p.loc[p["outcome"] == "operated", "capacity_mw_initial"].sum(),
    }
    total["cancelled_share_all_units"] = total["cancelled"] / total["units"]
    total["cancelled_share_resolved_units"] = total["cancelled"] / (total["cancelled"] + total["operated"])
    by_state = pd.concat([by_state, pd.DataFrame([total])], ignore_index=True)
    by_state["cancelled_capacity_share_resolved"] = by_state["capacity_mw_cancelled"] / (
        by_state["capacity_mw_cancelled"] + by_state["capacity_mw_operated"])
    save(by_state, "s_denominators_by_entry_state")

    # Stratified 24-month cancellation among units with 24 months of potential follow-up.
    f = p.loc[p["has_24_month_potential_followup"]].copy()
    months = f["event_ordinal"] - f["first_report_ordinal"]
    f["cancelled_24m"] = ((f["event_type"] == "C") & (months <= 24)).astype(int)
    f["phase"] = f["initial_state"].map(PHASE)
    f["region"] = f["state_code"].map(REGION).fillna("Other")
    f["size_class"] = pd.cut(f["capacity_mw_initial"], [-np.inf, 5, 50, 150, np.inf],
                             labels=["5 MW or less", "over 5 to 50 MW", "over 50 to 150 MW", "over 150 MW"])
    strata = []
    for dimension in ("phase", "technology_group", "region", "size_class"):
        for level, g in f.groupby(dimension, observed=True):
            if len(g) < 20:
                continue
            for phase, h in [("all phases", g)] + ([] if dimension == "phase" else
                                                    [(ph, g.loc[g["phase"] == ph]) for ph in ("pre_construction", "construction")]):
                if len(h) < 20:
                    continue
                strata.append({
                    "dimension": dimension, "level": str(level), "entry_phase": phase,
                    "units": len(h), "cancelled_within_24m": int(h["cancelled_24m"].sum()),
                    "share": h["cancelled_24m"].mean(),
                    "capacity_mw": h["capacity_mw_initial"].sum(),
                    "capacity_mw_cancelled_24m": h.loc[h["cancelled_24m"] == 1, "capacity_mw_initial"].sum(),
                })
    return by_state, save(pd.DataFrame(strata), "s_stratified_attrition_24m")


def main() -> None:
    rng = np.random.default_rng(SEED)
    results: dict[str, object] = {}
    c = next_state_class_c()
    t = threshold_metrics()
    e = equivalence(rng)
    d = decision_curves(rng)
    audit, timing, overlap = partition_audit()
    refit, curve = refits()
    ident = identity_audit()
    denom, strata = denominators()
    results["spanning_clusters"] = audit.attrs.get("spanning_clusters")
    results["plants_spanning_partitions"] = audit.attrs.get("plants_spanning")
    results["seed"] = SEED
    results["bootstrap_reps"] = REPS
    (OUT / "manifest.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("done", OUT)


if __name__ == "__main__":
    main()
