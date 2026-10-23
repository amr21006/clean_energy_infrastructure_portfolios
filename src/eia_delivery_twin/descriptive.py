from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import entropy
from statsmodels.stats.proportion import proportion_confint

from .config import AnalysisConfig
from .panel import STATE_RANK


def _save(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return table


def cohort_flow_table(projects: pd.DataFrame) -> pd.DataFrame:
    stages = [
        ("All EIA-860M units ever observed as planned", pd.Series(True, index=projects.index)),
        ("After 12-month archive burn-in", projects["passes_burn_in"]),
        (
            "Clean-infrastructure technologies",
            projects["passes_burn_in"] & projects["is_clean_primary"],
        ),
        (
            "No operation before or at first planned observation",
            projects["passes_burn_in"]
            & projects["is_clean_primary"]
            & ~projects["operation_before_or_at_entry"],
        ),
        ("Primary longitudinal cohort", projects["eligible_primary_cohort"]),
        (
            "Primary cohort with at least 24 months potential follow-up",
            projects["eligible_primary_cohort"]
            & projects["has_24_month_potential_followup"],
        ),
    ]
    rows = []
    previous = None
    for order, (label, mask) in enumerate(stages, start=1):
        count = int(mask.sum())
        clusters = int(projects.loc[mask, "development_cluster_id"].nunique())
        rows.append(
            {
                "order": order,
                "cohort_stage": label,
                "unit_count": count,
                "development_cluster_count": clusters,
                "removed_from_previous": previous - count if previous is not None else 0,
            }
        )
        previous = count
    return pd.DataFrame(rows)


def project_characteristics_table(projects: pd.DataFrame) -> pd.DataFrame:
    primary = projects.loc[projects["eligible_primary_cohort"]].copy()
    rows: list[dict[str, Any]] = []
    for split, group in primary.groupby("analysis_split", dropna=False):
        capacities = group["capacity_mw_initial"].dropna()
        rows.append(
            {
                "analysis_split": split,
                "units": len(group),
                "development_clusters": group["development_cluster_id"].nunique(),
                "entities": group["entity_id_initial"].nunique(),
                "states_or_territories": group["state_code"].nunique(),
                "capacity_mw_sum": capacities.sum(),
                "capacity_mw_median": capacities.median(),
                "capacity_mw_iqr_low": capacities.quantile(0.25),
                "capacity_mw_iqr_high": capacities.quantile(0.75),
                "operating_terminal_n": int(
                    group["terminal_category"].isin(["operating", "retired"]).sum()
                ),
                "canceled_terminal_n": int(
                    (group["terminal_category"] == "canceled").sum()
                ),
                "still_planned_terminal_n": int(
                    (group["terminal_category"] == "planned").sum()
                ),
                "unresolved_terminal_n": int(
                    (group["terminal_category"] == "unresolved").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def outcome_by_subgroup(projects: pd.DataFrame, subgroup: str) -> pd.DataFrame:
    primary = projects.loc[projects["eligible_primary_cohort"]].copy()
    primary["outcome"] = primary["terminal_category"].replace(
        {"retired": "operating_or_later_retired", "operating": "operating_or_later_retired"}
    )
    counts = (
        primary.groupby([subgroup, "outcome"], dropna=False)
        .agg(
            units=("unit_id", "size"),
            development_clusters=("development_cluster_id", "nunique"),
            capacity_mw=("capacity_mw_initial", "sum"),
        )
        .reset_index()
    )
    totals = counts.groupby(subgroup)["units"].transform("sum")
    counts["unit_share_within_subgroup"] = counts["units"] / totals
    return counts.sort_values([subgroup, "outcome"]).reset_index(drop=True)


def transition_matrices(transitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = transitions.loc[transitions["eligible_primary_cohort"]].copy()
    counts = pd.crosstab(primary["current_state"], primary["next_state"])
    all_states = ["P", "L", "T", "U", "V", "TS", "OTHER", "O", "C"]
    counts = counts.reindex(index=all_states[:-2], columns=all_states, fill_value=0)
    probabilities = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    counts_long = (
        counts.rename_axis("current_state")
        .reset_index()
        .melt(id_vars="current_state", var_name="next_state", value_name="count")
    )
    probabilities_long = (
        probabilities.rename_axis("current_state")
        .reset_index()
        .melt(
            id_vars="current_state", var_name="next_state", value_name="probability"
        )
    )
    return counts_long, probabilities_long


def transition_metrics(transitions: pd.DataFrame) -> pd.DataFrame:
    primary = transitions.loc[transitions["eligible_primary_cohort"]].copy()

    def transition_type(row: pd.Series) -> str:
        current = STATE_RANK.get(row["current_state"])
        following = STATE_RANK.get(row["next_state"])
        if row["next_state"] == "O":
            return "operation"
        if row["next_state"] == "C":
            return "cancellation"
        if current is None or following is None:
            return "other"
        if following > current:
            return "advance"
        if following < current:
            return "reversal"
        return "remain"

    primary["transition_type"] = primary.apply(transition_type, axis=1)
    rows = []
    for state, group in primary.groupby("current_state"):
        counts = group["transition_type"].value_counts()
        next_probabilities = group["next_state"].value_counts(normalize=True)
        rows.append(
            {
                "current_state": state,
                "transition_count": len(group),
                "remain_probability": counts.get("remain", 0) / len(group),
                "advance_probability": counts.get("advance", 0) / len(group),
                "reversal_probability": counts.get("reversal", 0) / len(group),
                "operation_probability": counts.get("operation", 0) / len(group),
                "cancellation_probability": counts.get("cancellation", 0) / len(group),
                "next_state_entropy": entropy(next_probabilities.to_numpy(), base=2),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "current_state", key=lambda s: s.map({**STATE_RANK, "OTHER": 99})
    )


def state_spell_table(panel: pd.DataFrame) -> pd.DataFrame:
    primary = panel.loc[panel["eligible_primary_cohort"]].copy()
    spell_rows: list[dict[str, Any]] = []
    for unit_id, history in primary.groupby("unit_id", sort=False):
        history = history.sort_values("report_ordinal")
        spell_id = (
            (history["current_state"] != history["current_state"].shift())
            | (history["report_ordinal"].diff().fillna(1) != 1)
        ).cumsum()
        for _, spell in history.groupby(spell_id):
            spell_rows.append(
                {
                    "unit_id": unit_id,
                    "technology_group": spell["technology_group"].iloc[0],
                    "state": spell["current_state"].iloc[0],
                    "spell_start_month": spell["report_month"].iloc[0],
                    "spell_end_month": spell["report_month"].iloc[-1],
                    "spell_duration_observed_months": len(spell),
                    "terminal_category": spell["terminal_category"].iloc[0],
                }
            )
    spells = pd.DataFrame(spell_rows)
    summary = (
        spells.groupby(["state", "technology_group"], dropna=False)[
            "spell_duration_observed_months"
        ]
        .agg(
            spells="size",
            mean_months="mean",
            median_months="median",
            p25_months=lambda x: x.quantile(0.25),
            p75_months=lambda x: x.quantile(0.75),
            p90_months=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )
    return summary


def schedule_and_path_table(panel: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
    last = (
        panel.sort_values(["unit_id", "report_ordinal"])
        .groupby("unit_id", as_index=False)
        .tail(1)
    )
    columns = [
        "unit_id",
        "schedule_revision_count",
        "cumulative_schedule_slip_months",
        "absolute_schedule_movement_months",
        "state_reversal_count",
    ]
    result = projects.merge(last[columns], on="unit_id", how="left")
    return result.loc[result["eligible_primary_cohort"]].copy()


def schedule_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grouping, group in schedule.groupby("technology_group", dropna=False):
        for outcome, subset in group.groupby("terminal_category", dropna=False):
            rows.append(
                {
                    "technology_group": grouping,
                    "terminal_category": outcome,
                    "units": len(subset),
                    "any_schedule_revision_share": (
                        subset["schedule_revision_count"] > 0
                    ).mean(),
                    "revision_count_median": subset[
                        "schedule_revision_count"
                    ].median(),
                    "net_schedule_slip_median_months": subset[
                        "cumulative_schedule_slip_months"
                    ].median(),
                    "net_schedule_slip_p75_months": subset[
                        "cumulative_schedule_slip_months"
                    ].quantile(0.75),
                    "absolute_schedule_movement_median_months": subset[
                        "absolute_schedule_movement_months"
                    ].median(),
                    "any_state_reversal_share": (
                        subset["state_reversal_count"] > 0
                    ).mean(),
                }
            )
    return pd.DataFrame(rows)


def pathway_table(panel: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
    project_map = projects.set_index("unit_id").to_dict("index")
    paths: list[dict[str, Any]] = []
    for unit_id, history in panel.loc[
        panel["eligible_primary_cohort"]
    ].groupby("unit_id", sort=False):
        states = history.sort_values("report_ordinal")["current_state"].tolist()
        collapsed = [state for index, state in enumerate(states) if index == 0 or state != states[index - 1]]
        project = project_map[unit_id]
        if project["event_type"] in {"O", "C"}:
            collapsed.append(project["event_type"])
        paths.append(
            {
                "unit_id": unit_id,
                "technology_group": project["technology_group"],
                "terminal_category": project["terminal_category"],
                "path": ">".join(collapsed),
                "distinct_state_count": len(set(collapsed)),
                "path_step_count": len(collapsed) - 1,
            }
        )
    paths_frame = pd.DataFrame(paths)
    summary = (
        paths_frame.groupby(
            ["path", "technology_group", "terminal_category"], dropna=False
        )
        .agg(
            units=("unit_id", "size"),
            distinct_states=("distinct_state_count", "median"),
            path_steps=("path_step_count", "median"),
        )
        .reset_index()
    )
    summary["share_within_technology"] = summary["units"] / summary.groupby(
        "technology_group"
    )["units"].transform("sum")
    return summary.sort_values(
        ["technology_group", "units"], ascending=[True, False]
    ).reset_index(drop=True)


def horizon_outcome_table(projects: pd.DataFrame) -> pd.DataFrame:
    primary = projects.loc[projects["eligible_primary_cohort"]].copy()
    rows = []
    for horizon in (6, 12, 24):
        enough_followup = primary["first_report_ordinal"] <= (
            primary["first_report_ordinal"].max() + 0
        )
        # The archive endpoint is inferred from the largest observed censor/event month.
        endpoint = int(
            max(
                primary["last_planned_report_ordinal"].max(),
                primary["event_ordinal"].dropna().max(),
            )
        )
        observable = (endpoint - primary["first_report_ordinal"] >= horizon) | (
            primary["event_ordinal"].notna()
            & (primary["event_ordinal"] - primary["first_report_ordinal"] <= horizon)
        )
        eligible = primary.loc[observable].copy()
        eligible["operated"] = (
            (eligible["event_type"] == "O")
            & (
                eligible["event_ordinal"] - eligible["first_report_ordinal"]
                <= horizon
            )
        )
        eligible["canceled"] = (
            (eligible["event_type"] == "C")
            & (
                eligible["event_ordinal"] - eligible["first_report_ordinal"]
                <= horizon
            )
        )
        for dimensions, group in eligible.groupby(
            ["initial_state", "technology_group"], dropna=False
        ):
            initial_state, technology = dimensions
            for outcome in ("operated", "canceled"):
                events = int(group[outcome].sum())
                lower, upper = proportion_confint(
                    events, len(group), alpha=0.05, method="wilson"
                )
                rows.append(
                    {
                        "horizon_months": horizon,
                        "initial_state": initial_state,
                        "technology_group": technology,
                        "outcome": outcome,
                        "eligible_units": len(group),
                        "events": events,
                        "probability": events / len(group),
                        "ci95_low": lower,
                        "ci95_high": upper,
                    }
                )
    return pd.DataFrame(rows)


def state_occupancy_table(panel: pd.DataFrame) -> pd.DataFrame:
    primary = panel.loc[panel["eligible_primary_cohort"]]
    return (
        primary.groupby(
            ["report_month", "current_state", "technology_group"], dropna=False
        )
        .agg(
            units=("unit_id", "nunique"),
            capacity_mw=("capacity_mw", "sum"),
        )
        .reset_index()
        .sort_values(["report_month", "current_state", "technology_group"])
    )


def development_cluster_table(projects: pd.DataFrame) -> pd.DataFrame:
    primary = projects.loc[projects["eligible_primary_cohort"]]
    return (
        primary.groupby("development_cluster_id")
        .agg(
            units=("unit_id", "size"),
            plant_id=("plant_id", "first"),
            technology_group=("technology_group", "first"),
            first_report_month=("first_report_month", "min"),
            capacity_mw=("capacity_mw_initial", "sum"),
            outcome_count=("terminal_category", "nunique"),
        )
        .reset_index()
    )


def generate_data_dictionary(
    projects: pd.DataFrame,
    panel: pd.DataFrame,
    transitions: pd.DataFrame,
    landmarks: pd.DataFrame,
) -> pd.DataFrame:
    descriptions = {
        "unit_id": "Deterministic EIA Plant ID plus Generator ID key.",
        "development_cluster_id": "Plant, technology group, and initial reported operation-month cluster used to reduce multi-unit pseudo-replication.",
        "current_state": "Harmonized monthly delivery state.",
        "next_state": "Observed following-month state or immediate terminal state.",
        "event_type": "First modeled terminal event: O operation, C cancellation, or censored.",
        "schedule_revision_count": "Cumulative count of reported operation-date changes available at the observation month.",
        "cumulative_schedule_slip_months": "Current reported operation month minus the initial reported operation month.",
        "state_reversal_count": "Cumulative number of backward movements in the ordered delivery-stage sequence.",
        "cluster_equalizing_weight": "Inverse number of generating units in the plant-development cluster.",
    }
    rows = []
    for dataset, frame in (
        ("projects", projects),
        ("unit_month_panel", panel),
        ("monthly_transitions", transitions),
        ("landmarks", landmarks),
    ):
        for column in frame.columns:
            rows.append(
                {
                    "dataset": dataset,
                    "variable": column,
                    "dtype": str(frame[column].dtype),
                    "description": descriptions.get(column, "Derived or source variable; see analysis code."),
                    "missing_fraction": float(frame[column].isna().mean()),
                    "unique_values": int(frame[column].nunique(dropna=True)),
                }
            )
    return pd.DataFrame(rows)


def generate_descriptive_outputs(
    config: AnalysisConfig,
    projects: pd.DataFrame,
    panel: pd.DataFrame,
    transitions: pd.DataFrame,
    landmarks: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    transition_counts, transition_probabilities = transition_matrices(transitions)
    schedule_paths = schedule_and_path_table(panel, projects)
    tables = {
        "cohort_flow": cohort_flow_table(projects),
        "project_characteristics_by_split": project_characteristics_table(projects),
        "outcomes_by_technology": outcome_by_subgroup(projects, "technology_group"),
        "outcomes_by_initial_state": outcome_by_subgroup(projects, "initial_state"),
        "outcomes_by_sector": outcome_by_subgroup(projects, "sector_group"),
        "transition_counts": transition_counts,
        "transition_probabilities": transition_probabilities,
        "transition_metrics": transition_metrics(transitions),
        "state_spell_summary": state_spell_table(panel),
        "schedule_and_reversal_summary": schedule_summary(schedule_paths),
        "top_pathways": pathway_table(panel, projects),
        "horizon_outcomes": horizon_outcome_table(projects),
        "state_occupancy_monthly": state_occupancy_table(panel),
        "development_clusters": development_cluster_table(projects),
        "data_dictionary": generate_data_dictionary(
            projects, panel, transitions, landmarks
        ),
    }
    for name, table in tables.items():
        _save(table, config.tables_dir / f"{name}.csv")
    return tables
