from __future__ import annotations

import bisect
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .io import (
    FastXlsxReader,
    discover_monthly_files,
    harmonize_status,
    normalize_header,
    normalize_identifier,
    period_from_year_month,
    record_value,
    safe_float,
    sector_group,
    technology_group,
)


STATE_RANK = {"P": 0, "L": 1, "T": 2, "U": 3, "V": 4, "TS": 5}
TERMINAL_PRIORITY = {"canceled": 1, "planned": 2, "retired": 3, "operating": 4}


def period_ordinal(period: pd.Period | None) -> float:
    return float(period.ordinal) if period is not None else np.nan


def period_string(period: pd.Period | None) -> str | None:
    return str(period) if period is not None else None


def _unit_id(plant_id: str | None, generator_id: str | None) -> str | None:
    if not plant_id or not generator_id:
        return None
    return f"{plant_id}|{generator_id}"


def _planned_record(record: dict[str, object], period: pd.Period) -> dict[str, Any] | None:
    plant_id = normalize_identifier(record_value(record, "Plant ID"))
    generator_id = normalize_identifier(record_value(record, "Generator ID"))
    unit_id = _unit_id(plant_id, generator_id)
    if unit_id is None or plant_id.startswith("NOTES"):
        return None
    technology = normalize_header(record_value(record, "Technology"))
    energy_source = normalize_header(record_value(record, "Energy Source Code"))
    prime_mover = normalize_header(record_value(record, "Prime Mover Code"))
    raw_sector = normalize_header(record_value(record, "Sector", "Sector Name"))
    planned_operation = period_from_year_month(
        record_value(record, "Planned Operation Year"),
        record_value(record, "Planned Operation Month"),
    )
    raw_status = normalize_header(record_value(record, "Status"))
    return {
        "unit_id": unit_id,
        "plant_id": plant_id,
        "generator_id": generator_id,
        "report_month": str(period),
        "report_ordinal": period.ordinal,
        "entity_id": normalize_identifier(record_value(record, "Entity ID")),
        "entity_name": normalize_header(record_value(record, "Entity Name")),
        "plant_name": normalize_header(record_value(record, "Plant Name")),
        "state_code": normalize_header(record_value(record, "Plant State")),
        "county": normalize_header(record_value(record, "County")),
        "balancing_authority_code": normalize_header(
            record_value(record, "Balancing Authority Code")
        ),
        "sector_name": raw_sector,
        "sector_group": sector_group(raw_sector),
        "technology_name": technology,
        "technology_group": technology_group(
            technology, energy_source, prime_mover
        ),
        "energy_source_code": energy_source,
        "prime_mover_code": prime_mover,
        "capacity_mw": safe_float(
            record_value(
                record, "Nameplate Capacity (MW)", "Net Summer Capacity (MW)"
            )
        ),
        "planned_operation_month": period_string(planned_operation),
        "planned_operation_ordinal": period_ordinal(planned_operation),
        "raw_status": raw_status,
        "current_state": harmonize_status(raw_status),
        "latitude": safe_float(record_value(record, "Latitude")),
        "longitude": safe_float(record_value(record, "Longitude")),
    }


def read_planned_observations(
    monthly_files: list[tuple[pd.Period, Path]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    monthly_counts: dict[str, int] = {}
    sheet_counts: Counter[str] = Counter()
    for period, path in monthly_files:
        period_count = 0
        sheets = FastXlsxReader(path).read_sheets(("Planned",))
        for sheet_name, sheet in sheets.items():
            sheet_counts[sheet_name] += 1
            for raw_record in sheet.records():
                record = _planned_record(raw_record, period)
                if record is None:
                    continue
                rows.append(record)
                period_count += 1
        monthly_counts[str(period)] = period_count
    observations = pd.DataFrame.from_records(rows)
    if observations.empty:
        raise RuntimeError("No planned observations were parsed from EIA-860M.")
    observations.sort_values(["unit_id", "report_ordinal"], inplace=True)
    duplicate_mask = observations.duplicated(["unit_id", "report_ordinal"], keep=False)
    audit = {
        "monthly_planned_rows": int(len(observations)),
        "unique_ever_planned_units": int(observations["unit_id"].nunique()),
        "duplicate_unit_month_rows": int(duplicate_mask.sum()),
        "monthly_counts": monthly_counts,
        "sheet_file_counts": dict(sheet_counts),
        "status_counts": observations["current_state"].value_counts().to_dict(),
        "technology_counts": observations["technology_group"].value_counts().to_dict(),
    }
    if duplicate_mask.any():
        observations = observations.drop_duplicates(
            ["unit_id", "report_ordinal"], keep="first"
        )
    return observations.reset_index(drop=True), audit


def read_first_cancellation_months(
    monthly_files: list[tuple[pd.Period, Path]], first_planned_ordinal: dict[str, int]
) -> dict[str, pd.Period]:
    first_canceled: dict[str, pd.Period] = {}
    for period, path in monthly_files:
        sheets = FastXlsxReader(path).read_sheets(("Canceled",))
        for sheet in sheets.values():
            for record in sheet.records():
                unit_id = _unit_id(
                    normalize_identifier(record_value(record, "Plant ID")),
                    normalize_identifier(record_value(record, "Generator ID")),
                )
                if (
                    unit_id in first_planned_ordinal
                    and period.ordinal >= first_planned_ordinal[unit_id]
                    and unit_id not in first_canceled
                ):
                    first_canceled[unit_id] = period
    return first_canceled


def read_terminal_inventory(
    final_period: pd.Period, final_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    overlaps: Counter[str] = Counter()
    sheets = FastXlsxReader(final_path).read_sheets(
        ("Operating", "Planned", "Retired", "Canceled")
    )
    for sheet_name, sheet in sheets.items():
        lower = sheet_name.lower()
        if lower.startswith("operating"):
            category = "operating"
        elif lower.startswith("planned"):
            category = "planned"
        elif lower.startswith("retired"):
            category = "retired"
        elif lower.startswith("canceled"):
            category = "canceled"
        else:
            continue
        for record in sheet.records():
            plant_id = normalize_identifier(record_value(record, "Plant ID"))
            generator_id = normalize_identifier(record_value(record, "Generator ID"))
            unit_id = _unit_id(plant_id, generator_id)
            if unit_id is None or (plant_id and plant_id.startswith("NOTES")):
                continue
            operating_period = period_from_year_month(
                record_value(record, "Operating Year"),
                record_value(record, "Operating Month"),
            )
            retirement_period = period_from_year_month(
                record_value(record, "Retirement Year"),
                record_value(record, "Retirement Month"),
            )
            row = {
                "unit_id": unit_id,
                "terminal_category": category,
                "operating_month": period_string(operating_period),
                "operating_ordinal": period_ordinal(operating_period),
                "retirement_month": period_string(retirement_period),
                "retirement_ordinal": period_ordinal(retirement_period),
                "terminal_inventory_month": str(final_period),
            }
            if unit_id in records:
                overlaps[f"{records[unit_id]['terminal_category']}|{category}"] += 1
            if unit_id not in records or TERMINAL_PRIORITY[category] > TERMINAL_PRIORITY[
                records[unit_id]["terminal_category"]
            ]:
                records[unit_id] = row
    terminal = pd.DataFrame.from_records(list(records.values()))
    audit = {
        "terminal_inventory_rows": int(len(terminal)),
        "terminal_category_counts": terminal["terminal_category"]
        .value_counts()
        .to_dict(),
        "terminal_category_overlaps": dict(overlaps),
    }
    return terminal, audit


def _mode_or_first(series: pd.Series) -> object:
    nonmissing = series.dropna()
    nonmissing = nonmissing[nonmissing.astype(str) != ""]
    if nonmissing.empty:
        return None
    modes = nonmissing.mode()
    return modes.iloc[0] if not modes.empty else nonmissing.iloc[0]


def build_project_table(
    observations: pd.DataFrame,
    terminal: pd.DataFrame,
    first_canceled: dict[str, pd.Period],
    config: AnalysisConfig,
    first_archive_period: pd.Period,
    final_archive_period: pd.Period,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    terminal_map = terminal.set_index("unit_id").to_dict("index")
    project_rows: list[dict[str, Any]] = []
    for unit_id, history in observations.groupby("unit_id", sort=False):
        history = history.sort_values("report_ordinal")
        first = history.iloc[0]
        last = history.iloc[-1]
        terminal_record = terminal_map.get(unit_id, {})
        terminal_category = terminal_record.get("terminal_category", "unresolved")
        operation_ordinal = terminal_record.get("operating_ordinal", np.nan)
        operation_ordinal = (
            float(operation_ordinal)
            if operation_ordinal is not None and pd.notna(operation_ordinal)
            else np.nan
        )
        cancel_period = first_canceled.get(unit_id)
        cancel_ordinal = (
            float(cancel_period.ordinal) if cancel_period is not None else np.nan
        )
        event_type = "censored"
        event_ordinal = np.nan
        operation_before_or_at_entry = False
        if terminal_category in {"operating", "retired"} and pd.notna(
            operation_ordinal
        ):
            if operation_ordinal <= float(first["report_ordinal"]):
                operation_before_or_at_entry = True
            else:
                event_type = "O"
                event_ordinal = operation_ordinal
        elif terminal_category == "canceled":
            event_type = "C"
            if pd.notna(cancel_ordinal) and cancel_ordinal >= float(
                first["report_ordinal"]
            ):
                event_ordinal = cancel_ordinal
            else:
                event_ordinal = float(last["report_ordinal"]) + 1

        initial_schedule = first["planned_operation_ordinal"]
        initial_schedule_key = (
            str(first["planned_operation_month"])
            if pd.notna(first["planned_operation_month"])
            else str(first["report_month"])
        )
        cluster_key = "|".join(
            (
                str(first["plant_id"]),
                str(first["technology_group"]),
                initial_schedule_key,
            )
        )
        first_period = pd.Period(str(first["report_month"]), freq="M")
        split = _assign_split(first_period, config)
        project_rows.append(
            {
                "unit_id": unit_id,
                "plant_id": first["plant_id"],
                "generator_id": first["generator_id"],
                "entity_id_initial": first["entity_id"],
                "entity_name_initial": first["entity_name"],
                "plant_name_initial": first["plant_name"],
                "state_code": first["state_code"],
                "county_initial": first["county"],
                "balancing_authority_initial": first["balancing_authority_code"],
                "sector_group": first["sector_group"],
                "technology_group": first["technology_group"],
                "technology_name_initial": first["technology_name"],
                "energy_source_code": first["energy_source_code"],
                "prime_mover_code": first["prime_mover_code"],
                "capacity_mw_initial": first["capacity_mw"],
                "first_report_month": first["report_month"],
                "first_report_ordinal": int(first["report_ordinal"]),
                "last_planned_report_month": last["report_month"],
                "last_planned_report_ordinal": int(last["report_ordinal"]),
                "planned_observation_count": int(len(history)),
                "initial_state": first["current_state"],
                "initial_planned_operation_month": first[
                    "planned_operation_month"
                ],
                "initial_planned_operation_ordinal": initial_schedule,
                "terminal_category": terminal_category,
                "event_type": event_type,
                "event_month": (
                    str(pd.Period(ordinal=int(event_ordinal), freq="M"))
                    if pd.notna(event_ordinal)
                    else None
                ),
                "event_ordinal": event_ordinal,
                "operating_month": terminal_record.get("operating_month"),
                "operating_ordinal": operation_ordinal,
                "first_canceled_month": period_string(cancel_period),
                "first_canceled_ordinal": cancel_ordinal,
                "operation_before_or_at_entry": operation_before_or_at_entry,
                "is_clean_primary": first["technology_group"]
                in config.clean_groups,
                "passes_burn_in": first["report_ordinal"]
                >= first_archive_period.ordinal + config.burn_in_months,
                "has_24_month_potential_followup": first["report_ordinal"]
                <= final_archive_period.ordinal - 24,
                "analysis_split": split,
                "development_cluster_id": cluster_key,
                "entity_id_count": int(history["entity_id"].nunique(dropna=True)),
                "technology_group_count": int(
                    history["technology_group"].nunique(dropna=True)
                ),
                "state_code_count": int(history["state_code"].nunique(dropna=True)),
            }
        )
    projects = pd.DataFrame.from_records(project_rows)
    cluster_sizes = projects.groupby("development_cluster_id")["unit_id"].transform(
        "count"
    )
    projects["development_cluster_size"] = cluster_sizes.astype(int)
    projects["cluster_equalizing_weight"] = 1.0 / cluster_sizes
    projects["eligible_primary_cohort"] = (
        projects["is_clean_primary"]
        & projects["passes_burn_in"]
        & ~projects["operation_before_or_at_entry"]
    )
    audit = {
        "project_rows": int(len(projects)),
        "primary_clean_cohort_units": int(projects["eligible_primary_cohort"].sum()),
        "primary_clean_development_clusters": int(
            projects.loc[
                projects["eligible_primary_cohort"], "development_cluster_id"
            ].nunique()
        ),
        "terminal_categories_primary": projects.loc[
            projects["eligible_primary_cohort"], "terminal_category"
        ]
        .value_counts()
        .to_dict(),
        "event_types_primary": projects.loc[
            projects["eligible_primary_cohort"], "event_type"
        ]
        .value_counts()
        .to_dict(),
        "operation_before_or_at_entry": int(
            projects["operation_before_or_at_entry"].sum()
        ),
        "unresolved_terminal_identity": int(
            (projects["terminal_category"] == "unresolved").sum()
        ),
        "units_with_entity_changes": int((projects["entity_id_count"] > 1).sum()),
        "units_with_technology_group_changes": int(
            (projects["technology_group_count"] > 1).sum()
        ),
        "units_with_state_changes": int((projects["state_code_count"] > 1).sum()),
        "split_counts_primary": projects.loc[
            projects["eligible_primary_cohort"], "analysis_split"
        ]
        .value_counts()
        .to_dict(),
    }
    return projects, audit


def _assign_split(first_period: pd.Period, config: AnalysisConfig) -> str:
    if first_period <= pd.Period(config.train_entry_end, freq="M"):
        return "train"
    if pd.Period(config.validation_entry_start, freq="M") <= first_period <= pd.Period(
        config.validation_entry_end, freq="M"
    ):
        return "validation"
    if pd.Period(config.test_entry_start, freq="M") <= first_period <= pd.Period(
        config.test_entry_end, freq="M"
    ):
        return "test"
    return "outside_test_window"


def enrich_observations(
    observations: pd.DataFrame, projects: pd.DataFrame
) -> pd.DataFrame:
    project_columns = [
        "unit_id",
        "first_report_ordinal",
        "initial_planned_operation_ordinal",
        "event_type",
        "event_ordinal",
        "terminal_category",
        "eligible_primary_cohort",
        "analysis_split",
        "development_cluster_id",
        "development_cluster_size",
        "cluster_equalizing_weight",
    ]
    panel = observations.merge(
        projects[project_columns], on="unit_id", how="left", validate="many_to_one"
    )
    panel.sort_values(["unit_id", "report_ordinal"], inplace=True)
    panel["months_since_first_report"] = (
        panel["report_ordinal"] - panel["first_report_ordinal"]
    ).astype(int)
    panel["initial_reported_lead_months"] = (
        panel["initial_planned_operation_ordinal"] - panel["first_report_ordinal"]
    )
    panel["months_to_reported_operation"] = (
        panel["planned_operation_ordinal"] - panel["report_ordinal"]
    )
    panel["log1p_capacity_mw"] = np.log1p(panel["capacity_mw"].clip(lower=0))

    enriched_groups: list[pd.DataFrame] = []
    for _, history in panel.groupby("unit_id", sort=False):
        history = history.sort_values("report_ordinal").copy()
        schedule = history["planned_operation_ordinal"].to_numpy(dtype=float)
        state = history["current_state"].astype(str).to_numpy()
        revision_count: list[int] = []
        absolute_movement: list[float] = []
        cumulative_slip: list[float] = []
        reversal_count: list[int] = []
        months_in_state: list[int] = []
        revisions = 0
        movement = 0.0
        reversals = 0
        spell_start = 0
        initial_schedule = schedule[0]
        previous_valid_schedule = schedule[0] if np.isfinite(schedule[0]) else np.nan
        for index in range(len(history)):
            current_schedule = schedule[index]
            if index > 0 and np.isfinite(current_schedule) and np.isfinite(
                previous_valid_schedule
            ):
                if current_schedule != previous_valid_schedule:
                    revisions += 1
                    movement += abs(current_schedule - previous_valid_schedule)
                previous_valid_schedule = current_schedule
            elif np.isfinite(current_schedule):
                previous_valid_schedule = current_schedule
            if index > 0 and state[index] != state[index - 1]:
                spell_start = index
                prior_rank = STATE_RANK.get(state[index - 1])
                current_rank = STATE_RANK.get(state[index])
                if (
                    prior_rank is not None
                    and current_rank is not None
                    and current_rank < prior_rank
                ):
                    reversals += 1
            revision_count.append(revisions)
            absolute_movement.append(movement)
            cumulative_slip.append(
                current_schedule - initial_schedule
                if np.isfinite(current_schedule) and np.isfinite(initial_schedule)
                else np.nan
            )
            reversal_count.append(reversals)
            months_in_state.append(index - spell_start + 1)
        history["schedule_revision_count"] = revision_count
        history["absolute_schedule_movement_months"] = absolute_movement
        history["cumulative_schedule_slip_months"] = cumulative_slip
        history["state_reversal_count"] = reversal_count
        history["months_in_current_state"] = months_in_state
        enriched_groups.append(history)
    panel = pd.concat(enriched_groups, ignore_index=True)

    panel["concurrent_entity_pipeline_units"] = panel.groupby(
        ["report_ordinal", "entity_id"], dropna=False
    )["unit_id"].transform("nunique")
    completion_ordinals: dict[str, list[int]] = defaultdict(list)
    completed = projects.loc[
        (projects["event_type"] == "O")
        & projects["event_ordinal"].notna()
        & projects["entity_id_initial"].notna(),
        ["entity_id_initial", "event_ordinal"],
    ]
    for entity_id, values in completed.groupby("entity_id_initial"):
        completion_ordinals[str(entity_id)] = sorted(values["event_ordinal"].astype(int))
    panel["prior_entity_completions"] = [
        bisect.bisect_left(
            completion_ordinals.get(str(entity_id), []), int(report_ordinal)
        )
        if entity_id is not None and pd.notna(entity_id)
        else 0
        for entity_id, report_ordinal in zip(
            panel["entity_id"], panel["report_ordinal"], strict=False
        )
    ]
    return panel.sort_values(["unit_id", "report_ordinal"]).reset_index(drop=True)


def build_transition_table(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    transition_rows: list[dict[str, Any]] = []
    internal_gaps = 0
    event_gaps: Counter[int] = Counter()
    for _, history in panel.groupby("unit_id", sort=False):
        history = history.sort_values("report_ordinal")
        records = history.to_dict("records")
        for current, following in zip(records, records[1:], strict=False):
            gap = int(following["report_ordinal"] - current["report_ordinal"])
            if gap != 1:
                internal_gaps += 1
                continue
            row = dict(current)
            row["next_state"] = following["current_state"]
            row["transition_gap_months"] = 1
            row["terminal_transition"] = False
            transition_rows.append(row)
        last = records[-1]
        if last["event_type"] in {"O", "C"} and pd.notna(last["event_ordinal"]):
            gap = int(last["event_ordinal"] - last["report_ordinal"])
            event_gaps[gap] += 1
            if gap in {0, 1}:
                row = dict(last)
                row["next_state"] = last["event_type"]
                row["transition_gap_months"] = gap
                row["terminal_transition"] = True
                transition_rows.append(row)
    transitions = pd.DataFrame.from_records(transition_rows)
    audit = {
        "transition_rows": int(len(transitions)),
        "internal_nonconsecutive_observation_gaps": int(internal_gaps),
        "terminal_event_gap_distribution": {
            str(key): int(value) for key, value in sorted(event_gaps.items())
        },
        "next_state_counts": transitions["next_state"].value_counts().to_dict(),
    }
    return transitions, audit


def build_landmarks(
    panel: pd.DataFrame, projects: pd.DataFrame, final_archive_period: pd.Period
) -> pd.DataFrame:
    project_map = projects.set_index("unit_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for unit_id, history in panel.groupby("unit_id", sort=False):
        project = project_map[unit_id]
        if not project["eligible_primary_cohort"]:
            continue
        by_age = history.set_index("months_since_first_report", drop=False)
        for landmark_age in (0, 6):
            if landmark_age not in by_age.index:
                continue
            landmark = by_age.loc[landmark_age]
            if isinstance(landmark, pd.DataFrame):
                landmark = landmark.iloc[0]
            origin = int(landmark["report_ordinal"])
            event_ordinal = project["event_ordinal"]
            event_type = project["event_type"]
            months_to_event = (
                int(event_ordinal - origin) if pd.notna(event_ordinal) else None
            )
            if months_to_event is not None and months_to_event <= 0:
                continue
            observable_12 = final_archive_period.ordinal - origin >= 12 or (
                months_to_event is not None and months_to_event <= 12
            )
            observable_24 = final_archive_period.ordinal - origin >= 24 or (
                months_to_event is not None and months_to_event <= 24
            )
            row = landmark.to_dict()
            row.update(
                {
                    "landmark_age_months": landmark_age,
                    "months_to_event": months_to_event,
                    "observable_12_month_outcome": bool(observable_12),
                    "observable_24_month_outcome": bool(observable_24),
                    "operation_within_12_months": int(
                        event_type == "O"
                        and months_to_event is not None
                        and months_to_event <= 12
                    ),
                    "cancellation_within_24_months": int(
                        event_type == "C"
                        and months_to_event is not None
                        and months_to_event <= 24
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _continuity_audit(monthly_files: list[tuple[pd.Period, Path]]) -> dict[str, Any]:
    periods = [period for period, _ in monthly_files]
    expected = list(pd.period_range(periods[0], periods[-1], freq="M"))
    missing = sorted(set(expected) - set(periods))
    duplicated = [
        str(period)
        for period, count in Counter(periods).items()
        if count > 1
    ]
    return {
        "first_month": str(periods[0]),
        "last_month": str(periods[-1]),
        "monthly_file_count": len(periods),
        "expected_month_count": len(expected),
        "missing_months": [str(period) for period in missing],
        "duplicated_months": duplicated,
        "continuous_monthly_archive": not missing and not duplicated,
    }


def build_longitudinal_data(
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    monthly_files = discover_monthly_files(config.monthly_dir)
    if not monthly_files:
        raise FileNotFoundError(f"No EIA-860M files found in {config.monthly_dir}")
    continuity = _continuity_audit(monthly_files)
    if not continuity["continuous_monthly_archive"]:
        raise ValueError(f"Monthly archive is not continuous: {continuity}")
    observations, observation_audit = read_planned_observations(monthly_files)
    first_planned_ordinal = (
        observations.groupby("unit_id")["report_ordinal"].min().astype(int).to_dict()
    )
    first_canceled = read_first_cancellation_months(
        monthly_files, first_planned_ordinal
    )
    final_period, final_path = monthly_files[-1]
    terminal, terminal_audit = read_terminal_inventory(final_period, final_path)
    projects, project_audit = build_project_table(
        observations,
        terminal,
        first_canceled,
        config,
        monthly_files[0][0],
        final_period,
    )
    panel = enrich_observations(observations, projects)
    transitions, transition_audit = build_transition_table(panel)
    landmarks = build_landmarks(panel, projects, final_period)

    primary_panel = panel.loc[panel["eligible_primary_cohort"]]
    missingness = {
        column: float(primary_panel[column].isna().mean())
        for column in (
            "capacity_mw",
            "planned_operation_ordinal",
            "balancing_authority_code",
            "latitude",
            "longitude",
            "entity_id",
        )
    }
    audit = {
        "continuity": continuity,
        "observations": observation_audit,
        "terminal_inventory": terminal_audit,
        "projects": project_audit,
        "transitions": transition_audit,
        "cancellation_dates_found": len(first_canceled),
        "primary_panel_rows": int(len(primary_panel)),
        "primary_landmark_rows": int(len(landmarks)),
        "primary_missingness_fraction": missingness,
    }
    return projects, panel, transitions, landmarks, audit


def save_longitudinal_data(
    config: AnalysisConfig,
    projects: pd.DataFrame,
    panel: pd.DataFrame,
    transitions: pd.DataFrame,
    landmarks: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    projects.to_parquet(config.processed_dir / "projects.parquet", index=False)
    panel.to_parquet(config.processed_dir / "unit_month_panel.parquet", index=False)
    transitions.to_parquet(config.processed_dir / "monthly_transitions.parquet", index=False)
    landmarks.to_parquet(config.processed_dir / "landmarks.parquet", index=False)
    with (config.results_dir / "data_quality_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)


def load_longitudinal_data(
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    projects = pd.read_parquet(config.processed_dir / "projects.parquet")
    panel = pd.read_parquet(config.processed_dir / "unit_month_panel.parquet")
    transitions = pd.read_parquet(config.processed_dir / "monthly_transitions.parquet")
    landmarks = pd.read_parquet(config.processed_dir / "landmarks.parquet")
    with (config.results_dir / "data_quality_audit.json").open(encoding="utf-8") as handle:
        audit = json.load(handle)
    return projects, panel, transitions, landmarks, audit
