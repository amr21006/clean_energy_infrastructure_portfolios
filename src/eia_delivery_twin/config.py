from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    raw_dir: Path
    output_dir: Path
    seed: int = 20260713
    bootstrap_reps: int = 500
    burn_in_months: int = 12
    primary_horizon_months: int = 12
    cancellation_horizon_months: int = 24
    train_entry_end: str = "2021-12"
    validation_entry_start: str = "2022-01"
    validation_entry_end: str = "2022-12"
    test_entry_start: str = "2023-01"
    test_entry_end: str = "2024-05"
    random_search_iterations: int = 2
    status_order: tuple[str, ...] = (
        "P",
        "L",
        "T",
        "U",
        "V",
        "TS",
        "O",
        "C",
        "OTHER",
    )
    clean_groups: tuple[str, ...] = (
        "solar",
        "wind",
        "storage",
        "hydro",
        "other_renewable",
    )
    stable_categorical_features: tuple[str, ...] = (
        "state_code",
        "technology_group",
        "sector_group",
        "energy_source_code",
        "prime_mover_code",
        "current_state",
    )
    numeric_features: tuple[str, ...] = (
        "log1p_capacity_mw",
        "months_since_first_report",
        "months_in_current_state",
        "months_to_reported_operation",
        "schedule_revision_count",
        "cumulative_schedule_slip_months",
        "absolute_schedule_movement_months",
        "state_reversal_count",
        "concurrent_entity_pipeline_units",
        "prior_entity_completions",
        "initial_reported_lead_months",
    )

    @property
    def monthly_dir(self) -> Path:
        return self.raw_dir / "eia860m"

    @property
    def annual_dir(self) -> Path:
        return self.raw_dir / "eia860_annual"

    @property
    def processed_dir(self) -> Path:
        return self.output_dir / "processed"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def predictions_dir(self) -> Path:
        return self.output_dir / "predictions"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    def ensure_directories(self) -> None:
        for path in (
            self.output_dir,
            self.processed_dir,
            self.tables_dir,
            self.models_dir,
            self.predictions_dir,
            self.results_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
