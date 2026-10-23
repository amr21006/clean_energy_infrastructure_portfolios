from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .config import AnalysisConfig
from .descriptive import generate_descriptive_outputs
from .modeling import train_and_evaluate_models
from .panel import (
    build_longitudinal_data,
    load_longitudinal_data,
    save_longitudinal_data,
)
from .reporting import (
    create_threats_to_validity,
    create_run_manifest,
    frontier_comparator_audit,
    generate_results_report,
    table_index,
)
from .validation import validate_analysis


def _configure_logging(config: AnalysisConfig) -> logging.Logger:
    logger = logging.getLogger("eia_delivery_twin")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(
        config.results_dir / "pipeline.log", encoding="utf-8", mode="w"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def run_pipeline(
    raw_dir: Path,
    output_dir: Path,
    seed: int = 20260713,
    bootstrap_reps: int = 500,
    rebuild_panel: bool = False,
) -> None:
    config = AnalysisConfig(
        raw_dir=raw_dir.resolve(),
        output_dir=output_dir.resolve(),
        seed=seed,
        bootstrap_reps=bootstrap_reps,
    )
    config.ensure_directories()
    logger = _configure_logging(config)
    start = time.time()
    create_run_manifest(config)

    cache_files = [
        config.processed_dir / "projects.parquet",
        config.processed_dir / "unit_month_panel.parquet",
        config.processed_dir / "monthly_transitions.parquet",
        config.processed_dir / "landmarks.parquet",
        config.results_dir / "data_quality_audit.json",
    ]
    if rebuild_panel or not all(path.exists() for path in cache_files):
        logger.info("Building longitudinal panel from raw EIA workbooks")
        projects, panel, transitions, landmarks, data_audit = build_longitudinal_data(config)
        save_longitudinal_data(
            config, projects, panel, transitions, landmarks, data_audit
        )
    else:
        logger.info("Loading verified longitudinal panel cache")
        projects, panel, transitions, landmarks, data_audit = load_longitudinal_data(
            config
        )
    logger.info(
        "Panel ready: %s projects, %s unit-month rows, %s transitions, %s landmarks",
        f"{len(projects):,}",
        f"{len(panel):,}",
        f"{len(transitions):,}",
        f"{len(landmarks):,}",
    )

    logger.info("Generating descriptive and multistate outputs")
    descriptive = generate_descriptive_outputs(
        config, projects, panel, transitions, landmarks
    )

    logger.info("Training and validating predictive models")
    models = train_and_evaluate_models(config, transitions, landmarks)

    logger.info("Running completion and consistency audit")
    validation = validate_analysis(
        config, projects, panel, transitions, landmarks, data_audit
    )
    create_threats_to_validity(config)
    frontier_comparator_audit(config)
    table_index(config)
    generate_results_report(
        config,
        projects,
        panel,
        transitions,
        landmarks,
        descriptive,
        models,
        validation,
    )
    table_index(config)
    elapsed = time.time() - start
    summary = {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "projects": len(projects),
        "panel_rows": len(panel),
        "transition_rows": len(transitions),
        "landmark_rows": len(landmarks),
        "validation_checks": len(validation),
        "validation_checks_passed": int(validation["passed"].sum()),
    }
    with (config.results_dir / "pipeline_completion.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Pipeline completed in %.1f seconds", elapsed)
