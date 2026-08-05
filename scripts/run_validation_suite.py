from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_forecasting.config import load_config, save_config
from energy_forecasting.pipeline import run_pipeline
from energy_forecasting.validation import (
    ablation_summary,
    aggregate_metrics,
    paired_error_tests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reviewer-facing validation suite.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--suite-name",
        default=None,
        help="Optional output suite folder name. A timestamp is appended automatically.",
    )
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Run only the main model across seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_name = args.suite_name or f"{config['experiment']['name']}_validation"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config["experiment"].get("output_dir", "results"))
    suite_dir = output_root / f"{base_name}_{timestamp}"
    runs_dir = suite_dir / "runs"
    suite_dir.mkdir(parents=True, exist_ok=False)
    runs_dir.mkdir(parents=True, exist_ok=True)

    validation_cfg = config.get("validation", {})
    seeds = [int(seed) for seed in validation_cfg.get("seeds", [config["experiment"].get("seed", 42)])]
    variants: list[dict[str, Any]] = [{"name": "main", "model_overrides": {}}]
    if not args.skip_ablations:
        variants.extend(validation_cfg.get("ablations", []))

    save_config(config, suite_dir / "suite_base_config.yaml")

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []

    for variant in variants:
        variant_name = str(variant["name"])
        for seed in seeds:
            run_cfg = copy.deepcopy(config)
            run_cfg["experiment"]["seed"] = seed
            run_cfg["experiment"]["output_dir"] = str(runs_dir)
            run_cfg["experiment"]["name"] = f"{config['experiment']['name']}_{variant_name}_seed{seed}"
            _deep_update(run_cfg["model"], variant.get("model_overrides", {}))
            run_dir = run_pipeline(run_cfg)

            leaderboard = pd.read_csv(run_dir / "06_leaderboard.csv")
            leaderboard.insert(0, "run_dir", str(run_dir))
            leaderboard.insert(0, "seed", seed)
            leaderboard.insert(0, "variant", variant_name)
            metric_frames.append(leaderboard)

            predictions = pd.read_csv(run_dir / "07_figure_data_predictions.csv")
            predictions.insert(0, "run_dir", str(run_dir))
            predictions.insert(0, "seed", seed)
            predictions.insert(0, "variant", variant_name)
            prediction_frames.append(predictions)

            with (run_dir / "08_run_manifest.json").open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest.update({"variant": variant_name, "seed": seed, "run_dir": str(run_dir)})
            manifests.append(manifest)

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)

    all_metrics.to_csv(suite_dir / "suite_per_run_metrics.csv", index=False)
    all_predictions.to_parquet(suite_dir / "suite_test_predictions.parquet", index=False)
    aggregate_metrics(all_metrics).to_csv(suite_dir / "suite_aggregate_metrics.csv", index=False)
    paired_error_tests(all_predictions[all_predictions["variant"] == "main"]).to_csv(
        suite_dir / "suite_paired_error_tests.csv",
        index=False,
    )
    ablation_summary(all_metrics).to_csv(suite_dir / "suite_ablation_summary.csv", index=False)

    suite_manifest = {
        "suite_dir": str(suite_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_config": args.config,
        "device_requested": config["experiment"].get("device"),
        "seeds": seeds,
        "variants": [str(variant["name"]) for variant in variants],
        "n_runs": len(manifests),
        "run_manifests": manifests,
        "saved_files": sorted(p.name for p in suite_dir.iterdir() if p.is_file()),
    }
    with (suite_dir / "suite_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(suite_manifest, f, indent=2)

    print(f"Validation suite complete. Results saved to: {suite_dir}")


def _deep_update(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


if __name__ == "__main__":
    main()
