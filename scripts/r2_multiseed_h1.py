"""R2: multi-seed replication at the one-hour horizon.

Why this exists
---------------
The revised manuscript reports a one-hour test RMSE of 17.20 kWh for PA-STT
against 17.39 kWh for the anchored LightGBM baseline - a margin of 0.19 kWh -
while its own Limitations section states that the results rest on a single
training seed and that the closest evidence on run-to-run variability is a pair
of independent refits differing by about 0.18 kWh. A margin that sits inside the
run-to-run spread cannot support a claim of architectural superiority.

This script settles the question by replication rather than by argument. It
retrains PA-STT under several seeds on the archived 95-building windows, using
the project's own `train_anchored_transformer`, the archived model configuration
and the archived target scaler, so the only thing that changes between runs is
the seed.

It then tests the anchored-boosting comparator for determinism, and reports:

  * mean, SD and a two-sided 95 percent confidence interval for PA-STT test RMSE
    and MAE across seeds;
  * a one-sample t-test of the PA-STT seed distribution against the anchored
    LightGBM value;
  * the seed-to-seed spread, which is the honest noise floor for every
    single-seed comparison in the manuscript.

The script does not decide what the manuscript should claim. It produces the
numbers; the claim follows from them.

Outputs
-------
  r2_multiseed_h1.json        per-seed metrics and the statistical summary
  r2_multiseed_h1_manifest.json  inputs, hashes, environment, config
"""
from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\pmo3.EDG.000\Desktop\EB Revision")
CODE = ROOT / "Final Submittions" / "revision_package" / "reproducibility" / "code"
RUN = ROOT / "revision_results" / "bdg2_pa_stt_revision_main_20260730_200352"
HERE = Path(__file__).parent
OUT = HERE / "r2_multiseed_h1.json"
MANIFEST = HERE / "r2_multiseed_h1_manifest.json"

sys.path.insert(0, str(CODE / "src"))
import torch  # noqa: E402
import lightgbm  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from scipy import stats  # noqa: E402

from energy_forecasting.features import (  # noqa: E402
    TargetScalerState, add_time_features, chronological_split, fit_target_scaler,
)
from energy_forecasting.metrics import regression_metrics  # noqa: E402
from energy_forecasting.train import train_anchored_transformer  # noqa: E402

# Archived model configuration (00_config_used.yaml, model block), verbatim.
MODEL_CFG = {
    "name": "anchored_temporal_transformer",
    "reporting_name": "pa_stt",
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.1,
    "dim_feedforward": 128,
    "residual_feature_index": 0,
    "ramp_weight": 0.0,
    "ramp_base_tolerance": 0.25,
    "ramp_temperature_coefficient": 3.0,
    "ramp_temperature_feature_index": 1,
    "max_epochs": 10,
    "batch_size": 2048,
    "evaluation_batch_size": 1024,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "patience": 3,
    "mixed_precision": True,
}

# Seed 42 is the archived main run; it is included from its stored metrics and
# the remaining seeds are trained here.
ARCHIVED_SEED = 42
NEW_SEEDS = [52, 62, 72, 82, 92]
ANCHORED_LIGHTGBM_RMSE = 17.39  # manuscript Table 8, reproduced by r33


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_arrays() -> dict[str, dict[str, np.ndarray]]:
    payload = np.load(RUN / "02_window_arrays.npz", allow_pickle=True)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        prefix = split + "_"
        arrays[split] = {k[len(prefix):]: payload[k] for k in payload.files
                         if k.startswith(prefix)}
    return arrays


def summary_feats(X: np.ndarray) -> np.ndarray:
    """32 window statistics, float64 accumulation (archived behaviour)."""
    Xd = X.astype(np.float64)
    return np.concatenate(
        [Xd[:, -1, :], Xd.mean(axis=1), Xd.std(axis=1), Xd[:, -1, :] - Xd[:, 0, :]],
        axis=1).astype(np.float32)


def main() -> int:
    t0 = time.time()
    print("loading archived windows ...", flush=True)
    arrays = load_arrays()
    for split, values in arrays.items():
        print(f"   {split}: X{values['X'].shape}", flush=True)

    print("rebuilding the archived target scaler ...", flush=True)
    clean = pd.read_parquet(RUN / "01_cleaned_long.parquet")
    split_frame = chronological_split(
        add_time_features(clean), {"train": 0.70, "validation": 0.15, "test": 0.15})
    target_scaler = fit_target_scaler(split_frame, "meter_reading")
    del clean, split_frame
    gc.collect()
    print(f"   target scaler mean={target_scaler.mean:.4f} std={target_scaler.std:.4f}",
          flush=True)

    # ---------------------------------------------------------------- seed 42
    archived = pd.read_csv(RUN / "05_pa_stt_metrics.csv")
    archived_test = archived[archived["split"] == "test"].iloc[0]
    seed_rows = [{
        "seed": ARCHIVED_SEED,
        "source": "archived main run",
        "rmse": float(archived_test["rmse"]),
        "mae": float(archived_test["mae"]),
        "cvrmse_pct": float(archived_test["cvrmse_pct"]),
        "r2": float(archived_test["r2"]),
        "training_seconds": None,
    }]
    print(f"\nseed {ARCHIVED_SEED} (archived): RMSE {seed_rows[0]['rmse']:.4f} "
          f"MAE {seed_rows[0]['mae']:.4f}", flush=True)

    # ------------------------------------------------------------ new seeds
    for seed in NEW_SEEDS:
        started = time.time()
        print(f"\n=== training seed {seed} ===", flush=True)
        result = train_anchored_transformer(
            arrays=arrays,
            model_cfg=dict(MODEL_CFG),
            target_scaler=target_scaler,
            seed=seed,
            device_name="cuda",
        )
        metrics = result.metric_frame
        test = metrics[metrics["split"] == "test"].iloc[0]
        row = {
            "seed": seed,
            "source": "replication",
            "rmse": float(test["rmse"]),
            "mae": float(test["mae"]),
            "cvrmse_pct": float(test["cvrmse_pct"]),
            "r2": float(test["r2"]),
            "training_seconds": float(time.time() - started),
        }
        seed_rows.append(row)
        print(f"   seed {seed}: RMSE {row['rmse']:.4f}  MAE {row['mae']:.4f}  "
              f"CV {row['cvrmse_pct']:.4f}%  [{row['training_seconds']:.0f}s]", flush=True)
        # Persist after every seed so a long run stays recoverable.
        OUT.write_text(json.dumps({"seeds": seed_rows}, indent=1), encoding="utf-8")
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------- anchored LightGBM determinism
    print("\n=== anchored LightGBM determinism check ===", flush=True)
    X_train = summary_feats(arrays["train"]["X"])
    y_train = arrays["train"]["y"] - arrays["train"]["last_target"]
    X_test = summary_feats(arrays["test"]["X"])
    lgbm_rows = []
    for seed in (42, 52, 62):
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                              random_state=seed, verbosity=-1, n_jobs=6)
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_test), dtype=np.float64) + arrays["test"]["last_target"]
        m = regression_metrics(arrays["test"]["y"], pred)
        lgbm_rows.append({"seed": seed, "rmse": float(m["rmse"]), "mae": float(m["mae"]),
                          "cvrmse_pct": float(m["cvrmse_pct"])})
        print(f"   lgbm seed {seed}: RMSE {m['rmse']:.4f}  MAE {m['mae']:.4f}", flush=True)
    del X_train, X_test
    gc.collect()
    lgbm_rmse = [r["rmse"] for r in lgbm_rows]
    lgbm_deterministic = bool(np.ptp(lgbm_rmse) < 1e-9)
    lgbm_reference = float(np.mean(lgbm_rmse))

    # ------------------------------------------------------------ statistics
    rmse = np.array([r["rmse"] for r in seed_rows], dtype=float)
    mae = np.array([r["mae"] for r in seed_rows], dtype=float)
    n = len(rmse)
    mean, sd = float(rmse.mean()), float(rmse.std(ddof=1))
    sem = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    ci = (mean - tcrit * sem, mean + tcrit * sem)
    # One-sample t-test: is the PA-STT seed mean below the anchored LightGBM value?
    tt = stats.ttest_1samp(rmse, lgbm_reference, alternative="less")

    stats_block = {
        "n_seeds": n,
        "seeds": [int(r["seed"]) for r in seed_rows],
        "rmse_mean": mean,
        "rmse_sd": sd,
        "rmse_min": float(rmse.min()),
        "rmse_max": float(rmse.max()),
        "rmse_range": float(rmse.max() - rmse.min()),
        "rmse_ci95": [float(ci[0]), float(ci[1])],
        "mae_mean": float(mae.mean()),
        "mae_sd": float(mae.std(ddof=1)),
        "anchored_lightgbm_rmse": lgbm_reference,
        "anchored_lightgbm_deterministic_across_seeds": lgbm_deterministic,
        "margin_point_estimate": lgbm_reference - float(archived_test["rmse"]),
        "margin_vs_seed_mean": lgbm_reference - mean,
        "t_statistic": float(tt.statistic),
        "p_one_sided_less": float(tt.pvalue),
        "significant_at_0_05": bool(tt.pvalue < 0.05),
        "lightgbm_inside_seed_range": bool(rmse.min() <= lgbm_reference <= rmse.max()),
    }

    print("\n" + "=" * 74)
    print("MULTI-SEED RESULT (one-hour horizon, test split)")
    print("=" * 74)
    for r in seed_rows:
        print(f"  seed {r['seed']:>3}  RMSE {r['rmse']:8.4f}  MAE {r['mae']:7.4f}  ({r['source']})")
    print(f"\n  PA-STT  n={n}  mean {mean:.4f}  SD {sd:.4f}  range {stats_block['rmse_range']:.4f}")
    print(f"  95% CI  [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  anchored LightGBM {lgbm_reference:.4f} "
          f"(deterministic across seeds: {lgbm_deterministic})")
    print(f"  margin vs seed mean {stats_block['margin_vs_seed_mean']:+.4f} kWh")
    print(f"  one-sided t-test p = {tt.pvalue:.4f} -> "
          f"{'SIGNIFICANT' if stats_block['significant_at_0_05'] else 'NOT SIGNIFICANT'} at 0.05")
    print(f"  LightGBM value inside the PA-STT seed range: "
          f"{stats_block['lightgbm_inside_seed_range']}")

    payload = {"seeds": seed_rows, "anchored_lightgbm": lgbm_rows, "statistics": stats_block}
    OUT.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    MANIFEST.write_text(json.dumps({
        "purpose": "Multi-seed replication of PA-STT at h=1 to test whether the 0.19 kWh "
                   "margin over anchored LightGBM exceeds run-to-run variability.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_run": str(RUN),
        "window_arrays_sha256": sha256(RUN / "02_window_arrays.npz"),
        "model_config": MODEL_CFG,
        "archived_seed": ARCHIVED_SEED,
        "replication_seeds": NEW_SEEDS,
        "training_function": "energy_forecasting.train.train_anchored_transformer",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "statistics": stats_block,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\ntotal {time.time() - t0:.0f}s -> {OUT.name}, {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
