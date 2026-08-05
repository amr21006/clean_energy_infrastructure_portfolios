"""Extended experiments for the manuscript: multi-seed stability and ablations.

Reuses the exact project training code (energy_forecasting.train.train_pg_sstt)
and the cached supervised window arrays from the main run, so results are faithful
reproductions of the original pipeline under changed seeds / model settings.
No metrics are fabricated: every number is produced by the same evaluation code.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from energy_forecasting.features import add_time_features, chronological_split, fit_target_scaler
from energy_forecasting.train import train_pg_sstt

RUN = Path("results/bdg2_pg_sstt_fast_final_20260704_130635")
OUT = Path("manuscript/derived")
OUT.mkdir(parents=True, exist_ok=True)

# Base model config, identical to 00_config_used.yaml.
BASE_CFG = dict(
    d_model=64,
    n_heads=4,
    n_layers=2,
    dropout=0.1,
    dim_feedforward=128,
    residual_feature_index=0,
    physics_weight=0.02,
    max_epochs=10,
    batch_size=2048,
    learning_rate=0.001,
    weight_decay=0.0001,
    patience=3,
    mixed_precision=True,
)

DEVICE = "cuda"


def load_arrays() -> dict:
    d = np.load(RUN / "02_window_arrays.npz", allow_pickle=True)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in ["train", "validation", "test"]:
        prefix = split + "_"
        arrays[split] = {k[len(prefix):]: d[k] for k in d.files if k.startswith(prefix)}
        arrays[split]["X"] = arrays[split]["X"].astype(np.float32)
        arrays[split]["y"] = arrays[split]["y"].astype(np.float32)
        arrays[split]["y_scaled"] = arrays[split]["y_scaled"].astype(np.float32)
        arrays[split]["last_target"] = arrays[split]["last_target"].astype(np.float32)
    return arrays


def build_target_scaler():
    raw = pd.read_parquet(RUN / "01_cleaned_long.parquet")
    featured = add_time_features(raw)
    split_df = chronological_split(featured, {"train": 0.7, "validation": 0.15, "test": 0.15})
    return fit_target_scaler(split_df, "meter_reading")


def append_metrics(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    arrays = load_arrays()
    target_scaler = build_target_scaler()
    print(f"target_scaler mean={target_scaler.mean:.4f} std={target_scaler.std:.4f}", flush=True)

    variants = [
        # (variant_name, seed, cfg_overrides)
        ("main", 52, {}),
        ("main", 62, {}),
        ("no_physics", 42, {"physics_weight": 0.0}),
        ("no_residual_anchor", 42, {"residual_feature_index": None}),
    ]

    metric_rows: list[dict] = []
    diag_rows: list[dict] = []
    for variant, seed, overrides in variants:
        cfg = dict(BASE_CFG)
        cfg.update(overrides)
        tag = f"{variant}_seed{seed}"
        print(f"=== training {tag} ({overrides}) ===", flush=True)
        t0 = time.perf_counter()
        try:
            result = train_pg_sstt(
                arrays=arrays,
                model_cfg=cfg,
                target_scaler=target_scaler,
                seed=seed,
                device_name=DEVICE,
            )
        except Exception as exc:  # keep partial results usable
            print(f"!!! {tag} failed: {exc}", flush=True)
            continue
        mf = result.metric_frame.copy()
        mf["variant"] = variant
        mf["seed"] = seed
        for _, r in mf.iterrows():
            metric_rows.append(r.to_dict())
        result.history.assign(variant=variant, seed=seed).to_csv(
            OUT / f"ext_history_{tag}.csv", index=False
        )
        diag = dict(result.diagnostics)
        diag.update({"variant": variant, "seed": seed, "wall_seconds": time.perf_counter() - t0})
        diag_rows.append(diag)
        # flush incrementally so partial runs are recoverable
        append_metrics(metric_rows, OUT / "ext_metrics.csv")
        pd.DataFrame(diag_rows).to_csv(OUT / "ext_diagnostics.csv", index=False)
        print(f"=== done {tag} in {time.perf_counter()-t0:.1f}s ===", flush=True)

    print("ALL_EXT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
