"""Anchored boosting across all forecast horizons.

Table 9 currently compares PA-STT against PLAIN LightGBM at 3/6/12/24 h. Now that
anchoring is known to transform boosting at one hour, that comparison is incomplete:
a reviewer will ask whether PA-STT's advantage survives against an ANCHORED booster.

For each horizon this refits LightGBM on the persistence-differenced target using the
project's own windowing (build_chronological_arrays equivalent), and refits the plain
variant as a CONTROL that must reproduce the archived Table 9 values.

Summary features are accumulated in float64, which is what reproduces the archived
runs (single precision shifts test RMSE by ~0.34 kWh).
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\pmo3.EDG.000\Desktop\EB Revision")
CODE = ROOT / "Final Submittions" / "revision_package" / "reproducibility" / "code"
RUN = ROOT / "revision_results" / "bdg2_pa_stt_revision_main_20260730_200352"
OUT = Path(__file__).parent / "r33_multihorizon.json"

sys.path.insert(0, str(CODE / "src"))
from energy_forecasting.features import (  # noqa: E402
    add_time_features, chronological_split, fit_feature_scaler,
    fit_target_scaler, make_windows,
)
from energy_forecasting.metrics import regression_metrics  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402

FEATURE_COLUMNS = ["meter_reading", "air_temperature", "hour_sin", "hour_cos",
                   "dow_sin", "dow_cos", "month_sin", "month_cos"]
SEED = 42

# archived Table 9 test RMSE, used as controls
ARCHIVED = {
    3: {"summary": 33.55, "full_lags": 35.87, "pa_stt": 29.90},
    6: {"summary": 42.86, "full_lags": 41.64, "pa_stt": 38.42},
    12: {"summary": 50.31, "full_lags": 48.86, "pa_stt": 48.76},
    24: {"summary": 49.94, "full_lags": 49.22, "pa_stt": 48.39},
}


def summary_feats(X):
    Xd = X.astype(np.float64)
    return np.concatenate(
        [Xd[:, -1, :], Xd.mean(axis=1), Xd.std(axis=1), Xd[:, -1, :] - Xd[:, 0, :]],
        axis=1).astype(np.float32)


def full_feats(X):
    return X.reshape(X.shape[0], -1)


def fit_eval(Xtr, ytr, Xte, yte, anchor_tr, anchor_te, anchored):
    m = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                      random_state=SEED, verbosity=-1)
    m.fit(Xtr, ytr - anchor_tr if anchored else ytr)
    pred = m.predict(Xte)
    if anchored:
        pred = pred + anchor_te
    return regression_metrics(yte, pred)


t0 = time.time()
print("loading cleaned data ...", flush=True)
clean = pd.read_parquet(RUN / "01_cleaned_long.parquet")
featured = add_time_features(clean)
split_frame = chronological_split(featured, {"train": 0.70, "validation": 0.15, "test": 0.15})
feature_scaler = fit_feature_scaler(split_frame, FEATURE_COLUMNS)
target_scaler = fit_target_scaler(split_frame, "meter_reading")
del clean, featured
gc.collect()
print(f"ready in {time.time()-t0:.0f}s", flush=True)

results = {}
for h in (3, 6, 12, 24):
    th = time.time()
    print(f"\n===== horizon {h} h : building windows ...", flush=True)
    A = make_windows(split_frame, feature_cols=FEATURE_COLUMNS, sequence_length=48,
                     horizon=h, feature_scaler=feature_scaler,
                     target_scaler=target_scaler, target_col="meter_reading",
                     require_contiguous_hourly=True)
    ytr, yte = A["train"]["y"], A["test"]["y"]
    atr, ate = A["train"]["last_target"], A["test"]["last_target"]
    print(f"   train {A['train']['X'].shape}  test {A['test']['X'].shape} "
          f"[{time.time()-th:.0f}s]", flush=True)
    row = {}
    for rep, flat in (("summary", summary_feats), ("full_lags", full_feats)):
        Xtr, Xte = flat(A["train"]["X"]), flat(A["test"]["X"])
        ctrl = fit_eval(Xtr, ytr, Xte, yte, atr, ate, anchored=False)
        anch = fit_eval(Xtr, ytr, Xte, yte, atr, ate, anchored=True)
        d = abs(ctrl["rmse"] - ARCHIVED[h][rep])
        print(f"   {rep:10s} control {ctrl['rmse']:7.2f} (archived {ARCHIVED[h][rep]:.2f}, "
              f"delta {d:.2f} -> {'MATCH' if d < 0.15 else 'CHECK'})   "
              f"anchored {anch['rmse']:7.2f}  MAE {anch['mae']:6.2f}", flush=True)
        row[rep] = {"control": ctrl, "anchored": anch}
        del Xtr, Xte
        gc.collect()
    row["pa_stt_archived_rmse"] = ARCHIVED[h]["pa_stt"]
    best_anch = min(row["summary"]["anchored"]["rmse"], row["full_lags"]["anchored"]["rmse"])
    row["pa_stt_margin_pct"] = 100 * (best_anch - ARCHIVED[h]["pa_stt"]) / best_anch
    print(f"   PA-STT {ARCHIVED[h]['pa_stt']:.2f} vs best anchored booster {best_anch:.2f}"
          f"  -> PA-STT better by {row['pa_stt_margin_pct']:+.1f}%  "
          f"[horizon took {time.time()-th:.0f}s]", flush=True)
    results[h] = row
    del A
    gc.collect()

OUT.write_text(json.dumps(results, indent=1, default=float))
print("\n===== SUMMARY: PA-STT vs best ANCHORED booster =====", flush=True)
print(f"  {'h':>3}  {'PA-STT':>8}  {'anch.summ':>10}  {'anch.full':>10}  {'PA-STT margin':>14}")
print(f"  {1:>3}  {17.20:>8.2f}  {17.39:>10.2f}  {17.72:>10.2f}  {100*(17.39-17.20)/17.39:>13.1f}%")
for h, r in results.items():
    print(f"  {h:>3}  {r['pa_stt_archived_rmse']:>8.2f}  "
          f"{r['summary']['anchored']['rmse']:>10.2f}  {r['full_lags']['anchored']['rmse']:>10.2f}  "
          f"{r['pa_stt_margin_pct']:>13.1f}%")
print(f"\ntotal {time.time()-t0:.0f}s -> {OUT}")
