"""R3.3 anchored boosting - corrected evidence run.

Why v2 exists
-------------
`r33_anchored_boosting.py` builds the 32 summary features with
`energy_forecasting.features.flatten_windows`, which calls `X.mean(axis=1)` and
`X.std(axis=1)` directly on the float32 window array. NumPy then accumulates the
48-step reduction in single precision, and the resulting features differ enough
from the archived pipeline that the unanchored control returns 25.09 kWh instead
of the published 24.80 kWh - a 0.29 kWh discrepancy that trips the script's own
"MISMATCH - DO NOT USE" guard.

`r33_multihorizon_anchored.py` already documents the cause in its docstring
("summary features are accumulated in float64, which is what reproduces the
archived runs; single precision shifts test RMSE by ~0.34 kWh") and defines a
float64 `summary_feats`. This script applies the same accumulation so that the
control reproduces the published value and the anchored figure is produced on
the same footing as everything else in Table 8.

The full-lag representation is a pure reshape with no reduction, so it is
unaffected by precision and is computed exactly as before.

Outputs
-------
  r33_results_v2.json      metrics for all four variants, both splits
  r33_manifest_v2.json     inputs, hashes, environment, control verdicts

Nothing is written to the manuscript by this script. Its purpose is to produce
the archived artefact that the Table 8 figures can be traced to.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\pmo3.EDG.000\Desktop\EB Revision")
CODE = ROOT / "Final Submittions" / "revision_package" / "reproducibility" / "code"
RUN = ROOT / "revision_results" / "bdg2_pa_stt_revision_main_20260730_200352"
HERE = Path(__file__).parent
OUT = HERE / "r33_results_v2.json"
MANIFEST = HERE / "r33_manifest_v2.json"

sys.path.insert(0, str(CODE / "src"))
from energy_forecasting.features import flatten_windows_full_lags  # noqa: E402
from energy_forecasting.metrics import regression_metrics  # noqa: E402
import lightgbm  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402

SEED = 42
# Published values these controls must reproduce (manuscript Table 4 / Table 8).
PUBLISHED = {"summary": 24.796693595725568, "full_lags": 29.51}
TOLERANCE = {"summary": 0.01, "full_lags": 0.02}


def summary_feats(X: np.ndarray) -> np.ndarray:
    """32 window statistics with float64 accumulation (archived behaviour)."""
    Xd = X.astype(np.float64)
    return np.concatenate(
        [Xd[:, -1, :], Xd.mean(axis=1), Xd.std(axis=1), Xd[:, -1, :] - Xd[:, 0, :]],
        axis=1,
    ).astype(np.float32)


def sha256(path: Path, limit: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(limit):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    t0 = time.time()
    print("loading archived window arrays ...", flush=True)
    payload = np.load(RUN / "02_window_arrays.npz")
    arrays = {
        split: {k[len(split) + 1:]: payload[k] for k in payload.files if k.startswith(split + "_")}
        for split in ("train", "validation", "test")
    }
    for split, values in arrays.items():
        print(f"   {split}: X{values['X'].shape}  dtype {values['X'].dtype}", flush=True)

    results: dict[str, dict] = {}
    verdicts: dict[str, dict] = {}

    def run(tag: str, flatten, anchored: bool) -> dict:
        started = time.time()
        print(f"\n=== {tag} (anchored={anchored}) ===", flush=True)
        X_train = flatten(arrays["train"]["X"])
        y_train = arrays["train"]["y"]
        if anchored:
            y_train = y_train - arrays["train"]["last_target"]
        print(f"   features {X_train.shape[1]}, fitting ...", flush=True)
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                              random_state=SEED, verbosity=-1, n_jobs=6)
        model.fit(X_train, y_train)
        del X_train
        out = {}
        for split in ("validation", "test"):
            X_split = flatten(arrays[split]["X"])
            pred = np.asarray(model.predict(X_split), dtype=np.float64)
            if anchored:
                pred = pred + arrays[split]["last_target"]
            out[split] = regression_metrics(arrays[split]["y"], pred)
            del X_split
        print(f"   test RMSE {out['test']['rmse']:.2f} kWh | "
              f"CV(RMSE) {out['test']['cvrmse_pct']:.2f}% | MAE {out['test']['mae']:.2f} kWh "
              f"[{time.time() - started:.0f}s]", flush=True)
        results[tag] = out
        return out

    for representation, flatten in (("summary", summary_feats),
                                    ("full_lags", flatten_windows_full_lags)):
        control = run(f"lightgbm_{representation}_CONTROL", flatten, anchored=False)
        delta = abs(control["test"]["rmse"] - PUBLISHED[representation])
        ok = delta < TOLERANCE[representation]
        verdicts[representation] = {
            "published_rmse": PUBLISHED[representation],
            "control_rmse": control["test"]["rmse"],
            "delta_kwh": delta,
            "tolerance_kwh": TOLERANCE[representation],
            "reproduced": bool(ok),
        }
        print(f"   control vs published {PUBLISHED[representation]:.2f}: delta {delta:.4f} kWh "
              f"-> {'MATCH' if ok else 'MISMATCH - DO NOT USE'}", flush=True)
        run(f"lightgbm_{representation}_ANCHORED", flatten, anchored=True)

    OUT.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")

    all_reproduced = all(v["reproduced"] for v in verdicts.values())
    manifest = {
        "purpose": "R3.3 anchored (persistence-differenced) boosting baselines, corrected "
                   "float64 summary accumulation.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_run": str(RUN),
        "inputs": {
            "window_arrays": str(RUN / "02_window_arrays.npz"),
            "window_arrays_sha256": sha256(RUN / "02_window_arrays.npz"),
        },
        "seed": SEED,
        "hyperparameters": "LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)",
        "representations": {
            "summary": "32 window statistics (last, mean, std, trend per channel), "
                       "float64 accumulation",
            "full_lags": "complete flattened 48 x 8 window (384 features)",
        },
        "controls": verdicts,
        "all_controls_reproduced": all_reproduced,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "results": results,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")

    print("\n=== SUMMARY (test split) ===", flush=True)
    for tag, value in results.items():
        print(f"  {tag:36s} RMSE {value['test']['rmse']:7.2f}  "
              f"CV {value['test']['cvrmse_pct']:6.2f}%  MAE {value['test']['mae']:6.2f}", flush=True)
    print("\nreference: PA-STT 17.20 / 11.37% / 5.78   persistence 19.52 / 12.90% / 6.44")
    print(f"controls reproduced: {all_reproduced}")
    print(f"total {time.time() - t0:.0f}s -> {OUT.name}, {MANIFEST.name}")
    return 0 if all_reproduced else 2


if __name__ == "__main__":
    raise SystemExit(main())
