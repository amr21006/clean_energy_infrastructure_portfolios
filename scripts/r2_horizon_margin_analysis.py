"""R2: at which horizons is the advantage actually established?

The second-round assessment proposes restricting superiority claims to
"horizons beyond three hours where their lead is genuine". That is the right
remedy in principle, but the boundary should be set by the evidence rather than
assumed, and the archived results make the picture less tidy than it first
appears.

Two baseline families behave in opposite directions with horizon:

  * plain boosting improves relative to PA-STT as the horizon lengthens, because
    calendar structure carries more of the signal than recent dynamics;
  * anchored boosting is strong at one hour and degrades badly beyond six, since
    differencing against a reference twelve or more hours old is a poor start.

So the identity of "the strongest baseline" changes with horizon. A margin
quoted against only one family can flatter the model at horizons where that
family happens to be weak. This script therefore compares PA-STT against the
best baseline of ANY family at each horizon, and expresses the margin in units
of the measured run-to-run noise floor.

Noise floor
-----------
Multi-seed replication was run at one hour, giving a standard deviation in test
RMSE. For other horizons no replication was run, so the floor is scaled by
holding the RELATIVE spread constant (sd / rmse). That is an assumption, it is
labelled as one in the output, and the verdicts are reported with the multiplier
so a reader can apply a different assumption.

A horizon is called established only when the margin exceeds three times the
noise floor at that horizon.

Output: r2_horizon_margin_analysis.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\pmo3.EDG.000\Desktop\EB Revision")
EXT = ROOT / "revision_results" / "extended_validation"
HERE = Path(__file__).parent
OUT = HERE / "r2_horizon_margin_analysis.json"

MULTISEED = HERE / "r2_multiseed_h1.json"
ANCHORED = HERE / "r33_multihorizon.json"
ANCHORED_H1 = HERE / "r33_results_v2.json"

ESTABLISHED_MULTIPLE = 3.0


def load_anchored() -> dict[int, float]:
    """Best-of-representation anchored-booster test RMSE per horizon."""
    values: dict[int, float] = {}
    if ANCHORED_H1.exists():
        data = json.loads(ANCHORED_H1.read_text(encoding="utf-8"))
        candidates = [v["test"]["rmse"] for k, v in data.items() if k.endswith("_ANCHORED")]
        if candidates:
            values[1] = min(candidates)
    if ANCHORED.exists():
        data = json.loads(ANCHORED.read_text(encoding="utf-8"))
        for horizon, row in data.items():
            options = [row[rep]["anchored"]["rmse"] for rep in ("summary", "full_lags")
                       if rep in row]
            if options:
                values[int(horizon)] = min(options)
    return values


def main() -> int:
    if not MULTISEED.exists():
        raise SystemExit("run r2_multiseed_h1.py first")
    seeds = json.loads(MULTISEED.read_text(encoding="utf-8"))
    stats = seeds["statistics"]
    sd_h1 = stats["rmse_sd"]
    mean_h1 = stats["rmse_mean"]
    relative_sd = sd_h1 / mean_h1

    metrics = pd.read_csv(EXT / "multihorizon_metrics.csv")
    test = metrics[metrics["split"] == "test"]
    pivot = test.pivot_table(index="horizon_hours", columns="model", values="rmse")

    anchored = load_anchored()
    missing = [h for h in pivot.index if h not in anchored]

    rows = []
    for horizon in sorted(pivot.index):
        # At one hour the replicated mean is the honest estimate; the archived
        # single-seed value was the second-best of six draws. Other horizons
        # remain single-seed and are flagged as such.
        if horizon == 1:
            pa = mean_h1
            pa_basis = f"multi-seed mean over {stats['n_seeds']} seeds"
        else:
            pa = float(pivot.loc[horizon, "pa_stt"])
            pa_basis = "single seed (42)"
        families = {
            "persistence": float(pivot.loc[horizon, "persistence"]),
            "lightgbm_summary": float(pivot.loc[horizon, "lightgbm_summary"]),
            "lightgbm_full_lags": float(pivot.loc[horizon, "lightgbm_full_lags"]),
        }
        if horizon in anchored:
            families["lightgbm_anchored"] = float(anchored[horizon])

        best_name = min(families, key=families.get)
        best_value = families[best_name]
        margin = best_value - pa
        noise = relative_sd * pa if horizon != 1 else sd_h1
        multiple = margin / noise if noise > 0 else float("nan")
        rows.append({
            "horizon_hours": int(horizon),
            "pa_stt_rmse": pa,
            "pa_stt_basis": pa_basis,
            "pa_stt_archived_single_seed": float(pivot.loc[horizon, "pa_stt"]),
            "baselines": families,
            "strongest_baseline": best_name,
            "strongest_baseline_rmse": best_value,
            "margin_kwh": margin,
            "margin_pct": 100.0 * margin / best_value,
            "noise_floor_kwh": noise,
            "noise_floor_source": ("measured, multi-seed" if horizon == 1
                                   else "scaled from the one-hour relative spread (assumption)"),
            "margin_in_noise_units": multiple,
            "established": bool(margin > ESTABLISHED_MULTIPLE * noise),
        })

    established = [r["horizon_hours"] for r in rows if r["established"]]
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "criterion": f"margin over the strongest baseline of any family exceeds "
                     f"{ESTABLISHED_MULTIPLE:g}x the run-to-run noise floor",
        "noise_floor_h1_kwh": sd_h1,
        "noise_floor_relative": relative_sd,
        "noise_floor_assumption": (
            "Replication was run at one hour only. For other horizons the floor is obtained by "
            "holding the relative spread (sd / rmse) constant. This is an assumption, not a "
            "measurement, and the margin is reported in noise units so a reader can vary it."
        ),
        "anchored_horizons_available": sorted(anchored),
        "anchored_horizons_missing": missing,
        "horizons": rows,
        "established_horizons": established,
        "verdict": (
            f"The advantage is established at {established} hour(s) ahead under the stated "
            f"criterion." if established else
            "No horizon meets the criterion on the evidence available."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    print("=" * 96)
    print("HORIZON MARGIN ANALYSIS - PA-STT vs the strongest baseline of any family")
    print("=" * 96)
    print(f"noise floor at h=1: {sd_h1:.4f} kWh (measured over {stats['n_seeds']} seeds); "
          f"relative {relative_sd * 100:.2f}%")
    if missing:
        print(f"WARNING: anchored-booster values missing for horizons {missing}; "
              f"those rows compare against plain baselines only")
    print()
    print(f"{'h':>3} {'PA-STT':>8} {'strongest baseline':>20} {'value':>8} "
          f"{'margin':>8} {'margin%':>8} {'noise':>7} {'xnoise':>7}  verdict")
    for r in rows:
        print(f"{r['horizon_hours']:>3} {r['pa_stt_rmse']:>8.2f} {r['strongest_baseline']:>20} "
              f"{r['strongest_baseline_rmse']:>8.2f} {r['margin_kwh']:>8.2f} "
              f"{r['margin_pct']:>7.1f}% {r['noise_floor_kwh']:>7.2f} "
              f"{r['margin_in_noise_units']:>7.1f}  "
              f"{'ESTABLISHED' if r['established'] else 'not established'}")
    print(f"\n{report['verdict']}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
