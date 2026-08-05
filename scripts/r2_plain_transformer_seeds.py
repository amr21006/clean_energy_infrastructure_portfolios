"""Matched-seed replication of the anchor ablation at the one-hour horizon.

Why this exists
---------------
Table 8 attributes 0.14 kWh of RMSE and 12.9% of MAE at one hour to the persistence anchor, by
comparing PA-STT against an otherwise identical unanchored Transformer. Both sides of that
comparison rested on a single training seed. A six-seed replication of PA-STT alone
(r2_multiseed_h1.py) put its seed-to-seed spread at 0.28 kWh, which is larger than the RMSE
effect being claimed, so the comparison could not be resolved.

The validation suite was intended to close this gap but did not: its manifest records
variants ["main"] and suite_ablation_summary.csv is empty, because it was invoked with
--skip-ablations. The plain Transformer therefore never ran at more than one seed.

This script trains the plain Transformer at the same seeds for which PA-STT values already
exist, on the same archived windows and target scaler, changing only residual_feature_index
(0 -> None, which disables the anchor in AnchoredTemporalTransformer.forward). It then pairs
the two models seed by seed, which is the correct comparison: it removes the seed as a source
of between-model variance instead of comparing two independent seed distributions.

Seed 42 is retrained rather than taken from the archive, so the run doubles as a check that
this harness reproduces the published plain-Transformer value of 17.345 kWh.

Outputs
-------
  results/seed_replication/r2_plain_transformer_seeds.json           per-seed and paired stats
  results/seed_replication/r2_plain_transformer_seeds_manifest.json  inputs, hashes, environment
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
PKG = (ROOT / "Final Submittion 3-8-2026" / "Gemini EB edits"
       / "github_upload_Code_and_Results")
OUTDIR = PKG / "results" / "seed_replication"
OUT = OUTDIR / "r2_plain_transformer_seeds.json"
MANIFEST = OUTDIR / "r2_plain_transformer_seeds_manifest.json"
PASTT_JSON = OUTDIR / "r2_multiseed_h1.json"

sys.path.insert(0, str(CODE / "src"))
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from energy_forecasting.features import (  # noqa: E402
    add_time_features, chronological_split, fit_target_scaler,
)
from energy_forecasting.train import train_anchored_transformer  # noqa: E402

# Archived model configuration, verbatim, with the anchor removed. Everything else - encoder
# capacity, head, schedule, budget - is identical to PA-STT, which is what makes this a matched
# ablation rather than a different model.
MODEL_CFG = {
    "name": "anchored_temporal_transformer",
    "reporting_name": "plain_transformer",
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.1,
    "dim_feedforward": 128,
    "residual_feature_index": None,   # <- the only change: no persistence anchor
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

SEEDS = [42, 52, 62]
PUBLISHED_PLAIN_RMSE = 17.345024116213693   # Table 8, component-ablation stage, seed 42
PUBLISHED_PLAIN_MAE = 6.642648712985935


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


def pastt_by_seed() -> dict[int, dict[str, float]]:
    """PA-STT test metrics at each seed, from the existing six-seed replication."""
    payload = json.loads(PASTT_JSON.read_text(encoding="utf-8"))
    return {int(r["seed"]): {"rmse": float(r["rmse"]), "mae": float(r["mae"]),
                            "cvrmse_pct": float(r["cvrmse_pct"])}
            for r in payload["seeds"]}


def main() -> int:
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("loading archived windows ...", flush=True)
    arrays = load_arrays()
    for split, values in arrays.items():
        print("   %-10s X%s" % (split, values["X"].shape), flush=True)

    print("rebuilding the archived target scaler ...", flush=True)
    clean = pd.read_parquet(RUN / "01_cleaned_long.parquet")
    split_frame = chronological_split(
        add_time_features(clean), {"train": 0.70, "validation": 0.15, "test": 0.15})
    target_scaler = fit_target_scaler(split_frame, "meter_reading")
    del clean, split_frame
    gc.collect()
    print("   mean=%.4f std=%.4f" % (target_scaler.mean, target_scaler.std), flush=True)

    pastt = pastt_by_seed()
    print("   PA-STT reference seeds available: %s" % sorted(pastt), flush=True)

    rows = []
    for seed in SEEDS:
        started = time.time()
        print("\n=== training plain Transformer, seed %d ===" % seed, flush=True)
        result = train_anchored_transformer(
            arrays=arrays,
            model_cfg=dict(MODEL_CFG),
            target_scaler=target_scaler,
            seed=seed,
            device_name="cuda",
        )
        test = result.metric_frame[result.metric_frame["split"] == "test"].iloc[0]
        row = {
            "seed": seed,
            "rmse": float(test["rmse"]),
            "mae": float(test["mae"]),
            "cvrmse_pct": float(test["cvrmse_pct"]),
            "r2": float(test["r2"]),
            "training_seconds": float(time.time() - started),
        }
        rows.append(row)
        print("   seed %d: RMSE %.4f  MAE %.4f  CV %.4f%%  [%.0fs]"
              % (seed, row["rmse"], row["mae"], row["cvrmse_pct"], row["training_seconds"]),
              flush=True)
        OUT.write_text(json.dumps({"plain_transformer": rows}, indent=1), encoding="utf-8")
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ paired analysis
    paired = []
    for r in rows:
        s = r["seed"]
        if s not in pastt:
            continue
        paired.append({
            "seed": s,
            "pa_stt_rmse": pastt[s]["rmse"], "plain_rmse": r["rmse"],
            "rmse_delta": r["rmse"] - pastt[s]["rmse"],
            "rmse_delta_pct": 100.0 * (r["rmse"] - pastt[s]["rmse"]) / r["rmse"],
            "pa_stt_mae": pastt[s]["mae"], "plain_mae": r["mae"],
            "mae_delta": r["mae"] - pastt[s]["mae"],
            "mae_delta_pct": 100.0 * (r["mae"] - pastt[s]["mae"]) / r["mae"],
        })

    def block(key_delta, key_pct):
        d = np.array([p[key_delta] for p in paired], dtype=float)
        pct = np.array([p[key_pct] for p in paired], dtype=float)
        n = len(d)
        out = {
            "n_paired_seeds": n,
            "mean_delta": float(d.mean()),
            "sd_delta": float(d.std(ddof=1)) if n > 1 else None,
            "min_delta": float(d.min()),
            "max_delta": float(d.max()),
            "mean_delta_pct": float(pct.mean()),
            "anchor_helps_in_all_seeds": bool((d > 0).all()),
        }
        if n > 1:
            tt = stats.ttest_1samp(d, 0.0, alternative="greater")
            out["t_statistic"] = float(tt.statistic)
            out["p_one_sided_greater"] = float(tt.pvalue)
            out["significant_at_0_05"] = bool(tt.pvalue < 0.05)
        return out

    plain_rmse = np.array([r["rmse"] for r in rows], dtype=float)
    plain_mae = np.array([r["mae"] for r in rows], dtype=float)
    pastt_rmse = np.array([pastt[r["seed"]]["rmse"] for r in rows if r["seed"] in pastt])

    harness = {
        "published_seed42_rmse": PUBLISHED_PLAIN_RMSE,
        "reproduced_seed42_rmse": rows[0]["rmse"] if rows[0]["seed"] == 42 else None,
        "abs_difference_kwh": (abs(rows[0]["rmse"] - PUBLISHED_PLAIN_RMSE)
                               if rows[0]["seed"] == 42 else None),
        "published_seed42_mae": PUBLISHED_PLAIN_MAE,
        "reproduced_seed42_mae": rows[0]["mae"] if rows[0]["seed"] == 42 else None,
    }

    summary = {
        "plain_transformer_rmse_mean": float(plain_rmse.mean()),
        "plain_transformer_rmse_sd": float(plain_rmse.std(ddof=1)),
        "plain_transformer_rmse_range": [float(plain_rmse.min()), float(plain_rmse.max())],
        "plain_transformer_mae_mean": float(plain_mae.mean()),
        "plain_transformer_mae_sd": float(plain_mae.std(ddof=1)),
        "pa_stt_rmse_mean_same_seeds": float(pastt_rmse.mean()),
        "seed_ranges_overlap": bool(plain_rmse.min() <= pastt_rmse.max()
                                    and pastt_rmse.min() <= plain_rmse.max()),
        "paired_rmse": block("rmse_delta", "rmse_delta_pct"),
        "paired_mae": block("mae_delta", "mae_delta_pct"),
        "harness_check_seed42": harness,
    }

    print("\n" + "=" * 78)
    print("MATCHED-SEED ANCHOR ABLATION (one-hour horizon, test split)")
    print("=" * 78)
    print("  %-6s %10s %10s %9s | %10s %10s %9s"
          % ("seed", "PA-STT", "plain", "dRMSE", "PA-STT", "plain", "dMAE"))
    for p in paired:
        print("  %-6d %10.4f %10.4f %+9.4f | %10.4f %10.4f %+9.4f"
              % (p["seed"], p["pa_stt_rmse"], p["plain_rmse"], p["rmse_delta"],
                 p["pa_stt_mae"], p["plain_mae"], p["mae_delta"]))
    pr, pm = summary["paired_rmse"], summary["paired_mae"]
    print("\n  RMSE  anchor worth %+.4f kWh (%.2f%%), positive in every seed: %s"
          % (pr["mean_delta"], pr["mean_delta_pct"], pr["anchor_helps_in_all_seeds"]))
    if "p_one_sided_greater" in pr:
        print("        paired t p = %.4f -> %s at 0.05"
              % (pr["p_one_sided_greater"],
                 "SIGNIFICANT" if pr["significant_at_0_05"] else "NOT SIGNIFICANT"))
    print("  MAE   anchor worth %+.4f kWh (%.2f%%), positive in every seed: %s"
          % (pm["mean_delta"], pm["mean_delta_pct"], pm["anchor_helps_in_all_seeds"]))
    if "p_one_sided_greater" in pm:
        print("        paired t p = %.4f -> %s at 0.05"
              % (pm["p_one_sided_greater"],
                 "SIGNIFICANT" if pm["significant_at_0_05"] else "NOT SIGNIFICANT"))
    if harness["reproduced_seed42_rmse"] is not None:
        print("\n  harness check, seed 42: published %.4f vs reproduced %.4f (|d| = %.4f kWh)"
              % (harness["published_seed42_rmse"], harness["reproduced_seed42_rmse"],
                 harness["abs_difference_kwh"]))

    OUT.write_text(json.dumps(
        {"plain_transformer": rows, "pa_stt_reference": pastt, "paired": paired,
         "summary": summary}, indent=1, default=float), encoding="utf-8")
    MANIFEST.write_text(json.dumps({
        "purpose": "Matched-seed replication of the persistence-anchor ablation at h=1, to test "
                   "whether the Table 8 anchor effect survives training-seed variability.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_run": str(RUN),
        "window_arrays_sha256": sha256(RUN / "02_window_arrays.npz"),
        "model_config": MODEL_CFG,
        "seeds": SEEDS,
        "pa_stt_reference_file": str(PASTT_JSON.name),
        "training_function": "energy_forecasting.train.train_anchored_transformer",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__,
        },
        "summary": summary,
    }, indent=2, default=float), encoding="utf-8")
    print("\ntotal %.0fs -> %s" % (time.time() - t0, OUT.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
