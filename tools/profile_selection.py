from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def js_divergence(left: pd.Series, right: pd.Series) -> float:
    labels = sorted(set(left.index).union(right.index))
    p = left.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    q = right.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl(values: np.ndarray, reference: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / reference[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def round_robin_original(
    meter_columns: list[str],
    metadata: pd.DataFrame,
    maximum: int,
) -> list[str]:
    eligible_meta = metadata[metadata["building_id"].isin(meter_columns)].copy()
    grouped = {
        str(site): sorted(group["building_id"].astype(str).tolist())
        for site, group in eligible_meta.groupby("site_id")
    }
    selected: list[str] = []
    while len(selected) < maximum and grouped:
        for site in sorted(list(grouped)):
            if grouped[site]:
                selected.append(grouped[site].pop(0))
                if len(selected) == maximum:
                    break
            if not grouped[site]:
                grouped.pop(site, None)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile the full BDG2 electricity cohort and reconstruct the reviewed selection."
    )
    parser.add_argument("--meter", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--reviewed-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-history", type=int, default=1000)
    parser.add_argument("--min-nonzero-fraction", type=float, default=0.20)
    parser.add_argument("--min-target-mean", type=float, default=1.0)
    parser.add_argument("--original-prefilter-maximum", type=int, default=50)
    args = parser.parse_args()

    for path in [args.meter, args.metadata, args.weather, args.reviewed_profile]:
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    meter_header = pd.read_csv(args.meter, nrows=0)
    timestamp_column = "timestamp" if "timestamp" in meter_header.columns else meter_header.columns[0]
    meter_columns = [column for column in meter_header.columns if column != timestamp_column]
    metadata = metadata[metadata["building_id"].isin(meter_columns)].copy()
    metadata["primary_use"] = metadata["primaryspaceusage"].fillna("Unknown")

    weather = pd.read_csv(
        args.weather,
        usecols=["timestamp", "site_id", "airTemperature"],
        parse_dates=["timestamp"],
    )
    weather["has_air_temperature"] = pd.to_numeric(
        weather["airTemperature"], errors="coerce"
    ).notna()
    weather_lookup = {
        str(site): (
            group.drop_duplicates(subset=["timestamp"], keep="last")
            .set_index("timestamp")["has_air_temperature"]
        )
        for site, group in weather.groupby("site_id")
    }

    site_by_building = metadata.set_index("building_id")["site_id"].astype(str).to_dict()
    count = pd.Series(0, index=meter_columns, dtype=np.int64)
    nonzero_count = pd.Series(0, index=meter_columns, dtype=np.int64)
    target_sum = pd.Series(0.0, index=meter_columns, dtype=np.float64)
    negative_count = pd.Series(0, index=meter_columns, dtype=np.int64)

    processed = 0
    for chunk_index, chunk in enumerate(pd.read_csv(args.meter, chunksize=512), start=1):
        timestamps = pd.to_datetime(chunk[timestamp_column], errors="coerce")
        values = chunk[meter_columns].apply(pd.to_numeric, errors="coerce")
        for site, site_metadata in metadata.groupby("site_id"):
            columns = [column for column in site_metadata["building_id"] if column in values.columns]
            if not columns:
                continue
            available = weather_lookup.get(str(site))
            if available is None:
                weather_mask = np.zeros(len(chunk), dtype=bool)
            else:
                weather_mask = (
                    available.reindex(timestamps)
                    .fillna(False)
                    .to_numpy(dtype=bool)
                )
            site_values = values[columns]
            valid = site_values.notna() & site_values.ge(0)
            valid = valid & weather_mask[:, None]
            count.loc[columns] += valid.sum(axis=0).to_numpy(dtype=np.int64)
            nonzero_count.loc[columns] += (
                valid & site_values.gt(0)
            ).sum(axis=0).to_numpy(dtype=np.int64)
            target_sum.loc[columns] += site_values.where(valid).sum(axis=0).to_numpy(dtype=float)
            negative_count.loc[columns] += site_values.lt(0).sum(axis=0).to_numpy(dtype=np.int64)
        processed += len(chunk)
        print(
            f"processed chunk={chunk_index} rows={processed}",
            file=sys.stderr,
            flush=True,
        )

    profile = pd.DataFrame(
        {
            "building_id": meter_columns,
            "history_count": count.reindex(meter_columns).to_numpy(),
            "nonzero_count": nonzero_count.reindex(meter_columns).to_numpy(),
            "target_sum": target_sum.reindex(meter_columns).to_numpy(),
            "negative_count": negative_count.reindex(meter_columns).to_numpy(),
        }
    )
    profile["nonzero_fraction"] = np.where(
        profile["history_count"] > 0,
        profile["nonzero_count"] / profile["history_count"],
        0.0,
    )
    profile["target_mean"] = np.where(
        profile["history_count"] > 0,
        profile["target_sum"] / profile["history_count"],
        np.nan,
    )
    profile = profile.merge(
        metadata[
            [
                "building_id",
                "site_id",
                "primary_use",
                "sqm",
                "yearbuilt",
            ]
        ],
        on="building_id",
        how="left",
    )
    profile["passes_quality"] = (
        (profile["history_count"] >= args.min_history)
        & (profile["nonzero_fraction"] >= args.min_nonzero_fraction)
        & (profile["target_mean"] >= args.min_target_mean)
    )

    original_prefilter = set(
        round_robin_original(
            meter_columns=meter_columns,
            metadata=metadata,
            maximum=args.original_prefilter_maximum,
        )
    )
    reviewed = set(
        json.loads(args.reviewed_profile.read_text(encoding="utf-8"))[
            "rows_by_building"
        ].keys()
    )
    profile["selected_before_quality_in_reviewed_code"] = profile["building_id"].isin(
        original_prefilter
    )
    profile["included_in_reviewed_manuscript"] = profile["building_id"].isin(reviewed)
    profile.sort_values(["site_id", "primary_use", "building_id"]).to_csv(
        args.output_dir / "full_electricity_cohort_profile.csv",
        index=False,
    )

    eligible = profile[profile["passes_quality"]].copy()
    reviewed_frame = profile[profile["included_in_reviewed_manuscript"]].copy()

    site_comparison = (
        profile.groupby("site_id")
        .agg(
            full_electricity_buildings=("building_id", "nunique"),
            eligible_buildings=("passes_quality", "sum"),
            reviewed_buildings=("included_in_reviewed_manuscript", "sum"),
        )
        .reset_index()
    )
    site_comparison.to_csv(args.output_dir / "site_count_comparison.csv", index=False)

    use_full = eligible["primary_use"].value_counts().sort_index()
    use_reviewed = reviewed_frame["primary_use"].value_counts().sort_index()
    use_comparison = pd.DataFrame(
        {
            "eligible_count": use_full,
            "reviewed_count": use_reviewed,
        }
    ).fillna(0)
    use_comparison["eligible_pct"] = (
        100.0 * use_comparison["eligible_count"] / use_comparison["eligible_count"].sum()
    )
    use_comparison["reviewed_pct"] = (
        100.0 * use_comparison["reviewed_count"] / use_comparison["reviewed_count"].sum()
    )
    use_comparison.reset_index(names="primary_use").to_csv(
        args.output_dir / "primary_use_comparison.csv",
        index=False,
    )

    summary = {
        "source_files": {
            "meter": {"path": str(args.meter), "sha256": sha256(args.meter)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "weather": {"path": str(args.weather), "sha256": sha256(args.weather)},
            "reviewed_profile": {
                "path": str(args.reviewed_profile),
                "sha256": sha256(args.reviewed_profile),
            },
        },
        "thresholds": {
            "minimum_history_count": args.min_history,
            "minimum_nonzero_fraction": args.min_nonzero_fraction,
            "minimum_target_mean_kwh": args.min_target_mean,
            "negative_readings_excluded": True,
            "missing_air_temperature_rows_excluded": True,
        },
        "counts": {
            "metadata_buildings": int(pd.read_csv(args.metadata)["building_id"].nunique()),
            "electricity_meter_buildings": len(meter_columns),
            "quality_eligible_electricity_buildings": int(profile["passes_quality"].sum()),
            "reviewed_prefilter_buildings": len(original_prefilter),
            "reviewed_final_buildings": len(reviewed),
            "reviewed_final_sites": int(reviewed_frame["site_id"].nunique()),
        },
        "selection_reconstruction": {
            "reviewed_ids_match_reconstructed_prefilter_after_quality": reviewed
            == set(
                profile.loc[
                    profile["selected_before_quality_in_reviewed_code"]
                    & profile["passes_quality"],
                    "building_id",
                ]
            ),
            "reviewed_ids_all_quality_eligible": reviewed.issubset(
                set(eligible["building_id"])
            ),
            "description": (
                "The reviewed code sorted building identifiers within each site, "
                "selected 50 by site round-robin, then applied the quality thresholds."
            ),
        },
        "representativeness": {
            "primary_use_jensen_shannon_divergence_bits": js_divergence(
                use_reviewed, use_full
            ),
            "eligible_primary_use_count": int(use_full.size),
            "reviewed_primary_use_count": int(use_reviewed.size),
            "eligible_floor_area_median_m2": float(eligible["sqm"].median()),
            "reviewed_floor_area_median_m2": float(reviewed_frame["sqm"].median()),
            "eligible_floor_area_iqr_m2": [
                float(eligible["sqm"].quantile(0.25)),
                float(eligible["sqm"].quantile(0.75)),
            ],
            "reviewed_floor_area_iqr_m2": [
                float(reviewed_frame["sqm"].quantile(0.25)),
                float(reviewed_frame["sqm"].quantile(0.75)),
            ],
        },
    }
    (args.output_dir / "selection_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
