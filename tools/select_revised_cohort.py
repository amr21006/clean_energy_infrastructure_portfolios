from __future__ import annotations

import argparse
import hashlib
import json
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
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)

    def kl(values: np.ndarray, reference: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / reference[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a deterministic equal-site sample from quality-eligible BDG2 meters."
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buildings-per-site", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    if not args.profile.exists():
        raise FileNotFoundError(args.profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = pd.read_csv(args.profile)
    required = {
        "building_id",
        "site_id",
        "primary_use",
        "sqm",
        "history_count",
        "nonzero_fraction",
        "target_mean",
        "passes_quality",
    }
    missing = required - set(profile.columns)
    if missing:
        raise ValueError(f"Profile is missing required columns: {sorted(missing)}")

    eligible = profile[profile["passes_quality"]].copy()
    rng = np.random.default_rng(args.seed)
    selected_indices: list[int] = []
    for site_id, group in eligible.groupby("site_id", sort=True):
        if len(group) < args.buildings_per_site:
            raise ValueError(
                f"Site {site_id} has only {len(group)} eligible buildings, fewer than "
                f"the requested {args.buildings_per_site}."
            )
        selected_indices.extend(
            rng.choice(
                group.index.to_numpy(),
                size=args.buildings_per_site,
                replace=False,
            ).tolist()
        )

    selected = eligible.loc[selected_indices].copy()
    selected["selection_seed"] = args.seed
    selected["selection_probability_within_site"] = selected["site_id"].map(
        eligible.groupby("site_id").size().rdiv(args.buildings_per_site)
    )
    selected = selected.sort_values(["site_id", "building_id"]).reset_index(drop=True)

    selection_columns = [
        "building_id",
        "site_id",
        "primary_use",
        "sqm",
        "history_count",
        "nonzero_fraction",
        "target_mean",
        "selection_seed",
        "selection_probability_within_site",
    ]
    selected[selection_columns].to_csv(
        args.output_dir / "revised_selected_buildings.csv",
        index=False,
    )

    site_counts = (
        eligible.groupby("site_id")
        .size()
        .rename("eligible_buildings")
        .to_frame()
        .join(
            selected.groupby("site_id")
            .size()
            .rename("selected_buildings"),
            how="left",
        )
        .fillna(0)
        .reset_index()
    )
    site_counts.to_csv(args.output_dir / "revised_site_counts.csv", index=False)

    eligible_uses = eligible["primary_use"].value_counts().sort_index()
    selected_uses = selected["primary_use"].value_counts().sort_index()
    use_counts = pd.DataFrame(
        {
            "eligible_count": eligible_uses,
            "selected_count": selected_uses,
        }
    ).fillna(0)
    use_counts["eligible_pct"] = (
        100.0 * use_counts["eligible_count"] / use_counts["eligible_count"].sum()
    )
    use_counts["selected_pct"] = (
        100.0 * use_counts["selected_count"] / use_counts["selected_count"].sum()
    )
    use_counts.reset_index(names="primary_use").to_csv(
        args.output_dir / "revised_primary_use_counts.csv",
        index=False,
    )

    reviewed = profile[profile["included_in_reviewed_manuscript"]]
    summary = {
        "profile": {
            "path": str(args.profile),
            "sha256": sha256(args.profile),
        },
        "algorithm": (
            "After applying the predeclared quality thresholds to all electricity "
            "meters, draw five buildings without replacement from each of the 19 "
            "sites using NumPy Generator PCG64 and the recorded seed."
        ),
        "seed": args.seed,
        "buildings_per_site": args.buildings_per_site,
        "counts": {
            "eligible_buildings": int(len(eligible)),
            "eligible_sites": int(eligible["site_id"].nunique()),
            "selected_buildings": int(len(selected)),
            "selected_sites": int(selected["site_id"].nunique()),
            "eligible_primary_uses": int(eligible["primary_use"].nunique()),
            "selected_primary_uses": int(selected["primary_use"].nunique()),
        },
        "representativeness": {
            "primary_use_jensen_shannon_divergence_bits": js_divergence(
                selected_uses,
                eligible_uses,
            ),
            "reviewed_primary_use_jensen_shannon_divergence_bits": js_divergence(
                reviewed["primary_use"].value_counts(),
                eligible_uses,
            ),
            "selected_floor_area_median_m2": float(selected["sqm"].median()),
            "eligible_floor_area_median_m2": float(eligible["sqm"].median()),
            "selected_floor_area_iqr_m2": [
                float(selected["sqm"].quantile(0.25)),
                float(selected["sqm"].quantile(0.75)),
            ],
            "eligible_floor_area_iqr_m2": [
                float(eligible["sqm"].quantile(0.25)),
                float(eligible["sqm"].quantile(0.75)),
            ],
        },
    }
    (args.output_dir / "revised_selection_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
