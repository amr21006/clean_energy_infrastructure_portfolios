"""Back-test of stratified portfolio contingency.

The rule tested: each stratum's expected attrition is its
observed 24-month cancellation rate in earlier entry cohorts multiplied by the announced
capacity of the units now entering. Strata are delivery stage at first report crossed with
technology group; a stratum with fewer than 20 source units takes the rate of its stage. The
portfolio-wide rate is the unstratified reference, and adding size classes is a supplementary
variant. Rates come from the training cohort (24-month labels resolved by December 2023) and
from the training and 2022 validation cohorts (resolved by December 2024); both are applied to
the test cohort (entry January 2023 to May 2024). Uncertainty: 500 resamples of whole
development clusters, drawn independently in the source and test cohorts.

Run from the repository root:
    python supplementary_analyses/contingency_backtest.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "outputs" / "delivery_twin"
OUT = ROOT / "outputs" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261016
REPS = 500
MIN_UNITS = 20
STAGE = {"P": "pre-construction", "L": "pre-construction", "T": "pre-construction",
         "U": "construction", "V": "construction", "TS": "commissioning"}
RULES = {
    "portfolio-wide rate": [],
    "stage and technology": ["stage", "technology_group"],
    "stage, technology and size": ["stage", "technology_group", "size_class"],
}
SOURCES = {
    "training cohort, labels resolved by December 2023": ("train",),
    "training and 2022 cohorts, labels resolved by December 2024": ("train", "validation"),
}


def frame() -> pd.DataFrame:
    landmarks = pd.read_parquet(PRIMARY / "processed" / "landmarks.parquet")
    f = landmarks.loc[(landmarks["landmark_age_months"] == 0)
                      & landmarks["observable_24_month_outcome"].astype(bool)
                      & landmarks["eligible_primary_cohort"].astype(bool)].copy()
    f["stage"] = f["current_state"].map(STAGE).fillna("other")
    f["size_class"] = pd.cut(f["capacity_mw"], [-np.inf, 5, 50, 150, np.inf],
                             labels=["5 MW or less", "over 5 to 50 MW", "over 50 to 150 MW",
                                     "over 150 MW"]).astype(str)
    f["y"] = f["cancellation_within_24_months"].astype(int)
    return f


def expected(source: pd.DataFrame, target: pd.DataFrame, keys: list[str]) -> tuple[float, float]:
    """Expected cancelled units and MW in the target under the stratified-rate rule."""
    overall = source["y"].mean()
    stage_rate = source.groupby("stage")["y"].mean()
    if keys:
        cells = source.groupby(keys)["y"].agg(["mean", "size"])
        cells = cells.loc[cells["size"] >= MIN_UNITS, "mean"].rename("rate")
        t = target.join(cells, on=keys)
        rate = t["rate"].fillna(t["stage"].map(stage_rate)).fillna(overall)
    else:
        rate = pd.Series(overall, index=target.index)
    return float(rate.sum()), float((rate * target["capacity_mw"]).sum())


def cluster_sample(f: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    codes, uniques = pd.factorize(f["development_cluster_id"])
    members = [np.flatnonzero(codes == k) for k in range(len(uniques))]
    draw = rng.integers(0, len(uniques), len(uniques))
    return f.iloc[np.concatenate([members[k] for k in draw])]


def main() -> None:
    f = frame()
    test = f.loc[f["analysis_split"] == "test"]
    observed_units = int(test["y"].sum())
    observed_mw = float((test["y"] * test["capacity_mw"]).sum())
    rng = np.random.default_rng(SEED)
    rows = []
    for source_name, splits in SOURCES.items():
        source = f.loc[f["analysis_split"].isin(splits)]
        for rule, keys in RULES.items():
            units, mw = expected(source, test, keys)
            boot_u, boot_mw = [], []
            for _ in range(REPS):
                s = cluster_sample(source, rng)
                t = cluster_sample(test, rng)
                u_hat, mw_hat = expected(s, t, keys)
                boot_u.append(u_hat / max(t["y"].sum(), 1))
                boot_mw.append(mw_hat / max((t["y"] * t["capacity_mw"]).sum(), 1e-9))
            rows.append({
                "rate_source": source_name, "rule": rule,
                "source_units": len(source), "source_events": int(source["y"].sum()),
                "source_prevalence": source["y"].mean(),
                "test_units": len(test), "observed_units": observed_units,
                "expected_units": units, "unit_ratio": units / observed_units,
                "unit_ratio_low": np.percentile(boot_u, 2.5),
                "unit_ratio_high": np.percentile(boot_u, 97.5),
                "test_capacity_mw": float(test["capacity_mw"].sum()),
                "observed_mw": observed_mw, "expected_mw": mw, "mw_ratio": mw / observed_mw,
                "mw_ratio_low": np.percentile(boot_mw, 2.5),
                "mw_ratio_high": np.percentile(boot_mw, 97.5),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "s_contingency_backtest.csv", index=False)
    cancelled = test.loc[test["y"] == 1, "capacity_mw"].sort_values(ascending=False)
    concentration = pd.DataFrame([{
        "cancelled_units": len(cancelled), "cancelled_mw": float(cancelled.sum()),
        "largest_unit_mw": float(cancelled.iloc[0]),
        "top5_share_of_cancelled_mw": float(cancelled.head(5).sum() / cancelled.sum()),
        "units_above_150mw": int((cancelled > 150).sum()),
        "share_mw_above_150mw": float(cancelled[cancelled > 150].sum() / cancelled.sum()),
    }])
    concentration.to_csv(OUT / "s_contingency_backtest_concentration.csv", index=False)
    print(result.round(3).to_string())
    print(concentration.round(3).to_string())


if __name__ == "__main__":
    main()
