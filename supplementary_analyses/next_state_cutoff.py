"""Sensitivity analysis: next-state prediction with training transitions restricted to calendar
months before test entry began (January 2023).

Partitions in the primary analysis are by unit entry month, so later monthly transitions of
training units overlap the test period in calendar time. This script refits the Markov chain,
the multinomial logistic regression and histogram gradient boosting on (a) all training
transitions and (b) only training transitions reported before January 2023, and scores both on
the same test transitions. Output: outputs/supplementary/s_next_state_calendar_cutoff.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eia_delivery_twin.config import AnalysisConfig  # noqa: E402
from eia_delivery_twin.modeling import (  # noqa: E402
    _logistic_pipeline,
    _prepare_features,
    _tree_pipeline,
    feature_columns,
    markov_probabilities,
)

PRIMARY = ROOT / "outputs" / "delivery_twin"
OUT = ROOT / "outputs" / "supplementary"
CONFIG = AnalysisConfig(raw_dir=ROOT / "data" / "raw", output_dir=PRIMARY)


def aligned(prob: np.ndarray, model_classes, classes) -> np.ndarray:
    frame = pd.DataFrame(prob, columns=list(model_classes))
    return frame.reindex(columns=list(classes), fill_value=0.0).to_numpy()


def score(name: str, setting: str, n_train: int, prob: np.ndarray, test: pd.DataFrame,
          classes) -> dict:
    y = test["next_state"].to_numpy()
    prob = np.clip(prob, 1e-12, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)
    pred = np.asarray(classes)[prob.argmax(axis=1)]
    c_index = list(classes).index("C")
    is_c = (y == "C").astype(int)
    return {
        "setting": setting,
        "model": name,
        "train_transitions": n_train,
        "test_transitions": len(y),
        "log_loss": log_loss(y, prob, labels=list(classes)),
        "macro_f1": f1_score(y, pred, average="macro", labels=[c for c in classes if c in set(y)]),
        "cancellation_roc_auc": roc_auc_score(is_c, prob[:, c_index]),
        "cancellation_average_precision": average_precision_score(is_c, prob[:, c_index]),
    }


def main() -> None:
    transitions = pd.read_parquet(PRIMARY / "processed" / "monthly_transitions.parquet")
    frame = transitions.loc[
        transitions["eligible_primary_cohort"]
        & transitions["analysis_split"].isin(["train", "validation", "test"])
    ].copy()
    classes = np.array(sorted(frame["next_state"].unique()))
    categorical, numeric = feature_columns(CONFIG)
    train_all = frame.loc[frame["analysis_split"] == "train"]
    test = frame.loc[frame["analysis_split"] == "test"]
    train_cut = train_all.loc[train_all["report_month"] < "2023-01"]
    rows = []
    for setting, train in (("all training transitions", train_all),
                           ("training transitions before January 2023", train_cut)):
        markov = markov_probabilities(train, test, classes)
        rows.append(score("Empirical Markov", setting, len(train), markov, test, classes))
        for name, pipe in (
            ("Multinomial logistic", _logistic_pipeline(categorical, numeric, multiclass=True)),
            ("Histogram gradient boosting",
             _tree_pipeline(categorical, numeric, "hist_gradient_boosting", CONFIG.seed + 302)),
        ):
            pipe.fit(_prepare_features(train, categorical, numeric), train["next_state"],
                     classifier__sample_weight=train["cluster_equalizing_weight"])
            prob = pipe.predict_proba(_prepare_features(test, categorical, numeric))
            prob = aligned(prob, pipe.named_steps["classifier"].classes_, classes)
            rows.append(score(name, setting, len(train), prob, test, classes))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "s_next_state_calendar_cutoff.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
