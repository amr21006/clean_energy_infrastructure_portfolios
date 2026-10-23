from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

from eia_delivery_twin.deep_models import DeliverySparseMoENetwork
from eia_delivery_twin.io import (
    excel_column_index,
    harmonize_status,
    normalize_identifier,
    period_from_year_month,
    sector_group,
    technology_group,
)
from eia_delivery_twin.modeling import (
    align_probabilities,
    paired_bootstrap_binary_comparisons,
    state_conditioned_delivery_ensemble,
)


def test_excel_column_index() -> None:
    assert excel_column_index("A1") == 0
    assert excel_column_index("Z99") == 25
    assert excel_column_index("AA10") == 26


def test_identifier_normalization() -> None:
    assert normalize_identifier(5.0) == "5"
    assert normalize_identifier(" a1 ") == "A1"
    assert normalize_identifier(None) is None


def test_period_parsing() -> None:
    assert period_from_year_month(2024, 7) == pd.Period("2024-07", freq="M")
    assert period_from_year_month(2024, 13) is None
    assert period_from_year_month(None, 7) is None


def test_status_harmonization() -> None:
    assert harmonize_status("(P) Planned for installation") == "P"
    assert harmonize_status("(L) Regulatory approvals pending") == "L"
    assert harmonize_status("(TS) Construction complete") == "TS"
    assert harmonize_status("unmapped") == "OTHER"


def test_technology_grouping() -> None:
    assert technology_group("Solar Photovoltaic", "SUN", "PV") == "solar"
    assert technology_group("Batteries", "MWH", "BA") == "storage"
    assert technology_group("Onshore Wind Turbine", "WND", "WT") == "wind"
    assert technology_group("Natural Gas Combined Cycle", "NG", "CT") == "gas"


def test_sector_grouping() -> None:
    assert sector_group("IPP Non-CHP") == "independent_power_producer"
    assert sector_group("Electric Utility") == "electric_utility"
    assert sector_group("Industrial CHP") == "industrial"


def test_probability_alignment() -> None:
    probabilities = align_probabilities(
        probabilities=[[0.8, 0.2]],
        source_classes=["A", "B"],
        target_classes=["B", "A", "C"],
    )
    assert probabilities.shape == (1, 3)
    assert abs(probabilities.sum() - 1.0) < 1e-9
    assert probabilities[0, 1] > probabilities[0, 0]


def test_delivery_ensemble_learns_nonnegative_simplex_weights() -> None:
    observed = np.tile([0, 0, 1, 1], 60)
    frame = pd.DataFrame(
        {"outcome": observed, "current_state": np.tile(["P", "L"], 120)}
    )
    good_positive = np.where(observed == 1, 0.9, 0.1)
    bad_positive = 1 - good_positive
    probability_map = {
        "good": np.column_stack([1 - good_positive, good_positive]),
        "bad": np.column_stack([1 - bad_positive, bad_positive]),
    }
    validation_probability, test_probability, weights = (
        state_conditioned_delivery_ensemble(
            frame,
            probability_map,
            frame,
            probability_map,
            "outcome",
            [0, 1],
            minimum_state_observations=20,
        )
    )
    assert validation_probability.shape == (len(frame), 2)
    assert np.allclose(validation_probability.sum(axis=1), 1.0)
    assert np.allclose(test_probability, validation_probability)
    assert np.allclose(weights[["good", "bad"]].sum(axis=1), 1.0)
    assert (weights[["good", "bad"]] >= 0).all().all()
    assert weights.loc[weights["state"] == "__GLOBAL__", "good"].iloc[0] > 0.99


def test_paired_bootstrap_reports_direction_of_difference() -> None:
    observed = np.tile([0, 1], 30)
    frame = pd.DataFrame(
        {
            "outcome": observed,
            "development_cluster_id": [f"c{index}" for index in range(len(observed))],
        }
    )
    proposed = np.where(observed == 1, 0.9, 0.1)
    comparator = np.full(len(observed), 0.5)
    result = paired_bootstrap_binary_comparisons(
        frame,
        "outcome",
        {"proposed": proposed, "comparator": comparator},
        "proposed",
        "development_cluster_id",
        repetitions=20,
        seed=7,
    )
    auc = result.loc[result["metric"] == "roc_auc"].iloc[0]
    brier = result.loc[result["metric"] == "brier"].iloc[0]
    assert auc["point_delta_proposed_minus_comparator"] > 0
    assert brier["point_delta_proposed_minus_comparator"] < 0


def test_sparse_moe_routing_and_probability_shapes() -> None:
    torch.manual_seed(11)
    reference_numeric = torch.randn(24, 2)
    bins = compute_bins(reference_numeric, n_bins=4)
    embeddings = PiecewiseLinearEmbeddings(
        bins, d_embedding=4, activation=False, version="B"
    )
    model = DeliverySparseMoENetwork(
        n_numeric=2,
        categorical_cardinalities=[3],
        classes=3,
        numerical_embeddings=embeddings,
        width=16,
        blocks=1,
        dropout=0.0,
        experts=4,
        top_k=2,
        prior_initial_scale=0.1,
    )
    numeric = torch.randn(5, 2)
    categorical = torch.tensor([[0], [1], [2], [0], [1]])
    log_prior = torch.log_softmax(torch.randn(5, 3), dim=1)
    logits, auxiliary, gates = model(numeric, categorical, log_prior)
    assert logits.shape == (5, 3)
    assert auxiliary.shape == (5, 5)
    assert gates.shape == (5, 4)
    assert torch.allclose(gates.sum(dim=1), torch.ones(5), atol=1e-6)
    assert torch.isfinite(logits).all()
