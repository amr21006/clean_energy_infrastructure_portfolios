from __future__ import annotations

import copy
import math
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from tabm import TabM

from .config import AnalysisConfig


EPSILON = 1e-7


def _prepare_arrays(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    categorical: Sequence[str],
    numeric: Sequence[str],
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    dict[str, Any],
]:
    active_numeric = [
        column
        for column in numeric
        if pd.to_numeric(train[column], errors="coerce").nunique(dropna=True) > 1
    ]
    dropped_numeric = [column for column in numeric if column not in active_numeric]
    mappings: dict[str, dict[str, int]] = {}
    cardinalities: list[int] = []
    categorical_arrays = [
        np.zeros((len(frame), len(categorical)), dtype=np.int64)
        for frame in (train, validation, test)
    ]
    for column_index, column in enumerate(categorical):
        levels = sorted(
            set(train[column].fillna("UNKNOWN").astype(str)) | {"UNKNOWN"}
        )
        mapping = {level: index for index, level in enumerate(levels)}
        mappings[column] = mapping
        cardinalities.append(len(levels))
        unknown = mapping["UNKNOWN"]
        for array, frame in zip(categorical_arrays, (train, validation, test)):
            array[:, column_index] = (
                frame[column]
                .fillna("UNKNOWN")
                .astype(str)
                .map(mapping)
                .fillna(unknown)
                .to_numpy(dtype=np.int64)
            )

    numeric_arrays = [
        np.zeros((len(frame), len(active_numeric)), dtype=np.float32)
        for frame in (train, validation, test)
    ]
    numeric_statistics: dict[str, dict[str, float]] = {}
    for column_index, column in enumerate(active_numeric):
        train_values = pd.to_numeric(train[column], errors="coerce")
        median = float(train_values.median()) if train_values.notna().any() else 0.0
        filled = train_values.fillna(median)
        mean = float(filled.mean())
        standard_deviation = float(filled.std(ddof=0))
        if not np.isfinite(standard_deviation) or standard_deviation < 1e-8:
            standard_deviation = 1.0
        numeric_statistics[column] = {
            "median": median,
            "mean": mean,
            "standard_deviation": standard_deviation,
        }
        for array, frame in zip(numeric_arrays, (train, validation, test)):
            values = pd.to_numeric(frame[column], errors="coerce").fillna(median)
            array[:, column_index] = (
                values.to_numpy(dtype=np.float32) - mean
            ) / standard_deviation
    metadata = {
        "categorical_columns": list(categorical),
        "categorical_mappings": mappings,
        "categorical_cardinalities": cardinalities,
        "numeric_columns": active_numeric,
        "dropped_constant_numeric_columns": dropped_numeric,
        "numeric_statistics": numeric_statistics,
    }
    return (
        (numeric_arrays[0], categorical_arrays[0]),
        (numeric_arrays[1], categorical_arrays[1]),
        (numeric_arrays[2], categorical_arrays[2]),
        metadata,
    )


def _embedding_dimensions(cardinalities: Sequence[int]) -> list[int]:
    return [min(24, max(4, int(math.ceil(math.sqrt(value) * 2)))) for value in cardinalities]


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(width)
        self.expand = nn.Linear(width, width * 2)
        self.contract = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.normalization(x)
        value, gate = self.expand(normalized).chunk(2, dim=-1)
        transformed = self.contract(F.gelu(value) * torch.sigmoid(gate))
        return x + self.dropout(transformed)


class DeliveryBackbone(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        categorical_cardinalities: Sequence[int],
        width: int,
        blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = _embedding_dimensions(categorical_cardinalities)
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dimension)
                for cardinality, dimension in zip(
                    categorical_cardinalities, dimensions
                )
            ]
        )
        self.input = nn.Sequential(
            nn.Linear(n_numeric + sum(dimensions), width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, dropout) for _ in range(blocks)]
        )
        self.output_normalization = nn.LayerNorm(width)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        x = self.input(torch.cat([numeric, *embedded], dim=1))
        for block in self.blocks:
            x = block(x)
        return self.output_normalization(x)


class DeliveryStateResidualNetwork(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        categorical_cardinalities: Sequence[int],
        classes: int,
        width: int,
        blocks: int,
        dropout: float,
        use_markov_prior: bool,
    ) -> None:
        super().__init__()
        self.backbone = DeliveryBackbone(
            n_numeric,
            categorical_cardinalities,
            width,
            blocks,
            dropout,
        )
        self.transition_head = nn.Linear(width, classes)
        self.auxiliary_head = nn.Linear(width, 5)
        self.use_markov_prior = use_markov_prior
        self.prior_scale_raw = nn.Parameter(torch.tensor(0.5413))

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        log_markov_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.backbone(numeric, categorical)
        logits = self.transition_head(representation)
        if self.use_markov_prior:
            logits = logits + F.softplus(self.prior_scale_raw) * log_markov_prior
        return logits, self.auxiliary_head(representation)


class DeliveryCompetingRiskNetwork(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        categorical_cardinalities: Sequence[int],
        width: int,
        blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = DeliveryBackbone(
            n_numeric,
            categorical_cardinalities,
            width,
            blocks,
            dropout,
        )
        self.risk_head = nn.Linear(width, 3)

    def forward(
        self, numeric: torch.Tensor, categorical: torch.Tensor
    ) -> torch.Tensor:
        return self.risk_head(self.backbone(numeric, categorical))


class DeliverySparseMoENetwork(nn.Module):
    """One end-to-end sparse MoE; it never pools comparator predictions."""

    def __init__(
        self,
        n_numeric: int,
        categorical_cardinalities: Sequence[int],
        classes: int,
        numerical_embeddings: nn.Module,
        width: int,
        blocks: int,
        dropout: float,
        experts: int = 4,
        top_k: int = 2,
        prior_initial_scale: float = 0.693147,
    ) -> None:
        super().__init__()
        self.encoder = TabM.make(
            n_num_features=n_numeric,
            cat_cardinalities=list(categorical_cardinalities),
            d_out=None,
            num_embeddings=numerical_embeddings,
            n_blocks=blocks,
            d_block=width,
            dropout=dropout,
            k=1,
            arch_type="tabm",
            start_scaling_init="normal",
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, width * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(width * 2, width),
                )
                for _ in range(experts)
            ]
        )
        self.gate = nn.Linear(width, experts)
        self.top_k = top_k
        self.transition_head = nn.Linear(width, classes)
        self.auxiliary_head = nn.Linear(width, 5)
        self.prior_scale_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(prior_initial_scale)))
        )

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        log_markov_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        representation = self.encoder(numeric, categorical)[:, 0, :]
        gate_probability = torch.softmax(self.gate(representation), dim=1)
        top_probability, top_index = torch.topk(
            gate_probability, k=self.top_k, dim=1
        )
        sparse_gate = torch.zeros_like(gate_probability).scatter(
            1, top_index, top_probability
        )
        sparse_gate = sparse_gate / sparse_gate.sum(dim=1, keepdim=True)
        expert_values = torch.stack(
            [expert(representation) for expert in self.experts], dim=1
        )
        mixed = representation + torch.sum(
            sparse_gate.unsqueeze(-1) * expert_values, dim=1
        )
        transition_logits = self.transition_head(mixed)
        transition_logits = (
            transition_logits
            + F.softplus(self.prior_scale_raw) * log_markov_prior
        )
        return transition_logits, self.auxiliary_head(mixed), gate_probability


def _weighted_markov_prior(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    classes: Sequence[str],
) -> np.ndarray:
    counts = train.pivot_table(
        index="current_state",
        columns="next_state",
        values="cluster_equalizing_weight",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(columns=classes, fill_value=0.0)
    global_counts = (
        train.groupby("next_state")["cluster_equalizing_weight"]
        .sum()
        .reindex(classes, fill_value=0.0)
    )
    global_probability = (global_counts + 1.0) / (
        global_counts.sum() + len(classes)
    )
    rows = []
    for state in frame["current_state"]:
        if state in counts.index:
            state_counts = counts.loc[state].astype(float) + 1.0
            rows.append((state_counts / state_counts.sum()).to_numpy())
        else:
            rows.append(global_probability.to_numpy())
    return np.log(np.clip(np.vstack(rows), EPSILON, 1.0)).astype(np.float32)


def _auxiliary_transition_target(frame: pd.DataFrame) -> np.ndarray:
    order = {state: index for index, state in enumerate(["P", "L", "T", "U", "V", "TS"])}
    values = []
    for current, next_state in zip(frame["current_state"], frame["next_state"]):
        if next_state == "C":
            values.append(3)
        elif next_state == "O":
            values.append(1)
        elif current == next_state:
            values.append(0)
        elif current in order and next_state in order and order[next_state] > order[current]:
            values.append(1)
        elif current in order and next_state in order and order[next_state] < order[current]:
            values.append(2)
        else:
            values.append(4)
    return np.asarray(values, dtype=np.int64)


def _state_network_predictions(
    model: DeliveryStateResidualNetwork,
    arrays: tuple[np.ndarray, np.ndarray],
    log_prior: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities = []
    numeric, categorical = arrays
    with torch.inference_mode():
        for start in range(0, len(numeric), 2048):
            logits, _ = model(
                torch.from_numpy(numeric[start : start + 2048]).to(device),
                torch.from_numpy(categorical[start : start + 2048]).to(device),
                torch.from_numpy(log_prior[start : start + 2048]).to(device),
            )
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def train_delivery_state_network(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    categorical: Sequence[str],
    numeric: Sequence[str],
    config: AnalysisConfig,
    use_markov_prior: bool = True,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("The delivery-state network requires CUDA.")
    classes = np.array(sorted(set(train["next_state"]) | set(validation["next_state"])))
    class_mapping = {label: index for index, label in enumerate(classes)}
    y_train = train["next_state"].map(class_mapping).to_numpy(dtype=np.int64)
    y_validation = validation["next_state"].map(class_mapping).to_numpy(dtype=np.int64)
    auxiliary_train = _auxiliary_transition_target(train)
    train_arrays, validation_arrays, test_arrays, preprocessing = _prepare_arrays(
        train, validation, test, categorical, numeric
    )
    train_prior = _weighted_markov_prior(train, train, classes)
    validation_prior = _weighted_markov_prior(train, validation, classes)
    test_prior = _weighted_markov_prior(train, test, classes)
    sample_weight = train["cluster_equalizing_weight"].to_numpy(dtype=np.float32)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_arrays[0]),
        torch.from_numpy(train_arrays[1]),
        torch.from_numpy(train_prior),
        torch.from_numpy(y_train),
        torch.from_numpy(auxiliary_train),
        torch.from_numpy(sample_weight),
    )
    device = torch.device("cuda:0")
    candidates = [
        {"width": 256, "blocks": 3, "dropout": 0.10, "learning_rate": 1e-3},
        {"width": 384, "blocks": 3, "dropout": 0.15, "learning_rate": 8e-4},
    ]
    tuning_rows = []
    best: dict[str, Any] | None = None
    for candidate_index, parameters in enumerate(candidates):
        seed = config.seed + 800 + candidate_index + (20 if not use_markov_prior else 0)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1024,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
            pin_memory=True,
        )
        model = DeliveryStateResidualNetwork(
            len(preprocessing["numeric_columns"]),
            preprocessing["categorical_cardinalities"],
            len(classes),
            parameters["width"],
            parameters["blocks"],
            parameters["dropout"],
            use_markov_prior,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=parameters["learning_rate"],
            weight_decay=1e-4,
        )
        best_state = None
        best_score = math.inf
        best_epoch = -1
        stale_epochs = 0
        start_time = time.perf_counter()
        for epoch in range(120):
            model.train()
            for x_num, x_cat, prior, target, auxiliary, weight in loader:
                x_num = x_num.to(device, non_blocking=True)
                x_cat = x_cat.to(device, non_blocking=True)
                prior = prior.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                auxiliary = auxiliary.to(device, non_blocking=True)
                weight = weight.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits, auxiliary_logits = model(x_num, x_cat, prior)
                primary_loss = F.cross_entropy(logits, target, reduction="none")
                auxiliary_loss = F.cross_entropy(
                    auxiliary_logits, auxiliary, reduction="none"
                )
                loss = ((primary_loss + 0.20 * auxiliary_loss) * weight).sum() / weight.sum()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            validation_probability = _state_network_predictions(
                model, validation_arrays, validation_prior, device
            )
            score = log_loss(
                y_validation,
                validation_probability,
                labels=list(range(len(classes))),
            )
            if score < best_score - 1e-5:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= 15:
                break
        fit_seconds = time.perf_counter() - start_time
        tuning_rows.append(
            {
                "candidate": candidate_index,
                **parameters,
                "use_markov_prior": use_markov_prior,
                "best_epoch": best_epoch,
                "epochs_completed": epoch + 1,
                "validation_log_loss": best_score,
                "fit_seconds": fit_seconds,
                "backend": "cuda_gpu",
            }
        )
        if best_state is not None and (
            best is None or best_score < best["validation_log_loss"]
        ):
            best = {
                "state_dict": best_state,
                "parameters": parameters,
                "seed": seed,
                "validation_log_loss": best_score,
                "fit_seconds": fit_seconds,
            }
        del model
        torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError("Delivery-state network failed to produce a checkpoint.")
    parameters = best["parameters"]
    model = DeliveryStateResidualNetwork(
        len(preprocessing["numeric_columns"]),
        preprocessing["categorical_cardinalities"],
        len(classes),
        parameters["width"],
        parameters["blocks"],
        parameters["dropout"],
        use_markov_prior,
    ).to(device)
    model.load_state_dict(best["state_dict"])
    validation_probability = _state_network_predictions(
        model, validation_arrays, validation_prior, device
    )
    test_start = time.perf_counter()
    test_probability = _state_network_predictions(model, test_arrays, test_prior, device)
    test_predict_seconds = time.perf_counter() - test_start
    artifact = {
        **best,
        "preprocessing": preprocessing,
        "classes": classes,
        "use_markov_prior": use_markov_prior,
        "markov_states": sorted(train["current_state"].astype(str).unique()),
        "markov_log_prior_by_state": {
            state: prior.tolist()
            for state, prior in zip(
                sorted(train["current_state"].astype(str).unique()),
                _weighted_markov_prior(
                    train,
                    pd.DataFrame(
                        {
                            "current_state": sorted(
                                train["current_state"].astype(str).unique()
                            )
                        }
                    ),
                    classes,
                ),
            )
        },
        "model_type": "DeliveryStateResidualNetwork",
    }
    tuning = pd.DataFrame(tuning_rows)
    tuning["test_predict_seconds_selected"] = test_predict_seconds
    del model
    torch.cuda.empty_cache()
    return artifact, validation_probability, test_probability, tuning, classes


def _moe_state_outputs(
    model: DeliverySparseMoENetwork,
    arrays: tuple[np.ndarray, np.ndarray],
    log_prior: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_parts = []
    gate_parts = []
    numeric, categorical = arrays
    with torch.inference_mode():
        for start in range(0, len(numeric), 2048):
            logits, _, gates = model(
                torch.from_numpy(numeric[start : start + 2048]).to(device),
                torch.from_numpy(categorical[start : start + 2048]).to(device),
                torch.from_numpy(log_prior[start : start + 2048]).to(device),
            )
            logits_parts.append(logits.cpu().numpy())
            gate_parts.append(gates.cpu().numpy())
    return np.concatenate(logits_parts), np.concatenate(gate_parts)


def _fit_temperature(logits: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        shifted = logits / temperature
        shifted -= shifted.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=1, keepdims=True)
        return float(
            log_loss(target, probability, labels=list(range(probability.shape[1])))
        )

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.25), math.log(4.0)),
        method="bounded",
        options={"xatol": 1e-4},
    )
    temperature = math.exp(float(result.x))
    return temperature, float(result.fun)


def _temperature_probability(logits: np.ndarray, temperature: float) -> np.ndarray:
    shifted = logits / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    return probability / probability.sum(axis=1, keepdims=True)


def train_delivery_sparse_moe_network(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    categorical: Sequence[str],
    numeric: Sequence[str],
    config: AnalysisConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Train a standalone sparse MoE and expose the test set once after selection."""
    if not torch.cuda.is_available():
        raise RuntimeError("The delivery sparse-MoE network requires CUDA.")
    classes = np.array(sorted(set(train["next_state"]) | set(validation["next_state"])))
    mapping = {label: index for index, label in enumerate(classes)}
    y_train = train["next_state"].map(mapping).to_numpy(dtype=np.int64)
    y_validation = validation["next_state"].map(mapping).to_numpy(dtype=np.int64)
    auxiliary_train = _auxiliary_transition_target(train)
    train_arrays, validation_arrays, test_arrays, preprocessing = _prepare_arrays(
        train, validation, test, categorical, numeric
    )
    train_prior = _weighted_markov_prior(train, train, classes)
    validation_prior = _weighted_markov_prior(train, validation, classes)
    test_prior = _weighted_markov_prior(train, test, classes)
    sample_weight = train["cluster_equalizing_weight"].to_numpy(dtype=np.float32)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_arrays[0]),
        torch.from_numpy(train_arrays[1]),
        torch.from_numpy(train_prior),
        torch.from_numpy(y_train),
        torch.from_numpy(auxiliary_train),
        torch.from_numpy(sample_weight),
    )
    bins = compute_bins(torch.from_numpy(train_arrays[0]), n_bins=32)
    device = torch.device("cuda:0")
    candidates = [
        {
            "width": 256,
            "blocks": 2,
            "dropout": 0.10,
            "learning_rate": 1e-3,
            "auxiliary_weight": 0.15,
            "balance_weight": 0.01,
            "prior_initial_scale": 0.693147,
        },
        {
            "width": 384,
            "blocks": 3,
            "dropout": 0.15,
            "learning_rate": 8e-4,
            "auxiliary_weight": 0.20,
            "balance_weight": 0.01,
            "prior_initial_scale": 0.693147,
        },
        {
            "width": 384,
            "blocks": 3,
            "dropout": 0.10,
            "learning_rate": 4e-4,
            "auxiliary_weight": 0.05,
            "balance_weight": 0.005,
            "prior_initial_scale": 0.10,
        },
        {
            "width": 512,
            "blocks": 2,
            "dropout": 0.05,
            "learning_rate": 3e-4,
            "auxiliary_weight": 0.0,
            "balance_weight": 0.005,
            "prior_initial_scale": 0.10,
        },
    ]
    tuning_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    total_fit_seconds = 0.0
    for candidate_index, parameters in enumerate(candidates):
        seed = config.seed + 950 + candidate_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=512,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
            pin_memory=True,
        )
        numerical_embeddings = PiecewiseLinearEmbeddings(
            bins, d_embedding=16, activation=False, version="B"
        )
        model = DeliverySparseMoENetwork(
            len(preprocessing["numeric_columns"]),
            preprocessing["categorical_cardinalities"],
            len(classes),
            numerical_embeddings,
            parameters["width"],
            parameters["blocks"],
            parameters["dropout"],
            experts=4,
            top_k=2,
            prior_initial_scale=parameters["prior_initial_scale"],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=parameters["learning_rate"], weight_decay=1e-4
        )
        best_state = None
        best_raw_loss = math.inf
        best_epoch = -1
        stale_epochs = 0
        start_time = time.perf_counter()
        for epoch in range(120):
            model.train()
            for x_num, x_cat, prior, target, auxiliary, weight in loader:
                x_num = x_num.to(device, non_blocking=True)
                x_cat = x_cat.to(device, non_blocking=True)
                prior = prior.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                auxiliary = auxiliary.to(device, non_blocking=True)
                weight = weight.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits, auxiliary_logits, gate_probability = model(
                    x_num, x_cat, prior
                )
                primary_loss = F.cross_entropy(logits, target, reduction="none")
                auxiliary_loss = F.cross_entropy(
                    auxiliary_logits, auxiliary, reduction="none"
                )
                mean_gate = gate_probability.mean(dim=0)
                balance_loss = 4.0 * torch.sum(mean_gate.square()) - 1.0
                loss = (
                    (
                        primary_loss
                        + parameters["auxiliary_weight"] * auxiliary_loss
                    )
                    * weight
                ).sum() / weight.sum() + parameters["balance_weight"] * balance_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            validation_logits, _ = _moe_state_outputs(
                model, validation_arrays, validation_prior, device
            )
            raw_probability = _temperature_probability(validation_logits, 1.0)
            raw_loss = log_loss(
                y_validation, raw_probability, labels=list(range(len(classes)))
            )
            if raw_loss < best_raw_loss - 1e-5:
                best_raw_loss = raw_loss
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= 15:
                break
        fit_seconds = time.perf_counter() - start_time
        total_fit_seconds += fit_seconds
        if best_state is None:
            del model
            torch.cuda.empty_cache()
            continue
        model.load_state_dict(best_state)
        validation_logits, validation_gates = _moe_state_outputs(
            model, validation_arrays, validation_prior, device
        )
        temperature, calibrated_loss = _fit_temperature(
            validation_logits, y_validation
        )
        utilization = validation_gates.mean(axis=0)
        tuning_rows.append(
            {
                "candidate": candidate_index,
                **parameters,
                "experts": 4,
                "top_k": 2,
                "best_epoch": best_epoch,
                "epochs_completed": epoch + 1,
                "validation_raw_log_loss": best_raw_loss,
                "validation_temperature": temperature,
                "validation_log_loss": calibrated_loss,
                "expert_utilization": " | ".join(f"{value:.6f}" for value in utilization),
                "minimum_expert_utilization": float(utilization.min()),
                "fit_seconds": fit_seconds,
                "backend": "cuda_gpu",
            }
        )
        if selected is None or calibrated_loss < selected["validation_log_loss"]:
            selected = {
                "state_dict": best_state,
                "parameters": parameters,
                "seed": seed,
                "temperature": temperature,
                "validation_log_loss": calibrated_loss,
                "validation_raw_log_loss": best_raw_loss,
                "validation_gate_utilization": utilization,
            }
        del model
        torch.cuda.empty_cache()
    if selected is None:
        raise RuntimeError("Delivery sparse-MoE failed to produce a checkpoint.")
    parameters = selected["parameters"]
    numerical_embeddings = PiecewiseLinearEmbeddings(
        bins, d_embedding=16, activation=False, version="B"
    )
    model = DeliverySparseMoENetwork(
        len(preprocessing["numeric_columns"]),
        preprocessing["categorical_cardinalities"],
        len(classes),
        numerical_embeddings,
        parameters["width"],
        parameters["blocks"],
        parameters["dropout"],
        experts=4,
        top_k=2,
        prior_initial_scale=parameters["prior_initial_scale"],
    ).to(device)
    model.load_state_dict(selected["state_dict"])
    validation_logits, validation_gates = _moe_state_outputs(
        model, validation_arrays, validation_prior, device
    )
    validation_probability = _temperature_probability(
        validation_logits, selected["temperature"]
    )
    test_start = time.perf_counter()
    test_logits, test_gates = _moe_state_outputs(
        model, test_arrays, test_prior, device
    )
    test_probability = _temperature_probability(test_logits, selected["temperature"])
    test_predict_seconds = time.perf_counter() - test_start
    artifact = {
        **selected,
        "fit_seconds": total_fit_seconds,
        "test_predict_seconds": test_predict_seconds,
        "preprocessing": preprocessing,
        "classes": classes,
        "piecewise_linear_bins": [value.cpu() for value in bins],
        "numeric_embedding_size": 16,
        "markov_states": sorted(train["current_state"].astype(str).unique()),
        "markov_log_prior_by_state": {
            state: prior.tolist()
            for state, prior in zip(
                sorted(train["current_state"].astype(str).unique()),
                _weighted_markov_prior(
                    train,
                    pd.DataFrame(
                        {
                            "current_state": sorted(
                                train["current_state"].astype(str).unique()
                            )
                        }
                    ),
                    classes,
                ),
            )
        },
        "validation_gate_utilization": validation_gates.mean(axis=0),
        "test_gate_utilization": test_gates.mean(axis=0),
        "model_type": "DeliverySparseMoENetwork",
        "no_comparator_prediction_inputs": True,
    }
    tuning = pd.DataFrame(tuning_rows)
    tuning["selected"] = tuning["candidate"].eq(
        int(tuning.loc[tuning["validation_log_loss"].idxmin(), "candidate"])
    )
    tuning["total_search_fit_seconds"] = total_fit_seconds
    tuning["test_predict_seconds_selected"] = test_predict_seconds
    del model
    torch.cuda.empty_cache()
    return artifact, validation_probability, test_probability, tuning, classes


def _transform_with_artifact(
    frame: pd.DataFrame, preprocessing: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    categorical_columns = preprocessing["categorical_columns"]
    numeric_columns = preprocessing["numeric_columns"]
    categorical = np.zeros((len(frame), len(categorical_columns)), dtype=np.int64)
    for index, column in enumerate(categorical_columns):
        mapping = preprocessing["categorical_mappings"][column]
        unknown = mapping["UNKNOWN"]
        categorical[:, index] = (
            frame[column]
            .fillna("UNKNOWN")
            .astype(str)
            .map(mapping)
            .fillna(unknown)
            .to_numpy(dtype=np.int64)
        )
    numeric = np.zeros((len(frame), len(numeric_columns)), dtype=np.float32)
    for index, column in enumerate(numeric_columns):
        statistics = preprocessing["numeric_statistics"][column]
        values = pd.to_numeric(frame[column], errors="coerce").fillna(
            statistics["median"]
        )
        numeric[:, index] = (
            values.to_numpy(dtype=np.float32) - statistics["mean"]
        ) / statistics["standard_deviation"]
    return numeric, categorical


def _restore_delivery_sparse_moe(
    artifact: dict[str, Any], device: torch.device
) -> DeliverySparseMoENetwork:
    parameters = artifact["parameters"]
    embeddings = PiecewiseLinearEmbeddings(
        artifact["piecewise_linear_bins"],
        d_embedding=artifact["numeric_embedding_size"],
        activation=False,
        version="B",
    )
    model = DeliverySparseMoENetwork(
        len(artifact["preprocessing"]["numeric_columns"]),
        artifact["preprocessing"]["categorical_cardinalities"],
        len(artifact["classes"]),
        embeddings,
        parameters["width"],
        parameters["blocks"],
        parameters["dropout"],
        experts=4,
        top_k=2,
        prior_initial_scale=parameters["prior_initial_scale"],
    ).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model


def _artifact_moe_probability(
    model: DeliverySparseMoENetwork,
    artifact: dict[str, Any],
    frame: pd.DataFrame,
    device: torch.device,
) -> np.ndarray:
    arrays = _transform_with_artifact(frame, artifact["preprocessing"])
    classes = list(artifact["classes"])
    prior_lookup = artifact["markov_log_prior_by_state"]
    fallback = np.full(len(classes), -math.log(len(classes)), dtype=np.float32)
    log_prior = np.vstack(
        [
            np.asarray(prior_lookup.get(str(state), fallback), dtype=np.float32)
            for state in frame["current_state"]
        ]
    )
    logits, _ = _moe_state_outputs(model, arrays, log_prior, device)
    return _temperature_probability(logits, artifact["temperature"])


def rollout_delivery_path_moe(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
    operation_horizon: int = 12,
    cancellation_horizon: int = 24,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Roll one transition law forward while retaining state-duration path mass."""
    if not torch.cuda.is_available():
        raise RuntimeError("Delivery-PathMoE rollout requires CUDA.")
    classes = [str(value) for value in artifact["classes"]]
    if "O" not in classes or "C" not in classes:
        raise RuntimeError("Delivery-PathMoE requires operation and cancellation states.")
    device = torch.device("cuda:0")
    model = _restore_delivery_sparse_moe(artifact, device)
    base = frame.reset_index(drop=True).copy()
    observations = len(base)
    initial_state = base["current_state"].fillna("OTHER").astype(str).to_numpy()
    initial_duration = (
        pd.to_numeric(base["months_in_current_state"], errors="coerce")
        .fillna(1)
        .clip(lower=1)
        .astype(int)
        .to_numpy()
    )
    paths: dict[tuple[str, int], np.ndarray] = {}
    for state, duration in sorted(set(zip(initial_state, initial_duration))):
        paths[(state, int(duration))] = (
            (initial_state == state) & (initial_duration == duration)
        ).astype(np.float64)
    maximum_horizon = max(operation_horizon, cancellation_horizon)
    operation_probability = np.zeros(observations, dtype=np.float64)
    cancellation_probability = np.zeros(observations, dtype=np.float64)
    maximum_mass_error = 0.0
    start_time = time.perf_counter()
    for step in range(1, maximum_horizon + 1):
        next_paths: dict[tuple[str, int], np.ndarray] = {}
        prediction_frames: list[pd.DataFrame] = []
        specifications: list[tuple[str, int, np.ndarray, np.ndarray, int]] = []
        cursor = 0
        for (state, duration), mass in paths.items():
            if state in {"O", "C"}:
                key = (state, 1)
                next_paths[key] = next_paths.get(
                    key, np.zeros(observations, dtype=np.float64)
                ) + mass
                continue
            indices = np.flatnonzero(mass > 0.0)
            if not len(indices):
                continue
            prediction_frame = base.iloc[indices].copy()
            prediction_frame["current_state"] = state
            prediction_frame["months_in_current_state"] = duration
            elapsed = step - 1
            prediction_frame["months_since_first_report"] = (
                pd.to_numeric(
                    prediction_frame["months_since_first_report"], errors="coerce"
                ).fillna(0)
                + elapsed
            )
            prediction_frame["months_to_reported_operation"] = (
                pd.to_numeric(
                    prediction_frame["months_to_reported_operation"], errors="coerce"
                )
                - elapsed
            )
            prediction_frames.append(prediction_frame)
            specifications.append((state, duration, indices, mass[indices], cursor))
            cursor += len(indices)
        if prediction_frames:
            combined = pd.concat(prediction_frames, ignore_index=True)
            probability = _artifact_moe_probability(model, artifact, combined, device)
            for state, duration, indices, mass, start in specifications:
                stop = start + len(indices)
                group_probability = probability[start:stop]
                for class_index, next_state in enumerate(classes):
                    next_duration = duration + 1 if next_state == state else 1
                    key = (next_state, next_duration)
                    if key not in next_paths:
                        next_paths[key] = np.zeros(observations, dtype=np.float64)
                    next_paths[key][indices] += mass * group_probability[:, class_index]
        paths = next_paths
        total_mass = np.sum(np.stack(list(paths.values())), axis=0)
        maximum_mass_error = max(
            maximum_mass_error, float(np.max(np.abs(total_mass - 1.0)))
        )
        if step == operation_horizon:
            operation_probability = np.sum(
                np.stack(
                    [mass for (state, _), mass in paths.items() if state == "O"]
                ),
                axis=0,
            )
        if step == cancellation_horizon:
            cancellation_probability = np.sum(
                np.stack(
                    [mass for (state, _), mass in paths.items() if state == "C"]
                ),
                axis=0,
            )
    predict_seconds = time.perf_counter() - start_time
    del model
    torch.cuda.empty_cache()
    if maximum_mass_error > 1e-5:
        raise RuntimeError(
            f"Delivery-PathMoE probability mass error is {maximum_mass_error:.3g}."
        )
    return (
        {
            "operation_12m": operation_probability,
            "cancellation_24m": cancellation_probability,
        },
        {
            "observations": observations,
            "operation_horizon": operation_horizon,
            "cancellation_horizon": cancellation_horizon,
            "maximum_probability_mass_error": maximum_mass_error,
            "predict_seconds": predict_seconds,
            "backend": "cuda_gpu",
            "future_cumulative_features_frozen": True,
            "deterministic_clock_features_advanced": True,
        },
    )


def train_delivery_sparse_moe_risk_network(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    categorical: Sequence[str],
    numeric: Sequence[str],
    config: AnalysisConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    """Apply the same standalone sparse-MoE family to competing landmarks."""
    if not torch.cuda.is_available():
        raise RuntimeError("The delivery sparse-MoE risk network requires CUDA.")

    def target(frame: pd.DataFrame) -> np.ndarray:
        overlap = frame["operation_within_12_months"].astype(bool) & frame[
            "cancellation_within_24_months"
        ].astype(bool)
        if overlap.any():
            raise RuntimeError(f"Competing-risk labels overlap for {int(overlap.sum())} rows.")
        return np.select(
            [
                frame["operation_within_12_months"].astype(bool),
                frame["cancellation_within_24_months"].astype(bool),
            ],
            [1, 2],
            default=0,
        ).astype(np.int64)

    y_train = target(train)
    y_validation = target(validation)
    train_arrays, validation_arrays, test_arrays, preprocessing = _prepare_arrays(
        train, validation, test, categorical, numeric
    )
    sample_weight = train["cluster_equalizing_weight"].to_numpy(dtype=np.float32)
    zero_prior_train = np.zeros((len(train), 3), dtype=np.float32)
    zero_prior_validation = np.zeros((len(validation), 3), dtype=np.float32)
    zero_prior_test = np.zeros((len(test), 3), dtype=np.float32)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_arrays[0]),
        torch.from_numpy(train_arrays[1]),
        torch.from_numpy(zero_prior_train),
        torch.from_numpy(y_train),
        torch.from_numpy(sample_weight),
    )
    bins = compute_bins(torch.from_numpy(train_arrays[0]), n_bins=32)
    cancellation_events = max(int((y_train == 2).sum()), 1)
    rare_weight = math.sqrt((len(y_train) - cancellation_events) / cancellation_events)
    candidates = [
        {
            "width": 128,
            "blocks": 2,
            "dropout": 0.10,
            "learning_rate": 8e-4,
            "cancellation_weight": 1.0,
        },
        {
            "width": 256,
            "blocks": 2,
            "dropout": 0.15,
            "learning_rate": 5e-4,
            "cancellation_weight": rare_weight,
        },
    ]
    device = torch.device("cuda:0")
    selected: dict[str, Any] | None = None
    tuning_rows: list[dict[str, Any]] = []
    total_fit_seconds = 0.0
    for candidate_index, parameters in enumerate(candidates):
        seed = config.seed + 980 + candidate_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=512,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
            pin_memory=True,
        )
        numerical_embeddings = PiecewiseLinearEmbeddings(
            bins, d_embedding=16, activation=False, version="B"
        )
        model = DeliverySparseMoENetwork(
            len(preprocessing["numeric_columns"]),
            preprocessing["categorical_cardinalities"],
            3,
            numerical_embeddings,
            parameters["width"],
            parameters["blocks"],
            parameters["dropout"],
            experts=4,
            top_k=2,
            prior_initial_scale=0.10,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=parameters["learning_rate"], weight_decay=1e-4
        )
        class_weight = torch.tensor(
            [1.0, 1.0, parameters["cancellation_weight"]], device=device
        )
        best_state = None
        best_score = math.inf
        best_operation_loss = math.inf
        best_cancellation_loss = math.inf
        best_epoch = -1
        stale_epochs = 0
        start_time = time.perf_counter()
        for epoch in range(160):
            model.train()
            for x_num, x_cat, prior, batch_target, weight in loader:
                x_num = x_num.to(device, non_blocking=True)
                x_cat = x_cat.to(device, non_blocking=True)
                prior = prior.to(device, non_blocking=True)
                batch_target = batch_target.to(device, non_blocking=True)
                weight = weight.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits, _, gate_probability = model(x_num, x_cat, prior)
                primary_loss = F.cross_entropy(
                    logits, batch_target, weight=class_weight, reduction="none"
                )
                mean_gate = gate_probability.mean(dim=0)
                balance_loss = 4.0 * torch.sum(mean_gate.square()) - 1.0
                loss = (primary_loss * weight).sum() / weight.sum() + 0.005 * balance_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            validation_logits, _ = _moe_state_outputs(
                model, validation_arrays, zero_prior_validation, device
            )
            validation_probability = _temperature_probability(validation_logits, 1.0)
            operation_loss = log_loss(
                validation["operation_within_12_months"],
                validation_probability[:, 1],
                labels=[0, 1],
            )
            cancellation_loss = log_loss(
                validation["cancellation_within_24_months"],
                validation_probability[:, 2],
                labels=[0, 1],
            )
            score = 0.5 * (operation_loss + cancellation_loss)
            if score < best_score - 1e-5:
                best_score = score
                best_operation_loss = operation_loss
                best_cancellation_loss = cancellation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= 20:
                break
        fit_seconds = time.perf_counter() - start_time
        total_fit_seconds += fit_seconds
        if best_state is None:
            del model
            torch.cuda.empty_cache()
            continue
        model.load_state_dict(best_state)
        _, validation_gates = _moe_state_outputs(
            model, validation_arrays, zero_prior_validation, device
        )
        utilization = validation_gates.mean(axis=0)
        tuning_rows.append(
            {
                "candidate": candidate_index,
                **parameters,
                "experts": 4,
                "top_k": 2,
                "best_epoch": best_epoch,
                "epochs_completed": epoch + 1,
                "validation_joint_log_loss": best_score,
                "validation_operation_log_loss": best_operation_loss,
                "validation_cancellation_log_loss": best_cancellation_loss,
                "expert_utilization": " | ".join(f"{value:.6f}" for value in utilization),
                "minimum_expert_utilization": float(utilization.min()),
                "fit_seconds": fit_seconds,
                "backend": "cuda_gpu",
            }
        )
        if selected is None or best_score < selected["validation_joint_log_loss"]:
            selected = {
                "state_dict": best_state,
                "parameters": parameters,
                "seed": seed,
                "validation_joint_log_loss": best_score,
                "validation_operation_log_loss": best_operation_loss,
                "validation_cancellation_log_loss": best_cancellation_loss,
            }
        del model
        torch.cuda.empty_cache()
    if selected is None:
        raise RuntimeError("Delivery sparse-MoE risk network produced no checkpoint.")
    parameters = selected["parameters"]
    numerical_embeddings = PiecewiseLinearEmbeddings(
        bins, d_embedding=16, activation=False, version="B"
    )
    model = DeliverySparseMoENetwork(
        len(preprocessing["numeric_columns"]),
        preprocessing["categorical_cardinalities"],
        3,
        numerical_embeddings,
        parameters["width"],
        parameters["blocks"],
        parameters["dropout"],
        experts=4,
        top_k=2,
        prior_initial_scale=0.10,
    ).to(device)
    model.load_state_dict(selected["state_dict"])
    validation_logits, validation_gates = _moe_state_outputs(
        model, validation_arrays, zero_prior_validation, device
    )
    validation_probability = _temperature_probability(validation_logits, 1.0)
    test_start = time.perf_counter()
    test_logits, test_gates = _moe_state_outputs(
        model, test_arrays, zero_prior_test, device
    )
    test_probability = _temperature_probability(test_logits, 1.0)
    test_predict_seconds = time.perf_counter() - test_start
    artifact = {
        **selected,
        "fit_seconds": total_fit_seconds,
        "test_predict_seconds": test_predict_seconds,
        "preprocessing": preprocessing,
        "piecewise_linear_bins": [value.cpu() for value in bins],
        "numeric_embedding_size": 16,
        "validation_gate_utilization": validation_gates.mean(axis=0),
        "test_gate_utilization": test_gates.mean(axis=0),
        "model_type": "DeliverySparseMoERiskNetwork",
        "no_comparator_prediction_inputs": True,
    }
    tuning = pd.DataFrame(tuning_rows)
    tuning["selected"] = tuning["candidate"].eq(
        int(tuning.loc[tuning["validation_joint_log_loss"].idxmin(), "candidate"])
    )
    tuning["total_search_fit_seconds"] = total_fit_seconds
    tuning["test_predict_seconds_selected"] = test_predict_seconds
    del model
    torch.cuda.empty_cache()
    return (
        artifact,
        {
            "operation_12m": validation_probability[:, 1],
            "cancellation_24m": validation_probability[:, 2],
        },
        {
            "operation_12m": test_probability[:, 1],
            "cancellation_24m": test_probability[:, 2],
        },
        tuning,
    )


def _risk_predictions(
    model: DeliveryCompetingRiskNetwork,
    arrays: tuple[np.ndarray, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities = []
    numeric, categorical = arrays
    with torch.inference_mode():
        for start in range(0, len(numeric), 2048):
            logits = model(
                torch.from_numpy(numeric[start : start + 2048]).to(device),
                torch.from_numpy(categorical[start : start + 2048]).to(device),
            )
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def train_delivery_competing_risk_network(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    categorical: Sequence[str],
    numeric: Sequence[str],
    config: AnalysisConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    if not torch.cuda.is_available():
        raise RuntimeError("The competing-risk network requires CUDA.")
    overlap = int(
        (
            train["operation_within_12_months"].astype(bool)
            & train["cancellation_within_24_months"].astype(bool)
        ).sum()
    )
    if overlap:
        raise RuntimeError(f"Competing-risk labels overlap for {overlap} training rows.")

    def target(frame: pd.DataFrame) -> np.ndarray:
        return np.select(
            [
                frame["operation_within_12_months"].astype(bool),
                frame["cancellation_within_24_months"].astype(bool),
            ],
            [1, 2],
            default=0,
        ).astype(np.int64)

    y_train = target(train)
    y_validation = target(validation)
    train_arrays, validation_arrays, test_arrays, preprocessing = _prepare_arrays(
        train, validation, test, categorical, numeric
    )
    sample_weight = train["cluster_equalizing_weight"].to_numpy(dtype=np.float32)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_arrays[0]),
        torch.from_numpy(train_arrays[1]),
        torch.from_numpy(y_train),
        torch.from_numpy(sample_weight),
    )
    device = torch.device("cuda:0")
    cancellation_events = max(int((y_train == 2).sum()), 1)
    rare_weight = math.sqrt((len(y_train) - cancellation_events) / cancellation_events)
    candidates = [
        {"width": 256, "blocks": 3, "dropout": 0.10, "learning_rate": 1e-3, "cancellation_weight": 1.0},
        {"width": 256, "blocks": 3, "dropout": 0.15, "learning_rate": 8e-4, "cancellation_weight": rare_weight},
    ]
    tuning_rows = []
    best: dict[str, Any] | None = None
    for candidate_index, parameters in enumerate(candidates):
        seed = config.seed + 900 + candidate_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=512,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
            pin_memory=True,
        )
        model = DeliveryCompetingRiskNetwork(
            len(preprocessing["numeric_columns"]),
            preprocessing["categorical_cardinalities"],
            parameters["width"],
            parameters["blocks"],
            parameters["dropout"],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=parameters["learning_rate"],
            weight_decay=1e-4,
        )
        class_weights = torch.tensor(
            [1.0, 1.0, parameters["cancellation_weight"]],
            device=device,
        )
        best_state = None
        best_score = math.inf
        best_operation_loss = math.inf
        best_cancellation_loss = math.inf
        best_epoch = -1
        stale_epochs = 0
        start_time = time.perf_counter()
        for epoch in range(160):
            model.train()
            for x_num, x_cat, batch_y, weight in loader:
                x_num = x_num.to(device, non_blocking=True)
                x_cat = x_cat.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                weight = weight.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_num, x_cat)
                losses = F.cross_entropy(
                    logits, batch_y, weight=class_weights, reduction="none"
                )
                loss = (losses * weight).sum() / weight.sum()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            validation_probability = _risk_predictions(
                model, validation_arrays, device
            )
            operation_loss = log_loss(
                validation["operation_within_12_months"],
                validation_probability[:, 1],
                labels=[0, 1],
            )
            cancellation_loss = log_loss(
                validation["cancellation_within_24_months"],
                validation_probability[:, 2],
                labels=[0, 1],
            )
            score = 0.5 * (operation_loss + cancellation_loss)
            if score < best_score - 1e-5:
                best_score = score
                best_operation_loss = operation_loss
                best_cancellation_loss = cancellation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= 20:
                break
        fit_seconds = time.perf_counter() - start_time
        tuning_rows.append(
            {
                "candidate": candidate_index,
                **parameters,
                "best_epoch": best_epoch,
                "epochs_completed": epoch + 1,
                "validation_joint_log_loss": best_score,
                "validation_operation_log_loss": best_operation_loss,
                "validation_cancellation_log_loss": best_cancellation_loss,
                "fit_seconds": fit_seconds,
                "backend": "cuda_gpu",
            }
        )
        if best_state is not None and (
            best is None or best_score < best["validation_joint_log_loss"]
        ):
            best = {
                "state_dict": best_state,
                "parameters": parameters,
                "seed": seed,
                "validation_joint_log_loss": best_score,
                "fit_seconds": fit_seconds,
            }
        del model
        torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError("Competing-risk network failed to produce a checkpoint.")
    parameters = best["parameters"]
    model = DeliveryCompetingRiskNetwork(
        len(preprocessing["numeric_columns"]),
        preprocessing["categorical_cardinalities"],
        parameters["width"],
        parameters["blocks"],
        parameters["dropout"],
    ).to(device)
    model.load_state_dict(best["state_dict"])
    validation_probability = _risk_predictions(model, validation_arrays, device)
    test_start = time.perf_counter()
    test_probability = _risk_predictions(model, test_arrays, device)
    test_predict_seconds = time.perf_counter() - test_start
    artifact = {
        **best,
        "preprocessing": preprocessing,
        "classes": ["neither", "operation_12m", "cancellation_24m"],
        "model_type": "DeliveryCompetingRiskNetwork",
        "label_overlap": overlap,
    }
    tuning = pd.DataFrame(tuning_rows)
    tuning["test_predict_seconds_selected"] = test_predict_seconds
    del model
    torch.cuda.empty_cache()
    return (
        artifact,
        {
            "operation_12m": validation_probability[:, 1],
            "cancellation_24m": validation_probability[:, 2],
        },
        {
            "operation_12m": test_probability[:, 1],
            "cancellation_24m": test_probability[:, 2],
        },
        tuning,
    )
