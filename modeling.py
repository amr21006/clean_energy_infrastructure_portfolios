from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import torch
import torch.nn.functional as torch_functional
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.optimize import minimize
from scipy.special import expit, logit
from tabm import TabM
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import AnalysisConfig
from .deep_models import (
    rollout_delivery_path_moe,
    train_delivery_competing_risk_network,
    train_delivery_sparse_moe_network,
    train_delivery_sparse_moe_risk_network,
    train_delivery_state_network,
)


EPSILON = 1e-6


@dataclass
class ModelResult:
    name: str
    probabilities: np.ndarray
    classes: np.ndarray
    model: Any
    backend: str = "cpu"
    fit_seconds: float = math.nan
    predict_seconds: float = math.nan


def feature_columns(config: AnalysisConfig) -> tuple[list[str], list[str]]:
    return list(config.stable_categorical_features), list(config.numeric_features)


def _prepare_features(
    frame: pd.DataFrame, categorical: Sequence[str], numeric: Sequence[str]
) -> pd.DataFrame:
    result = frame.loc[:, [*categorical, *numeric]].copy()
    for column in categorical:
        result[column] = result[column].fillna("UNKNOWN").astype(str)
        result.loc[result[column].str.strip() == "", column] = "UNKNOWN"
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _logistic_pipeline(
    categorical: Sequence[str], numeric: Sequence[str], multiclass: bool = False
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=5,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                list(categorical),
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(numeric),
            ),
        ]
    )
    classifier = LogisticRegression(
        solver="lbfgs",
        max_iter=1500,
        C=1.0,
        class_weight=None,
        n_jobs=None,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _tree_pipeline(
    categorical: Sequence[str],
    numeric: Sequence[str],
    algorithm: str,
    seed: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                list(categorical),
            ),
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                list(numeric),
            ),
        ]
    )
    if algorithm == "random_forest":
        classifier: Any = RandomForestClassifier(
            n_estimators=160,
            max_features="sqrt",
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed,
        )
    elif algorithm == "extra_trees":
        classifier = ExtraTreesClassifier(
            n_estimators=160,
            max_features="sqrt",
            max_depth=24,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed,
        )
    elif algorithm == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=100,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported classic tree algorithm: {algorithm}")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _prepare_lightgbm_frames(
    train: pd.DataFrame,
    other_frames: Sequence[pd.DataFrame],
    categorical: Sequence[str],
    numeric: Sequence[str],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    train_result = train.copy()
    results = [frame.copy() for frame in other_frames]
    for column in categorical:
        train_values = train_result[column].fillna("UNKNOWN").astype(str)
        categories = sorted(set(train_values) | {"UNKNOWN"})
        train_result[column] = pd.Categorical(train_values, categories=categories)
        for frame in results:
            values = frame[column].fillna("UNKNOWN").astype(str)
            values = values.where(values.isin(categories), "UNKNOWN")
            frame[column] = pd.Categorical(values, categories=categories)
    for column in numeric:
        median = pd.to_numeric(train_result[column], errors="coerce").median()
        if pd.isna(median):
            median = 0.0
        train_result[column] = pd.to_numeric(
            train_result[column], errors="coerce"
        ).fillna(median)
        for frame in results:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(
                median
            )
    return train_result, results


def multiclass_brier(y_true: Sequence[str], probabilities: np.ndarray, classes: Sequence[str]) -> float:
    class_to_index = {label: index for index, label in enumerate(classes)}
    observed = np.zeros_like(probabilities, dtype=float)
    for row, label in enumerate(y_true):
        if label in class_to_index:
            observed[row, class_to_index[label]] = 1.0
    return float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))


def multiclass_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    classes: Sequence[str],
    sample_weight: pd.Series | None = None,
) -> dict[str, float]:
    classes_array = np.asarray(classes)
    predicted = classes_array[np.argmax(probabilities, axis=1)]
    result = {
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "macro_f1": f1_score(y_true, predicted, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true, predicted, average="weighted", zero_division=0
        ),
        "log_loss": log_loss(y_true, probabilities, labels=list(classes_array)),
        "multiclass_brier": multiclass_brier(y_true, probabilities, classes_array),
    }
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        class_to_index = {label: index for index, label in enumerate(classes_array)}
        observed = np.zeros_like(probabilities, dtype=float)
        for row, label in enumerate(y_true):
            observed[row, class_to_index[label]] = 1.0
        row_brier = np.sum((probabilities - observed) ** 2, axis=1)
        result.update(
            {
                "cluster_weighted_accuracy": accuracy_score(
                    y_true, predicted, sample_weight=weights
                ),
                "cluster_weighted_macro_f1": f1_score(
                    y_true,
                    predicted,
                    average="macro",
                    zero_division=0,
                    sample_weight=weights,
                ),
                "cluster_weighted_log_loss": log_loss(
                    y_true,
                    probabilities,
                    labels=list(classes_array),
                    sample_weight=weights,
                ),
                "cluster_weighted_multiclass_brier": float(
                    np.average(row_brier, weights=weights)
                ),
            }
        )
    return result


def per_class_metrics(
    y_true: pd.Series, probabilities: np.ndarray, classes: Sequence[str], model_name: str
) -> pd.DataFrame:
    classes_array = np.asarray(classes)
    predicted = classes_array[np.argmax(probabilities, axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=classes_array, zero_division=0
    )
    return pd.DataFrame(
        {
            "model": model_name,
            "class": classes_array,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def markov_probabilities(
    train: pd.DataFrame, target: pd.DataFrame, classes: Sequence[str]
) -> np.ndarray:
    if "cluster_equalizing_weight" in train:
        counts = train.pivot_table(
            index="current_state",
            columns="next_state",
            values="cluster_equalizing_weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        global_counts = train.groupby("next_state")[
            "cluster_equalizing_weight"
        ].sum()
    else:
        counts = pd.crosstab(train["current_state"], train["next_state"])
        global_counts = train["next_state"].value_counts()
    counts = counts.reindex(columns=classes, fill_value=0)
    global_counts = global_counts.reindex(classes, fill_value=0)
    global_probability = (global_counts + 1) / (global_counts.sum() + len(classes))
    rows: list[np.ndarray] = []
    for state in target["current_state"]:
        if state in counts.index:
            state_counts = counts.loc[state].astype(float) + 1.0
            rows.append((state_counts / state_counts.sum()).to_numpy())
        else:
            rows.append(global_probability.to_numpy())
    return np.vstack(rows)


def _candidate_boosting_parameters(seed: int, count: int) -> list[dict[str, Any]]:
    del seed
    candidates = [
        {
            "num_leaves": 31,
            "learning_rate": 0.08,
            "min_child_samples": 40,
            "reg_lambda": 3.0,
            "feature_fraction": 0.9,
        },
        {
            "num_leaves": 63,
            "learning_rate": 0.05,
            "min_child_samples": 75,
            "reg_lambda": 6.0,
            "feature_fraction": 0.9,
        },
    ]
    return candidates[:count]


def tune_lightgbm(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    categorical: Sequence[str],
    config: AnalysisConfig,
    binary: bool,
    positive_weight: float | None = None,
    sample_weight: pd.Series | None = None,
) -> tuple[LGBMClassifier, pd.DataFrame]:
    candidate_count = min(config.random_search_iterations, 2)
    candidates = _candidate_boosting_parameters(
        config.seed + (11 if binary else 0), candidate_count
    )
    tuning_rows: list[dict[str, Any]] = []
    best_model: LGBMClassifier | None = None
    best_score = math.inf
    sample = train_x
    sampled_y = train_y
    sampled_weight = sample_weight
    for index, parameters in enumerate(candidates):
        model_parameters: dict[str, Any] = {
            "n_estimators": 300,
            "objective": "binary" if binary else "multiclass",
            "random_state": config.seed + index,
            "n_jobs": -1,
            "verbosity": -1,
            "subsample": 0.9,
            "colsample_bytree": parameters["feature_fraction"],
            **{key: value for key, value in parameters.items() if key != "feature_fraction"},
        }
        if binary and positive_weight is not None:
            model_parameters["scale_pos_weight"] = positive_weight
        model = LGBMClassifier(**model_parameters)
        fit_start = time.perf_counter()
        model.fit(
            sample,
            sampled_y,
            sample_weight=sampled_weight,
            eval_set=(validation_x, validation_y),
            eval_metric="binary_logloss" if binary else "multi_logloss",
            categorical_feature=list(categorical),
            callbacks=[early_stopping(40, verbose=False), log_evaluation(0)],
        )
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        probabilities = model.predict_proba(validation_x)
        validation_predict_seconds = time.perf_counter() - predict_start
        score = log_loss(
            validation_y,
            probabilities,
            labels=list(model.classes_),
        )
        tuning_rows.append(
            {
                "candidate": index,
                **parameters,
                "best_iteration": int(model.best_iteration_),
                "validation_log_loss": score,
                "fit_seconds": fit_seconds,
                "validation_predict_seconds": validation_predict_seconds,
                "backend": "cpu",
            }
        )
        if score < best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError("LightGBM tuning failed to produce a model.")
    return best_model, pd.DataFrame(tuning_rows)


def _catboost_candidates(binary: bool) -> list[dict[str, Any]]:
    if binary:
        return [
            {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 4.0},
            {"depth": 8, "learning_rate": 0.035, "l2_leaf_reg": 7.0},
        ]
    return [
        {"depth": 6, "learning_rate": 0.06, "l2_leaf_reg": 4.0},
        {"depth": 8, "learning_rate": 0.04, "l2_leaf_reg": 7.0},
    ]


def tune_catboost(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    categorical: Sequence[str],
    config: AnalysisConfig,
    binary: bool,
    positive_weight: float | None = None,
    sample_weight: pd.Series | None = None,
) -> tuple[CatBoostClassifier, pd.DataFrame]:
    candidates = _catboost_candidates(binary)[: min(config.random_search_iterations, 2)]
    tuning_rows: list[dict[str, Any]] = []
    best_model: CatBoostClassifier | None = None
    best_score = math.inf
    for index, parameters in enumerate(candidates):
        model_parameters: dict[str, Any] = {
            "iterations": 500,
            "loss_function": "Logloss" if binary else "MultiClass",
            "eval_metric": "Logloss" if binary else "MultiClass",
            "task_type": "GPU",
            "devices": "0",
            "random_seed": config.seed + 100 + index,
            "allow_writing_files": False,
            "verbose": False,
            "random_strength": 0.5,
            **parameters,
        }
        if binary and positive_weight is not None:
            model_parameters["scale_pos_weight"] = positive_weight
        model = CatBoostClassifier(**model_parameters)
        fit_start = time.perf_counter()
        model.fit(
            train_x,
            train_y,
            cat_features=list(categorical),
            sample_weight=sample_weight,
            eval_set=(validation_x, validation_y),
            use_best_model=True,
            early_stopping_rounds=50,
            verbose=False,
        )
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        probabilities = model.predict_proba(validation_x)
        validation_predict_seconds = time.perf_counter() - predict_start
        score = log_loss(validation_y, probabilities, labels=list(model.classes_))
        tuning_rows.append(
            {
                "candidate": index,
                **parameters,
                "best_iteration": int(model.get_best_iteration()),
                "validation_log_loss": score,
                "fit_seconds": fit_seconds,
                "validation_predict_seconds": validation_predict_seconds,
                "backend": "cuda_gpu",
            }
        )
        if score < best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError("CatBoost tuning failed to produce a model.")
    return best_model, pd.DataFrame(tuning_rows)


def _xgboost_candidates(binary: bool) -> list[dict[str, Any]]:
    if binary:
        return [
            {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 3.0},
            {"max_depth": 8, "learning_rate": 0.035, "min_child_weight": 5.0},
        ]
    return [
        {"max_depth": 6, "learning_rate": 0.06, "min_child_weight": 3.0},
        {"max_depth": 8, "learning_rate": 0.04, "min_child_weight": 5.0},
    ]


def tune_xgboost(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    config: AnalysisConfig,
    binary: bool,
    positive_weight: float | None = None,
    sample_weight: pd.Series | None = None,
) -> tuple[XGBClassifier, pd.DataFrame, np.ndarray]:
    candidates = _xgboost_candidates(binary)[: min(config.random_search_iterations, 2)]
    if binary:
        model_classes = np.array([0, 1])
        encoded_train = train_y.astype(int)
        encoded_validation = validation_y.astype(int)
    else:
        model_classes = np.array(sorted(set(train_y) | set(validation_y)))
        mapping = {label: index for index, label in enumerate(model_classes)}
        encoded_train = train_y.map(mapping).astype(int)
        encoded_validation = validation_y.map(mapping).astype(int)
    tuning_rows: list[dict[str, Any]] = []
    best_model: XGBClassifier | None = None
    best_score = math.inf
    for index, parameters in enumerate(candidates):
        model_parameters: dict[str, Any] = {
            "n_estimators": 500,
            "objective": "binary:logistic" if binary else "multi:softprob",
            "eval_metric": "logloss" if binary else "mlogloss",
            "tree_method": "hist",
            "device": "cuda",
            "enable_categorical": True,
            "max_bin": 256,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 3.0,
            "random_state": config.seed + 200 + index,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
            **parameters,
        }
        if not binary:
            model_parameters["num_class"] = len(model_classes)
        if binary and positive_weight is not None:
            model_parameters["scale_pos_weight"] = positive_weight
        model = XGBClassifier(**model_parameters)
        fit_start = time.perf_counter()
        model.fit(
            train_x,
            encoded_train,
            sample_weight=sample_weight,
            eval_set=[(validation_x, encoded_validation)],
            verbose=False,
        )
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        probabilities = model.predict_proba(validation_x)
        validation_predict_seconds = time.perf_counter() - predict_start
        score = log_loss(
            encoded_validation,
            probabilities,
            labels=list(range(probabilities.shape[1])),
        )
        tuning_rows.append(
            {
                "candidate": index,
                **parameters,
                "best_iteration": int(model.best_iteration),
                "validation_log_loss": score,
                "fit_seconds": fit_seconds,
                "validation_predict_seconds": validation_predict_seconds,
                "backend": "cuda_gpu",
            }
        )
        if score < best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError("XGBoost tuning failed to produce a model.")
    return best_model, pd.DataFrame(tuning_rows), model_classes


def _prepare_tabm_arrays(
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
    categorical_mappings: dict[str, dict[str, int]] = {}
    train_categorical = np.zeros((len(train), len(categorical)), dtype=np.int64)
    validation_categorical = np.zeros(
        (len(validation), len(categorical)), dtype=np.int64
    )
    test_categorical = np.zeros((len(test), len(categorical)), dtype=np.int64)
    cardinalities = []
    for column_index, column in enumerate(categorical):
        train_values = train[column].fillna("UNKNOWN").astype(str)
        levels = sorted(set(train_values) | {"UNKNOWN"})
        mapping = {level: index for index, level in enumerate(levels)}
        unknown = mapping["UNKNOWN"]
        categorical_mappings[column] = mapping
        cardinalities.append(len(levels))
        train_categorical[:, column_index] = train_values.map(mapping).fillna(unknown)
        validation_categorical[:, column_index] = (
            validation[column]
            .fillna("UNKNOWN")
            .astype(str)
            .map(mapping)
            .fillna(unknown)
        )
        test_categorical[:, column_index] = (
            test[column]
            .fillna("UNKNOWN")
            .astype(str)
            .map(mapping)
            .fillna(unknown)
        )

    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    standard_deviations: dict[str, float] = {}
    numeric_arrays = []
    for frame in (train, validation, test):
        numeric_arrays.append(np.zeros((len(frame), len(numeric)), dtype=np.float32))
    for column_index, column in enumerate(numeric):
        train_values = pd.to_numeric(train[column], errors="coerce")
        median = float(train_values.median()) if train_values.notna().any() else 0.0
        filled_train = train_values.fillna(median)
        mean = float(filled_train.mean())
        standard_deviation = float(filled_train.std(ddof=0))
        if not np.isfinite(standard_deviation) or standard_deviation < 1e-8:
            standard_deviation = 1.0
        medians[column] = median
        means[column] = mean
        standard_deviations[column] = standard_deviation
        for array, frame in zip(numeric_arrays, (train, validation, test)):
            values = pd.to_numeric(frame[column], errors="coerce").fillna(median)
            array[:, column_index] = (
                (values.to_numpy(dtype=np.float32) - mean) / standard_deviation
            )
    metadata = {
        "categorical_columns": list(categorical),
        "numeric_columns": list(numeric),
        "categorical_mappings": categorical_mappings,
        "cat_cardinalities": cardinalities,
        "numeric_medians": medians,
        "numeric_means": means,
        "numeric_standard_deviations": standard_deviations,
    }
    return (
        (numeric_arrays[0], train_categorical),
        (numeric_arrays[1], validation_categorical),
        (numeric_arrays[2], test_categorical),
        metadata,
    )


def train_tabm(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    test_x: pd.DataFrame,
    categorical: Sequence[str],
    numeric: Sequence[str],
    config: AnalysisConfig,
    binary: bool,
    sample_weight: pd.Series | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("TabM GPU benchmark requires CUDA, but CUDA is unavailable.")
    torch.manual_seed(config.seed + (701 if binary else 700))
    torch.cuda.manual_seed_all(config.seed + (701 if binary else 700))
    device = torch.device("cuda:0")
    active_numeric = [
        column
        for column in numeric
        if pd.to_numeric(train_x[column], errors="coerce").nunique(dropna=True) > 1
    ]
    dropped_constant_numeric = [
        column for column in numeric if column not in active_numeric
    ]
    if not active_numeric:
        raise RuntimeError("TabM requires at least one nonconstant numerical feature.")
    (train_num, train_cat), (validation_num, validation_cat), (test_num, test_cat), metadata = (
        _prepare_tabm_arrays(
            train_x, validation_x, test_x, categorical, active_numeric
        )
    )
    metadata["dropped_constant_numeric_columns"] = dropped_constant_numeric
    if binary:
        classes = np.array([0, 1])
        encoded_train = train_y.astype(int).to_numpy()
        encoded_validation = validation_y.astype(int).to_numpy()
        output_size = 1
    else:
        classes = np.array(sorted(set(train_y) | set(validation_y)))
        mapping = {label: index for index, label in enumerate(classes)}
        encoded_train = train_y.map(mapping).astype(int).to_numpy()
        encoded_validation = validation_y.map(mapping).astype(int).to_numpy()
        output_size = len(classes)
    weights = (
        np.ones(len(train_x), dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_num),
        torch.from_numpy(train_cat),
        torch.from_numpy(encoded_train),
        torch.from_numpy(weights),
    )
    generator = torch.Generator().manual_seed(config.seed + 702)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    train_num_tensor = torch.from_numpy(train_num)
    bins = compute_bins(train_num_tensor, n_bins=32)
    numerical_embeddings = PiecewiseLinearEmbeddings(
        bins,
        d_embedding=16,
        activation=False,
        version="B",
    )
    model_kwargs = {
        "n_num_features": len(active_numeric),
        "cat_cardinalities": metadata["cat_cardinalities"],
        "d_out": output_size,
        "n_blocks": 2,
        "d_block": 256,
        "dropout": 0.15,
        "k": 16,
        "arch_type": "tabm",
        "start_scaling_init": "normal",
    }
    model = TabM.make(
        **model_kwargs,
        num_embeddings=numerical_embeddings,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def predict_arrays(
        numeric_array: np.ndarray, categorical_array: np.ndarray
    ) -> np.ndarray:
        model.eval()
        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(numeric_array), 1024):
                x_num = torch.from_numpy(numeric_array[start : start + 1024]).to(
                    device
                )
                x_cat = torch.from_numpy(
                    categorical_array[start : start + 1024]
                ).to(device)
                logits = model(x_num, x_cat)
                if binary:
                    positive = torch.sigmoid(logits.squeeze(-1)).mean(dim=1)
                    batch_probability = torch.stack([1 - positive, positive], dim=1)
                else:
                    batch_probability = torch.softmax(logits, dim=-1).mean(dim=1)
                probabilities.append(batch_probability.cpu().numpy())
        return np.concatenate(probabilities)

    best_state: dict[str, torch.Tensor] | None = None
    best_score = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    maximum_epochs = 120
    patience = 15
    fit_start = time.perf_counter()
    for epoch in range(maximum_epochs):
        model.train()
        for batch_num, batch_cat, batch_y, batch_weight in train_loader:
            batch_num = batch_num.to(device, non_blocking=True)
            batch_cat = batch_cat.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_weight = batch_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_num, batch_cat)
            if binary:
                expanded_y = batch_y[:, None].expand(-1, model.k).float()
                losses = torch_functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), expanded_y, reduction="none"
                )
            else:
                expanded_y = batch_y[:, None].expand(-1, model.k).reshape(-1)
                losses = torch_functional.cross_entropy(
                    logits.reshape(-1, output_size), expanded_y, reduction="none"
                ).reshape(-1, model.k)
            loss = (losses.mean(dim=1) * batch_weight).sum() / batch_weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        validation_probability = predict_arrays(validation_num, validation_cat)
        score = log_loss(
            encoded_validation,
            validation_probability,
            labels=list(range(validation_probability.shape[1])),
        )
        if score < best_score - 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    fit_seconds = time.perf_counter() - fit_start
    if best_state is None:
        raise RuntimeError("TabM training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    validation_predict_start = time.perf_counter()
    validation_probability = predict_arrays(validation_num, validation_cat)
    validation_predict_seconds = time.perf_counter() - validation_predict_start
    test_predict_start = time.perf_counter()
    test_probability = predict_arrays(test_num, test_cat)
    test_predict_seconds = time.perf_counter() - test_predict_start
    artifact = {
        "state_dict": best_state,
        "model_kwargs": model_kwargs,
        "piecewise_linear_bins": [bin_.cpu() for bin_ in bins],
        "piecewise_linear_embedding_size": 16,
        "preprocessing": metadata,
        "classes": classes,
        "binary": binary,
        "seed": config.seed + (701 if binary else 700),
    }
    tuning = pd.DataFrame(
        [
            {
                "algorithm": "tabm_gpu",
                "candidate": 0,
                "best_epoch": best_epoch,
                "epochs_completed": epoch + 1,
                "validation_log_loss": best_score,
                "fit_seconds": fit_seconds,
                "validation_predict_seconds": validation_predict_seconds,
                "test_predict_seconds": test_predict_seconds,
                "backend": "cuda_gpu",
                "d_block": 256,
                "n_blocks": 2,
                "ensemble_size": 16,
                "numeric_embedding_size": 16,
                "dropout": 0.15,
            }
        ]
    )
    del model
    torch.cuda.empty_cache()
    return artifact, validation_probability, test_probability, tuning, classes


def train_next_state_models(
    config: AnalysisConfig, transitions: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    frame = transitions.loc[
        transitions["eligible_primary_cohort"]
        & transitions["analysis_split"].isin(["train", "validation", "test"])
    ].copy()
    categorical, numeric = feature_columns(config)
    feature_list = [*categorical, *numeric]
    train = frame.loc[frame["analysis_split"] == "train"].copy()
    validation = frame.loc[frame["analysis_split"] == "validation"].copy()
    test = frame.loc[frame["analysis_split"] == "test"].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("One or more next-state temporal splits are empty.")
    classes = np.array(sorted(frame["next_state"].unique()))

    results: list[ModelResult] = []
    validation_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    markov_start = time.perf_counter()
    markov_validation = markov_probabilities(train, validation, classes)
    markov = markov_probabilities(train, test, classes)
    results.append(
        ModelResult(
            "empirical_markov",
            markov,
            classes,
            None,
            backend="cpu",
            fit_seconds=time.perf_counter() - markov_start,
            predict_seconds=0.0,
        )
    )
    validation_results["empirical_markov"] = (markov_validation, classes)

    logistic = _logistic_pipeline(categorical, numeric, multiclass=True)
    logistic_start = time.perf_counter()
    logistic.fit(
        _prepare_features(train, categorical, numeric),
        train["next_state"],
        classifier__sample_weight=train["cluster_equalizing_weight"],
    )
    logistic_fit_seconds = time.perf_counter() - logistic_start
    logistic_predict_start = time.perf_counter()
    logistic_validation_probabilities = logistic.predict_proba(
        _prepare_features(validation, categorical, numeric)
    )
    logistic_probabilities = logistic.predict_proba(
        _prepare_features(test, categorical, numeric)
    )
    logistic_predict_seconds = time.perf_counter() - logistic_predict_start
    results.append(
        ModelResult(
            "multinomial_logistic",
            logistic_probabilities,
            logistic.named_steps["classifier"].classes_,
            logistic,
            backend="cpu",
            fit_seconds=logistic_fit_seconds,
            predict_seconds=logistic_predict_seconds,
        )
    )
    validation_results["multinomial_logistic"] = (
        logistic_validation_probabilities,
        logistic.named_steps["classifier"].classes_,
    )
    joblib.dump(logistic, config.models_dir / "next_state_multinomial_logistic.joblib")

    for offset, algorithm in enumerate(
        ("random_forest", "extra_trees", "hist_gradient_boosting")
    ):
        classic_model = _tree_pipeline(
            categorical, numeric, algorithm, config.seed + 300 + offset
        )
        fit_start = time.perf_counter()
        classic_model.fit(
            _prepare_features(train, categorical, numeric),
            train["next_state"],
            classifier__sample_weight=train["cluster_equalizing_weight"],
        )
        fit_seconds = time.perf_counter() - fit_start
        validation_probability = classic_model.predict_proba(
            _prepare_features(validation, categorical, numeric)
        )
        predict_start = time.perf_counter()
        test_probability = classic_model.predict_proba(
            _prepare_features(test, categorical, numeric)
        )
        predict_seconds = time.perf_counter() - predict_start
        model_classes = classic_model.named_steps["classifier"].classes_
        results.append(
            ModelResult(
                algorithm,
                test_probability,
                model_classes,
                classic_model,
                backend="cpu",
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
            )
        )
        validation_results[algorithm] = (validation_probability, model_classes)
        joblib.dump(classic_model, config.models_dir / f"next_state_{algorithm}.joblib")

    raw_train = _prepare_features(train, categorical, numeric)
    raw_validation = _prepare_features(validation, categorical, numeric)
    raw_test = _prepare_features(test, categorical, numeric)
    native_train, (native_validation, native_test) = _prepare_lightgbm_frames(
        raw_train, [raw_validation, raw_test], categorical, numeric
    )
    lightgbm_model, lightgbm_tuning = tune_lightgbm(
        native_train,
        train["next_state"],
        native_validation,
        validation["next_state"],
        categorical,
        config,
        binary=False,
        sample_weight=train["cluster_equalizing_weight"],
    )
    lightgbm_validation_probabilities = lightgbm_model.predict_proba(native_validation)
    predict_start = time.perf_counter()
    lightgbm_probabilities = lightgbm_model.predict_proba(native_test)
    lightgbm_predict_seconds = time.perf_counter() - predict_start
    results.append(
        ModelResult(
            "lightgbm_cpu",
            lightgbm_probabilities,
            lightgbm_model.classes_,
            lightgbm_model,
            backend="cpu",
            fit_seconds=float(
                lightgbm_tuning.loc[
                    lightgbm_tuning["validation_log_loss"].idxmin(), "fit_seconds"
                ]
            ),
            predict_seconds=lightgbm_predict_seconds,
        )
    )
    validation_results["lightgbm_cpu"] = (
        lightgbm_validation_probabilities,
        lightgbm_model.classes_,
    )
    joblib.dump(lightgbm_model, config.models_dir / "next_state_lightgbm_cpu.joblib")

    catboost_model, catboost_tuning = tune_catboost(
        raw_train,
        train["next_state"],
        raw_validation,
        validation["next_state"],
        categorical,
        config,
        binary=False,
        sample_weight=train["cluster_equalizing_weight"],
    )
    catboost_validation_probabilities = catboost_model.predict_proba(raw_validation)
    predict_start = time.perf_counter()
    catboost_probabilities = catboost_model.predict_proba(raw_test)
    catboost_predict_seconds = time.perf_counter() - predict_start
    results.append(
        ModelResult(
            "catboost_gpu",
            catboost_probabilities,
            catboost_model.classes_,
            catboost_model,
            backend="cuda_gpu",
            fit_seconds=float(
                catboost_tuning.loc[
                    catboost_tuning["validation_log_loss"].idxmin(), "fit_seconds"
                ]
            ),
            predict_seconds=catboost_predict_seconds,
        )
    )
    validation_results["catboost_gpu"] = (
        catboost_validation_probabilities,
        catboost_model.classes_,
    )
    catboost_model.save_model(str(config.models_dir / "next_state_catboost_gpu.cbm"))

    xgboost_model, xgboost_tuning, xgboost_classes = tune_xgboost(
        native_train,
        train["next_state"],
        native_validation,
        validation["next_state"],
        config,
        binary=False,
        sample_weight=train["cluster_equalizing_weight"],
    )
    xgboost_validation_probabilities = xgboost_model.predict_proba(native_validation)
    predict_start = time.perf_counter()
    xgboost_probabilities = xgboost_model.predict_proba(native_test)
    xgboost_predict_seconds = time.perf_counter() - predict_start
    results.append(
        ModelResult(
            "xgboost_gpu",
            xgboost_probabilities,
            xgboost_classes,
            xgboost_model,
            backend="cuda_gpu",
            fit_seconds=float(
                xgboost_tuning.loc[
                    xgboost_tuning["validation_log_loss"].idxmin(), "fit_seconds"
                ]
            ),
            predict_seconds=xgboost_predict_seconds,
        )
    )
    validation_results["xgboost_gpu"] = (
        xgboost_validation_probabilities,
        xgboost_classes,
    )
    xgboost_model.save_model(config.models_dir / "next_state_xgboost_gpu.json")

    tabm_artifact, tabm_validation, tabm_test, tabm_tuning, tabm_classes = train_tabm(
        raw_train,
        train["next_state"],
        raw_validation,
        validation["next_state"],
        raw_test,
        categorical,
        numeric,
        config,
        binary=False,
        sample_weight=train["cluster_equalizing_weight"],
    )
    results.append(
        ModelResult(
            "tabm_gpu",
            tabm_test,
            tabm_classes,
            None,
            backend="cuda_gpu",
            fit_seconds=float(tabm_tuning.loc[0, "fit_seconds"]),
            predict_seconds=float(tabm_tuning.loc[0, "test_predict_seconds"]),
        )
    )
    validation_results["tabm_gpu"] = (tabm_validation, tabm_classes)
    torch.save(
        tabm_artifact,
        config.models_dir / "next_state_tabm_gpu.pt",
    )

    (
        delivery_moe_artifact,
        delivery_moe_validation,
        delivery_moe_test,
        delivery_moe_tuning,
        delivery_moe_classes,
    ) = train_delivery_sparse_moe_network(
        train,
        validation,
        test,
        categorical,
        numeric,
        config,
    )
    delivery_moe_tuning.insert(0, "algorithm", "delivery_sparse_moe_gpu")
    results.append(
        ModelResult(
            "delivery_sparse_moe_gpu",
            delivery_moe_test,
            delivery_moe_classes,
            None,
            backend="cuda_gpu",
            fit_seconds=float(delivery_moe_artifact["fit_seconds"]),
            predict_seconds=float(delivery_moe_artifact["test_predict_seconds"]),
        )
    )
    validation_results["delivery_sparse_moe_gpu"] = (
        delivery_moe_validation,
        delivery_moe_classes,
    )
    torch.save(
        delivery_moe_artifact,
        config.models_dir / "next_state_delivery_sparse_moe_gpu.pt",
    )

    (
        delivery_state_artifact,
        delivery_state_validation,
        delivery_state_test,
        delivery_state_tuning,
        delivery_state_classes,
    ) = train_delivery_state_network(
        train,
        validation,
        test,
        categorical,
        numeric,
        config,
        use_markov_prior=True,
    )
    delivery_state_tuning.insert(0, "algorithm", "delivery_state_residual_net_gpu")
    results.append(
        ModelResult(
            "delivery_state_residual_net_gpu",
            delivery_state_test,
            delivery_state_classes,
            None,
            backend="cuda_gpu",
            fit_seconds=float(delivery_state_artifact["fit_seconds"]),
            predict_seconds=float(
                delivery_state_tuning["test_predict_seconds_selected"].iloc[0]
            ),
        )
    )
    validation_results["delivery_state_residual_net_gpu"] = (
        delivery_state_validation,
        delivery_state_classes,
    )
    torch.save(
        delivery_state_artifact,
        config.models_dir / "next_state_delivery_state_residual_net_gpu.pt",
    )

    (
        no_markov_artifact,
        no_markov_validation,
        no_markov_test,
        no_markov_tuning,
        no_markov_classes,
    ) = train_delivery_state_network(
        train,
        validation,
        test,
        categorical,
        numeric,
        config,
        use_markov_prior=False,
    )
    no_markov_tuning.insert(0, "algorithm", "delivery_net_no_markov_ablation")
    results.append(
        ModelResult(
            "delivery_net_no_markov_ablation",
            no_markov_test,
            no_markov_classes,
            None,
            backend="cuda_gpu",
            fit_seconds=float(no_markov_artifact["fit_seconds"]),
            predict_seconds=float(
                no_markov_tuning["test_predict_seconds_selected"].iloc[0]
            ),
        )
    )
    validation_results["delivery_net_no_markov_ablation"] = (
        no_markov_validation,
        no_markov_classes,
    )
    torch.save(
        no_markov_artifact,
        config.models_dir / "next_state_delivery_net_no_markov_ablation.pt",
    )

    lightgbm_tuning.insert(0, "algorithm", "lightgbm_cpu")
    catboost_tuning.insert(0, "algorithm", "catboost_gpu")
    xgboost_tuning.insert(0, "algorithm", "xgboost_gpu")
    tuning = pd.concat(
        [
            lightgbm_tuning,
            catboost_tuning,
            xgboost_tuning,
            tabm_tuning,
            delivery_moe_tuning,
            delivery_state_tuning,
            no_markov_tuning,
        ],
        ignore_index=True,
    )
    tuning.to_csv(
        config.tables_dir / "next_state_boosting_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aligned_validation = {
        name: align_probabilities(probability, source_classes, classes)
        for name, (probability, source_classes) in validation_results.items()
    }
    aligned_test = {
        result.name: align_probabilities(result.probabilities, result.classes, classes)
        for result in results
    }
    ensemble_sources = [
        "empirical_markov",
        "multinomial_logistic",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "lightgbm_cpu",
        "catboost_gpu",
        "xgboost_gpu",
    ]
    validation_source_map = {
        name: aligned_validation[name] for name in ensemble_sources
    }
    test_source_map = {name: aligned_test[name] for name in ensemble_sources}
    equal_validation = np.mean(list(validation_source_map.values()), axis=0)
    equal_test = np.mean(list(test_source_map.values()), axis=0)
    results.append(
        ModelResult(
            "equal_weight_pool_ablation",
            equal_test,
            classes,
            None,
            backend="hybrid",
            fit_seconds=0.0,
            predict_seconds=0.0,
        )
    )
    validation_results["equal_weight_pool_ablation"] = (equal_validation, classes)
    ensemble_start = time.perf_counter()
    proposed_validation, proposed_test, ensemble_weights = (
        state_conditioned_delivery_ensemble(
            validation,
            validation_source_map,
            test,
            test_source_map,
            "next_state",
            classes,
        )
    )
    ensemble_fit_seconds = time.perf_counter() - ensemble_start
    results.append(
        ModelResult(
            "proposed_delivery_ensemble",
            proposed_test,
            classes,
            {"weights": ensemble_weights, "sources": ensemble_sources},
            backend="hybrid",
            fit_seconds=ensemble_fit_seconds,
            predict_seconds=0.0,
        )
    )
    validation_results["proposed_delivery_ensemble"] = (
        proposed_validation,
        classes,
    )
    ensemble_weights.to_csv(
        config.tables_dir / "next_state_proposed_ensemble_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    joblib.dump(
        {"weights": ensemble_weights, "sources": ensemble_sources, "classes": classes},
        config.models_dir / "next_state_proposed_delivery_ensemble.joblib",
    )

    validation_rows = []
    for name, (probability, source_classes) in validation_results.items():
        aligned = align_probabilities(probability, source_classes, classes)
        validation_rows.append(
            {
                "model": name,
                **multiclass_metrics(
                    validation["next_state"],
                    aligned,
                    classes,
                    sample_weight=validation["cluster_equalizing_weight"],
                ),
            }
        )
    validation_performance = pd.DataFrame(validation_rows)
    validation_performance.to_csv(
        config.tables_dir / "next_state_validation_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    performance_rows: list[dict[str, Any]] = []
    class_tables: list[pd.DataFrame] = []
    test_probability_map: dict[str, np.ndarray] = {}
    prediction_frame = test[
        [
            "unit_id",
            "development_cluster_id",
            "entity_id",
            "report_month",
            "current_state",
            "next_state",
            "technology_group",
        ]
    ].reset_index(drop=True)
    for result in results:
        aligned = align_probabilities(result.probabilities, result.classes, classes)
        test_probability_map[result.name] = aligned
        metrics = multiclass_metrics(
            test["next_state"],
            aligned,
            classes,
            sample_weight=test["cluster_equalizing_weight"],
        )
        performance_rows.append(
            {
                "model": result.name,
                "backend": result.backend,
                "fit_seconds": result.fit_seconds,
                "test_predict_seconds": result.predict_seconds,
                **metrics,
            }
        )
        class_tables.append(
            per_class_metrics(test["next_state"], aligned, classes, result.name)
        )
        for index, state in enumerate(classes):
            prediction_frame[f"{result.name}_prob_{state}"] = aligned[:, index]
    performance = pd.DataFrame(performance_rows)
    per_class = pd.concat(class_tables, ignore_index=True)
    prediction_frame.to_parquet(
        config.predictions_dir / "next_state_test_predictions.parquet", index=False
    )
    performance.to_csv(
        config.tables_dir / "next_state_model_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_class.to_csv(
        config.tables_dir / "next_state_per_class_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired_comparisons = paired_bootstrap_multiclass_comparisons(
        test,
        "next_state",
        test_probability_map,
        "proposed_delivery_ensemble",
        classes,
        "development_cluster_id",
        config.bootstrap_reps,
        config.seed + 610,
    )
    paired_comparisons.to_csv(
        config.tables_dir / "next_state_proposed_paired_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    custom_deep_comparisons = paired_bootstrap_multiclass_comparisons(
        test,
        "next_state",
        test_probability_map,
        "delivery_state_residual_net_gpu",
        classes,
        "development_cluster_id",
        config.bootstrap_reps,
        config.seed + 611,
    )
    custom_deep_comparisons.to_csv(
        config.tables_dir / "next_state_custom_deep_paired_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    delivery_moe_comparisons = paired_bootstrap_multiclass_comparisons(
        test,
        "next_state",
        test_probability_map,
        "delivery_sparse_moe_gpu",
        classes,
        "development_cluster_id",
        config.bootstrap_reps,
        config.seed + 612,
    )
    delivery_moe_comparisons.to_csv(
        config.tables_dir / "next_state_delivery_sparse_moe_paired_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    importance_tables = []
    for algorithm, model in (
        ("lightgbm_cpu", lightgbm_model),
        ("catboost_gpu", catboost_model),
        ("xgboost_gpu", xgboost_model),
    ):
        importance_tables.append(
            pd.DataFrame(
                {
                    "algorithm": algorithm,
                    "feature": feature_list,
                    "importance": model.feature_importances_,
                }
            )
        )
    importances = pd.concat(importance_tables, ignore_index=True).sort_values(
        ["algorithm", "importance"], ascending=[True, False]
    )
    importances.to_csv(
        config.tables_dir / "next_state_boosting_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "next_state_performance": performance,
        "next_state_per_class": per_class,
        "next_state_feature_importance": importances,
        "next_state_tuning": tuning,
        "next_state_validation_performance": validation_performance,
        "next_state_ensemble_weights": ensemble_weights,
        "next_state_paired_comparisons": paired_comparisons,
        "next_state_custom_deep_comparisons": custom_deep_comparisons,
        "next_state_delivery_moe_comparisons": delivery_moe_comparisons,
        "next_state_delivery_moe_artifact": delivery_moe_artifact,
    }


def align_probabilities(
    probabilities: np.ndarray,
    source_classes: Sequence[str],
    target_classes: Sequence[str],
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    source = {label: index for index, label in enumerate(source_classes)}
    aligned = np.full((len(probabilities), len(target_classes)), EPSILON)
    for index, label in enumerate(target_classes):
        if label in source:
            aligned[:, index] = probabilities[:, source[label]]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def _simplex_weights(
    y_true: pd.Series,
    probability_map: dict[str, np.ndarray],
    classes: Sequence[Any],
) -> tuple[np.ndarray, float]:
    names = list(probability_map)
    stack = np.stack([probability_map[name] for name in names], axis=0)

    def objective(weights: np.ndarray) -> float:
        pooled = np.tensordot(weights, stack, axes=(0, 0))
        pooled = np.clip(pooled, EPSILON, 1 - EPSILON)
        pooled /= pooled.sum(axis=1, keepdims=True)
        return float(log_loss(y_true, pooled, labels=list(classes)))

    initial = np.full(len(names), 1.0 / len(names))
    fitted = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 300, "ftol": 1e-10},
    )
    weights = fitted.x if fitted.success else initial
    weights = np.clip(weights, 0.0, 1.0)
    weights /= weights.sum()
    return weights, objective(weights)


def state_conditioned_delivery_ensemble(
    validation: pd.DataFrame,
    validation_probabilities: dict[str, np.ndarray],
    test: pd.DataFrame,
    test_probabilities: dict[str, np.ndarray],
    outcome_column: str,
    classes: Sequence[Any],
    state_column: str = "current_state",
    shrinkage_observations: int = 750,
    minimum_state_observations: int = 100,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Validation-fitted, state-conditioned convex pool with partial pooling.

    This is the proposed delivery-aware ensemble. Each delivery state receives a
    locally optimized set of nonnegative learner weights. Sparse states are
    shrunk toward the global optimum using a prespecified reliability factor.
    Test outcomes are never supplied to this function.
    """
    names = list(validation_probabilities)
    global_weights, global_loss = _simplex_weights(
        validation[outcome_column], validation_probabilities, classes
    )
    weight_rows = [
        {
            "state": "__GLOBAL__",
            "validation_n": len(validation),
            "reliability": 1.0,
            "validation_log_loss": global_loss,
            **{name: weight for name, weight in zip(names, global_weights)},
        }
    ]
    weights_by_state: dict[str, np.ndarray] = {}
    validation_state_values = validation[state_column].fillna("UNKNOWN").astype(str)
    for state, indices in validation.groupby(validation_state_values).groups.items():
        positions = validation.index.get_indexer(indices)
        state_y = validation.loc[indices, outcome_column]
        state_weights = global_weights.copy()
        reliability = 0.0
        state_loss = np.nan
        if len(indices) >= minimum_state_observations and state_y.nunique() > 1:
            local_map = {
                name: probability[positions]
                for name, probability in validation_probabilities.items()
            }
            local_weights, _ = _simplex_weights(state_y, local_map, classes)
            reliability = len(indices) / (len(indices) + shrinkage_observations)
            state_weights = reliability * local_weights + (1 - reliability) * global_weights
            pooled = sum(
                weight * local_map[name]
                for name, weight in zip(names, state_weights)
            )
            state_loss = log_loss(state_y, pooled, labels=list(classes))
        weights_by_state[str(state)] = state_weights
        weight_rows.append(
            {
                "state": state,
                "validation_n": len(indices),
                "reliability": reliability,
                "validation_log_loss": state_loss,
                **{name: weight for name, weight in zip(names, state_weights)},
            }
        )

    def pool(frame: pd.DataFrame, probability_map: dict[str, np.ndarray]) -> np.ndarray:
        pooled = np.zeros_like(next(iter(probability_map.values())), dtype=float)
        states = frame[state_column].fillna("UNKNOWN").astype(str).to_numpy()
        for position, state in enumerate(states):
            weights = weights_by_state.get(state, global_weights)
            pooled[position] = sum(
                weight * probability_map[name][position]
                for name, weight in zip(names, weights)
            )
        pooled = np.clip(pooled, EPSILON, 1 - EPSILON)
        pooled /= pooled.sum(axis=1, keepdims=True)
        return pooled

    return (
        pool(validation, validation_probabilities),
        pool(test, test_probabilities),
        pd.DataFrame(weight_rows),
    )


def calibration_intercept_slope(y_true: pd.Series, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, EPSILON, 1 - EPSILON)
    predictor = logit(clipped)
    try:
        model = sm.GLM(
            np.asarray(y_true, dtype=float),
            sm.add_constant(predictor),
            family=sm.families.Binomial(),
        ).fit()
        return float(model.params[0]), float(model.params[1])
    except Exception:
        return np.nan, np.nan


def fit_platt(y_true: pd.Series, probability: np.ndarray) -> LogisticRegression:
    predictor = logit(np.clip(probability, EPSILON, 1 - EPSILON)).reshape(-1, 1)
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(predictor, y_true)
    return calibrator


def apply_platt(calibrator: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    predictor = logit(np.clip(probability, EPSILON, 1 - EPSILON)).reshape(-1, 1)
    return calibrator.predict_proba(predictor)[:, 1]


def binary_metrics(
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    predicted = probability >= threshold
    result = {
        "roc_auc": roc_auc_score(y_true, probability) if y_true.nunique() > 1 else np.nan,
        "average_precision": average_precision_score(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
        "log_loss": log_loss(y_true, probability, labels=[0, 1]),
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "f1": f1_score(y_true, predicted, zero_division=0),
    }
    intercept, slope = calibration_intercept_slope(y_true, probability)
    result["calibration_intercept"] = intercept
    result["calibration_slope"] = slope
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        result.update(
            {
                "cluster_weighted_roc_auc": roc_auc_score(
                    y_true, probability, sample_weight=weights
                ),
                "cluster_weighted_average_precision": average_precision_score(
                    y_true, probability, sample_weight=weights
                ),
                "cluster_weighted_brier": brier_score_loss(
                    y_true, probability, sample_weight=weights
                ),
                "cluster_weighted_log_loss": log_loss(
                    y_true, probability, labels=[0, 1], sample_weight=weights
                ),
            }
        )
    return result


def choose_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    thresholds = np.linspace(0.02, 0.98, 97)
    scores = [
        f1_score(y_true, probability >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def calibration_bins(
    y_true: pd.Series,
    probability: np.ndarray,
    model_name: str,
    outcome: str,
    bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame({"observed": np.asarray(y_true), "predicted": probability})
    frame["bin"] = pd.qcut(
        frame["predicted"], q=bins, duplicates="drop", labels=False
    )
    result = (
        frame.groupby("bin", dropna=False)
        .agg(
            observations=("observed", "size"),
            events=("observed", "sum"),
            mean_predicted=("predicted", "mean"),
            observed_rate=("observed", "mean"),
            min_predicted=("predicted", "min"),
            max_predicted=("predicted", "max"),
        )
        .reset_index()
    )
    result.insert(0, "outcome", outcome)
    result.insert(0, "model", model_name)
    return result


def decision_curve(
    y_true: pd.Series, probability: np.ndarray, model_name: str, outcome: str
) -> pd.DataFrame:
    observed = np.asarray(y_true, dtype=int)
    n = len(observed)
    prevalence = observed.mean()
    rows = []
    for threshold in np.linspace(0.01, 0.50, 50):
        predicted = probability >= threshold
        true_positive = np.sum(predicted & (observed == 1))
        false_positive = np.sum(predicted & (observed == 0))
        odds = threshold / (1 - threshold)
        rows.append(
            {
                "model": model_name,
                "outcome": outcome,
                "threshold": threshold,
                "net_benefit": true_positive / n - false_positive / n * odds,
                "treat_all_net_benefit": prevalence - (1 - prevalence) * odds,
                "treat_none_net_benefit": 0.0,
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap_binary_metrics(
    frame: pd.DataFrame,
    outcome_column: str,
    probability_column: str,
    threshold: float,
    cluster_column: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cluster_to_indices = {
        cluster: indices.to_numpy()
        for cluster, indices in frame.groupby(cluster_column).groups.items()
    }
    clusters = np.array(list(cluster_to_indices), dtype=object)
    rows: list[dict[str, float]] = []
    for repetition in range(repetitions):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([cluster_to_indices[cluster] for cluster in sampled])
        sample = frame.loc[indices]
        if sample[outcome_column].nunique() < 2:
            continue
        observed = sample[outcome_column]
        probability = sample[probability_column].to_numpy()
        predicted = probability >= threshold
        metrics = {
            "roc_auc": roc_auc_score(observed, probability),
            "average_precision": average_precision_score(observed, probability),
            "brier": brier_score_loss(observed, probability),
            "log_loss": log_loss(observed, probability, labels=[0, 1]),
            "balanced_accuracy": balanced_accuracy_score(observed, predicted),
            "f1": f1_score(observed, predicted, zero_division=0),
        }
        rows.append({"bootstrap_rep": repetition, **metrics})
    return pd.DataFrame(rows)


def bootstrap_summary(
    bootstrap: pd.DataFrame, model_name: str, outcome: str
) -> pd.DataFrame:
    rows = []
    for metric in (
        "roc_auc",
        "average_precision",
        "brier",
        "log_loss",
        "balanced_accuracy",
        "f1",
        "calibration_intercept",
        "calibration_slope",
    ):
        if metric not in bootstrap:
            continue
        values = bootstrap[metric].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "model": model_name,
                "outcome": outcome,
                "metric": metric,
                "bootstrap_repetitions": len(values),
                "bootstrap_mean": values.mean(),
                "ci95_low": values.quantile(0.025),
                "ci95_high": values.quantile(0.975),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_binary_comparisons(
    frame: pd.DataFrame,
    outcome_column: str,
    probability_map: dict[str, np.ndarray],
    proposed_name: str,
    cluster_column: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    working = frame.reset_index(drop=True)
    cluster_to_indices = {
        cluster: np.asarray(indices, dtype=int)
        for cluster, indices in working.groupby(cluster_column).groups.items()
    }
    clusters = np.array(list(cluster_to_indices), dtype=object)

    def scores(indices: np.ndarray, probability: np.ndarray) -> dict[str, float]:
        observed = working.loc[indices, outcome_column].to_numpy()
        selected = probability[indices]
        return {
            "roc_auc": roc_auc_score(observed, selected),
            "average_precision": average_precision_score(observed, selected),
            "brier": brier_score_loss(observed, selected),
            "log_loss": log_loss(observed, selected, labels=[0, 1]),
        }

    full_indices = np.arange(len(frame))
    proposed_point = scores(full_indices, probability_map[proposed_name])
    comparisons = [name for name in probability_map if name != proposed_name]
    point = {
        name: scores(full_indices, probability_map[name]) for name in comparisons
    }
    samples: dict[tuple[str, str], list[float]] = {
        (name, metric): []
        for name in comparisons
        for metric in proposed_point
    }
    for _ in range(repetitions):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate(
            [cluster_to_indices[cluster] for cluster in sampled_clusters]
        )
        if working.loc[indices, outcome_column].nunique() < 2:
            continue
        proposed_scores = scores(indices, probability_map[proposed_name])
        for name in comparisons:
            comparator_scores = scores(indices, probability_map[name])
            for metric in proposed_scores:
                samples[(name, metric)].append(
                    proposed_scores[metric] - comparator_scores[metric]
                )
    rows = []
    for name in comparisons:
        for metric in proposed_point:
            values = np.asarray(samples[(name, metric)], dtype=float)
            higher_is_better = metric in {"roc_auc", "average_precision"}
            rows.append(
                {
                    "proposed_model": proposed_name,
                    "comparator": name,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "point_delta_proposed_minus_comparator": proposed_point[metric]
                    - point[name][metric],
                    "bootstrap_repetitions": len(values),
                    "bootstrap_mean_delta": values.mean(),
                    "ci95_low": np.quantile(values, 0.025),
                    "ci95_high": np.quantile(values, 0.975),
                    "probability_proposed_better": float(
                        np.mean(values > 0) if higher_is_better else np.mean(values < 0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_bootstrap_multiclass_comparisons(
    frame: pd.DataFrame,
    outcome_column: str,
    probability_map: dict[str, np.ndarray],
    proposed_name: str,
    classes: Sequence[str],
    cluster_column: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    working = frame.reset_index(drop=True)
    cluster_to_indices = {
        cluster: np.asarray(indices, dtype=int)
        for cluster, indices in working.groupby(cluster_column).groups.items()
    }
    clusters = np.array(list(cluster_to_indices), dtype=object)
    observed = working[outcome_column]

    def scores(indices: np.ndarray, probability: np.ndarray) -> dict[str, float]:
        selected_y = observed.iloc[indices]
        selected_probability = probability[indices]
        predicted = np.asarray(classes)[np.argmax(selected_probability, axis=1)]
        return {
            "log_loss": log_loss(
                selected_y, selected_probability, labels=list(classes)
            ),
            "multiclass_brier": multiclass_brier(
                selected_y, selected_probability, classes
            ),
            "macro_f1": f1_score(
                selected_y, predicted, average="macro", zero_division=0
            ),
        }

    full_indices = np.arange(len(frame))
    proposed_point = scores(full_indices, probability_map[proposed_name])
    comparisons = [name for name in probability_map if name != proposed_name]
    point = {
        name: scores(full_indices, probability_map[name]) for name in comparisons
    }
    samples: dict[tuple[str, str], list[float]] = {
        (name, metric): []
        for name in comparisons
        for metric in proposed_point
    }
    for _ in range(repetitions):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate(
            [cluster_to_indices[cluster] for cluster in sampled_clusters]
        )
        proposed_scores = scores(indices, probability_map[proposed_name])
        for name in comparisons:
            comparator_scores = scores(indices, probability_map[name])
            for metric in proposed_scores:
                samples[(name, metric)].append(
                    proposed_scores[metric] - comparator_scores[metric]
                )
    rows = []
    for name in comparisons:
        for metric in proposed_point:
            values = np.asarray(samples[(name, metric)], dtype=float)
            higher_is_better = metric == "macro_f1"
            rows.append(
                {
                    "proposed_model": proposed_name,
                    "comparator": name,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "point_delta_proposed_minus_comparator": proposed_point[metric]
                    - point[name][metric],
                    "bootstrap_repetitions": len(values),
                    "bootstrap_mean_delta": values.mean(),
                    "ci95_low": np.quantile(values, 0.025),
                    "ci95_high": np.quantile(values, 0.975),
                    "probability_proposed_better": float(
                        np.mean(values > 0) if higher_is_better else np.mean(values < 0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _simple_features() -> tuple[list[str], list[str]]:
    return ["current_state", "technology_group"], ["months_to_reported_operation"]


def train_custom_deep_landmark_model(
    config: AnalysisConfig, landmarks: pd.DataFrame
) -> dict[str, Any]:
    frame = landmarks.loc[
        (landmarks["landmark_age_months"] == 0)
        & landmarks["observable_12_month_outcome"]
        & landmarks["observable_24_month_outcome"]
        & landmarks["analysis_split"].isin(["train", "validation", "test"])
    ].copy()
    frame.reset_index(drop=True, inplace=True)
    train = frame.loc[frame["analysis_split"] == "train"].copy()
    validation = frame.loc[frame["analysis_split"] == "validation"].copy()
    test = frame.loc[frame["analysis_split"] == "test"].copy()
    categorical, numeric = feature_columns(config)
    artifact, validation_probabilities, test_probabilities, tuning = (
        train_delivery_competing_risk_network(
            train,
            validation,
            test,
            categorical,
            numeric,
            config,
        )
    )
    torch.save(
        artifact,
        config.models_dir / "landmark_delivery_competing_risk_net_gpu.pt",
    )
    tuning.insert(0, "algorithm", "delivery_competing_risk_net_gpu")
    tuning.to_csv(
        config.tables_dir / "landmark_custom_deep_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "validation_unit_ids": validation["unit_id"].to_numpy(),
        "test_unit_ids": test["unit_id"].to_numpy(),
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
        "tuning": tuning,
        "artifact": artifact,
        "fit_seconds": float(artifact["fit_seconds"]),
        "test_predict_seconds": float(
            tuning["test_predict_seconds_selected"].iloc[0]
        ),
    }


def train_custom_moe_landmark_model(
    config: AnalysisConfig, landmarks: pd.DataFrame
) -> dict[str, Any]:
    frame = landmarks.loc[
        (landmarks["landmark_age_months"] == 0)
        & landmarks["observable_12_month_outcome"]
        & landmarks["observable_24_month_outcome"]
        & landmarks["analysis_split"].isin(["train", "validation", "test"])
    ].copy()
    frame.reset_index(drop=True, inplace=True)
    train = frame.loc[frame["analysis_split"] == "train"].copy()
    validation = frame.loc[frame["analysis_split"] == "validation"].copy()
    test = frame.loc[frame["analysis_split"] == "test"].copy()
    categorical, numeric = feature_columns(config)
    artifact, validation_probabilities, test_probabilities, tuning = (
        train_delivery_sparse_moe_risk_network(
            train, validation, test, categorical, numeric, config
        )
    )
    torch.save(
        artifact,
        config.models_dir / "landmark_delivery_sparse_moe_gpu.pt",
    )
    tuning.insert(0, "algorithm", "delivery_sparse_moe_gpu")
    tuning.to_csv(
        config.tables_dir / "landmark_delivery_sparse_moe_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "validation_unit_ids": validation["unit_id"].to_numpy(),
        "test_unit_ids": test["unit_id"].to_numpy(),
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
        "tuning": tuning,
        "artifact": artifact,
        "fit_seconds": float(artifact["fit_seconds"]),
        "test_predict_seconds": float(artifact["test_predict_seconds"]),
    }


def train_delivery_path_moe_landmark_model(
    config: AnalysisConfig,
    landmarks: pd.DataFrame,
    transition_artifact: dict[str, Any],
) -> dict[str, Any]:
    """Derive horizon risks by coherent rollout of the selected transition MoE."""
    frame = landmarks.loc[
        (landmarks["landmark_age_months"] == 0)
        & landmarks["observable_12_month_outcome"]
        & landmarks["observable_24_month_outcome"]
        & landmarks["analysis_split"].isin(["validation", "test"])
    ].copy()
    frame.reset_index(drop=True, inplace=True)
    validation = frame.loc[frame["analysis_split"] == "validation"].copy()
    test = frame.loc[frame["analysis_split"] == "test"].copy()
    validation_probabilities, validation_audit = rollout_delivery_path_moe(
        transition_artifact, validation
    )
    test_probabilities, test_audit = rollout_delivery_path_moe(
        transition_artifact, test
    )
    audit = pd.DataFrame(
        [
            {"split": "validation", **validation_audit},
            {"split": "test", **test_audit},
        ]
    )
    audit.to_csv(
        config.tables_dir / "landmark_delivery_path_moe_rollout_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tuning = pd.DataFrame(
        [
            {
                "algorithm": "delivery_path_moe_gpu",
                "candidate": 0,
                "source_transition_checkpoint": "validation_selected_delivery_sparse_moe_gpu",
                "operation_horizon": 12,
                "cancellation_horizon": 24,
                "validation_rollout_seconds": validation_audit["predict_seconds"],
                "test_predict_seconds": test_audit["predict_seconds"],
                "maximum_probability_mass_error": max(
                    validation_audit["maximum_probability_mass_error"],
                    test_audit["maximum_probability_mass_error"],
                ),
                "backend": "cuda_gpu",
                "no_additional_test_fitted_parameters": True,
            }
        ]
    )
    return {
        "validation_unit_ids": validation["unit_id"].to_numpy(),
        "test_unit_ids": test["unit_id"].to_numpy(),
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
        "tuning": tuning,
        "audit": audit,
        "fit_seconds": 0.0,
        "test_predict_seconds": float(test_audit["predict_seconds"]),
    }


def train_binary_landmark_models(
    config: AnalysisConfig,
    landmarks: pd.DataFrame,
    outcome_column: str,
    observable_column: str,
    outcome_name: str,
    landmark_age: int = 0,
    custom_deep: dict[str, Any] | None = None,
    custom_moe: dict[str, Any] | None = None,
    custom_rollout: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    frame = landmarks.loc[
        (landmarks["landmark_age_months"] == landmark_age)
        & landmarks[observable_column]
        & landmarks["analysis_split"].isin(["train", "validation", "test"])
    ].copy()
    frame.reset_index(drop=True, inplace=True)
    train = frame.loc[frame["analysis_split"] == "train"].copy()
    validation = frame.loc[frame["analysis_split"] == "validation"].copy()
    test = frame.loc[frame["analysis_split"] == "test"].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError(f"One or more {outcome_name} temporal splits are empty.")
    if min(
        train[outcome_column].nunique(),
        validation[outcome_column].nunique(),
        test[outcome_column].nunique(),
    ) < 2:
        raise RuntimeError(f"One or more {outcome_name} splits contain one outcome class.")

    full_categorical, full_numeric = feature_columns(config)
    simple_categorical, simple_numeric = _simple_features()
    records: dict[str, dict[str, Any]] = {}

    def fit_pipeline_model(
        name: str,
        model: Pipeline,
        categorical: Sequence[str],
        numeric: Sequence[str],
    ) -> None:
        fit_start = time.perf_counter()
        model.fit(
            _prepare_features(train, categorical, numeric),
            train[outcome_column],
            classifier__sample_weight=train["cluster_equalizing_weight"],
        )
        fit_seconds = time.perf_counter() - fit_start
        validation_probability = model.predict_proba(
            _prepare_features(validation, categorical, numeric)
        )[:, 1]
        predict_start = time.perf_counter()
        test_probability = model.predict_proba(
            _prepare_features(test, categorical, numeric)
        )[:, 1]
        records[name] = {
            "model": model,
            "validation_probability": validation_probability,
            "test_probability": test_probability,
            "backend": "cpu",
            "fit_seconds": fit_seconds,
            "test_predict_seconds": time.perf_counter() - predict_start,
        }

    fit_pipeline_model(
        "status_schedule_logistic",
        _logistic_pipeline(simple_categorical, simple_numeric),
        simple_categorical,
        simple_numeric,
    )
    fit_pipeline_model(
        "full_logistic",
        _logistic_pipeline(full_categorical, full_numeric),
        full_categorical,
        full_numeric,
    )
    for offset, name in enumerate(
        ("random_forest", "extra_trees", "hist_gradient_boosting")
    ):
        fit_pipeline_model(
            name,
            _tree_pipeline(
                full_categorical, full_numeric, name, config.seed + 400 + offset
            ),
            full_categorical,
            full_numeric,
        )

    custom_tuning: pd.DataFrame | None = None
    if custom_deep is not None:
        validation_lookup = dict(
            zip(
                custom_deep["validation_unit_ids"],
                custom_deep["validation_probabilities"][outcome_name],
            )
        )
        test_lookup = dict(
            zip(
                custom_deep["test_unit_ids"],
                custom_deep["test_probabilities"][outcome_name],
            )
        )
        validation_probability = validation["unit_id"].map(validation_lookup)
        test_probability = test["unit_id"].map(test_lookup)
        if validation_probability.isna().any() or test_probability.isna().any():
            raise RuntimeError(
                f"Custom deep probabilities do not align for {outcome_name}."
            )
        records["delivery_competing_risk_net_gpu"] = {
            "model": None,
            "validation_probability": validation_probability.to_numpy(),
            "test_probability": test_probability.to_numpy(),
            "backend": "cuda_gpu",
            "fit_seconds": custom_deep["fit_seconds"],
            "test_predict_seconds": custom_deep["test_predict_seconds"],
        }
        custom_tuning = custom_deep["tuning"].copy()

    custom_moe_tuning: pd.DataFrame | None = None
    if custom_moe is not None:
        validation_lookup = dict(
            zip(
                custom_moe["validation_unit_ids"],
                custom_moe["validation_probabilities"][outcome_name],
            )
        )
        test_lookup = dict(
            zip(
                custom_moe["test_unit_ids"],
                custom_moe["test_probabilities"][outcome_name],
            )
        )
        validation_probability = validation["unit_id"].map(validation_lookup)
        test_probability = test["unit_id"].map(test_lookup)
        if validation_probability.isna().any() or test_probability.isna().any():
            raise RuntimeError(f"Custom MoE probabilities do not align for {outcome_name}.")
        records["delivery_sparse_moe_gpu"] = {
            "model": None,
            "validation_probability": validation_probability.to_numpy(),
            "test_probability": test_probability.to_numpy(),
            "backend": "cuda_gpu",
            "fit_seconds": custom_moe["fit_seconds"],
            "test_predict_seconds": custom_moe["test_predict_seconds"],
        }
        custom_moe_tuning = custom_moe["tuning"].copy()

    custom_rollout_tuning: pd.DataFrame | None = None
    if custom_rollout is not None:
        validation_lookup = dict(
            zip(
                custom_rollout["validation_unit_ids"],
                custom_rollout["validation_probabilities"][outcome_name],
            )
        )
        test_lookup = dict(
            zip(
                custom_rollout["test_unit_ids"],
                custom_rollout["test_probabilities"][outcome_name],
            )
        )
        validation_probability = validation["unit_id"].map(validation_lookup)
        test_probability = test["unit_id"].map(test_lookup)
        if validation_probability.isna().any() or test_probability.isna().any():
            raise RuntimeError(
                f"Delivery-PathMoE probabilities do not align for {outcome_name}."
            )
        records["delivery_path_moe_gpu"] = {
            "model": None,
            "validation_probability": validation_probability.to_numpy(),
            "test_probability": test_probability.to_numpy(),
            "backend": "cuda_gpu",
            "fit_seconds": custom_rollout["fit_seconds"],
            "test_predict_seconds": custom_rollout["test_predict_seconds"],
        }
        custom_rollout_tuning = custom_rollout["tuning"].copy()

    raw_train = _prepare_features(train, full_categorical, full_numeric)
    raw_validation = _prepare_features(validation, full_categorical, full_numeric)
    raw_test = _prepare_features(test, full_categorical, full_numeric)
    native_train, (native_validation, native_test) = _prepare_lightgbm_frames(
        raw_train, [raw_validation, raw_test], full_categorical, full_numeric
    )
    prevalence = train[outcome_column].mean()
    positive_weight = (
        math.sqrt((1 - prevalence) / max(prevalence, EPSILON))
        if prevalence < 0.20
        else None
    )

    lightgbm_model, lightgbm_tuning = tune_lightgbm(
        native_train,
        train[outcome_column],
        native_validation,
        validation[outcome_column],
        full_categorical,
        config,
        binary=True,
        positive_weight=positive_weight,
        sample_weight=train["cluster_equalizing_weight"],
    )
    predict_start = time.perf_counter()
    lightgbm_test_probability = lightgbm_model.predict_proba(native_test)[:, 1]
    records["lightgbm_cpu"] = {
        "model": lightgbm_model,
        "validation_probability": lightgbm_model.predict_proba(native_validation)[:, 1],
        "test_probability": lightgbm_test_probability,
        "backend": "cpu",
        "fit_seconds": float(
            lightgbm_tuning.loc[
                lightgbm_tuning["validation_log_loss"].idxmin(), "fit_seconds"
            ]
        ),
        "test_predict_seconds": time.perf_counter() - predict_start,
    }

    catboost_model, catboost_tuning = tune_catboost(
        raw_train,
        train[outcome_column],
        raw_validation,
        validation[outcome_column],
        full_categorical,
        config,
        binary=True,
        positive_weight=positive_weight,
        sample_weight=train["cluster_equalizing_weight"],
    )
    predict_start = time.perf_counter()
    catboost_test_probability = catboost_model.predict_proba(raw_test)[:, 1]
    records["catboost_gpu"] = {
        "model": catboost_model,
        "validation_probability": catboost_model.predict_proba(raw_validation)[:, 1],
        "test_probability": catboost_test_probability,
        "backend": "cuda_gpu",
        "fit_seconds": float(
            catboost_tuning.loc[
                catboost_tuning["validation_log_loss"].idxmin(), "fit_seconds"
            ]
        ),
        "test_predict_seconds": time.perf_counter() - predict_start,
    }

    xgboost_model, xgboost_tuning, _ = tune_xgboost(
        native_train,
        train[outcome_column],
        native_validation,
        validation[outcome_column],
        config,
        binary=True,
        positive_weight=positive_weight,
        sample_weight=train["cluster_equalizing_weight"],
    )
    predict_start = time.perf_counter()
    xgboost_test_probability = xgboost_model.predict_proba(native_test)[:, 1]
    records["xgboost_gpu"] = {
        "model": xgboost_model,
        "validation_probability": xgboost_model.predict_proba(native_validation)[:, 1],
        "test_probability": xgboost_test_probability,
        "backend": "cuda_gpu",
        "fit_seconds": float(
            xgboost_tuning.loc[
                xgboost_tuning["validation_log_loss"].idxmin(), "fit_seconds"
            ]
        ),
        "test_predict_seconds": time.perf_counter() - predict_start,
    }

    tabm_artifact, tabm_validation, tabm_test, tabm_tuning, _ = train_tabm(
        raw_train,
        train[outcome_column],
        raw_validation,
        validation[outcome_column],
        raw_test,
        full_categorical,
        full_numeric,
        config,
        binary=True,
        sample_weight=train["cluster_equalizing_weight"],
    )
    records["tabm_gpu"] = {
        "model": None,
        "validation_probability": tabm_validation[:, 1],
        "test_probability": tabm_test[:, 1],
        "backend": "cuda_gpu",
        "fit_seconds": float(tabm_tuning.loc[0, "fit_seconds"]),
        "test_predict_seconds": float(tabm_tuning.loc[0, "test_predict_seconds"]),
    }
    torch.save(
        tabm_artifact,
        config.models_dir / f"{outcome_name}_tabm_gpu.pt",
    )

    lightgbm_tuning.insert(0, "algorithm", "lightgbm_cpu")
    catboost_tuning.insert(0, "algorithm", "catboost_gpu")
    xgboost_tuning.insert(0, "algorithm", "xgboost_gpu")
    tuning_parts = [lightgbm_tuning, catboost_tuning, xgboost_tuning, tabm_tuning]
    if custom_tuning is not None:
        tuning_parts.append(custom_tuning)
    if custom_moe_tuning is not None:
        tuning_parts.append(custom_moe_tuning)
    if custom_rollout_tuning is not None:
        tuning_parts.append(custom_rollout_tuning)
    tuning = pd.concat(tuning_parts, ignore_index=True)
    tuning.insert(0, "outcome", outcome_name)
    tuning.to_csv(
        config.tables_dir / f"{outcome_name}_boosting_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    calibrated_validation: dict[str, np.ndarray] = {}
    calibrated_test: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    calibrators: dict[str, LogisticRegression] = {}
    for name, record in records.items():
        calibrator = fit_platt(
            validation[outcome_column], record["validation_probability"]
        )
        calibrators[name] = calibrator
        calibrated_validation[name] = apply_platt(
            calibrator, record["validation_probability"]
        )
        calibrated_test[name] = apply_platt(calibrator, record["test_probability"])
        thresholds[name] = choose_threshold(
            validation[outcome_column], calibrated_validation[name]
        )

    ensemble_sources = [
        "status_schedule_logistic",
        "full_logistic",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "lightgbm_cpu",
        "catboost_gpu",
        "xgboost_gpu",
    ]
    validation_source_map = {
        name: np.column_stack(
            [1 - calibrated_validation[name], calibrated_validation[name]]
        )
        for name in ensemble_sources
    }
    test_source_map = {
        name: np.column_stack([1 - calibrated_test[name], calibrated_test[name]])
        for name in ensemble_sources
    }
    equal_validation = np.mean(list(validation_source_map.values()), axis=0)[:, 1]
    equal_test = np.mean(list(test_source_map.values()), axis=0)[:, 1]
    calibrated_validation["equal_weight_pool_ablation"] = equal_validation
    calibrated_test["equal_weight_pool_ablation"] = equal_test
    thresholds["equal_weight_pool_ablation"] = choose_threshold(
        validation[outcome_column], equal_validation
    )
    records["equal_weight_pool_ablation"] = {
        "model": None,
        "backend": "hybrid",
        "fit_seconds": 0.0,
        "test_predict_seconds": 0.0,
    }

    ensemble_start = time.perf_counter()
    proposed_validation_matrix, proposed_test_matrix, ensemble_weights = (
        state_conditioned_delivery_ensemble(
            validation,
            validation_source_map,
            test,
            test_source_map,
            outcome_column,
            [0, 1],
        )
    )
    calibrated_validation["proposed_delivery_ensemble"] = proposed_validation_matrix[
        :, 1
    ]
    calibrated_test["proposed_delivery_ensemble"] = proposed_test_matrix[:, 1]
    thresholds["proposed_delivery_ensemble"] = choose_threshold(
        validation[outcome_column],
        calibrated_validation["proposed_delivery_ensemble"],
    )
    records["proposed_delivery_ensemble"] = {
        "model": {"weights": ensemble_weights, "sources": ensemble_sources},
        "backend": "hybrid",
        "fit_seconds": time.perf_counter() - ensemble_start,
        "test_predict_seconds": 0.0,
    }
    ensemble_weights.to_csv(
        config.tables_dir / f"{outcome_name}_proposed_ensemble_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_columns = [
        "unit_id",
        "development_cluster_id",
        "entity_id",
        "report_month",
        "current_state",
        "technology_group",
        "state_code",
        "analysis_split",
        outcome_column,
    ]
    predictions = test[prediction_columns].reset_index(drop=True)
    performance_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    calibration_tables: list[pd.DataFrame] = []
    decision_tables: list[pd.DataFrame] = []
    bootstrap_tables: list[pd.DataFrame] = []
    for model_index, (name, record) in enumerate(records.items()):
        probability = calibrated_test[name]
        validation_probability = calibrated_validation[name]
        threshold = thresholds[name]
        validation_metrics = binary_metrics(
            validation[outcome_column],
            validation_probability,
            threshold,
            sample_weight=validation["cluster_equalizing_weight"],
        )
        test_metrics = binary_metrics(
            test[outcome_column],
            probability,
            threshold,
            sample_weight=test["cluster_equalizing_weight"],
        )
        common = {
            "outcome": outcome_name,
            "landmark_age_months": landmark_age,
            "model": name,
            "backend": record["backend"],
            "fit_seconds": record["fit_seconds"],
            "test_predict_seconds": record["test_predict_seconds"],
            "train_n": len(train),
            "validation_n": len(validation),
            "test_n": len(test),
            "train_events": int(train[outcome_column].sum()),
            "validation_events": int(validation[outcome_column].sum()),
            "test_events": int(test[outcome_column].sum()),
        }
        performance_rows.append({**common, **test_metrics})
        validation_rows.append({**common, **validation_metrics})
        probability_column = f"{name}_probability"
        predictions[probability_column] = probability
        calibration_tables.append(
            calibration_bins(test[outcome_column], probability, name, outcome_name)
        )
        decision_tables.append(
            decision_curve(test[outcome_column], probability, name, outcome_name)
        )
        bootstrap_frame = predictions[
            ["development_cluster_id", outcome_column, probability_column]
        ].copy()
        bootstrap = cluster_bootstrap_binary_metrics(
            bootstrap_frame,
            outcome_column,
            probability_column,
            threshold,
            "development_cluster_id",
            config.bootstrap_reps,
            config.seed + 500 + model_index,
        )
        bootstrap_tables.append(bootstrap_summary(bootstrap, name, outcome_name))
        artifact = {
            "model": record["model"],
            "calibrator": calibrators.get(name),
            "threshold": threshold,
        }
        joblib.dump(artifact, config.models_dir / f"{outcome_name}_{name}.joblib")

    catboost_model.save_model(str(config.models_dir / f"{outcome_name}_catboost_gpu.cbm"))
    xgboost_model.save_model(config.models_dir / f"{outcome_name}_xgboost_gpu.json")
    joblib.dump(lightgbm_model, config.models_dir / f"{outcome_name}_lightgbm_cpu.joblib")

    importance_tables = []
    for algorithm, model in (
        ("random_forest", records["random_forest"]["model"].named_steps["classifier"]),
        ("extra_trees", records["extra_trees"]["model"].named_steps["classifier"]),
        ("lightgbm_cpu", lightgbm_model),
        ("catboost_gpu", catboost_model),
        ("xgboost_gpu", xgboost_model),
    ):
        importance_tables.append(
            pd.DataFrame(
                {
                    "outcome": outcome_name,
                    "algorithm": algorithm,
                    "feature": [*full_categorical, *full_numeric],
                    "importance": model.feature_importances_,
                }
            )
        )
    importance = pd.concat(importance_tables, ignore_index=True).sort_values(
        ["algorithm", "importance"], ascending=[True, False]
    )
    paired_comparisons = paired_bootstrap_binary_comparisons(
        predictions,
        outcome_column,
        calibrated_test,
        "proposed_delivery_ensemble",
        "development_cluster_id",
        config.bootstrap_reps,
        config.seed + 620,
    )
    custom_deep_comparisons = None
    if "delivery_competing_risk_net_gpu" in calibrated_test:
        custom_deep_comparisons = paired_bootstrap_binary_comparisons(
            predictions,
            outcome_column,
            calibrated_test,
            "delivery_competing_risk_net_gpu",
            "development_cluster_id",
            config.bootstrap_reps,
            config.seed + 621,
        )
    custom_moe_comparisons = None
    if "delivery_sparse_moe_gpu" in calibrated_test:
        custom_moe_comparisons = paired_bootstrap_binary_comparisons(
            predictions,
            outcome_column,
            calibrated_test,
            "delivery_sparse_moe_gpu",
            "development_cluster_id",
            config.bootstrap_reps,
            config.seed + 622,
        )
    path_moe_comparisons = None
    if "delivery_path_moe_gpu" in calibrated_test:
        path_moe_comparisons = paired_bootstrap_binary_comparisons(
            predictions,
            outcome_column,
            calibrated_test,
            "delivery_path_moe_gpu",
            "development_cluster_id",
            config.bootstrap_reps,
            config.seed + 623,
        )
    performance = pd.DataFrame(performance_rows)
    validation_performance = pd.DataFrame(validation_rows)
    calibration = pd.concat(calibration_tables, ignore_index=True)
    decisions = pd.concat(decision_tables, ignore_index=True)
    bootstraps = pd.concat(bootstrap_tables, ignore_index=True)
    predictions.to_parquet(
        config.predictions_dir / f"{outcome_name}_test_predictions.parquet",
        index=False,
    )
    for name, table in (
        (f"{outcome_name}_model_performance.csv", performance),
        (f"{outcome_name}_validation_performance.csv", validation_performance),
        (f"{outcome_name}_calibration_bins.csv", calibration),
        (f"{outcome_name}_decision_curve.csv", decisions),
        (f"{outcome_name}_bootstrap_ci.csv", bootstraps),
        (f"{outcome_name}_proposed_paired_comparisons.csv", paired_comparisons),
        (f"{outcome_name}_feature_importance.csv", importance),
    ):
        table.to_csv(config.tables_dir / name, index=False, encoding="utf-8-sig")
    if custom_deep_comparisons is not None:
        custom_deep_comparisons.to_csv(
            config.tables_dir / f"{outcome_name}_custom_deep_paired_comparisons.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if custom_moe_comparisons is not None:
        custom_moe_comparisons.to_csv(
            config.tables_dir / f"{outcome_name}_delivery_sparse_moe_paired_comparisons.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if path_moe_comparisons is not None:
        path_moe_comparisons.to_csv(
            config.tables_dir / f"{outcome_name}_delivery_path_moe_paired_comparisons.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return {
        f"{outcome_name}_performance": performance,
        f"{outcome_name}_validation_performance": validation_performance,
        f"{outcome_name}_calibration": calibration,
        f"{outcome_name}_decision_curve": decisions,
        f"{outcome_name}_bootstrap": bootstraps,
        f"{outcome_name}_importance": importance,
        f"{outcome_name}_tuning": tuning,
        f"{outcome_name}_ensemble_weights": ensemble_weights,
        f"{outcome_name}_paired_comparisons": paired_comparisons,
        f"{outcome_name}_custom_deep_comparisons": custom_deep_comparisons,
        f"{outcome_name}_custom_moe_comparisons": custom_moe_comparisons,
        f"{outcome_name}_path_moe_comparisons": path_moe_comparisons,
    }


def adjusted_association_models(
    config: AnalysisConfig, landmarks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    entry = landmarks.loc[landmarks["landmark_age_months"] == 0].copy()
    entry["entry_year_centered"] = (
        entry["report_month"].astype(str).str.slice(0, 4).astype(float) - 2021.0
    )
    entry["entity_cluster"] = entry["entity_id"].fillna(entry["unit_id"]).astype(str)
    entry["delivery_phase"] = entry["current_state"].map(
        {
            "P": "early",
            "L": "early",
            "T": "construction",
            "U": "construction",
            "V": "advanced",
            "TS": "advanced",
        }
    )
    entry["technology_collapsed"] = entry["technology_group"].where(
        entry["technology_group"].isin(["solar", "storage", "wind"]),
        "other_clean",
    )
    entry["sector_collapsed"] = entry["sector_group"].where(
        entry["sector_group"].isin(
            ["independent_power_producer", "electric_utility"]
        ),
        "other_sector",
    )
    categorical = ["delivery_phase", "technology_collapsed", "sector_collapsed"]
    numeric = [
        "entry_year_centered",
        "log1p_capacity_mw",
        "months_to_reported_operation",
        "concurrent_entity_pipeline_units",
        "prior_entity_completions",
    ]
    for column in categorical:
        entry[column] = entry[column].fillna("UNKNOWN").astype(str)
    for column in numeric:
        values = pd.to_numeric(entry[column], errors="coerce")
        entry[column] = values.fillna(values.median())
    formula = (
        "{outcome} ~ C(delivery_phase, Treatment(reference='early')) "
        "+ C(technology_collapsed, Treatment(reference='solar')) "
        "+ C(sector_collapsed, Treatment(reference='independent_power_producer')) "
        "+ entry_year_centered + log1p_capacity_mw "
        "+ months_to_reported_operation + concurrent_entity_pipeline_units "
        "+ prior_entity_completions"
    )
    coefficient_tables: list[pd.DataFrame] = []
    model_audit_rows: list[dict[str, Any]] = []
    for outcome, observable in (
        ("operation_within_12_months", "observable_12_month_outcome"),
        ("cancellation_within_24_months", "observable_24_month_outcome"),
    ):
        observable_data = entry.loc[entry[observable]].copy()
        data = observable_data.loc[
            observable_data["delivery_phase"].notna()
            & observable_data["delivery_phase"].ne("UNKNOWN")
        ].copy()
        try:
            model = smf.glm(
                formula=formula.format(outcome=outcome),
                data=data,
                family=sm.families.Binomial(),
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": data["entity_cluster"]},
                maxiter=200,
            )
            finite = np.isfinite(model.params).all() and np.isfinite(model.bse).all()
            converged = bool(getattr(model, "converged", False))
            if not finite or not converged:
                raise RuntimeError(
                    f"Unstable GLM: converged={converged}, finite={finite}"
                )
            intervals = model.conf_int()
            table = pd.DataFrame(
                {
                    "outcome": outcome,
                    "term": model.params.index,
                    "log_odds": model.params.values,
                    "standard_error": model.bse.values,
                    "odds_ratio": np.exp(model.params.values),
                    "ci95_low": np.exp(intervals[0].values),
                    "ci95_high": np.exp(intervals[1].values),
                    "p_value": model.pvalues.values,
                }
            )
            coefficient_tables.append(table)
            model_audit_rows.append(
                {
                    "outcome": outcome,
                    "status": "fitted",
                    "n": len(data),
                    "events": int(data[outcome].sum()),
                    "clusters": data["entity_cluster"].nunique(),
                    "excluded_sparse_or_nonplanned_entry_states": len(observable_data)
                    - len(data),
                    "cluster_robust_covariance": True,
                    "cluster_equalizing_weights_used": False,
                    "collinear_initial_lead_removed": True,
                    "status_collapsed_to_three_delivery_phases": True,
                    "converged": converged,
                    "aic": model.aic,
                    "deviance": model.deviance,
                    "iterations": model.fit_history.get("iteration"),
                }
            )
        except Exception as error:
            model_audit_rows.append(
                {
                    "outcome": outcome,
                    "status": "failed",
                    "n": len(data),
                    "events": int(data[outcome].sum()),
                    "clusters": data["entity_cluster"].nunique(),
                    "excluded_sparse_or_nonplanned_entry_states": len(observable_data)
                    - len(data),
                    "error": repr(error),
                }
            )
    coefficients = (
        pd.concat(coefficient_tables, ignore_index=True)
        if coefficient_tables
        else pd.DataFrame()
    )
    audit = pd.DataFrame(model_audit_rows)
    coefficients.to_csv(
        config.tables_dir / "adjusted_association_odds_ratios.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(
        config.tables_dir / "adjusted_association_model_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return coefficients, audit


def subgroup_generalization(
    config: AnalysisConfig,
    landmarks: pd.DataFrame,
    outcome_name: str,
    outcome_column: str,
) -> pd.DataFrame:
    predictions = pd.read_parquet(
        config.predictions_dir / f"{outcome_name}_test_predictions.parquet"
    )
    train_entities = set(
        landmarks.loc[
            (landmarks["landmark_age_months"] == 0)
            & (landmarks["analysis_split"] == "train"),
            "entity_id",
        ].dropna()
    )
    predictions["entity_exposure"] = np.where(
        predictions["entity_id"].isin(train_entities), "seen_entity", "unseen_entity"
    )
    rows = []
    probability_columns = [
        column for column in predictions.columns if column.endswith("_probability")
    ]
    for model_column in probability_columns:
        model_name = model_column.removesuffix("_probability")
        for dimension in ("technology_group", "current_state", "entity_exposure"):
            for subgroup, data in predictions.groupby(dimension, dropna=False):
                if len(data) < 30 or data[outcome_column].nunique() < 2:
                    continue
                probability = data[model_column].to_numpy()
                rows.append(
                    {
                        "outcome": outcome_name,
                        "model": model_name,
                        "dimension": dimension,
                        "subgroup": subgroup,
                        "n": len(data),
                        "events": int(data[outcome_column].sum()),
                        "roc_auc": roc_auc_score(data[outcome_column], probability),
                        "average_precision": average_precision_score(
                            data[outcome_column], probability
                        ),
                        "brier": brier_score_loss(data[outcome_column], probability),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(
        config.tables_dir / f"{outcome_name}_subgroup_generalization.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def train_and_evaluate_models(
    config: AnalysisConfig,
    transitions: pd.DataFrame,
    landmarks: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    next_state_outputs = train_next_state_models(config, transitions)
    transition_moe_artifact = next_state_outputs.pop(
        "next_state_delivery_moe_artifact"
    )
    outputs.update(next_state_outputs)
    custom_deep = train_custom_deep_landmark_model(config, landmarks)
    outputs["landmark_custom_deep_tuning"] = custom_deep["tuning"]
    custom_moe = train_custom_moe_landmark_model(config, landmarks)
    outputs["landmark_custom_moe_tuning"] = custom_moe["tuning"]
    custom_rollout = train_delivery_path_moe_landmark_model(
        config, landmarks, transition_moe_artifact
    )
    outputs["landmark_delivery_path_moe_tuning"] = custom_rollout["tuning"]
    outputs["landmark_delivery_path_moe_audit"] = custom_rollout["audit"]
    operation = train_binary_landmark_models(
        config,
        landmarks,
        "operation_within_12_months",
        "observable_12_month_outcome",
        "operation_12m",
        landmark_age=0,
        custom_deep=custom_deep,
        custom_moe=custom_moe,
        custom_rollout=custom_rollout,
    )
    cancellation = train_binary_landmark_models(
        config,
        landmarks,
        "cancellation_within_24_months",
        "observable_24_month_outcome",
        "cancellation_24m",
        landmark_age=0,
        custom_deep=custom_deep,
        custom_moe=custom_moe,
        custom_rollout=custom_rollout,
    )
    outputs.update(operation)
    outputs.update(cancellation)
    coefficients, association_audit = adjusted_association_models(config, landmarks)
    outputs["adjusted_associations"] = coefficients
    outputs["adjusted_association_audit"] = association_audit
    outputs["operation_12m_subgroups"] = subgroup_generalization(
        config, landmarks, "operation_12m", "operation_within_12_months"
    )
    outputs["cancellation_24m_subgroups"] = subgroup_generalization(
        config, landmarks, "cancellation_24m", "cancellation_within_24_months"
    )
    return outputs
