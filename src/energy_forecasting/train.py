from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from energy_forecasting.features import TargetScalerState
from energy_forecasting.metrics import regression_metrics
from energy_forecasting.models import AnchoredTemporalTransformer


@dataclass
class TrainResult:
    model: AnchoredTemporalTransformer
    history: pd.DataFrame
    prediction_frames: pd.DataFrame
    metric_frame: pd.DataFrame
    diagnostics: dict[str, Any]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    return torch.device(requested)


def train_anchored_transformer(
    arrays: dict[str, dict[str, np.ndarray]],
    model_cfg: dict[str, Any],
    target_scaler: TargetScalerState,
    seed: int,
    device_name: str,
) -> TrainResult:
    start_time = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    n_features = arrays["train"]["X"].shape[2]
    model = AnchoredTemporalTransformer(
        n_features=n_features,
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
        residual_feature_index=model_cfg.get("residual_feature_index", 0),
    ).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    train_loader = _loader(
        arrays["train"],
        int(model_cfg["batch_size"]),
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = _loader(
        arrays["validation"],
        int(model_cfg["batch_size"]),
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
    )
    mse = nn.MSELoss()
    use_amp = bool(model_cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ramp_weight = float(
        model_cfg.get("ramp_weight", model_cfg.get("physics_weight", 0.0))
    )
    ramp_base_tolerance = float(model_cfg.get("ramp_base_tolerance", 0.25))
    ramp_temperature_coefficient = float(
        model_cfg.get("ramp_temperature_coefficient", 3.0)
    )
    ramp_temperature_feature_index = int(
        model_cfg.get("ramp_temperature_feature_index", 1)
    )
    reporting_name = str(model_cfg.get("reporting_name", "pa_stt"))
    best_state = None
    best_val = float("inf")
    patience = int(model_cfg.get("patience", 5))
    stale_epochs = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, int(model_cfg["max_epochs"]) + 1):
        model.train()
        train_losses: list[float] = []
        for X, y_scaled, last_target in train_loader:
            X = X.to(device, non_blocking=True)
            y_scaled = y_scaled.to(device, non_blocking=True)
            last_target = last_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_scaled = model(X)
                loss = mse(pred_scaled, y_scaled)
                if ramp_weight > 0:
                    last_scaled = (last_target - target_scaler.mean) / target_scaler.std
                    if X.shape[-1] <= ramp_temperature_feature_index:
                        raise ValueError(
                            "ramp_temperature_feature_index is outside the feature array"
                        )
                    temp_delta = torch.abs(
                        X[:, -1, ramp_temperature_feature_index]
                        - X[:, 0, ramp_temperature_feature_index]
                    )
                    tolerance = (
                        ramp_base_tolerance
                        + ramp_temperature_coefficient * temp_delta
                    )
                    ramp_loss = torch.relu(
                        torch.abs(pred_scaled - last_scaled) - tolerance
                    ).mean()
                    loss = loss + ramp_weight * ramp_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        val_rmse = _scaled_rmse(model, val_loader, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "validation_rmse_scaled": val_rmse})
        print(
            f"[{reporting_name}] epoch={epoch} train_loss={train_loss:.6f} validation_rmse_scaled={val_rmse:.6f}",
            flush=True,
        )

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = pd.DataFrame(history_rows)
    prediction_frames, metric_frame = evaluate_anchored_transformer(
        model,
        arrays,
        target_scaler,
        device,
        reporting_name=reporting_name,
        batch_size=int(
            model_cfg.get(
                "evaluation_batch_size",
                model_cfg.get("batch_size", 2048),
            )
        ),
    )
    diagnostics = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "total_parameter_count": int(total_params),
        "trainable_parameter_count": int(trainable_params),
        "epochs_ran": int(len(history)),
        "best_validation_rmse_scaled": float(best_val),
        "training_seconds": float(time.perf_counter() - start_time),
        "reporting_name": reporting_name,
        "residual_feature_index": model_cfg.get("residual_feature_index", 0),
        "ramp_regularizer": {
            "weight": ramp_weight,
            "base_tolerance": ramp_base_tolerance,
            "temperature_coefficient": ramp_temperature_coefficient,
            "temperature_feature_index": ramp_temperature_feature_index,
            "classification": "heuristic_domain_informed_regularizer",
        },
    }
    return TrainResult(
        model=model,
        history=history,
        prediction_frames=prediction_frames,
        metric_frame=metric_frame,
        diagnostics=diagnostics,
    )


def _loader(split_arrays: dict[str, np.ndarray], batch_size: int, shuffle: bool, pin_memory: bool = False) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(split_arrays["X"], dtype=torch.float32),
        torch.tensor(split_arrays["y_scaled"], dtype=torch.float32),
        torch.tensor(split_arrays["last_target"], dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


def _scaled_rmse(
    model: AnchoredTemporalTransformer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    preds: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    with torch.no_grad():
        for X, y_scaled, _ in loader:
            pred = model(X.to(device, non_blocking=True)).detach().cpu().numpy()
            preds.append(pred)
            actuals.append(y_scaled.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(actuals)
    return float(np.mean((y_pred - y_true) ** 2) ** 0.5)


def evaluate_anchored_transformer(
    model: AnchoredTemporalTransformer,
    arrays: dict[str, dict[str, np.ndarray]],
    target_scaler: TargetScalerState,
    device: torch.device,
    reporting_name: str = "pa_stt",
    batch_size: int = 2048,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if batch_size < 1:
        raise ValueError("Evaluation batch_size must be at least 1.")
    frames: list[pd.DataFrame] = []
    metrics: list[dict[str, float | str]] = []
    model.eval()
    for split in ["validation", "test"]:
        split_X = arrays[split]["X"]
        scaled_batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(split_X), batch_size):
                X_batch = torch.as_tensor(
                    split_X[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                scaled_batches.append(
                    model(X_batch).detach().cpu().numpy()
                )
        pred_scaled = (
            np.concatenate(scaled_batches)
            if scaled_batches
            else np.empty(0, dtype=float)
        )
        y_pred = target_scaler.inverse_transform(pred_scaled)
        y_true = arrays[split]["y"]
        split_metrics = regression_metrics(y_true, y_pred)
        split_metrics.update({"model": reporting_name, "split": split})
        metrics.append(split_metrics)
        frames.append(
            pd.DataFrame(
                {
                    "model": reporting_name,
                    "split": split,
                    "timestamp": arrays[split]["timestamp"],
                    "window_start_timestamp": arrays[split]["window_start_timestamp"],
                    "window_end_timestamp": arrays[split]["window_end_timestamp"],
                    "building_id": arrays[split]["building_id"],
                    "site_id": arrays[split].get(
                        "site_id",
                        np.full(len(y_true), "", dtype=str),
                    ),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "residual": y_pred - y_true,
                }
            )
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(metrics)


# Backward-compatible function aliases for the originally reviewed scripts.
train_pg_sstt = train_anchored_transformer
evaluate_pg_sstt = evaluate_anchored_transformer
