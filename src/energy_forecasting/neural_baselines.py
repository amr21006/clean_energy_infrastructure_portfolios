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
from energy_forecasting.models import RNNForecaster, TCNForecaster
from energy_forecasting.train import resolve_device


@dataclass
class NeuralBaselineResult:
    models: dict[str, nn.Module]
    history: pd.DataFrame
    prediction_frames: pd.DataFrame
    metric_frame: pd.DataFrame
    diagnostics: dict[str, Any]


def train_neural_baselines(
    arrays: dict[str, dict[str, np.ndarray]],
    baseline_cfg: dict[str, Any],
    target_scaler: TargetScalerState,
    seed: int,
    device_name: str,
) -> NeuralBaselineResult:
    enabled = list(baseline_cfg.get("enabled", []))
    if not enabled:
        return NeuralBaselineResult(
            models={},
            history=pd.DataFrame(),
            prediction_frames=pd.DataFrame(),
            metric_frame=pd.DataFrame(),
            diagnostics={"enabled": []},
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    n_features = int(arrays["train"]["X"].shape[2])
    models: dict[str, nn.Module] = {}
    histories: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {"enabled": enabled, "device": str(device), "models": {}}

    for name in enabled:
        print(f"[neural_baseline:{name}] training started", flush=True)
        model_start = time.perf_counter()
        model = _make_sequence_model(name, n_features, baseline_cfg).to(device)
        model, history = _fit_model(
            model=model,
            arrays=arrays,
            cfg=baseline_cfg,
            seed=seed,
            device=device,
        )
        pred_frame, metric_frame = _evaluate_model(
            model=model,
            arrays=arrays,
            target_scaler=target_scaler,
            device=device,
            model_name=name,
        )
        params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        diagnostics["models"][name] = {
            "total_parameter_count": int(params),
            "trainable_parameter_count": int(trainable_params),
            "epochs_ran": int(len(history)),
            "training_seconds": float(time.perf_counter() - model_start),
        }
        history.insert(0, "model", name)
        histories.append(history)
        metrics.append(metric_frame)
        predictions.append(pred_frame)
        models[name] = model
        print(f"[neural_baseline:{name}] training complete", flush=True)

    return NeuralBaselineResult(
        models=models,
        history=pd.concat(histories, ignore_index=True) if histories else pd.DataFrame(),
        prediction_frames=pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        metric_frame=pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(),
        diagnostics=diagnostics,
    )


def _make_sequence_model(name: str, n_features: int, cfg: dict[str, Any]) -> nn.Module:
    hidden_size = int(cfg.get("hidden_size", 64))
    dropout = float(cfg.get("dropout", 0.1))
    if name in {"lstm", "gru"}:
        return RNNForecaster(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=int(cfg.get("num_layers", 1)),
            dropout=dropout,
            cell_type=name,
        )
    if name in {"tcn", "residual_tcn"}:
        return TCNForecaster(
            n_features=n_features,
            channels=int(cfg.get("tcn_channels", hidden_size)),
            levels=int(cfg.get("tcn_levels", 3)),
            kernel_size=int(cfg.get("tcn_kernel_size", 3)),
            dropout=dropout,
            residual_feature_index=0 if name == "residual_tcn" else None,
        )
    raise ValueError(f"Unsupported neural baseline: {name}")


def _fit_model(
    model: nn.Module,
    arrays: dict[str, dict[str, np.ndarray]],
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame]:
    torch.manual_seed(seed)
    pin_memory = device.type == "cuda"
    train_loader = _loader(arrays["train"], int(cfg.get("batch_size", 256)), shuffle=True, pin_memory=pin_memory)
    val_loader = _loader(arrays["validation"], int(cfg.get("batch_size", 256)), shuffle=False, pin_memory=pin_memory)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 0.001)),
        weight_decay=float(cfg.get("weight_decay", 0.0001)),
    )
    loss_fn = nn.MSELoss()
    use_amp = bool(cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    patience = int(cfg.get("patience", 5))
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, int(cfg.get("max_epochs", 20)) + 1):
        model.train()
        losses: list[float] = []
        for X, y_scaled in train_loader:
            X = X.to(device, non_blocking=True)
            y_scaled = y_scaled.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(X)
                loss = loss_fn(pred, y_scaled)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val_rmse = _scaled_rmse(model, val_loader, device)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_rmse_scaled": val_rmse,
            }
        )
        print(
            f"[sequence_model] epoch={epoch} train_loss={train_loss:.6f} validation_rmse_scaled={val_rmse:.6f}",
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
    return model, pd.DataFrame(history_rows)


def _loader(split_arrays: dict[str, np.ndarray], batch_size: int, shuffle: bool, pin_memory: bool = False) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(split_arrays["X"], dtype=torch.float32),
        torch.tensor(split_arrays["y_scaled"], dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


def _scaled_rmse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    preds: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    with torch.no_grad():
        for X, y_scaled in loader:
            pred = model(X.to(device, non_blocking=True)).detach().cpu().numpy()
            preds.append(pred)
            actuals.append(y_scaled.numpy())
    return float(np.mean((np.concatenate(preds) - np.concatenate(actuals)) ** 2) ** 0.5)


def _evaluate_model(
    model: nn.Module,
    arrays: dict[str, dict[str, np.ndarray]],
    target_scaler: TargetScalerState,
    device: torch.device,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, float | str]] = []
    model.eval()
    for split in ["validation", "test"]:
        split_loader = _loader(arrays[split], batch_size=4096, shuffle=False, pin_memory=device.type == "cuda")
        pred_scaled_batches: list[np.ndarray] = []
        with torch.no_grad():
            for X, _ in split_loader:
                pred_scaled_batches.append(model(X.to(device, non_blocking=True)).detach().cpu().numpy())
        pred_scaled = np.concatenate(pred_scaled_batches)
        y_pred = target_scaler.inverse_transform(pred_scaled)
        y_true = arrays[split]["y"]
        split_metrics = regression_metrics(y_true, y_pred)
        split_metrics.update({"model": model_name, "split": split})
        rows.append(split_metrics)
        frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
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
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rows)
