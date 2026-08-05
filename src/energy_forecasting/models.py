from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AnchoredTemporalTransformer(nn.Module):
    """Supervised temporal Transformer with an optional persistence anchor.

    The network predicts a standardized correction to the most recent observed
    load when ``residual_feature_index`` is not ``None``. Any optional ramp
    regularizer is implemented in the training routine, not in this architecture.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
        residual_feature_index: int | None = 0,
    ) -> None:
        super().__init__()
        self.residual_feature_index = residual_feature_index
        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_projection(x)
        z = self.positional_encoding(z)
        z = self.encoder(z)
        pooled = z[:, -1, :]
        forecast = self.head(pooled).squeeze(-1)
        if self.residual_feature_index is not None:
            forecast = forecast + x[:, -1, self.residual_feature_index]
        return forecast


# Backward-compatible import name for the originally reviewed code and model
# checkpoints. Revised experiments use ``AnchoredTemporalTransformer``.
PGSSTT = AnchoredTemporalTransformer


class RNNForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        cell_type: str,
    ) -> None:
        super().__init__()
        if cell_type not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported RNN cell type: {cell_type}")
        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.pad(x, (self.left_padding, 0)))
        y = self.dropout(F.gelu(self.norm1(y)))
        y = self.conv2(F.pad(y, (self.left_padding, 0)))
        y = self.dropout(F.gelu(self.norm2(y)))
        return F.gelu(y + self.residual(x))


class TCNForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: int,
        levels: int,
        kernel_size: int,
        dropout: float,
        residual_feature_index: int | None = None,
    ) -> None:
        super().__init__()
        self.residual_feature_index = residual_feature_index
        blocks: list[nn.Module] = []
        in_channels = n_features
        for level in range(levels):
            blocks.append(
                TemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            in_channels = channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, 1),
        )
        if self.residual_feature_index is not None:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.network(z)
        forecast = self.head(z[:, :, -1]).squeeze(-1)
        if self.residual_feature_index is not None:
            forecast = forecast + x[:, -1, self.residual_feature_index]
        return forecast
