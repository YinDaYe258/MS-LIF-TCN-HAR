from __future__ import annotations

import torch
from torch import nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,K,T,C], got {tuple(x.shape)}")
        bsz, context_len, window_size, channels = x.shape
        flat = x.reshape(bsz * context_len, window_size, channels)
        output, _ = self.gru(flat)
        features = output[:, -1]
        logits = self.classifier(features).reshape(bsz, context_len, -1)
        return {"logits": logits}
