from __future__ import annotations

import torch
from torch import nn


class WindowGRU(nn.Module):
    """Per-window temporal encoder followed by a GRU across context windows."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.window_encoder = nn.Sequential(
            nn.Conv1d(input_channels, self.hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,K,T,C], got {tuple(x.shape)}")
        bsz, context_len, window_size, channels = x.shape
        flat = x.reshape(bsz * context_len, window_size, channels).permute(0, 2, 1)
        embeddings = self.window_encoder(flat).squeeze(-1).reshape(bsz, context_len, self.hidden_dim)
        outputs, _ = self.gru(embeddings)
        logits = self.classifier(outputs)
        return {"logits": logits}
