from __future__ import annotations

import torch
from torch import nn

from .ms_cmg_lif_snn import MultiScaleTemporalEncoder


class MSCNN1D(nn.Module):
    """Multi-scale temporal CNN baseline."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        branch_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = MultiScaleTemporalEncoder(input_channels, int(hidden_dim), int(branch_dim), dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(int(hidden_dim), num_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,K,T,C], got {tuple(x.shape)}")
        bsz, context_len, window_size, channels = x.shape
        flat = x.reshape(bsz * context_len, window_size, channels).permute(0, 2, 1)
        features = self.pool(self.encoder(flat)).squeeze(-1)
        logits = self.classifier(features).reshape(bsz, context_len, -1)
        return {"logits": logits}
