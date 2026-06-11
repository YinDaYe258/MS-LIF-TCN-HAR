from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .surrogate import inverse_softplus, surrogate_spike


class LIFSNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        beta: float = 0.9,
        theta_init: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = float(beta)
        self.encoder = nn.Linear(input_channels, hidden_dim)
        self.theta_raw = nn.Parameter(torch.full((hidden_dim,), inverse_softplus(theta_init)))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,K,T,C], got {tuple(x.shape)}")
        bsz, context_len, window_size, channels = x.shape
        flat = x.reshape(bsz * context_len, window_size, channels)
        currents = self.encoder(flat)
        theta = F.softplus(self.theta_raw) + 1e-3
        spikes = self._run_lif(currents, theta)
        rates = spikes.mean(dim=1)
        logits = self.classifier(rates).reshape(bsz, context_len, -1)
        return {
            "logits": logits,
            "spike_rate": spikes.mean(),
            "spike_repr": rates.reshape(bsz, context_len, self.hidden_dim),
        }

    def _run_lif(self, currents: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        membrane = currents.new_zeros((currents.shape[0], currents.shape[-1]))
        spikes = []
        theta = theta.reshape(1, -1)
        for time_idx in range(currents.shape[1]):
            membrane = self.beta * membrane + currents[:, time_idx]
            spike = surrogate_spike(membrane - theta)
            membrane = membrane - spike * theta
            spikes.append(spike)
        return torch.stack(spikes, dim=1)
