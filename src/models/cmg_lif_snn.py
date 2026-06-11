from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .surrogate import inverse_softplus, surrogate_spike


class CMGLIFSNN(nn.Module):
    """Context-memory gated LIF SNN with adaptive thresholds."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        beta: float = 0.9,
        alpha: float = 0.8,
        theta_init: float = 1.0,
        theta_scale: float = 0.5,
        dropout: float = 0.2,
        threshold_modulation: bool = True,
        context_memory: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.theta_scale = float(theta_scale)
        self.threshold_modulation = bool(threshold_modulation)
        self.context_memory = bool(context_memory)
        self.encoder = nn.Linear(input_channels, hidden_dim)
        self.memory_to_theta = nn.Linear(hidden_dim, hidden_dim)
        self.theta_raw = nn.Parameter(torch.full((hidden_dim,), inverse_softplus(theta_init)))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,K,T,C], got {tuple(x.shape)}")
        bsz, context_len, _, _ = x.shape
        memory = x.new_zeros((bsz, self.hidden_dim))
        logits_by_window = []
        spike_rates = []
        context_states = []

        for window_idx in range(context_len):
            currents = self.encoder(x[:, window_idx])
            theta = self._threshold_from_memory(memory)
            spikes = self._run_lif(currents, theta)
            spike_repr = spikes.mean(dim=1)
            logits_by_window.append(self.classifier(spike_repr))
            spike_rates.append(spikes.mean())

            if self.context_memory:
                memory = self.alpha * memory + (1.0 - self.alpha) * spike_repr
            else:
                memory = torch.zeros_like(memory)
            context_states.append(memory)

        logits = torch.stack(logits_by_window, dim=1)
        return {
            "logits": logits,
            "spike_rate": torch.stack(spike_rates).mean(),
            "context_states": torch.stack(context_states, dim=1),
        }

    def _threshold_from_memory(self, memory: torch.Tensor) -> torch.Tensor:
        base = F.softplus(self.theta_raw).reshape(1, -1) + 1e-3
        if not self.threshold_modulation:
            return base.expand(memory.shape[0], -1)
        delta = self.theta_scale * torch.tanh(self.memory_to_theta(memory))
        return torch.clamp(base + delta, min=0.05)

    def _run_lif(self, currents: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        membrane = currents.new_zeros((currents.shape[0], currents.shape[-1]))
        spikes = []
        for time_idx in range(currents.shape[1]):
            membrane = self.beta * membrane + currents[:, time_idx]
            spike = surrogate_spike(membrane - theta)
            membrane = membrane - spike * theta
            spikes.append(spike)
        return torch.stack(spikes, dim=1)
