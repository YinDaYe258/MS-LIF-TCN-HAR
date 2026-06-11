from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .surrogate import inverse_softplus, surrogate_spike


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int, branch_dim: int, dropout: float) -> None:
        super().__init__()
        self.branch3 = self._branch(input_channels, branch_dim, kernel_size=3, padding=1)
        self.branch5 = self._branch(input_channels, branch_dim, kernel_size=5, padding=2)
        self.branch9 = self._branch(input_channels, branch_dim, kernel_size=9, padding=4)
        self.project = nn.Sequential(
            nn.Conv1d(branch_dim * 3, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _branch(input_channels: int, branch_dim: int, kernel_size: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(input_channels, branch_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(branch_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat([self.branch3(x), self.branch5(x), self.branch9(x)], dim=1)
        return self.project(features)


class MSCMGLIFSNN(nn.Module):
    """Multi-scale temporal encoder with lightweight context-memory gated LIF dynamics."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        branch_dim: int = 32,
        beta: float = 0.9,
        alpha: float = 0.8,
        theta_init: float = 1.0,
        theta_scale: float = 0.1,
        num_groups: int = 8,
        dropout: float = 0.2,
        threshold_modulation: bool = True,
        context_memory: bool = True,
    ) -> None:
        super().__init__()
        if num_groups < 1:
            raise ValueError("num_groups must be >= 1")
        if num_groups > hidden_dim:
            raise ValueError("num_groups must be <= hidden_dim")
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.theta_scale = float(theta_scale)
        self.num_groups = int(num_groups)
        self.threshold_modulation = bool(threshold_modulation)
        self.context_memory = bool(context_memory)

        self.encoder = MultiScaleTemporalEncoder(input_channels, self.hidden_dim, int(branch_dim), dropout)
        self.memory_to_gate = nn.Linear(self.hidden_dim, self.num_groups)
        self.theta_raw = nn.Parameter(torch.full((self.hidden_dim,), inverse_softplus(theta_init)))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.hidden_dim, num_classes))

        nn.init.zeros_(self.memory_to_gate.weight)
        nn.init.zeros_(self.memory_to_gate.bias)
        group_ids = torch.div(
            torch.arange(self.hidden_dim) * self.num_groups,
            self.hidden_dim,
            rounding_mode="floor",
        ).long()
        self.register_buffer("group_ids", group_ids, persistent=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,K,T,C], got {tuple(x.shape)}")
        bsz, context_len, window_size, channels = x.shape
        flat = x.reshape(bsz * context_len, window_size, channels).permute(0, 2, 1)
        encoded = self.encoder(flat).permute(0, 2, 1).reshape(bsz, context_len, window_size, self.hidden_dim)

        memory = x.new_zeros((bsz, self.hidden_dim))
        logits_by_window = []
        spike_rates = []
        context_states = []

        for window_idx in range(context_len):
            currents = encoded[:, window_idx]
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

        return {
            "logits": torch.stack(logits_by_window, dim=1),
            "spike_rate": torch.stack(spike_rates).mean(),
            "context_states": torch.stack(context_states, dim=1),
        }

    def _threshold_from_memory(self, memory: torch.Tensor) -> torch.Tensor:
        base = F.softplus(self.theta_raw).reshape(1, -1) + 1e-3
        if not self.threshold_modulation:
            return base.expand(memory.shape[0], -1)
        gate = torch.tanh(self.memory_to_gate(memory))
        gate_hidden = gate[:, self.group_ids]
        theta = base * (1.0 + self.theta_scale * gate_hidden)
        return torch.clamp(theta, min=0.5, max=2.5)

    def _run_lif(self, currents: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        membrane = currents.new_zeros((currents.shape[0], currents.shape[-1]))
        spikes = []
        for time_idx in range(currents.shape[1]):
            membrane = self.beta * membrane + currents[:, time_idx]
            spike = surrogate_spike(membrane - theta)
            membrane = membrane - spike * theta
            spikes.append(spike)
        return torch.stack(spikes, dim=1)
