from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ms_cmg_lif_snn import MultiScaleTemporalEncoder
from .surrogate import inverse_softplus, surrogate_spike


class CausalDepthwiseTCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = int((kernel_size - 1) * dilation)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.pad(x, (self.left_padding, 0))
        out = self.depthwise(out)
        out = self.pointwise(out)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return residual + out


class WindowTemporalTCN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
        dilations: tuple[int, ...] | None = None,
        tcn_layers: int = 2,
    ) -> None:
        super().__init__()
        if dilations is None:
            if tcn_layers < 1:
                raise ValueError("tcn_layers must be >= 1")
            dilations = tuple(2**idx for idx in range(int(tcn_layers)))
        self.blocks = nn.Sequential(
            *[CausalDepthwiseTCNBlock(hidden_dim, kernel_size, dilation, dropout) for dilation in dilations]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,K,H], got {tuple(x.shape)}")
        out = self.blocks(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(out)


class WindowAttentionGate(nn.Module):
    """Causal attention pooling over window-level representations."""

    def __init__(self, hidden_dim: int, attention_hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, int(attention_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(attention_hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,K,H], got {tuple(x.shape)}")
        scores = self.scorer(x).squeeze(-1)
        context_len = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(context_len, context_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        masked_scores = scores[:, None, :].expand(-1, context_len, -1).masked_fill(causal_mask[None], -1e9)
        weights = torch.softmax(masked_scores, dim=-1)
        attended = torch.matmul(weights, x)
        return attended, weights


class SingleScaleTemporalEncoder(nn.Module):
    """Single-kernel temporal encoder used only for multi-scale ablation."""

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        branch_dim: int,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("single-scale kernel size must be a positive odd integer")
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, branch_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(branch_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(branch_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def build_window_encoder(
    input_channels: int,
    hidden_dim: int,
    branch_dim: int,
    dropout: float,
    encoder_mode: str = "multi",
    single_kernel_size: int = 5,
) -> nn.Module:
    mode = str(encoder_mode).lower().replace("-", "_")
    if mode in {"multi", "multiscale", "multi_scale"}:
        return MultiScaleTemporalEncoder(input_channels, hidden_dim, branch_dim, dropout)
    if mode in {"single", "single_scale"}:
        return SingleScaleTemporalEncoder(
            input_channels,
            hidden_dim,
            branch_dim,
            kernel_size=int(single_kernel_size),
            dropout=dropout,
        )
    raise ValueError(f"Unknown encoder_mode: {encoder_mode}")


class MSLIFTCNSNN(nn.Module):
    """Multi-scale LIF SNN with a causal window-level TCN over spike representations."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        branch_dim: int = 32,
        beta: float = 0.9,
        theta_init: float = 1.0,
        dropout: float = 0.2,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.2,
        tcn_layers: int = 2,
        encoder_mode: str = "multi",
        single_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)
        self.encoder = build_window_encoder(
            input_channels,
            self.hidden_dim,
            int(branch_dim),
            dropout,
            encoder_mode=encoder_mode,
            single_kernel_size=single_kernel_size,
        )
        self.theta_raw = nn.Parameter(torch.full((self.hidden_dim,), inverse_softplus(theta_init)))
        self.tcn_layers = int(tcn_layers)
        self.window_tcn = (
            nn.Identity()
            if self.tcn_layers == 0
            else WindowTemporalTCN(
                self.hidden_dim,
                kernel_size=tcn_kernel_size,
                dropout=tcn_dropout,
                tcn_layers=self.tcn_layers,
            )
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,K,T,C], got {tuple(x.shape)}")
        bsz, context_len, window_size, channels = x.shape
        flat = x.reshape(bsz * context_len, window_size, channels).permute(0, 2, 1)
        currents = self.encoder(flat).permute(0, 2, 1)
        theta = F.softplus(self.theta_raw).reshape(1, -1) + 1e-3
        spikes = self._run_lif(currents, theta)
        spike_repr = spikes.mean(dim=1).reshape(bsz, context_len, self.hidden_dim)
        context_repr = self.window_tcn(spike_repr)
        logits = self.classifier(context_repr)
        return {
            "logits": logits,
            "spike_rate": spikes.mean(),
            "spike_repr": spike_repr,
            "window_context": context_repr,
            "features": context_repr,
        }

    def _run_lif(self, currents: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        membrane = currents.new_zeros((currents.shape[0], currents.shape[-1]))
        spikes = []
        for time_idx in range(currents.shape[1]):
            membrane = self.beta * membrane + currents[:, time_idx]
            spike = surrogate_spike(membrane - theta)
            membrane = membrane - spike * theta
            spikes.append(spike)
        return torch.stack(spikes, dim=1)


class MSLIFTCNGateSNN(nn.Module):
    """MS-LIF-TCN with a residual gate between current-window and TCN context features."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        branch_dim: int = 32,
        beta: float = 0.9,
        theta_init: float = 1.0,
        dropout: float = 0.2,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.2,
        tcn_layers: int = 2,
        gate_hidden_dim: int = 64,
        gate_dropout: float = 0.1,
        gate_mode: str = "scalar",
        encoder_mode: str = "multi",
        single_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)
        self.gate_mode = str(gate_mode).lower().replace("-", "_")
        if self.gate_mode not in {"scalar", "channel"}:
            raise ValueError("gate_mode must be 'scalar' or 'channel'")
        self.encoder = build_window_encoder(
            input_channels,
            self.hidden_dim,
            int(branch_dim),
            dropout,
            encoder_mode=encoder_mode,
            single_kernel_size=single_kernel_size,
        )
        self.theta_raw = nn.Parameter(torch.full((self.hidden_dim,), inverse_softplus(theta_init)))
        self.tcn_layers = int(tcn_layers)
        self.window_tcn = (
            nn.Identity()
            if self.tcn_layers == 0
            else WindowTemporalTCN(
                self.hidden_dim,
                kernel_size=tcn_kernel_size,
                dropout=tcn_dropout,
                tcn_layers=self.tcn_layers,
            )
        )
        gate_out_dim = 1 if self.gate_mode == "scalar" else self.hidden_dim
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, int(gate_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(gate_dropout)),
            nn.Linear(int(gate_hidden_dim), gate_out_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,K,T,C], got {tuple(x.shape)}")
        bsz, context_len, window_size, channels = x.shape
        flat = x.reshape(bsz * context_len, window_size, channels).permute(0, 2, 1)
        currents = self.encoder(flat).permute(0, 2, 1)
        theta = F.softplus(self.theta_raw).reshape(1, -1) + 1e-3
        spikes = self._run_lif(currents, theta)
        spike_repr = spikes.mean(dim=1).reshape(bsz, context_len, self.hidden_dim)
        context_repr = self.window_tcn(spike_repr)
        gate_input = torch.cat([spike_repr, context_repr, context_repr - spike_repr], dim=-1)
        gate = self.gate(gate_input)
        fused_repr = gate * context_repr + (1.0 - gate) * spike_repr
        logits = self.classifier(fused_repr)
        return {
            "logits": logits,
            "spike_rate": spikes.mean(),
            "spike_repr": spike_repr,
            "window_context": context_repr,
            "gate": gate,
            "gate_mean": gate.mean(),
            "gate_std": gate.std(unbiased=False),
            "features": fused_repr,
        }

    def _run_lif(self, currents: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        membrane = currents.new_zeros((currents.shape[0], currents.shape[-1]))
        spikes = []
        for time_idx in range(currents.shape[1]):
            membrane = self.beta * membrane + currents[:, time_idx]
            spike = surrogate_spike(membrane - theta)
            membrane = membrane - spike * theta
            spikes.append(spike)
        return torch.stack(spikes, dim=1)


class MSLIFTCNAttnSNN(nn.Module):
    """MS-LIF-TCN with lightweight causal attention over context windows."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        branch_dim: int = 32,
        beta: float = 0.9,
        theta_init: float = 1.0,
        dropout: float = 0.2,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.2,
        tcn_layers: int = 2,
        attention_hidden_dim: int = 64,
        encoder_mode: str = "multi",
        single_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)
        self.encoder = build_window_encoder(
            input_channels,
            self.hidden_dim,
            int(branch_dim),
            dropout,
            encoder_mode=encoder_mode,
            single_kernel_size=single_kernel_size,
        )
        self.theta_raw = nn.Parameter(torch.full((self.hidden_dim,), inverse_softplus(theta_init)))
        self.window_tcn = WindowTemporalTCN(
            self.hidden_dim,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
            tcn_layers=tcn_layers,
        )
        self.attention = WindowAttentionGate(self.hidden_dim, attention_hidden_dim, dropout)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,K,T,C], got {tuple(x.shape)}")
        bsz, context_len, window_size, channels = x.shape
        flat = x.reshape(bsz * context_len, window_size, channels).permute(0, 2, 1)
        currents = self.encoder(flat).permute(0, 2, 1)
        theta = F.softplus(self.theta_raw).reshape(1, -1) + 1e-3
        spikes = self._run_lif(currents, theta)
        spike_repr = spikes.mean(dim=1).reshape(bsz, context_len, self.hidden_dim)
        context_repr = self.window_tcn(spike_repr)
        attended_context, attention_matrix = self.attention(context_repr)
        logits = self.classifier(attended_context)
        return {
            "logits": logits,
            "spike_rate": spikes.mean(),
            "spike_repr": spike_repr,
            "window_context": context_repr,
            "attended_context": attended_context,
            "attention_weights": attention_matrix[:, -1, :],
            "attention_matrix": attention_matrix,
            "features": attended_context,
        }

    def _run_lif(self, currents: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        membrane = currents.new_zeros((currents.shape[0], currents.shape[-1]))
        spikes = []
        for time_idx in range(currents.shape[1]):
            membrane = self.beta * membrane + currents[:, time_idx]
            spike = surrogate_spike(membrane - theta)
            membrane = membrane - spike * theta
            spikes.append(spike)
        return torch.stack(spikes, dim=1)


class MSANNTCN(nn.Module):
    """Non-spiking multi-scale encoder with the same causal window TCN head.

    This is a structural control for MS-LIF-TCN: it keeps the multi-scale
    window encoder and causal window-level TCN, but replaces LIF spike
    dynamics with continuous average-pooled features.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        branch_dim: int = 32,
        dropout: float = 0.2,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.2,
        tcn_layers: int = 2,
        encoder_mode: str = "multi",
        single_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.encoder = build_window_encoder(
            input_channels,
            self.hidden_dim,
            int(branch_dim),
            dropout,
            encoder_mode=encoder_mode,
            single_kernel_size=single_kernel_size,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.window_tcn = WindowTemporalTCN(
            self.hidden_dim,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
            tcn_layers=tcn_layers,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,K,T,C], got {tuple(x.shape)}")
        bsz, context_len, window_size, channels = x.shape
        flat = x.reshape(bsz * context_len, window_size, channels).permute(0, 2, 1)
        encoded = self.encoder(flat)
        window_repr = self.pool(encoded).squeeze(-1).reshape(bsz, context_len, self.hidden_dim)
        context_repr = self.window_tcn(window_repr)
        logits = self.classifier(context_repr)
        return {
            "logits": logits,
            "window_repr": window_repr,
            "window_context": context_repr,
            "features": context_repr,
        }


class MSCMGTCNLIFSNN(nn.Module):
    """MS-CMG-LIF with an additional causal window-level TCN classifier head."""

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
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.2,
        tcn_layers: int = 2,
        encoder_mode: str = "multi",
        single_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if num_groups < 1 or num_groups > hidden_dim:
            raise ValueError("num_groups must be in [1, hidden_dim]")
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.theta_scale = float(theta_scale)
        self.num_groups = int(num_groups)
        self.threshold_modulation = bool(threshold_modulation)
        self.context_memory = bool(context_memory)
        self.encoder = build_window_encoder(
            input_channels,
            self.hidden_dim,
            int(branch_dim),
            dropout,
            encoder_mode=encoder_mode,
            single_kernel_size=single_kernel_size,
        )
        self.memory_to_gate = nn.Linear(self.hidden_dim, self.num_groups)
        self.theta_raw = nn.Parameter(torch.full((self.hidden_dim,), inverse_softplus(theta_init)))
        self.window_tcn = WindowTemporalTCN(
            self.hidden_dim,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
            tcn_layers=tcn_layers,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.hidden_dim, num_classes))

        nn.init.zeros_(self.memory_to_gate.weight)
        nn.init.zeros_(self.memory_to_gate.bias)
        group_ids = torch.div(torch.arange(self.hidden_dim) * self.num_groups, self.hidden_dim, rounding_mode="floor").long()
        self.register_buffer("group_ids", group_ids, persistent=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,K,T,C], got {tuple(x.shape)}")
        bsz, context_len, window_size, channels = x.shape
        flat = x.reshape(bsz * context_len, window_size, channels).permute(0, 2, 1)
        encoded = self.encoder(flat).permute(0, 2, 1).reshape(bsz, context_len, window_size, self.hidden_dim)
        memory = x.new_zeros((bsz, self.hidden_dim))
        spike_reprs = []
        spike_rates = []
        context_states = []
        for window_idx in range(context_len):
            theta = self._threshold_from_memory(memory)
            spikes = self._run_lif(encoded[:, window_idx], theta)
            spike_repr = spikes.mean(dim=1)
            spike_reprs.append(spike_repr)
            spike_rates.append(spikes.mean())
            if self.context_memory:
                memory = self.alpha * memory + (1.0 - self.alpha) * spike_repr
            else:
                memory = torch.zeros_like(memory)
            context_states.append(memory)
        spike_repr_seq = torch.stack(spike_reprs, dim=1)
        context_repr = self.window_tcn(spike_repr_seq)
        logits = self.classifier(context_repr)
        return {
            "logits": logits,
            "spike_rate": torch.stack(spike_rates).mean(),
            "spike_repr": spike_repr_seq,
            "context_states": torch.stack(context_states, dim=1),
            "window_context": context_repr,
            "features": context_repr,
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
