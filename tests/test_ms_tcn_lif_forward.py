from __future__ import annotations

import torch

from src.models import (
    MSANNTCN,
    MSCMGTCNLIFSNN,
    MSLIFTCNAttnSNN,
    MSLIFTCNGateSNN,
    MSLIFTCNSNN,
    SingleScaleTemporalEncoder,
    WindowTemporalTCN,
)


def test_ms_lif_tcn_forward_shape() -> None:
    model = MSLIFTCNSNN(input_channels=9, num_classes=6, hidden_dim=32, branch_dim=8)
    x = torch.randn(2, 4, 128, 9)
    outputs = model(x)
    assert outputs["logits"].shape == (2, 4, 6)
    assert outputs["spike_rate"].ndim == 0
    assert outputs["window_context"].shape == (2, 4, 32)


def test_ms_lif_tcn_gate_forward_shape() -> None:
    model = MSLIFTCNGateSNN(input_channels=9, num_classes=6, hidden_dim=32, branch_dim=8)
    x = torch.randn(2, 4, 128, 9)
    outputs = model(x)
    assert outputs["logits"].shape == (2, 4, 6)
    assert outputs["spike_rate"].ndim == 0
    assert outputs["spike_repr"].shape == (2, 4, 32)
    assert outputs["window_context"].shape == (2, 4, 32)
    assert outputs["features"].shape == (2, 4, 32)
    assert outputs["gate"].shape == (2, 4, 1)


def test_ms_lif_tcn_gate_outputs_gate_stats() -> None:
    model = MSLIFTCNGateSNN(input_channels=9, num_classes=6, hidden_dim=16, branch_dim=8)
    x = torch.randn(2, 3, 64, 9)
    outputs = model(x)
    assert outputs["gate_mean"].ndim == 0
    assert outputs["gate_std"].ndim == 0
    assert 0.0 <= float(outputs["gate_mean"]) <= 1.0
    assert float(outputs["gate_std"]) >= 0.0


def test_gate_model_can_fallback_to_last_window_shape() -> None:
    model = MSLIFTCNGateSNN(input_channels=9, num_classes=6, hidden_dim=16, branch_dim=8, tcn_layers=0)
    x = torch.randn(2, 5, 64, 9)
    outputs = model(x)
    assert outputs["logits"][:, -1, :].shape == (2, 6)
    assert outputs["features"][:, -1, :].shape == (2, 16)


def test_ms_lif_tcn_gate_channel_forward_shape() -> None:
    model = MSLIFTCNGateSNN(
        input_channels=9,
        num_classes=6,
        hidden_dim=16,
        branch_dim=8,
        gate_mode="channel",
    )
    x = torch.randn(2, 4, 64, 9)
    outputs = model(x)
    assert outputs["logits"].shape == (2, 4, 6)
    assert outputs["gate"].shape == (2, 4, 16)


def test_ms_lif_tcn_tcn0_forward_shape_and_no_window_context() -> None:
    torch.manual_seed(0)
    model = MSLIFTCNSNN(input_channels=9, num_classes=6, hidden_dim=16, branch_dim=8, tcn_layers=0)
    model.eval()
    x = torch.randn(2, 4, 128, 9)
    with torch.no_grad():
        outputs = model(x)
        changed_history = x.clone()
        changed_history[:, 0, :, :] += 100.0
        changed_outputs = model(changed_history)
    assert outputs["logits"].shape == (2, 4, 6)
    assert outputs["window_context"].shape == (2, 4, 16)
    assert torch.allclose(outputs["spike_repr"], outputs["window_context"])
    assert torch.allclose(outputs["logits"][:, -1, :], changed_outputs["logits"][:, -1, :], atol=1e-5)


def test_ms_cmg_tcn_forward_shape() -> None:
    model = MSCMGTCNLIFSNN(input_channels=9, num_classes=6, hidden_dim=32, branch_dim=8, num_groups=4)
    x = torch.randn(2, 4, 128, 9)
    outputs = model(x)
    assert outputs["logits"].shape == (2, 4, 6)
    assert outputs["spike_rate"].ndim == 0
    assert outputs["context_states"].shape == (2, 4, 32)


def test_ms_lif_tcn_attn_forward_shape_and_weights() -> None:
    model = MSLIFTCNAttnSNN(input_channels=9, num_classes=6, hidden_dim=32, branch_dim=8, attention_hidden_dim=16)
    x = torch.randn(2, 4, 128, 9)
    outputs = model(x)
    assert outputs["logits"].shape == (2, 4, 6)
    assert outputs["spike_rate"].ndim == 0
    assert outputs["attention_weights"].shape == (2, 4)
    assert outputs["features"].shape == (2, 4, 32)
    assert torch.allclose(outputs["attention_weights"].sum(dim=1), torch.ones(2), atol=1e-5)


def test_ms_ann_tcn_forward_shape_without_spike_rate() -> None:
    model = MSANNTCN(input_channels=9, num_classes=6, hidden_dim=32, branch_dim=8)
    x = torch.randn(2, 4, 128, 9)
    outputs = model(x)
    assert outputs["logits"].shape == (2, 4, 6)
    assert outputs["window_context"].shape == (2, 4, 32)
    assert "spike_rate" not in outputs


def test_window_tcn_layer_count() -> None:
    one_layer = WindowTemporalTCN(hidden_dim=8, kernel_size=3, dropout=0.0, tcn_layers=1)
    two_layers = WindowTemporalTCN(hidden_dim=8, kernel_size=3, dropout=0.0, tcn_layers=2)
    assert len(one_layer.blocks) == 1
    assert len(two_layers.blocks) == 2


def test_single_scale_encoder_preserves_time_length() -> None:
    encoder = SingleScaleTemporalEncoder(input_channels=9, hidden_dim=16, branch_dim=8, kernel_size=5, dropout=0.0)
    x = torch.randn(2, 9, 128)
    y = encoder(x)
    assert y.shape == (2, 16, 128)


def test_ms_lif_tcn_single_scale_forward_shape() -> None:
    model = MSLIFTCNSNN(
        input_channels=9,
        num_classes=6,
        hidden_dim=16,
        branch_dim=8,
        encoder_mode="single",
        single_kernel_size=5,
    )
    x = torch.randn(2, 4, 128, 9)
    outputs = model(x)
    assert outputs["logits"].shape == (2, 4, 6)
    assert outputs["spike_rate"].ndim == 0


def test_window_tcn_is_causal_for_earlier_outputs() -> None:
    torch.manual_seed(0)
    tcn = WindowTemporalTCN(hidden_dim=8, kernel_size=3, dropout=0.0)
    tcn.eval()
    x = torch.randn(2, 6, 8)
    y1 = tcn(x)
    changed_future = x.clone()
    changed_future[:, 5, :] += 100.0
    y2 = tcn(changed_future)
    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-5)
    assert not torch.allclose(y1[:, 5, :], y2[:, 5, :])
