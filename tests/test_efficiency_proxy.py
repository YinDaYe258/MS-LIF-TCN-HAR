from __future__ import annotations

import pytest

from src.analysis.efficiency import (
    estimate_gate_macs,
    estimate_multiscale_encoder_macs,
    estimate_spike_count_per_sample,
    estimate_window_gru_ops,
    summarize_efficiency,
)


def test_spike_count_per_sample_formula() -> None:
    assert estimate_spike_count_per_sample(0.25, context_len=4, window_size=128, hidden_dim=64) == pytest.approx(
        8192.0
    )


def test_lif_gate_macs_is_zero() -> None:
    assert estimate_gate_macs("lif_snn", context_len=4, hidden_dim=128) == 0


def test_cmg_lif_lite_gate_macs() -> None:
    assert estimate_gate_macs("cmg_lif_lite", context_len=4, hidden_dim=128, num_groups=8) == 4096


def test_dense_cmg_lif_gate_macs() -> None:
    assert estimate_gate_macs("cmg_lif", context_len=4, hidden_dim=128) == 65536


def test_ms_cmg_lif_gate_macs_uses_group_gate() -> None:
    assert estimate_gate_macs("ms_cmg_lif", context_len=8, hidden_dim=128, num_groups=8) == 8192


def test_ms_lif_and_ms_cnn_have_no_gate_macs() -> None:
    assert estimate_gate_macs("ms_lif_snn", context_len=8, hidden_dim=128, num_groups=8) == 0
    assert estimate_gate_macs("ms_cnn1d", context_len=8, hidden_dim=128, num_groups=8) == 0


def test_multiscale_encoder_macs_formula() -> None:
    assert estimate_multiscale_encoder_macs(
        context_len=2,
        window_size=128,
        input_channels=9,
        hidden_dim=64,
        branch_dim=16,
    ) == 2 * 128 * ((9 * 16 * (3 + 5 + 9)) + (16 * 3 * 64))


def test_ms_cnn_summary_has_no_spike_count_without_spike_rate() -> None:
    summary = summarize_efficiency(
        {"dataset": "UCI-HAR", "model": "ms_cnn1d", "context_len": 8, "target_mode": "last", "params": 10},
        {
            "dataset": {"window_size": 128, "input_channels": 9, "num_classes": 6},
            "model": {"hidden_dim": 128, "branch_dim": 32, "num_groups": 8},
            "training": {"target_mode": "last"},
        },
        "ms_cnn1d",
    )
    assert summary["spike_count_per_sample"] == 0.0
    assert "non_spiking_model" in summary["note"]


def test_window_gru_ops_are_positive() -> None:
    ops = estimate_window_gru_ops(
        context_len=8,
        window_size=128,
        input_channels=9,
        hidden_dim=128,
        num_classes=6,
        target_mode="last",
    )
    assert ops["total_ops_proxy"] > 0
    assert ops["recurrent_ops"] > 0
