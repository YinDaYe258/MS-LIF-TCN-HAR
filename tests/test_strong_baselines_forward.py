from __future__ import annotations

import torch

from src.training.utils import build_model


def test_ms_lif_snn_forward_shape() -> None:
    x = torch.randn(2, 4, 128, 9)
    model = build_model(
        "ms_lif_snn",
        input_channels=9,
        num_classes=6,
        model_cfg={"hidden_dim": 16, "branch_dim": 8, "beta": 0.9, "dropout": 0.0},
    )
    outputs = model(x)

    assert outputs["logits"].shape == (2, 4, 6)
    assert outputs["spike_rate"].ndim == 0


def test_ms_cnn1d_forward_shape() -> None:
    x = torch.randn(2, 4, 128, 9)
    model = build_model(
        "ms_cnn1d",
        input_channels=9,
        num_classes=6,
        model_cfg={"hidden_dim": 16, "branch_dim": 8, "dropout": 0.0},
    )
    outputs = model(x)

    assert outputs["logits"].shape == (2, 4, 6)


def test_window_gru_forward_shape() -> None:
    x = torch.randn(2, 4, 128, 9)
    model = build_model(
        "window_gru",
        input_channels=9,
        num_classes=6,
        model_cfg={"hidden_dim": 16, "dropout": 0.0},
    )
    outputs = model(x)

    assert outputs["logits"].shape == (2, 4, 6)
