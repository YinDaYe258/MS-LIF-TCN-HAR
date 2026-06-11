from __future__ import annotations

import torch

from src.training.utils import build_model


def test_ms_cmg_lif_forward_shapes() -> None:
    x = torch.randn(2, 4, 128, 9)
    model = build_model(
        "ms_cmg_lif",
        input_channels=9,
        num_classes=6,
        model_cfg={
            "hidden_dim": 16,
            "branch_dim": 8,
            "beta": 0.9,
            "alpha": 0.8,
            "num_groups": 4,
            "dropout": 0.0,
        },
    )
    outputs = model(x)

    assert outputs["logits"].shape == (2, 4, 6)
    assert outputs["spike_rate"].ndim == 0
    assert outputs["context_states"].shape == (2, 4, 16)
