from __future__ import annotations

import torch

from src.training.utils import build_model


def test_models_forward_shapes() -> None:
    x = torch.randn(2, 3, 128, 9)
    for model_name in ["cnn1d", "gru", "lif_snn", "cmg_lif", "cmg_lif_lite", "ms_lif_tcn_gate"]:
        model = build_model(
            model_name,
            input_channels=9,
            num_classes=6,
            model_cfg={"hidden_dim": 16, "beta": 0.9, "alpha": 0.8, "num_groups": 4, "dropout": 0.0},
        )
        outputs = model(x)
        assert outputs["logits"].shape == (2, 3, 6)
        if model_name in {"lif_snn", "cmg_lif", "cmg_lif_lite", "ms_lif_tcn_gate"}:
            assert "spike_rate" in outputs
            assert outputs["spike_rate"].ndim == 0
        if model_name == "ms_lif_tcn_gate":
            assert "gate_mean" in outputs
            assert outputs["features"].shape == (2, 3, 16)
