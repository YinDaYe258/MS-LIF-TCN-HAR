from __future__ import annotations

import pandas as pd
import torch

from scripts.run_ms_lif_tcn_plus_diagnostic import row_exists
from src.training.augmentation import apply_sensor_augmentation


def test_sensor_augmentation_changes_training_batch() -> None:
    torch.manual_seed(0)
    x = torch.ones(2, 3, 8, 4)
    augmented = apply_sensor_augmentation(
        x,
        jitter_std=0.05,
        scaling_std=0.10,
        channel_dropout_prob=0.25,
        temporal_shift_max=1,
    )
    assert augmented.shape == x.shape
    assert not torch.allclose(augmented, x)


def test_plus_diagnostic_row_exists(tmp_path) -> None:
    path = tmp_path / "rows.csv"
    pd.DataFrame(
        [
            {"dataset_key": "ucihar", "variant": "attn_ce", "seed": 42},
            {"dataset_key": "hapt6", "variant": "attn_ce", "seed": 42},
        ]
    ).to_csv(path, index=False)
    assert row_exists(path, "ucihar", "attn_ce", 42)
    assert not row_exists(path, "ucihar", "attn_ce", 43)
