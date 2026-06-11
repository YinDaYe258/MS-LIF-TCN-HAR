from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.analysis.summarize_robustness import summarize_robustness_table
from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.checkpoints import find_checkpoint


def test_checkpoint_discovery_returns_matching_path(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    checkpoint = results_dir / "model.pt"
    torch.save({"model_state_dict": {}, "config": {}}, checkpoint)
    pd.DataFrame(
        [
            {
                "model": "ms_lif_snn",
                "seed": 42,
                "context_len": 8,
                "target_mode": "last",
                "synthetic_data": False,
                "smoke_test": False,
                "checkpoint": str(checkpoint),
            }
        ]
    ).to_csv(results_dir / "ucihar_strong_baseline_results.csv", index=False)

    found = find_checkpoint(results_dir, "ms_lif_snn", context_len=8, target_mode="last", seed=42)

    assert found == checkpoint


def test_perturbation_only_changes_test_dataset(tmp_path: Path) -> None:
    config = {
        "seed": 11,
        "dataset": {
            "root": str(tmp_path / "missing"),
            "context_len": 2,
            "val_subject_fraction": 0.25,
            "smoke_max_train_sequences": 4,
            "smoke_max_val_sequences": 4,
            "smoke_max_test_sequences": 4,
        },
        "training": {"batch_size": 2, "num_workers": 0},
    }
    clean_loaders, _ = create_ucihar_dataloaders(config, smoke_test=True)
    noisy_loaders, _ = create_ucihar_dataloaders(config, smoke_test=True, perturbation={"noise_std": 0.2})

    clean_train = clean_loaders["train"].dataset[0]["x"]
    noisy_train = noisy_loaders["train"].dataset[0]["x"]
    clean_test = clean_loaders["test"].dataset[0]["x"]
    noisy_test = noisy_loaders["test"].dataset[0]["x"]

    assert torch.equal(clean_train, noisy_train)
    assert not torch.equal(clean_test, noisy_test)


def test_robustness_summary_computes_drops() -> None:
    rows = pd.DataFrame(
        [
            {"model": "a", "perturbation_type": "clean", "perturbation_level": "0", "macro_f1": 0.9},
            {"model": "a", "perturbation_type": "gaussian_noise", "perturbation_level": "0.20", "macro_f1": 0.7},
            {"model": "a", "perturbation_type": "channel_dropout", "perturbation_level": "0.30", "macro_f1": 0.6},
            {"model": "a", "perturbation_type": "modality_dropout", "perturbation_level": "acc", "macro_f1": 0.5},
            {"model": "a", "perturbation_type": "modality_dropout", "perturbation_level": "gyro", "macro_f1": 0.8},
        ]
    )

    summary = summarize_robustness_table(rows)
    row = summary.iloc[0]

    assert row["noise_0.20_drop"] == pytest.approx(0.2)
    assert row["dropout_0.30_drop"] == pytest.approx(0.3)
    assert row["acc_dropout_drop"] == pytest.approx(0.4)
    assert row["gyro_dropout_drop"] == pytest.approx(0.1)
