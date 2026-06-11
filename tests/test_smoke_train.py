from __future__ import annotations

import torch

from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.trainer import Trainer
from src.training.utils import build_model, set_seed


def test_one_smoke_training_epoch(tmp_path) -> None:
    config = {
        "seed": 11,
        "device": "cpu",
        "dataset": {
            "root": "missing/path",
            "context_len": 2,
            "val_subject_fraction": 0.25,
            "smoke_max_train_sequences": 8,
            "smoke_max_val_sequences": 4,
            "smoke_max_test_sequences": 4,
        },
        "model": {"hidden_dim": 16, "beta": 0.9, "alpha": 0.8, "dropout": 0.0},
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "patience": 2,
            "spike_reg_lambda": 0.0001,
            "num_workers": 0,
        },
        "results": {"dir": str(tmp_path)},
    }
    set_seed(config["seed"])
    loaders, meta = create_ucihar_dataloaders(config, smoke_test=True)
    model = build_model("cmg_lif", meta.num_channels, meta.num_classes, config["model"])
    trainer = Trainer(
        model,
        loaders,
        config,
        torch.device("cpu"),
        "test_smoke",
        results_dir=tmp_path,
        num_classes=meta.num_classes,
    )
    metrics = trainer.fit()
    assert "macro_f1" in metrics
    assert (tmp_path / "test_smoke_best.pt").exists()
    assert (tmp_path / "test_smoke_epoch_log.csv").exists()
