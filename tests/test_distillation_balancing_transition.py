from __future__ import annotations

import torch

from src.datasets.hapt import create_hapt_dataloaders, load_hapt_arrays
from src.training.distillation import distillation_kl_loss
from src.training.losses import compute_class_weights_from_loader, sequence_classification_loss


def test_distillation_kl_loss_has_student_gradient() -> None:
    student_logits = torch.randn(5, 4, requires_grad=True)
    teacher_logits = torch.randn(5, 4)

    loss = distillation_kl_loss(student_logits, teacher_logits, temperature=4.0)
    loss.backward()

    assert loss.ndim == 0
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_class_weights_use_loader_labels_and_ignore_missing_class() -> None:
    batches = [
        {"y": torch.tensor([[0, 0], [0, 1], [0, 0]], dtype=torch.long)},
        {"y": torch.tensor([[1, 0]], dtype=torch.long)},
    ]

    weights = compute_class_weights_from_loader(batches, num_classes=3, target_mode="last")

    assert weights.shape == (3,)
    assert weights[1] > weights[0]
    assert weights[2] == 0


def test_weighted_focal_sequence_loss_runs() -> None:
    outputs = {
        "logits": torch.randn(2, 3, 4, requires_grad=True),
        "spike_rate": torch.tensor(0.25),
    }
    targets = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    weights = torch.ones(4)

    loss, details = sequence_classification_loss(
        outputs,
        targets,
        spike_reg_lambda=0.001,
        target_mode="last",
        loss_type="weighted_focal",
        class_weights=weights,
        focal_gamma=2.0,
    )

    assert loss.ndim == 0
    assert details["ce_loss"] > 0


def test_hapt_transition_binary_raw_mapping(tmp_path) -> None:
    root = tmp_path / "HAPT Dataset"
    raw_dir = root / "RawData"
    train_dir = root / "Train"
    test_dir = root / "Test"
    raw_dir.mkdir(parents=True)
    train_dir.mkdir()
    test_dir.mkdir()
    (train_dir / "subject_id_train.txt").write_text("1\n", encoding="utf-8")
    (test_dir / "subject_id_test.txt").write_text("2\n", encoding="utf-8")
    (raw_dir / "labels.txt").write_text(
        "1 1 1 1 128\n"
        "1 1 7 129 256\n"
        "2 2 1 1 128\n"
        "2 2 7 129 256\n",
        encoding="utf-8",
    )
    signal = "\n".join("0.1 0.2 0.3" for _ in range(256))
    for prefix, exp, user in [("acc", 1, 1), ("gyro", 1, 1), ("acc", 2, 2), ("gyro", 2, 2)]:
        (raw_dir / f"{prefix}_exp{exp:02d}_user{user:02d}.txt").write_text(signal, encoding="utf-8")

    _, y_train, _ = load_hapt_arrays(root, "train", window_size=128, stride=128, task="transitionbinary")
    _, y_test, _ = load_hapt_arrays(root, "test", window_size=128, stride=128, task="transitionbinary")

    assert set(y_train.tolist()) == {0, 1}
    assert set(y_test.tolist()) == {0, 1}


def test_synthetic_transition_binary_loader_uses_two_classes() -> None:
    config = {
        "seed": 7,
        "dataset": {
            "root": "missing/path",
            "task": "transitionbinary",
            "context_len": 2,
            "window_size": 128,
            "stride": 64,
            "val_subject_fraction": 0.25,
            "smoke_max_train_sequences": 4,
            "smoke_max_val_sequences": 4,
            "smoke_max_test_sequences": 4,
        },
        "training": {"batch_size": 2, "num_workers": 0},
    }
    loaders, meta = create_hapt_dataloaders(config, smoke_test=True)
    batch = next(iter(loaders["train"]))

    assert meta.num_classes == 2
    assert int(batch["y"].max()) < 2
