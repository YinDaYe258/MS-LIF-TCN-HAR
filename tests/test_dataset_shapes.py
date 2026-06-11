from __future__ import annotations

import numpy as np

from src.datasets.hapt import create_hapt_dataloaders, load_hapt_arrays
from src.datasets.mhealth import create_mhealth_dataloaders, load_mhealth_windows
from src.datasets.pamap2 import create_pamap2_dataloaders, load_pamap2_windows
from src.datasets.preprocessing import apply_train_preprocessor, fit_train_preprocessor
from src.datasets.sequence_dataset import SequenceWindowDataset
from src.datasets.ucihar import create_ucihar_dataloaders


def test_sequence_dataset_shape() -> None:
    x = np.random.randn(10, 128, 9).astype(np.float32)
    y = np.arange(10, dtype=np.int64) % 6
    subjects = np.array([1] * 5 + [2] * 5, dtype=np.int64)
    dataset = SequenceWindowDataset(x, y, subjects, context_len=3)

    sample = dataset[0]
    assert sample["x"].shape == (3, 128, 9)
    assert sample["y"].shape == (3,)
    assert int(sample["subject"]) == 1
    assert len(dataset) == 6


def test_sequence_dataset_respects_group_ids() -> None:
    x = np.random.randn(8, 128, 6).astype(np.float32)
    y = np.arange(8, dtype=np.int64) % 2
    subjects = np.array([1] * 8, dtype=np.int64)
    groups = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    dataset = SequenceWindowDataset(x, y, subjects, context_len=3, group_ids=groups)

    assert len(dataset) == 4
    for sample in dataset:
        assert len(set(sample["group"].numpy().tolist())) == 1


def test_synthetic_ucihar_loader_shape() -> None:
    config = {
        "seed": 7,
        "dataset": {
            "root": "missing/path",
            "context_len": 2,
            "val_subject_fraction": 0.25,
            "smoke_max_train_sequences": 4,
            "smoke_max_val_sequences": 4,
            "smoke_max_test_sequences": 4,
        },
        "training": {"batch_size": 2, "num_workers": 0},
    }
    loaders, meta = create_ucihar_dataloaders(config, smoke_test=True)
    batch = next(iter(loaders["train"]))

    assert meta.synthetic is True
    assert batch["x"].shape == (2, 2, 128, 9)
    assert batch["y"].shape == (2, 2)


def test_synthetic_hapt_loader_shape() -> None:
    config = {
        "seed": 7,
        "dataset": {
            "root": "missing/path",
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

    assert meta.synthetic is True
    assert meta.num_classes == 12
    assert batch["x"].shape == (2, 2, 128, 6)
    assert batch["y"].shape == (2, 2)


def test_synthetic_pamap2_loader_shape() -> None:
    config = {
        "seed": 7,
        "dataset": {
            "root": "missing/path",
            "context_len": 8,
            "window_size": 256,
            "stride": 128,
            "val_subject_fraction": 0.25,
            "smoke_max_train_sequences": 4,
            "smoke_max_val_sequences": 4,
            "smoke_max_test_sequences": 4,
        },
        "training": {"batch_size": 2, "num_workers": 0},
    }
    loaders, meta = create_pamap2_dataloaders(config, smoke_test=True)
    batch = next(iter(loaders["train"]))

    assert meta.synthetic is True
    assert meta.num_classes == 12
    assert batch["x"].shape == (2, 8, 256, 18)
    assert batch["y"].shape == (2, 8)


def test_synthetic_mhealth_loader_shape() -> None:
    config = {
        "seed": 7,
        "dataset": {
            "root": "missing/path",
            "context_len": 8,
            "window_size": 128,
            "stride": 64,
            "val_subject_fraction": 0.25,
            "smoke_max_train_sequences": 4,
            "smoke_max_val_sequences": 4,
            "smoke_max_test_sequences": 4,
        },
        "training": {"batch_size": 2, "num_workers": 0},
    }
    loaders, meta = create_mhealth_dataloaders(config, smoke_test=True)
    batch = next(iter(loaders["train"]))

    assert meta.synthetic is True
    assert meta.num_classes == 12
    assert batch["x"].shape == (2, 8, 128, 15)
    assert batch["y"].shape == (2, 8)


def test_train_only_preprocessor_uses_train_statistics() -> None:
    train_x = np.asarray([[[1.0], [3.0]], [[5.0], [7.0]]], dtype=np.float32)
    test_x = np.asarray([[[100.0], [np.nan]]], dtype=np.float32)
    stats = fit_train_preprocessor(train_x, normalize="train_zscore", impute_missing="train_channel_mean")
    transformed_train = apply_train_preprocessor(train_x, stats)
    transformed_test = apply_train_preprocessor(test_x, stats)

    assert abs(float(transformed_train.mean())) < 1e-6
    assert abs(float(transformed_train.std()) - 1.0) < 1e-6
    assert abs(float(stats["channel_mean"][0]) - 4.0) < 1e-6
    assert np.isfinite(transformed_test).all()


def test_hapt_raw_loader_tiny_dataset(tmp_path) -> None:
    root = tmp_path / "HAPT Dataset"
    raw_dir = root / "RawData"
    train_dir = root / "Train"
    test_dir = root / "Test"
    raw_dir.mkdir(parents=True)
    train_dir.mkdir()
    test_dir.mkdir()
    (train_dir / "subject_id_train.txt").write_text("1\n", encoding="utf-8")
    (test_dir / "subject_id_test.txt").write_text("2\n", encoding="utf-8")
    (raw_dir / "labels.txt").write_text("1 1 5 1 256\n2 2 1 1 256\n", encoding="utf-8")

    signal = "\n".join("0.1 0.2 0.3" for _ in range(256))
    for prefix, exp, user in [("acc", 1, 1), ("gyro", 1, 1), ("acc", 2, 2), ("gyro", 2, 2)]:
        (raw_dir / f"{prefix}_exp{exp:02d}_user{user:02d}.txt").write_text(signal, encoding="utf-8")

    x_train, y_train, subjects_train = load_hapt_arrays(root, "train", window_size=128, stride=64)
    x_test, y_test, subjects_test = load_hapt_arrays(root, "test", window_size=128, stride=64)
    _, y_train_hapt6, _ = load_hapt_arrays(root, "train", window_size=128, stride=64, task="hapt6")

    assert x_train.shape == (3, 128, 6)
    assert x_test.shape == (3, 128, 6)
    assert set(y_train.tolist()) == {4}
    assert set(y_train_hapt6.tolist()) == {4}
    assert set(y_test.tolist()) == {0}
    assert set(subjects_train.tolist()) == {1}
    assert set(subjects_test.tolist()) == {2}


def test_pamap2_raw_loader_tiny_dataset(tmp_path) -> None:
    root = tmp_path / "PAMAP2_Dataset"
    protocol = root / "Protocol"
    protocol.mkdir(parents=True)

    def write_subject(subject_id: int, activity_id: int) -> None:
        rows = []
        for idx in range(384):
            values = np.zeros(54, dtype=np.float32)
            values[0] = idx * 0.01
            values[1] = activity_id
            values[3:] = 0.1 + subject_id * 0.001
            rows.append(" ".join(f"{value:.4f}" for value in values))
        (protocol / f"subject{subject_id}.dat").write_text("\n".join(rows), encoding="utf-8")

    write_subject(101, 1)
    write_subject(105, 2)

    x_train, y_train, subjects_train, meta_train = load_pamap2_windows(
        root,
        "train",
        window_size=256,
        stride=128,
        test_subjects=(105,),
    )
    x_test, y_test, subjects_test, meta_test = load_pamap2_windows(
        root,
        "test",
        window_size=256,
        stride=128,
        test_subjects=(105,),
    )

    assert x_train.shape == (2, 256, 18)
    assert x_test.shape == (2, 256, 18)
    assert set(y_train.tolist()) == {0}
    assert set(y_test.tolist()) == {1}
    assert set(subjects_train.tolist()) == {101}
    assert set(subjects_test.tolist()) == {105}
    assert len(set(meta_train["segment_ids"].tolist())) == 1
    assert len(set(meta_test["segment_ids"].tolist())) == 1


def test_mhealth_raw_loader_tiny_dataset(tmp_path) -> None:
    root = tmp_path / "MHEALTHDATASET"
    root.mkdir()

    def write_subject(subject_id: int, label_id: int) -> None:
        rows = []
        for idx in range(256):
            values = np.zeros(24, dtype=np.float32)
            values[:23] = 0.2 + subject_id * 0.001
            values[-1] = label_id
            rows.append(" ".join(f"{value:.4f}" for value in values))
        (root / f"mHealth_subject{subject_id}.log").write_text("\n".join(rows), encoding="utf-8")

    write_subject(1, 3)
    write_subject(9, 4)

    x_train, y_train, subjects_train, meta_train = load_mhealth_windows(
        root,
        "train",
        window_size=128,
        stride=64,
        test_subjects=(9,),
    )
    x_test, y_test, subjects_test, meta_test = load_mhealth_windows(
        root,
        "test",
        window_size=128,
        stride=64,
        test_subjects=(9,),
    )

    assert x_train.shape == (3, 128, 15)
    assert x_test.shape == (3, 128, 15)
    assert set(y_train.tolist()) == {2}
    assert set(y_test.tolist()) == {3}
    assert set(subjects_train.tolist()) == {1}
    assert set(subjects_test.tolist()) == {9}
    assert len(set(meta_train["segment_ids"].tolist())) == 1
    assert len(set(meta_test["segment_ids"].tolist())) == 1
