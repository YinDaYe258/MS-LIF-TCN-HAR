from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset

from .sequence_dataset import SequenceDatasetMeta, SequenceWindowDataset

UCIHAR_CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)


class UCIHARWindowDataset(Dataset):
    """Official UCI-HAR inertial-signal windows."""

    def __init__(self, root: str | Path, split: str) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.root = Path(root)
        self.split = split
        self.x, self.y, self.subjects = load_ucihar_arrays(self.root, split)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, int]:
        return self.x[idx], int(self.y[idx]), int(self.subjects[idx])


def load_ucihar_arrays(root: str | Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(root)
    signal_dir = root / split / "Inertial Signals"
    if not signal_dir.exists():
        raise FileNotFoundError(
            f"UCI-HAR inertial signal directory not found: {signal_dir}. "
            "Run scripts/download_ucihar.py or place the official dataset under data/."
        )

    signals = []
    for channel in UCIHAR_CHANNELS:
        path = signal_dir / f"{channel}_{split}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing UCI-HAR signal file: {path}")
        signals.append(np.loadtxt(path, dtype=np.float32))
    x = np.stack(signals, axis=-1)

    y_path = root / split / f"y_{split}.txt"
    subject_path = root / split / f"subject_{split}.txt"
    y = np.loadtxt(y_path, dtype=np.int64) - 1
    subjects = np.loadtxt(subject_path, dtype=np.int64)
    return x, y, subjects


def split_train_val_by_subject(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    val_subject_fraction: float,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    unique_subjects = np.unique(subjects)
    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_subject_fraction)))
    val_subjects = set(shuffled[:val_count].tolist())
    val_mask = np.array([subject in val_subjects for subject in subjects], dtype=bool)
    train_mask = ~val_mask
    return (
        (x[train_mask], y[train_mask], subjects[train_mask]),
        (x[val_mask], y[val_mask], subjects[val_mask]),
    )


def create_synthetic_ucihar(
    n_train_subjects: int = 4,
    n_test_subjects: int = 2,
    windows_per_subject: int = 18,
    window_size: int = 128,
    num_channels: int = 9,
    num_classes: int = 6,
    seed: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)

    def make_split(subject_offset: int, n_subjects: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        subs: list[int] = []
        time = np.linspace(0.0, 1.0, window_size, dtype=np.float32)
        class_templates = np.stack(
            [np.sin(2.0 * np.pi * (label + 1) * time) for label in range(num_classes)]
        ).astype(np.float32)
        for local_subject in range(n_subjects):
            subject_id = subject_offset + local_subject + 1
            subject_bias = rng.normal(0.0, 0.1, size=(num_channels,)).astype(np.float32)
            for window_idx in range(windows_per_subject):
                label = (window_idx + local_subject) % num_classes
                base = class_templates[label][:, None]
                channel_scale = rng.uniform(0.5, 1.5, size=(num_channels,)).astype(np.float32)
                noise = rng.normal(0.0, 0.05, size=(window_size, num_channels)).astype(np.float32)
                xs.append(base * channel_scale + subject_bias + noise)
                ys.append(label)
                subs.append(subject_id)
        return np.stack(xs), np.asarray(ys, dtype=np.int64), np.asarray(subs, dtype=np.int64)

    return {
        "train": make_split(0, n_train_subjects),
        "test": make_split(n_train_subjects, n_test_subjects),
    }


def create_ucihar_dataloaders(
    config: dict[str, Any],
    model_name: str | None = None,
    smoke_test: bool = False,
    perturbation: dict[str, Any] | None = None,
) -> tuple[dict[str, DataLoader], SequenceDatasetMeta]:
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    root = Path(dataset_cfg.get("root", "data/UCI HAR Dataset"))
    context_len = int(dataset_cfg.get("context_len", 1))
    seed = int(config.get("seed", 0))
    perturbation = perturbation or {}
    synthetic = False

    if root.exists():
        train_arrays = load_ucihar_arrays(root, "train")
        test_arrays = load_ucihar_arrays(root, "test")
    elif smoke_test:
        synthetic = True
        synthetic_data = create_synthetic_ucihar(seed=seed)
        train_arrays = synthetic_data["train"]
        test_arrays = synthetic_data["test"]
    else:
        raise FileNotFoundError(
            f"UCI-HAR dataset not found at {root}. Run scripts/download_ucihar.py first. "
            "For code-only validation, pass --smoke_test to use synthetic data."
        )

    train_arrays, val_arrays = split_train_val_by_subject(
        *train_arrays,
        val_subject_fraction=float(dataset_cfg.get("val_subject_fraction", 0.2)),
        seed=seed,
    )

    max_train = dataset_cfg.get("smoke_max_train_sequences") if smoke_test else None
    max_val = dataset_cfg.get("smoke_max_val_sequences") if smoke_test else None
    max_test = dataset_cfg.get("smoke_max_test_sequences") if smoke_test else None

    train_dataset = SequenceWindowDataset(
        *train_arrays,
        context_len=context_len,
        channel_names=UCIHAR_CHANNELS,
        max_sequences=max_train,
        seed=seed,
    )
    val_dataset = SequenceWindowDataset(
        *val_arrays,
        context_len=context_len,
        channel_names=UCIHAR_CHANNELS,
        max_sequences=max_val,
        seed=seed + 10_000,
    )
    test_dataset = SequenceWindowDataset(
        *test_arrays,
        context_len=context_len,
        channel_names=UCIHAR_CHANNELS,
        max_sequences=max_test,
        seed=seed + 20_000,
        **perturbation,
    )

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))
    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    meta = SequenceDatasetMeta(
        context_len=context_len,
        num_channels=9,
        window_size=128,
        num_classes=6,
        channel_names=UCIHAR_CHANNELS,
        synthetic=synthetic,
    )
    return loaders, meta
