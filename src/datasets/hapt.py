from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset

from .sequence_dataset import SequenceDatasetMeta, SequenceWindowDataset
from .ucihar import split_train_val_by_subject

HAPT_CHANNELS = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)

HAPT_NUM_CLASSES = 12
HAPT6_NUM_CLASSES = 6
HAPT_TRANSITION_BINARY_CLASSES = 2


class HAPTWindowDataset(Dataset):
    """Raw-window HAPT dataset built from official labels and raw inertial files."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        window_size: int = 128,
        stride: int = 64,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.root = Path(root)
        self.split = split
        self.x, self.y, self.subjects = load_hapt_arrays(self.root, split, window_size, stride)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, int]:
        return self.x[idx], int(self.y[idx]), int(self.subjects[idx])


def load_hapt_arrays(
    root: str | Path,
    split: str,
    window_size: int = 128,
    stride: int = 64,
    task: str = "hapt12",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, subjects, _ = load_hapt_windows(root, split, window_size=window_size, stride=stride, task=task)
    return x, y, subjects


def load_hapt_windows(
    root: str | Path,
    split: str,
    window_size: int = 128,
    stride: int = 64,
    task: str = "hapt12",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    root = Path(root)
    raw_dir = root / "RawData"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"HAPT RawData directory not found: {raw_dir}. "
            "Run scripts/download_hapt.py or place the official dataset under data/."
        )
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive")
    normalized_task = normalize_hapt_task(task)

    split_dir = root / ("Train" if split == "train" else "Test")
    subject_path = split_dir / f"subject_id_{split}.txt"
    if not subject_path.exists():
        raise FileNotFoundError(f"Missing HAPT subject split file: {subject_path}")
    split_subjects = set(np.loadtxt(subject_path, dtype=np.int64).reshape(-1).tolist())

    labels_path = raw_dir / "labels.txt"
    labels = np.loadtxt(labels_path, dtype=np.int64)
    labels = np.atleast_2d(labels)
    labels = labels[np.lexsort((labels[:, 3], labels[:, 0], labels[:, 1]))]

    signal_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    experiment_ids: list[int] = []
    segment_ids: list[int] = []
    activity_ids: list[int] = []
    segment_counter = 0

    for exp_id, user_id, activity_id, start, end in labels:
        if int(user_id) not in split_subjects:
            continue
        if normalized_task == "hapt6" and int(activity_id) > 6:
            continue
        if normalized_task == "transitionbinary":
            label = 0 if int(activity_id) <= 6 else 1
        else:
            label = int(activity_id) - 1
        acc, gyro = _load_raw_pair(raw_dir, int(exp_id), int(user_id), signal_cache)
        start_idx = max(0, int(start) - 1)
        end_idx = min(len(acc), int(end))
        if end_idx - start_idx < window_size:
            continue
        segment_counter += 1
        for offset in range(start_idx, end_idx - window_size + 1, stride):
            window = np.concatenate(
                [acc[offset : offset + window_size], gyro[offset : offset + window_size]],
                axis=1,
            ).astype(np.float32, copy=False)
            xs.append(window)
            ys.append(label)
            subjects.append(int(user_id))
            experiment_ids.append(int(exp_id))
            segment_ids.append(segment_counter)
            activity_ids.append(int(activity_id))

    if not xs:
        raise ValueError(f"No HAPT windows were generated for split={split} at {root}")
    return (
        np.stack(xs).astype(np.float32, copy=False),
        np.asarray(ys, dtype=np.int64),
        np.asarray(subjects, dtype=np.int64),
        {
            "experiment_ids": np.asarray(experiment_ids, dtype=np.int64),
            "segment_ids": np.asarray(segment_ids, dtype=np.int64),
            "activity_ids": np.asarray(activity_ids, dtype=np.int64),
        },
    )


def normalize_hapt_task(task: str) -> str:
    normalized = task.lower().replace("-", "").replace("_", "")
    if normalized in {"hapt", "hapt12", "12"}:
        return "hapt12"
    if normalized in {"hapt6", "6"}:
        return "hapt6"
    if normalized in {"hapttransitionbinary", "transitionbinary", "haptbinary", "binary"}:
        return "transitionbinary"
    raise ValueError(f"Unsupported HAPT task: {task}")


def _load_raw_pair(
    raw_dir: Path,
    exp_id: int,
    user_id: int,
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    key = (exp_id, user_id)
    if key not in cache:
        suffix = f"exp{exp_id:02d}_user{user_id:02d}.txt"
        acc_path = raw_dir / f"acc_{suffix}"
        gyro_path = raw_dir / f"gyro_{suffix}"
        if not acc_path.exists() or not gyro_path.exists():
            raise FileNotFoundError(f"Missing HAPT raw pair: {acc_path} / {gyro_path}")
        acc = np.loadtxt(acc_path, dtype=np.float32)
        gyro = np.loadtxt(gyro_path, dtype=np.float32)
        if acc.shape != gyro.shape or acc.ndim != 2 or acc.shape[1] != 3:
            raise ValueError(f"Unexpected HAPT raw shape for {suffix}: acc={acc.shape}, gyro={gyro.shape}")
        cache[key] = (acc, gyro)
    return cache[key]


def create_synthetic_hapt(
    n_train_subjects: int = 4,
    n_test_subjects: int = 2,
    windows_per_subject: int = 24,
    window_size: int = 128,
    num_channels: int = 6,
    num_classes: int = HAPT_NUM_CLASSES,
    seed: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)

    def make_split(subject_offset: int, n_subjects: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        subs: list[int] = []
        time = np.linspace(0.0, 1.0, window_size, dtype=np.float32)
        templates = np.stack(
            [np.sin(2.0 * np.pi * (1 + label % 6) * time + 0.25 * label) for label in range(num_classes)]
        ).astype(np.float32)
        for local_subject in range(n_subjects):
            subject_id = subject_offset + local_subject + 1
            subject_bias = rng.normal(0.0, 0.08, size=(num_channels,)).astype(np.float32)
            for window_idx in range(windows_per_subject):
                label = (window_idx + local_subject) % num_classes
                channel_scale = rng.uniform(0.4, 1.6, size=(num_channels,)).astype(np.float32)
                noise = rng.normal(0.0, 0.05, size=(window_size, num_channels)).astype(np.float32)
                xs.append(templates[label][:, None] * channel_scale + subject_bias + noise)
                ys.append(label)
                subs.append(subject_id)
        return np.stack(xs), np.asarray(ys, dtype=np.int64), np.asarray(subs, dtype=np.int64)

    return {
        "train": make_split(0, n_train_subjects),
        "test": make_split(n_train_subjects, n_test_subjects),
    }


def create_hapt_dataloaders(
    config: dict[str, Any],
    model_name: str | None = None,
    smoke_test: bool = False,
    perturbation: dict[str, Any] | None = None,
) -> tuple[dict[str, DataLoader], SequenceDatasetMeta]:
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    root = Path(dataset_cfg.get("root", "data/HAPT Dataset"))
    context_len = int(dataset_cfg.get("context_len", 1))
    seed = int(config.get("seed", 0))
    window_size = int(dataset_cfg.get("window_size", 128))
    stride = int(dataset_cfg.get("stride", 64))
    task = normalize_hapt_task(str(dataset_cfg.get("task", dataset_cfg.get("name", "hapt12"))))
    sequence_within_segment = bool(dataset_cfg.get("sequence_within_segment", True))
    perturbation = perturbation or {}
    synthetic = False

    if root.exists():
        train_arrays = load_hapt_windows(root, "train", window_size=window_size, stride=stride, task=task)
        test_arrays = load_hapt_windows(root, "test", window_size=window_size, stride=stride, task=task)
    elif smoke_test:
        synthetic = True
        synthetic_num_classes = HAPT_TRANSITION_BINARY_CLASSES if task == "transitionbinary" else HAPT_NUM_CLASSES
        synthetic_data = create_synthetic_hapt(
            window_size=window_size,
            num_classes=synthetic_num_classes,
            seed=seed,
        )
        train_arrays = (*synthetic_data["train"], _synthetic_groups(synthetic_data["train"][0]))
        test_arrays = (*synthetic_data["test"], _synthetic_groups(synthetic_data["test"][0]))
    else:
        raise FileNotFoundError(
            f"HAPT dataset not found at {root}. Run scripts/download_hapt.py first. "
            "For code-only validation, pass --smoke_test to use synthetic data."
        )

    train_base = train_arrays[:3]
    test_base = test_arrays[:3]
    train_meta = train_arrays[3]
    test_meta = test_arrays[3]
    (train_split, val_split) = split_train_val_by_subject(
        *train_base,
        val_subject_fraction=float(dataset_cfg.get("val_subject_fraction", 0.2)),
        seed=seed,
    )
    train_mask = _rows_in_split(train_base[2], train_split[2])
    val_mask = _rows_in_split(train_base[2], val_split[2])
    train_groups = _select_group_ids(train_meta, train_mask, sequence_within_segment)
    val_groups = _select_group_ids(train_meta, val_mask, sequence_within_segment)
    test_groups = _select_group_ids(test_meta, np.ones(len(test_base[0]), dtype=bool), sequence_within_segment)

    max_train = dataset_cfg.get("smoke_max_train_sequences") if smoke_test else None
    max_val = dataset_cfg.get("smoke_max_val_sequences") if smoke_test else None
    max_test = dataset_cfg.get("smoke_max_test_sequences") if smoke_test else None

    train_dataset = SequenceWindowDataset(
        *train_split,
        context_len=context_len,
        channel_names=HAPT_CHANNELS,
        max_sequences=max_train,
        group_ids=train_groups,
        seed=seed,
    )
    val_dataset = SequenceWindowDataset(
        *val_split,
        context_len=context_len,
        channel_names=HAPT_CHANNELS,
        max_sequences=max_val,
        group_ids=val_groups,
        seed=seed + 10_000,
    )
    test_dataset = SequenceWindowDataset(
        *test_base,
        context_len=context_len,
        channel_names=HAPT_CHANNELS,
        max_sequences=max_test,
        group_ids=test_groups,
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
    if task == "hapt6":
        num_classes = HAPT6_NUM_CLASSES
    elif task == "transitionbinary":
        num_classes = HAPT_TRANSITION_BINARY_CLASSES
    else:
        num_classes = HAPT_NUM_CLASSES
    meta = SequenceDatasetMeta(
        context_len=context_len,
        num_channels=len(HAPT_CHANNELS),
        window_size=window_size,
        num_classes=num_classes,
        channel_names=HAPT_CHANNELS,
        synthetic=synthetic,
    )
    return loaders, meta


def _synthetic_groups(x: np.ndarray) -> dict[str, np.ndarray]:
    ids = np.zeros(len(x), dtype=np.int64)
    return {"experiment_ids": ids, "segment_ids": ids, "activity_ids": ids}


def _rows_in_split(all_subjects: np.ndarray, selected_subjects: np.ndarray) -> np.ndarray:
    selected = set(selected_subjects.tolist())
    return np.asarray([int(subject) in selected for subject in all_subjects], dtype=bool)


def _select_group_ids(meta: dict[str, np.ndarray], mask: np.ndarray, sequence_within_segment: bool) -> np.ndarray:
    if sequence_within_segment:
        return meta["segment_ids"][mask]
    return meta["experiment_ids"][mask]
