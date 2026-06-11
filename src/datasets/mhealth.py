from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from .preprocessing import apply_train_preprocessor, fit_train_preprocessor, stats_to_serializable
from .sequence_dataset import SequenceDatasetMeta, SequenceWindowDataset
from .ucihar import split_train_val_by_subject

MHEALTH_NUM_CLASSES = 12

_ACC_GYRO_INDICES = (
    0,
    1,
    2,  # chest acceleration
    5,
    6,
    7,
    8,
    9,
    10,  # left ankle acceleration + gyro
    14,
    15,
    16,
    17,
    18,
    19,  # right lower-arm acceleration + gyro
)

MHEALTH_CHANNELS = (
    "chest_acc_x",
    "chest_acc_y",
    "chest_acc_z",
    "ankle_acc_x",
    "ankle_acc_y",
    "ankle_acc_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "arm_acc_x",
    "arm_acc_y",
    "arm_acc_z",
    "arm_gyro_x",
    "arm_gyro_y",
    "arm_gyro_z",
)


def resolve_mhealth_root(root: str | Path) -> Path:
    root = Path(root)
    candidates = [root, root / "MHEALTHDATASET", root / "MHEALTH Dataset" / "MHEALTHDATASET"]
    for candidate in candidates:
        if list(candidate.glob("mHealth_subject*.log")):
            return candidate
    return root


def load_mhealth_windows(
    root: str | Path,
    split: str,
    window_size: int = 128,
    stride: int = 64,
    test_subjects: tuple[int, ...] | list[int] = (9, 10),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    root = resolve_mhealth_root(root)
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    files = sorted(root.glob("mHealth_subject*.log"))
    if not files:
        raise FileNotFoundError(
            f"No MHEALTH mHealth_subject*.log files found under {root}. "
            "Run scripts/download_mhealth.py or place MHEALTHDATASET under data/."
        )
    test_set = {int(subject) for subject in test_subjects}
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    segment_ids: list[int] = []
    activity_ids: list[int] = []
    segment_counter = 0
    for path in files:
        subject_id = _parse_subject_id(path)
        is_test = subject_id in test_set
        if (split == "test") != is_test:
            continue
        raw = np.loadtxt(path, dtype=np.float32)
        raw = np.atleast_2d(raw)
        labels = raw[:, -1].astype(np.int64)
        signal = raw[:, list(_ACC_GYRO_INDICES)].astype(np.float32, copy=False)
        for start, end, label_id in _label_segments(labels):
            if int(label_id) <= 0:
                continue
            if end - start < window_size:
                continue
            segment_counter += 1
            label = int(label_id) - 1
            for offset in range(start, end - window_size + 1, stride):
                xs.append(signal[offset : offset + window_size].astype(np.float32, copy=False))
                ys.append(label)
                subjects.append(subject_id)
                segment_ids.append(segment_counter)
                activity_ids.append(int(label_id))
    if not xs:
        raise ValueError(f"No MHEALTH windows generated for split={split} at {root}")
    return (
        np.stack(xs).astype(np.float32, copy=False),
        np.asarray(ys, dtype=np.int64),
        np.asarray(subjects, dtype=np.int64),
        {
            "segment_ids": np.asarray(segment_ids, dtype=np.int64),
            "activity_ids": np.asarray(activity_ids, dtype=np.int64),
        },
    )


def _parse_subject_id(path: Path) -> int:
    stem = path.stem.lower().replace("mhealth_subject", "")
    return int(stem)


def _label_segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    if len(labels) == 0:
        return segments
    start = 0
    current = int(labels[0])
    for idx in range(1, len(labels)):
        label = int(labels[idx])
        if label != current:
            segments.append((start, idx, current))
            start = idx
            current = label
    segments.append((start, len(labels), current))
    return segments


def create_synthetic_mhealth(
    n_train_subjects: int = 6,
    n_test_subjects: int = 2,
    windows_per_subject: int = 30,
    window_size: int = 128,
    num_channels: int = len(MHEALTH_CHANNELS),
    num_classes: int = MHEALTH_NUM_CLASSES,
    seed: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)

    def make_split(subject_offset: int, n_subjects: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        subs: list[int] = []
        groups: list[int] = []
        time = np.linspace(0.0, 1.0, window_size, dtype=np.float32)
        for local_subject in range(n_subjects):
            subject_id = subject_offset + local_subject + 1
            bias = rng.normal(0.0, 0.05, size=(num_channels,)).astype(np.float32)
            for window_idx in range(windows_per_subject):
                label = (window_idx + local_subject) % num_classes
                template = np.cos(2 * np.pi * (1 + label % 6) * time + 0.15 * label)[:, None]
                noise = rng.normal(0.0, 0.04, size=(window_size, num_channels)).astype(np.float32)
                xs.append(template.astype(np.float32) + bias + noise)
                ys.append(label)
                subs.append(subject_id)
                groups.append(subject_id * 1000)
        return (
            np.stack(xs).astype(np.float32),
            np.asarray(ys, dtype=np.int64),
            np.asarray(subs, dtype=np.int64),
            {"segment_ids": np.asarray(groups, dtype=np.int64), "activity_ids": np.asarray(ys, dtype=np.int64)},
        )

    return {"train": make_split(0, n_train_subjects), "test": make_split(n_train_subjects, n_test_subjects)}


def create_mhealth_dataloaders(
    config: dict[str, Any],
    model_name: str | None = None,
    smoke_test: bool = False,
    perturbation: dict[str, Any] | None = None,
) -> tuple[dict[str, DataLoader], SequenceDatasetMeta]:
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    root = Path(dataset_cfg.get("root", "data/MHEALTHDATASET"))
    context_len = int(dataset_cfg.get("context_len", 8))
    seed = int(config.get("seed", 0))
    window_size = int(dataset_cfg.get("window_size", 128))
    stride = int(dataset_cfg.get("stride", 64))
    test_subjects = tuple(int(subject) for subject in dataset_cfg.get("test_subjects", [9, 10]))
    normalize = str(dataset_cfg.get("normalize", "none"))
    impute_missing = str(dataset_cfg.get("impute_missing", "none"))
    perturbation = perturbation or {}
    synthetic = False

    if list(resolve_mhealth_root(root).glob("mHealth_subject*.log")):
        train_arrays = load_mhealth_windows(root, "train", window_size, stride, test_subjects)
        test_arrays = load_mhealth_windows(root, "test", window_size, stride, test_subjects)
    elif smoke_test:
        synthetic = True
        train_arrays = create_synthetic_mhealth(window_size=window_size, seed=seed)["train"]
        test_arrays = create_synthetic_mhealth(window_size=window_size, seed=seed)["test"]
    else:
        raise FileNotFoundError(
            f"MHEALTH dataset not found at {root}. Run scripts/download_mhealth.py first. "
            "For code-only validation, pass --smoke_test."
        )

    train_base = train_arrays[:3]
    test_base = test_arrays[:3]
    train_meta = train_arrays[3]
    test_meta = test_arrays[3]
    train_split, val_split = split_train_val_by_subject(
        *train_base,
        val_subject_fraction=float(dataset_cfg.get("val_subject_fraction", 0.2)),
        seed=seed,
    )
    train_mask = _rows_in_split(train_base[2], train_split[2])
    val_mask = _rows_in_split(train_base[2], val_split[2])
    if normalize != "none" or impute_missing != "none":
        pre_stats = fit_train_preprocessor(train_split[0], normalize=normalize, impute_missing=impute_missing)
        train_split = (
            apply_train_preprocessor(train_split[0], pre_stats),
            train_split[1],
            train_split[2],
        )
        val_split = (
            apply_train_preprocessor(val_split[0], pre_stats),
            val_split[1],
            val_split[2],
        )
        test_base = (
            apply_train_preprocessor(test_base[0], pre_stats),
            test_base[1],
            test_base[2],
        )
        config["_dataset_runtime"] = {
            "preprocessing": stats_to_serializable(pre_stats),
        }
    max_train = dataset_cfg.get("smoke_max_train_sequences") if smoke_test else None
    max_val = dataset_cfg.get("smoke_max_val_sequences") if smoke_test else None
    max_test = dataset_cfg.get("smoke_max_test_sequences") if smoke_test else None

    train_dataset = SequenceWindowDataset(*train_split, context_len=context_len, channel_names=MHEALTH_CHANNELS, group_ids=train_meta["segment_ids"][train_mask], max_sequences=max_train, seed=seed)
    val_dataset = SequenceWindowDataset(*val_split, context_len=context_len, channel_names=MHEALTH_CHANNELS, group_ids=train_meta["segment_ids"][val_mask], max_sequences=max_val, seed=seed + 10_000)
    test_dataset = SequenceWindowDataset(*test_base, context_len=context_len, channel_names=MHEALTH_CHANNELS, group_ids=test_meta["segment_ids"], max_sequences=max_test, seed=seed + 20_000, **perturbation)
    loaders = {
        "train": DataLoader(train_dataset, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=True, num_workers=int(train_cfg.get("num_workers", 0))),
        "val": DataLoader(val_dataset, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0))),
        "test": DataLoader(test_dataset, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0))),
    }
    meta = SequenceDatasetMeta(
        context_len=context_len,
        num_channels=len(MHEALTH_CHANNELS),
        window_size=window_size,
        num_classes=MHEALTH_NUM_CLASSES,
        channel_names=MHEALTH_CHANNELS,
        synthetic=synthetic,
    )
    return loaders, meta


def _rows_in_split(all_subjects: np.ndarray, selected_subjects: np.ndarray) -> np.ndarray:
    selected = set(selected_subjects.tolist())
    return np.asarray([int(subject) in selected for subject in all_subjects], dtype=bool)
