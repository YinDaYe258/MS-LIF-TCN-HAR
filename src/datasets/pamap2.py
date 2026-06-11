from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from .preprocessing import apply_train_preprocessor, fit_train_preprocessor, stats_to_serializable
from .sequence_dataset import SequenceDatasetMeta, SequenceWindowDataset
from .ucihar import split_train_val_by_subject

PAMAP2_ACTIVITY_IDS = (1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24)
PAMAP2_LABEL_MAP = {activity_id: idx for idx, activity_id in enumerate(PAMAP2_ACTIVITY_IDS)}

_BLOCKS = {
    "hand": 3,
    "chest": 20,
    "ankle": 37,
}


def pamap2_channel_indices(channel_set: str = "acc_gyro") -> tuple[list[int], tuple[str, ...]]:
    """Return raw-column indices for the selected PAMAP2 sensor subset."""
    normalized = channel_set.lower().replace("-", "_")
    if normalized not in {"acc_gyro", "acc_gyro_mag"}:
        raise ValueError(f"Unsupported PAMAP2 channel_set: {channel_set}")
    indices: list[int] = []
    names: list[str] = []
    for sensor, start in _BLOCKS.items():
        # IMU block layout after timestamp/activity/HR:
        # temperature, acc16g xyz, acc6g xyz, gyro xyz, mag xyz, orientation wxyz.
        for axis, offset in zip(("x", "y", "z"), (1, 2, 3)):
            indices.append(start + offset)
            names.append(f"{sensor}_acc_{axis}")
        for axis, offset in zip(("x", "y", "z"), (7, 8, 9)):
            indices.append(start + offset)
            names.append(f"{sensor}_gyro_{axis}")
        if normalized == "acc_gyro_mag":
            for axis, offset in zip(("x", "y", "z"), (10, 11, 12)):
                indices.append(start + offset)
                names.append(f"{sensor}_mag_{axis}")
    return indices, tuple(names)


PAMAP2_CHANNELS = pamap2_channel_indices("acc_gyro")[1]


def resolve_pamap2_root(root: str | Path) -> Path:
    root = Path(root)
    candidates = [
        root,
        root / "PAMAP2_Dataset",
        root / "pamap2+physical+activity+monitoring" / "PAMAP2_Dataset",
    ]
    for candidate in candidates:
        if (candidate / "Protocol").exists():
            return candidate
    return root


def load_pamap2_windows(
    root: str | Path,
    split: str,
    window_size: int = 256,
    stride: int = 128,
    channel_set: str = "acc_gyro",
    test_subjects: tuple[int, ...] | list[int] = (105, 106),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    root = resolve_pamap2_root(root)
    protocol_dir = root / "Protocol"
    if not protocol_dir.exists():
        raise FileNotFoundError(
            f"PAMAP2 Protocol directory not found: {protocol_dir}. "
            "Run scripts/download_pamap2.py or place PAMAP2_Dataset under data/."
        )
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive")

    test_set = {int(subject) for subject in test_subjects}
    channel_indices, _ = pamap2_channel_indices(channel_set)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    segment_ids: list[int] = []
    activity_ids: list[int] = []
    segment_counter = 0

    files = sorted(protocol_dir.glob("subject*.dat"))
    if not files:
        raise FileNotFoundError(f"No PAMAP2 subject*.dat files found under {protocol_dir}")

    for file_path in files:
        subject_id = int(file_path.stem.replace("subject", ""))
        is_test = subject_id in test_set
        if (split == "test") != is_test:
            continue
        raw = np.loadtxt(file_path, dtype=np.float32)
        raw = np.atleast_2d(raw)
        labels = raw[:, 1].astype(np.int64)
        signal = raw[:, channel_indices].astype(np.float32, copy=False)
        for start, end, activity_id in _label_segments(labels):
            if int(activity_id) not in PAMAP2_LABEL_MAP:
                continue
            if end - start < window_size:
                continue
            segment_counter += 1
            label = PAMAP2_LABEL_MAP[int(activity_id)]
            for offset in range(start, end - window_size + 1, stride):
                xs.append(signal[offset : offset + window_size].astype(np.float32, copy=False))
                ys.append(label)
                subjects.append(subject_id)
                segment_ids.append(segment_counter)
                activity_ids.append(int(activity_id))

    if not xs:
        raise ValueError(f"No PAMAP2 windows generated for split={split} at {root}")
    return (
        np.stack(xs).astype(np.float32, copy=False),
        np.asarray(ys, dtype=np.int64),
        np.asarray(subjects, dtype=np.int64),
        {
            "segment_ids": np.asarray(segment_ids, dtype=np.int64),
            "activity_ids": np.asarray(activity_ids, dtype=np.int64),
        },
    )


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


def create_synthetic_pamap2(
    n_train_subjects: int = 5,
    n_test_subjects: int = 2,
    windows_per_subject: int = 30,
    window_size: int = 256,
    num_channels: int = 18,
    num_classes: int = len(PAMAP2_ACTIVITY_IDS),
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
            subject_id = 101 + subject_offset + local_subject
            bias = rng.normal(0.0, 0.06, size=(num_channels,)).astype(np.float32)
            for window_idx in range(windows_per_subject):
                label = (window_idx + local_subject) % num_classes
                freq = 1 + (label % 8)
                template = np.sin(2 * np.pi * freq * time + 0.1 * label)[:, None]
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


def create_pamap2_dataloaders(
    config: dict[str, Any],
    model_name: str | None = None,
    smoke_test: bool = False,
    perturbation: dict[str, Any] | None = None,
) -> tuple[dict[str, DataLoader], SequenceDatasetMeta]:
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    root = Path(dataset_cfg.get("root", "data/PAMAP2_Dataset"))
    context_len = int(dataset_cfg.get("context_len", 8))
    seed = int(config.get("seed", 0))
    window_size = int(dataset_cfg.get("window_size", 256))
    stride = int(dataset_cfg.get("stride", 128))
    channel_set = str(dataset_cfg.get("channel_set", "acc_gyro"))
    test_subjects = tuple(int(subject) for subject in dataset_cfg.get("test_subjects", [105, 106]))
    perturbation = perturbation or {}
    synthetic = False
    if smoke_test:
        dataset_cfg["_smoke_test"] = True

    if resolve_pamap2_root(root).joinpath("Protocol").exists():
        train_arrays = load_pamap2_windows(root, "train", window_size, stride, channel_set, test_subjects)
        test_arrays = load_pamap2_windows(root, "test", window_size, stride, channel_set, test_subjects)
    elif smoke_test:
        synthetic = True
        train_arrays = create_synthetic_pamap2(window_size=window_size, seed=seed)["train"]
        test_arrays = create_synthetic_pamap2(window_size=window_size, seed=seed)["test"]
    else:
        raise FileNotFoundError(
            f"PAMAP2 dataset not found at {root}. Run scripts/download_pamap2.py first. "
            "For code-only validation, pass --smoke_test."
        )

    return _build_loaders(config, train_arrays, test_arrays, context_len, channel_set, window_size, synthetic, perturbation)


def _build_loaders(
    config: dict[str, Any],
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]],
    test_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]],
    context_len: int,
    channel_set: str,
    window_size: int,
    synthetic: bool,
    perturbation: dict[str, Any],
) -> tuple[dict[str, DataLoader], SequenceDatasetMeta]:
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    seed = int(config.get("seed", 0))
    normalize = str(dataset_cfg.get("normalize", "none"))
    impute_missing = str(dataset_cfg.get("impute_missing", "none"))
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
    train_groups = train_meta["segment_ids"][train_mask]
    val_groups = train_meta["segment_ids"][val_mask]
    test_groups = test_meta["segment_ids"]
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
    max_train = dataset_cfg.get("smoke_max_train_sequences") if dataset_cfg.get("_smoke_test", False) else None
    max_val = dataset_cfg.get("smoke_max_val_sequences") if dataset_cfg.get("_smoke_test", False) else None
    max_test = dataset_cfg.get("smoke_max_test_sequences") if dataset_cfg.get("_smoke_test", False) else None
    _, channel_names = pamap2_channel_indices(channel_set)

    train_dataset = SequenceWindowDataset(*train_split, context_len=context_len, channel_names=channel_names, group_ids=train_groups, max_sequences=max_train, seed=seed)
    val_dataset = SequenceWindowDataset(*val_split, context_len=context_len, channel_names=channel_names, group_ids=val_groups, max_sequences=max_val, seed=seed + 10_000)
    test_dataset = SequenceWindowDataset(*test_base, context_len=context_len, channel_names=channel_names, group_ids=test_groups, max_sequences=max_test, seed=seed + 20_000, **perturbation)
    loaders = {
        "train": DataLoader(train_dataset, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=True, num_workers=int(train_cfg.get("num_workers", 0))),
        "val": DataLoader(val_dataset, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0))),
        "test": DataLoader(test_dataset, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0))),
    }
    meta = SequenceDatasetMeta(
        context_len=context_len,
        num_channels=len(channel_names),
        window_size=window_size,
        num_classes=len(PAMAP2_ACTIVITY_IDS),
        channel_names=channel_names,
        synthetic=synthetic,
    )
    return loaders, meta


def _rows_in_split(all_subjects: np.ndarray, selected_subjects: np.ndarray) -> np.ndarray:
    selected = set(selected_subjects.tolist())
    return np.asarray([int(subject) in selected for subject in all_subjects], dtype=bool)
