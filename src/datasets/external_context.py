from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .aligned_sequence_dataset import AlignedSequenceWindowDataset
from .preprocessing import apply_train_preprocessor, fit_train_preprocessor, stats_to_serializable
from .sequence_dataset import SequenceDatasetMeta, SequenceWindowDataset

WISDM_AR_LABELS = ("Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing")
WISDM_AT_LABELS = ("Walking", "Jogging", "Stairs", "Sitting", "Standing", "LyingDown")
CHEST_ACCEL_LABELS = tuple(str(idx) for idx in range(1, 8))
EXTERNAL_CHANNELS_3 = ("acc_x", "acc_y", "acc_z")
CAPTURE24_MOVEMENT4_LABELS = ("sleep_rest", "sedentary", "light_activity", "moderate_vigorous")
CAPTURE24_MOVEMENT4_PATTERNS: tuple[tuple[str, str], ...] = (
    ("sleep_rest", "sleep|rest"),
    ("sedentary", "sedentary|sit|sitting|lying|lie|desk|computer|tv"),
    ("light_activity", "light|household|chores|standing|slow|walk"),
    ("moderate_vigorous", "moderate|vigorous|mvpa|sport|run|running|cycle|cycling|exercise"),
)
HUGADB_LABELS = {
    1: "Walking",
    2: "Running",
    3: "Going up",
    4: "Going down",
    5: "Sitting",
    6: "Sitting down",
    7: "Standing up",
    8: "Standing",
    9: "Bicycling",
    10: "Up by elevator",
    11: "Down by elevator",
    12: "Sitting in car",
}
HUGADB_TASKS: dict[str, dict[str, Any]] = {
    "hugadb": {"labels": tuple(HUGADB_LABELS), "class_names": tuple(HUGADB_LABELS.values())},
    "hugadb_inertial": {"labels": tuple(HUGADB_LABELS), "class_names": tuple(HUGADB_LABELS.values())},
    "hugadb_locomotion": {
        "labels": (1, 2, 3, 4, 8),
        "class_names": ("Walking", "Running", "Going up", "Going down", "Standing"),
    },
    "hugadb_posture_transition": {
        "labels": (5, 6, 7, 8),
        "class_names": ("Sitting", "Sitting down", "Standing up", "Standing"),
    },
    "hugadb_stairs": {
        "labels": (1, 3, 4, 8),
        "class_names": ("Walking", "Going up", "Going down", "Standing"),
    },
    "hugadb_transition_binary": {
        "labels": (5, 6, 7, 8),
        "binary_map": {5: 0, 8: 0, 6: 1, 7: 1},
        "class_names": ("Stable posture", "Transition"),
    },
}
DEFAULT_SPLIT_SEED = 20260604


@dataclass(frozen=True)
class ExternalArrays:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    group_ids: np.ndarray
    class_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    runs: np.ndarray | None = None


def create_external_context_dataloaders(
    config: dict[str, Any],
    model_name: str | None = None,
    smoke_test: bool = False,
    perturbation: dict[str, Any] | None = None,
) -> tuple[dict[str, DataLoader], SequenceDatasetMeta]:
    dataset_cfg = config.get("dataset", {})
    name = str(dataset_cfg.get("name", "")).lower()
    seed = int(config.get("seed", 0))
    perturbation = perturbation or {}
    synthetic = False

    if smoke_test:
        dataset_cfg["_smoke_test"] = True

    if name in {"wisdm_ar", "wisdm_at"}:
        if smoke_test and not bool(dataset_cfg.get("smoke_use_real_data", False)):
            arrays = create_synthetic_external(
                num_classes=len(WISDM_AR_LABELS if name == "wisdm_ar" else WISDM_AT_LABELS),
                window_size=int(dataset_cfg.get("window_size", 200)),
                num_channels=3,
                seed=seed,
                class_names=WISDM_AR_LABELS if name == "wisdm_ar" else WISDM_AT_LABELS,
            )
            synthetic = True
        else:
            arrays = load_wisdm_windows(
                dataset_cfg.get("root", f"data/external/{name}"),
                dataset_key=name,
                window_size=int(dataset_cfg.get("window_size", 200)),
                stride=int(dataset_cfg.get("stride", 100)),
                inspection_dir=dataset_cfg.get("inspection_dir", dataset_cfg.get("support_audit_dir", "results/external_context_v1")),
                sort_by_subject_timestamp=bool(dataset_cfg.get("sort_by_subject_timestamp", False)),
            )
            config.setdefault("_dataset_runtime", {})["inspection_path"] = str(
                Path(dataset_cfg.get("inspection_dir", dataset_cfg.get("support_audit_dir", "results/external_context_v1")))
                / f"inspection_{name}.json"
            )
    elif name == "chest_accel":
        if smoke_test and not bool(dataset_cfg.get("smoke_use_real_data", False)):
            arrays = create_synthetic_external(
                num_classes=7,
                window_size=int(dataset_cfg.get("window_size", 128)),
                num_channels=3,
                seed=seed,
                class_names=CHEST_ACCEL_LABELS,
            )
            synthetic = True
        else:
            arrays = load_chest_accel_windows(
                dataset_cfg.get("root", "data/external/chest_accel"),
                window_size=int(dataset_cfg.get("window_size", 128)),
                stride=int(dataset_cfg.get("stride", 64)),
                inspection_dir=dataset_cfg.get("inspection_dir", dataset_cfg.get("support_audit_dir", "results/external_context_v1")),
            )
            config.setdefault("_dataset_runtime", {})["inspection_path"] = str(
                Path(dataset_cfg.get("inspection_dir", dataset_cfg.get("support_audit_dir", "results/external_context_v1")))
                / "inspection_chest_accel.json"
            )
    elif name in HUGADB_TASKS:
        task_spec = hugadb_task_spec(name)
        channel_set = str(dataset_cfg.get("channel_set", "inertial36"))
        num_channels = 38 if channel_set == "inertial36_emg2" else 36
        if smoke_test and not bool(dataset_cfg.get("smoke_use_real_data", False)):
            arrays = create_synthetic_external(
                num_classes=len(task_spec["class_names"]),
                window_size=int(dataset_cfg.get("window_size", 256)),
                num_channels=num_channels,
                seed=seed,
                class_names=tuple(task_spec["class_names"]),
            )
            synthetic = True
        else:
            arrays = load_hugadb_windows(
                dataset_cfg.get("root", "data/external/hugadb"),
                window_size=int(dataset_cfg.get("window_size", 256)),
                stride=int(dataset_cfg.get("stride", 128)),
                channel_set=channel_set,
                inspection_dir=dataset_cfg.get("inspection_dir", dataset_cfg.get("support_audit_dir", "results/external_context_v1")),
                exclude_known_corrupted_gyro=bool(dataset_cfg.get("exclude_known_corrupted_gyro", True)),
                dataset_key=name,
            )
            config.setdefault("_dataset_runtime", {})["inspection_path"] = str(
                Path(dataset_cfg.get("inspection_dir", dataset_cfg.get("support_audit_dir", "results/external_context_v1")))
                / f"inspection_{name}.json"
            )
    elif name in {"opportunity_locomotion_subject", "opportunity_locomotion_session", "opportunity_locomotion"}:
        if smoke_test and not bool(dataset_cfg.get("smoke_use_real_data", False)):
            arrays = create_synthetic_external(
                num_classes=4,
                window_size=int(dataset_cfg.get("window_size", 64)),
                num_channels=int(dataset_cfg.get("smoke_num_channels", 24)),
                seed=seed,
                class_names=("Stand", "Walk", "Sit", "Lie"),
            )
            synthetic = True
        else:
            arrays = load_opportunity_windows(
                dataset_cfg.get("root", "data/external/opportunity"),
                dataset_key=name,
                task="locomotion",
                protocol_type=str(dataset_cfg.get("protocol_type", "subject_disjoint")),
                feature_set=str(dataset_cfg.get("feature_set", "body_worn")),
                window_size=int(dataset_cfg.get("window_size", 64)),
                stride=int(dataset_cfg.get("stride", 32)),
                inspection_dir=dataset_cfg.get("inspection_dir", dataset_cfg.get("support_audit_dir", "results/external_context_v1")),
                drop_feature_missing_threshold=float(dataset_cfg.get("drop_feature_missing_threshold", 0.8)),
            )
            config.setdefault("_dataset_runtime", {})["inspection_path"] = str(
                Path(dataset_cfg.get("inspection_dir", dataset_cfg.get("support_audit_dir", "results/external_context_v1")))
                / f"inspection_{name}.json"
            )
    elif name in {"capture24_movement4", "capture24_activity6"}:
        if smoke_test and not bool(dataset_cfg.get("smoke_use_real_data", False)):
            arrays = create_synthetic_external(
                num_classes=len(CAPTURE24_MOVEMENT4_LABELS),
                window_size=int(dataset_cfg.get("window_size", 1000)),
                num_channels=3,
                seed=seed,
                class_names=CAPTURE24_MOVEMENT4_LABELS,
            )
            synthetic = True
        else:
            arrays = load_capture24_windows(
                dataset_cfg.get("root", "data/external/capture24"),
                dataset_key=name,
                window_size=int(dataset_cfg.get("window_size", 1000)),
                stride=int(dataset_cfg.get("stride", 500)),
                inspection_dir=dataset_cfg.get("inspection_dir", dataset_cfg.get("support_audit_dir", "results/robust_context_v4")),
                label_map_path=dataset_cfg.get("label_map_path"),
                max_files=dataset_cfg.get("max_files"),
                downsample_bins=dataset_cfg.get("downsample_bins"),
                max_windows_per_subject_class=dataset_cfg.get("max_windows_per_subject_class"),
            )
            config.setdefault("_dataset_runtime", {})["inspection_path"] = str(
                Path(dataset_cfg.get("inspection_dir", dataset_cfg.get("support_audit_dir", "results/robust_context_v4")))
                / f"capture24_inspection.json"
            )
    else:
        raise ValueError(f"Unsupported external context dataset: {name}")

    return _build_external_loaders(
        config,
        arrays,
        synthetic=synthetic,
        perturbation=perturbation,
    )


def load_wisdm_windows(
    root: str | Path,
    dataset_key: str = "wisdm_ar",
    window_size: int = 200,
    stride: int = 100,
    inspection_dir: str | Path | None = None,
    sort_by_subject_timestamp: bool = False,
) -> ExternalArrays:
    root = Path(root)
    raw_file = find_wisdm_raw_file(root, dataset_key)
    class_names = WISDM_AR_LABELS if dataset_key == "wisdm_ar" else WISDM_AT_LABELS
    label_map = {label.lower(): idx for idx, label in enumerate(class_names)}
    rows: list[tuple[int, str, float, np.ndarray]] = []
    num_raw_rows = 0
    with raw_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            num_raw_rows += 1
            parsed = parse_wisdm_line(line)
            if parsed is None:
                continue
            subject, activity, timestamp, values = parsed
            if activity.lower() not in label_map:
                continue
            rows.append((subject, activity, timestamp, values))
    if not rows:
        raise ValueError(f"No labeled WISDM rows parsed from {raw_file}")
    warnings = wisdm_timestamp_warnings(rows)
    if sort_by_subject_timestamp:
        rows = sorted(rows, key=lambda item: (item[0], item[2]))
    x, y, subjects, groups, segment_info = _window_wisdm_rows(rows, label_map, window_size, stride)
    if inspection_dir is not None:
        write_wisdm_inspection(
            Path(inspection_dir),
            dataset_key,
            raw_file,
            rows,
            label_map,
            y,
            num_raw_rows,
            segment_info,
            warnings,
        )
    return ExternalArrays(x, y, subjects, groups, class_names, EXTERNAL_CHANNELS_3)


def find_wisdm_raw_file(root: Path, dataset_key: str) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"WISDM root not found: {root}")
    preferred = "WISDM_ar_v1.1_raw.txt" if dataset_key == "wisdm_ar" else "WISDM_at_v2.0_raw.txt"
    matches = [path for path in root.rglob(preferred) if path.is_file()]
    if not matches:
        prefix = "WISDM_ar" if dataset_key == "wisdm_ar" else "WISDM_at"
        matches = [
            path
            for path in root.rglob("*.txt")
            if prefix.lower() in path.name.lower()
            and "raw" in path.name.lower()
            and "unlabeled" not in path.name.lower()
            and "about" not in path.name.lower()
        ]
    if not matches:
        raise FileNotFoundError(f"No WISDM raw file found under {root} for {dataset_key}")
    return sorted(matches)[0]


def parse_wisdm_line(line: str) -> tuple[int, str, float, np.ndarray] | None:
    cleaned = line.strip().rstrip(";")
    if not cleaned:
        return None
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) < 6:
        return None
    try:
        subject = int(parts[0])
        activity = parts[1]
        timestamp = float(parts[2])
        values = np.asarray([float(parts[3]), float(parts[4]), float(parts[5])], dtype=np.float32)
    except ValueError:
        return None
    if not np.isfinite(values).all():
        return None
    return subject, activity, timestamp, values


def _window_wisdm_rows(
    rows: list[tuple[int, str, float, np.ndarray]],
    label_map: dict[str, int],
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    group_ids: list[int] = []
    segment_info: list[dict[str, Any]] = []
    group_counter = 0
    start = 0
    while start < len(rows):
        subject, activity, _, _ = rows[start]
        end = start + 1
        while end < len(rows) and rows[end][0] == subject and rows[end][1] == activity:
            end += 1
        values = np.stack([row[3] for row in rows[start:end]]).astype(np.float32, copy=False)
        segment_windows = 0
        if len(values) >= window_size:
            group_counter += 1
            label = label_map[activity.lower()]
            for offset in range(0, len(values) - window_size + 1, stride):
                xs.append(values[offset : offset + window_size])
                ys.append(label)
                subjects.append(subject)
                group_ids.append(group_counter)
                segment_windows += 1
        segment_info.append(
            {
                "subject": int(subject),
                "activity": str(activity),
                "raw_length": int(len(values)),
                "num_windows": int(segment_windows),
                "short_segment": bool(len(values) < window_size),
            }
        )
        start = end
    if not xs:
        raise ValueError("No WISDM windows generated; check window_size/stride and raw file format")
    return (
        np.stack(xs).astype(np.float32, copy=False),
        np.asarray(ys, dtype=np.int64),
        np.asarray(subjects, dtype=np.int64),
        np.asarray(group_ids, dtype=np.int64),
        segment_info,
    )


def wisdm_timestamp_warnings(rows: list[tuple[int, str, float, np.ndarray]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    last_by_subject: dict[int, float] = {}
    for index, (subject, _activity, timestamp, _values) in enumerate(rows):
        previous = last_by_subject.get(int(subject))
        if previous is not None and float(timestamp) < float(previous):
            warnings.append(
                {
                    "type": "timestamp_decrease",
                    "subject": int(subject),
                    "row_index": int(index),
                    "previous_timestamp": float(previous),
                    "timestamp": float(timestamp),
                }
            )
        last_by_subject[int(subject)] = float(timestamp)
    return warnings


def write_wisdm_inspection(
    output_dir: Path,
    dataset_key: str,
    raw_file: Path,
    rows: list[tuple[int, str, float, np.ndarray]],
    label_map: dict[str, int],
    y_windows: np.ndarray,
    num_raw_rows: int,
    segment_info: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_counts = {label: 0 for label in label_map}
    for _subject, activity, _timestamp, _values in rows:
        raw_counts[activity.lower()] = raw_counts.get(activity.lower(), 0) + 1
    windows_per_class: dict[str, int] = {}
    inverse_labels = {idx: label for label, idx in label_map.items()}
    for label_id in range(len(label_map)):
        windows_per_class[inverse_labels[label_id]] = int((y_windows == label_id).sum())
    window_counts = list(windows_per_class.values())
    inspection = {
        "dataset_key": dataset_key,
        "raw_file": str(raw_file),
        "num_raw_rows": int(num_raw_rows),
        "num_parsed_rows": int(len(rows)),
        "subjects": sorted(int(subject) for subject in {row[0] for row in rows}),
        "class_counts_raw": raw_counts,
        "num_segments": int(len(segment_info)),
        "windows_per_class": windows_per_class,
        "min_windows_per_class": int(min(window_counts)) if window_counts else 0,
        "max_windows_per_class": int(max(window_counts)) if window_counts else 0,
        "suspected_short_segments": int(sum(1 for item in segment_info if item["short_segment"])),
        "warnings": warnings,
    }
    (output_dir / f"inspection_{dataset_key}.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")


def load_chest_accel_windows(
    root: str | Path,
    window_size: int = 128,
    stride: int = 64,
    inspection_dir: str | Path | None = None,
) -> ExternalArrays:
    root = Path(root)
    files = find_chest_accel_files(root)
    raw_rows: list[tuple[int, int, np.ndarray]] = []
    labels_seen: set[int] = set()
    file_inspection: list[dict[str, Any]] = []
    for file_path in files:
        subject = _subject_from_path(file_path)
        frame = pd.read_csv(file_path, header=None, sep=r"[,\s]+", engine="python")
        if frame.shape[1] < 5:
            file_inspection.append(
                {
                    "file": str(file_path),
                    "subject": int(subject),
                    "parsed_rows": 0,
                    "label_counts": {},
                    "warning": "fewer_than_five_columns",
                }
            )
            continue
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        keep = numeric.iloc[:, [1, 2, 3, numeric.shape[1] - 1]].notna().all(axis=1)
        numeric = numeric[keep]
        values = numeric.iloc[:, 1:4].to_numpy(dtype=np.float32)
        labels = numeric.iloc[:, -1].to_numpy(dtype=np.int64)
        label_counts = pd.Series(labels).value_counts().sort_index().to_dict()
        file_inspection.append(
            {
                "file": str(file_path),
                "subject": int(subject),
                "parsed_rows": int(len(labels)),
                "label_counts": {str(int(k)): int(v) for k, v in label_counts.items()},
            }
        )
        for value, label in zip(values, labels):
            if int(label) <= 0:
                continue
            labels_seen.add(int(label))
            raw_rows.append((subject, int(label), value.astype(np.float32, copy=False)))
    if not raw_rows:
        raise ValueError(f"No chest accelerometer rows parsed from {root}")
    subjects_seen = sorted({int(row[0]) for row in raw_rows})
    if len(subjects_seen) < 3:
        if inspection_dir is not None:
            write_chest_inspection(Path(inspection_dir), files, file_inspection, labels_seen, subjects_seen)
        raise ValueError("not enough subjects for subject-disjoint split: chest_accel requires at least 3 subjects")
    class_ids = sorted(labels_seen)
    label_map = {label: idx for idx, label in enumerate(class_ids)}
    class_names = tuple(str(label) for label in class_ids)
    x, y, subjects, groups = _window_chest_rows(raw_rows, label_map, window_size, stride)
    if inspection_dir is not None:
        write_chest_inspection(Path(inspection_dir), files, file_inspection, labels_seen, subjects_seen)
    return ExternalArrays(x, y, subjects, groups, class_names, EXTERNAL_CHANNELS_3)


def write_chest_inspection(
    output_dir: Path,
    files: list[Path],
    file_inspection: list[dict[str, Any]],
    labels_seen: set[int],
    subjects_seen: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_subject: dict[str, dict[str, int]] = {}
    for item in file_inspection:
        subject_key = str(item["subject"])
        by_subject.setdefault(subject_key, {})
        for label, count in item.get("label_counts", {}).items():
            by_subject[subject_key][label] = by_subject[subject_key].get(label, 0) + int(count)
    inspection = {
        "dataset_key": "chest_accel",
        "files": [str(path) for path in files],
        "inferred_subject_ids": subjects_seen,
        "num_subjects": int(len(subjects_seen)),
        "original_label_ids": sorted(int(label) for label in labels_seen),
        "label_counts_by_subject": by_subject,
        "file_inspection": file_inspection,
    }
    (output_dir / "inspection_chest_accel.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")


def find_chest_accel_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Chest accelerometer root not found: {root}")
    files = [path for path in root.rglob("*.csv") if path.is_file()]
    if not files:
        files = [path for path in root.rglob("*.txt") if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No chest accelerometer CSV/TXT files found under {root}")
    return sorted(files)


def _subject_from_path(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if digits:
        return int(digits)
    return abs(hash(path.stem)) % 10_000


def _window_chest_rows(
    rows: list[tuple[int, int, np.ndarray]],
    label_map: dict[int, int],
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    group_ids: list[int] = []
    group_counter = 0
    start = 0
    while start < len(rows):
        subject, label, _ = rows[start]
        end = start + 1
        while end < len(rows) and rows[end][0] == subject and rows[end][1] == label:
            end += 1
        values = np.stack([row[2] for row in rows[start:end]]).astype(np.float32, copy=False)
        if len(values) >= window_size:
            group_counter += 1
            mapped = label_map[label]
            for offset in range(0, len(values) - window_size + 1, stride):
                xs.append(values[offset : offset + window_size])
                ys.append(mapped)
                subjects.append(subject)
                group_ids.append(group_counter)
        start = end
    if not xs:
        raise ValueError("No chest accelerometer windows generated; check window_size/stride and labels")
    return (
        np.stack(xs).astype(np.float32, copy=False),
        np.asarray(ys, dtype=np.int64),
        np.asarray(subjects, dtype=np.int64),
        np.asarray(group_ids, dtype=np.int64),
    )


def load_hugadb_windows(
    root: str | Path,
    window_size: int = 256,
    stride: int = 128,
    channel_set: str = "inertial36",
    inspection_dir: str | Path | None = None,
    exclude_known_corrupted_gyro: bool = True,
    dataset_key: str = "hugadb",
) -> ExternalArrays:
    root = Path(root)
    task_spec = hugadb_task_spec(dataset_key)
    keep_labels = set(int(label) for label in task_spec["labels"])
    binary_map = task_spec.get("binary_map")
    files = find_hugadb_files(root)
    all_values: list[np.ndarray] = []
    all_labels: list[int] = []
    all_original_window_labels: list[int] = []
    all_subjects: list[int] = []
    all_groups: list[int] = []
    file_summaries: list[dict[str, Any]] = []
    skipped_files: list[dict[str, str]] = []
    warnings: list[str] = []
    raw_rows = 0
    valid_rows = 0
    group_counter = 0
    labels_seen: set[int] = set()
    raw_labels_seen: set[int] = set()

    for file_path in files:
        try:
            subject_id, session_id = parse_hugadb_subject_session(file_path)
            numeric_rows, file_raw_rows, file_valid_rows = parse_hugadb_file(file_path)
        except Exception as exc:  # noqa: BLE001 - inspect and continue across files
            skipped_files.append({"file": str(file_path), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        raw_rows += file_raw_rows
        valid_rows += file_valid_rows
        if numeric_rows.size == 0:
            skipped_files.append({"file": str(file_path), "reason": "no valid 39-column numeric rows"})
            continue
        values, original_labels = select_hugadb_channels(numeric_rows, channel_set)
        label_row_counts = pd.Series(original_labels).value_counts().sort_index().to_dict()
        for label in label_row_counts:
            label_int = int(label)
            if label_int > 0:
                raw_labels_seen.add(label_int)
        start = 0
        file_windows = 0
        file_segments = 0
        while start < len(original_labels):
            label = int(original_labels[start])
            end = start + 1
            while end < len(original_labels) and int(original_labels[end]) == label:
                end += 1
            segment_values = values[start:end]
            if label in keep_labels and len(segment_values) >= window_size:
                labels_seen.add(label)
                group_counter += 1
                file_segments += 1
                mapped_label = int(binary_map[label]) if binary_map else hugadb_label_index(label, task_spec)
                for offset in range(0, len(segment_values) - window_size + 1, stride):
                    all_values.append(segment_values[offset : offset + window_size])
                    all_labels.append(mapped_label)
                    all_original_window_labels.append(label)
                    all_subjects.append(subject_id)
                    all_groups.append(group_counter)
                    file_windows += 1
            start = end
        file_summaries.append(
            {
                "file": str(file_path),
                "subject": int(subject_id),
                "session": int(session_id),
                "raw_rows": int(file_raw_rows),
                "valid_rows": int(file_valid_rows),
                "segments": int(file_segments),
                "windows": int(file_windows),
                "observed_label_ids": sorted(int(label) for label in set(original_labels.tolist()) if int(label) > 0),
                "label_row_counts": {str(int(label)): int(count) for label, count in label_row_counts.items() if int(label) > 0},
            }
        )

    if not all_values:
        raise ValueError(f"No HuGaDB windows generated under {root}; check files/window_size/stride")
    subjects_seen = sorted(set(int(subject) for subject in all_subjects))
    if len(subjects_seen) < 5:
        if inspection_dir is not None:
            write_hugadb_inspection(
                Path(inspection_dir),
                dataset_key,
                root,
                files,
                raw_rows,
                valid_rows,
                subjects_seen,
                sorted(labels_seen),
                np.asarray(all_original_window_labels, dtype=np.int64),
                file_summaries,
                skipped_files,
                warnings,
                channel_set,
                window_size,
                stride,
                exclude_known_corrupted_gyro,
                task_spec,
                dataset_key,
            )
        raise ValueError("HuGaDB requires at least five subjects for subject-disjoint screening")

    original_label_ids = sorted(labels_seen)
    mapped_labels = np.asarray(all_labels, dtype=np.int64)
    class_names = tuple(task_spec["class_names"])
    x = np.stack(all_values).astype(np.float32, copy=False)
    y = mapped_labels
    subjects = np.asarray(all_subjects, dtype=np.int64)
    groups = np.asarray(all_groups, dtype=np.int64)
    if inspection_dir is not None:
        write_hugadb_inspection(
            Path(inspection_dir),
            dataset_key,
            root,
            files,
            raw_rows,
            valid_rows,
            subjects_seen,
            original_label_ids,
            np.asarray(all_original_window_labels, dtype=np.int64),
            file_summaries,
            skipped_files,
            warnings,
            channel_set,
            window_size,
            stride,
            exclude_known_corrupted_gyro,
            task_spec,
            dataset_key,
        )
    return ExternalArrays(x, y, subjects, groups, class_names, hugadb_channel_names(channel_set))


def hugadb_task_spec(dataset_key: str) -> dict[str, Any]:
    key = dataset_key.lower()
    if key not in HUGADB_TASKS:
        raise ValueError(f"Unsupported HuGaDB task: {dataset_key}")
    return HUGADB_TASKS[key]


def hugadb_label_index(label: int, task_spec: dict[str, Any]) -> int:
    labels = tuple(int(value) for value in task_spec["labels"])
    return labels.index(int(label))


def find_hugadb_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"HuGaDB root not found: {root}")
    files = [
        path
        for pattern in ("*.txt", "*.csv", "*.dat")
        for path in root.rglob(pattern)
        if path.is_file()
        and "readme" not in path.name.lower()
        and "description" not in path.name.lower()
        and path.stat().st_size > 0
    ]
    if not files:
        raise FileNotFoundError(f"No HuGaDB TXT/CSV/DAT files found under {root}")
    return sorted(files)


def parse_hugadb_subject_session(path: Path) -> tuple[int, int]:
    tokens = re.findall(r"\d+", path.stem)
    numbers = [int(token) for token in tokens]
    if len(numbers) >= 2:
        return numbers[-2], numbers[-1]
    if len(numbers) == 1:
        return numbers[0], 0
    raise ValueError(f"Cannot infer HuGaDB participant/session from filename: {path.name}")


def parse_hugadb_file(path: Path) -> tuple[np.ndarray, int, int]:
    rows: list[np.ndarray] = []
    raw_rows = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            raw_rows += 1
            values = parse_numeric_row(line)
            if values is None or len(values) < 39:
                continue
            values = values[-39:]
            label = int(round(values[-1]))
            if label <= 0:
                continue
            if not np.isfinite(values).all():
                continue
            rows.append(values.astype(np.float32, copy=False))
    if not rows:
        return np.empty((0, 39), dtype=np.float32), raw_rows, 0
    frame = np.stack(rows).astype(np.float32, copy=False)
    return frame, raw_rows, int(len(frame))


def parse_numeric_row(line: str) -> np.ndarray | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = [part for part in re.split(r"[\s,;]+", stripped) if part]
    values: list[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            return None
    return np.asarray(values, dtype=np.float32)


def select_hugadb_channels(rows: np.ndarray, channel_set: str) -> tuple[np.ndarray, np.ndarray]:
    channel_set = channel_set.lower()
    if channel_set == "inertial36":
        values = rows[:, :36]
    elif channel_set == "inertial36_emg2":
        values = rows[:, :38]
    else:
        raise ValueError(f"Unsupported HuGaDB channel_set: {channel_set}")
    labels = rows[:, -1].round().astype(np.int64)
    return values.astype(np.float32, copy=False), labels


def hugadb_channel_names(channel_set: str) -> tuple[str, ...]:
    names = tuple(f"inertial_{idx:02d}" for idx in range(36))
    if channel_set.lower() == "inertial36_emg2":
        return names + ("emg_00", "emg_01")
    return names


def write_hugadb_inspection(
    output_dir: Path,
    dataset_key: str,
    root: Path,
    files: list[Path],
    num_raw_rows: int,
    num_valid_rows: int,
    subjects_seen: list[int],
    original_label_ids: list[int],
    original_window_labels: np.ndarray,
    file_summaries: list[dict[str, Any]],
    skipped_files: list[dict[str, str]],
    warnings: list[str],
    channel_set: str,
    window_size: int,
    stride: int,
    exclude_known_corrupted_gyro: bool,
    task_spec: dict[str, Any] | None = None,
    output_dataset_key: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_spec = task_spec or hugadb_task_spec(dataset_key)
    output_dataset_key = output_dataset_key or dataset_key
    observed_raw_labels = sorted(
        {
            int(label)
            for item in file_summaries
            for label in item.get("label_row_counts", {}).keys()
            if int(label) > 0
        }
    )
    selected_labels = [int(label) for label in task_spec["labels"]]
    dropped_labels = [label for label in observed_raw_labels if label not in selected_labels]
    windows_per_class = {
        HUGADB_LABELS.get(label, f"activity_{label}"): int((original_window_labels == label).sum())
        for label in original_label_ids
    }
    window_counts = list(windows_per_class.values())
    inspection = {
        "dataset_key": output_dataset_key,
        "parent_dataset": "hugadb",
        "task_name": output_dataset_key,
        "root": str(root),
        "num_files_found": int(len(files)),
        "num_files_parsed": int(len(file_summaries)),
        "num_raw_rows": int(num_raw_rows),
        "num_valid_rows": int(num_valid_rows),
        "subjects": [int(subject) for subject in subjects_seen],
        "num_subjects": int(len(subjects_seen)),
        "observed_original_label_ids": observed_raw_labels,
        "original_label_ids": [int(label) for label in original_label_ids],
        "remapped_class_names": list(task_spec["class_names"]),
        "dropped_label_ids": dropped_labels,
        "dropped_class_names": [HUGADB_LABELS.get(label, f"activity_{label}") for label in dropped_labels],
        "binary_map": {str(int(k)): int(v) for k, v in task_spec.get("binary_map", {}).items()},
        "class_counts_raw": aggregate_hugadb_raw_counts(file_summaries),
        "num_segments": int(sum(int(item.get("segments", 0)) for item in file_summaries)),
        "windows_per_class": windows_per_class,
        "min_windows_per_class": int(min(window_counts)) if window_counts else 0,
        "max_windows_per_class": int(max(window_counts)) if window_counts else 0,
        "channel_set": channel_set,
        "window_size": int(window_size),
        "stride": int(stride),
        "exclude_known_corrupted_gyro": bool(exclude_known_corrupted_gyro),
        "suspected_corrupted_files": [],
        "skipped_files": skipped_files,
        "file_summaries": file_summaries,
        "warnings": warnings,
    }
    (output_dir / f"inspection_{output_dataset_key}.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")


def aggregate_hugadb_raw_counts(file_summaries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in file_summaries:
        label_counts = item.get("label_row_counts", {})
        if label_counts:
            for label, count in label_counts.items():
                name = HUGADB_LABELS.get(int(label), f"activity_{int(label)}")
                counts[name] = counts.get(name, 0) + int(count)
            continue
        for label in item.get("observed_label_ids", []):
            name = HUGADB_LABELS.get(int(label), f"activity_{int(label)}")
            counts[name] = counts.get(name, 0) + 1
    return counts


def load_opportunity_windows(
    root: str | Path,
    dataset_key: str,
    task: str = "locomotion",
    protocol_type: str = "subject_disjoint",
    feature_set: str = "body_worn",
    window_size: int = 64,
    stride: int = 32,
    inspection_dir: str | Path | None = None,
    drop_feature_missing_threshold: float = 0.8,
) -> ExternalArrays:
    if task != "locomotion":
        raise ValueError("Only OPPORTUNITY locomotion is implemented in this phase")
    root = Path(root)
    files = find_opportunity_files(root)
    if not files:
        raise FileNotFoundError(f"No OPPORTUNITY .dat/.csv files found under {root}")
    parsed_files: list[dict[str, Any]] = []
    skipped_files: list[dict[str, str]] = []
    warnings: list[str] = []
    all_values: list[np.ndarray] = []
    all_original_labels: list[int] = []
    all_subjects: list[int] = []
    all_runs: list[int] = []
    all_groups: list[int] = []
    total_rows = 0
    total_valid_rows = 0
    group_counter = 0
    column_map: dict[str, Any] | None = build_opportunity_column_map_from_metadata(root, feature_set=feature_set, task=task)
    read_usecols = column_map.get("_read_usecols") if column_map else None
    metadata_feature_count = int(column_map.get("_selected_feature_count", 0)) if column_map else 0

    for file_path in files:
        try:
            subject_id, run_id, run_name = parse_opportunity_subject_run(file_path)
            matrix = read_opportunity_numeric_file(file_path, usecols=read_usecols)
        except Exception as exc:  # noqa: BLE001
            skipped_files.append({"file": str(file_path), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        total_rows += int(matrix.shape[0])
        if matrix.size == 0 or matrix.shape[1] < 8:
            skipped_files.append({"file": str(file_path), "reason": "empty or too few numeric columns"})
            continue
        if metadata_feature_count > 0:
            file_column_map = column_map
            feature_columns = list(range(metadata_feature_count))
            label_column = metadata_feature_count
        else:
            try:
                file_column_map = build_opportunity_column_map(matrix, feature_set=feature_set, task=task)
            except ValueError:
                if column_map is None:
                    raise
                file_column_map = column_map
            if column_map is None:
                column_map = file_column_map
            feature_columns = file_column_map["feature_columns"]
            label_column = int(file_column_map["label_columns"]["locomotion"])
        values = matrix[:, feature_columns].astype(np.float32, copy=False)
        raw_labels = matrix[:, label_column]
        valid_mask = np.isfinite(raw_labels) & (np.rint(raw_labels).astype(np.int64) > 0)
        valid_mask &= np.isfinite(values).any(axis=1)
        if not valid_mask.any():
            skipped_files.append({"file": str(file_path), "reason": "no valid locomotion labels"})
            continue
        values = values[valid_mask]
        labels = np.rint(raw_labels[valid_mask]).astype(np.int64)
        total_valid_rows += int(len(labels))
        label_counts = pd.Series(labels).value_counts().sort_index().to_dict()
        start = 0
        file_windows = 0
        file_segments = 0
        while start < len(labels):
            label = int(labels[start])
            end = start + 1
            while end < len(labels) and int(labels[end]) == label:
                end += 1
            segment_values = values[start:end]
            if len(segment_values) >= window_size:
                group_counter += 1
                file_segments += 1
                for offset in range(0, len(segment_values) - window_size + 1, stride):
                    all_values.append(segment_values[offset : offset + window_size])
                    all_original_labels.append(label)
                    all_subjects.append(subject_id)
                    all_runs.append(run_id)
                    all_groups.append(group_counter)
                    file_windows += 1
            start = end
        parsed_files.append(
            {
                "file": str(file_path),
                "subject": int(subject_id),
                "run": int(run_id),
                "run_name": run_name,
                "rows": int(matrix.shape[0]),
                "valid_locomotion_rows": int(len(labels)),
                "segments": int(file_segments),
                "windows": int(file_windows),
                "label_row_counts": {str(int(k)): int(v) for k, v in label_counts.items()},
            }
        )

    if not all_values:
        if inspection_dir is not None:
            write_opportunity_inspection(
                Path(inspection_dir),
                root,
                dataset_key,
                protocol_type,
                feature_set,
                files,
                parsed_files,
                skipped_files,
                warnings + ["no windows generated"],
                column_map,
                total_rows,
                total_valid_rows,
                (),
                (),
                window_size,
                stride,
                drop_feature_missing_threshold,
            )
        raise ValueError("No OPPORTUNITY windows generated; inspect labels/window_size/protocol")

    x_raw = np.stack(all_values).astype(np.float32, copy=False)
    missing_fraction = np.isnan(x_raw).mean(axis=(0, 1))
    missing_fraction_total = float(np.isnan(x_raw).mean())
    keep_features = missing_fraction <= float(drop_feature_missing_threshold)
    if not keep_features.any():
        raise ValueError("All OPPORTUNITY feature columns exceed missing-value threshold")
    if not np.all(keep_features):
        warnings.append(f"dropped {int((~keep_features).sum())} high-missing feature columns")
    x = x_raw[:, :, keep_features].astype(np.float32, copy=False)
    original_label_ids = sorted(set(int(label) for label in all_original_labels))
    label_map = {label: idx for idx, label in enumerate(original_label_ids)}
    y = np.asarray([label_map[int(label)] for label in all_original_labels], dtype=np.int64)
    class_names = tuple(opportunity_locomotion_name(label) for label in original_label_ids)
    subjects = np.asarray(all_subjects, dtype=np.int64)
    runs = np.asarray(all_runs, dtype=np.int64)
    groups = np.asarray(all_groups, dtype=np.int64)
    channel_names = tuple(f"opp_ch_{idx:03d}" for idx in range(x.shape[-1]))
    if inspection_dir is not None:
        write_opportunity_inspection(
            Path(inspection_dir),
            root,
            dataset_key,
            protocol_type,
            feature_set,
            files,
            parsed_files,
            skipped_files,
            warnings,
            column_map,
            total_rows,
            total_valid_rows,
            original_label_ids,
            class_names,
            window_size,
            stride,
            drop_feature_missing_threshold,
            kept_feature_count=int(x.shape[-1]),
            total_feature_count=int(x_raw.shape[-1]),
            missing_value_fraction_total=missing_fraction_total,
            missing_value_fraction_by_sensor_group={
                "selected_features_before_drop_mean": float(missing_fraction.mean()),
                "selected_features_before_drop_max": float(missing_fraction.max()),
            },
        )
        write_opportunity_column_map(Path(inspection_dir), column_map, feature_count=int(x.shape[-1]))
    return ExternalArrays(x, y, subjects, groups, class_names, channel_names, runs=runs)


def find_opportunity_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"OPPORTUNITY root not found: {root}")
    preferred = [
        path
        for path in root.rglob("*.dat")
        if path.is_file() and re.search(r"S\d+-(ADL\d+|Drill)", path.name, flags=re.IGNORECASE)
    ]
    if preferred:
        return sorted(preferred)
    files = [
        path
        for pattern in ("*.dat", "*.data", "*.csv", "*.txt")
        for path in root.rglob(pattern)
        if path.is_file() and "readme" not in path.name.lower()
    ]
    return sorted(files)


def parse_opportunity_subject_run(path: Path) -> tuple[int, int, str]:
    text = path.stem
    subject_match = re.search(r"S(\d+)", text, flags=re.IGNORECASE)
    subject = int(subject_match.group(1)) if subject_match else 0
    run_match = re.search(r"ADL(\d+)", text, flags=re.IGNORECASE)
    if run_match:
        run = int(run_match.group(1))
        return subject, run, f"ADL{run}"
    if re.search(r"Drill", text, flags=re.IGNORECASE):
        return subject, 0, "Drill"
    numbers = [int(token) for token in re.findall(r"\d+", text)]
    run = numbers[-1] if numbers else -1
    return subject, run, f"run_{run}"


def read_opportunity_numeric_file(path: Path, usecols: list[int] | None = None) -> np.ndarray:
    try:
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            engine="c",
            na_values=["NaN", "nan", "?"],
            usecols=usecols,
        )
    except Exception:
        frame = pd.read_csv(path, header=None, na_values=["NaN", "nan", "?"], usecols=usecols)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.to_numpy(dtype=np.float32)


def build_opportunity_column_map_from_metadata(
    root: Path,
    feature_set: str = "body_worn",
    task: str = "locomotion",
) -> dict[str, Any] | None:
    column_file = next((path for path in root.rglob("column_names.txt") if path.is_file()), None)
    if column_file is None:
        return None
    entries: list[tuple[int, str]] = []
    for line in column_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"\s*Column:\s*(\d+)\s+(.+)$", line)
        if not match:
            continue
        entries.append((int(match.group(1)) - 1, match.group(2).strip()))
    if not entries:
        return None
    locomotion = next((idx for idx, name in entries if name.lower() == "locomotion"), None)
    if locomotion is None:
        return None
    label_indices = [idx for idx, name in entries if idx >= locomotion]
    if feature_set == "body_worn":
        source_features = [
            idx
            for idx, name in entries
            if 0 < idx < locomotion
            and (
                name.startswith("Accelerometer")
                or name.startswith("InertialMeasurementUnit")
            )
            and not any(
                token in name
                for token in (
                    "CUP",
                    "SALAMI",
                    "WATER",
                    "CHEESE",
                    "BREAD",
                    "KNIFE",
                    "MILK",
                    "SPOON",
                    "SUGAR",
                    "PLATE",
                    "GLASS",
                    "DOOR",
                    "DISHWASHER",
                    "DRAWER",
                    "FRIDGE",
                    "LAZYCHAIR",
                )
            )
        ]
    else:
        source_features = [idx for idx, _ in entries if 0 < idx < locomotion]
    if not source_features:
        source_features = [idx for idx, _ in entries if 0 < idx < locomotion]
    read_usecols = [*source_features, locomotion]
    selected_names = {idx: name for idx, name in entries}
    feature_columns = list(range(len(source_features)))
    return {
        "time_column": 0,
        "subject_column": None,
        "run_column": None,
        "feature_set": feature_set if feature_set == "body_worn" else "all_numeric_sensors",
        "feature_columns": feature_columns,
        "feature_column_count": len(feature_columns),
        "source_feature_columns": source_features,
        "source_feature_names": [selected_names.get(idx, f"Column {idx + 1}") for idx in source_features],
        "sensor_groups": {
            "body_worn": feature_columns if feature_set == "body_worn" else [],
            "object": [],
            "ambient": [],
            "unknown": [] if feature_set == "body_worn" else feature_columns,
        },
        "label_columns": {"locomotion": len(source_features)},
        "source_label_columns": {"locomotion": locomotion},
        "all_source_label_columns": label_indices,
        "label_candidates": [{"index": len(source_features), "source_index": locomotion, "track": "locomotion"}],
        "ignored_columns": label_indices,
        "detection_confidence": "official_column_names",
        "task": task,
        "_read_usecols": read_usecols,
        "_selected_feature_count": len(source_features),
    }


def build_opportunity_column_map(matrix: np.ndarray, feature_set: str = "body_worn", task: str = "locomotion") -> dict[str, Any]:
    label_candidates = opportunity_label_candidates(matrix)
    if not label_candidates:
        raise ValueError("Could not detect OPPORTUNITY label columns")
    locomotion = choose_locomotion_label_candidate(label_candidates)
    label_columns = [candidate["index"] for candidate in label_candidates]
    first_label = min(label_columns)
    feature_columns = [idx for idx in range(first_label) if idx != 0]
    if not feature_columns:
        feature_columns = [idx for idx in range(matrix.shape[1]) if idx not in set(label_columns)]
    return {
        "time_column": 0,
        "subject_column": None,
        "run_column": None,
        "feature_set": feature_set if feature_set == "body_worn" else "all_numeric_sensors",
        "feature_columns": feature_columns,
        "feature_column_count": len(feature_columns),
        "sensor_groups": {
            "body_worn": feature_columns,
            "object": [],
            "ambient": [],
            "unknown": [],
        },
        "label_columns": {
            "locomotion": int(locomotion["index"]),
        },
        "label_candidates": label_candidates,
        "ignored_columns": label_columns,
        "detection_confidence": "heuristic_low_cardinality_tail_columns",
        "task": task,
    }


def opportunity_label_candidates(matrix: np.ndarray) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    start = max(1, matrix.shape[1] - 24)
    for idx in range(start, matrix.shape[1]):
        column = matrix[:, idx]
        finite = column[np.isfinite(column)]
        if len(finite) == 0:
            continue
        rounded = np.rint(finite)
        if not np.allclose(finite, rounded, atol=1e-4):
            continue
        unique = sorted(int(value) for value in set(rounded.astype(np.int64).tolist()))
        valid = [value for value in unique if value > 0]
        if 2 <= len(valid) <= 12 and len(unique) <= 16:
            candidates.append(
                {
                    "index": int(idx),
                    "unique_values": unique,
                    "valid_values": valid,
                    "valid_fraction": float((rounded > 0).mean()),
                    "num_valid_classes": int(len(valid)),
                }
            )
    return candidates


def choose_locomotion_label_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    preferred_values = {1, 2, 4, 5}
    def score(candidate: dict[str, Any]) -> tuple[int, float, int]:
        values = set(int(value) for value in candidate["valid_values"])
        overlap = len(values & preferred_values)
        valid_fraction = float(candidate["valid_fraction"])
        # Prefer low-cardinality locomotion-like labels over richer gesture/action tracks.
        return overlap, valid_fraction, -abs(int(candidate["num_valid_classes"]) - 4)

    return max(candidates, key=score)


def opportunity_locomotion_name(label: int) -> str:
    names = {
        1: "Stand",
        2: "Walk",
        4: "Sit",
        5: "Lie",
    }
    return names.get(int(label), f"locomotion_{int(label)}")


def write_opportunity_inspection(
    output_dir: Path,
    root: Path,
    dataset_key: str,
    protocol_type: str,
    feature_set: str,
    files: list[Path],
    parsed_files: list[dict[str, Any]],
    skipped_files: list[dict[str, str]],
    warnings: list[str],
    column_map: dict[str, Any] | None,
    num_rows: int,
    num_valid_rows: int,
    original_label_ids: Iterable[int],
    class_names: Iterable[str],
    window_size: int,
    stride: int,
    drop_feature_missing_threshold: float,
    kept_feature_count: int | None = None,
    total_feature_count: int | None = None,
    missing_value_fraction_total: float | None = None,
    missing_value_fraction_by_sensor_group: dict[str, float] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_ids = sorted({int(item["subject"]) for item in parsed_files})
    run_ids = sorted({int(item["run"]) for item in parsed_files})
    label_track_counts: dict[str, int] = {}
    for item in parsed_files:
        for label, count in item.get("label_row_counts", {}).items():
            label_track_counts[label] = label_track_counts.get(label, 0) + int(count)
    inspection = {
        "dataset_key": dataset_key,
        "parent_dataset": "opportunity",
        "data_source": "archive_or_manual_files",
        "root": str(root),
        "files_found": [str(path) for path in files[:200]],
        "num_files_found": len(files),
        "num_files_parsed": len(parsed_files),
        "num_rows": int(num_rows),
        "num_valid_locomotion_rows": int(num_valid_rows),
        "num_columns": int((column_map or {}).get("feature_column_count", 0)) if column_map else 0,
        "missing_value_fraction_total": missing_value_fraction_total,
        "missing_value_fraction_by_sensor_group": missing_value_fraction_by_sensor_group or {},
        "detected_subjects": subject_ids,
        "detected_runs": run_ids,
        "detected_label_tracks": list((column_map or {}).get("label_columns", {}).keys()),
        "label_track_counts": label_track_counts,
        "candidate_tasks": ["opportunity_locomotion"],
        "task": "locomotion",
        "protocol_type": protocol_type,
        "feature_set": feature_set,
        "original_label_ids": [int(label) for label in original_label_ids],
        "class_names": list(class_names),
        "window_size": int(window_size),
        "stride": int(stride),
        "drop_feature_missing_threshold": float(drop_feature_missing_threshold),
        "kept_feature_count": kept_feature_count,
        "total_feature_count": total_feature_count,
        "skipped_files": skipped_files,
        "warnings": warnings,
    }
    (output_dir / f"inspection_{dataset_key}.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")


def write_opportunity_column_map(output_dir: Path, column_map: dict[str, Any] | None, feature_count: int) -> None:
    if column_map is None:
        return
    serializable = dict(column_map)
    serializable["selected_feature_count_after_missing_drop"] = int(feature_count)
    (output_dir / "opportunity_column_map.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_capture24_windows(
    root: str | Path,
    dataset_key: str = "capture24_movement4",
    window_size: int = 1000,
    stride: int = 500,
    inspection_dir: str | Path | None = None,
    label_map_path: str | Path | None = None,
    max_files: int | None = None,
    downsample_bins: int | None = None,
    max_windows_per_subject_class: int | None = None,
    use_cache: bool = True,
) -> ExternalArrays:
    root = Path(root)
    cache_path = capture24_cache_path(root, dataset_key, window_size, stride, downsample_bins, max_windows_per_subject_class, max_files)
    if use_cache and cache_path.exists():
        arrays = read_capture24_cache(cache_path)
        if inspection_dir is not None and not (Path(inspection_dir) / "capture24_inspection.json").exists():
            write_capture24_inspection(
                Path(inspection_dir),
                dataset_key,
                root,
                find_capture24_files(root)[: max_files if max_files is not None else None],
                [],
                [],
                0,
                0,
                {label: 0 for label in CAPTURE24_MOVEMENT4_LABELS},
                window_size,
                stride,
                downsample_bins=int(downsample_bins) if downsample_bins is not None else 0,
                max_windows_per_subject_class=int(max_windows_per_subject_class) if max_windows_per_subject_class is not None else 0,
                windows_per_class={
                    CAPTURE24_MOVEMENT4_LABELS[idx]: int((arrays.y == idx).sum())
                    for idx in range(len(CAPTURE24_MOVEMENT4_LABELS))
                },
                subjects=sorted(int(subject) for subject in np.unique(arrays.subjects)),
            )
        return arrays
    files = find_capture24_files(root)
    if not files:
        raise FileNotFoundError(
            f"No Capture-24 participant CSV files found under {root}. "
            "Run the downloader and place the extracted dataset there first."
        )
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    label_map = load_capture24_label_map(label_map_path)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    groups: list[int] = []
    file_summaries: list[dict[str, Any]] = []
    skipped_files: list[dict[str, str]] = []
    group_counter = 0
    num_raw_rows = 0
    num_valid_rows = 0
    class_counts_raw = {label: 0 for label in CAPTURE24_MOVEMENT4_LABELS}
    cap_counts: dict[tuple[int, int], int] = {}
    downsample_bins_int = int(downsample_bins) if downsample_bins is not None else 0
    cap_per_subject_class = int(max_windows_per_subject_class) if max_windows_per_subject_class is not None else 0

    for file_path in files:
        try:
            subject = parse_capture24_subject(file_path)
            frame = read_capture24_frame(file_path)
            num_raw_rows += int(len(frame))
            columns = detect_capture24_columns(frame)
            values = frame[columns["acc_cols"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            labels_raw = frame[columns["label_col"]].astype(str).to_numpy()
            order = np.arange(len(frame))
            if columns.get("time_col"):
                order = pd.to_numeric(frame[columns["time_col"]], errors="coerce").ffill().fillna(0).to_numpy()
                sort_idx = np.argsort(order, kind="stable")
                values = values[sort_idx]
                labels_raw = labels_raw[sort_idx]
            mapped = np.asarray([map_capture24_label(label, label_map) for label in labels_raw], dtype=object)
            valid = np.asarray([label is not None for label in mapped], dtype=bool)
            finite = np.isfinite(values).all(axis=1)
            keep = valid & finite
            values = values[keep]
            mapped_labels = np.asarray([label for label in mapped[keep]], dtype=object)
            num_valid_rows += int(len(values))
            if len(values) < int(window_size):
                skipped_files.append({"file": str(file_path), "reason": "not enough valid rows"})
                continue
            label_ids = np.asarray([CAPTURE24_MOVEMENT4_LABELS.index(str(label)) for label in mapped_labels], dtype=np.int64)
            for label_name in mapped_labels:
                class_counts_raw[str(label_name)] = class_counts_raw.get(str(label_name), 0) + 1
            file_windows = 0
            for start, end in contiguous_label_segments(label_ids):
                segment_values = values[start:end]
                segment_label = int(label_ids[start])
                if len(segment_values) < int(window_size):
                    continue
                group_counter += 1
                for window_start in range(0, len(segment_values) - int(window_size) + 1, int(stride)):
                    window = segment_values[window_start : window_start + int(window_size)]
                    cap_key = (int(subject), int(segment_label))
                    if cap_per_subject_class > 0 and cap_counts.get(cap_key, 0) >= cap_per_subject_class:
                        continue
                    if downsample_bins_int > 0:
                        window = downsample_capture24_window(window, downsample_bins_int)
                    xs.append(window.astype(np.float32, copy=False))
                    ys.append(segment_label)
                    subjects.append(subject)
                    groups.append(group_counter)
                    cap_counts[cap_key] = cap_counts.get(cap_key, 0) + 1
                    file_windows += 1
            file_summaries.append(
                {
                    "file": str(file_path.relative_to(root) if file_path.is_relative_to(root) else file_path),
                    "subject": int(subject),
                    "rows": int(len(frame)),
                    "valid_rows": int(len(values)),
                    "windows": int(file_windows),
                }
            )
        except Exception as exc:  # noqa: BLE001 - inspection should preserve parser failures
            skipped_files.append({"file": str(file_path), "reason": str(exc)})

    if not xs:
        if inspection_dir is not None:
            write_capture24_inspection(
                Path(inspection_dir),
                dataset_key,
                root,
                files,
                file_summaries,
                skipped_files,
                num_raw_rows,
                num_valid_rows,
                class_counts_raw,
                window_size,
                stride,
                downsample_bins=downsample_bins_int,
                max_windows_per_subject_class=cap_per_subject_class,
            )
        raise ValueError("Capture-24 parser produced no windows; inspect capture24_inspection.json")

    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    subject_arr = np.asarray(subjects, dtype=np.int64)
    group_arr = np.asarray(groups, dtype=np.int64)
    if inspection_dir is not None:
        write_capture24_inspection(
            Path(inspection_dir),
            dataset_key,
            root,
            files,
            file_summaries,
            skipped_files,
            num_raw_rows,
            num_valid_rows,
            class_counts_raw,
            window_size,
            stride,
            downsample_bins=downsample_bins_int,
            max_windows_per_subject_class=cap_per_subject_class,
            windows_per_class={
                CAPTURE24_MOVEMENT4_LABELS[idx]: int((y == idx).sum())
                for idx in range(len(CAPTURE24_MOVEMENT4_LABELS))
            },
            subjects=sorted(int(subject) for subject in np.unique(subject_arr)),
        )
    arrays = ExternalArrays(
        x,
        y,
        subject_arr,
        group_arr,
        CAPTURE24_MOVEMENT4_LABELS,
        ("acc_x", "acc_y", "acc_z"),
    )
    if use_cache:
        write_capture24_cache(cache_path, arrays)
    return arrays


def capture24_cache_path(
    root: Path,
    dataset_key: str,
    window_size: int,
    stride: int,
    downsample_bins: int | None,
    max_windows_per_subject_class: int | None,
    max_files: int | None,
) -> Path:
    cache_dir = root / ".cache"
    suffix = (
        f"{dataset_key}_w{int(window_size)}_s{int(stride)}_"
        f"bins{int(downsample_bins) if downsample_bins is not None else 0}_"
        f"cap{int(max_windows_per_subject_class) if max_windows_per_subject_class is not None else 0}_"
        f"files{int(max_files) if max_files is not None else 0}.npz"
    )
    return cache_dir / suffix


def write_capture24_cache(path: Path, arrays: ExternalArrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=arrays.x,
        y=arrays.y,
        subjects=arrays.subjects,
        group_ids=arrays.group_ids,
    )


def read_capture24_cache(path: Path) -> ExternalArrays:
    loaded = np.load(path)
    return ExternalArrays(
        loaded["x"].astype(np.float32, copy=False),
        loaded["y"].astype(np.int64, copy=False),
        loaded["subjects"].astype(np.int64, copy=False),
        loaded["group_ids"].astype(np.int64, copy=False),
        CAPTURE24_MOVEMENT4_LABELS,
        ("acc_x", "acc_y", "acc_z"),
    )


def find_capture24_files(root: Path) -> list[Path]:
    patterns = ("*.csv", "*.csv.gz", "*.parquet")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    ignored = ("annotation", "dictionary", "metadata", "readme", "label_map", "support_audit", "inspection")
    return sorted(
        path
        for path in files
        if not any(token in path.name.lower() for token in ignored)
    )


def parse_capture24_subject(path: Path) -> int:
    match = re.search(r"(?:^|[_\-\/\\])P?(\d{1,4})(?:[_\-.]|$)", str(path), flags=re.IGNORECASE)
    if not match:
        match = re.search(r"P(\d{1,4})", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not infer Capture-24 participant ID from {path.name}")
    return int(match.group(1))


def read_capture24_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def detect_capture24_columns(frame: pd.DataFrame) -> dict[str, Any]:
    lowered = {str(column).lower(): column for column in frame.columns}
    acc_candidates = [
        ("x", "y", "z"),
        ("acc_x", "acc_y", "acc_z"),
        ("accelerometer_x", "accelerometer_y", "accelerometer_z"),
        ("enmo_x", "enmo_y", "enmo_z"),
    ]
    acc_cols: tuple[Any, Any, Any] | None = None
    for names in acc_candidates:
        if all(name in lowered for name in names):
            acc_cols = tuple(lowered[name] for name in names)
            break
    if acc_cols is None:
        x_col = first_column_containing(frame, ("acc_x", "x"))
        y_col = first_column_containing(frame, ("acc_y", "y"))
        z_col = first_column_containing(frame, ("acc_z", "z"))
        if x_col is not None and y_col is not None and z_col is not None:
            acc_cols = (x_col, y_col, z_col)
    if acc_cols is None:
        raise ValueError("Could not detect Capture-24 x/y/z accelerometer columns")
    label_col = first_column_containing(frame, ("annotation", "label", "activity", "class"))
    if label_col is None:
        raise ValueError("Could not detect Capture-24 label/annotation column")
    time_col = first_column_containing(frame, ("time", "timestamp", "datetime"))
    return {"acc_cols": list(acc_cols), "label_col": label_col, "time_col": time_col}


def first_column_containing(frame: pd.DataFrame, candidates: tuple[str, ...]) -> Any | None:
    for candidate in candidates:
        for column in frame.columns:
            lowered = str(column).lower()
            if lowered == candidate or candidate in lowered:
                return column
    return None


def load_capture24_label_map(label_map_path: str | Path | None) -> dict[str, str]:
    if not label_map_path:
        return {}
    path = Path(label_map_path)
    if not path.exists():
        raise FileNotFoundError(f"Capture-24 label map not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else None
    if not isinstance(data, dict):
        raise ValueError("Capture-24 label map must be a JSON object mapping raw labels to movement4 labels")
    allowed = set(CAPTURE24_MOVEMENT4_LABELS)
    mapping = {str(raw).lower(): str(mapped) for raw, mapped in data.items()}
    invalid = sorted(set(mapping.values()) - allowed)
    if invalid:
        raise ValueError(f"Invalid Capture-24 movement4 labels in map: {invalid}")
    return mapping


def map_capture24_label(raw_label: str, explicit_map: dict[str, str]) -> str | None:
    normalized = str(raw_label).strip().lower()
    if not normalized or normalized in {"nan", "none", "null", "unknown", "other"}:
        return None
    if normalized in explicit_map:
        return explicit_map[normalized]
    if normalized in CAPTURE24_MOVEMENT4_LABELS:
        return normalized
    if normalized in {"0", "1", "2", "3"}:
        return CAPTURE24_MOVEMENT4_LABELS[int(normalized)]
    for target, pattern in CAPTURE24_MOVEMENT4_PATTERNS:
        if re.search(pattern, normalized):
            return target
    return None


def contiguous_label_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    if len(labels) == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(labels)):
        if int(labels[idx]) != int(labels[start]):
            segments.append((start, idx))
            start = idx
    segments.append((start, len(labels)))
    return segments


def downsample_capture24_window(window: np.ndarray, bins: int) -> np.ndarray:
    if bins < 1:
        return window
    if len(window) < bins:
        raise ValueError("downsample bins cannot exceed raw window length")
    chunks = np.array_split(window, bins, axis=0)
    return np.stack([chunk.mean(axis=0) for chunk in chunks], axis=0).astype(np.float32)


def write_capture24_inspection(
    output_dir: Path,
    dataset_key: str,
    root: Path,
    files: list[Path],
    file_summaries: list[dict[str, Any]],
    skipped_files: list[dict[str, str]],
    num_raw_rows: int,
    num_valid_rows: int,
    class_counts_raw: dict[str, int],
    window_size: int,
    stride: int,
    downsample_bins: int = 0,
    max_windows_per_subject_class: int = 0,
    windows_per_class: dict[str, int] | None = None,
    subjects: list[int] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    inspection = {
        "dataset_key": dataset_key,
        "parent_dataset": "capture24",
        "root": str(root),
        "num_files_found": int(len(files)),
        "num_files_parsed": int(len(file_summaries)),
        "num_raw_rows": int(num_raw_rows),
        "num_valid_rows": int(num_valid_rows),
        "subjects": subjects or sorted({int(row["subject"]) for row in file_summaries if "subject" in row}),
        "num_subjects": int(len(subjects or sorted({int(row["subject"]) for row in file_summaries if "subject" in row}))),
        "class_names": list(CAPTURE24_MOVEMENT4_LABELS),
        "class_counts_raw": {str(key): int(value) for key, value in class_counts_raw.items()},
        "windows_per_class": windows_per_class or {},
        "raw_window_size": int(window_size),
        "window_size": int(downsample_bins) if downsample_bins > 0 else int(window_size),
        "stride": int(stride),
        "downsample_bins": int(downsample_bins),
        "max_windows_per_subject_class": int(max_windows_per_subject_class),
        "feature_set": "wrist_accelerometer_xyz",
        "label_mapping": "movement4",
        "file_summaries": file_summaries[:200],
        "skipped_files": skipped_files[:200],
        "warnings": [
            "Capture-24 labels are mapped to movement4 only when raw labels match the configured broad taxonomy or an explicit label_map_path.",
            "This is algorithmic dataset inspection; no training claim is made without support_ok audit.",
        ],
    }
    (output_dir / "capture24_inspection.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")


def create_synthetic_external(
    num_classes: int,
    window_size: int,
    num_channels: int,
    seed: int,
    class_names: Iterable[str],
    train_subjects: int = 5,
    test_subjects: int = 2,
    windows_per_class: int = 12,
) -> ExternalArrays:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    groups: list[int] = []
    time = np.linspace(0.0, 1.0, window_size, dtype=np.float32)
    group_counter = 0
    for subject in range(1, train_subjects + test_subjects + 1):
        subject_bias = rng.normal(0.0, 0.05, size=(num_channels,)).astype(np.float32)
        for label in range(num_classes):
            group_counter += 1
            freq = 1 + (label % 7)
            base = np.sin(2 * np.pi * freq * time + 0.1 * subject).astype(np.float32)[:, None]
            for _ in range(windows_per_class):
                noise = rng.normal(0.0, 0.03, size=(window_size, num_channels)).astype(np.float32)
                xs.append(base + subject_bias + noise + label * 0.01)
                ys.append(label)
                subjects.append(subject)
                groups.append(group_counter)
    runs = ((np.asarray(groups, dtype=np.int64) - 1) % 5) + 1
    return ExternalArrays(
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(subjects, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
        tuple(class_names),
        tuple(f"ch_{idx}" for idx in range(num_channels)),
        runs=runs.astype(np.int64),
    )


def _build_external_loaders(
    config: dict[str, Any],
    arrays: ExternalArrays,
    synthetic: bool,
    perturbation: dict[str, Any],
) -> tuple[dict[str, DataLoader], SequenceDatasetMeta]:
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    seed = int(config.get("seed", 0))
    split_seed = int(dataset_cfg.get("split_seed", DEFAULT_SPLIT_SEED))
    context_len = int(dataset_cfg.get("context_len", 8))
    aligned_final_targets = bool(dataset_cfg.get("aligned_final_targets", False))
    aligned_context_max = dataset_cfg.get("aligned_context_max")
    protocol_type = str(dataset_cfg.get("protocol_type", "subject_disjoint"))
    if aligned_final_targets:
        if aligned_context_max is None:
            raise ValueError("aligned_context_max is required when aligned_final_targets=True")
        aligned_context_max = int(aligned_context_max)
        if context_len > aligned_context_max:
            raise ValueError("context_len must be <= aligned_context_max when aligned_final_targets=True")
    runtime = config.setdefault("_dataset_runtime", {})
    if protocol_type == "session_disjoint":
        splits = split_external_by_session(
            arrays,
            train_runs=dataset_cfg.get("train_runs", [1, 2, 3]),
            val_runs=dataset_cfg.get("val_runs", [4]),
            test_runs=dataset_cfg.get("test_runs", [5]),
            exclude_runs=dataset_cfg.get("exclude_runs", [0]),
        )
        runtime["split_runs"] = {
            "train": sorted(int(run) for run in np.unique(splits["train"][4])),
            "val": sorted(int(run) for run in np.unique(splits["val"][4])),
            "test": sorted(int(run) for run in np.unique(splits["test"][4])),
        }
    elif bool(dataset_cfg.get("coverage_split", False)):
        splits, split_report_path = select_subject_split_with_class_coverage(
            arrays=arrays,
            context_len=context_len,
            split_seed=split_seed,
            val_subject_fraction=float(dataset_cfg.get("val_subject_fraction", 0.2)),
            test_subject_fraction=float(dataset_cfg.get("test_subject_fraction", 0.2)),
            min_train_support_per_class=int(dataset_cfg.get("min_train_support_per_class", 50)),
            min_val_support_per_class=int(dataset_cfg.get("min_val_support_per_class", 10)),
            min_test_support_per_class=int(dataset_cfg.get("min_test_support_per_class", dataset_cfg.get("min_test_support", 20))),
            output_dir=Path(dataset_cfg.get("support_audit_dir", "results/external_context_v1")),
            dataset_key=str(dataset_cfg.get("name", "external")),
            class_names=arrays.class_names,
            aligned_context_max=int(aligned_context_max) if aligned_final_targets else None,
            max_candidates=int(dataset_cfg.get("max_split_search_candidates", 20_000)),
        )
        runtime["split_search_report_path"] = str(split_report_path)
    else:
        splits = split_external_by_subject(
            arrays,
            seed=split_seed,
            val_subject_fraction=float(dataset_cfg.get("val_subject_fraction", 0.2)),
            test_subject_fraction=float(dataset_cfg.get("test_subject_fraction", 0.2)),
            explicit_test_subjects=dataset_cfg.get("test_subjects"),
            explicit_val_subjects=dataset_cfg.get("val_subjects"),
        )
    runtime["split_seed"] = split_seed
    runtime["split_subjects"] = {
        split_name: sorted(int(subject) for subject in np.unique(values[2]))
        for split_name, values in splits.items()
    }
    runtime["protocol_type"] = protocol_type

    normalize = str(dataset_cfg.get("normalize", "train_zscore"))
    impute_missing = str(dataset_cfg.get("impute_missing", "train_channel_mean"))
    if normalize != "none" or impute_missing != "none":
        stats = fit_train_preprocessor(splits["train"][0], normalize=normalize, impute_missing=impute_missing)
        for split_name in ("train", "val", "test"):
            x, y, subjects, groups, *rest = splits[split_name]
            splits[split_name] = (apply_train_preprocessor(x, stats), y, subjects, groups, *rest)
        runtime["preprocessing"] = stats_to_serializable(stats)

    max_train = dataset_cfg.get("smoke_max_train_sequences") if dataset_cfg.get("_smoke_test", False) else None
    max_val = dataset_cfg.get("smoke_max_val_sequences") if dataset_cfg.get("_smoke_test", False) else None
    max_test = dataset_cfg.get("smoke_max_test_sequences") if dataset_cfg.get("_smoke_test", False) else None
    dataset_cls = AlignedSequenceWindowDataset if aligned_final_targets else SequenceWindowDataset

    def make_dataset(
        split_name: str,
        max_sequences: int | None,
        dataset_seed: int,
        extra: dict[str, Any] | None = None,
    ) -> SequenceWindowDataset:
        kwargs: dict[str, Any] = {
            "x": splits[split_name][0],
            "y": splits[split_name][1],
            "subjects": splits[split_name][2],
            "context_len": context_len,
            "channel_names": arrays.channel_names,
            "group_ids": splits[split_name][3],
            "max_sequences": max_sequences,
            "seed": dataset_seed,
        }
        if aligned_final_targets:
            kwargs["aligned_context_max"] = int(aligned_context_max)
        if extra:
            kwargs.update(extra)
        return dataset_cls(**kwargs)

    train_dataset = make_dataset("train", max_train, seed)
    val_dataset = make_dataset("val", max_val, seed + 10_000)
    test_dataset = make_dataset("test", max_test, seed + 20_000, perturbation)
    maybe_write_support_audit(
        config,
        arrays.class_names,
        {"train": train_dataset, "val": val_dataset, "test": test_dataset},
        split_seed=split_seed,
    )
    runtime["support_audit_path"] = str(
        support_audit_path(
            dataset_key=str(dataset_cfg.get("name", "external")),
            split_seed=split_seed,
            output_dir=Path(dataset_cfg.get("support_audit_dir", "results/external_context_v1")),
            context_len=context_len,
            aligned_context_max=int(aligned_context_max) if aligned_final_targets else None,
        )
    )
    maybe_update_inspection_with_split_runtime(config, {"train": train_dataset, "val": val_dataset, "test": test_dataset})

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))
    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    meta = SequenceDatasetMeta(
        context_len=context_len,
        num_channels=arrays.x.shape[-1],
        window_size=arrays.x.shape[1],
        num_classes=len(arrays.class_names),
        channel_names=arrays.channel_names,
        synthetic=synthetic,
    )
    return loaders, meta


def maybe_update_inspection_with_split_runtime(
    config: dict[str, Any],
    datasets: dict[str, SequenceWindowDataset],
) -> None:
    runtime = config.get("_dataset_runtime", {})
    inspection_path = runtime.get("inspection_path")
    support_audit = runtime.get("support_audit_path")
    if not inspection_path:
        return
    path = Path(inspection_path)
    if not path.exists():
        return
    try:
        inspection = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    support_by_split: dict[str, dict[str, int]] = {}
    for split_name, dataset in datasets.items():
        counts: dict[str, int] = {}
        for indices in dataset.indices:
            class_id = int(dataset.y[indices[-1]])
            class_key = str(class_id)
            counts[class_key] = counts.get(class_key, 0) + 1
        support_by_split[split_name] = counts
    if support_audit and Path(support_audit).exists():
        audit = pd.read_csv(support_audit)
        support_by_split = {
            split: {
                str(row.class_name): int(getattr(row, f"{split}_support"))
                for row in audit.itertuples(index=False)
            }
            for split in ("train", "val", "test")
        }
    inspection.update(
        {
            "coverage_split": bool(config.get("dataset", {}).get("coverage_split", False)),
            "selected_train_subjects": runtime.get("split_subjects", {}).get("train", []),
            "selected_val_subjects": runtime.get("split_subjects", {}).get("val", []),
            "selected_test_subjects": runtime.get("split_subjects", {}).get("test", []),
            "selected_train_runs": runtime.get("split_runs", {}).get("train", []),
            "selected_val_runs": runtime.get("split_runs", {}).get("val", []),
            "selected_test_runs": runtime.get("split_runs", {}).get("test", []),
            "protocol_type": runtime.get("protocol_type", config.get("dataset", {}).get("protocol_type", "subject_disjoint")),
            "support_by_class": support_by_split,
            "num_train_sequences": int(len(datasets["train"])),
            "num_val_sequences": int(len(datasets["val"])),
            "num_test_sequences": int(len(datasets["test"])),
        }
    )
    if runtime.get("split_search_report_path"):
        inspection["split_search_report_path"] = runtime["split_search_report_path"]
    path.write_text(json.dumps(inspection, indent=2), encoding="utf-8")


def split_external_by_subject(
    arrays: ExternalArrays,
    seed: int,
    val_subject_fraction: float,
    test_subject_fraction: float,
    explicit_test_subjects: Iterable[int] | None = None,
    explicit_val_subjects: Iterable[int] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    subjects = np.unique(arrays.subjects)
    if len(subjects) < 3:
        raise ValueError("External datasets need at least three subjects for subject-disjoint train/val/test splits")
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    test_subjects = set(int(s) for s in explicit_test_subjects) if explicit_test_subjects else set(
        int(s) for s in shuffled[: max(1, int(round(len(subjects) * test_subject_fraction)))]
    )
    remaining = np.asarray([s for s in subjects if int(s) not in test_subjects], dtype=np.int64)
    if explicit_val_subjects:
        val_subjects = set(int(s) for s in explicit_val_subjects)
    else:
        rng.shuffle(remaining)
        val_count = max(1, int(round(len(remaining) * val_subject_fraction)))
        val_subjects = set(int(s) for s in remaining[:val_count])
    train_subjects = set(int(s) for s in subjects if int(s) not in test_subjects and int(s) not in val_subjects)
    if not train_subjects or not val_subjects or not test_subjects:
        raise ValueError("Subject split produced an empty train/val/test group")

    def take(selected: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask = np.asarray([int(subject) in selected for subject in arrays.subjects], dtype=bool)
        return arrays.x[mask], arrays.y[mask], arrays.subjects[mask], arrays.group_ids[mask]

    return {"train": take(train_subjects), "val": take(val_subjects), "test": take(test_subjects)}


def split_external_by_session(
    arrays: ExternalArrays,
    train_runs: Iterable[int],
    val_runs: Iterable[int],
    test_runs: Iterable[int],
    exclude_runs: Iterable[int] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if arrays.runs is None:
        raise ValueError("Session-disjoint protocol requires run/session IDs")
    exclude = set(int(run) for run in (exclude_runs or []))
    run_sets = {
        "train": set(int(run) for run in train_runs),
        "val": set(int(run) for run in val_runs),
        "test": set(int(run) for run in test_runs),
    }
    if any(run in exclude for selected in run_sets.values() for run in selected):
        raise ValueError("Run cannot be both selected and excluded in session-disjoint split")

    def take(selected: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask = np.asarray([int(run) in selected and int(run) not in exclude for run in arrays.runs], dtype=bool)
        if not mask.any():
            raise ValueError(f"Session-disjoint split produced empty run set: {sorted(selected)}")
        return arrays.x[mask], arrays.y[mask], arrays.subjects[mask], arrays.group_ids[mask], arrays.runs[mask]

    return {split: take(selected) for split, selected in run_sets.items()}


def select_subject_split_with_class_coverage(
    arrays: ExternalArrays,
    context_len: int,
    split_seed: int,
    val_subject_fraction: float,
    test_subject_fraction: float,
    min_train_support_per_class: int,
    min_val_support_per_class: int,
    min_test_support_per_class: int,
    output_dir: Path,
    dataset_key: str,
    class_names: tuple[str, ...],
    aligned_context_max: int | None = None,
    max_candidates: int = 20_000,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], Path]:
    subjects = np.asarray(sorted(int(subject) for subject in np.unique(arrays.subjects)), dtype=np.int64)
    if len(subjects) < 3:
        raise ValueError("Coverage split requires at least three subjects")
    num_classes = len(class_names)
    subject_support = final_window_support_by_subject(arrays, context_len, num_classes, aligned_context_max)
    test_count = max(1, int(round(len(subjects) * test_subject_fraction)))
    val_count = max(1, int(round((len(subjects) - test_count) * val_subject_fraction)))
    rng = np.random.default_rng(split_seed)
    feasible: list[dict[str, Any]] = []
    evaluated = 0
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    for test_subjects, val_subjects, candidate_index in subject_split_candidates(
        subjects,
        test_count,
        val_count,
        rng,
        max_candidates,
    ):
        key = (tuple(sorted(test_subjects)), tuple(sorted(val_subjects)))
        if key in seen:
            continue
        seen.add(key)
        evaluated += 1
        train_subjects = tuple(int(subject) for subject in subjects if int(subject) not in set(test_subjects) | set(val_subjects))
        supports = {
            "train": support_for_subjects(subject_support, train_subjects, num_classes),
            "val": support_for_subjects(subject_support, val_subjects, num_classes),
            "test": support_for_subjects(subject_support, test_subjects, num_classes),
        }
        thresholds_ok = (
            np.all(supports["train"] >= int(min_train_support_per_class))
            and np.all(supports["val"] >= int(min_val_support_per_class))
            and np.all(supports["test"] >= int(min_test_support_per_class))
        )
        if not thresholds_ok:
            continue
        test_mean = float(supports["test"].mean()) if len(supports["test"]) else 0.0
        test_balance = float(supports["test"].std() / (test_mean + 1e-9)) if test_mean > 0 else float("inf")
        subject_balance = abs(len(test_subjects) - test_count) + abs(len(val_subjects) - val_count)
        feasible.append(
            {
                "train_subjects": train_subjects,
                "val_subjects": tuple(int(subject) for subject in val_subjects),
                "test_subjects": tuple(int(subject) for subject in test_subjects),
                "supports": supports,
                "score": (
                    int(supports["test"].min()),
                    -test_balance,
                    -subject_balance,
                    -int(candidate_index),
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"split_search_{dataset_key}_split{int(split_seed)}.json"
    thresholds = {
        "min_train_support_per_class": int(min_train_support_per_class),
        "min_val_support_per_class": int(min_val_support_per_class),
        "min_test_support_per_class": int(min_test_support_per_class),
        "context_len": int(context_len),
        "aligned_context_max": int(aligned_context_max) if aligned_context_max is not None else None,
    }
    if not feasible:
        best_effort = best_effort_subject_support(subject_support, subjects, num_classes)
        write_split_search_report(
            report_path,
            dataset_key,
            split_seed,
            evaluated,
            feasible,
            thresholds,
            class_names,
            reason="no feasible subject-disjoint split meets class support thresholds",
            warnings=["coverage_split_failed"],
            best_effort=best_effort,
        )
        raise ValueError(
            f"No coverage-aware split for {dataset_key} meets support thresholds; see {report_path}"
        )

    selected = max(feasible, key=lambda item: item["score"])
    write_split_search_report(
        report_path,
        dataset_key,
        split_seed,
        evaluated,
        feasible,
        thresholds,
        class_names,
        reason="selected feasible coverage-aware subject split",
        warnings=[],
        selected=selected,
    )
    splits = {
        "train": take_subjects(arrays, set(selected["train_subjects"])),
        "val": take_subjects(arrays, set(selected["val_subjects"])),
        "test": take_subjects(arrays, set(selected["test_subjects"])),
    }
    return splits, report_path


def final_window_support_by_subject(
    arrays: ExternalArrays,
    context_len: int,
    num_classes: int,
    aligned_context_max: int | None = None,
) -> dict[int, np.ndarray]:
    supports: dict[int, np.ndarray] = {
        int(subject): np.zeros(num_classes, dtype=np.int64) for subject in np.unique(arrays.subjects)
    }
    required_history = int(aligned_context_max) if aligned_context_max is not None else int(context_len)
    pairs = np.stack([arrays.subjects, arrays.group_ids], axis=1)
    for subject_id, group_id in np.unique(pairs, axis=0):
        group_indices = np.flatnonzero((arrays.subjects == subject_id) & (arrays.group_ids == group_id))
        if len(group_indices) < required_history:
            continue
        for final_index in group_indices[required_history - 1 :]:
            supports[int(subject_id)][int(arrays.y[final_index])] += 1
    return supports


def subject_split_candidates(
    subjects: np.ndarray,
    test_count: int,
    val_count: int,
    rng: np.random.Generator,
    max_candidates: int,
) -> Iterable[tuple[tuple[int, ...], tuple[int, ...], int]]:
    subject_list = tuple(int(subject) for subject in subjects)
    if len(subject_list) <= 10:
        candidate_index = 0
        for test_subjects in combinations(subject_list, test_count):
            remaining = tuple(subject for subject in subject_list if subject not in set(test_subjects))
            for val_subjects in combinations(remaining, val_count):
                yield tuple(test_subjects), tuple(val_subjects), candidate_index
                candidate_index += 1
        return
    for candidate_index in range(int(max_candidates)):
        shuffled = rng.permutation(subjects).astype(int).tolist()
        test_subjects = tuple(sorted(shuffled[:test_count]))
        remaining = shuffled[test_count:]
        val_subjects = tuple(sorted(remaining[:val_count]))
        yield test_subjects, val_subjects, candidate_index


def support_for_subjects(
    subject_support: dict[int, np.ndarray],
    subjects: Iterable[int],
    num_classes: int,
) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for subject in subjects:
        counts += subject_support.get(int(subject), np.zeros(num_classes, dtype=np.int64))
    return counts


def take_subjects(
    arrays: ExternalArrays,
    selected: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray([int(subject) in selected for subject in arrays.subjects], dtype=bool)
    return arrays.x[mask], arrays.y[mask], arrays.subjects[mask], arrays.group_ids[mask]


def best_effort_subject_support(
    subject_support: dict[int, np.ndarray],
    subjects: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    rows = []
    for subject in subjects:
        support = subject_support.get(int(subject), np.zeros(num_classes, dtype=np.int64))
        rows.append({"subject": int(subject), "support": support.astype(int).tolist()})
    total = support_for_subjects(subject_support, subjects, num_classes)
    return {"per_subject": rows, "total_support": total.astype(int).tolist()}


def write_split_search_report(
    path: Path,
    dataset_key: str,
    split_seed: int,
    candidate_count: int,
    feasible: list[dict[str, Any]],
    thresholds: dict[str, Any],
    class_names: tuple[str, ...],
    reason: str,
    warnings: list[str],
    selected: dict[str, Any] | None = None,
    best_effort: dict[str, Any] | None = None,
) -> None:
    report: dict[str, Any] = {
        "dataset_key": dataset_key,
        "split_seed": int(split_seed),
        "candidate_count": int(candidate_count),
        "feasible_count": int(len(feasible)),
        "thresholds": thresholds,
        "class_names": list(class_names),
        "reason": reason,
        "warnings": warnings,
    }
    if selected is not None:
        report.update(
            {
                "selected_train_subjects": [int(value) for value in selected["train_subjects"]],
                "selected_val_subjects": [int(value) for value in selected["val_subjects"]],
                "selected_test_subjects": [int(value) for value in selected["test_subjects"]],
                "support_by_split_and_class": {
                    split: {
                        class_names[idx]: int(count)
                        for idx, count in enumerate(selected["supports"][split])
                    }
                    for split in ("train", "val", "test")
                },
            }
        )
    if best_effort is not None:
        report["best_effort"] = best_effort
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def maybe_write_support_audit(
    config: dict[str, Any],
    class_names: tuple[str, ...],
    datasets: dict[str, SequenceWindowDataset],
    split_seed: int,
) -> None:
    dataset_cfg = config.get("dataset", {})
    if not bool(dataset_cfg.get("write_support_audit", True)):
        return
    output_dir = Path(dataset_cfg.get("support_audit_dir", "results/external_context_v1"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_key = str(dataset_cfg.get("name", "external"))
    min_support = int(dataset_cfg.get("min_test_support", 20))
    rows: list[dict[str, Any]] = []
    supports: dict[str, np.ndarray] = {}
    for split_name, dataset in datasets.items():
        counts = np.zeros(len(class_names), dtype=np.int64)
        for indices in dataset.indices:
            counts[int(dataset.y[indices[-1]])] += 1
        supports[split_name] = counts
    for class_id, class_name in enumerate(class_names):
        test_support = int(supports["test"][class_id])
        rows.append(
            {
                "dataset_key": dataset_key,
                "class_id": class_id,
                "class_name": class_name,
                "train_support": int(supports["train"][class_id]),
                "val_support": int(supports["val"][class_id]),
                "test_support": test_support,
                "min_support_flag": bool(0 < test_support < min_support),
                "zero_support_flag": bool(test_support == 0),
            }
        )
    frame = pd.DataFrame(rows)
    aligned_final_targets = bool(dataset_cfg.get("aligned_final_targets", False))
    path = support_audit_path(
        dataset_key=dataset_key,
        split_seed=split_seed,
        output_dir=output_dir,
        context_len=int(dataset_cfg.get("context_len", 8)),
        aligned_context_max=int(dataset_cfg["aligned_context_max"]) if aligned_final_targets else None,
    )
    if path.exists():
        existing = pd.read_csv(path)
        if list(existing.columns) == list(frame.columns) and existing.astype(str).equals(frame.astype(str)):
            return
    frame.to_csv(path, index=False)


def support_audit_path(
    dataset_key: str,
    split_seed: int,
    output_dir: Path,
    context_len: int | None = None,
    aligned_context_max: int | None = None,
) -> Path:
    if aligned_context_max is not None:
        if context_len is None:
            raise ValueError("context_len is required for aligned support audit paths")
        return output_dir / (
            f"support_audit_{dataset_key}_split{int(split_seed)}_k{int(context_len)}_aligned{int(aligned_context_max)}.csv"
        )
    return output_dir / f"support_audit_{dataset_key}_split{int(split_seed)}.csv"
