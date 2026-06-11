from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.hapt import load_hapt_windows


CANDIDATE_DIRS = (
    Path("data/HAPT Dataset"),
    Path("data/Smartphone-Based Recognition of Human Activities and Postural Transitions"),
    Path("data/smartphone_based_recognition_of_human_activities_and_postural_transitions"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect HAPT raw dataset integrity.")
    parser.add_argument("--root", default=None)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--context_len", type=int, default=8)
    return parser.parse_args()


def detect_root(root_arg: str | None) -> Path:
    candidates = [Path(root_arg)] if root_arg else list(CANDIDATE_DIRS)
    for candidate in candidates:
        if (candidate / "RawData" / "labels.txt").exists():
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find HAPT RawData/labels.txt. Checked: {checked}")


def main() -> None:
    args = parse_args()
    root = detect_root(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = root / "RawData"
    acc_files = sorted(raw_dir.glob("acc_exp*_user*.txt"))
    gyro_files = sorted(raw_dir.glob("gyro_exp*_user*.txt"))
    labels = np.loadtxt(raw_dir / "labels.txt", dtype=np.int64)
    labels = np.atleast_2d(labels)
    activities = read_activity_labels(root / "activity_labels.txt")
    train_subjects = set(np.loadtxt(root / "Train" / "subject_id_train.txt", dtype=np.int64).reshape(-1).tolist())
    test_subjects = set(np.loadtxt(root / "Test" / "subject_id_test.txt", dtype=np.int64).reshape(-1).tolist())

    raw_keys = {path.name.replace("acc_", "").replace("gyro_", "") for path in acc_files + gyro_files}
    label_keys = {f"exp{int(exp):02d}_user{int(user):02d}.txt" for exp, user in labels[:, :2]}
    segment_lengths = labels[:, 4] - labels[:, 3] + 1
    class_distribution = pd.DataFrame(
        [
            {
                "activity_id": int(activity_id),
                "activity_name": activities.get(int(activity_id), f"class_{int(activity_id)}"),
                "segments": int(np.sum(labels[:, 2] == activity_id)),
                "raw_window_count": int(
                    np.sum(
                        np.maximum(
                            0,
                            ((labels[labels[:, 2] == activity_id, 4] - labels[labels[:, 2] == activity_id, 3] + 1) - args.window_size)
                            // args.stride
                            + 1,
                        )
                    )
                ),
            }
            for activity_id in sorted(np.unique(labels[:, 2]).tolist())
        ]
    )

    sequence_counts: dict[str, dict[str, Any]] = {}
    context_lens = sorted({2, 4, int(args.context_len)})
    for task in ("hapt6", "hapt12"):
        task_counts: dict[str, Any] = {}
        for split in ("train", "test"):
            x, y, subjects, meta = load_hapt_windows(root, split, args.window_size, args.stride, task=task)
            groups = meta["segment_ids"]
            task_counts[split] = {
                "windows": int(len(y)),
                "subjects": sorted(int(v) for v in np.unique(subjects).tolist()),
                "class_distribution": {
                    str(int(label)): int(count) for label, count in zip(*np.unique(y, return_counts=True))
                },
            }
            for context_len in context_lens:
                sequence_count, sequence_label_distribution = count_sequences_by_last_label(
                    y,
                    subjects,
                    groups,
                    context_len,
                )
                task_counts[split][f"k{context_len}_sequence_count_within_segment"] = int(sequence_count)
                task_counts[split][f"k{context_len}_sequence_label_distribution_within_segment"] = sequence_label_distribution
        sequence_counts[task] = task_counts

    summary = {
        "root": str(root),
        "raw_acc_files": len(acc_files),
        "raw_gyro_files": len(gyro_files),
        "label_rows": int(len(labels)),
        "activity_labels": {str(k): v for k, v in activities.items()},
        "train_subjects": sorted(int(v) for v in train_subjects),
        "test_subjects": sorted(int(v) for v in test_subjects),
        "train_test_subject_overlap": sorted(int(v) for v in train_subjects & test_subjects),
        "raw_files_cover_label_experiments": sorted(label_keys - raw_keys),
        "unlabeled_raw_files": sorted(raw_keys - label_keys),
        "segment_length": {
            "min": int(segment_lengths.min()),
            "max": int(segment_lengths.max()),
            "mean": float(segment_lengths.mean()),
        },
        "sequence_counts": sequence_counts,
    }
    (output_dir / "hapt_dataset_inspection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    class_distribution.to_csv(output_dir / "hapt_class_distribution.csv", index=False)

    print(json.dumps(summary, indent=2))
    print(f"Saved {output_dir / 'hapt_dataset_inspection.json'}")
    print(f"Saved {output_dir / 'hapt_class_distribution.csv'}")


def read_activity_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            labels[int(parts[0])] = parts[1]
    return labels


def count_sequences_by_last_label(
    y: np.ndarray,
    subjects: np.ndarray,
    groups: np.ndarray,
    context_len: int,
) -> tuple[int, dict[str, int]]:
    sequence_count = 0
    sequence_labels: list[int] = []
    for subject_id, group_id in np.unique(np.stack([subjects, groups], axis=1), axis=0):
        indices = np.flatnonzero((subjects == subject_id) & (groups == group_id))
        valid_sequences = max(0, len(indices) - context_len + 1)
        sequence_count += valid_sequences
        for start in range(valid_sequences):
            sequence_labels.append(int(y[indices[start + context_len - 1]]))
    if not sequence_labels:
        return sequence_count, {}
    labels, counts = np.unique(np.asarray(sequence_labels, dtype=np.int64), return_counts=True)
    return sequence_count, {str(int(label)): int(count) for label, count in zip(labels, counts)}


if __name__ == "__main__":
    main()
