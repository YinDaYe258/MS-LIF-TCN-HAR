from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.mhealth import MHEALTH_CHANNELS, load_mhealth_windows, resolve_mhealth_root
from src.datasets.pamap2 import PAMAP2_ACTIVITY_IDS, pamap2_channel_indices, load_pamap2_windows, resolve_pamap2_root
from src.datasets.preprocessing import fit_train_preprocessor, stats_to_serializable
from src.datasets.sequence_dataset import SequenceWindowDataset
from src.datasets.ucihar import split_train_val_by_subject


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect PAMAP2 or MHEALTH data protocol.")
    parser.add_argument("--dataset", choices=["pamap2", "mhealth"], required=True)
    parser.add_argument("--root")
    parser.add_argument("--output_dir", default="results/final_paper_v3")
    parser.add_argument("--context_len", type=int, default=8)
    parser.add_argument("--low_support_threshold", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_subject_fraction", type=float, default=0.2)
    parser.add_argument("--normalize", default="train_zscore")
    parser.add_argument("--impute_missing", default="train_channel_mean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "pamap2":
        root = resolve_pamap2_root(args.root or "data/PAMAP2_Dataset")
        train = load_pamap2_windows(root, "train")
        test = load_pamap2_windows(root, "test")
        channel_names = pamap2_channel_indices("acc_gyro")[1]
        label_count = len(PAMAP2_ACTIVITY_IDS)
    else:
        root = resolve_mhealth_root(args.root or "data/MHEALTHDATASET")
        train = load_mhealth_windows(root, "train")
        test = load_mhealth_windows(root, "test")
        channel_names = MHEALTH_CHANNELS
        label_count = 12
    sequence_distribution = class_distribution(train, test, label_count, args.context_len, args.low_support_threshold)
    report = build_report(
        args.dataset,
        root,
        train,
        test,
        channel_names,
        label_count,
        sequence_distribution,
        args.context_len,
        args.low_support_threshold,
    )
    (output_dir / f"{args.dataset}_inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    stats_path = write_preprocessing_stats(
        output_dir,
        args.dataset,
        train,
        seed=args.seed,
        val_subject_fraction=args.val_subject_fraction,
        normalize=args.normalize,
        impute_missing=args.impute_missing,
    )
    report["preprocessing_stats_path"] = str(stats_path)
    (output_dir / f"{args.dataset}_inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    sequence_distribution.to_csv(output_dir / f"{args.dataset}_class_distribution.csv", index=False)
    sequence_distribution.to_csv(output_dir / f"{args.dataset}_sequence_class_distribution_k{args.context_len}.csv", index=False)
    print(json.dumps(report, indent=2))
    print(f"Saved inspection files under {output_dir}")


def build_report(
    dataset: str,
    root: Path,
    train,
    test,
    channel_names: tuple[str, ...],
    label_count: int,
    distribution: pd.DataFrame,
    context_len: int,
    low_support_threshold: int,
) -> dict[str, object]:
    x_train, y_train, subjects_train, meta_train = train
    x_test, y_test, subjects_test, meta_test = test
    train_subjects = sorted(int(subject) for subject in np.unique(subjects_train))
    test_subjects = sorted(int(subject) for subject in np.unique(subjects_test))
    support_col = f"sequence_support_k{context_len}_final_label"
    train_sequences = int(distribution[distribution["split"].eq("train")][support_col].sum())
    test_sequences = int(distribution[distribution["split"].eq("test")][support_col].sum())
    test_support = distribution[distribution["split"].eq("test")]
    nonzero_support = test_support[test_support[support_col] > 0][support_col]
    low_support = test_support[test_support["low_support_flag"]]["class_id"].astype(int).tolist()
    zero_support = test_support[test_support[support_col].eq(0)]["class_id"].astype(int).tolist()
    return {
        "dataset": dataset,
        "root": str(root),
        "window_shape": list(x_train.shape[1:]),
        "num_channels": len(channel_names),
        "channel_names": list(channel_names),
        "num_classes": label_count,
        "train_windows": int(len(x_train)),
        "test_windows": int(len(x_test)),
        f"train_sequences_k{context_len}": train_sequences,
        f"test_sequences_k{context_len}": test_sequences,
        f"min_test_sequences_per_class_k{context_len}": int(nonzero_support.min()) if not nonzero_support.empty else 0,
        f"classes_with_low_support_k{context_len}": low_support,
        f"classes_with_zero_support_k{context_len}": zero_support,
        "low_support_threshold": int(low_support_threshold),
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "subject_overlap": sorted(set(train_subjects) & set(test_subjects)),
        "train_segments": int(len(np.unique(meta_train["segment_ids"]))),
        "test_segments": int(len(np.unique(meta_test["segment_ids"]))),
    }


def class_distribution(train, test, label_count: int, context_len: int, low_support_threshold: int) -> pd.DataFrame:
    rows = []
    for split, arrays in [("train", train), ("test", test)]:
        x, labels, subjects, meta = arrays
        sequence_labels, sequence_groups = final_sequence_labels_and_groups(x, labels, subjects, meta["segment_ids"], context_len)
        for label in range(label_count):
            label_sequence_groups = sequence_groups[sequence_labels == label]
            segment_counts = pd.Series(label_sequence_groups).value_counts() if len(label_sequence_groups) else pd.Series(dtype=int)
            sequence_support = int(len(label_sequence_groups))
            rows.append(
                {
                    "split": split,
                    "class_id": label,
                    "window_support": int((labels == label).sum()),
                    f"sequence_support_k{context_len}_final_label": sequence_support,
                    "num_segments": int((meta["segment_ids"][labels == label].size and len(np.unique(meta["segment_ids"][labels == label])))),
                    "min_segment_sequences": int(segment_counts.min()) if not segment_counts.empty else 0,
                    "context_len": int(context_len),
                    "low_support_flag": bool(split == "test" and sequence_support < low_support_threshold),
                }
            )
    return pd.DataFrame(rows)


def final_sequence_labels_and_groups(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    group_ids: np.ndarray,
    context_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = SequenceWindowDataset(x, y, subjects, context_len=context_len, group_ids=group_ids)
    if len(dataset.indices) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    return (
        np.asarray([int(y[indices[-1]]) for indices in dataset.indices], dtype=np.int64),
        np.asarray([int(group_ids[indices[-1]]) for indices in dataset.indices], dtype=np.int64),
    )


def write_preprocessing_stats(
    output_dir: Path,
    dataset: str,
    train,
    seed: int,
    val_subject_fraction: float,
    normalize: str,
    impute_missing: str,
) -> Path:
    x_train, y_train, subjects_train, _ = train
    train_split, _ = split_train_val_by_subject(
        x_train,
        y_train,
        subjects_train,
        val_subject_fraction=val_subject_fraction,
        seed=seed,
    )
    stats = stats_to_serializable(
        fit_train_preprocessor(train_split[0], normalize=normalize, impute_missing=impute_missing)
    )
    stats.update(
        {
            "dataset": dataset,
            "seed": int(seed),
            "computed_from": "train_subjects_only_after_train_val_split",
            "nan_imputation": impute_missing,
            "normalization": normalize,
            "val_subject_fraction": float(val_subject_fraction),
        }
    )
    stats_dir = output_dir / "preprocessing_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / f"{dataset}_seed{seed}_train_stats.json"
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
