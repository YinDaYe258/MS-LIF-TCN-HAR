from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_final_paper_v3 import build_name_for
from src.datasets.hapt import create_hapt_dataloaders, load_hapt_windows
from src.datasets.sequence_dataset import SequenceWindowDataset
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed

V3_DIR = Path("results/final_paper_v3")
ARTIFACT_DIR = V3_DIR / "artifacts"
HAPT12_AUDIT_PATH = V3_DIR / "hapt12_transition_support_audit.csv"
BINARY_AUDIT_PATH = V3_DIR / "hapt_transition_binary_support_audit.csv"
HAPT12_OUTPUT_PATH = V3_DIR / "hapt12_transition_diagnostic.csv"
BINARY_OUTPUT_PATH = V3_DIR / "hapt_transition_binary_diagnostic.csv"
HAPT12_SMOKE_PATH = V3_DIR / "hapt12_transition_diagnostic_smoke.csv"
BINARY_SMOKE_PATH = V3_DIR / "hapt_transition_binary_diagnostic_smoke.csv"

HAPT12_CONFIG = "configs/hapt12_k2_last.yaml"
BINARY_CONFIG = "configs/hapt_transition_binary_k2_last.yaml"
DEFAULT_MODELS = ["ms_lif_snn_wide", "ms_ann_tcn", "ms_lif_tcn"]
DEFAULT_CONTEXT_LENS = [2, 4]
LOW_SUPPORT_THRESHOLD = 20
BINARY_POSITIVE_WEAK_THRESHOLD = 50

HAPT12_CLASS_NAMES = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
    6: "STAND_TO_SIT",
    7: "SIT_TO_STAND",
    8: "SIT_TO_LIE",
    9: "LIE_TO_SIT",
    10: "STAND_TO_LIE",
    11: "LIE_TO_STAND",
}
BINARY_CLASS_NAMES = {0: "non_transition", 1: "transition"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 HAPT transition supplementary diagnostics.")
    parser.add_argument("--tasks", nargs="+", default=["hapt12", "binary"], choices=["hapt12", "binary"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--context_lens", nargs="+", type=int, default=DEFAULT_CONTEXT_LENS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--allow_low_support_hapt12", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    V3_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    hapt12_audit, binary_audit = write_support_audits(args.context_lens)
    if args.audit_only:
        print(f"Wrote {HAPT12_AUDIT_PATH}")
        print(f"Wrote {BINARY_AUDIT_PATH}")
        return

    for task in args.tasks:
        for context_len in args.context_lens:
            if task == "hapt12" and not args.allow_low_support_hapt12 and not hapt12_context_is_supported(
                hapt12_audit, context_len
            ):
                print(f"Skipping HAPT-12 K={context_len}: transition-class support below threshold.")
                continue
            if task == "binary" and not binary_context_has_positive_support(binary_audit, context_len):
                print(f"Skipping transition-binary K={context_len}: no positive transition test support.")
                continue
            for model_label in args.models:
                for seed in args.seeds:
                    budget = budget_for_run(task, context_len, model_label, int(seed), args)
                    output_path = output_path_for(task, args.smoke_test)
                    if not args.force and row_exists(
                        output_path,
                        task,
                        int(context_len),
                        model_label,
                        int(seed),
                        budget["training_budget"],
                    ):
                        print(f"Skipping existing HAPT transition row: {task} K={context_len} {model_label} seed{seed}")
                        continue
                    row = run_one(task, int(context_len), model_label, int(seed), args, hapt12_audit, binary_audit)
                    append_csv_row(output_path, row)
                    transition = ""
                    if task == "binary":
                        transition = f", transition_f1={row['transition_f1']:.4f}"
                    print(
                        "Saved HAPT transition row: "
                        f"{task} K={context_len} {model_label} seed{seed} -> macro_f1={row['macro_f1']:.4f}{transition}"
                    )


def run_one(
    task: str,
    context_len: int,
    model_label: str,
    seed: int,
    args: argparse.Namespace,
    hapt12_audit: pd.DataFrame,
    binary_audit: pd.DataFrame,
) -> dict[str, Any]:
    config = make_config(task, context_len, model_label, seed, args)
    set_seed(seed)
    build_name = build_name_for(model_label)
    loaders, meta = create_hapt_dataloaders(config, model_name=build_name, smoke_test=args.smoke_test)
    device = get_device(config.get("device", "auto"))
    model = build_model(build_name, meta.num_channels, meta.num_classes, config.get("model", {}))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    suffix = "_smoke" if args.smoke_test else ""
    run_name = f"v3_hapt_transition_{task}_k{context_len}_{model_label}_{target_mode}_seed{seed}{suffix}"
    trainer = Trainer(
        model,
        loaders,
        config,
        device,
        run_name,
        results_dir=ARTIFACT_DIR,
        num_classes=meta.num_classes,
    )
    metrics = trainer.fit()
    model_cfg = config.get("model", {})
    training_meta = training_metadata(config)
    spike_rate = None if model_label == "ms_ann_tcn" else float(metrics.get("spike_rate", 0.0))
    audit_status = audit_status_for(task, context_len, hapt12_audit, binary_audit)
    row: dict[str, Any] = {
        "dataset": "HAPT-12" if task == "hapt12" else "HAPT Transition Binary",
        "dataset_key": "hapt12_transition" if task == "hapt12" else "hapt_transition_binary",
        "task": task,
        "diagnostic": "hapt_transition",
        "model": model_label,
        "build_model": build_name,
        "seed": int(seed),
        "context_len": int(meta.context_len),
        "window_size": int(meta.window_size),
        "num_channels": int(meta.num_channels),
        "num_classes": int(meta.num_classes),
        "target_mode": target_mode,
        "sequence_protocol": f"fixed_k{context_len}_within_segment",
        "synthetic_data": bool(meta.synthetic),
        "smoke_test": bool(args.smoke_test),
        **training_meta,
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "branch_dim": int(model_cfg.get("branch_dim", 32)),
        "tcn_layers": int(model_cfg.get("tcn_layers", 0)),
        "spike_reg_lambda": float(config.get("training", {}).get("spike_reg_lambda", 0.0)),
        "params": int(count_parameters(model)),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
        "loss": float(metrics.get("loss", 0.0)),
        "spike_rate": spike_rate,
        "best_epoch": int(metrics.get("best_epoch", 0)),
        "best_val_macro_f1": float(metrics.get("best_val_macro_f1", 0.0)),
        "checkpoint": metrics.get("checkpoint", ""),
        "epoch_log": metrics.get("epoch_log", ""),
        "confusion_matrix_path": metrics.get("confusion_matrix_path", ""),
        "support_audit_status": audit_status,
        "note": "Supplementary diagnostic only; HAPT-12/transition tasks are not primary main-result datasets.",
    }
    if task == "binary":
        row.update(binary_transition_metrics(np.asarray(metrics["confusion_matrix"], dtype=np.int64)))
    return row


def make_config(task: str, context_len: int, model_label: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    config_path = HAPT12_CONFIG if task == "hapt12" else BINARY_CONFIG
    config = copy.deepcopy(load_config(config_path))
    config["seed"] = int(seed)
    config.setdefault("results", {})["dir"] = str(ARTIFACT_DIR)
    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["context_len"] = int(context_len)
    if task == "hapt12":
        dataset_cfg["name"] = "hapt12"
        dataset_cfg["task"] = "hapt12"
    else:
        dataset_cfg["name"] = "hapt_transition_binary"
        dataset_cfg["task"] = "transitionbinary"
    model_cfg = config.setdefault("model", {})
    if model_label == "ms_lif_snn_wide":
        model_cfg["hidden_dim"] = 224
        model_cfg["branch_dim"] = 64
        model_cfg["tcn_layers"] = 0
    elif model_label in {"ms_lif_tcn", "ms_ann_tcn"}:
        model_cfg["tcn_layers"] = int(model_cfg.get("tcn_layers", 2))
    training = config.setdefault("training", {})
    training["epochs"] = int(args.epochs)
    training["patience"] = int(args.patience)
    if args.batch_size is not None:
        training["batch_size"] = int(args.batch_size)
    if args.smoke_test:
        training["epochs"] = 1
        training["patience"] = 1
        dataset_cfg["smoke_max_train_sequences"] = min(int(dataset_cfg.get("smoke_max_train_sequences", 64)), 64)
        dataset_cfg["smoke_max_val_sequences"] = min(int(dataset_cfg.get("smoke_max_val_sequences", 32)), 32)
        dataset_cfg["smoke_max_test_sequences"] = min(int(dataset_cfg.get("smoke_max_test_sequences", 32)), 32)
    return config


def write_support_audits(context_lens: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    hapt12 = build_support_audit("hapt12", context_lens)
    binary = build_support_audit("binary", context_lens)
    hapt12.to_csv(HAPT12_AUDIT_PATH, index=False)
    binary.to_csv(BINARY_AUDIT_PATH, index=False)
    return hapt12, binary


def build_support_audit(task: str, context_lens: list[int], root: str | Path = "data/HAPT Dataset") -> pd.DataFrame:
    dataset_task = "hapt12" if task == "hapt12" else "transitionbinary"
    class_names = HAPT12_CLASS_NAMES if task == "hapt12" else BINARY_CLASS_NAMES
    num_classes = len(class_names)
    rows: list[dict[str, Any]] = []
    for context_len in sorted(set(int(value) for value in context_lens)):
        for split in ["train", "test"]:
            x, y, subjects, meta = load_hapt_windows(root, split, task=dataset_task)
            groups = np.asarray(meta["segment_ids"], dtype=np.int64)
            dataset = SequenceWindowDataset(x, y, subjects, context_len=context_len, group_ids=groups)
            final_labels = np.asarray([int(y[indices[-1]]) for indices in dataset.indices], dtype=np.int64)
            final_groups = np.asarray([int(groups[indices[-1]]) for indices in dataset.indices], dtype=np.int64)
            window_support = np.bincount(y, minlength=num_classes)
            sequence_support = np.bincount(final_labels, minlength=num_classes)
            for class_id in range(num_classes):
                class_final_groups = final_groups[final_labels == class_id]
                per_segment_counts = Counter(class_final_groups.tolist())
                min_segment_sequences = min(per_segment_counts.values()) if per_segment_counts else 0
                if task == "hapt12":
                    raw_activity_id = class_id + 1
                    is_transition = raw_activity_id >= 7
                    low_threshold = LOW_SUPPORT_THRESHOLD
                else:
                    raw_activity_id = class_id
                    is_transition = class_id == 1
                    low_threshold = BINARY_POSITIVE_WEAK_THRESHOLD if class_id == 1 else LOW_SUPPORT_THRESHOLD
                support = int(sequence_support[class_id])
                rows.append(
                    {
                        "task": task,
                        "context_len": context_len,
                        "split": split,
                        "class_id": class_id,
                        "activity_id": raw_activity_id,
                        "class_name": class_names[class_id],
                        "is_transition": bool(is_transition),
                        "window_support": int(window_support[class_id]),
                        "sequence_support": support,
                        "num_segments": int(len(np.unique(groups[y == class_id]))),
                        "min_segment_sequences": int(min_segment_sequences),
                        "low_support_threshold": int(low_threshold),
                        "low_support_flag": bool(support < low_threshold),
                        "zero_support_flag": bool(support == 0),
                    }
                )
    return pd.DataFrame(rows)


def hapt12_context_is_supported(audit: pd.DataFrame, context_len: int) -> bool:
    subset = audit[
        audit["context_len"].astype(int).eq(int(context_len))
        & audit["split"].astype(str).eq("test")
        & audit["is_transition"].astype(bool)
    ]
    if subset.empty:
        return False
    return not bool(subset["low_support_flag"].astype(bool).any() or subset["zero_support_flag"].astype(bool).any())


def binary_context_has_positive_support(audit: pd.DataFrame, context_len: int) -> bool:
    subset = audit[
        audit["context_len"].astype(int).eq(int(context_len))
        & audit["split"].astype(str).eq("test")
        & audit["class_id"].astype(int).eq(1)
    ]
    if subset.empty:
        return False
    return int(subset["sequence_support"].iloc[0]) > 0


def audit_status_for(task: str, context_len: int, hapt12_audit: pd.DataFrame, binary_audit: pd.DataFrame) -> str:
    if task == "hapt12":
        return "support_ok" if hapt12_context_is_supported(hapt12_audit, context_len) else "low_transition_support"
    subset = binary_audit[
        binary_audit["context_len"].astype(int).eq(int(context_len))
        & binary_audit["split"].astype(str).eq("test")
        & binary_audit["class_id"].astype(int).eq(1)
    ]
    if subset.empty or int(subset["sequence_support"].iloc[0]) == 0:
        return "zero_transition_support"
    if bool(subset["low_support_flag"].iloc[0]):
        return "weak_transition_support"
    return "support_ok"


def binary_transition_metrics(matrix: np.ndarray) -> dict[str, float | int]:
    if matrix.shape != (2, 2):
        raise ValueError(f"Expected 2x2 binary confusion matrix, got {matrix.shape}")
    tp = float(matrix[1, 1])
    fp = float(matrix[0, 1])
    fn = float(matrix[1, 0])
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        "transition_precision": precision,
        "transition_recall": recall,
        "transition_f1": f1,
        "transition_support": int(matrix[1, :].sum()),
    }


def safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def output_path_for(task: str, smoke_test: bool) -> Path:
    if task == "hapt12":
        return HAPT12_SMOKE_PATH if smoke_test else HAPT12_OUTPUT_PATH
    return BINARY_SMOKE_PATH if smoke_test else BINARY_OUTPUT_PATH


def budget_for_run(
    task: str,
    context_len: int,
    model_label: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = make_config(task, context_len, model_label, seed, args)
    return training_metadata(config)


def training_metadata(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    context_len = int(config.get("dataset", {}).get("context_len", 0))
    max_epochs = int(training.get("epochs", 0))
    patience = int(training.get("patience", 0))
    batch_size = int(training.get("batch_size", 0))
    learning_rate = float(training.get("learning_rate", 0.0))
    weight_decay = float(training.get("weight_decay", 0.0))
    return {
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "training_budget": f"fixedK{context_len}_e{max_epochs}_p{patience}_b{batch_size}",
    }


def row_exists(
    path: Path,
    task: str,
    context_len: int,
    model_label: str,
    seed: int,
    training_budget: str | None = None,
) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    required = {"task", "context_len", "model", "seed"}
    if rows.empty or not required.issubset(rows.columns):
        return False
    mask = (
        rows["task"].astype(str).eq(task)
        & rows["context_len"].astype(int).eq(int(context_len))
        & rows["model"].astype(str).eq(model_label)
        & rows["seed"].astype(int).eq(int(seed))
    )
    if training_budget is not None:
        if "training_budget" not in rows.columns:
            return False
        mask &= rows["training_budget"].astype(str).eq(str(training_budget))
    return bool(mask.any())


if __name__ == "__main__":
    main()
