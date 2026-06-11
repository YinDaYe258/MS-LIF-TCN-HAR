from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_final_paper_v3 import DATASETS, NON_SPIKING_MODELS
from src.datasets.sequence_dataset import SequenceWindowDataset
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed

V3_DIR = Path("results/final_paper_v3")
ARTIFACT_DIR = V3_DIR / "artifacts"
OUTPUT_PATH = V3_DIR / "context_length_ablation.csv"
SMOKE_PATH = V3_DIR / "context_length_ablation_smoke.csv"

CONTEXT_MODELS = ["ms_lif_snn", "ms_ann_tcn", "ms_lif_tcn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 context-length ablation.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=sorted(DATASETS))
    parser.add_argument("--models", nargs="+", default=CONTEXT_MODELS, choices=CONTEXT_MODELS)
    parser.add_argument("--context_lens", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument(
        "--aligned_kmax",
        type=int,
        default=0,
        help="Restrict all K values to final-window targets available under this maximum K. Use 8 for the strict v3 K sweep.",
    )
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    V3_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SMOKE_PATH if args.smoke_test else OUTPUT_PATH
    for dataset_key in args.datasets:
        for context_len in args.context_lens:
            for seed in args.seeds:
                for model_label in args.models:
                    budget = budget_for_run(dataset_key, model_label, int(context_len), int(seed), args)
                    if not args.force and row_exists(
                        output_path,
                        dataset_key,
                        model_label,
                        seed,
                        context_len,
                        args.aligned_kmax,
                        budget["training_budget"],
                    ):
                        print(f"Skipping existing context row: {dataset_key} {model_label} k{context_len} seed{seed}")
                        continue
                    row = run_one(dataset_key, model_label, int(context_len), int(seed), args)
                    append_csv_row(output_path, row)
                    print(
                        f"Saved context row: {dataset_key} {model_label} k{context_len} seed{seed} -> {row['macro_f1']:.4f}"
                    )


def run_one(dataset_key: str, model_label: str, context_len: int, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    spec = DATASETS[dataset_key]
    config = make_config(str(spec["config"]), model_label, context_len, seed, args)
    set_seed(seed)
    loader_fn: Callable[..., Any] = spec["loader"]
    loaders, meta = loader_fn(config, model_name=model_label, smoke_test=args.smoke_test)
    if args.aligned_kmax:
        if context_len > args.aligned_kmax:
            raise ValueError(f"context_len={context_len} cannot be aligned to aligned_kmax={args.aligned_kmax}")
        align_loaders_to_final_targets(loaders, int(args.aligned_kmax))
    device = get_device(config.get("device", "auto"))
    model = build_model(model_label, meta.num_channels, meta.num_classes, config.get("model", {}))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    suffix = "_smoke" if args.smoke_test else ""
    align_suffix = f"_alignedK{int(args.aligned_kmax)}" if args.aligned_kmax else ""
    run_name = f"v3_context_{dataset_key}_{model_label}_k{meta.context_len}_{target_mode}_seed{seed}{align_suffix}{suffix}"
    normalization_stats_path = write_runtime_stats(config, run_name)
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
    spike_rate = None if model_label in NON_SPIKING_MODELS else float(metrics.get("spike_rate", 0.0))
    model_cfg = config.get("model", {})
    training_meta = training_metadata(config, int(args.aligned_kmax))
    return {
        "dataset": spec["display"],
        "dataset_key": dataset_key,
        "task": spec["task"],
        "ablation": "context_length",
        "variant": f"k{int(meta.context_len)}",
        "aligned_final_window": bool(args.aligned_kmax),
        "aligned_kmax": int(args.aligned_kmax),
        "sequence_protocol": f"aligned_kmax_{int(args.aligned_kmax)}" if args.aligned_kmax else "native_k",
        "model": model_label,
        "seed": int(seed),
        "context_len": int(meta.context_len),
        "window_size": int(meta.window_size),
        "num_channels": int(meta.num_channels),
        "num_classes": int(meta.num_classes),
        "target_mode": target_mode,
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
        "normalization_stats_path": normalization_stats_path,
    }


def make_config(
    config_path: str,
    model_label: str,
    context_len: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = copy.deepcopy(load_config(config_path))
    config["seed"] = int(seed)
    config.setdefault("results", {})["dir"] = str(ARTIFACT_DIR)
    config.setdefault("dataset", {})["context_len"] = int(context_len)
    model_cfg = config.setdefault("model", {})
    if model_label == "ms_lif_snn":
        model_cfg["tcn_layers"] = int(model_cfg.get("tcn_layers", 2))
    if args.smoke_test:
        training = config.setdefault("training", {})
        training["epochs"] = 1
        dataset_cfg = config.setdefault("dataset", {})
        dataset_cfg["smoke_max_train_sequences"] = min(int(dataset_cfg.get("smoke_max_train_sequences", 64)), 64)
        dataset_cfg["smoke_max_val_sequences"] = min(int(dataset_cfg.get("smoke_max_val_sequences", 32)), 32)
        dataset_cfg["smoke_max_test_sequences"] = min(int(dataset_cfg.get("smoke_max_test_sequences", 32)), 32)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.patience is not None:
        config.setdefault("training", {})["patience"] = int(args.patience)
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(args.batch_size)
    return config


def budget_for_run(
    dataset_key: str,
    model_label: str,
    context_len: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    spec = DATASETS[dataset_key]
    config = make_config(str(spec["config"]), model_label, int(context_len), int(seed), args)
    return training_metadata(config, int(args.aligned_kmax))


def training_metadata(config: dict[str, Any], aligned_kmax: int) -> dict[str, Any]:
    training = config.get("training", {})
    max_epochs = int(training.get("epochs", 0))
    patience = int(training.get("patience", 0))
    batch_size = int(training.get("batch_size", 0))
    learning_rate = float(training.get("learning_rate", 0.0))
    weight_decay = float(training.get("weight_decay", 0.0))
    protocol = f"alignedK{int(aligned_kmax)}" if aligned_kmax else "nativeK"
    return {
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "training_budget": f"{protocol}_e{max_epochs}_p{patience}_b{batch_size}",
    }


def write_runtime_stats(config: dict[str, Any], run_name: str) -> str:
    preprocessing = config.get("_dataset_runtime", {}).get("preprocessing")
    if not preprocessing:
        return ""
    path = ARTIFACT_DIR / f"{run_name}_normalization_stats.json"
    path.write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    return str(path)


def align_loaders_to_final_targets(loaders: dict[str, Any], aligned_kmax: int) -> None:
    for loader in loaders.values():
        dataset = getattr(loader, "dataset", None)
        if isinstance(dataset, SequenceWindowDataset):
            max_sequences = len(dataset.indices)
            dataset.indices = aligned_indices(dataset, aligned_kmax)[:max_sequences]


def aligned_indices(dataset: SequenceWindowDataset, aligned_kmax: int) -> list[Any]:
    context_len = int(dataset.context_len)
    if context_len > aligned_kmax:
        raise ValueError(f"context_len={context_len} cannot exceed aligned_kmax={aligned_kmax}")
    rebuilt: list[Any] = []
    if dataset.group_ids is None:
        for subject_id in sorted(np.unique(dataset.subjects)):
            group_indices = np.flatnonzero(dataset.subjects == subject_id)
            append_aligned_group_indices(rebuilt, group_indices, context_len, aligned_kmax)
    else:
        pairs = np.stack([dataset.subjects, dataset.group_ids], axis=1)
        for subject_id, group_id in np.unique(pairs, axis=0):
            group_indices = np.flatnonzero((dataset.subjects == subject_id) & (dataset.group_ids == group_id))
            append_aligned_group_indices(rebuilt, group_indices, context_len, aligned_kmax)
    return rebuilt


def append_aligned_group_indices(
    output: list[Any],
    group_indices: Any,
    context_len: int,
    aligned_kmax: int,
) -> None:
    if len(group_indices) < aligned_kmax:
        return
    for final_pos in range(aligned_kmax - 1, len(group_indices)):
        start = final_pos - context_len + 1
        output.append(group_indices[start : final_pos + 1])


def row_exists(
    path: Path,
    dataset_key: str,
    model_label: str,
    seed: int,
    context_len: int,
    aligned_kmax: int = 0,
    training_budget: str | None = None,
) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    required = {"dataset_key", "model", "seed", "context_len"}
    if rows.empty or not required.issubset(rows.columns):
        return False
    if "aligned_kmax" not in rows.columns:
        rows["aligned_kmax"] = 0
    rows["aligned_kmax"] = pd.to_numeric(rows["aligned_kmax"], errors="coerce").fillna(0).astype(int)
    mask = (
        rows["dataset_key"].astype(str).eq(dataset_key)
        & rows["model"].astype(str).eq(model_label)
        & rows["seed"].astype(int).eq(int(seed))
        & rows["context_len"].astype(int).eq(int(context_len))
        & rows["aligned_kmax"].eq(int(aligned_kmax))
    )
    if training_budget is not None:
        if "training_budget" not in rows.columns:
            return False
        mask &= rows["training_budget"].astype(str).eq(str(training_budget))
    return bool(mask.any())


if __name__ == "__main__":
    main()
