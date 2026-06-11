from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.hapt import create_hapt_dataloaders
from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed


DATASETS: dict[str, dict[str, Any]] = {
    "hapt6": {"config": "configs/hapt6_k8_last.yaml", "loader": create_hapt_dataloaders, "dataset": "HAPT", "task": "hapt6"},
    "hapt12_k2": {"config": "configs/hapt12_k2_last.yaml", "loader": create_hapt_dataloaders, "dataset": "HAPT", "task": "hapt12"},
    "ucihar": {"config": "configs/ucihar_k8_last.yaml", "loader": create_ucihar_dataloaders, "dataset": "UCI-HAR", "task": "ucihar"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run class-balanced loss diagnostics.")
    parser.add_argument("--datasets", nargs="+", default=["hapt6", "hapt12_k2"], choices=sorted(DATASETS))
    parser.add_argument("--models", nargs="+", default=["lif_snn", "cmg_lif_lite", "ms_lif_snn", "ms_cmg_lif"])
    parser.add_argument("--loss_types", nargs="+", default=["weighted_ce", "focal", "weighted_focal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, help="Override training epochs for quick diagnostics.")
    parser.add_argument("--patience", type=int, help="Override early-stopping patience.")
    parser.add_argument("--batch_size", type=int, help="Override batch size.")
    parser.add_argument("--output", default="results/class_balanced_seed42_results.csv")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def row_exists(path: Path, dataset_key: str, model: str, seed: int, loss_type: str) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    required = {"dataset_key", "model", "seed", "loss_type"}
    if rows.empty or not required.issubset(rows.columns):
        return False
    return bool(
        (
            rows["dataset_key"].astype(str).eq(dataset_key)
            & rows["model"].astype(str).eq(model)
            & rows["seed"].astype(int).eq(int(seed))
            & rows["loss_type"].astype(str).eq(loss_type)
        ).any()
    )


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.patience is not None:
        config.setdefault("training", {})["patience"] = int(args.patience)
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(args.batch_size)
    return config


def run_one(dataset_key: str, model_name: str, seed: int, loss_type: str, args: argparse.Namespace) -> dict[str, Any]:
    spec = DATASETS[dataset_key]
    config = apply_overrides(copy.deepcopy(load_config(spec["config"])), args)
    config["seed"] = int(seed)
    config.setdefault("training", {})["loss_type"] = loss_type
    config["training"]["class_weight_source"] = "train_labels"
    config["training"].setdefault("focal_gamma", 2.0)
    set_seed(seed)
    loader_fn: Callable[..., Any] = spec["loader"]
    loaders, meta = loader_fn(config, model_name=model_name, smoke_test=False)
    device = get_device(config.get("device", "auto"))
    model = build_model(model_name, meta.num_channels, meta.num_classes, config.get("model", {}))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    run_name = f"{dataset_key}_{model_name}_{loss_type}_k{meta.context_len}_{target_mode}_seed{seed}"
    trainer = Trainer(
        model,
        loaders,
        config,
        device,
        run_name,
        results_dir=config.get("results", {}).get("dir", "results"),
        num_classes=meta.num_classes,
    )
    metrics = trainer.fit()
    return {
        "dataset_key": dataset_key,
        "dataset": spec["dataset"],
        "task": spec["task"],
        "model": model_name,
        "seed": int(seed),
        "context_len": int(meta.context_len),
        "target_mode": target_mode,
        "loss_type": loss_type,
        "focal_gamma": float(config["training"].get("focal_gamma", 2.0)),
        "params": count_parameters(model),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
        "loss": metrics["loss"],
        "spike_rate": metrics.get("spike_rate", 0.0),
        "best_epoch": metrics["best_epoch"],
        "best_val_macro_f1": metrics["best_val_macro_f1"],
        "checkpoint": metrics["checkpoint"],
        "epoch_log": metrics["epoch_log"],
        "confusion_matrix_path": metrics.get("confusion_matrix_path", ""),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    for dataset_key in args.datasets:
        for model in args.models:
            if dataset_key == "hapt12_k2" and model in {"ms_lif_snn", "ms_cmg_lif"}:
                continue
            for loss_type in args.loss_types:
                if not args.force and row_exists(output, dataset_key, model, args.seed, loss_type):
                    print(f"Skipping existing class-balanced row {dataset_key} {model} {loss_type}")
                    continue
                row = run_one(dataset_key, model, args.seed, loss_type, args)
                append_csv_row(output, row)
                print(f"Saved row to {output}")
                print(row)


if __name__ == "__main__":
    main()
