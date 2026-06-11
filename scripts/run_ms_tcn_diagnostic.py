from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.hapt import create_hapt_dataloaders
from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed


DATASET_CONFIGS = {
    "ucihar": "configs/ucihar_ms_tcn_snn.yaml",
    "hapt6": "configs/hapt6_ms_tcn_snn.yaml",
}
MODEL_ALIASES = {
    "ms_lif_tcn": "ms_lif_tcn",
    "ms_lif_tcn_snn": "ms_lif_tcn",
    "ms_cmg_tcn": "ms_cmg_tcn",
    "ms_cmg_tcn_snn": "ms_cmg_tcn",
}
DEFAULT_MODELS = ["ms_lif_tcn", "ms_cmg_tcn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seed-42 MS-TCN-SNN diagnostic experiments.")
    parser.add_argument("--datasets", nargs="+", default=["ucihar", "hapt6"], choices=["ucihar", "hapt6"])
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", default="results/ms_tcn_seed42_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    for dataset_key in args.datasets:
        config = load_config(DATASET_CONFIGS[dataset_key])
        config["seed"] = int(args.seed)
        if args.epochs is not None:
            config.setdefault("training", {})["epochs"] = int(args.epochs)
        if args.patience is not None:
            config.setdefault("training", {})["patience"] = int(args.patience)
        if args.batch_size is not None:
            config.setdefault("training", {})["batch_size"] = int(args.batch_size)
        for model_name in selected_models(args.models):
            if not args.force and row_exists(output, dataset_key, model_name, int(args.seed)):
                print(f"Skipping existing MS-TCN row: {dataset_key} {model_name} seed{args.seed}")
                continue
            row = run_one(dataset_key, model_name, copy.deepcopy(config))
            append_csv_row(output, row)
            print(f"Saved row to {output}")
            print(row)


def selected_models(model_args: list[str]) -> list[str]:
    if len(model_args) == 1 and model_args[0].lower() == "all":
        return DEFAULT_MODELS
    selected = []
    for model_arg in model_args:
        normalized = MODEL_ALIASES.get(model_arg.lower())
        if normalized is None:
            raise ValueError(f"Unknown model: {model_arg}")
        selected.append(normalized)
    return selected


def run_one(dataset_key: str, model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get("seed", 42)))
    if dataset_key == "ucihar":
        loaders, meta = create_ucihar_dataloaders(config, model_name=model_name, smoke_test=False)
        dataset_label = "UCI-HAR"
        task = "ucihar"
    elif dataset_key == "hapt6":
        loaders, meta = create_hapt_dataloaders(config, model_name=model_name, smoke_test=False)
        dataset_label = "HAPT"
        task = "hapt6"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_key}")
    device = get_device(config.get("device", "auto"))
    model = build_model(model_name, meta.num_channels, meta.num_classes, config.get("model", {}))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    run_name = f"{dataset_key}_{model_name}_k{meta.context_len}_{target_mode}_seed{config.get('seed', 42)}"
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
        "dataset": dataset_label,
        "dataset_key": dataset_key,
        "task": task,
        "model": model_name,
        "seed": int(config.get("seed", 42)),
        "context_len": meta.context_len,
        "target_mode": target_mode,
        "synthetic_data": meta.synthetic,
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


def row_exists(path: Path, dataset_key: str, model: str, seed: int) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    if rows.empty:
        return False
    required = {"dataset_key", "model", "seed"}
    if not required.issubset(rows.columns):
        return False
    return bool(
        (
            rows["dataset_key"].astype(str).eq(str(dataset_key))
            & rows["model"].astype(str).eq(str(model))
            & rows["seed"].astype(int).eq(int(seed))
        ).any()
    )


if __name__ == "__main__":
    main()

