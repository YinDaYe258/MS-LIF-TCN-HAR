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
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed

MODEL_ALIASES = {
    "cnn1d": "cnn1d",
    "cnn": "cnn1d",
    "gru": "gru",
    "ms_cnn": "ms_cnn1d",
    "ms_cnn1d": "ms_cnn1d",
    "window_gru": "window_gru",
    "lif": "lif_snn",
    "lif_snn": "lif_snn",
    "ms_lif": "ms_lif_snn",
    "ms_lif_snn": "ms_lif_snn",
    "cmg_lif": "cmg_lif",
    "cmg_lif_snn": "cmg_lif",
    "cmg_lif_lite": "cmg_lif_lite",
    "cmg_lif_lite_snn": "cmg_lif_lite",
    "ms_cmg_lif": "ms_cmg_lif",
    "ms_cmg_lif_snn": "ms_cmg_lif",
}
DEFAULT_MODELS = ["cnn1d", "gru", "lif_snn", "cmg_lif_lite", "ms_lif_snn", "ms_cmg_lif"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HAPT baselines from raw inertial windows.")
    parser.add_argument("--config", default="configs/hapt_k8_last.yaml")
    parser.add_argument("--model", default=None, help="Backward-compatible single model argument.")
    parser.add_argument("--models", nargs="+", default=None, help="Model list or all.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--context_len", type=int)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.context_len is not None:
        config.setdefault("dataset", {})["context_len"] = args.context_len
    if args.smoke_test:
        config.setdefault("training", {})["epochs"] = min(int(config["training"].get("epochs", 1)), 1)
    return config


def selected_models(args: argparse.Namespace) -> list[str]:
    model_args = args.models if args.models is not None else [args.model or "all"]
    if len(model_args) == 1 and model_args[0] == "all":
        return DEFAULT_MODELS
    selected = []
    for model_arg in model_args:
        normalized = MODEL_ALIASES.get(model_arg.lower())
        if normalized is None:
            raise ValueError(f"Unknown model: {model_arg}")
        selected.append(normalized)
    return selected


def result_path_for(config: dict[str, Any], smoke_test: bool) -> Path:
    results_dir = Path(config.get("results", {}).get("dir", "results"))
    if smoke_test:
        return results_dir / "hapt_smoke_results.csv"
    task = str(config.get("dataset", {}).get("task", config.get("dataset", {}).get("name", "hapt12")))
    context_len = int(config.get("dataset", {}).get("context_len", 1))
    seed = int(config.get("seed", 0))
    if task == "hapt12":
        return results_dir / f"{task}_k{context_len}_seed{seed}_results.csv"
    return results_dir / f"{task}_seed{seed}_results.csv"


def row_already_exists(path: Path, model_name: str, config: dict[str, Any], smoke_test: bool) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    if rows.empty:
        return False
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    task = str(dataset_cfg.get("task", dataset_cfg.get("name", "hapt12")))
    required = {"model", "seed", "context_len", "target_mode", "task", "smoke_test"}
    if not required.issubset(rows.columns):
        return False
    mask = (
        (rows["model"].astype(str) == model_name)
        & (rows["seed"].astype(int) == int(config.get("seed", 0)))
        & (rows["context_len"].astype(int) == int(dataset_cfg.get("context_len", 1)))
        & (rows["target_mode"].astype(str) == str(training_cfg.get("target_mode", "last")))
        & (rows["task"].astype(str) == task)
        & (rows["smoke_test"].astype(str).str.lower() == str(bool(smoke_test)).lower())
    )
    return bool(mask.any())


def run_one(model_name: str, config: dict[str, Any], smoke_test: bool) -> dict[str, Any]:
    set_seed(int(config.get("seed", 0)))
    loaders, meta = create_hapt_dataloaders(config, model_name=model_name, smoke_test=smoke_test)
    device = get_device(config.get("device", "auto"))
    model = build_model(model_name, meta.num_channels, meta.num_classes, config.get("model", {}))
    target_mode = config.get("training", {}).get("target_mode", "last")
    task = str(config.get("dataset", {}).get("task", config.get("dataset", {}).get("name", "hapt12")))
    run_name = f"hapt_{task}_{model_name}_k{meta.context_len}_{target_mode}_seed{config.get('seed', 0)}"
    if smoke_test:
        run_name += "_smoke"
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
    row = {
        "dataset": "HAPT",
        "task": task,
        "model": model_name,
        "seed": int(config.get("seed", 0)),
        "context_len": meta.context_len,
        "window_size": meta.window_size,
        "target_mode": target_mode,
        "sequence_within_segment": bool(config.get("dataset", {}).get("sequence_within_segment", True)),
        "num_classes": meta.num_classes,
        "effective_num_test_classes": effective_num_classes(loaders["test"].dataset, target_mode),
        "synthetic_data": meta.synthetic,
        "smoke_test": smoke_test,
        "params": count_parameters(model),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "loss": metrics["loss"],
        "spike_rate": metrics.get("spike_rate", 0.0),
        "best_epoch": metrics["best_epoch"],
        "best_val_macro_f1": metrics["best_val_macro_f1"],
        "checkpoint": metrics["checkpoint"],
        "epoch_log": metrics["epoch_log"],
        "confusion_matrix_path": metrics.get("confusion_matrix_path", ""),
    }
    result_path = result_path_for(config, smoke_test)
    append_csv_row(result_path, row)
    print(f"Saved row to {result_path}")
    print(row)
    return row


def effective_num_classes(dataset: Any, target_mode: str) -> int:
    labels: list[int] = []
    for indices in getattr(dataset, "indices", []):
        if target_mode == "last":
            labels.append(int(dataset.y[indices[-1]]))
        else:
            labels.extend(int(label) for label in dataset.y[indices])
    return len(set(labels))


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    result_path = result_path_for(config, args.smoke_test)
    for model_name in selected_models(args):
        if not args.force and row_already_exists(result_path, model_name, config, args.smoke_test):
            print(f"Skipping existing HAPT row for {model_name}")
            continue
        run_one(model_name, config, args.smoke_test)


if __name__ == "__main__":
    main()
