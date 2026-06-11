from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.ucihar import create_ucihar_dataloaders
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
DEFAULT_MODELS = ["cnn1d", "gru", "lif_snn", "cmg_lif"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UCI-HAR baselines.")
    parser.add_argument("--config", default="configs/ucihar_main.yaml")
    parser.add_argument(
        "--model",
        default="all",
        help=(
            "One of cnn1d, gru, ms_cnn1d, window_gru, lif_snn, ms_lif_snn, "
            "cmg_lif, cmg_lif_lite, ms_cmg_lif, or all."
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--context_len", type=int)
    parser.add_argument("--smoke_test", action="store_true")
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


def selected_models(model_arg: str) -> list[str]:
    if model_arg == "all":
        return DEFAULT_MODELS
    normalized = MODEL_ALIASES.get(model_arg.lower())
    if normalized is None:
        raise ValueError(f"Unknown model: {model_arg}")
    return [normalized]


def run_one(model_name: str, config: dict[str, Any], smoke_test: bool) -> dict[str, Any]:
    set_seed(int(config.get("seed", 0)))
    loaders, meta = create_ucihar_dataloaders(config, model_name=model_name, smoke_test=smoke_test)
    device = get_device(config.get("device", "auto"))
    model = build_model(model_name, meta.num_channels, meta.num_classes, config.get("model", {}))
    run_name = f"ucihar_{model_name}_k{meta.context_len}_seed{config.get('seed', 0)}"
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
        "dataset": "UCI-HAR",
        "model": model_name,
        "seed": int(config.get("seed", 0)),
        "context_len": meta.context_len,
        "synthetic_data": meta.synthetic,
        "smoke_test": smoke_test,
        "params": count_parameters(model),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "loss": metrics["loss"],
        "spike_rate": metrics.get("spike_rate", 0.0),
        "target_mode": config.get("training", {}).get("target_mode", "all"),
        "best_epoch": metrics["best_epoch"],
        "best_val_macro_f1": metrics["best_val_macro_f1"],
        "checkpoint": metrics["checkpoint"],
        "epoch_log": metrics["epoch_log"],
        "confusion_matrix_path": metrics.get("confusion_matrix_path", ""),
    }
    result_path = Path(config.get("results", {}).get("dir", "results")) / "ucihar_main_results.csv"
    append_csv_row(result_path, row)
    print(f"Saved row to {result_path}")
    print(row)
    return row


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    for model_name in selected_models(args.model):
        run_one(model_name, config, args.smoke_test)


if __name__ == "__main__":
    main()
