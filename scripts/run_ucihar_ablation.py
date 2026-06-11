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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UCI-HAR CMG-LIF ablations.")
    parser.add_argument("--config", default="configs/ucihar_main.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def apply_base_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.smoke_test:
        config.setdefault("training", {})["epochs"] = min(int(config["training"].get("epochs", 1)), 1)
    return config


def ablation_configs(base_config: dict[str, Any], smoke_test: bool) -> list[tuple[str, dict[str, Any]]]:
    context_values = [1, 2] if smoke_test else [1, 2, 4, 8]
    alpha_values = [0.7] if smoke_test else [0.5, 0.7, 0.9]
    variants: list[tuple[str, dict[str, Any]]] = []
    for context_len in context_values:
        cfg = copy.deepcopy(base_config)
        cfg["dataset"]["context_len"] = context_len
        variants.append((f"context_len_{context_len}", cfg))
    for alpha in alpha_values:
        cfg = copy.deepcopy(base_config)
        cfg["model"]["alpha"] = alpha
        variants.append((f"alpha_{alpha}", cfg))

    cfg = copy.deepcopy(base_config)
    cfg["model"]["threshold_modulation"] = False
    variants.append(("threshold_modulation_disabled", cfg))

    cfg = copy.deepcopy(base_config)
    cfg["training"]["spike_reg_lambda"] = 0.0
    variants.append(("spike_regularization_disabled", cfg))

    cfg = copy.deepcopy(base_config)
    cfg["model"]["context_memory"] = False
    variants.append(("context_memory_disabled", cfg))
    return variants


def run_variant(name: str, config: dict[str, Any], smoke_test: bool) -> dict[str, Any]:
    set_seed(int(config.get("seed", 0)))
    loaders, meta = create_ucihar_dataloaders(config, model_name="cmg_lif", smoke_test=smoke_test)
    device = get_device(config.get("device", "auto"))
    model = build_model("cmg_lif", meta.num_channels, meta.num_classes, config.get("model", {}))
    run_name = f"ucihar_cmg_lif_{name}_seed{config.get('seed', 0)}"
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
        "model": "cmg_lif",
        "ablation": name,
        "seed": int(config.get("seed", 0)),
        "context_len": meta.context_len,
        "alpha": float(config.get("model", {}).get("alpha", 0.8)),
        "threshold_modulation": bool(config.get("model", {}).get("threshold_modulation", True)),
        "context_memory": bool(config.get("model", {}).get("context_memory", True)),
        "spike_reg_lambda": float(config.get("training", {}).get("spike_reg_lambda", 0.0)),
        "synthetic_data": meta.synthetic,
        "smoke_test": smoke_test,
        "params": count_parameters(model),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "spike_rate": metrics.get("spike_rate", 0.0),
        "checkpoint": metrics["checkpoint"],
    }
    result_path = Path(config.get("results", {}).get("dir", "results")) / "ucihar_ablation_results.csv"
    append_csv_row(result_path, row)
    print(f"Saved row to {result_path}")
    print(row)
    return row


def main() -> None:
    args = parse_args()
    base_config = apply_base_overrides(load_config(args.config), args)
    for name, config in ablation_configs(base_config, args.smoke_test):
        run_variant(name, config, args.smoke_test)


if __name__ == "__main__":
    main()
