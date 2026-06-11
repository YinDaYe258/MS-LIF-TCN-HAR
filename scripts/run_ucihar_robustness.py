from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, get_device, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained UCI-HAR models under perturbations.")
    parser.add_argument("--config", default="configs/ucihar_main.yaml")
    parser.add_argument("--model", default="cmg_lif")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--context_len", type=int)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.seed is not None:
        config["seed"] = args.seed
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.context_len is not None:
        config.setdefault("dataset", {})["context_len"] = args.context_len
    return config


def perturbations(smoke_test: bool) -> list[tuple[str, dict[str, Any]]]:
    noise_values = [0.05] if smoke_test else [0.05, 0.10, 0.20]
    dropout_values = [0.1] if smoke_test else [0.1, 0.2, 0.3]
    items: list[tuple[str, dict[str, Any]]] = [("clean", {})]
    items.extend((f"gaussian_noise_{std}", {"noise_std": std}) for std in noise_values)
    items.extend((f"channel_dropout_{prob}", {"channel_dropout_prob": prob}) for prob in dropout_values)
    items.append(("drop_accelerometer", {"modality_dropout": "accelerometer"}))
    items.append(("drop_gyroscope", {"modality_dropout": "gyroscope"}))
    return items


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    set_seed(int(config.get("seed", 0)))
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device(config.get("device", "auto"))
    result_path = Path(config.get("results", {}).get("dir", "results")) / "ucihar_robustness_results.csv"

    for perturbation_name, perturbation_cfg in perturbations(args.smoke_test):
        loaders, meta = create_ucihar_dataloaders(
            config,
            model_name=args.model,
            smoke_test=args.smoke_test,
            perturbation=perturbation_cfg,
        )
        model = build_model(args.model, meta.num_channels, meta.num_classes, config.get("model", {})).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer = Trainer(
            model,
            loaders,
            config,
            device,
            run_name=f"ucihar_{args.model}_robustness_{perturbation_name}",
            results_dir=config.get("results", {}).get("dir", "results"),
            num_classes=meta.num_classes,
        )
        metrics = trainer.evaluate("test")
        row = {
            "dataset": "UCI-HAR",
            "model": args.model,
            "seed": int(config.get("seed", 0)),
            "context_len": meta.context_len,
            "synthetic_data": meta.synthetic,
            "smoke_test": args.smoke_test,
            "perturbation": perturbation_name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "spike_rate": metrics.get("spike_rate", 0.0),
            "checkpoint": str(checkpoint_path),
        }
        append_csv_row(result_path, row)
        print(row)


if __name__ == "__main__":
    main()
