from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.checkpoints import find_checkpoint, load_model_from_checkpoint
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, count_parameters, get_device, load_config, set_seed


DEFAULT_MODELS = [
    "cnn1d",
    "gru",
    "ms_cnn1d",
    "window_gru",
    "lif_snn",
    "cmg_lif_lite",
    "ms_lif_snn",
    "ms_cmg_lif",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UCI-HAR checkpoint robustness.")
    parser.add_argument("--config", default="configs/ucihar_k8_last.yaml")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default="all", help="Comma-separated list or 'all'.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_models(model_arg: str) -> list[str]:
    if model_arg.lower() == "all":
        return DEFAULT_MODELS
    return [item.strip() for item in model_arg.split(",") if item.strip()]


def perturbations() -> list[tuple[str, str, Any, dict[str, Any]]]:
    return [
        ("clean", "0", "0", {}),
        ("gaussian_noise", "0.05", "0.05", {"noise_std": 0.05}),
        ("gaussian_noise", "0.10", "0.10", {"noise_std": 0.10}),
        ("gaussian_noise", "0.20", "0.20", {"noise_std": 0.20}),
        ("channel_dropout", "0.10", "0.10", {"channel_dropout_prob": 0.10}),
        ("channel_dropout", "0.20", "0.20", {"channel_dropout_prob": 0.20}),
        ("channel_dropout", "0.30", "0.30", {"channel_dropout_prob": 0.30}),
        ("modality_dropout", "acc", "acc", {"modality_dropout": "accelerometer"}),
        ("modality_dropout", "gyro", "gyro", {"modality_dropout": "gyroscope"}),
    ]


def row_exists(
    result_path: Path,
    model: str,
    seed: int,
    context_len: int,
    target_mode: str,
    perturbation_type: str,
    perturbation_level: str,
) -> bool:
    if not result_path.exists():
        return False
    rows = pd.read_csv(result_path)
    if rows.empty:
        return False
    mask = (
        (rows["model"] == model)
        & (rows["seed"].astype(int) == int(seed))
        & (rows["context_len"].astype(int) == int(context_len))
        & (rows["target_mode"].astype(str) == str(target_mode))
        & (rows["perturbation_type"].astype(str) == str(perturbation_type))
        & (rows["perturbation_level"].astype(str) == str(perturbation_level))
    )
    return bool(mask.any())


def main() -> None:
    args = parse_args()
    config = copy.deepcopy(load_config(args.config))
    config["seed"] = int(args.seed)
    set_seed(int(args.seed))

    results_dir = Path(args.results_dir)
    result_path = results_dir / "ucihar_robustness_suite.csv"
    context_len = int(config.get("dataset", {}).get("context_len", 8))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    device = get_device(config.get("device", "auto"))

    for model_name in selected_models(args.models):
        checkpoint_path = find_checkpoint(results_dir, model_name, context_len, target_mode, int(args.seed))
        model = None
        params = 0

        for perturbation_type, perturbation_level, raw_level, perturbation_cfg in perturbations():
            if not args.force and row_exists(
                result_path,
                model_name,
                int(args.seed),
                context_len,
                target_mode,
                perturbation_type,
                perturbation_level,
            ):
                print(f"Skipping existing robustness row: {model_name} {perturbation_type} {perturbation_level}")
                continue

            loaders, meta = create_ucihar_dataloaders(
                config,
                model_name=model_name,
                smoke_test=False,
                perturbation=perturbation_cfg,
            )
            if model is None:
                model = load_model_from_checkpoint(
                    model_name,
                    config,
                    meta.num_channels,
                    meta.num_classes,
                    checkpoint_path,
                    device,
                )
                params = count_parameters(model)

            trainer = Trainer(
                model,
                loaders,
                config,
                device,
                run_name=f"ucihar_robust_{model_name}_{perturbation_type}_{perturbation_level}",
                results_dir=results_dir,
                num_classes=meta.num_classes,
            )
            metrics = trainer.evaluate("test")
            row = {
                "dataset": "UCI-HAR",
                "model": model_name,
                "seed": int(args.seed),
                "context_len": meta.context_len,
                "target_mode": target_mode,
                "perturbation_type": perturbation_type,
                "perturbation_level": perturbation_level,
                "synthetic_data": meta.synthetic,
                "checkpoint": str(checkpoint_path),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "loss": metrics["loss"],
                "spike_rate": metrics.get("spike_rate", 0.0),
                "params": params,
            }
            append_csv_row(result_path, row)
            print(row)


if __name__ == "__main__":
    main()
