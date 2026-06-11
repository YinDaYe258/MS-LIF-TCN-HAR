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
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed


MODELS = [
    "cnn1d",
    "gru",
    "ms_cnn1d",
    "window_gru",
    "lif_snn",
    "cmg_lif_lite",
    "ms_lif_snn",
    "ms_cmg_lif",
]

SOURCE_RESULT_FILES = [
    "ucihar_strong_baseline_results.csv",
    "ucihar_main_results.csv",
    "ucihar_matched_protocol_results.csv",
    "ucihar_cmg_diagnostic_results.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fair UCI-HAR K=8/last strong baselines.")
    parser.add_argument("--config", default="configs/ucihar_k8_last.yaml")
    parser.add_argument("--force", action="store_true", help="Rerun even if a matching row already exists.")
    return parser.parse_args()


def row_matches(row: pd.Series, model_name: str, config: dict[str, Any]) -> bool:
    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    target_mode = row.get("target_mode", "all")
    return (
        str(row.get("model", "")) == model_name
        and int(row.get("seed", -1)) == int(config.get("seed", 42))
        and int(row.get("context_len", -1)) == int(dataset_cfg.get("context_len", 8))
        and str(target_mode) == str(training_cfg.get("target_mode", "last"))
        and not bool(row.get("synthetic_data", False))
        and not bool(row.get("smoke_test", False))
    )


def existing_row(model_name: str, config: dict[str, Any], results_dir: Path) -> dict[str, Any] | None:
    for file_name in SOURCE_RESULT_FILES:
        path = results_dir / file_name
        if not path.exists():
            continue
        rows = pd.read_csv(path)
        if rows.empty:
            continue
        for _, row in rows.iterrows():
            if row_matches(row, model_name, config):
                return row.to_dict()
    return None


def normalize_for_output(row: dict[str, Any], model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    return {
        "dataset": row.get("dataset", "UCI-HAR"),
        "model": model_name,
        "seed": int(row.get("seed", config.get("seed", 42))),
        "context_len": int(row.get("context_len", dataset_cfg.get("context_len", 8))),
        "target_mode": row.get("target_mode", training_cfg.get("target_mode", "last")),
        "synthetic_data": bool(row.get("synthetic_data", False)),
        "params": int(row.get("params", 0)),
        "accuracy": float(row.get("accuracy", 0.0)),
        "macro_f1": float(row.get("macro_f1", 0.0)),
        "weighted_f1": float(row.get("weighted_f1", 0.0)),
        "loss": float(row.get("loss", 0.0)),
        "spike_rate": float(row.get("spike_rate", 0.0)),
        "best_epoch": int(row.get("best_epoch", 0)),
        "best_val_macro_f1": float(row.get("best_val_macro_f1", 0.0)),
        "checkpoint": row.get("checkpoint", ""),
        "epoch_log": row.get("epoch_log", ""),
        "confusion_matrix_path": row.get("confusion_matrix_path", ""),
    }


def run_one(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get("seed", 42)))
    loaders, meta = create_ucihar_dataloaders(config, model_name=model_name, smoke_test=False)
    device = get_device(config.get("device", "auto"))
    model = build_model(model_name, meta.num_channels, meta.num_classes, config.get("model", {}))
    run_name = f"ucihar_strong_{model_name}_k{meta.context_len}_{config['training'].get('target_mode', 'last')}_seed{config.get('seed', 42)}"
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
        "dataset": "UCI-HAR",
        "model": model_name,
        "seed": int(config.get("seed", 42)),
        "context_len": meta.context_len,
        "target_mode": config.get("training", {}).get("target_mode", "last"),
        "synthetic_data": meta.synthetic,
        "params": count_parameters(model),
        "accuracy": metrics["accuracy"],
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = copy.deepcopy(config)
    config["seed"] = 42
    results_dir = Path(config.get("results", {}).get("dir", "results"))
    result_path = results_dir / "ucihar_strong_baseline_results.csv"

    for model_name in MODELS:
        if not args.force:
            row = existing_row(model_name, config, results_dir)
            if row is not None:
                output_row = normalize_for_output(row, model_name, config)
                if existing_row(model_name, config, result_path.parent) is None or not result_path.exists():
                    append_csv_row(result_path, output_row)
                elif not row_already_in_output(result_path, model_name, config):
                    append_csv_row(result_path, output_row)
                print(f"Reused existing row for {model_name}")
                continue

        output_row = run_one(model_name, config)
        append_csv_row(result_path, output_row)
        print(f"Saved row to {result_path}")
        print(output_row)


def row_already_in_output(result_path: Path, model_name: str, config: dict[str, Any]) -> bool:
    if not result_path.exists():
        return False
    rows = pd.read_csv(result_path)
    return any(row_matches(row, model_name, config) for _, row in rows.iterrows())


if __name__ == "__main__":
    main()
