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

SOURCE_RESULT_FILES = [
    "ucihar_formal_multiseed_results.csv",
    "ucihar_strong_baseline_results.csv",
    "ucihar_matched_protocol_results.csv",
    "ucihar_main_results.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal multi-seed UCI-HAR experiments.")
    parser.add_argument("--config", default="configs/ucihar_k8_last.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--models", default="all", help="Comma-separated list or 'all'.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_models(model_arg: str) -> list[str]:
    if model_arg.lower() == "all":
        return DEFAULT_MODELS
    return [item.strip() for item in model_arg.split(",") if item.strip()]


def _normal_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _normal_target_mode(value: Any) -> str:
    if value is None or pd.isna(value) or value == "":
        return "all"
    return str(value)


def row_matches(row: pd.Series, model: str, seed: int, context_len: int, target_mode: str) -> bool:
    return (
        str(row.get("model", "")) == model
        and int(row.get("seed", -1)) == int(seed)
        and int(row.get("context_len", -1)) == int(context_len)
        and _normal_target_mode(row.get("target_mode")) == str(target_mode)
        and not _normal_bool(row.get("synthetic_data", False))
        and not _normal_bool(row.get("smoke_test", False))
    )


def existing_row(
    model: str,
    seed: int,
    context_len: int,
    target_mode: str,
    results_dir: Path,
    source_files: list[str] | None = None,
) -> dict[str, Any] | None:
    for file_name in source_files or SOURCE_RESULT_FILES:
        path = results_dir / file_name
        if not path.exists():
            continue
        rows = pd.read_csv(path)
        for _, row in rows.iterrows():
            if row_matches(row, model, seed, context_len, target_mode):
                return row.to_dict()
    return None


def row_already_in_output(result_path: Path, model: str, seed: int, context_len: int, target_mode: str) -> bool:
    if not result_path.exists():
        return False
    rows = pd.read_csv(result_path)
    return any(row_matches(row, model, seed, context_len, target_mode) for _, row in rows.iterrows())


def normalize_for_output(row: dict[str, Any], model: str, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    return {
        "dataset": row.get("dataset", "UCI-HAR"),
        "model": model,
        "seed": int(row.get("seed", seed)),
        "context_len": int(row.get("context_len", dataset_cfg.get("context_len", 8))),
        "target_mode": _normal_target_mode(row.get("target_mode", training_cfg.get("target_mode", "last"))),
        "synthetic_data": _normal_bool(row.get("synthetic_data", False)),
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


def append_unique_formal_row(result_path: Path, row: dict[str, Any]) -> bool:
    if row_already_in_output(
        result_path,
        str(row["model"]),
        int(row["seed"]),
        int(row["context_len"]),
        str(row["target_mode"]),
    ):
        return False
    append_csv_row(result_path, row)
    return True


def run_one(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get("seed", 42)))
    loaders, meta = create_ucihar_dataloaders(config, model_name=model_name, smoke_test=False)
    device = get_device(config.get("device", "auto"))
    model = build_model(model_name, meta.num_channels, meta.num_classes, config.get("model", {}))
    run_name = (
        f"ucihar_formal_{model_name}_k{meta.context_len}_"
        f"{config['training'].get('target_mode', 'last')}_seed{config.get('seed', 42)}"
    )
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
    base_config = load_config(args.config)
    results_dir = Path(base_config.get("results", {}).get("dir", "results"))
    result_path = results_dir / "ucihar_formal_multiseed_results.csv"
    context_len = int(base_config.get("dataset", {}).get("context_len", 8))
    target_mode = str(base_config.get("training", {}).get("target_mode", "last"))

    for seed in args.seeds:
        for model_name in selected_models(args.models):
            if not args.force:
                reused = existing_row(model_name, seed, context_len, target_mode, results_dir)
                if reused is not None:
                    row = normalize_for_output(reused, model_name, seed, base_config)
                    append_unique_formal_row(result_path, row)
                    print(f"Reused existing row for {model_name} seed {seed}")
                    continue

            config = copy.deepcopy(base_config)
            config["seed"] = int(seed)
            row = run_one(model_name, config)
            append_unique_formal_row(result_path, row)
            print(f"Saved row to {result_path}")
            print(row)


if __name__ == "__main__":
    main()
