from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.hapt import create_hapt_dataloaders
from src.datasets.mhealth import create_mhealth_dataloaders
from src.datasets.pamap2 import create_pamap2_dataloaders
from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed

V3_DIR = Path("results/final_paper_v3")
ARTIFACT_DIR = V3_DIR / "artifacts"
RAW_PATH = V3_DIR / "main_results_raw.csv"
SMOKE_PATH = V3_DIR / "smoke_results.csv"

DATASETS: dict[str, dict[str, Any]] = {
    "ucihar": {
        "display": "UCI-HAR",
        "task": "ucihar",
        "config": "configs/ucihar_ms_tcn_snn.yaml",
        "loader": create_ucihar_dataloaders,
    },
    "hapt6": {
        "display": "HAPT-6",
        "task": "hapt6",
        "config": "configs/hapt6_ms_tcn_snn.yaml",
        "loader": create_hapt_dataloaders,
    },
    "pamap2": {
        "display": "PAMAP2",
        "task": "pamap2",
        "config": "configs/pamap2_k8_last.yaml",
        "loader": create_pamap2_dataloaders,
    },
    "mhealth": {
        "display": "MHEALTH",
        "task": "mhealth",
        "config": "configs/mhealth_k8_last.yaml",
        "loader": create_mhealth_dataloaders,
    },
}

MAIN_MODELS = ["cnn1d", "window_gru", "ms_lif_snn", "ms_lif_snn_wide", "ms_ann_tcn", "ms_lif_tcn"]
DIAGNOSTIC_MODELS = ["ms_lif_tcn_plus"]
NON_SPIKING_MODELS = {"cnn1d", "window_gru", "ms_ann_tcn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final_paper_v3 controlled experiments.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=sorted(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--models", nargs="+", default=["main"], help="Model names or groups: main, diagnostic, all.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    V3_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SMOKE_PATH if args.smoke_test else RAW_PATH
    models = selected_models(args.models)
    for dataset_key in args.datasets:
        for seed in args.seeds:
            for model_label in models:
                if not args.force and row_exists(output_path, dataset_key, model_label, seed):
                    print(f"Skipping existing v3 row: {dataset_key} {model_label} seed{seed}")
                    continue
                row = run_one(dataset_key, model_label, seed, args)
                append_csv_row(output_path, row)
                print(f"Saved v3 row: {dataset_key} {model_label} seed{seed} -> {row['macro_f1']:.4f}")


def selected_models(values: list[str]) -> list[str]:
    selected: list[str] = []
    for value in values:
        normalized = value.lower().replace("-", "_")
        if normalized == "main":
            selected.extend(MAIN_MODELS)
        elif normalized == "diagnostic":
            selected.extend(DIAGNOSTIC_MODELS)
        elif normalized == "all":
            selected.extend(MAIN_MODELS + DIAGNOSTIC_MODELS)
        elif normalized in MAIN_MODELS or normalized in DIAGNOSTIC_MODELS:
            selected.append(normalized)
        else:
            raise ValueError(f"Unknown v3 model/group: {value}")
    deduped: list[str] = []
    for model in selected:
        if model not in deduped:
            deduped.append(model)
    return deduped


def run_one(dataset_key: str, model_label: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    spec = DATASETS[dataset_key]
    config = make_config(spec["config"], model_label, seed, args)
    set_seed(seed)
    loader_fn: Callable[..., Any] = spec["loader"]
    build_name = build_name_for(model_label)
    loaders, meta = loader_fn(config, model_name=build_name, smoke_test=args.smoke_test)
    device = get_device(config.get("device", "auto"))
    model = build_model(build_name, meta.num_channels, meta.num_classes, config.get("model", {}))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    suffix = "_smoke" if args.smoke_test else ""
    run_name = f"v3_{dataset_key}_{model_label}_k{meta.context_len}_{target_mode}_seed{seed}{suffix}"
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
    return {
        "dataset": spec["display"],
        "dataset_key": dataset_key,
        "task": spec["task"],
        "model": model_label,
        "build_model": build_name,
        "seed": int(seed),
        "context_len": int(meta.context_len),
        "window_size": int(meta.window_size),
        "num_channels": int(meta.num_channels),
        "num_classes": int(meta.num_classes),
        "target_mode": target_mode,
        "synthetic_data": bool(meta.synthetic),
        "smoke_test": bool(args.smoke_test),
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


def write_runtime_stats(config: dict[str, Any], run_name: str) -> str:
    preprocessing = config.get("_dataset_runtime", {}).get("preprocessing")
    if not preprocessing:
        return ""
    path = ARTIFACT_DIR / f"{run_name}_normalization_stats.json"
    path.write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    return str(path)


def make_config(config_path: str, model_label: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(load_config(config_path))
    config["seed"] = int(seed)
    config.setdefault("results", {})["dir"] = str(ARTIFACT_DIR)
    model_cfg = config.setdefault("model", {})
    if model_label in {"cnn1d", "window_gru", "ms_lif_snn"}:
        model_cfg["tcn_layers"] = int(model_cfg.get("tcn_layers", 2))
    if model_label == "ms_lif_snn_wide":
        model_cfg["hidden_dim"] = 224
        model_cfg["branch_dim"] = 64
    if model_label == "ms_lif_tcn_plus":
        model_cfg["attention_hidden_dim"] = 64
        training = config.setdefault("training", {})
        supcon = training.setdefault("aux_loss", {}).setdefault("supervised_contrastive", {})
        supcon.update({"enabled": True, "weight": 0.10, "temperature": 0.2})
    if args.smoke_test:
        training = config.setdefault("training", {})
        training["epochs"] = 1
        config.setdefault("dataset", {})["smoke_max_train_sequences"] = min(int(config["dataset"].get("smoke_max_train_sequences", 64)), 64)
        config.setdefault("dataset", {})["smoke_max_val_sequences"] = min(int(config["dataset"].get("smoke_max_val_sequences", 32)), 32)
        config.setdefault("dataset", {})["smoke_max_test_sequences"] = min(int(config["dataset"].get("smoke_max_test_sequences", 32)), 32)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.patience is not None:
        config.setdefault("training", {})["patience"] = int(args.patience)
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(args.batch_size)
    return config


def build_name_for(model_label: str) -> str:
    if model_label == "ms_lif_snn_wide":
        return "ms_lif_snn"
    if model_label == "ms_lif_tcn_plus":
        return "ms_lif_tcn_attn"
    return model_label


def row_exists(path: Path, dataset_key: str, model_label: str, seed: int) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    if rows.empty or not {"dataset_key", "model", "seed"}.issubset(rows.columns):
        return False
    return bool(
        (
            rows["dataset_key"].astype(str).eq(dataset_key)
            & rows["model"].astype(str).eq(model_label)
            & rows["seed"].astype(int).eq(int(seed))
        ).any()
    )


if __name__ == "__main__":
    main()
