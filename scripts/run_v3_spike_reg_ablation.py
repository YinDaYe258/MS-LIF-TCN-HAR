from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_final_paper_v3 import DATASETS, build_name_for
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed

V3_DIR = Path("results/final_paper_v3")
ARTIFACT_DIR = V3_DIR / "artifacts"
OUTPUT_PATH = V3_DIR / "spike_reg_ablation.csv"
SMOKE_PATH = V3_DIR / "spike_reg_ablation_smoke.csv"

DEFAULT_DATASETS = ["ucihar", "pamap2"]
DEFAULT_MODELS = ["ms_lif_snn_wide", "ms_lif_tcn"]
DEFAULT_LAMBDAS = [0.0, 1e-5, 1e-4, 1e-3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 spike-regularization ablation.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=sorted(DATASETS))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=sorted(DEFAULT_MODELS))
    parser.add_argument("--spike_reg_lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    V3_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SMOKE_PATH if args.smoke_test else OUTPUT_PATH
    for dataset_key in args.datasets:
        for model_label in args.models:
            for spike_lambda in args.spike_reg_lambdas:
                for seed in args.seeds:
                    budget = budget_for_run(dataset_key, model_label, float(spike_lambda), int(seed), args)
                    if not args.force and row_exists(
                        output_path,
                        dataset_key,
                        model_label,
                        float(spike_lambda),
                        int(seed),
                        budget["training_budget"],
                    ):
                        print(
                            "Skipping existing spike-reg row: "
                            f"{dataset_key} {model_label} lambda={lambda_label(float(spike_lambda))} seed{seed}"
                        )
                        continue
                    row = run_one(dataset_key, model_label, float(spike_lambda), int(seed), args)
                    append_csv_row(output_path, row)
                    print(
                        "Saved spike-reg row: "
                        f"{dataset_key} {model_label} lambda={row['spike_reg_lambda_label']} "
                        f"seed{seed} -> macro_f1={row['macro_f1']:.4f}, spike_rate={row['spike_rate']:.4f}"
                    )


def run_one(dataset_key: str, model_label: str, spike_lambda: float, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    spec = DATASETS[dataset_key]
    config = make_config(str(spec["config"]), model_label, spike_lambda, seed, args)
    set_seed(seed)
    loader_fn: Callable[..., Any] = spec["loader"]
    build_name = build_name_for(model_label)
    loaders, meta = loader_fn(config, model_name=build_name, smoke_test=args.smoke_test)
    device = get_device(config.get("device", "auto"))
    model = build_model(build_name, meta.num_channels, meta.num_classes, config.get("model", {}))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    suffix = "_smoke" if args.smoke_test else ""
    lambda_name = lambda_label(spike_lambda).replace("-", "m")
    run_name = f"v3_spike_reg_{dataset_key}_{model_label}_lam{lambda_name}_k{meta.context_len}_{target_mode}_seed{seed}{suffix}"
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
    model_cfg = config.get("model", {})
    training_meta = training_metadata(config)
    return {
        "dataset": spec["display"],
        "dataset_key": dataset_key,
        "task": spec["task"],
        "ablation": "spike_reg",
        "model": model_label,
        "build_model": build_name,
        "seed": int(seed),
        "spike_reg_lambda": float(spike_lambda),
        "spike_reg_lambda_label": lambda_label(spike_lambda),
        "context_len": int(meta.context_len),
        "window_size": int(meta.window_size),
        "num_channels": int(meta.num_channels),
        "num_classes": int(meta.num_classes),
        "target_mode": target_mode,
        "sequence_protocol": "fixed_k8",
        "synthetic_data": bool(meta.synthetic),
        "smoke_test": bool(args.smoke_test),
        **training_meta,
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "branch_dim": int(model_cfg.get("branch_dim", 32)),
        "tcn_layers": int(model_cfg.get("tcn_layers", 0)),
        "params": int(count_parameters(model)),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
        "loss": float(metrics.get("loss", 0.0)),
        "spike_rate": float(metrics.get("spike_rate", 0.0)),
        "best_epoch": int(metrics.get("best_epoch", 0)),
        "best_val_macro_f1": float(metrics.get("best_val_macro_f1", 0.0)),
        "checkpoint": metrics.get("checkpoint", ""),
        "epoch_log": metrics.get("epoch_log", ""),
        "confusion_matrix_path": metrics.get("confusion_matrix_path", ""),
        "normalization_stats_path": normalization_stats_path,
        "note": "Reduced spike-regularization diagnostic; proxy only, not measured energy.",
    }


def make_config(
    config_path: str,
    model_label: str,
    spike_lambda: float,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = copy.deepcopy(load_config(config_path))
    config["seed"] = int(seed)
    config.setdefault("results", {})["dir"] = str(ARTIFACT_DIR)
    config.setdefault("dataset", {})["context_len"] = 8
    model_cfg = config.setdefault("model", {})
    if model_label == "ms_lif_snn_wide":
        model_cfg["hidden_dim"] = 224
        model_cfg["branch_dim"] = 64
        model_cfg["tcn_layers"] = 0
    elif model_label == "ms_lif_tcn":
        model_cfg["tcn_layers"] = int(model_cfg.get("tcn_layers", 2))
    training = config.setdefault("training", {})
    training["epochs"] = int(args.epochs)
    training["patience"] = int(args.patience)
    training["spike_reg_lambda"] = float(spike_lambda)
    if args.batch_size is not None:
        training["batch_size"] = int(args.batch_size)
    if args.smoke_test:
        training["epochs"] = 1
        training["patience"] = 1
        dataset_cfg = config.setdefault("dataset", {})
        dataset_cfg["smoke_max_train_sequences"] = min(int(dataset_cfg.get("smoke_max_train_sequences", 64)), 64)
        dataset_cfg["smoke_max_val_sequences"] = min(int(dataset_cfg.get("smoke_max_val_sequences", 32)), 32)
        dataset_cfg["smoke_max_test_sequences"] = min(int(dataset_cfg.get("smoke_max_test_sequences", 32)), 32)
    return config


def budget_for_run(
    dataset_key: str,
    model_label: str,
    spike_lambda: float,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    spec = DATASETS[dataset_key]
    config = make_config(str(spec["config"]), model_label, float(spike_lambda), int(seed), args)
    return training_metadata(config)


def training_metadata(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    max_epochs = int(training.get("epochs", 0))
    patience = int(training.get("patience", 0))
    batch_size = int(training.get("batch_size", 0))
    learning_rate = float(training.get("learning_rate", 0.0))
    weight_decay = float(training.get("weight_decay", 0.0))
    return {
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "training_budget": f"fixedK8_e{max_epochs}_p{patience}_b{batch_size}",
    }


def write_runtime_stats(config: dict[str, Any], run_name: str) -> str:
    preprocessing = config.get("_dataset_runtime", {}).get("preprocessing")
    if not preprocessing:
        return ""
    path = ARTIFACT_DIR / f"{run_name}_normalization_stats.json"
    path.write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    return str(path)


def lambda_label(value: float) -> str:
    if math.isclose(float(value), 0.0, abs_tol=1e-15):
        return "0"
    return f"{float(value):.0e}"


def row_exists(
    path: Path,
    dataset_key: str,
    model_label: str,
    spike_lambda: float,
    seed: int,
    training_budget: str | None = None,
) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    required = {"dataset_key", "model", "spike_reg_lambda", "seed"}
    if rows.empty or not required.issubset(rows.columns):
        return False
    lambdas = pd.to_numeric(rows["spike_reg_lambda"], errors="coerce")
    mask = (
        rows["dataset_key"].astype(str).eq(dataset_key)
        & rows["model"].astype(str).eq(model_label)
        & rows["seed"].astype(int).eq(int(seed))
        & (lambdas.sub(float(spike_lambda)).abs() < 1e-15)
    )
    if training_budget is not None:
        if "training_budget" not in rows.columns:
            return False
        mask &= rows["training_budget"].astype(str).eq(str(training_budget))
    return bool(mask.any())


if __name__ == "__main__":
    main()
