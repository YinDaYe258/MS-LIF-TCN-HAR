from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.hapt import create_hapt_dataloaders
from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.distillation import DistillationTrainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed


DATASETS: dict[str, dict[str, Any]] = {
    "ucihar": {
        "config": "configs/ucihar_k8_last.yaml",
        "dataset": "UCI-HAR",
        "task": "ucihar",
        "loader": create_ucihar_dataloaders,
        "result_sources": [
            "ucihar_formal_multiseed_results.csv",
            "ucihar_strong_baseline_results.csv",
            "ucihar_main_results.csv",
        ],
    },
    "hapt6": {
        "config": "configs/hapt6_k8_last.yaml",
        "dataset": "HAPT",
        "task": "hapt6",
        "loader": create_hapt_dataloaders,
        "result_sources": [
            "hapt6_multiseed_results.csv",
            "hapt6_seed{seed}_results.csv",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Window-GRU distilled MS-SNN students.")
    parser.add_argument("--datasets", nargs="+", default=["ucihar", "hapt6"], choices=sorted(DATASETS))
    parser.add_argument("--models", nargs="+", default=["ms_lif_snn", "ms_cmg_lif"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[4.0])
    parser.add_argument("--kd_weights", nargs="+", type=float, default=[0.5])
    parser.add_argument("--epochs", type=int, help="Override training epochs for pilot runs.")
    parser.add_argument("--patience", type=int, help="Override early-stopping patience.")
    parser.add_argument("--batch_size", type=int, help="Override batch size.")
    parser.add_argument("--grid_output", default="results/distill_seed42_grid.csv")
    parser.add_argument("--output", default="results/distill_multiseed_results.csv")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_model(model: str) -> str:
    normalized = model.lower().replace("-", "_")
    if normalized in {"ms_lif", "ms_lif_snn"}:
        return "ms_lif_snn"
    if normalized in {"ms_cmg", "ms_cmg_lif", "ms_cmg_lif_snn"}:
        return "ms_cmg_lif"
    raise ValueError(f"Unsupported distilled student: {model}")


def result_exists(path: Path, dataset: str, model: str, seed: int, temperature: float, kd_weight: float) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    if rows.empty:
        return False
    required = {"dataset_key", "base_model", "seed", "temperature", "kd_weight"}
    if not required.issubset(rows.columns):
        return False
    mask = (
        rows["dataset_key"].astype(str).eq(dataset)
        & rows["base_model"].astype(str).eq(model)
        & rows["seed"].astype(int).eq(int(seed))
        & rows["temperature"].astype(float).eq(float(temperature))
        & rows["kd_weight"].astype(float).eq(float(kd_weight))
    )
    return bool(mask.any())


def find_teacher_checkpoint(dataset_key: str, config: dict[str, Any], seed: int) -> Path:
    spec = DATASETS[dataset_key]
    results_dir = Path(config.get("results", {}).get("dir", "results"))
    context_len = int(config.get("dataset", {}).get("context_len", 8))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    task = str(config.get("dataset", {}).get("task", spec["task"]))
    for source in spec["result_sources"]:
        source_path = results_dir / source.format(seed=seed)
        if not source_path.exists():
            continue
        rows = pd.read_csv(source_path)
        if rows.empty:
            continue
        mask = (
            rows["model"].astype(str).eq("window_gru")
            & rows["seed"].astype(int).eq(int(seed))
            & rows["context_len"].astype(int).eq(context_len)
            & rows["target_mode"].astype(str).eq(target_mode)
        )
        if "task" in rows.columns:
            mask &= rows["task"].astype(str).eq(task)
        match = rows[mask]
        if match.empty:
            continue
        checkpoint = Path(str(match.iloc[-1]["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if checkpoint.exists():
            return checkpoint
    raise FileNotFoundError(f"Missing window_gru teacher checkpoint for {dataset_key} seed={seed}")


def load_teacher(checkpoint_path: Path, fallback_config: dict[str, Any], input_channels: int, num_classes: int, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", fallback_config)
    teacher = build_model("window_gru", input_channels, num_classes, config.get("model", {})).to(device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    return teacher


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.patience is not None:
        config.setdefault("training", {})["patience"] = int(args.patience)
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(args.batch_size)
    return config


def run_one(
    dataset_key: str,
    base_model: str,
    seed: int,
    temperature: float,
    kd_weight: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    spec = DATASETS[dataset_key]
    config = apply_overrides(copy.deepcopy(load_config(spec["config"])), args)
    config["seed"] = int(seed)
    config.setdefault("training", {}).setdefault("distillation", {})
    config["training"]["distillation"] = {
        "enabled": True,
        "teacher_model": "window_gru",
        "temperature": float(temperature),
        "kd_weight": float(kd_weight),
    }
    set_seed(seed)
    loader_fn: Callable[..., Any] = spec["loader"]
    loaders, meta = loader_fn(config, model_name=base_model, smoke_test=False)
    device = get_device(config.get("device", "auto"))
    teacher_checkpoint = find_teacher_checkpoint(dataset_key, config, seed)
    teacher = load_teacher(teacher_checkpoint, config, meta.num_channels, meta.num_classes, device)
    student = build_model(base_model, meta.num_channels, meta.num_classes, config.get("model", {}))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    model_name = f"{base_model}_distill"
    run_name = (
        f"{dataset_key}_{model_name}_k{meta.context_len}_{target_mode}_"
        f"t{temperature:g}_kd{kd_weight:g}_seed{seed}"
    ).replace(".", "p")
    trainer = DistillationTrainer(
        student,
        teacher,
        loaders,
        config,
        device,
        run_name,
        results_dir=config.get("results", {}).get("dir", "results"),
        num_classes=meta.num_classes,
    )
    metrics = trainer.fit()
    row = {
        "dataset_key": dataset_key,
        "dataset": spec["dataset"],
        "task": str(config.get("dataset", {}).get("task", spec["task"])),
        "model": model_name,
        "base_model": base_model,
        "teacher_model": "window_gru",
        "teacher_checkpoint": str(teacher_checkpoint),
        "seed": int(seed),
        "context_len": int(meta.context_len),
        "target_mode": target_mode,
        "temperature": float(temperature),
        "kd_weight": float(kd_weight),
        "params": count_parameters(trainer.student),
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
    return row


def main() -> None:
    args = parse_args()
    models = [normalize_model(model) for model in args.models]
    output = Path(args.output)
    grid_output = Path(args.grid_output)
    is_grid = len(args.seeds) == 1 and args.seeds[0] == 42 and (
        len(args.temperatures) > 1 or len(args.kd_weights) > 1
    )
    target_output = grid_output if is_grid else output
    for dataset_key in args.datasets:
        for model in models:
            for seed in args.seeds:
                for temperature in args.temperatures:
                    for kd_weight in args.kd_weights:
                        if not args.force and result_exists(target_output, dataset_key, model, seed, temperature, kd_weight):
                            print(f"Skipping existing distill row {dataset_key} {model} seed={seed} T={temperature} kd={kd_weight}")
                            continue
                        row = run_one(dataset_key, model, seed, temperature, kd_weight, args)
                        append_csv_row(target_output, row)
                        print(f"Saved distillation row to {target_output}")
                        print(row)


if __name__ == "__main__":
    main()
