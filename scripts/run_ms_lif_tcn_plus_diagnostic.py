from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.hapt import create_hapt_dataloaders
from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed


DATASETS: dict[str, dict[str, Any]] = {
    "ucihar": {
        "config": "configs/ucihar_ms_lif_tcn_attn.yaml",
        "loader": create_ucihar_dataloaders,
        "dataset": "UCI-HAR",
        "task": "ucihar",
    },
    "hapt6": {
        "config": "configs/hapt6_ms_lif_tcn_attn.yaml",
        "loader": create_hapt_dataloaders,
        "dataset": "HAPT",
        "task": "hapt6",
    },
}

VARIANTS: dict[str, dict[str, Any]] = {
    "baseline_ms_lif_tcn": {
        "model": "ms_lif_tcn",
        "attention_enabled": False,
        "supcon_enabled": False,
        "supcon_weight": 0.0,
        "focal_enabled": False,
        "augmentation_enabled": False,
    },
    "attn_ce": {
        "model": "ms_lif_tcn_attn",
        "attention_enabled": True,
        "supcon_enabled": False,
        "supcon_weight": 0.0,
        "focal_enabled": False,
        "augmentation_enabled": False,
    },
    "attn_supcon_0.05": {
        "model": "ms_lif_tcn_attn",
        "attention_enabled": True,
        "supcon_enabled": True,
        "supcon_weight": 0.05,
        "focal_enabled": False,
        "augmentation_enabled": False,
    },
    "attn_supcon_0.1": {
        "model": "ms_lif_tcn_attn",
        "attention_enabled": True,
        "supcon_enabled": True,
        "supcon_weight": 0.10,
        "focal_enabled": False,
        "augmentation_enabled": False,
    },
    "attn_weighted_focal": {
        "model": "ms_lif_tcn_attn",
        "attention_enabled": True,
        "supcon_enabled": False,
        "supcon_weight": 0.0,
        "focal_enabled": True,
        "augmentation_enabled": False,
    },
    "attn_supcon_0.05_aug": {
        "model": "ms_lif_tcn_attn",
        "attention_enabled": True,
        "supcon_enabled": True,
        "supcon_weight": 0.05,
        "focal_enabled": False,
        "augmentation_enabled": True,
    },
    "tcn_supcon_0.05_aug": {
        "model": "ms_lif_tcn",
        "attention_enabled": False,
        "supcon_enabled": True,
        "supcon_weight": 0.05,
        "focal_enabled": False,
        "augmentation_enabled": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MS-LIF-TCN+ seed-42 diagnostics.")
    parser.add_argument("--datasets", nargs="+", default=["ucihar", "hapt6"], choices=sorted(DATASETS))
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--output", default="results/ms_lif_tcn_plus_diagnostic.csv")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_variants(args: list[str]) -> list[str]:
    if len(args) == 1 and args[0].lower() == "all":
        return list(VARIANTS)
    unknown = [name for name in args if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    for dataset_key in args.datasets:
        for variant_name in selected_variants(args.variants):
            if not args.force and row_exists(output, dataset_key, variant_name, int(args.seed)):
                print(f"Skipping existing plus diagnostic row: {dataset_key} {variant_name} seed{args.seed}")
                continue
            if variant_name == "baseline_ms_lif_tcn":
                baseline = load_existing_baseline(dataset_key, int(args.seed))
                if baseline is not None:
                    append_csv_row(output, baseline)
                    print(f"Reused baseline row for {dataset_key} seed{args.seed}")
                    continue
            row = run_one(dataset_key, variant_name, int(args.seed), args)
            append_csv_row(output, row)
            print(f"Saved row to {output}")
            print(row)


def apply_variant_config(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    training = config.setdefault("training", {})
    aux_loss = training.setdefault("aux_loss", {})
    supcon = aux_loss.setdefault("supervised_contrastive", {})
    supcon["enabled"] = bool(variant["supcon_enabled"])
    supcon["weight"] = float(variant["supcon_weight"])
    supcon.setdefault("temperature", 0.2)
    training["loss_type"] = "weighted_focal" if bool(variant["focal_enabled"]) else "ce"
    training["class_weight_source"] = "train_labels"
    training.setdefault("focal_gamma", 2.0)
    aug = training.setdefault("augmentation", {})
    aug["enabled"] = bool(variant["augmentation_enabled"])
    aug.setdefault("jitter_std", 0.03)
    aug.setdefault("scaling_std", 0.10)
    aug.setdefault("channel_dropout_prob", 0.05)
    aug.setdefault("temporal_shift_max", 4)
    return config


def apply_arg_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.patience is not None:
        config.setdefault("training", {})["patience"] = int(args.patience)
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(args.batch_size)
    return config


def run_one(dataset_key: str, variant_name: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    spec = DATASETS[dataset_key]
    variant = VARIANTS[variant_name]
    config = copy.deepcopy(load_config(spec["config"]))
    config["seed"] = int(seed)
    config = apply_variant_config(config, variant)
    config = apply_arg_overrides(config, args)
    set_seed(seed)
    loader_fn: Callable[..., Any] = spec["loader"]
    model_name = str(variant["model"])
    loaders, meta = loader_fn(config, model_name=model_name, smoke_test=False)
    device = get_device(config.get("device", "auto"))
    model = build_model(model_name, meta.num_channels, meta.num_classes, config.get("model", {}))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    run_name = f"{dataset_key}_{variant_name}_k{meta.context_len}_{target_mode}_seed{seed}"
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
    return make_result_row(
        dataset_key,
        spec,
        model_name,
        variant_name,
        seed,
        meta.context_len,
        target_mode,
        bool(meta.synthetic),
        count_parameters(model),
        metrics,
        variant,
        config,
    )


def make_result_row(
    dataset_key: str,
    spec: dict[str, Any],
    model_name: str,
    variant_name: str,
    seed: int,
    context_len: int,
    target_mode: str,
    synthetic_data: bool,
    params: int,
    metrics: dict[str, Any],
    variant: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    training = config.get("training", {})
    supcon = training.get("aux_loss", {}).get("supervised_contrastive", {})
    aug = training.get("augmentation", {})
    return {
        "dataset": spec["dataset"],
        "dataset_key": dataset_key,
        "task": spec["task"],
        "model": model_name,
        "variant": variant_name,
        "seed": int(seed),
        "context_len": int(context_len),
        "target_mode": target_mode,
        "synthetic_data": synthetic_data,
        "attention_enabled": bool(variant["attention_enabled"]),
        "supcon_enabled": bool(supcon.get("enabled", False)),
        "supcon_weight": float(supcon.get("weight", 0.0)),
        "supcon_temperature": float(supcon.get("temperature", 0.2)),
        "focal_enabled": bool(variant["focal_enabled"]),
        "loss_type": str(training.get("loss_type", "ce")),
        "augmentation_enabled": bool(aug.get("enabled", False)),
        "jitter_std": float(aug.get("jitter_std", 0.0)),
        "scaling_std": float(aug.get("scaling_std", 0.0)),
        "channel_dropout_prob": float(aug.get("channel_dropout_prob", 0.0)),
        "temporal_shift_max": int(aug.get("temporal_shift_max", 0)),
        "params": int(params),
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


def row_exists(path: Path, dataset_key: str, variant: str, seed: int) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    required = {"dataset_key", "variant", "seed"}
    if rows.empty or not required.issubset(rows.columns):
        return False
    return bool(
        (
            rows["dataset_key"].astype(str).eq(dataset_key)
            & rows["variant"].astype(str).eq(variant)
            & rows["seed"].astype(int).eq(int(seed))
        ).any()
    )


def load_existing_baseline(dataset_key: str, seed: int) -> dict[str, Any] | None:
    path = Path("results/ms_tcn_multiseed_results.csv")
    if not path.exists():
        return None
    rows = pd.read_csv(path)
    required = {"dataset_key", "model", "seed"}
    if rows.empty or not required.issubset(rows.columns):
        return None
    match = rows[
        rows["dataset_key"].astype(str).eq(dataset_key)
        & rows["model"].astype(str).eq("ms_lif_tcn")
        & rows["seed"].astype(int).eq(int(seed))
    ]
    if match.empty:
        return None
    row = match.iloc[-1].to_dict()
    spec = DATASETS[dataset_key]
    metrics = {
        "accuracy": float(row["accuracy"]),
        "macro_f1": float(row["macro_f1"]),
        "weighted_f1": float(row["weighted_f1"]),
        "balanced_accuracy": float(row.get("balanced_accuracy", 0.0)),
        "loss": float(row.get("loss", 0.0)),
        "spike_rate": float(row.get("spike_rate", 0.0)),
        "best_epoch": int(row.get("best_epoch", 0)),
        "best_val_macro_f1": float(row.get("best_val_macro_f1", 0.0)),
        "checkpoint": row.get("checkpoint", ""),
        "epoch_log": row.get("epoch_log", ""),
        "confusion_matrix_path": row.get("confusion_matrix_path", ""),
    }
    config = load_config(spec["config"])
    return make_result_row(
        dataset_key,
        spec,
        "ms_lif_tcn",
        "baseline_ms_lif_tcn",
        seed,
        int(row["context_len"]),
        str(row.get("target_mode", "last")),
        bool_from_value(row.get("synthetic_data", False)),
        int(row["params"]),
        metrics,
        VARIANTS["baseline_ms_lif_tcn"],
        config,
    )


def bool_from_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    main()
