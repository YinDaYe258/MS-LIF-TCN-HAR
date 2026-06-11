from __future__ import annotations

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


def matched_variants(base_config: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    specs = [
        ("lif_snn", 4, "all", 0.0001),
        ("cmg_lif_lite", 4, "all", 0.0001),
        ("lif_snn", 4, "last", 0.0001),
        ("cmg_lif_lite", 4, "last", 0.0001),
        ("lif_snn", 8, "last", 0.0001),
        ("cmg_lif_lite", 8, "last", 0.0001),
    ]
    variants = []
    for model_name, context_len, target_mode, spike_reg_lambda in specs:
        cfg = copy.deepcopy(base_config)
        cfg["seed"] = 42
        cfg.setdefault("dataset", {})["context_len"] = context_len
        cfg.setdefault("training", {})["target_mode"] = target_mode
        cfg.setdefault("training", {})["spike_reg_lambda"] = spike_reg_lambda
        name = f"{model_name}_k{context_len}_{target_mode}_reg{spike_reg_lambda:g}"
        variants.append((name, model_name, cfg))
    return variants


def run_variant(name: str, model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get("seed", 42)))
    loaders, meta = create_ucihar_dataloaders(config, model_name=model_name, smoke_test=False)
    device = get_device(config.get("device", "auto"))
    model = build_model(model_name, meta.num_channels, meta.num_classes, config.get("model", {}))
    run_name = f"ucihar_matched_{name}_seed42"
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
        "seed": 42,
        "context_len": meta.context_len,
        "target_mode": config.get("training", {}).get("target_mode", "all"),
        "spike_reg_lambda": float(config.get("training", {}).get("spike_reg_lambda", 0.0)),
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
    result_path = Path(config.get("results", {}).get("dir", "results")) / "ucihar_matched_protocol_results.csv"
    append_csv_row(result_path, row)
    print(f"Saved row to {result_path}")
    print(row)
    return row


def already_completed(result_path: Path, model_name: str, config: dict[str, Any]) -> bool:
    if not result_path.exists():
        return False
    existing = pd.read_csv(result_path)
    if existing.empty:
        return False
    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    mask = (
        (existing["model"] == model_name)
        & (existing["seed"] == int(config.get("seed", 42)))
        & (existing["context_len"] == int(dataset_cfg.get("context_len", 1)))
        & (existing["target_mode"] == str(training_cfg.get("target_mode", "all")))
        & (existing["spike_reg_lambda"].astype(float) == float(training_cfg.get("spike_reg_lambda", 0.0)))
    )
    return bool(mask.any())


def main() -> None:
    config = load_config("configs/ucihar_cmg_lite.yaml")
    result_path = Path(config.get("results", {}).get("dir", "results")) / "ucihar_matched_protocol_results.csv"
    for name, model_name, variant_config in matched_variants(config):
        if already_completed(result_path, model_name, variant_config):
            print(f"Skipping completed matched run: {name}")
            continue
        run_variant(name, model_name, variant_config)


if __name__ == "__main__":
    main()
