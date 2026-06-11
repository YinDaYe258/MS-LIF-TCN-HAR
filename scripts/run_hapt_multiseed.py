from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_hapt_baselines import DEFAULT_MODELS, MODEL_ALIASES, apply_overrides, effective_num_classes, run_one
from src.datasets.hapt import create_hapt_dataloaders
from src.training.utils import append_csv_row, load_config


PROTOCOLS = {
    "hapt6": {
        "config": "configs/hapt6_k8_last.yaml",
        "output": "results/hapt6_multiseed_results.csv",
        "models": ["cnn1d", "gru", "lif_snn", "cmg_lif_lite", "ms_lif_snn", "ms_cmg_lif"],
        "seed_result_template": "results/hapt6_seed{seed}_results.csv",
    },
    "hapt12_k2": {
        "config": "configs/hapt12_k2_last.yaml",
        "output": "results/hapt12_k2_multiseed_results.csv",
        "models": ["lif_snn", "cmg_lif_lite"],
        "seed_result_template": "results/hapt12_k2_seed{seed}_results.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal HAPT multi-seed experiments.")
    parser.add_argument("--protocols", nargs="+", default=["hapt6", "hapt12_k2"], choices=sorted(PROTOCOLS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--models", nargs="+", default=None, help="Optional model list shared by selected protocols.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_models(model_args: list[str] | None, default_models: list[str]) -> list[str]:
    if model_args is None:
        return default_models
    if len(model_args) == 1 and model_args[0] == "all":
        return DEFAULT_MODELS
    models: list[str] = []
    for item in model_args:
        model = MODEL_ALIASES.get(item.lower())
        if model is None:
            raise ValueError(f"Unknown HAPT model: {item}")
        models.append(model)
    return models


def main() -> None:
    args = parse_args()
    for protocol_name in args.protocols:
        protocol = PROTOCOLS[protocol_name]
        models = normalize_models(args.models, protocol["models"])
        for seed in args.seeds:
            config = apply_overrides(load_config(protocol["config"]), argparse.Namespace(
                seed=seed,
                epochs=None,
                batch_size=None,
                context_len=None,
                smoke_test=False,
            ))
            for model_name in models:
                output = Path(protocol["output"])
                if args.force:
                    remove_existing(output, model_name, config)
                elif has_row(output, model_name, config):
                    print(f"Skipping existing {protocol_name} {model_name} seed={seed}")
                    continue

                reused = load_existing_seed_row(Path(protocol["seed_result_template"].format(seed=seed)), model_name, config)
                if reused is not None and not args.force:
                    row = enrich_row(reused, config, model_name)
                    append_csv_row(output, row)
                    print(f"Reused {model_name} seed={seed} into {output}")
                    continue

                row = run_one(model_name, config, smoke_test=False)
                row = enrich_row(row, config, model_name)
                append_csv_row(output, row)
                print(f"Saved {model_name} seed={seed} into {output}")


def row_mask(rows: pd.DataFrame, model_name: str, config: dict[str, Any]) -> pd.Series:
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    required = {"model", "seed", "context_len", "target_mode"}
    if not required.issubset(rows.columns):
        return pd.Series(False, index=rows.index)
    task = str(dataset_cfg.get("task", dataset_cfg.get("name", "hapt12")))
    mask = (
        (rows["model"].astype(str) == model_name)
        & (rows["seed"].astype(int) == int(config.get("seed", 0)))
        & (rows["context_len"].astype(int) == int(dataset_cfg.get("context_len", 1)))
        & (rows["target_mode"].astype(str) == str(training_cfg.get("target_mode", "last")))
    )
    if "task" in rows.columns:
        mask &= rows["task"].astype(str).eq(task)
    return mask


def has_row(path: Path, model_name: str, config: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    if rows.empty:
        return False
    return bool(row_mask(rows, model_name, config).any())


def remove_existing(path: Path, model_name: str, config: dict[str, Any]) -> None:
    if not path.exists():
        return
    rows = pd.read_csv(path)
    kept = rows.loc[~row_mask(rows, model_name, config)]
    kept.to_csv(path, index=False)


def load_existing_seed_row(path: Path, model_name: str, config: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows = pd.read_csv(path)
    mask = row_mask(rows, model_name, config)
    if not mask.any():
        return None
    return rows.loc[mask].iloc[-1].to_dict()


def enrich_row(row: dict[str, Any], config: dict[str, Any], model_name: str) -> dict[str, Any]:
    row = copy.deepcopy(row)
    dataset_cfg = config.get("dataset", {})
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    row["sequence_within_segment"] = bool(dataset_cfg.get("sequence_within_segment", True))
    loaders, meta = create_hapt_dataloaders(config, model_name=model_name, smoke_test=False)
    row["num_classes"] = int(meta.num_classes)
    row["effective_num_test_classes"] = effective_num_classes(loaders["test"].dataset, target_mode)
    if "balanced_accuracy" not in row or pd.isna(row["balanced_accuracy"]):
        row["balanced_accuracy"] = balanced_accuracy_from_path(row.get("confusion_matrix_path", ""))
    ordered = [
        "dataset",
        "task",
        "model",
        "seed",
        "context_len",
        "window_size",
        "target_mode",
        "sequence_within_segment",
        "num_classes",
        "effective_num_test_classes",
        "synthetic_data",
        "smoke_test",
        "params",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "loss",
        "spike_rate",
        "best_epoch",
        "best_val_macro_f1",
        "checkpoint",
        "epoch_log",
        "confusion_matrix_path",
    ]
    return {key: row.get(key, "") for key in ordered}


def balanced_accuracy_from_path(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        return float("nan")
    matrix = np.asarray(pd.read_json(path).values, dtype=np.float64)
    support = matrix.sum(axis=1)
    valid = support > 0
    if not valid.any():
        return float("nan")
    recall = np.divide(np.diag(matrix), support, out=np.zeros_like(support), where=support > 0)
    return float(recall[valid].mean())


if __name__ == "__main__":
    main()
