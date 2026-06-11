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


PROTOCOLS: dict[str, dict[str, Any]] = {
    "ucihar": {
        "config": "configs/ucihar_k8_last.yaml",
        "output": "results/ucihar_param_matched_results.csv",
        "dataset": "UCI-HAR",
        "task": "ucihar",
        "loader": create_ucihar_dataloaders,
    },
    "hapt6": {
        "config": "configs/hapt6_k8_last.yaml",
        "output": "results/hapt6_param_matched_results.csv",
        "dataset": "HAPT",
        "task": "hapt6",
        "loader": create_hapt_dataloaders,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter-matched LIF-SNN controls.")
    parser.add_argument("--protocols", nargs="+", default=["ucihar", "hapt6"], choices=sorted(PROTOCOLS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def row_mask(rows: pd.DataFrame, protocol: str, seed: int, hidden_dim: int) -> pd.Series:
    if rows.empty:
        return pd.Series(False, index=rows.index)
    required = {"dataset", "model", "seed", "context_len", "target_mode", "hidden_dim"}
    if not required.issubset(rows.columns):
        return pd.Series(False, index=rows.index)
    spec = PROTOCOLS[protocol]
    config = load_param_config(spec["config"], seed, hidden_dim)
    return (
        rows["dataset"].astype(str).eq(str(spec["dataset"]))
        & rows["model"].astype(str).eq(f"lif_snn_h{hidden_dim}")
        & rows["seed"].astype(int).eq(int(seed))
        & rows["context_len"].astype(int).eq(int(config.get("dataset", {}).get("context_len", 1)))
        & rows["target_mode"].astype(str).eq(str(config.get("training", {}).get("target_mode", "last")))
        & rows["hidden_dim"].astype(int).eq(int(hidden_dim))
    )


def has_existing_row(path: Path, protocol: str, seed: int, hidden_dim: int) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    return bool(row_mask(rows, protocol, seed, hidden_dim).any())


def remove_existing_row(path: Path, protocol: str, seed: int, hidden_dim: int) -> None:
    if not path.exists():
        return
    rows = pd.read_csv(path)
    rows.loc[~row_mask(rows, protocol, seed, hidden_dim)].to_csv(path, index=False)


def load_param_config(config_path: str | Path, seed: int, hidden_dim: int) -> dict[str, Any]:
    config = copy.deepcopy(load_config(config_path))
    config["seed"] = int(seed)
    config.setdefault("model", {})["hidden_dim"] = int(hidden_dim)
    return config


def run_protocol(protocol: str, seed: int, hidden_dim: int) -> dict[str, Any]:
    spec = PROTOCOLS[protocol]
    config = load_param_config(spec["config"], seed, hidden_dim)
    set_seed(seed)
    loader_fn: Callable[..., Any] = spec["loader"]
    loaders, meta = loader_fn(config, model_name="lif_snn", smoke_test=False)
    device = get_device(config.get("device", "auto"))
    model = build_model("lif_snn", meta.num_channels, meta.num_classes, config.get("model", {}))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    task = str(config.get("dataset", {}).get("task", spec["task"]))
    run_name = f"{protocol}_param_lif_snn_h{hidden_dim}_k{meta.context_len}_{target_mode}_seed{seed}"
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
        "dataset": spec["dataset"],
        "task": task,
        "model": f"lif_snn_h{hidden_dim}",
        "base_model": "lif_snn",
        "variant": f"h{hidden_dim}",
        "seed": int(seed),
        "context_len": int(meta.context_len),
        "window_size": int(meta.window_size),
        "target_mode": target_mode,
        "hidden_dim": int(hidden_dim),
        "num_classes": int(meta.num_classes),
        "synthetic_data": bool(meta.synthetic),
        "params": count_parameters(model),
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
    if protocol == "hapt6":
        row["sequence_within_segment"] = bool(config.get("dataset", {}).get("sequence_within_segment", True))
    return row


def main() -> None:
    args = parse_args()
    for protocol in args.protocols:
        output = Path(PROTOCOLS[protocol]["output"])
        for seed in args.seeds:
            if args.force:
                remove_existing_row(output, protocol, seed, args.hidden_dim)
            elif has_existing_row(output, protocol, seed, args.hidden_dim):
                print(f"Skipping existing {protocol} lif_snn_h{args.hidden_dim} seed={seed}")
                continue
            row = run_protocol(protocol, seed, args.hidden_dim)
            append_csv_row(output, row)
            print(f"Saved {protocol} lif_snn_h{args.hidden_dim} seed={seed} to {output}")
            print(row)


if __name__ == "__main__":
    main()
