from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_hapt_baselines import effective_num_classes, run_one
from src.datasets.hapt import create_hapt_dataloaders
from src.training.utils import append_csv_row, load_config


CONFIG_BY_K = {
    1: "configs/hapt_transition_binary_k1_last.yaml",
    2: "configs/hapt_transition_binary_k2_last.yaml",
    4: "configs/hapt_transition_binary_k4_last.yaml",
}
DEFAULT_MODELS = ["cnn1d", "window_gru", "lif_snn", "cmg_lif_lite", "ms_lif_snn", "ms_cmg_lif"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HAPT basic-vs-transition diagnostic.")
    parser.add_argument("--context_lens", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, help="Override training epochs for pilot runs.")
    parser.add_argument("--patience", type=int, help="Override early-stopping patience.")
    parser.add_argument("--batch_size", type=int, help="Override batch size.")
    parser.add_argument("--output", default="results/hapt_transition_binary_results.csv")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def row_exists(path: Path, model: str, seed: int, context_len: int) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    required = {"model", "seed", "context_len", "task"}
    if rows.empty or not required.issubset(rows.columns):
        return False
    return bool(
        (
            rows["model"].astype(str).eq(model)
            & rows["seed"].astype(int).eq(int(seed))
            & rows["context_len"].astype(int).eq(int(context_len))
            & rows["task"].astype(str).eq("transitionbinary")
        ).any()
    )


def transition_recall_precision(confusion_path: str | Path) -> tuple[float, float]:
    matrix = pd.DataFrame(json.loads(Path(confusion_path).read_text(encoding="utf-8"))).values
    support = matrix[1].sum()
    predicted = matrix[:, 1].sum()
    tp = matrix[1, 1]
    recall = float(tp / support) if support > 0 else float("nan")
    precision = float(tp / predicted) if predicted > 0 else float("nan")
    return recall, precision


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.patience is not None:
        config.setdefault("training", {})["patience"] = int(args.patience)
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(args.batch_size)
    return config


def run_with_config(model: str, seed: int, context_len: int, args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(load_config(CONFIG_BY_K[context_len]))
    config = apply_overrides(config, args)
    config["seed"] = int(seed)
    row = run_one(model, config, smoke_test=False)
    row["dataset_key"] = "hapt_transition_binary"
    row["task"] = "transitionbinary"
    loaders, _ = create_hapt_dataloaders(config, model_name=model, smoke_test=False)
    row["effective_num_test_classes"] = effective_num_classes(
        loaders["test"].dataset,
        str(config.get("training", {}).get("target_mode", "last")),
    )
    recall, precision = transition_recall_precision(row["confusion_matrix_path"])
    row["transition_recall"] = recall
    row["transition_precision"] = precision
    return row


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    for context_len in args.context_lens:
        if context_len not in CONFIG_BY_K:
            raise ValueError(f"Unsupported context_len: {context_len}")
        for seed in args.seeds:
            for model in args.models:
                if not args.force and row_exists(output, model, seed, context_len):
                    print(f"Skipping existing transition row {model} K={context_len} seed={seed}")
                    continue
                row = run_with_config(model, seed, context_len, args)
                append_csv_row(output, row)
                print(f"Saved transition row to {output}")
                print(row)


if __name__ == "__main__":
    main()
