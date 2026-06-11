from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.hapt import create_hapt_dataloaders
from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed


DATASET_CONFIGS = {
    "ucihar": "configs/ucihar_ms_tcn_snn.yaml",
    "hapt6": "configs/hapt6_ms_tcn_snn.yaml",
}
OUTPUT = Path("results/final_paper/single_scale_ablation.csv")
TABLE = Path("results/final_paper/table_single_scale_ablation.tex")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal single-scale MS-LIF-TCN ablation.")
    parser.add_argument("--datasets", nargs="+", default=["ucihar", "hapt6"], choices=["ucihar", "hapt6"])
    parser.add_argument("--kernels", nargs="+", type=int, default=[5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for dataset_key in args.datasets:
        full = reusable_full_row(dataset_key, int(args.seed))
        if full is not None and (args.force or not row_exists(output, dataset_key, "full_multi_scale", int(args.seed))):
            append_csv_row(output, full)
            print(f"Reused full MS-LIF-TCN row for {dataset_key}")
        for kernel in args.kernels:
            variant = f"single_scale_k{int(kernel)}"
            if not args.force and row_exists(output, dataset_key, variant, int(args.seed)):
                print(f"Skipping existing single-scale row: {dataset_key} {variant} seed{args.seed}")
                continue
            row = run_single(dataset_key, int(kernel), int(args.seed), args)
            append_csv_row(output, row)
            print(f"Saved {dataset_key} {variant}: macro_f1={row['macro_f1']:.4f}")
    write_table(pd.read_csv(output), TABLE)
    print(f"Wrote {output}")
    print(f"Wrote {TABLE}")


def run_single(dataset_key: str, kernel: int, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(load_config(DATASET_CONFIGS[dataset_key]))
    config["seed"] = int(seed)
    model_cfg = config.setdefault("model", {})
    model_cfg["encoder_mode"] = "single"
    model_cfg["single_kernel_size"] = int(kernel)
    apply_overrides(config, args)
    set_seed(seed)
    if dataset_key == "ucihar":
        loaders, meta = create_ucihar_dataloaders(config, model_name="ms_lif_tcn", smoke_test=False)
        dataset_label = "UCI-HAR"
        task = "ucihar"
    else:
        loaders, meta = create_hapt_dataloaders(config, model_name="ms_lif_tcn", smoke_test=False)
        dataset_label = "HAPT-6"
        task = "hapt6"
    device = get_device(config.get("device", "auto"))
    model = build_model("ms_lif_tcn", meta.num_channels, meta.num_classes, config.get("model", {}))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    run_name = f"{dataset_key}_ms_lif_tcn_single_k{kernel}_k{meta.context_len}_{target_mode}_seed{seed}"
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
        "dataset": dataset_label,
        "dataset_key": dataset_key,
        "task": task,
        "model": "ms_lif_tcn",
        "variant": f"single_scale_k{kernel}",
        "seed": seed,
        "context_len": meta.context_len,
        "target_mode": target_mode,
        "encoder_mode": "single",
        "single_kernel_size": int(kernel),
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
        "status": "available",
        "note": "Single-scale encoder ablation; same TCN and training protocol.",
    }


def reusable_full_row(dataset_key: str, seed: int) -> dict[str, Any] | None:
    path = Path("results/ms_tcn_multiseed_results.csv")
    if not path.exists():
        return None
    rows = pd.read_csv(path)
    match = rows[
        rows["dataset_key"].astype(str).eq(dataset_key)
        & rows["model"].astype(str).eq("ms_lif_tcn")
        & rows["seed"].astype(int).eq(int(seed))
    ]
    if match.empty:
        return None
    row = match.iloc[-1].to_dict()
    row.update(
        {
            "variant": "full_multi_scale",
            "encoder_mode": "multi",
            "single_kernel_size": "",
            "status": "available",
            "note": "Reused full multi-scale MS-LIF-TCN seed42 result.",
        }
    )
    if dataset_key == "hapt6":
        row["dataset"] = "HAPT-6"
    return row


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    training = config.setdefault("training", {})
    if args.epochs is not None:
        training["epochs"] = int(args.epochs)
    if args.patience is not None:
        training["patience"] = int(args.patience)
    if args.batch_size is not None:
        training["batch_size"] = int(args.batch_size)


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


def write_table(rows: pd.DataFrame, output_path: Path) -> None:
    table = rows.copy()
    keep = [
        "dataset",
        "variant",
        "seed",
        "params",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "spike_rate",
        "note",
    ]
    table = table[[col for col in keep if col in table.columns]]
    table.to_latex(output_path, index=False, escape=False, float_format=lambda value: f"{value:.4f}")


if __name__ == "__main__":
    main()
