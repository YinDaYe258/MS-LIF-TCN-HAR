from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_final_paper_v3 import DATASETS
from scripts.run_v3_context_length_ablation import align_loaders_to_final_targets, make_config
from scripts.profile_v3_context_runtime import markdown_table
from src.training.trainer import Trainer, prefix_keys
from src.training.utils import build_model, count_parameters, get_device, set_seed

OUT_DIR = Path("results/final_paper_v3/runtime_profile")
PROFILE_CSV = OUT_DIR / "trainer_runtime_profile.csv"
PROFILE_REPORT = OUT_DIR / "trainer_runtime_profile_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile v3 Trainer runtime stages without writing ablation rows.")
    parser.add_argument("--dataset", default="ucihar", choices=sorted(DATASETS))
    parser.add_argument("--model", default="ms_lif_tcn")
    parser.add_argument("--context_len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aligned_kmax", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row = profile_trainer(args)
    frame = pd.DataFrame([row])
    frame.to_csv(PROFILE_CSV, index=False)
    write_report(frame, args)
    print(f"Wrote trainer runtime profile to {PROFILE_CSV}")
    print(f"Wrote trainer runtime profile report to {PROFILE_REPORT}")


def profile_trainer(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    spec = DATASETS[args.dataset]
    config_args = SimpleNamespace(smoke_test=False, epochs=2, patience=1, batch_size=args.batch_size)
    config = make_config(str(spec["config"]), args.model, int(args.context_len), int(args.seed), config_args)
    if args.device is not None:
        config["device"] = args.device

    loader_fn = spec["loader"]
    build_start = time.perf_counter()
    loaders, meta = loader_fn(config, model_name=args.model, smoke_test=False)
    loader_build_s = time.perf_counter() - build_start
    align_start = time.perf_counter()
    align_loaders_to_final_targets(loaders, int(args.aligned_kmax))
    align_s = time.perf_counter() - align_start

    device = get_device(config.get("device", "auto"))
    model = build_model(args.model, meta.num_channels, meta.num_classes, config.get("model", {}))
    trainer = Trainer(
        model,
        loaders,
        config,
        device,
        f"profile_v3_trainer_{args.dataset}_{args.model}_k{args.context_len}_seed{args.seed}_alignedK{args.aligned_kmax}",
        results_dir=OUT_DIR,
        num_classes=meta.num_classes,
    )
    optimizer = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=float(config.get("training", {}).get("learning_rate", 1e-3)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 1e-4)),
    )

    first_batch = next(iter(loaders["train"]))
    batch_shape = tuple(int(dim) for dim in first_batch["x"].shape)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    train_start = time.perf_counter()
    train_metrics = trainer._run_train_epoch(optimizer)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_epoch_s = time.perf_counter() - train_start

    val_start = time.perf_counter()
    val_metrics = trainer.evaluate("val")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    val_eval_s = time.perf_counter() - val_start

    log_start = time.perf_counter()
    row = {"epoch": 1, **prefix_keys(train_metrics, "train"), **prefix_keys(val_metrics, "val")}
    pd.DataFrame([row]).to_csv(trainer.epoch_log_path, index=False)
    log_write_ms = (time.perf_counter() - log_start) * 1000.0

    ckpt_start = time.perf_counter()
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "config": config,
            "run_name": trainer.run_name,
            "best_epoch": 1,
            "best_val_macro_f1": float(val_metrics["macro_f1"]),
        },
        trainer.checkpoint_path,
    )
    checkpoint_save_ms = (time.perf_counter() - ckpt_start) * 1000.0

    test_start = time.perf_counter()
    test_metrics = trainer.evaluate("test")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_eval_s = time.perf_counter() - test_start

    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
        device_name = torch.cuda.get_device_name(device)
    else:
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0
        device_name = str(device)

    return {
        "dataset": spec["display"],
        "dataset_key": args.dataset,
        "model": args.model,
        "seed": int(args.seed),
        "context_len": int(args.context_len),
        "aligned_kmax": int(args.aligned_kmax),
        "sequence_protocol": f"aligned_kmax_{int(args.aligned_kmax)}",
        "device": str(device),
        "device_name": device_name,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "num_train_sequences": int(len(loaders["train"].dataset)),
        "num_val_sequences": int(len(loaders["val"].dataset)),
        "num_test_sequences": int(len(loaders["test"].dataset)),
        "num_train_batches": int(len(loaders["train"])),
        "num_val_batches": int(len(loaders["val"])),
        "num_test_batches": int(len(loaders["test"])),
        "batch_size": int(args.batch_size),
        "batch_x_shape": str(batch_shape),
        "params": int(count_parameters(trainer.model)),
        "loader_build_time_s": loader_build_s,
        "align_time_s": align_s,
        "train_epoch_time_s": train_epoch_s,
        "train_iter_time_ms_mean_est": train_epoch_s * 1000.0 / max(1, len(loaders["train"])),
        "val_eval_time_s": val_eval_s,
        "val_iter_time_ms_mean_est": val_eval_s * 1000.0 / max(1, len(loaders["val"])),
        "log_write_time_ms": log_write_ms,
        "checkpoint_save_time_ms": checkpoint_save_ms,
        "test_eval_time_s": test_eval_s,
        "test_iter_time_ms_mean_est": test_eval_s * 1000.0 / max(1, len(loaders["test"])),
        "train_macro_f1": float(train_metrics["macro_f1"]),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_confusion_matrix_path": str(test_metrics.get("confusion_matrix_path", "")),
        "gpu_peak_memory_allocated_mb": peak_allocated_mb,
        "gpu_peak_memory_reserved_mb": peak_reserved_mb,
    }


def write_report(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    row = frame.iloc[0]
    lines = [
        "# v3 Trainer Runtime Profile",
        "",
        "This report profiles the full Trainer stages for one context-length run. It does not write context ablation rows.",
        "",
        f"Dataset: `{args.dataset}`",
        f"Model: `{args.model}`",
        f"Context length: `{args.context_len}`",
        f"Seed: `{args.seed}`",
        f"Aligned final-window protocol: `aligned_kmax_{args.aligned_kmax}`",
        "",
        "## Stage Timing",
        "",
        markdown_table(
            frame[
                [
                    "context_len",
                    "device",
                    "batch_x_shape",
                    "num_train_batches",
                    "train_epoch_time_s",
                    "train_iter_time_ms_mean_est",
                    "val_eval_time_s",
                    "test_eval_time_s",
                    "log_write_time_ms",
                    "checkpoint_save_time_ms",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
    ]
    if str(row["device"]) == "cpu":
        lines.append("- WARNING: this profile ran on CPU.")
    else:
        lines.append("- Device check: this profile ran on CUDA.")
    lines.append(
        f"- Full Trainer train epoch estimate: `{float(row['train_iter_time_ms_mean_est']):.1f}` ms/iter over `{int(row['num_train_batches'])}` batches."
    )
    lines.append(
        "- If this estimate is much larger than the raw forward/backward profile, inspect Trainer-side metrics, gradient clipping, validation, and artifact writing."
    )
    lines.append("")
    PROFILE_REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
