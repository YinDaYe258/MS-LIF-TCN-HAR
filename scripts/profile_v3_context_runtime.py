from __future__ import annotations

import argparse
import copy
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
from src.training.losses import sequence_classification_loss
from src.training.utils import build_model, count_parameters, get_device, set_seed

OUT_DIR = Path("results/final_paper_v3")
PROFILE_CSV = OUT_DIR / "context_runtime_profile.csv"
PROFILE_REPORT = OUT_DIR / "context_runtime_profile_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile v3 aligned context runtime without writing ablation results.")
    parser.add_argument("--dataset", default="ucihar", choices=sorted(DATASETS))
    parser.add_argument("--model", default="ms_lif_tcn")
    parser.add_argument("--context_lens", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aligned_kmax", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--profile_iters", type=int, default=3)
    parser.add_argument("--warmup_iters", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [profile_one(args, context_len) for context_len in args.context_lens]
    frame = pd.DataFrame(rows)
    frame.to_csv(PROFILE_CSV, index=False)
    write_report(frame, args)
    print(f"Wrote context runtime profile to {PROFILE_CSV}")
    print(f"Wrote context runtime profile report to {PROFILE_REPORT}")


def profile_one(args: argparse.Namespace, context_len: int) -> dict[str, Any]:
    if context_len > args.aligned_kmax:
        raise ValueError(f"context_len={context_len} exceeds aligned_kmax={args.aligned_kmax}")
    set_seed(args.seed)
    spec = DATASETS[args.dataset]
    config_args = SimpleNamespace(
        smoke_test=False,
        epochs=None,
        patience=None,
        batch_size=args.batch_size,
    )
    config = make_config(str(spec["config"]), args.model, int(context_len), int(args.seed), config_args)
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
    model = build_model(args.model, meta.num_channels, meta.num_classes, config.get("model", {})).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("training", {}).get("learning_rate", 1e-3)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 1e-4)),
    )
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    spike_reg_lambda = float(config.get("training", {}).get("spike_reg_lambda", 0.0))

    first_batch_start = time.perf_counter()
    batch_iter = iter(loaders["train"])
    first_batch = next(batch_iter)
    first_batch_fetch_ms = (time.perf_counter() - first_batch_start) * 1000.0
    batch_shape = tuple(first_batch["x"].shape)
    batch_dtype = str(first_batch["x"].dtype)
    batch_device_before_move = str(first_batch["x"].device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Warmup keeps CUDA initialization out of the measured loop.
    reusable_batches = [first_batch]
    for _ in range(max(0, int(args.profile_iters) + int(args.warmup_iters) - 1)):
        try:
            reusable_batches.append(next(batch_iter))
        except StopIteration:
            batch_iter = iter(loaders["train"])
            reusable_batches.append(next(batch_iter))
    for batch in reusable_batches[: int(args.warmup_iters)]:
        run_train_iteration(model, optimizer, batch, device, target_mode, spike_reg_lambda, measure=False)

    forward_times: list[float] = []
    backward_times: list[float] = []
    iter_times: list[float] = []
    losses: list[float] = []
    spike_rates: list[float] = []
    for batch in reusable_batches[int(args.warmup_iters) : int(args.warmup_iters) + int(args.profile_iters)]:
        result = run_train_iteration(model, optimizer, batch, device, target_mode, spike_reg_lambda, measure=True)
        forward_times.append(result["forward_ms"])
        backward_times.append(result["backward_ms"])
        iter_times.append(result["iter_ms"])
        losses.append(result["loss"])
        if result["spike_rate"] is not None:
            spike_rates.append(result["spike_rate"])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory_allocated_mb = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
        memory_reserved_mb = torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
        peak_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        peak_memory_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
        device_name = torch.cuda.get_device_name(device)
    else:
        memory_allocated_mb = 0.0
        memory_reserved_mb = 0.0
        peak_memory_allocated_mb = 0.0
        peak_memory_reserved_mb = 0.0
        device_name = str(device)

    train_sequences = len(loaders["train"].dataset)
    val_sequences = len(loaders["val"].dataset)
    test_sequences = len(loaders["test"].dataset)
    num_train_batches = len(loaders["train"])
    iter_mean = mean(iter_times)
    return {
        "dataset": spec["display"],
        "dataset_key": args.dataset,
        "model": args.model,
        "seed": int(args.seed),
        "context_len": int(context_len),
        "aligned_kmax": int(args.aligned_kmax),
        "sequence_protocol": f"aligned_kmax_{int(args.aligned_kmax)}",
        "device": str(device),
        "device_name": device_name,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "num_train_sequences": int(train_sequences),
        "num_val_sequences": int(val_sequences),
        "num_test_sequences": int(test_sequences),
        "num_train_batches": int(num_train_batches),
        "batch_size": int(args.batch_size),
        "batch_x_shape": str(tuple(int(dim) for dim in batch_shape)),
        "batch_dtype": batch_dtype,
        "batch_device_before_move": batch_device_before_move,
        "params": int(count_parameters(model)),
        "loader_build_time_s": loader_build_s,
        "align_time_s": align_s,
        "first_batch_fetch_time_ms": first_batch_fetch_ms,
        "forward_time_ms_mean": mean(forward_times),
        "forward_time_ms_std": std(forward_times),
        "backward_time_ms_mean": mean(backward_times),
        "backward_time_ms_std": std(backward_times),
        "train_iter_time_ms_mean": iter_mean,
        "train_iter_time_ms_std": std(iter_times),
        "one_epoch_time_s_est": iter_mean * float(num_train_batches) / 1000.0,
        "loss_mean": mean(losses),
        "spike_rate_mean": mean(spike_rates) if spike_rates else float("nan"),
        "gpu_memory_allocated_mb": memory_allocated_mb,
        "gpu_memory_reserved_mb": memory_reserved_mb,
        "gpu_peak_memory_allocated_mb": peak_memory_allocated_mb,
        "gpu_peak_memory_reserved_mb": peak_memory_reserved_mb,
    }


def run_train_iteration(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    target_mode: str,
    spike_reg_lambda: float,
    measure: bool,
) -> dict[str, float | None]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    iter_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_start = time.perf_counter()
    outputs = model(x)
    loss, _ = sequence_classification_loss(
        outputs,
        y,
        spike_reg_lambda=spike_reg_lambda,
        target_mode=target_mode,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_end = time.perf_counter()
    backward_start = time.perf_counter()
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    iter_end = time.perf_counter()
    spike_rate = outputs.get("spike_rate")
    return {
        "forward_ms": (forward_end - forward_start) * 1000.0 if measure else 0.0,
        "backward_ms": (iter_end - backward_start) * 1000.0 if measure else 0.0,
        "iter_ms": (iter_end - iter_start) * 1000.0 if measure else 0.0,
        "loss": float(loss.detach().cpu()),
        "spike_rate": float(spike_rate.detach().cpu()) if spike_rate is not None else None,
    }


def write_report(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# v3 Context Runtime Profile",
        "",
        "This report profiles context-length runtime only. It does not train to convergence and does not write context ablation rows.",
        "",
        f"Dataset: `{args.dataset}`",
        f"Model: `{args.model}`",
        f"Seed: `{args.seed}`",
        f"Aligned final-window protocol: `aligned_kmax_{args.aligned_kmax}`",
        f"Profile iterations per K: `{args.profile_iters}` after `{args.warmup_iters}` warmup iterations",
        "",
        "## Runtime Table",
        "",
        markdown_table(
            frame[
                [
                    "context_len",
                    "device",
                    "batch_x_shape",
                    "num_train_sequences",
                    "num_train_batches",
                    "forward_time_ms_mean",
                    "backward_time_ms_mean",
                    "train_iter_time_ms_mean",
                    "one_epoch_time_s_est",
                    "gpu_peak_memory_allocated_mb",
                ]
            ]
        ),
        "",
        "## Interpretation Checklist",
        "",
    ]
    if frame["device"].astype(str).str.contains("cpu").any():
        lines.append("- WARNING: at least one profile row ran on CPU. Do not resume K sweep until device handling is fixed.")
    else:
        lines.append("- Device check: profile rows did not report CPU execution.")
    if len(frame) >= 2:
        k1 = frame[frame["context_len"].eq(1)]
        k4 = frame[frame["context_len"].eq(4)]
        k8 = frame[frame["context_len"].eq(8)]
        ratios: list[float] = []
        if not k1.empty and not k4.empty:
            ratio = float(k4.iloc[0]["train_iter_time_ms_mean"]) / max(float(k1.iloc[0]["train_iter_time_ms_mean"]), 1e-9)
            ratios.append(ratio)
            lines.append(f"- K=4/K=1 train-iteration ratio: `{ratio:.2f}`.")
        if not k1.empty and not k8.empty:
            ratio = float(k8.iloc[0]["train_iter_time_ms_mean"]) / max(float(k1.iloc[0]["train_iter_time_ms_mean"]), 1e-9)
            ratios.append(ratio)
            lines.append(f"- K=8/K=1 train-iteration ratio: `{ratio:.2f}`.")
        if ratios and max(ratios) <= 2.0:
            lines.append(
                "- Raw forward/backward profiling did not reproduce the earlier full-Trainer `~3s/iter` slow path."
            )
    lines.extend(
        [
            "- CUDA peak-memory values are a coarse diagnostic because allocator reuse can affect short profiles; timing, device, batch shape, and sequence counts are the primary checks here.",
            "- If K=4/K=8 runtime remains much higher than K=1/K=2, inspect model forward and dataloader behavior before launching reduced K screening.",
            "- If runtime is acceptable, resume with a staged `ms_lif_tcn`-only aligned K diagnostic before adding `ms_ann_tcn`.",
            "",
        ]
    )
    PROFILE_REPORT.write_text("\n".join(lines), encoding="utf-8")


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    value = torch.tensor(values, dtype=torch.float64)
    return float(value.std(unbiased=True).item())


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.astype(object).itertuples(index=False, name=None)]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


if __name__ == "__main__":
    main()
