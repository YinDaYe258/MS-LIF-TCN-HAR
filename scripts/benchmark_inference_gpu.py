from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.gpu_benchmark import model_for_inference, summarize_gpu_benchmark, write_benchmark_figures, write_latex_summary
from src.analysis.gpu_monitor import NVMLPowerMonitor
from src.datasets.hapt import create_hapt_dataloaders
from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.utils import build_model, count_parameters, load_config, set_seed


DEFAULT_MODELS = [
    "cnn1d",
    "gru",
    "window_gru",
    "lif_snn",
    "cmg_lif_lite",
    "ms_lif_snn",
    "ms_cmg_lif",
    "ms_lif_snn_distill",
    "ms_cmg_lif_distill",
    "ms_lif_tcn",
    "ms_lif_tcn_plus",
]

RESULT_FILES = {
    "ucihar": [
        "ms_lif_tcn_plus_multiseed.csv",
        "ms_tcn_multiseed_results.csv",
        "distill_multiseed_results.csv",
        "ucihar_formal_multiseed_results.csv",
        "ucihar_strong_baseline_results.csv",
        "ucihar_main_results.csv",
    ],
    "hapt6": [
        "ms_lif_tcn_plus_multiseed.csv",
        "ms_tcn_multiseed_results.csv",
        "distill_multiseed_results.csv",
        "hapt6_multiseed_results.csv",
        "hapt6_seed42_results.csv",
        "hapt_main_results.csv",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RTX GPU inference latency and NVML power.")
    parser.add_argument("--dataset", choices=["ucihar", "hapt6"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 16, 64])
    parser.add_argument("--warmup_batches", type=int, default=30)
    parser.add_argument("--benchmark_batches", type=int, default=200)
    parser.add_argument("--sample_interval_ms", type=int, default=20)
    parser.add_argument("--idle_duration_s", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--results_dir", default="results/gpu_benchmark")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_models(model_args: list[str]) -> list[str]:
    if len(model_args) == 1 and model_args[0].lower() == "all":
        return DEFAULT_MODELS
    return [model.lower() for model in model_args]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU inference benchmark requires an NVIDIA CUDA GPU.")
    try:
        import pynvml  # noqa: F401
    except Exception as exc:
        raise RuntimeError("NVML is unavailable. Install nvidia-ml-py with `pip install -r requirements.txt`.") from exc

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / "gpu_inference_raw.csv"
    config = load_config(args.config)
    config["seed"] = int(args.seed)
    config.setdefault("training", {})["num_workers"] = 0
    set_seed(int(args.seed))

    rows = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    all_rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        batch_config = copy.deepcopy(config)
        batch_config.setdefault("training", {})["batch_size"] = int(batch_size)
        loaders, meta = create_loaders(args.dataset, batch_config)
        test_loader = loaders["test"]
        for model_name in selected_models(args.models):
            checkpoint_row = find_checkpoint_row(args.dataset, model_name, int(args.seed), int(meta.context_len), "last")
            for repeat in range(int(args.repeats)):
                if not args.force and existing_raw_row(rows, args.dataset, model_name, args.seed, batch_size, repeat):
                    print(f"Skipping existing benchmark row: {args.dataset} {model_name} b{batch_size} repeat{repeat}")
                    continue
                row = benchmark_one(
                    dataset_key=args.dataset,
                    model_name=model_name,
                    config=batch_config,
                    checkpoint_row=checkpoint_row,
                    input_channels=meta.num_channels,
                    num_classes=meta.num_classes,
                    test_loader=test_loader,
                    batch_size=int(batch_size),
                    repeat=repeat,
                    warmup_batches=int(args.warmup_batches),
                    benchmark_batches=int(args.benchmark_batches),
                    sample_interval_ms=int(args.sample_interval_ms),
                    idle_duration_s=float(args.idle_duration_s),
                )
                all_rows.append(row)
                append_raw(raw_path, row)

    raw = pd.read_csv(raw_path)
    summary = summarize_gpu_benchmark(raw)
    summary_path = results_dir / "gpu_inference_summary.csv"
    summary.to_csv(summary_path, index=False)
    write_latex_summary(summary, results_dir / "table_gpu_inference_summary.tex")
    write_benchmark_figures(summary, results_dir)
    write_report(summary, results_dir, args)
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")


def create_loaders(dataset_key: str, config: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    if dataset_key == "ucihar":
        return create_ucihar_dataloaders(config, smoke_test=False)
    if dataset_key == "hapt6":
        dataset_cfg = config.setdefault("dataset", {})
        dataset_cfg["task"] = "hapt6"
        return create_hapt_dataloaders(config, smoke_test=False)
    raise ValueError(f"Unsupported dataset: {dataset_key}")


def find_checkpoint_row(dataset_key: str, model_name: str, seed: int, context_len: int, target_mode: str) -> dict[str, Any]:
    results_dir = Path("results")
    want_distilled = model_name.endswith("_distill")
    for file_name in RESULT_FILES[dataset_key]:
        path = results_dir / file_name
        if not path.exists():
            continue
        rows = pd.read_csv(path)
        if "model" not in rows.columns or "checkpoint" not in rows.columns:
            continue
        for _, row in rows.iterrows():
            if "dataset_key" in rows.columns and str(row.get("dataset_key", "")) != dataset_key:
                continue
            row_model = str(row.get("model", ""))
            if model_name == "ms_lif_tcn_plus":
                if row_model != "ms_lif_tcn_attn" or str(row.get("variant", "")) != "attn_supcon_0.1":
                    continue
            elif row_model != model_name:
                continue
            if want_distilled and "distill" not in file_name:
                continue
            if not want_distilled and "distill" in file_name:
                continue
            if int(row.get("seed", -1)) != int(seed):
                continue
            if int(row.get("context_len", -1)) != int(context_len):
                continue
            if str(row.get("target_mode", target_mode)) != str(target_mode):
                continue
            if dataset_key == "hapt6" and "task" in row and str(row.get("task")) not in {"hapt6", ""}:
                continue
            if _truthy(row.get("synthetic_data", False)) or _truthy(row.get("smoke_test", False)):
                continue
            checkpoint = str(row.get("checkpoint", ""))
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = ROOT / checkpoint_path
            if checkpoint_path.exists():
                result = row.to_dict()
                result["checkpoint_path"] = str(checkpoint_path)
                result["result_file"] = str(path)
                return result
    raise FileNotFoundError(
        f"No checkpoint found for dataset={dataset_key}, model={model_name}, seed={seed}, "
        f"context_len={context_len}, target_mode={target_mode}"
    )


def benchmark_one(
    dataset_key: str,
    model_name: str,
    config: dict[str, Any],
    checkpoint_row: dict[str, Any],
    input_channels: int,
    num_classes: int,
    test_loader: Any,
    batch_size: int,
    repeat: int,
    warmup_batches: int,
    benchmark_batches: int,
    sample_interval_ms: int,
    idle_duration_s: float,
) -> dict[str, Any]:
    device = torch.device("cuda")
    base_model = model_for_inference(model_name)
    checkpoint = torch.load(checkpoint_row["checkpoint_path"], map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", config)
    model_cfg = checkpoint_config.get("model", config.get("model", {})) if isinstance(checkpoint_config, dict) else config.get("model", {})
    model = build_model(base_model, input_channels, num_classes, model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    params = count_parameters(model)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch in iter_n_batches(test_loader, warmup_batches):
            _ = model(batch["x"].to(device, non_blocking=True))
    torch.cuda.synchronize()

    monitor = NVMLPowerMonitor(gpu_index=0)
    idle_power_w = monitor.measure_idle_power(duration_s=idle_duration_s, sample_interval_ms=sample_interval_ms)
    monitor.idle_power_w = idle_power_w
    total_samples = 0
    spike_rates: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    monitor.start(sample_interval_ms=sample_interval_ms)
    with torch.no_grad():
        for batch in iter_n_batches(test_loader, benchmark_batches):
            x = batch["x"].to(device, non_blocking=True)
            outputs = model(x)
            total_samples += int(x.shape[0])
            if isinstance(outputs, dict) and "spike_rate" in outputs:
                spike_rates.append(float(outputs["spike_rate"].detach().cpu()))
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    monitor.stop()
    peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    latency_ms_per_batch = elapsed_s * 1000.0 / max(benchmark_batches, 1)
    latency_ms_per_sample = elapsed_s * 1000.0 / max(total_samples, 1)
    throughput = total_samples / elapsed_s if elapsed_s > 0 else 0.0
    energy_j = monitor.active_energy_j
    net_energy_j = monitor.net_energy_j
    energy_note = (
        "measurement_noise_after_idle_subtraction"
        if net_energy_j <= 1e-12
        else "positive_net_energy_after_idle_subtraction"
    )
    row = {
        "dataset": "UCI-HAR" if dataset_key == "ucihar" else "HAPT-6",
        "dataset_key": dataset_key,
        "model": model_name,
        "base_model": base_model,
        "seed": int(config.get("seed", 42)),
        "context_len": int(config.get("dataset", {}).get("context_len", 8)),
        "target_mode": str(config.get("training", {}).get("target_mode", "last")),
        "batch_size": batch_size,
        "repeat": repeat,
        "checkpoint": checkpoint_row["checkpoint_path"],
        "macro_f1": _optional_float(checkpoint_row.get("macro_f1")),
        "accuracy": _optional_float(checkpoint_row.get("accuracy")),
        "params": params,
        "spike_rate": float(sum(spike_rates) / len(spike_rates)) if spike_rates else None,
        "warmup_batches": warmup_batches,
        "benchmark_batches": benchmark_batches,
        "idle_duration_s": float(idle_duration_s),
        "elapsed_s": elapsed_s,
        "total_samples": total_samples,
        "latency_ms_per_sample": latency_ms_per_sample,
        "latency_ms_per_batch": latency_ms_per_batch,
        "throughput_samples_per_s": throughput,
        "idle_power_w": idle_power_w,
        "avg_power_w": monitor.average_power_w,
        "peak_power_w": monitor.peak_power_w,
        "gpu_energy_j": energy_j,
        "net_gpu_energy_j": net_energy_j,
        "energy_mj_per_sample": energy_j * 1000.0 / max(total_samples, 1),
        "net_energy_mj_per_sample": net_energy_j * 1000.0 / max(total_samples, 1),
        "peak_memory_mb": peak_memory_mb,
        "peak_nvml_memory_mb": monitor.peak_mem_used_mb,
        "avg_gpu_util": monitor.average_gpu_util,
        "energy_note": energy_note,
        "note": "gpu_software_stack_not_neuromorphic_power",
    }
    print(
        f"{row['dataset']} {model_name} b{batch_size} r{repeat}: "
        f"{latency_ms_per_sample:.3f} ms/sample, {row['net_energy_mj_per_sample']:.3f} net mJ/sample"
    )
    return row


def iter_n_batches(loader: Any, n_batches: int):
    iterator = iter(loader)
    for _ in range(int(n_batches)):
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            yield next(iterator)


def existing_raw_row(rows: pd.DataFrame, dataset_key: str, model: str, seed: int, batch_size: int, repeat: int) -> bool:
    if rows.empty:
        return False
    required = {"dataset_key", "model", "seed", "batch_size", "repeat"}
    if not required.issubset(rows.columns):
        return False
    mask = (
        rows["dataset_key"].astype(str).eq(str(dataset_key))
        & rows["model"].astype(str).eq(str(model))
        & rows["seed"].astype(int).eq(int(seed))
        & rows["batch_size"].astype(int).eq(int(batch_size))
        & rows["repeat"].astype(int).eq(int(repeat))
    )
    return bool(mask.any())


def append_raw(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True, sort=False)
    else:
        updated = pd.DataFrame([row])
    updated.to_csv(path, index=False)


def write_report(summary: pd.DataFrame, results_dir: Path, args: argparse.Namespace) -> None:
    device_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda or "unknown"
    torch_version = torch.__version__
    lines = [
        "# RTX GPU Inference Benchmark Report",
        "",
        f"- Hardware: {device_name}",
        f"- CUDA: {cuda_version}",
        f"- PyTorch: {torch_version}",
        "- NVML: available",
        "- Scope: measured GPU inference power/latency under this PyTorch CUDA software stack.",
        "- Caveat: not measured neuromorphic hardware power and not evidence of event-driven SNN energy savings.",
        f"- Benchmark protocol: warmup_batches={args.warmup_batches}, benchmark_batches={args.benchmark_batches}, repeats={args.repeats}.",
        f"- Idle power sampling duration: {args.idle_duration_s} s.",
        "",
    ]
    subset = summary[summary["batch_size"].astype(int).eq(1)] if not summary.empty else summary
    if not subset.empty:
        latency_row = subset.loc[subset["latency_ms_per_sample_mean"].idxmin()]
        positive_energy = subset[pd.to_numeric(subset["net_energy_mj_per_sample_mean"], errors="coerce").gt(1e-6)]
        energy_row = positive_energy.loc[positive_energy["net_energy_mj_per_sample_mean"].idxmin()] if not positive_energy.empty else None
        valid = positive_energy.dropna(subset=["macro_f1_mean", "net_energy_mj_per_sample_mean"]).copy()
        if not valid.empty:
            valid["macro_f1_per_mj"] = valid["macro_f1_mean"] / valid["net_energy_mj_per_sample_mean"].clip(lower=1e-9)
            tradeoff_row = valid.loc[valid["macro_f1_per_mj"].idxmax()]
        else:
            tradeoff_row = None
        lines.extend(
            [
                "## Batch-1 Highlights",
                "",
                f"- Lowest latency: {latency_row.dataset} {latency_row.model} ({latency_row.latency_ms_per_sample_mean:.3f} ms/sample).",
            ]
        )
        if energy_row is not None:
            lines.append(
                f"- Lowest positive net GPU energy/sample: {energy_row.dataset} {energy_row.model} "
                f"({energy_row.net_energy_mj_per_sample_mean:.3f} mJ/sample)."
            )
        if tradeoff_row is not None:
            lines.append(
                f"- Best Macro-F1 per mJ: {tradeoff_row.dataset} {tradeoff_row.model} "
                f"({tradeoff_row.macro_f1_per_mj:.4f})."
            )
        snn = subset[subset["spike_rate_mean"].notna()]
        ann = subset[subset["spike_rate_mean"].isna()]
        if not snn.empty and not ann.empty:
            lines.append(
                f"- Mean SNN net mJ/sample: {snn['net_energy_mj_per_sample_mean'].mean():.3f}; "
                f"mean non-SNN net mJ/sample: {ann['net_energy_mj_per_sample_mean'].mean():.3f}."
            )
            if snn["net_energy_mj_per_sample_mean"].mean() < ann["net_energy_mj_per_sample_mean"].mean():
                lines.append("- In this benchmark, SNN models use lower average GPU energy per sample than non-SNN models.")
            else:
                lines.append(
                    "- In this benchmark, SNN models do not show lower average GPU energy per sample; PyTorch GPU execution does not exploit neuromorphic event-driven sparsity."
                )
        zero_count = int((subset["net_energy_mj_per_sample_mean"].fillna(0.0) <= 0.0).sum())
        if zero_count:
            lines.append(
                f"- {zero_count} batch-1 summary rows have zero net energy after idle subtraction; "
                "rerun with more benchmark batches/repeats for paper-final energy numbers."
            )
    lines.append("")
    (results_dir / "gpu_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def _truthy(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
