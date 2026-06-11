from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


DISTILLED_TO_BASE = {
    "ms_lif_snn_distill": "ms_lif_snn",
    "ms_cmg_lif_distill": "ms_cmg_lif",
    "ms_lif_tcn_plus": "ms_lif_tcn_attn",
}


def model_for_inference(model_name: str) -> str:
    return DISTILLED_TO_BASE.get(model_name, model_name)


def is_distilled_model(model_name: str) -> bool:
    return model_name in DISTILLED_TO_BASE


def format_optional(value: Any, decimals: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def summarize_gpu_benchmark(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    group_cols = ["dataset", "model", "batch_size"]
    metric_cols = [
        "macro_f1",
        "params",
        "spike_rate",
        "latency_ms_per_sample",
        "latency_ms_per_batch",
        "throughput_samples_per_s",
        "idle_power_w",
        "avg_power_w",
        "peak_power_w",
        "gpu_energy_j",
        "net_gpu_energy_j",
        "energy_mj_per_sample",
        "net_energy_mj_per_sample",
        "peak_memory_mb",
        "peak_nvml_memory_mb",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in raw.groupby(group_cols, dropna=False):
        dataset, model, batch_size = keys
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "batch_size": int(batch_size),
            "repeats": int(len(group)),
            "note": "gpu_software_stack_not_neuromorphic_power",
        }
        for col in metric_cols:
            if col not in group.columns:
                continue
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean()) if values.notna().any() else None
            row[f"{col}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        if "energy_note" in group.columns and group["energy_note"].astype(str).str.contains("measurement_noise").any():
            row["energy_note"] = "some_repeats_zero_or_negative_after_idle_subtraction"
        else:
            row["energy_note"] = "positive_net_energy_after_idle_subtraction"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "batch_size", "model"]).reset_index(drop=True)


def write_latex_summary(summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Dataset & Model & Batch & Macro-F1 & Lat. ms/sample & Net mJ/sample & Spike \\\\",
        "\\midrule",
    ]
    for row in summary.itertuples():
        macro = format_optional(getattr(row, "macro_f1_mean", None))
        latency = f"{format_optional(getattr(row, 'latency_ms_per_sample_mean', None), 3)} $\\pm$ {format_optional(getattr(row, 'latency_ms_per_sample_std', None), 3)}"
        energy = f"{format_optional(getattr(row, 'net_energy_mj_per_sample_mean', None), 3)} $\\pm$ {format_optional(getattr(row, 'net_energy_mj_per_sample_std', None), 3)}"
        spike = format_optional(getattr(row, "spike_rate_mean", None))
        lines.append(f"{row.dataset} & {row.model} & {row.batch_size} & {macro} & {latency} & {energy} & {spike} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_benchmark_figures(summary: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    batch_one = summary[summary["batch_size"].astype(int).eq(1)].copy()
    if batch_one.empty:
        batch_one = summary.copy()
    _scatter(
        batch_one,
        x_col="latency_ms_per_sample_mean",
        y_col="macro_f1_mean",
        output_path=output_dir / "fig_latency_vs_macro_f1.png",
        xlabel="Latency (ms/sample)",
        ylabel="Macro-F1",
    )
    _scatter(
        batch_one,
        x_col="net_energy_mj_per_sample_mean",
        y_col="macro_f1_mean",
        output_path=output_dir / "fig_energy_vs_macro_f1.png",
        xlabel="Net GPU energy (mJ/sample)",
        ylabel="Macro-F1",
    )
    _scatter(
        batch_one,
        x_col="params_mean",
        y_col="net_energy_mj_per_sample_mean",
        output_path=output_dir / "fig_params_vs_energy.png",
        xlabel="Parameters",
        ylabel="Net GPU energy (mJ/sample)",
    )


def _scatter(summary: pd.DataFrame, x_col: str, y_col: str, output_path: Path, xlabel: str, ylabel: str) -> None:
    if x_col not in summary.columns or y_col not in summary.columns:
        return
    values = summary[[x_col, y_col, "dataset", "model"]].dropna()
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, group in values.groupby("dataset"):
        ax.scatter(group[x_col], group[y_col], label=str(dataset), alpha=0.8)
        for row in group.itertuples():
            ax.annotate(str(row.model), (getattr(row, x_col), getattr(row, y_col)), fontsize=7, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
