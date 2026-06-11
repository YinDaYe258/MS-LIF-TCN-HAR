from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V3_DIR = Path("results/final_paper_v3")
INPUT_PATH = V3_DIR / "context_length_ablation.csv"
SUMMARY_PATH = V3_DIR / "context_length_ablation_aligned_summary.csv"
TABLE_PATH = V3_DIR / "table_context_length_ablation_aligned.tex"
REPORT_PATH = V3_DIR / "context_length_ablation_report.md"
FIGURE_PATH = V3_DIR / "fig_context_length_vs_macro_f1.png"

EXPECTED_DATASETS = ["ucihar", "hapt6", "pamap2", "mhealth"]
EXPECTED_MODELS = ["ms_lif_tcn", "ms_ann_tcn"]
EXPECTED_CONTEXT_LENS = [1, 2, 4, 8]
EXPECTED_SEEDS = [42, 43, 44]
SEQUENCE_PROTOCOL = "aligned_kmax_8"
TRAINING_BUDGET = "alignedK8_e20_p5_b64"


def main() -> None:
    V3_DIR.mkdir(parents=True, exist_ok=True)
    filtered = load_filtered_results(INPUT_PATH)
    validate_filtered_results(filtered)
    summary = summarize(filtered)
    summary.to_csv(SUMMARY_PATH, index=False)
    write_latex_table(summary)
    write_figure(summary)
    write_report(summary)
    print(f"Wrote aligned K diagnostic summary to {SUMMARY_PATH}")
    print(f"Wrote aligned K diagnostic report to {REPORT_PATH}")


def load_filtered_results(path: Path = INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing context-length ablation CSV: {path}")
    df = pd.read_csv(path)
    smoke = as_bool_series(df.get("smoke_test", pd.Series(False, index=df.index)))
    synthetic = as_bool_series(df.get("synthetic_data", pd.Series(False, index=df.index)))
    filtered = df[
        df["model"].astype(str).isin(EXPECTED_MODELS)
        & df["sequence_protocol"].astype(str).eq(SEQUENCE_PROTOCOL)
        & df["training_budget"].astype(str).eq(TRAINING_BUDGET)
        & ~smoke
        & ~synthetic
    ].copy()
    return filtered


def validate_filtered_results(df: pd.DataFrame) -> None:
    expected_total = len(EXPECTED_DATASETS) * len(EXPECTED_MODELS) * len(EXPECTED_CONTEXT_LENS) * len(EXPECTED_SEEDS)
    if len(df) != expected_total:
        raise ValueError(f"Expected {expected_total} aligned K rows, found {len(df)}")
    missing: list[str] = []
    for dataset in EXPECTED_DATASETS:
        for model in EXPECTED_MODELS:
            for context_len in EXPECTED_CONTEXT_LENS:
                subset = df[
                    df["dataset_key"].astype(str).eq(dataset)
                    & df["model"].astype(str).eq(model)
                    & df["context_len"].astype(int).eq(context_len)
                ]
                seeds = sorted(int(seed) for seed in subset["seed"].dropna().unique())
                if seeds != EXPECTED_SEEDS:
                    missing.append(f"{dataset} {model} K={context_len}: seeds={seeds}")
    if missing:
        raise ValueError("Incomplete aligned K diagnostic rows: " + "; ".join(missing))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset_key, model, context_len), group in df.groupby(["dataset_key", "model", "context_len"], sort=False):
        group = group.sort_values("seed")
        spike = pd.to_numeric(group.get("spike_rate", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "dataset": str(group["dataset"].iloc[0]),
                "dataset_key": str(dataset_key),
                "model": str(model),
                "context_len": int(context_len),
                "sequence_protocol": SEQUENCE_PROTOCOL,
                "training_budget": TRAINING_BUDGET,
                "num_seeds": int(group["seed"].nunique()),
                "seeds": " ".join(str(int(seed)) for seed in sorted(group["seed"].unique())),
                "accuracy_mean": float(group["accuracy"].mean()),
                "accuracy_std": sample_std(group["accuracy"]),
                "macro_f1_mean": float(group["macro_f1"].mean()),
                "macro_f1_std": sample_std(group["macro_f1"]),
                "weighted_f1_mean": float(group["weighted_f1"].mean()),
                "weighted_f1_std": sample_std(group["weighted_f1"]),
                "balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
                "balanced_accuracy_std": sample_std(group["balanced_accuracy"]),
                "spike_rate_mean": float(spike.mean()) if not spike.dropna().empty else np.nan,
                "spike_rate_std": sample_std(spike) if not spike.dropna().empty else np.nan,
                "best_epoch_mean": float(group["best_epoch"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["dataset_order"] = summary["dataset_key"].map({name: idx for idx, name in enumerate(EXPECTED_DATASETS)})
    summary["model_order"] = summary["model"].map({name: idx for idx, name in enumerate(EXPECTED_MODELS)})
    summary = summary.sort_values(["dataset_order", "model_order", "context_len"]).drop(
        columns=["dataset_order", "model_order"]
    )
    summary["delta_vs_k1_macro_f1"] = 0.0
    for dataset in EXPECTED_DATASETS:
        for model in EXPECTED_MODELS:
            mask = summary["dataset_key"].eq(dataset) & summary["model"].eq(model)
            k1 = float(summary[mask & summary["context_len"].eq(1)]["macro_f1_mean"].iloc[0])
            summary.loc[mask, "delta_vs_k1_macro_f1"] = summary.loc[mask, "macro_f1_mean"] - k1
    return summary


def write_latex_table(summary: pd.DataFrame) -> None:
    table = summary[
        [
            "dataset_key",
            "model",
            "context_len",
            "num_seeds",
            "macro_f1_mean",
            "macro_f1_std",
            "delta_vs_k1_macro_f1",
            "spike_rate_mean",
        ]
    ].copy()
    table.to_latex(TABLE_PATH, index=False, escape=False, float_format=lambda value: f"{value:.4f}")


def write_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes_flat = axes.reshape(-1)
    for axis, dataset in zip(axes_flat, EXPECTED_DATASETS, strict=True):
        for model in EXPECTED_MODELS:
            group = summary[summary["dataset_key"].eq(dataset) & summary["model"].eq(model)].sort_values("context_len")
            axis.errorbar(
                group["context_len"].to_numpy(),
                group["macro_f1_mean"].to_numpy(),
                yerr=group["macro_f1_std"].to_numpy(),
                marker="o",
                capsize=3,
                label=model,
            )
        axis.set_title(dataset)
        axis.set_xticks(EXPECTED_CONTEXT_LENS)
        axis.grid(True, alpha=0.3)
        axis.set_ylabel("Macro-F1")
    for axis in axes_flat[-2:]:
        axis.set_xlabel("Context length K")
    axes_flat[0].legend(loc="best", fontsize=8)
    fig.suptitle("Aligned context-length diagnostic")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    trends = build_trend_rows(summary)
    model_comparison = build_model_comparison(summary)
    lines = [
        "# Aligned Context-Length Diagnostic",
        "",
        "This reduced K screening uses aligned final-window targets with `aligned_kmax=8`, three seeds, `epochs=20`, `patience=5`, and `batch_size=64`.",
        "It is a diagnostic ablation and does not replace the 10-seed v3 main protocol.",
        "",
        "## Completeness",
        "",
        "- Input rows after filtering: 96.",
        f"- Required protocol: `{SEQUENCE_PROTOCOL}`.",
        f"- Required training budget: `{TRAINING_BUDGET}`.",
        f"- Models: {' '.join(EXPECTED_MODELS)}.",
        f"- Datasets: {' '.join(EXPECTED_DATASETS)}.",
        f"- Context lengths: {' '.join(str(k) for k in EXPECTED_CONTEXT_LENS)}.",
        f"- Seeds: {' '.join(str(seed) for seed in EXPECTED_SEEDS)}.",
        "",
        "## Context Trends",
        "",
        markdown_table(
            trends[
                [
                    "dataset_key",
                    "model",
                    "k1_macro_f1_mean",
                    "best_context_len",
                    "best_macro_f1_mean",
                    "best_delta_vs_k1",
                    "num_k_gt_1_above_k1",
                    "interpretation",
                ]
            ]
        ),
        "",
        "## MS-LIF-TCN versus MS-ANN-TCN by K",
        "",
        markdown_table(model_comparison),
        "",
        "## Mean Macro-F1 by K",
        "",
        markdown_table(
            summary[
                [
                    "dataset_key",
                    "model",
                    "context_len",
                    "macro_f1_mean",
                    "macro_f1_std",
                    "delta_vs_k1_macro_f1",
                    "spike_rate_mean",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
    ]
    lif_trends = trends[trends["model"].eq("ms_lif_tcn")]
    ann_trends = trends[trends["model"].eq("ms_ann_tcn")]
    lines.append(f"- MS-LIF-TCN best K is greater than 1 on `{int((lif_trends['best_context_len'] > 1).sum())}/4` datasets.")
    lines.append(f"- MS-ANN-TCN best K is greater than 1 on `{int((ann_trends['best_context_len'] > 1).sum())}/4` datasets.")
    lines.append(
        "- The diagnostic supports cross-window context as a useful factor, but K effects are dataset- and architecture-dependent."
    )
    lines.append(
        "- MS-ANN-TCN is a strong continuous-valued counterpart; this ablation should not be used to claim universal SNN superiority."
    )
    lines.append(
        "- K=16 remains excluded from the main K sweep because the support audit found zero-support classes in HAPT-6 and MHEALTH."
    )
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_trend_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for model in EXPECTED_MODELS:
            group = summary[summary["dataset_key"].eq(dataset) & summary["model"].eq(model)].sort_values("context_len")
            k1 = group[group["context_len"].eq(1)].iloc[0]
            context_rows = group[group["context_len"].gt(1)]
            best = group.sort_values(["macro_f1_mean", "context_len"], ascending=[False, True]).iloc[0]
            context_above = context_rows[context_rows["macro_f1_mean"].gt(float(k1["macro_f1_mean"]))]
            rows.append(
                {
                    "dataset_key": dataset,
                    "model": model,
                    "k1_macro_f1_mean": float(k1["macro_f1_mean"]),
                    "best_context_len": int(best["context_len"]),
                    "best_macro_f1_mean": float(best["macro_f1_mean"]),
                    "best_delta_vs_k1": float(best["macro_f1_mean"] - k1["macro_f1_mean"]),
                    "num_k_gt_1_above_k1": int(len(context_above)),
                    "interpretation": interpret(best, k1, context_rows),
                }
            )
    return pd.DataFrame(rows)


def build_model_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for context_len in EXPECTED_CONTEXT_LENS:
            lif = summary[
                summary["dataset_key"].eq(dataset)
                & summary["model"].eq("ms_lif_tcn")
                & summary["context_len"].eq(context_len)
            ].iloc[0]
            ann = summary[
                summary["dataset_key"].eq(dataset)
                & summary["model"].eq("ms_ann_tcn")
                & summary["context_len"].eq(context_len)
            ].iloc[0]
            rows.append(
                {
                    "dataset_key": dataset,
                    "context_len": int(context_len),
                    "ms_lif_tcn_macro_f1_mean": float(lif["macro_f1_mean"]),
                    "ms_ann_tcn_macro_f1_mean": float(ann["macro_f1_mean"]),
                    "lif_minus_ann_macro_f1": float(lif["macro_f1_mean"] - ann["macro_f1_mean"]),
                }
            )
    return pd.DataFrame(rows)


def interpret(best: pd.Series, k1: pd.Series, context_rows: pd.DataFrame) -> str:
    if int(best["context_len"]) == 1:
        return "k1_best_context_not_supported"
    if (context_rows["macro_f1_mean"] > float(k1["macro_f1_mean"])).all():
        return "all_context_lengths_above_k1"
    return "best_context_length_above_k1_mixed_short_context"


def sample_std(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) <= 1:
        return 0.0
    return float(numeric.std(ddof=1))


def as_bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append([format_cell(row[column]) for column in headers])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(float(value)):
            return "N/A"
        return f"{float(value):.4f}"
    return str(value)


if __name__ == "__main__":
    main()
