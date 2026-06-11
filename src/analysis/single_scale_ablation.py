from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V3_DIR = Path("results/final_paper_v3")
INPUT_PATH = V3_DIR / "single_scale_ablation.csv"
SUMMARY_PATH = V3_DIR / "single_scale_ablation_summary.csv"
TABLE_PATH = V3_DIR / "table_single_scale_ablation.tex"
FIGURE_PATH = V3_DIR / "fig_single_scale_vs_multiscale.png"
REPORT_PATH = V3_DIR / "single_scale_ablation_report.md"

EXPECTED_DATASETS = ["ucihar", "hapt6", "pamap2", "mhealth"]
EXPECTED_VARIANTS = ["multi_scale_full", "single_k3", "single_k5", "single_k9"]
EXPECTED_SEEDS = [42, 43, 44]
TRAINING_BUDGET = "fixedK8_e20_p5_b64"
SEQUENCE_PROTOCOL = "fixed_k8"


def main() -> None:
    V3_DIR.mkdir(parents=True, exist_ok=True)
    filtered = load_filtered(INPUT_PATH)
    validate(filtered)
    summary = summarize(filtered)
    summary.to_csv(SUMMARY_PATH, index=False)
    write_latex_table(summary)
    write_figure(summary)
    write_report(summary)
    print(f"Wrote single-scale summary to {SUMMARY_PATH}")
    print(f"Wrote single-scale report to {REPORT_PATH}")


def load_filtered(path: Path = INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing single-scale ablation CSV: {path}")
    df = pd.read_csv(path)
    smoke = as_bool_series(df.get("smoke_test", pd.Series(False, index=df.index)))
    synthetic = as_bool_series(df.get("synthetic_data", pd.Series(False, index=df.index)))
    return df[
        df["ablation"].astype(str).eq("single_scale")
        & df["variant_key"].astype(str).isin(EXPECTED_VARIANTS)
        & df["sequence_protocol"].astype(str).eq(SEQUENCE_PROTOCOL)
        & df["training_budget"].astype(str).eq(TRAINING_BUDGET)
        & ~smoke
        & ~synthetic
    ].copy()


def validate(df: pd.DataFrame) -> None:
    expected_total = len(EXPECTED_DATASETS) * len(EXPECTED_VARIANTS) * len(EXPECTED_SEEDS)
    if len(df) != expected_total:
        raise ValueError(f"Expected {expected_total} single-scale rows, found {len(df)}")
    missing: list[str] = []
    for dataset in EXPECTED_DATASETS:
        for variant in EXPECTED_VARIANTS:
            subset = df[df["dataset_key"].astype(str).eq(dataset) & df["variant_key"].astype(str).eq(variant)]
            seeds = sorted(int(seed) for seed in subset["seed"].dropna().unique())
            if seeds != EXPECTED_SEEDS:
                missing.append(f"{dataset} {variant}: seeds={seeds}")
    if missing:
        raise ValueError("Incomplete single-scale rows: " + "; ".join(missing))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset_key, variant_key), group in df.groupby(["dataset_key", "variant_key"], sort=False):
        group = group.sort_values("seed")
        rows.append(
            {
                "dataset": str(group["dataset"].iloc[0]),
                "dataset_key": str(dataset_key),
                "model": "ms_lif_tcn",
                "variant": str(variant_key),
                "variant_key": str(variant_key),
                "encoder_mode": str(group["encoder_mode"].iloc[0]),
                "single_kernel_size": int(group["single_kernel_size"].iloc[0]),
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
                "spike_rate_mean": float(pd.to_numeric(group["spike_rate"], errors="coerce").mean()),
                "spike_rate_std": sample_std(group["spike_rate"]),
                "params": int(round(group["params"].mean())),
                "best_epoch_mean": float(group["best_epoch"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["dataset_order"] = summary["dataset_key"].map({name: idx for idx, name in enumerate(EXPECTED_DATASETS)})
    summary["variant_order"] = summary["variant_key"].map({name: idx for idx, name in enumerate(EXPECTED_VARIANTS)})
    summary = summary.sort_values(["dataset_order", "variant_order"]).drop(columns=["dataset_order", "variant_order"])
    summary["delta_vs_multi_scale_macro_f1"] = 0.0
    for dataset in EXPECTED_DATASETS:
        mask = summary["dataset_key"].eq(dataset)
        full = float(summary[mask & summary["variant_key"].eq("multi_scale_full")]["macro_f1_mean"].iloc[0])
        summary.loc[mask, "delta_vs_multi_scale_macro_f1"] = summary.loc[mask, "macro_f1_mean"] - full
    return summary


def write_latex_table(summary: pd.DataFrame) -> None:
    table = summary[
        [
            "dataset_key",
            "variant",
            "encoder_mode",
            "single_kernel_size",
            "num_seeds",
            "macro_f1_mean",
            "macro_f1_std",
            "delta_vs_multi_scale_macro_f1",
            "params",
            "spike_rate_mean",
        ]
    ].copy()
    table.to_latex(TABLE_PATH, index=False, escape=False, float_format=lambda value: f"{value:.4f}")


def write_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
    axes_flat = axes.reshape(-1)
    for axis, dataset in zip(axes_flat, EXPECTED_DATASETS, strict=True):
        group = summary[summary["dataset_key"].eq(dataset)].copy()
        group["order"] = group["variant_key"].map({variant: idx for idx, variant in enumerate(EXPECTED_VARIANTS)})
        group = group.sort_values("order")
        axis.bar(group["variant_key"], group["macro_f1_mean"], yerr=group["macro_f1_std"], capsize=3)
        axis.axhline(
            float(group[group["variant_key"].eq("multi_scale_full")]["macro_f1_mean"].iloc[0]),
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )
        axis.set_title(dataset)
        axis.set_ylabel("Macro-F1")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Single-scale vs multi-scale encoder diagnostic")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    trends = trend_rows(summary)
    lines = [
        "# Single-Scale Encoder Diagnostic",
        "",
        "This reduced single-scale ablation uses fixed `K=8`, three seeds, `epochs=20`, `patience=5`, and `batch_size=64`.",
        "It is a diagnostic ablation and does not replace the 10-seed v3 main protocol.",
        "",
        "## Completeness",
        "",
        "- Input rows after filtering: 48.",
        f"- Required protocol: `{SEQUENCE_PROTOCOL}`.",
        f"- Required training budget: `{TRAINING_BUDGET}`.",
        f"- Variants: {' '.join(EXPECTED_VARIANTS)}.",
        f"- Datasets: {' '.join(EXPECTED_DATASETS)}.",
        f"- Seeds: {' '.join(str(seed) for seed in EXPECTED_SEEDS)}.",
        "",
        "## Dataset Trends",
        "",
        markdown_table(trends),
        "",
        "## Mean Macro-F1 by Encoder Variant",
        "",
        markdown_table(
            summary[
                [
                    "dataset_key",
                    "variant",
                    "encoder_mode",
                    "single_kernel_size",
                    "macro_f1_mean",
                    "macro_f1_std",
                    "delta_vs_multi_scale_macro_f1",
                    "params",
                    "spike_rate_mean",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
    ]
    full_best = int(trends["multi_scale_is_best"].sum())
    full_above_single_mean = int(trends["multi_scale_above_mean_single"].sum())
    lines.append(f"- Multi-scale is the best reduced-budget mean on `{full_best}/4` datasets.")
    lines.append(f"- Multi-scale is above the average of single-kernel variants on `{full_above_single_mean}/4` datasets.")
    lines.append("- If single-k variants are close or better, multi-scale should be described as a design component rather than the sole source of gains.")
    lines.append("- The primary v3 evidence for cross-window modeling remains the main run, aligned K diagnostic, and TCN-depth diagnostic.")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def trend_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in EXPECTED_DATASETS:
        group = summary[summary["dataset_key"].eq(dataset)]
        full = group[group["variant_key"].eq("multi_scale_full")].iloc[0]
        singles = group[~group["variant_key"].eq("multi_scale_full")]
        best = group.sort_values(["macro_f1_mean", "variant_key"], ascending=[False, True]).iloc[0]
        best_single = singles.sort_values(["macro_f1_mean", "variant_key"], ascending=[False, True]).iloc[0]
        single_mean = float(singles["macro_f1_mean"].mean())
        full_macro = float(full["macro_f1_mean"])
        best_single_delta = full_macro - float(best_single["macro_f1_mean"])
        if str(best["variant_key"]) == "multi_scale_full" and best_single_delta >= 0.005:
            judgement = "strong_multiscale_support"
        elif str(best["variant_key"]) == "multi_scale_full" or full_macro >= single_mean:
            judgement = "modest_or_dataset_dependent_support"
        else:
            judgement = "single_scale_competitive"
        rows.append(
            {
                "dataset_key": dataset,
                "multi_scale_macro_f1_mean": full_macro,
                "best_variant": str(best["variant_key"]),
                "best_macro_f1_mean": float(best["macro_f1_mean"]),
                "best_single_variant": str(best_single["variant_key"]),
                "best_single_macro_f1_mean": float(best_single["macro_f1_mean"]),
                "multi_minus_best_single": best_single_delta,
                "multi_minus_single_mean": full_macro - single_mean,
                "multi_scale_is_best": str(best["variant_key"]) == "multi_scale_full",
                "multi_scale_above_mean_single": full_macro >= single_mean,
                "judgement": judgement,
            }
        )
    return pd.DataFrame(rows)


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
    rows = [[format_cell(row[column]) for column in headers] for _, row in df.iterrows()]
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
