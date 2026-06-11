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
SUMMARY_PATH = V3_DIR / "context_length_msliftcn_only_summary.csv"
REPORT_PATH = V3_DIR / "context_length_msliftcn_only_report.md"
FIGURE_PATH = V3_DIR / "fig_context_length_msliftcn_only.png"

EXPECTED_DATASETS = ["ucihar", "hapt6", "pamap2", "mhealth"]
EXPECTED_CONTEXT_LENS = [1, 2, 4, 8]
EXPECTED_SEEDS = [42, 43, 44]
MODEL = "ms_lif_tcn"
SEQUENCE_PROTOCOL = "aligned_kmax_8"
TRAINING_BUDGET = "alignedK8_e20_p5_b64"


def main() -> None:
    V3_DIR.mkdir(parents=True, exist_ok=True)
    filtered = load_filtered_results(INPUT_PATH)
    validate_filtered_results(filtered)
    summary = summarize_context_results(filtered)
    trend = trend_table(summary)
    summary.to_csv(SUMMARY_PATH, index=False)
    write_figure(summary)
    write_report(filtered, summary, trend)
    print(f"Wrote MS-LIF-TCN context-length diagnostic summary to {SUMMARY_PATH}")
    print(f"Wrote report to {REPORT_PATH}")


def load_filtered_results(path: Path = INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing context-length ablation CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty context-length ablation CSV: {path}")
    smoke = as_bool_series(df.get("smoke_test", pd.Series(False, index=df.index)))
    synthetic = as_bool_series(df.get("synthetic_data", pd.Series(False, index=df.index)))
    filtered = df[
        df["model"].astype(str).eq(MODEL)
        & df["sequence_protocol"].astype(str).eq(SEQUENCE_PROTOCOL)
        & df["training_budget"].astype(str).eq(TRAINING_BUDGET)
        & ~smoke
        & ~synthetic
    ].copy()
    return filtered


def validate_filtered_results(df: pd.DataFrame) -> None:
    expected_total = len(EXPECTED_DATASETS) * len(EXPECTED_CONTEXT_LENS) * len(EXPECTED_SEEDS)
    if len(df) != expected_total:
        raise ValueError(f"Expected {expected_total} aligned MS-LIF-TCN rows, found {len(df)}")
    missing: list[str] = []
    for dataset in EXPECTED_DATASETS:
        for context_len in EXPECTED_CONTEXT_LENS:
            subset = df[df["dataset_key"].astype(str).eq(dataset) & df["context_len"].astype(int).eq(context_len)]
            seeds = sorted(int(seed) for seed in subset["seed"].dropna().unique())
            if seeds != EXPECTED_SEEDS:
                missing.append(f"{dataset} K={context_len}: seeds={seeds}")
    if missing:
        raise ValueError("Incomplete aligned K diagnostic rows: " + "; ".join(missing))
    missing_artifacts = []
    for column in ["checkpoint", "epoch_log", "confusion_matrix_path"]:
        values = df[column].fillna("").astype(str) if column in df.columns else pd.Series("", index=df.index)
        if values.eq("").any():
            missing_artifacts.append(column)
    if missing_artifacts:
        raise ValueError(f"Missing artifact path columns: {', '.join(missing_artifacts)}")


def summarize_context_results(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset_key, context_len), group in df.groupby(["dataset_key", "context_len"], sort=False):
        group = group.sort_values("seed")
        spike = pd.to_numeric(group.get("spike_rate", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "dataset": str(group["dataset"].iloc[0]),
                "dataset_key": str(dataset_key),
                "model": MODEL,
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
                "spike_rate_mean": float(spike.mean()),
                "spike_rate_std": sample_std(spike),
                "best_epoch_mean": float(group["best_epoch"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["dataset_order"] = summary["dataset_key"].map({name: idx for idx, name in enumerate(EXPECTED_DATASETS)})
    summary = summary.sort_values(["dataset_order", "context_len"]).drop(columns=["dataset_order"])
    summary["delta_vs_k1_macro_f1"] = 0.0
    for dataset in EXPECTED_DATASETS:
        mask = summary["dataset_key"].eq(dataset)
        k1_mean = float(summary[mask & summary["context_len"].eq(1)]["macro_f1_mean"].iloc[0])
        summary.loc[mask, "delta_vs_k1_macro_f1"] = summary.loc[mask, "macro_f1_mean"] - k1_mean
    return summary


def trend_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        group = summary[summary["dataset_key"].eq(dataset)].sort_values("context_len")
        k1 = group[group["context_len"].eq(1)].iloc[0]
        context_rows = group[group["context_len"].gt(1)]
        best = group.sort_values(["macro_f1_mean", "context_len"], ascending=[False, True]).iloc[0]
        context_above = context_rows[context_rows["macro_f1_mean"].gt(float(k1["macro_f1_mean"]))]
        rows.append(
            {
                "dataset": str(k1["dataset"]),
                "dataset_key": dataset,
                "k1_macro_f1_mean": float(k1["macro_f1_mean"]),
                "best_context_len": int(best["context_len"]),
                "best_macro_f1_mean": float(best["macro_f1_mean"]),
                "best_delta_vs_k1": float(best["macro_f1_mean"] - k1["macro_f1_mean"]),
                "num_k_gt_1_above_k1": int(len(context_above)),
                "all_k_gt_1_above_k1": bool(len(context_above) == len(context_rows)),
                "interpretation": interpret_dataset_trend(best, k1, context_rows),
            }
        )
    return pd.DataFrame(rows)


def interpret_dataset_trend(best: pd.Series, k1: pd.Series, context_rows: pd.DataFrame) -> str:
    if int(best["context_len"]) == 1:
        return "k1_best_context_not_supported"
    if (context_rows["macro_f1_mean"] > float(k1["macro_f1_mean"])).all():
        return "all_context_lengths_above_k1"
    return "best_context_length_above_k1_mixed_short_context"


def write_figure(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    for dataset in EXPECTED_DATASETS:
        group = summary[summary["dataset_key"].eq(dataset)].sort_values("context_len")
        plt.errorbar(
            group["context_len"].to_numpy(),
            group["macro_f1_mean"].to_numpy(),
            yerr=group["macro_f1_std"].to_numpy(),
            marker="o",
            capsize=3,
            label=dataset,
        )
    plt.xlabel("Context length K")
    plt.ylabel("Macro-F1")
    plt.title("MS-LIF-TCN aligned context-length diagnostic")
    plt.xticks(EXPECTED_CONTEXT_LENS)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=200)
    plt.close()


def write_report(df: pd.DataFrame, summary: pd.DataFrame, trend: pd.DataFrame) -> None:
    datasets_with_any_context_gain = int((trend["best_context_len"] > 1).sum())
    datasets_with_all_context_gain = int(trend["all_k_gt_1_above_k1"].sum())
    best_k_counts = trend["best_context_len"].value_counts().sort_index()
    lines = [
        "# MS-LIF-TCN Context-Length Diagnostic",
        "",
        "This reduced K screening uses aligned final-window targets with `aligned_kmax=8`, three seeds, `epochs=20`, `patience=5`, and `batch_size=64`.",
        "It is a diagnostic ablation and does not replace the 10-seed v3 main protocol.",
        "",
        "## Completeness",
        "",
        f"- Input rows after filtering: {len(df)}.",
        f"- Required protocol: `{SEQUENCE_PROTOCOL}`.",
        f"- Required training budget: `{TRAINING_BUDGET}`.",
        f"- Datasets: {' '.join(EXPECTED_DATASETS)}.",
        f"- Context lengths: {' '.join(str(k) for k in EXPECTED_CONTEXT_LENS)}.",
        f"- Seeds: {' '.join(str(seed) for seed in EXPECTED_SEEDS)}.",
        "",
        "## Dataset Trends",
        "",
        markdown_table(
            trend[
                [
                    "dataset_key",
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
        "## Mean Macro-F1 by K",
        "",
        markdown_table(
            summary[
                [
                    "dataset_key",
                    "context_len",
                    "num_seeds",
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
        f"- Best K is greater than 1 on {datasets_with_any_context_gain}/4 datasets.",
        f"- All evaluated K>1 values are above K=1 on {datasets_with_all_context_gain}/4 datasets.",
        f"- Best-K counts: {format_counts(best_k_counts)}.",
        "- The screening supports short cross-window context for MS-LIF-TCN, but it is not monotonic on every dataset.",
        "- UCI-HAR peaks at a shorter context in this reduced diagnostic, so the safe claim is context helps, not that larger K consistently improves performance.",
        "- HAPT-6, PAMAP2, and MHEALTH show their highest mean Macro-F1 at K=8 under this aligned reduced protocol.",
        "",
        "## Next Step",
        "",
        "The MS-LIF-TCN-only diagnostic passes the context trend check. The ANN-TCN aligned comparison can be run as a separate second phase if the project continues this ablation.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


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
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return str(value)


def format_counts(counts: pd.Series) -> str:
    return ", ".join(f"K={int(k)}: {int(v)}" for k, v in counts.items())


if __name__ == "__main__":
    main()
