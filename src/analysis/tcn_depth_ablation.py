from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V3_DIR = Path("results/final_paper_v3")
INPUT_PATH = V3_DIR / "tcn_depth_ablation.csv"
SUMMARY_PATH = V3_DIR / "tcn_depth_ablation_summary.csv"
TABLE_PATH = V3_DIR / "table_tcn_depth_ablation.tex"
FIGURE_PATH = V3_DIR / "fig_tcn_depth_vs_macro_f1.png"
REPORT_PATH = V3_DIR / "tcn_depth_ablation_report.md"

EXPECTED_DATASETS = ["ucihar", "hapt6", "pamap2", "mhealth"]
EXPECTED_VARIANTS = ["tcn0", "tcn1", "tcn2", "tcn3"]
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
    print(f"Wrote TCN-depth summary to {SUMMARY_PATH}")
    print(f"Wrote TCN-depth report to {REPORT_PATH}")


def load_filtered(path: Path = INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing TCN-depth ablation CSV: {path}")
    df = pd.read_csv(path)
    smoke = as_bool_series(df.get("smoke_test", pd.Series(False, index=df.index)))
    synthetic = as_bool_series(df.get("synthetic_data", pd.Series(False, index=df.index)))
    return df[
        df["ablation"].astype(str).eq("tcn_depth")
        & df["variant_key"].astype(str).isin(EXPECTED_VARIANTS)
        & df["sequence_protocol"].astype(str).eq(SEQUENCE_PROTOCOL)
        & df["training_budget"].astype(str).eq(TRAINING_BUDGET)
        & ~smoke
        & ~synthetic
    ].copy()


def validate(df: pd.DataFrame) -> None:
    expected_total = len(EXPECTED_DATASETS) * len(EXPECTED_VARIANTS) * len(EXPECTED_SEEDS)
    if len(df) != expected_total:
        raise ValueError(f"Expected {expected_total} TCN-depth rows, found {len(df)}")
    missing: list[str] = []
    for dataset in EXPECTED_DATASETS:
        for variant in EXPECTED_VARIANTS:
            subset = df[df["dataset_key"].astype(str).eq(dataset) & df["variant_key"].astype(str).eq(variant)]
            seeds = sorted(int(seed) for seed in subset["seed"].dropna().unique())
            if seeds != EXPECTED_SEEDS:
                missing.append(f"{dataset} {variant}: seeds={seeds}")
    if missing:
        raise ValueError("Incomplete TCN-depth rows: " + "; ".join(missing))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset_key, variant_key), group in df.groupby(["dataset_key", "variant_key"], sort=False):
        group = group.sort_values("seed")
        rows.append(
            {
                "dataset": str(group["dataset"].iloc[0]),
                "dataset_key": str(dataset_key),
                "model": "ms_lif_tcn",
                "variant": str(group["variant"].iloc[0]),
                "variant_key": str(variant_key),
                "tcn_layers": int(group["tcn_layers"].iloc[0]),
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
    summary["delta_vs_tcn0_macro_f1"] = 0.0
    for dataset in EXPECTED_DATASETS:
        mask = summary["dataset_key"].eq(dataset)
        tcn0 = float(summary[mask & summary["variant_key"].eq("tcn0")]["macro_f1_mean"].iloc[0])
        summary.loc[mask, "delta_vs_tcn0_macro_f1"] = summary.loc[mask, "macro_f1_mean"] - tcn0
    return summary


def write_latex_table(summary: pd.DataFrame) -> None:
    table = summary[
        [
            "dataset_key",
            "variant",
            "tcn_layers",
            "num_seeds",
            "macro_f1_mean",
            "macro_f1_std",
            "delta_vs_tcn0_macro_f1",
            "params",
            "spike_rate_mean",
        ]
    ].copy()
    table.to_latex(TABLE_PATH, index=False, escape=False, float_format=lambda value: f"{value:.4f}")


def write_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes_flat = axes.reshape(-1)
    for axis, dataset in zip(axes_flat, EXPECTED_DATASETS, strict=True):
        group = summary[summary["dataset_key"].eq(dataset)].sort_values("tcn_layers")
        axis.errorbar(
            group["tcn_layers"].to_numpy(),
            group["macro_f1_mean"].to_numpy(),
            yerr=group["macro_f1_std"].to_numpy(),
            marker="o",
            capsize=3,
        )
        axis.set_title(dataset)
        axis.set_xticks([0, 1, 2, 3])
        axis.grid(True, alpha=0.3)
        axis.set_ylabel("Macro-F1")
    for axis in axes_flat[-2:]:
        axis.set_xlabel("TCN layers")
    fig.suptitle("MS-LIF-TCN depth diagnostic")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    trends = trend_rows(summary)
    lines = [
        "# TCN-Depth Diagnostic",
        "",
        "This reduced TCN-depth ablation uses fixed `K=8`, three seeds, `epochs=20`, `patience=5`, and `batch_size=64`.",
        "It is a diagnostic ablation and does not replace the 10-seed v3 main protocol.",
        "",
        "`tcn0_no_temporal_context` keeps the multi-scale spiking window encoder and classifier interface, but replaces the causal window-level TCN with an identity mapping over per-window spike representations.",
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
        "## Mean Macro-F1 by TCN Depth",
        "",
        markdown_table(
            summary[
                [
                    "dataset_key",
                    "variant",
                    "tcn_layers",
                    "macro_f1_mean",
                    "macro_f1_std",
                    "delta_vs_tcn0_macro_f1",
                    "params",
                    "spike_rate_mean",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
    ]
    datasets_with_tcn_gain = int((trends["best_tcn_layers"] > 0).sum())
    datasets_main_best = int((trends["best_tcn_layers"] == 2).sum())
    lines.append(f"- At least one TCN depth is better than TCN-0 on `{datasets_with_tcn_gain}/4` datasets.")
    lines.append(f"- The two-layer main depth is the best reduced-budget mean on `{datasets_main_best}/4` datasets.")
    lines.append("- If one TCN layer is better on some datasets, this should be described as shallow context being sufficient rather than as a failure.")
    lines.append("- This table should be interpreted alongside the aligned K diagnostic; it is not a claim of statistical optimality.")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def trend_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in EXPECTED_DATASETS:
        group = summary[summary["dataset_key"].eq(dataset)].sort_values("tcn_layers")
        tcn0 = group[group["variant_key"].eq("tcn0")].iloc[0]
        best = group.sort_values(["macro_f1_mean", "tcn_layers"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "dataset_key": dataset,
                "tcn0_macro_f1_mean": float(tcn0["macro_f1_mean"]),
                "best_tcn_layers": int(best["tcn_layers"]),
                "best_variant": str(best["variant"]),
                "best_macro_f1_mean": float(best["macro_f1_mean"]),
                "best_delta_vs_tcn0": float(best["macro_f1_mean"] - tcn0["macro_f1_mean"]),
                "tcn2_delta_vs_tcn0": float(
                    group[group["variant_key"].eq("tcn2")]["macro_f1_mean"].iloc[0] - tcn0["macro_f1_mean"]
                ),
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
