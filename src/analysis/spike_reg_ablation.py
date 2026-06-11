from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V3_DIR = Path("results/final_paper_v3")
INPUT_PATH = V3_DIR / "spike_reg_ablation.csv"
SUMMARY_PATH = V3_DIR / "spike_reg_ablation_summary.csv"
TABLE_PATH = V3_DIR / "table_spike_reg_ablation.tex"
FIG_SPIKE_PATH = V3_DIR / "fig_macro_f1_vs_spike_rate.png"
FIG_SYNOPS_PATH = V3_DIR / "fig_macro_f1_vs_synops_proxy.png"
REPORT_PATH = V3_DIR / "spike_reg_ablation_report.md"

EXPECTED_DATASETS = ["ucihar", "pamap2"]
EXPECTED_MODELS = ["ms_lif_snn_wide", "ms_lif_tcn"]
EXPECTED_LAMBDAS = [0.0, 1e-5, 1e-4, 1e-3]
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
    write_tradeoff_figure(summary, "spike_rate_mean", FIG_SPIKE_PATH, "Spike rate")
    write_tradeoff_figure(summary, "synops_proxy_mean", FIG_SYNOPS_PATH, "SynOps proxy")
    write_report(summary)
    print(f"Wrote spike-reg summary to {SUMMARY_PATH}")
    print(f"Wrote spike-reg report to {REPORT_PATH}")


def load_filtered(path: Path = INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing spike-reg ablation CSV: {path}")
    df = pd.read_csv(path)
    smoke = as_bool_series(df.get("smoke_test", pd.Series(False, index=df.index)))
    synthetic = as_bool_series(df.get("synthetic_data", pd.Series(False, index=df.index)))
    lambdas = pd.to_numeric(df["spike_reg_lambda"], errors="coerce")
    lambda_mask = pd.Series(False, index=df.index)
    for value in EXPECTED_LAMBDAS:
        lambda_mask |= lambdas.sub(float(value)).abs() < 1e-15
    return df[
        df["ablation"].astype(str).eq("spike_reg")
        & df["dataset_key"].astype(str).isin(EXPECTED_DATASETS)
        & df["model"].astype(str).isin(EXPECTED_MODELS)
        & lambda_mask
        & df["sequence_protocol"].astype(str).eq(SEQUENCE_PROTOCOL)
        & df["training_budget"].astype(str).eq(TRAINING_BUDGET)
        & ~smoke
        & ~synthetic
    ].copy()


def validate(df: pd.DataFrame) -> None:
    expected_total = len(EXPECTED_DATASETS) * len(EXPECTED_MODELS) * len(EXPECTED_LAMBDAS) * len(EXPECTED_SEEDS)
    if len(df) != expected_total:
        raise ValueError(f"Expected {expected_total} spike-reg rows, found {len(df)}")
    missing: list[str] = []
    for dataset in EXPECTED_DATASETS:
        for model in EXPECTED_MODELS:
            for spike_lambda in EXPECTED_LAMBDAS:
                mask = (
                    df["dataset_key"].astype(str).eq(dataset)
                    & df["model"].astype(str).eq(model)
                    & (pd.to_numeric(df["spike_reg_lambda"], errors="coerce").sub(spike_lambda).abs() < 1e-15)
                )
                seeds = sorted(int(seed) for seed in df[mask]["seed"].dropna().unique())
                if seeds != EXPECTED_SEEDS:
                    missing.append(f"{dataset} {model} lambda={format_lambda(spike_lambda)} seeds={seeds}")
    if missing:
        raise ValueError("Incomplete spike-reg rows: " + "; ".join(missing))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset_key, model, spike_lambda), group in df.groupby(["dataset_key", "model", "spike_reg_lambda"], sort=False):
        group = group.sort_values("seed")
        hidden_dim = int(round(pd.to_numeric(group["hidden_dim"], errors="coerce").mean()))
        context_len = int(round(pd.to_numeric(group["context_len"], errors="coerce").mean()))
        window_size = int(round(pd.to_numeric(group["window_size"], errors="coerce").mean()))
        spike_rate_mean = float(pd.to_numeric(group["spike_rate"], errors="coerce").mean())
        spike_count_proxy = spike_rate_mean * context_len * window_size * hidden_dim
        rows.append(
            {
                "dataset": str(group["dataset"].iloc[0]),
                "dataset_key": str(dataset_key),
                "model": str(model),
                "spike_reg_lambda": float(spike_lambda),
                "spike_reg_lambda_label": format_lambda(float(spike_lambda)),
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
                "spike_rate_mean": spike_rate_mean,
                "spike_rate_std": sample_std(group["spike_rate"]),
                "spike_count_proxy_mean": float(spike_count_proxy),
                "synops_proxy_mean": float(spike_count_proxy * hidden_dim),
                "params": int(round(group["params"].mean())),
                "context_len": context_len,
                "window_size": window_size,
                "hidden_dim": hidden_dim,
                "best_epoch_mean": float(group["best_epoch"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["dataset_order"] = summary["dataset_key"].map({name: idx for idx, name in enumerate(EXPECTED_DATASETS)})
    summary["model_order"] = summary["model"].map({name: idx for idx, name in enumerate(EXPECTED_MODELS)})
    summary["lambda_order"] = summary["spike_reg_lambda"].map({value: idx for idx, value in enumerate(EXPECTED_LAMBDAS)})
    summary = summary.sort_values(["dataset_order", "model_order", "lambda_order"]).drop(
        columns=["dataset_order", "model_order", "lambda_order"]
    )
    summary["delta_macro_f1_vs_lambda0"] = 0.0
    summary["delta_spike_rate_vs_lambda0"] = 0.0
    summary["spike_rate_reduction_pct_vs_lambda0"] = 0.0
    for dataset in EXPECTED_DATASETS:
        for model in EXPECTED_MODELS:
            mask = summary["dataset_key"].eq(dataset) & summary["model"].eq(model)
            base = summary[mask & np.isclose(summary["spike_reg_lambda"].astype(float), 0.0)]
            if base.empty:
                continue
            base_macro = float(base["macro_f1_mean"].iloc[0])
            base_spike = float(base["spike_rate_mean"].iloc[0])
            summary.loc[mask, "delta_macro_f1_vs_lambda0"] = summary.loc[mask, "macro_f1_mean"] - base_macro
            summary.loc[mask, "delta_spike_rate_vs_lambda0"] = summary.loc[mask, "spike_rate_mean"] - base_spike
            if base_spike > 0:
                summary.loc[mask, "spike_rate_reduction_pct_vs_lambda0"] = (
                    (base_spike - summary.loc[mask, "spike_rate_mean"]) / base_spike * 100.0
                )
    return summary


def write_latex_table(summary: pd.DataFrame) -> None:
    table = summary[
        [
            "dataset_key",
            "model",
            "spike_reg_lambda_label",
            "num_seeds",
            "macro_f1_mean",
            "macro_f1_std",
            "spike_rate_mean",
            "spike_rate_std",
            "delta_macro_f1_vs_lambda0",
            "spike_rate_reduction_pct_vs_lambda0",
        ]
    ].copy()
    table.to_latex(TABLE_PATH, index=False, escape=False, float_format=lambda value: f"{value:.4f}")


def write_tradeoff_figure(summary: pd.DataFrame, x_column: str, path: Path, xlabel: str) -> None:
    fig, axes = plt.subplots(1, len(EXPECTED_DATASETS), figsize=(11, 4), sharey=True)
    if len(EXPECTED_DATASETS) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, EXPECTED_DATASETS, strict=True):
        subset = summary[summary["dataset_key"].eq(dataset)]
        for model in EXPECTED_MODELS:
            group = subset[subset["model"].eq(model)].sort_values("spike_reg_lambda")
            axis.plot(group[x_column], group["macro_f1_mean"], marker="o", label=model)
            for _, row in group.iterrows():
                axis.annotate(
                    str(row["spike_reg_lambda_label"]),
                    (float(row[x_column]), float(row["macro_f1_mean"])),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
        axis.set_title(dataset)
        axis.set_xlabel(xlabel)
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Macro-F1")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(f"Spike regularization trade-off: Macro-F1 vs {xlabel}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    tradeoff = tradeoff_rows(summary)
    lines = [
        "# Spike-Regularization Diagnostic",
        "",
        "This reduced spike-regularization sweep uses fixed `K=8`, three seeds, `epochs=20`, `patience=5`, and `batch_size=64`.",
        "It is a diagnostic ablation and does not replace the 10-seed v3 main protocol.",
        "The SynOps values are rough algorithmic proxies derived from spike rate and hidden dimension; they are not measured energy.",
        "",
        "## Completeness",
        "",
        "- Input rows after filtering: 48.",
        f"- Required protocol: `{SEQUENCE_PROTOCOL}`.",
        f"- Required training budget: `{TRAINING_BUDGET}`.",
        f"- Datasets: {' '.join(EXPECTED_DATASETS)}.",
        f"- Models: {' '.join(EXPECTED_MODELS)}.",
        f"- Lambda values: {' '.join(format_lambda(value) for value in EXPECTED_LAMBDAS)}.",
        f"- Seeds: {' '.join(str(seed) for seed in EXPECTED_SEEDS)}.",
        "",
        "## Accuracy-Sparsity Trade-Off",
        "",
        markdown_table(tradeoff),
        "",
        "## Mean Metrics by Lambda",
        "",
        markdown_table(
            summary[
                [
                    "dataset_key",
                    "model",
                    "spike_reg_lambda_label",
                    "macro_f1_mean",
                    "macro_f1_std",
                    "spike_rate_mean",
                    "spike_rate_std",
                    "delta_macro_f1_vs_lambda0",
                    "spike_rate_reduction_pct_vs_lambda0",
                    "synops_proxy_mean",
                ]
            ]
        ),
        "",
        "## Interpretation Rules",
        "",
        "- A useful trade-off requires lower spike rate while preserving Macro-F1 near the no-regularization baseline.",
        "- If strong regularization lowers spike rate but materially reduces Macro-F1, it should be described as over-regularization.",
        "- These results characterize sparsity behavior only; they do not support measured neuromorphic or hardware energy claims.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def tradeoff_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for model in EXPECTED_MODELS:
            group = summary[summary["dataset_key"].eq(dataset) & summary["model"].eq(model)].copy()
            base = group[np.isclose(group["spike_reg_lambda"].astype(float), 0.0)].iloc[0]
            candidates = group[~np.isclose(group["spike_reg_lambda"].astype(float), 0.0)].copy()
            candidates["macro_drop"] = float(base["macro_f1_mean"]) - candidates["macro_f1_mean"]
            candidates["spike_drop"] = float(base["spike_rate_mean"]) - candidates["spike_rate_mean"]
            base_spike = float(base["spike_rate_mean"])
            candidates["spike_reduction_pct"] = 0.0
            if base_spike > 0:
                candidates["spike_reduction_pct"] = candidates["spike_drop"] / base_spike * 100.0
            macro_gain = candidates[(candidates["spike_drop"] > 0) & (candidates["macro_drop"] <= 0)]
            useful = candidates[(candidates["spike_reduction_pct"] >= 1.0) & (candidates["macro_drop"] <= 0.01)]
            if not macro_gain.empty:
                best = macro_gain.sort_values(["macro_f1_mean", "spike_reduction_pct"], ascending=[False, False]).iloc[0]
                if float(best["spike_reduction_pct"]) >= 1.0:
                    judgement = "accuracy_gain_with_spike_reduction"
                else:
                    judgement = "accuracy_gain_with_small_spike_reduction"
            elif not useful.empty:
                best = useful.sort_values(["spike_reduction_pct", "macro_f1_mean"], ascending=[False, False]).iloc[0]
                judgement = "useful_tradeoff"
            else:
                best = candidates.sort_values(["macro_f1_mean", "spike_reduction_pct"], ascending=[False, False]).iloc[0]
                if float(best["spike_drop"]) <= 0:
                    judgement = "limited_spike_reduction"
                elif float(best["spike_reduction_pct"]) < 1.0:
                    judgement = "small_spike_reduction"
                elif float(best["macro_drop"]) > 0.02:
                    judgement = "accuracy_cost_high"
                else:
                    judgement = "mixed_tradeoff"
            strong = candidates[np.isclose(candidates["spike_reg_lambda"].astype(float), 1e-3)]
            strong_note = ""
            if not strong.empty:
                strong_drop = float(base["macro_f1_mean"]) - float(strong["macro_f1_mean"].iloc[0])
                strong_spike_drop = float(base["spike_rate_mean"]) - float(strong["spike_rate_mean"].iloc[0])
                if strong_spike_drop > 0 and strong_drop > 0.02:
                    strong_note = "lambda_1e-3_reduces_spikes_but_hurts_macro_f1"
            rows.append(
                {
                    "dataset_key": dataset,
                    "model": model,
                    "lambda0_macro_f1": float(base["macro_f1_mean"]),
                    "lambda0_spike_rate": float(base["spike_rate_mean"]),
                    "selected_lambda": str(best["spike_reg_lambda_label"]),
                    "selected_macro_f1": float(best["macro_f1_mean"]),
                    "selected_spike_rate": float(best["spike_rate_mean"]),
                    "macro_delta_vs_lambda0": float(best["macro_f1_mean"]) - float(base["macro_f1_mean"]),
                    "spike_rate_delta_vs_lambda0": float(best["spike_rate_mean"]) - float(base["spike_rate_mean"]),
                    "judgement": judgement,
                    "strong_regularization_note": strong_note,
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


def format_lambda(value: float) -> str:
    if np.isclose(float(value), 0.0, atol=1e-15):
        return "0"
    return f"{float(value):.0e}"


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
