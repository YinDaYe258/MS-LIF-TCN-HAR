from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


RESULTS_DIR = Path("results")
RAW_RESULTS = RESULTS_DIR / "ms_tcn_seed42_results.csv"
CANONICAL_RESULTS = RESULTS_DIR / "ms_tcn_multiseed_results.csv"
SUMMARY_CSV = RESULTS_DIR / "ms_tcn_summary.csv"
SUMMARY_TEX = RESULTS_DIR / "table_ms_tcn_summary.tex"
COMPARISON_CSV = RESULTS_DIR / "ms_tcn_baseline_comparison.csv"
REPORT_MD = RESULTS_DIR / "ms_tcn_readiness_report.md"

MODEL_ORDER = ["ms_lif_tcn", "ms_cmg_tcn"]
DATASET_ORDER = ["ucihar", "hapt6"]

BASELINE_SUMMARIES = {
    "ucihar": RESULTS_DIR / "ucihar_formal_multiseed_summary.csv",
    "hapt6": RESULTS_DIR / "hapt6_multiseed_summary.csv",
}
DISTILL_SUMMARY = RESULTS_DIR / "distill_multiseed_summary.csv"


def mean_std_text(mean: float, std: float, decimals: int = 4) -> str:
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def read_results(path: Path = RAW_RESULTS) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing MS-TCN results: {path}")
    rows = pd.read_csv(path)
    if rows.empty:
        raise ValueError(f"MS-TCN results are empty: {path}")
    for column in ("seed", "params", "best_epoch"):
        if column in rows:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    metric_columns = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "loss",
        "spike_rate",
        "best_val_macro_f1",
    ]
    for column in metric_columns:
        if column in rows:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows.sort_values(["dataset_key", "model", "seed"]).reset_index(drop=True)


def summarize_ms_tcn(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dataset_key, model), group in rows.groupby(["dataset_key", "model"], sort=False):
        records.append(
            {
                "dataset_key": dataset_key,
                "dataset": group["dataset"].iloc[0],
                "task": group["task"].iloc[0],
                "model": model,
                "num_seeds": int(group["seed"].nunique()),
                "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"].dropna().unique())),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=1),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=1),
                "weighted_f1_mean": group["weighted_f1"].mean(),
                "weighted_f1_std": group["weighted_f1"].std(ddof=1),
                "balanced_accuracy_mean": group["balanced_accuracy"].mean(),
                "balanced_accuracy_std": group["balanced_accuracy"].std(ddof=1),
                "params": int(round(group["params"].mean())),
                "spike_rate_mean": group["spike_rate"].mean(),
                "spike_rate_std": group["spike_rate"].std(ddof=1),
                "best_epoch_mean": group["best_epoch"].mean(),
            }
        )
    summary = pd.DataFrame(records)
    summary["dataset_key"] = pd.Categorical(summary["dataset_key"], categories=DATASET_ORDER, ordered=True)
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    return summary.sort_values(["dataset_key", "model"]).reset_index(drop=True)


def latex_table(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dataset": summary["dataset_key"].astype(str),
            "Model": summary["model"].astype(str),
            "Seeds": summary["num_seeds"],
            "Acc": [mean_std_text(row.accuracy_mean, row.accuracy_std) for row in summary.itertuples()],
            "Macro-F1": [mean_std_text(row.macro_f1_mean, row.macro_f1_std) for row in summary.itertuples()],
            "Weighted-F1": [
                mean_std_text(row.weighted_f1_mean, row.weighted_f1_std) for row in summary.itertuples()
            ],
            "Balanced Acc": [
                mean_std_text(row.balanced_accuracy_mean, row.balanced_accuracy_std)
                for row in summary.itertuples()
            ],
            "Params": summary["params"].astype(int),
            "Spike Rate": [mean_std_text(row.spike_rate_mean, row.spike_rate_std) for row in summary.itertuples()],
        }
    )


def load_baseline_rows() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset_key, path in BASELINE_SUMMARIES.items():
        if not path.exists():
            continue
        rows = pd.read_csv(path)
        for row in rows.itertuples(index=False):
            records.append(
                {
                    "dataset_key": dataset_key,
                    "model": str(row.model),
                    "source": path.name,
                    "macro_f1_mean": float(row.macro_f1_mean),
                    "macro_f1_std": float(row.macro_f1_std) if not pd.isna(row.macro_f1_std) else 0.0,
                    "accuracy_mean": float(row.accuracy_mean),
                    "params": int(round(float(row.params))),
                    "spike_rate_mean": float(row.spike_rate_mean) if hasattr(row, "spike_rate_mean") else 0.0,
                }
            )
    if DISTILL_SUMMARY.exists():
        rows = pd.read_csv(DISTILL_SUMMARY)
        for row in rows.itertuples(index=False):
            dataset_key = str(row.dataset_key) if hasattr(row, "dataset_key") else str(row.task)
            if dataset_key not in DATASET_ORDER:
                continue
            records.append(
                {
                    "dataset_key": dataset_key,
                    "model": str(row.model),
                    "source": DISTILL_SUMMARY.name,
                    "macro_f1_mean": float(row.macro_f1_mean),
                    "macro_f1_std": float(row.macro_f1_std) if not pd.isna(row.macro_f1_std) else 0.0,
                    "accuracy_mean": float(row.accuracy_mean),
                    "params": int(round(float(row.params))),
                    "spike_rate_mean": float(row.spike_rate_mean),
                }
            )
    return pd.DataFrame(records)


def compare_to_baselines(ms_summary: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    records = []
    if baselines.empty:
        return pd.DataFrame(records)
    target_baselines = [
        "cnn1d",
        "gru",
        "ms_cnn1d",
        "window_gru",
        "ms_lif_snn",
        "ms_cmg_lif",
        "ms_lif_snn_distill",
        "ms_cmg_lif_distill",
    ]
    for ms_row in ms_summary.itertuples(index=False):
        if str(ms_row.model) != "ms_lif_tcn":
            continue
        dataset_baselines = baselines[baselines["dataset_key"].astype(str) == str(ms_row.dataset_key)]
        for baseline_name in target_baselines:
            matches = dataset_baselines[dataset_baselines["model"].astype(str) == baseline_name]
            if matches.empty:
                continue
            baseline = matches.iloc[-1]
            records.append(
                {
                    "dataset_key": str(ms_row.dataset_key),
                    "model": str(ms_row.model),
                    "baseline_model": baseline_name,
                    "model_macro_f1_mean": float(ms_row.macro_f1_mean),
                    "baseline_macro_f1_mean": float(baseline["macro_f1_mean"]),
                    "macro_f1_diff": float(ms_row.macro_f1_mean - baseline["macro_f1_mean"]),
                    "model_params": int(ms_row.params),
                    "baseline_params": int(baseline["params"]),
                    "source": baseline["source"],
                }
            )
    return pd.DataFrame(records)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(format_markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if pd.isna(value):
        return ""
    return str(value)


def write_report(summary: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    compact_summary = pd.DataFrame(
        {
            "dataset": summary["dataset_key"].astype(str),
            "model": summary["model"].astype(str),
            "seeds": summary["seeds"],
            "macro_f1": [
                mean_std_text(row.macro_f1_mean, row.macro_f1_std) for row in summary.itertuples()
            ],
            "accuracy": [mean_std_text(row.accuracy_mean, row.accuracy_std) for row in summary.itertuples()],
            "params": summary["params"].astype(int),
            "spike_rate": [
                mean_std_text(row.spike_rate_mean, row.spike_rate_std) for row in summary.itertuples()
            ],
        }
    )
    lines = [
        "# MS-TCN Readiness Report",
        "",
        "This report audits the window-temporal TCN SNN diagnostic results.",
        "It does not claim measured neuromorphic energy or real low-power hardware behavior.",
        "",
        "## Multiseed Summary",
        "",
        dataframe_to_markdown(compact_summary),
        "",
        "## Baseline Comparisons",
        "",
        dataframe_to_markdown(comparisons),
        "",
        "## Conservative Interpretation",
        "",
        "- `ms_lif_tcn` is the strongest new model from this diagnostic stage.",
        "- On UCI-HAR, `ms_lif_tcn` has a strong seed-42 result but the three-seed mean is slightly below `window_gru`; it should not be claimed to dominate `window_gru` on UCI-HAR.",
        "- On UCI-HAR, `ms_lif_tcn` improves over the previous SNN variants and the distilled SNN variants in mean Macro-F1.",
        "- On HAPT-6, `ms_lif_tcn` exceeds the current `window_gru` mean Macro-F1 in this run, while using fewer parameters.",
        "- `ms_cmg_tcn` was only run for seed 42 and was lower than `ms_lif_tcn`; it should remain diagnostic unless more seeds are needed for an ablation.",
        "- The useful paper claim is window-level temporal aggregation, not a blanket claim that CMG always improves SNNs.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = read_results()
    rows.to_csv(CANONICAL_RESULTS, index=False)
    summary = summarize_ms_tcn(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    latex_table(summary).to_latex(SUMMARY_TEX, index=False, escape=False)
    comparisons = compare_to_baselines(summary, load_baseline_rows())
    comparisons.to_csv(COMPARISON_CSV, index=False)
    write_report(summary, comparisons)
    print(f"Saved {CANONICAL_RESULTS}")
    print(f"Saved {SUMMARY_CSV}")
    print(f"Saved {SUMMARY_TEX}")
    print(f"Saved {COMPARISON_CSV}")
    print(f"Saved {REPORT_MD}")
    print(summary)


if __name__ == "__main__":
    main()
