from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


RESULTS_DIR = Path("results")
INPUT_CSV = RESULTS_DIR / "ms_lif_tcn_plus_diagnostic.csv"
MULTISEED_CSV = RESULTS_DIR / "ms_lif_tcn_plus_multiseed.csv"
SUMMARY_CSV = RESULTS_DIR / "ms_lif_tcn_plus_summary.csv"
SUMMARY_TEX = RESULTS_DIR / "table_ms_lif_tcn_plus.tex"
COMPARISON_CSV = RESULTS_DIR / "ms_lif_tcn_plus_comparison.csv"
REPORT_MD = RESULTS_DIR / "ms_lif_tcn_plus_readiness_report.md"


def mean_std_text(mean: float, std: float, decimals: int = 4) -> str:
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dataset_key, variant), group in rows.groupby(["dataset_key", "variant"], sort=False):
        records.append(
            {
                "dataset_key": dataset_key,
                "dataset": group["dataset"].iloc[0],
                "task": group["task"].iloc[0],
                "model": group["model"].iloc[0],
                "variant": variant,
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
                "best_val_macro_f1_mean": group["best_val_macro_f1"].mean(),
            }
        )
    return pd.DataFrame(records).sort_values(["dataset_key", "variant"]).reset_index(drop=True)


def latex_table(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dataset": summary["dataset_key"],
            "Variant": summary["variant"],
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


def baseline_comparisons(plus_summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    baseline_files = [
        RESULTS_DIR / "ms_tcn_summary.csv",
        RESULTS_DIR / "ucihar_formal_multiseed_summary.csv",
        RESULTS_DIR / "hapt6_multiseed_summary.csv",
    ]
    baseline_rows = []
    for path in baseline_files:
        if not path.exists():
            continue
        rows = pd.read_csv(path)
        for row in rows.itertuples(index=False):
            dataset_key = str(getattr(row, "dataset_key", "ucihar" if "ucihar" in path.name else "hapt6"))
            baseline_rows.append(
                {
                    "dataset_key": dataset_key,
                    "model": str(row.model),
                    "macro_f1_mean": float(row.macro_f1_mean),
                    "params": int(round(float(row.params))),
                    "source": path.name,
                }
            )
    baselines = pd.DataFrame(baseline_rows)
    for row in plus_summary.itertuples(index=False):
        if row.variant != "attn_supcon_0.1":
            continue
        candidates = baselines[baselines["dataset_key"].astype(str).eq(str(row.dataset_key))]
        for baseline_name in ("ms_lif_tcn", "window_gru", "cnn1d", "ms_lif_snn", "ms_cmg_lif"):
            match = candidates[candidates["model"].astype(str).eq(baseline_name)]
            if match.empty:
                continue
            baseline = match.iloc[-1]
            records.append(
                {
                    "dataset_key": row.dataset_key,
                    "variant": row.variant,
                    "baseline_model": baseline_name,
                    "variant_macro_f1_mean": row.macro_f1_mean,
                    "baseline_macro_f1_mean": baseline["macro_f1_mean"],
                    "macro_f1_diff": row.macro_f1_mean - baseline["macro_f1_mean"],
                    "variant_params": row.params,
                    "baseline_params": baseline["params"],
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
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if pd.isna(value):
        return ""
    return str(value)


def write_report(summary: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    compact = pd.DataFrame(
        {
            "dataset": summary["dataset_key"],
            "variant": summary["variant"],
            "seeds": summary["seeds"],
            "macro_f1": [mean_std_text(row.macro_f1_mean, row.macro_f1_std) for row in summary.itertuples()],
            "accuracy": [mean_std_text(row.accuracy_mean, row.accuracy_std) for row in summary.itertuples()],
            "params": summary["params"].astype(int),
            "spike_rate": [
                mean_std_text(row.spike_rate_mean, row.spike_rate_std) for row in summary.itertuples()
            ],
        }
    )
    lines = [
        "# MS-LIF-TCN+ Readiness Report",
        "",
        "This report summarizes the attention + supervised contrastive diagnostic.",
        "It does not claim measured neuromorphic power.",
        "",
        "## Summary",
        "",
        dataframe_to_markdown(compact),
        "",
        "## Comparisons",
        "",
        dataframe_to_markdown(comparisons),
        "",
        "## Conservative Interpretation",
        "",
        "- Seed-42 diagnostics show large gains from lightweight window attention plus supervised contrastive loss.",
        "- The selected variant is `attn_supcon_0.1`, chosen by validation Macro-F1 rather than test Macro-F1.",
        "- On HAPT-6, `attn_supcon_0.1` improves over `ms_lif_tcn` and `window_gru` in three-seed mean Macro-F1.",
        "- On UCI-HAR, `attn_supcon_0.1` has a strong seed-42 result but poor seed-44 behavior; its three-seed mean is not an improvement over `ms_lif_tcn`.",
        "- The final paper should present MS-LIF-TCN as the robust main model and MS-LIF-TCN+ as an HAPT-strong enhanced diagnostic unless further UCI stabilization is done.",
        "- Weighted focal and naive augmentation did not consistently improve the diagnostic results.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing diagnostic CSV: {INPUT_CSV}")
    rows = pd.read_csv(INPUT_CSV)
    for column in (
        "seed",
        "params",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "spike_rate",
        "best_epoch",
        "best_val_macro_f1",
    ):
        if column in rows:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    multiseed = rows[(rows["variant"] == "attn_supcon_0.1") & rows["seed"].isin([42, 43, 44])].copy()
    multiseed.to_csv(MULTISEED_CSV, index=False)
    summary = summarize(multiseed)
    summary.to_csv(SUMMARY_CSV, index=False)
    latex_table(summary).to_latex(SUMMARY_TEX, index=False, escape=False)
    comparisons = baseline_comparisons(summary)
    comparisons.to_csv(COMPARISON_CSV, index=False)
    write_report(summary, comparisons)
    print(f"Saved {MULTISEED_CSV}")
    print(f"Saved {SUMMARY_CSV}")
    print(f"Saved {SUMMARY_TEX}")
    print(f"Saved {COMPARISON_CSV}")
    print(f"Saved {REPORT_MD}")
    print(summary)


if __name__ == "__main__":
    main()
