from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

V3_DIR = Path("results/final_paper_v3")
HAPT12_AUDIT_PATH = V3_DIR / "hapt12_transition_support_audit.csv"
BINARY_AUDIT_PATH = V3_DIR / "hapt_transition_binary_support_audit.csv"
HAPT12_INPUT_PATH = V3_DIR / "hapt12_transition_diagnostic.csv"
BINARY_INPUT_PATH = V3_DIR / "hapt_transition_binary_diagnostic.csv"
HAPT12_TABLE_PATH = V3_DIR / "table_hapt12_transition_diagnostic.tex"
BINARY_TABLE_PATH = V3_DIR / "table_hapt_transition_binary_diagnostic.tex"
HAPT12_REPORT_PATH = V3_DIR / "hapt12_transition_diagnostic_report.md"
BINARY_REPORT_PATH = V3_DIR / "hapt_transition_binary_report.md"
HAPT12_SUMMARY_PATH = V3_DIR / "hapt12_transition_diagnostic_summary.csv"
BINARY_SUMMARY_PATH = V3_DIR / "hapt_transition_binary_diagnostic_summary.csv"

EXPECTED_MODELS = ["ms_lif_snn_wide", "ms_ann_tcn", "ms_lif_tcn"]
EXPECTED_SEEDS = [42, 43, 44]
EXPECTED_CONTEXT_LENS = [2, 4]


def main() -> None:
    V3_DIR.mkdir(parents=True, exist_ok=True)
    hapt12_audit = read_csv(HAPT12_AUDIT_PATH)
    binary_audit = read_csv(BINARY_AUDIT_PATH)
    hapt12_rows = load_diagnostic_rows(HAPT12_INPUT_PATH, "hapt12")
    binary_rows = load_diagnostic_rows(BINARY_INPUT_PATH, "binary")

    hapt12_summary = summarize(hapt12_rows, include_transition_metrics=False)
    binary_summary = summarize(binary_rows, include_transition_metrics=True)
    hapt12_summary.to_csv(HAPT12_SUMMARY_PATH, index=False)
    binary_summary.to_csv(BINARY_SUMMARY_PATH, index=False)
    write_hapt12_table(hapt12_summary)
    write_binary_table(binary_summary)
    write_hapt12_report(hapt12_audit, hapt12_summary)
    write_binary_report(binary_audit, binary_summary)
    print(f"Wrote {HAPT12_REPORT_PATH}")
    print(f"Wrote {BINARY_REPORT_PATH}")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_diagnostic_rows(path: Path, task: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    smoke = as_bool_series(df.get("smoke_test", pd.Series(False, index=df.index)))
    synthetic = as_bool_series(df.get("synthetic_data", pd.Series(False, index=df.index)))
    return df[
        df["task"].astype(str).eq(task)
        & df["model"].astype(str).isin(EXPECTED_MODELS)
        & df["context_len"].astype(int).isin(EXPECTED_CONTEXT_LENS)
        & ~smoke
        & ~synthetic
    ].copy()


def summarize(df: pd.DataFrame, include_transition_metrics: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (context_len, model), group in df.groupby(["context_len", "model"], sort=False):
        group = group.sort_values("seed")
        row: dict[str, Any] = {
            "dataset": str(group["dataset"].iloc[0]),
            "dataset_key": str(group["dataset_key"].iloc[0]),
            "task": str(group["task"].iloc[0]),
            "context_len": int(context_len),
            "model": str(model),
            "training_budget": str(group["training_budget"].iloc[0]),
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
            "params": int(round(group["params"].mean())),
            "spike_rate_mean": maybe_mean(group.get("spike_rate", pd.Series(dtype=float))),
            "support_audit_status": " ".join(sorted(set(str(value) for value in group["support_audit_status"]))),
        }
        if include_transition_metrics:
            for metric in ["transition_precision", "transition_recall", "transition_f1", "transition_support"]:
                row[f"{metric}_mean"] = maybe_mean(group.get(metric, pd.Series(dtype=float)))
                if metric != "transition_support":
                    row[f"{metric}_std"] = sample_std(group.get(metric, pd.Series(dtype=float)))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["context_order"] = summary["context_len"]
    summary["model_order"] = summary["model"].map({model: idx for idx, model in enumerate(EXPECTED_MODELS)})
    return summary.sort_values(["context_order", "model_order"]).drop(columns=["context_order", "model_order"])


def write_hapt12_table(summary: pd.DataFrame) -> None:
    if summary.empty:
        HAPT12_TABLE_PATH.write_text("% No HAPT-12 transition diagnostic rows were run due support audit.\n", encoding="utf-8")
        return
    table = summary[
        [
            "context_len",
            "model",
            "num_seeds",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "params",
            "spike_rate_mean",
            "support_audit_status",
        ]
    ]
    table.to_latex(HAPT12_TABLE_PATH, index=False, escape=False, float_format=lambda value: f"{value:.4f}")


def write_binary_table(summary: pd.DataFrame) -> None:
    if summary.empty:
        BINARY_TABLE_PATH.write_text("% No HAPT transition-binary diagnostic rows were available.\n", encoding="utf-8")
        return
    table = summary[
        [
            "context_len",
            "model",
            "num_seeds",
            "macro_f1_mean",
            "macro_f1_std",
            "transition_precision_mean",
            "transition_recall_mean",
            "transition_f1_mean",
            "params",
            "spike_rate_mean",
            "support_audit_status",
        ]
    ]
    table.to_latex(BINARY_TABLE_PATH, index=False, escape=False, float_format=lambda value: f"{value:.4f}")


def write_hapt12_report(audit: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# HAPT-12 Transition Multiclass Diagnostic",
        "",
        "This is a supplementary diagnostic only. HAPT-12 is not used as a primary main-result dataset.",
        "K-window filtering changes transition-class support, so support counts are reported before any metric interpretation.",
        "",
        "## Support Audit",
        "",
        markdown_table(hapt12_support_overview(audit)),
        "",
    ]
    if summary.empty:
        lines.extend(
            [
                "## Diagnostic Runs",
                "",
                "No HAPT-12 multiclass diagnostic rows were available. By default, low-support transition settings are skipped rather than promoted to paper evidence.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Diagnostic Summary",
                "",
                markdown_table(
                    summary[
                        [
                            "context_len",
                            "model",
                            "num_seeds",
                            "macro_f1_mean",
                            "macro_f1_std",
                            "balanced_accuracy_mean",
                            "support_audit_status",
                        ]
                    ]
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "- Treat this table as transition-aware supplementary evidence only.",
            "- Do not claim HAPT-12 as a complete primary 12-class main result if transition support is low.",
            "- Do not infer measured energy or neuromorphic power from this diagnostic.",
            "",
        ]
    )
    HAPT12_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_binary_report(audit: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# HAPT Transition Binary Diagnostic",
        "",
        "This is a supplementary diagnostic only. The task maps labels 1-6 to non-transition and labels 7-12 to transition.",
        "K-window filtering changes positive-class support, so transition precision, recall, F1, and support are reported.",
        "",
        "## Support Audit",
        "",
        markdown_table(binary_support_overview(audit)),
        "",
    ]
    if summary.empty:
        lines.extend(["## Diagnostic Summary", "", "No transition-binary diagnostic rows were available.", ""])
    else:
        lines.extend(
            [
                "## Diagnostic Summary",
                "",
                markdown_table(
                    summary[
                        [
                            "context_len",
                            "model",
                            "num_seeds",
                            "macro_f1_mean",
                            "macro_f1_std",
                            "transition_precision_mean",
                            "transition_recall_mean",
                            "transition_f1_mean",
                            "support_audit_status",
                        ]
                    ]
                ),
                "",
                "## Interpretation Boundary",
                "",
                "- If MS-LIF-TCN and MS-ANN-TCN both improve over the SNN-wide baseline, the safe claim is that temporal context helps transition-sensitive settings.",
                "- If the transition class is weakly supported, treat Macro-F1 and transition recall as diagnostic rather than primary evidence.",
                "- Do not claim measured energy savings from this diagnostic.",
                "",
            ]
        )
    BINARY_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def hapt12_support_overview(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    test = audit[audit["split"].astype(str).eq("test") & audit["is_transition"].astype(bool)].copy()
    if test.empty:
        return pd.DataFrame()
    rows = []
    for context_len, group in test.groupby("context_len", sort=True):
        rows.append(
            {
                "context_len": int(context_len),
                "min_transition_test_sequence_support": int(group["sequence_support"].min()),
                "low_transition_classes": " ".join(group[group["low_support_flag"].astype(bool)]["class_name"].astype(str)),
                "zero_transition_classes": " ".join(group[group["zero_support_flag"].astype(bool)]["class_name"].astype(str)),
                "support_ok_for_multiclass": not bool(group["low_support_flag"].astype(bool).any()),
            }
        )
    return pd.DataFrame(rows)


def binary_support_overview(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    rows = []
    test = audit[audit["split"].astype(str).eq("test") & audit["class_id"].astype(int).eq(1)].copy()
    for _, row in test.sort_values("context_len").iterrows():
        rows.append(
            {
                "context_len": int(row["context_len"]),
                "transition_test_sequence_support": int(row["sequence_support"]),
                "weak_positive_support": bool(row["low_support_flag"]),
                "zero_positive_support": bool(row["zero_support_flag"]),
            }
        )
    return pd.DataFrame(rows)


def sample_std(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) <= 1:
        return 0.0
    return float(numeric.std(ddof=1))


def maybe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) == 0:
        return float("nan")
    return float(numeric.mean())


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
