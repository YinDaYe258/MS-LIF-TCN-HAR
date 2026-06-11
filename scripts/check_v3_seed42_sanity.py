"""Generate a seed-42 sanity report for final_paper_v3 external datasets.

This check is intentionally narrow: it validates that PAMAP2 and MHEALTH
seed-42 full-main runs completed cleanly before launching the 10-seed run.
It does not mark the v3 result package as paper-ready.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


MAIN_MODELS = [
    "cnn1d",
    "window_gru",
    "ms_lif_snn",
    "ms_lif_snn_wide",
    "ms_ann_tcn",
    "ms_lif_tcn",
]

EXTERNAL_DATASETS = ["pamap2", "mhealth"]
NON_SPIKING_MODELS = {"cnn1d", "window_gru", "ms_ann_tcn"}
SNN_MODELS = {"ms_lif_snn", "ms_lif_snn_wide", "ms_lif_tcn"}


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def _exists(path_value: object) -> bool:
    if pd.isna(path_value):
        return False
    path = str(path_value).strip()
    return bool(path) and Path(path).exists()


def _format_float(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _load_low_support(results_dir: Path, dataset_key: str) -> tuple[list[int], list[int], int | None]:
    inspection_path = results_dir / f"{dataset_key}_inspection.json"
    if not inspection_path.exists():
        return [], [], None
    data = json.loads(inspection_path.read_text(encoding="utf-8"))
    return (
        list(data.get("classes_with_low_support_k8", [])),
        list(data.get("classes_with_zero_support_k8", [])),
        data.get("min_test_sequences_per_class_k8"),
    )


def build_report(results_dir: Path, output_path: Path) -> tuple[str, list[str]]:
    raw_path = results_dir / "main_results_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing main results file: {raw_path}")

    df = pd.read_csv(raw_path)
    subset = df[
        df["dataset_key"].isin(EXTERNAL_DATASETS)
        & df["seed"].eq(42)
        & df["model"].isin(MAIN_MODELS)
    ].copy()

    issues: list[str] = []
    warnings: list[str] = []
    table_rows: list[list[str]] = []

    for dataset_key in EXTERNAL_DATASETS:
        ddf = subset[subset["dataset_key"].eq(dataset_key)]
        present = set(ddf["model"])
        missing = [model for model in MAIN_MODELS if model not in present]
        extra_count = len(ddf) - len(present)
        if missing:
            issues.append(f"{dataset_key}: missing models: {', '.join(missing)}")
        if extra_count:
            issues.append(f"{dataset_key}: duplicate model rows detected for seed 42")

        if not ddf.empty:
            synthetic = _bool_series(ddf["synthetic_data"])
            smoke = _bool_series(ddf["smoke_test"])
            if synthetic.isna().any() or synthetic.any():
                issues.append(f"{dataset_key}: synthetic_data is not all False")
            if smoke.isna().any() or smoke.any():
                issues.append(f"{dataset_key}: smoke_test is not all False")

        ordered = ddf.set_index("model", drop=False)
        for model_name in MAIN_MODELS:
            if model_name not in ordered.index:
                continue
            row = ordered.loc[model_name]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            model = str(row["model"])
            missing_paths = [
                column
                for column in ["checkpoint", "epoch_log", "confusion_matrix_path"]
                if not _exists(row.get(column))
            ]
            if missing_paths:
                issues.append(f"{dataset_key}/{model}: missing path(s): {', '.join(missing_paths)}")

            spike_rate = row.get("spike_rate")
            if model in NON_SPIKING_MODELS and not pd.isna(spike_rate):
                issues.append(f"{dataset_key}/{model}: non-spiking model has non-empty spike_rate")
            if model in SNN_MODELS and pd.isna(spike_rate):
                issues.append(f"{dataset_key}/{model}: SNN model has empty spike_rate")

            macro_f1 = float(row["macro_f1"])
            if model == "ms_lif_tcn" and macro_f1 < 0.30:
                issues.append(f"{dataset_key}/ms_lif_tcn: Macro-F1 below 0.30 hard-stop threshold")
            if model in NON_SPIKING_MODELS and macro_f1 < 0.25:
                issues.append(f"{dataset_key}/{model}: ANN baseline Macro-F1 below 0.25 sanity threshold")

            table_rows.append(
                [
                    dataset_key,
                    model,
                    _format_float(row["macro_f1"]),
                    _format_float(row["accuracy"]),
                    str(int(row["params"])),
                    _format_float(spike_rate),
                    str(int(row["best_epoch"])),
                ]
            )

        low_support, zero_support, min_support = _load_low_support(results_dir, dataset_key)
        if zero_support:
            issues.append(f"{dataset_key}: zero-support K=8 test classes: {zero_support}")
        if low_support:
            warnings.append(
                f"{dataset_key}: low-support K=8 test classes {low_support}; "
                f"minimum test sequence support is {min_support}"
            )

    if issues:
        conclusion = "FAIL"
        conclusion_detail = "Stop before 10-seed main runs and fix the listed issues."
    elif warnings:
        conclusion = "WARNING"
        conclusion_detail = (
            "Seed-42 sanity passed model/protocol checks, but low-support classes "
            "must be handled in per-class analysis and paper limitations."
        )
    else:
        conclusion = "PASS"
        conclusion_detail = "Seed-42 sanity passed; it is reasonable to proceed to 10-seed main runs."

    lines = [
        "# v3 Seed-42 Sanity Report",
        "",
        f"Conclusion: **{conclusion}**",
        "",
        conclusion_detail,
        "",
        "This report is a pre-flight check for PAMAP2 and MHEALTH seed-42 full-main runs. "
        "It is not a final v3 paper result table.",
        "",
        "## Seed-42 Results",
        "",
        _markdown_table(
            ["dataset", "model", "macro_f1", "accuracy", "params", "spike_rate", "best_epoch"],
            table_rows,
        ),
        "",
        "## Checks",
        "",
        "- Expected datasets: pamap2, mhealth",
        "- Expected models per dataset: " + ", ".join(MAIN_MODELS),
        "- Required flags: synthetic_data=False and smoke_test=False",
        "- Required artifacts: checkpoint, epoch_log, confusion_matrix_path",
        "- Spike convention: non-spiking models have blank spike_rate; SNN models report spike_rate",
        "- Random-level guard: 12-class Macro-F1 near 0.08 is random; MS-LIF-TCN must be >= 0.30",
        "",
    ]

    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    if issues:
        lines.extend(["## Issues", ""])
        lines.extend(f"- {issue}" for issue in issues)
        lines.append("")
    if not warnings and not issues:
        lines.extend(["## Warnings", "", "- None", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    output_path.write_text(report + "\n", encoding="utf-8")
    return conclusion, issues + warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/final_paper_v3")
    parser.add_argument("--output", default="results/final_paper_v3/seed42_sanity_report.md")
    args = parser.parse_args()

    conclusion, messages = build_report(Path(args.results_dir), Path(args.output))
    print(f"Wrote {args.output}")
    print(f"Conclusion: {conclusion}")
    for message in messages:
        print(f"- {message}")


if __name__ == "__main__":
    main()
