from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.efficiency import summarize_efficiency
from src.training.utils import ensure_dir, load_config


RESULT_FILES = [
    "ucihar_formal_multiseed_results.csv",
    "ucihar_main_results.csv",
    "ucihar_cmg_diagnostic_results.csv",
    "ucihar_matched_protocol_results.csv",
    "ucihar_strong_baseline_results.csv",
]


def checkpoint_config(row: pd.Series, fallback_config: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = row.get("checkpoint", "")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        return fallback_config
    path = Path(checkpoint_path)
    if not path.exists():
        return fallback_config
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return fallback_config
    config = checkpoint.get("config")
    return config if isinstance(config, dict) else fallback_config


def normalize_row(row: pd.Series) -> pd.Series:
    row = row.copy()
    target_mode = row.get("target_mode", "all")
    if pd.isna(target_mode) or target_mode == "":
        row["target_mode"] = "all"
    if "synthetic_data" not in row or pd.isna(row.get("synthetic_data")):
        row["synthetic_data"] = False
    if "smoke_test" not in row or pd.isna(row.get("smoke_test")):
        row["smoke_test"] = False
    return row


def read_result_rows(results_dir: Path) -> pd.DataFrame:
    frames = []
    for file_name in RESULT_FILES:
        path = results_dir / file_name
        if path.exists():
            frame = pd.read_csv(path)
            frame["source_file"] = file_name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    results_dir = ensure_dir("results")
    fallback_config = load_config("configs/ucihar_cmg_lite.yaml")
    rows = read_result_rows(results_dir)
    if rows.empty:
        raise FileNotFoundError("No UCI-HAR result CSV files found under results/.")

    summaries = []
    for _, row in rows.iterrows():
        row = normalize_row(row)
        model_name = str(row.get("model", ""))
        config = checkpoint_config(row, fallback_config)
        summary = summarize_efficiency(row, config, model_name)
        summary["source_file"] = row.get("source_file", "")
        summaries.append(summary)

    output = pd.DataFrame(summaries)
    output_path = results_dir / "ucihar_efficiency_proxy.csv"
    output.to_csv(output_path, index=False)
    table_path = results_dir / "table_ucihar_efficiency_proxy.tex"
    table = summarize_efficiency_table(output)
    table.to_latex(table_path, index=False, escape=False)
    print(f"Saved efficiency proxy table to {output_path}")
    print(f"Saved efficiency proxy LaTeX table to {table_path}")
    print(output)


def _mean_std(mean: float, std: float, decimals: int = 4) -> str:
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def summarize_efficiency_table(rows: pd.DataFrame) -> pd.DataFrame:
    formal = rows[rows["source_file"].eq("ucihar_formal_multiseed_results.csv")]
    source = formal if not formal.empty else rows
    grouped = source.groupby("model", sort=False)
    records = []
    for model, group in grouped:
        records.append(
            {
                "Model": model,
                "Params": int(round(group["params"].mean())),
                "Model Size MB": f"{group['model_size_mb'].mean():.4f}",
                "Spike Rate": _mean_std(group["spike_rate"].mean(), group["spike_rate"].std(ddof=1)),
                "Spike Count": _mean_std(
                    group["spike_count_per_sample"].mean(),
                    group["spike_count_per_sample"].std(ddof=1),
                    decimals=1,
                ),
                "Total Ops Proxy": int(round(group["total_ops_proxy"].mean())),
                "Note": "proxy_only_not_measured_power",
            }
        )
    return pd.DataFrame(records)


if __name__ == "__main__":
    main()
