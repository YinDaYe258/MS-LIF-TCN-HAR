from __future__ import annotations

from pathlib import Path

import pandas as pd


def _mean_std(mean: float, std: float, decimals: int = 4) -> str:
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset_key", "task", "model", "base_model", "temperature", "kd_weight"]
    records = []
    for key, group in rows.groupby(group_cols, sort=False):
        dataset_key, task, model, base_model, temperature, kd_weight = key
        records.append(
            {
                "dataset_key": dataset_key,
                "task": task,
                "model": model,
                "base_model": base_model,
                "temperature": float(temperature),
                "kd_weight": float(kd_weight),
                "num_seeds": int(group["seed"].nunique()),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=1),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=1),
                "weighted_f1_mean": group["weighted_f1"].mean(),
                "weighted_f1_std": group["weighted_f1"].std(ddof=1),
                "balanced_accuracy_mean": group.get("balanced_accuracy", pd.Series([0.0])).mean(),
                "balanced_accuracy_std": group.get("balanced_accuracy", pd.Series([0.0])).std(ddof=1),
                "params": int(round(group["params"].mean())),
                "spike_rate_mean": group["spike_rate"].mean(),
                "spike_rate_std": group["spike_rate"].std(ddof=1),
                "best_epoch_mean": group["best_epoch"].mean(),
                "best_val_macro_f1_mean": group["best_val_macro_f1"].mean(),
            }
        )
    return pd.DataFrame(records)


def latex_table(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dataset": summary["dataset_key"],
            "Model": summary["model"],
            "T": summary["temperature"],
            "KD": summary["kd_weight"],
            "Seeds": summary["num_seeds"],
            "Acc": [_mean_std(row.accuracy_mean, row.accuracy_std) for row in summary.itertuples()],
            "Macro-F1": [_mean_std(row.macro_f1_mean, row.macro_f1_std) for row in summary.itertuples()],
            "Weighted-F1": [_mean_std(row.weighted_f1_mean, row.weighted_f1_std) for row in summary.itertuples()],
            "Params": summary["params"].astype(int),
            "Spike Rate": [_mean_std(row.spike_rate_mean, row.spike_rate_std) for row in summary.itertuples()],
        }
    )


def main() -> None:
    results_dir = Path("results")
    sources = [
        results_dir / "distill_multiseed_results.csv",
        results_dir / "distill_seed42_grid.csv",
    ]
    existing = [pd.read_csv(path) for path in sources if path.exists()]
    if not existing:
        print("No distillation CSV files found.")
        return
    rows = pd.concat(existing, ignore_index=True, sort=False)
    if rows.empty:
        print("Distillation CSV files are empty.")
        return
    summary = summarize(rows)
    summary_path = results_dir / "distill_multiseed_summary.csv"
    table_path = results_dir / "table_distill_multiseed.tex"
    summary.to_csv(summary_path, index=False)
    latex_table(summary).to_latex(table_path, index=False, escape=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
