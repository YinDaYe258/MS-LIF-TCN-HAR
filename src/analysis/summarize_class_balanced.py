from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for key, group in rows.groupby(["dataset_key", "task", "model", "loss_type"], sort=False):
        dataset_key, task, model, loss_type = key
        records.append(
            {
                "dataset_key": dataset_key,
                "task": task,
                "model": model,
                "loss_type": loss_type,
                "num_runs": len(group),
                "accuracy": group["accuracy"].mean(),
                "macro_f1": group["macro_f1"].mean(),
                "weighted_f1": group["weighted_f1"].mean(),
                "balanced_accuracy": group.get("balanced_accuracy", pd.Series([0.0])).mean(),
                "params": int(round(group["params"].mean())),
                "spike_rate": group["spike_rate"].mean(),
                "best_epoch": group["best_epoch"].mean(),
                "best_val_macro_f1": group["best_val_macro_f1"].mean(),
            }
        )
    summary = pd.DataFrame(records)
    if summary.empty:
        return summary
    summary["rank_within_dataset_model"] = summary.groupby(["dataset_key", "model"])["macro_f1"].rank(
        ascending=False,
        method="min",
    )
    return summary.sort_values(["dataset_key", "model", "rank_within_dataset_model"]).reset_index(drop=True)


def main() -> None:
    results_dir = Path("results")
    input_path = results_dir / "class_balanced_seed42_results.csv"
    if not input_path.exists():
        print(f"Missing {input_path}")
        return
    rows = pd.read_csv(input_path)
    if rows.empty:
        print(f"{input_path} is empty")
        return
    summary = summarize(rows)
    summary_path = results_dir / "class_balanced_summary.csv"
    table_path = results_dir / "table_class_balanced_seed42.tex"
    summary.to_csv(summary_path, index=False)
    summary.to_latex(table_path, index=False, escape=False, float_format="%.4f")
    print(f"Wrote {summary_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
