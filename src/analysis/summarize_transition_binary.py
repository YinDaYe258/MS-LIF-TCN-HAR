from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


MODEL_ORDER = ["cnn1d", "window_gru", "lif_snn", "cmg_lif_lite", "ms_lif_snn", "ms_cmg_lif"]


def mean_std_text(mean: float, std: float, decimals: int = 4) -> str:
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (context_len, model), group in rows.groupby(["context_len", "model"], sort=False):
        records.append(
            {
                "context_len": int(context_len),
                "model": model,
                "num_seeds": int(group["seed"].nunique()),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=1),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=1),
                "balanced_accuracy_mean": group["balanced_accuracy"].mean(),
                "balanced_accuracy_std": group["balanced_accuracy"].std(ddof=1),
                "transition_recall_mean": group["transition_recall"].mean(),
                "transition_recall_std": group["transition_recall"].std(ddof=1),
                "transition_precision_mean": group["transition_precision"].mean(),
                "transition_precision_std": group["transition_precision"].std(ddof=1),
                "params": int(round(group["params"].mean())),
                "spike_rate_mean": group["spike_rate"].mean(),
            }
        )
    summary = pd.DataFrame(records)
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    return summary.sort_values(["context_len", "model"]).reset_index(drop=True)


def latex_table(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "K": summary["context_len"],
            "Model": summary["model"].astype(str),
            "Seeds": summary["num_seeds"],
            "Macro-F1": [mean_std_text(row.macro_f1_mean, row.macro_f1_std) for row in summary.itertuples()],
            "Balanced Acc": [
                mean_std_text(row.balanced_accuracy_mean, row.balanced_accuracy_std) for row in summary.itertuples()
            ],
            "Transition Recall": [
                mean_std_text(row.transition_recall_mean, row.transition_recall_std) for row in summary.itertuples()
            ],
            "Transition Precision": [
                mean_std_text(row.transition_precision_mean, row.transition_precision_std)
                for row in summary.itertuples()
            ],
            "Params": summary["params"],
        }
    )


def main() -> None:
    input_path = Path("results/hapt_transition_binary_results.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}")
    rows = pd.read_csv(input_path)
    summary = summarize(rows)
    summary.to_csv("results/hapt_transition_binary_summary.csv", index=False)
    latex_table(summary).to_latex("results/table_hapt_transition_binary.tex", index=False, escape=False)
    print("Saved results/hapt_transition_binary_summary.csv")
    print("Saved results/table_hapt_transition_binary.tex")
    print(summary)


if __name__ == "__main__":
    main()
