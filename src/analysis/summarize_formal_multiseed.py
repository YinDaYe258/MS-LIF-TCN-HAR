from __future__ import annotations

from pathlib import Path

import pandas as pd


MODEL_ORDER = [
    "cnn1d",
    "gru",
    "ms_cnn1d",
    "window_gru",
    "lif_snn",
    "cmg_lif_lite",
    "ms_lif_snn",
    "ms_cmg_lif",
]


def mean_std_text(mean: float, std: float, decimals: int = 4) -> str:
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def summarize_formal(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby("model", sort=False)
    records = []
    for model, group in grouped:
        records.append(
            {
                "model": model,
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=1),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=1),
                "weighted_f1_mean": group["weighted_f1"].mean(),
                "weighted_f1_std": group["weighted_f1"].std(ddof=1),
                "params": int(round(group["params"].mean())),
                "spike_rate_mean": group["spike_rate"].mean(),
                "spike_rate_std": group["spike_rate"].std(ddof=1),
                "best_epoch_mean": group["best_epoch"].mean(),
                "num_seeds": int(group["seed"].nunique()),
            }
        )
    summary = pd.DataFrame(records)
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    return summary.sort_values("model").reset_index(drop=True)


def latex_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Model": summary["model"].astype(str),
            "Acc": [mean_std_text(row.accuracy_mean, row.accuracy_std) for row in summary.itertuples()],
            "Macro-F1": [mean_std_text(row.macro_f1_mean, row.macro_f1_std) for row in summary.itertuples()],
            "Weighted-F1": [mean_std_text(row.weighted_f1_mean, row.weighted_f1_std) for row in summary.itertuples()],
            "Params": summary["params"].astype(int),
            "Spike Rate": [mean_std_text(row.spike_rate_mean, row.spike_rate_std) for row in summary.itertuples()],
        }
    )
    return table


def main() -> None:
    results_dir = Path("results")
    input_path = results_dir / "ucihar_formal_multiseed_results.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing formal results: {input_path}")
    rows = pd.read_csv(input_path)
    summary = summarize_formal(rows)
    csv_path = results_dir / "ucihar_formal_multiseed_summary.csv"
    tex_path = results_dir / "table_ucihar_formal_multiseed.tex"
    summary.to_csv(csv_path, index=False)
    latex_table(summary).to_latex(tex_path, index=False, escape=False)
    print(f"Saved {csv_path}")
    print(f"Saved {tex_path}")
    print(summary)


if __name__ == "__main__":
    main()
