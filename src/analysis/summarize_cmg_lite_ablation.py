from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_subset(rows: pd.DataFrame, name_prefix: str, x_col: str, out_path: Path, xlabel: str) -> None:
    subset = rows[rows["ablation_name"].astype(str).str.startswith(name_prefix)].copy()
    if subset.empty:
        return
    subset = subset.sort_values(x_col)
    plt.figure(figsize=(6, 4))
    plt.plot(subset[x_col].astype(str), subset["macro_f1"], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Macro-F1")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    results_dir = Path("results")
    input_path = results_dir / "ucihar_cmg_lite_ablation_results.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing CMG-Lite ablation results: {input_path}")
    rows = pd.read_csv(input_path)
    table_path = results_dir / "table_ucihar_cmg_lite_ablation.tex"
    table_cols = [
        "ablation_name",
        "model",
        "context_len",
        "alpha",
        "num_groups",
        "context_memory",
        "threshold_modulation",
        "params",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "spike_rate",
        "best_epoch",
    ]
    rows[table_cols].to_latex(table_path, index=False, float_format="%.4f")
    plot_subset(rows, "context_len_", "context_len", results_dir / "fig_cmg_lite_context_len_macro_f1.png", "Context Length")
    plot_subset(rows, "alpha_", "alpha", results_dir / "fig_cmg_lite_alpha_macro_f1.png", "Alpha")
    plot_subset(rows, "num_groups_", "num_groups", results_dir / "fig_cmg_lite_groups_macro_f1.png", "Number of Groups")
    print(f"Saved {table_path}")


if __name__ == "__main__":
    main()
