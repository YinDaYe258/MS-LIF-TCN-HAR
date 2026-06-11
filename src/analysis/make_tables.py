from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)
    rows = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metric_cols:
            if metric in group:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std(ddof=0) if len(group) > 1 else 0.0
        row["num_seeds"] = group["seed"].nunique() if "seed" in group else len(group)
        row["result_scope"] = "multi-seed" if row["num_seeds"] > 1 else "single-seed preliminary"
        rows.append(row)
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from experiment CSV files.")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    metric_cols = ["accuracy", "macro_f1", "weighted_f1", "spike_rate", "params"]

    main_paths = [results_dir / "ucihar_main_results.csv", results_dir / "wisdm_main_results.csv"]
    main_frames = [pd.read_csv(path) for path in main_paths if path.exists()]
    if main_frames:
        main_df = pd.concat(main_frames, ignore_index=True, sort=False)
        write_table(summarize(main_df, ["dataset", "model", "context_len"], metric_cols), results_dir / "table_main_results.tex")
        efficiency_cols = [col for col in ["params", "spike_rate", "accuracy", "macro_f1"] if col in main_df]
        write_table(summarize(main_df, ["dataset", "model"], efficiency_cols), results_dir / "table_efficiency.tex")

    ablation_paths = [results_dir / "ucihar_ablation_results.csv", results_dir / "wisdm_ablation_results.csv"]
    ablation_frames = [pd.read_csv(path) for path in ablation_paths if path.exists()]
    if ablation_frames:
        ablation_df = pd.concat(ablation_frames, ignore_index=True, sort=False)
        write_table(summarize(ablation_df, ["dataset", "ablation"], metric_cols), results_dir / "table_ablation.tex")

    robustness_paths = [results_dir / "ucihar_robustness_results.csv", results_dir / "wisdm_robustness_results.csv"]
    robustness_frames = [pd.read_csv(path) for path in robustness_paths if path.exists()]
    if robustness_frames:
        robustness_df = pd.concat(robustness_frames, ignore_index=True, sort=False)
        write_table(summarize(robustness_df, ["dataset", "model", "perturbation"], metric_cols), results_dir / "table_robustness.tex")


if __name__ == "__main__":
    main()
