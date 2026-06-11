from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


MODEL_ORDER = ["lif_snn", "lif_snn_h192", "cmg_lif_lite"]


def mean_std_text(mean: float, std: float, decimals: int = 4) -> str:
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def load_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    sources = [
        (Path("results/ucihar_formal_multiseed_results.csv"), "UCI-HAR", "ucihar"),
        (Path("results/ucihar_param_matched_results.csv"), "UCI-HAR", "ucihar"),
        (Path("results/hapt6_multiseed_results.csv"), "HAPT", "hapt6"),
        (Path("results/hapt6_param_matched_results.csv"), "HAPT", "hapt6"),
    ]
    for path, dataset, task in sources:
        if not path.exists():
            continue
        rows = pd.read_csv(path)
        if rows.empty or "model" not in rows.columns:
            continue
        rows = rows[rows["model"].isin(MODEL_ORDER)].copy()
        if rows.empty:
            continue
        rows["dataset"] = rows.get("dataset", dataset)
        rows["task"] = rows.get("task", task)
        frames.append(rows)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    dedupe_cols = ["dataset", "task", "model", "seed", "context_len", "target_mode"]
    combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    return combined


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (dataset, task, model), group in rows.groupby(["dataset", "task", "model"], sort=False):
        records.append(
            {
                "dataset": dataset,
                "task": task,
                "model": model,
                "num_seeds": int(group["seed"].nunique()),
                "context_len": int(round(group["context_len"].mean())),
                "target_mode": str(group["target_mode"].iloc[0]),
                "params": int(round(group["params"].mean())),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=1),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=1),
                "weighted_f1_mean": group["weighted_f1"].mean(),
                "weighted_f1_std": group["weighted_f1"].std(ddof=1),
                "spike_rate_mean": group["spike_rate"].mean(),
                "spike_rate_std": group["spike_rate"].std(ddof=1),
            }
        )
    summary = pd.DataFrame(records)
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    return summary.sort_values(["dataset", "task", "model"]).reset_index(drop=True)


def pairwise(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    comparisons = [("cmg_lif_lite", "lif_snn"), ("cmg_lif_lite", "lif_snn_h192")]
    for (dataset, task), group in rows.groupby(["dataset", "task"], sort=False):
        for left_model, right_model in comparisons:
            left = group[group["model"] == left_model].set_index("seed")
            right = group[group["model"] == right_model].set_index("seed")
            seeds = sorted(set(left.index) & set(right.index))
            diffs: list[float] = []
            wins = 0
            ties = 0
            for seed in seeds:
                left_value = float(left.loc[seed, "macro_f1"])
                right_value = float(right.loc[seed, "macro_f1"])
                diff = left_value - right_value
                diffs.append(diff)
                if diff > 0:
                    wins += 1
                    winner = left_model
                elif diff < 0:
                    winner = right_model
                else:
                    ties += 1
                    winner = "tie"
                detailed_records.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "comparison": f"{left_model} - {right_model}",
                        "seed": int(seed),
                        "left_macro_f1": left_value,
                        "right_macro_f1": right_value,
                        "macro_f1_diff": diff,
                        "winner": winner,
                    }
                )
            if diffs:
                diff_series = pd.Series(diffs, dtype="float64")
                summary_records.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "comparison": f"{left_model} - {right_model}",
                        "num_paired_seeds": len(diffs),
                        "mean_macro_f1_diff": diff_series.mean(),
                        "std_macro_f1_diff": diff_series.std(ddof=1),
                        "wins": wins,
                        "losses": len(diffs) - wins - ties,
                        "ties": ties,
                    }
                )
    return pd.DataFrame(detailed_records), pd.DataFrame(summary_records)


def latex_table(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dataset": summary["dataset"].astype(str),
            "Task": summary["task"].astype(str),
            "Model": summary["model"].astype(str),
            "Seeds": summary["num_seeds"],
            "Acc": [mean_std_text(row.accuracy_mean, row.accuracy_std) for row in summary.itertuples()],
            "Macro-F1": [mean_std_text(row.macro_f1_mean, row.macro_f1_std) for row in summary.itertuples()],
            "Weighted-F1": [
                mean_std_text(row.weighted_f1_mean, row.weighted_f1_std) for row in summary.itertuples()
            ],
            "Params": summary["params"].astype(int),
            "Spike Rate": [mean_std_text(row.spike_rate_mean, row.spike_rate_std) for row in summary.itertuples()],
        }
    )


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
        if pd.isna(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def write_report(summary: pd.DataFrame, pair_summary: pd.DataFrame) -> None:
    lines = [
        "# Parameter-Matched LIF Audit",
        "",
        "This audit checks whether CMG-LIF-Lite gains can be explained by extra LIF capacity.",
        "The widened control is the same vanilla LIF-SNN with hidden_dim=192.",
        "",
        "## Summary",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Paired Macro-F1 Differences",
        "",
        dataframe_to_markdown(pair_summary),
        "",
        "## Conservative Interpretation",
        "",
        "- If CMG-LIF-Lite beats both LIF-SNN h128 and h192, the context-gated lightweight claim is stronger.",
        "- If LIF-SNN h192 matches or beats CMG-LIF-Lite, the paper should state that part of the gain may come from added capacity.",
        "- These rows do not measure hardware power; spike rate remains a proxy-only quantity.",
    ]
    Path("results/param_matched_readiness_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    results_dir = Path("results")
    rows = load_rows()
    if rows.empty:
        raise FileNotFoundError("No parameter-matched or baseline rows were found.")
    combined_path = results_dir / "param_matched_combined_results.csv"
    rows.to_csv(combined_path, index=False)
    summary = summarize(rows)
    detailed, pair_summary = pairwise(rows)
    summary_path = results_dir / "param_matched_summary.csv"
    detailed_path = results_dir / "param_matched_pairwise_detailed.csv"
    pair_summary_path = results_dir / "param_matched_pairwise_summary.csv"
    tex_path = results_dir / "table_param_matched.tex"
    summary.to_csv(summary_path, index=False)
    detailed.to_csv(detailed_path, index=False)
    pair_summary.to_csv(pair_summary_path, index=False)
    latex_table(summary).to_latex(tex_path, index=False, escape=False)
    write_dataset_specific_outputs(summary, detailed, pair_summary, results_dir)
    write_report(summary, pair_summary)
    print(f"Saved {combined_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {detailed_path}")
    print(f"Saved {pair_summary_path}")
    print(f"Saved {tex_path}")
    print("Saved results/param_matched_readiness_report.md")
    print(summary)
    print(pair_summary)


def write_dataset_specific_outputs(
    summary: pd.DataFrame,
    detailed: pd.DataFrame,
    pair_summary: pd.DataFrame,
    results_dir: Path,
) -> None:
    dataset_specs = {
        "UCI-HAR": "ucihar",
        "HAPT": "hapt6",
    }
    for dataset, prefix in dataset_specs.items():
        dataset_summary = summary[summary["dataset"].astype(str) == dataset].copy()
        if dataset_summary.empty:
            continue
        dataset_detailed = detailed[detailed["dataset"].astype(str) == dataset].copy()
        dataset_pairs = pair_summary[pair_summary["dataset"].astype(str) == dataset].copy()
        dataset_summary.to_csv(results_dir / f"{prefix}_param_matched_summary.csv", index=False)
        dataset_detailed.to_csv(results_dir / f"{prefix}_param_matched_pairwise_detailed.csv", index=False)
        dataset_pairs.to_csv(results_dir / f"{prefix}_param_matched_pairwise_summary.csv", index=False)
        latex_table(dataset_summary).to_latex(
            results_dir / f"table_{prefix}_param_matched.tex",
            index=False,
            escape=False,
        )


if __name__ == "__main__":
    main()
