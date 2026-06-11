from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.hapt import create_hapt_dataloaders
from src.training.utils import load_config


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


def summarize_multiseed(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model, group in rows.groupby("model", sort=False):
        records.append(
            {
                "model": model,
                "num_seeds": int(group["seed"].nunique()),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=1),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=1),
                "weighted_f1_mean": group["weighted_f1"].mean(),
                "weighted_f1_std": group["weighted_f1"].std(ddof=1),
                "balanced_accuracy_mean": group["balanced_accuracy"].mean(),
                "balanced_accuracy_std": group["balanced_accuracy"].std(ddof=1),
                "params": int(round(group["params"].mean())),
                "spike_rate_mean": group["spike_rate"].mean(),
                "spike_rate_std": group["spike_rate"].std(ddof=1),
                "best_epoch_mean": group["best_epoch"].mean(),
            }
        )
    summary = pd.DataFrame(records)
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    return summary.sort_values("model").reset_index(drop=True)


def latex_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": summary["model"].astype(str),
            "Seeds": summary["num_seeds"],
            "Acc": [mean_std_text(row.accuracy_mean, row.accuracy_std) for row in summary.itertuples()],
            "Macro-F1": [mean_std_text(row.macro_f1_mean, row.macro_f1_std) for row in summary.itertuples()],
            "Weighted-F1": [mean_std_text(row.weighted_f1_mean, row.weighted_f1_std) for row in summary.itertuples()],
            "Balanced Acc": [
                mean_std_text(row.balanced_accuracy_mean, row.balanced_accuracy_std) for row in summary.itertuples()
            ],
            "Params": summary["params"].astype(int),
            "Spike Rate": [mean_std_text(row.spike_rate_mean, row.spike_rate_std) for row in summary.itertuples()],
        }
    )


def pairwise_differences(rows: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    records = []
    for better, baseline in pairs:
        left = rows[rows["model"] == better].set_index("seed")
        right = rows[rows["model"] == baseline].set_index("seed")
        for seed in sorted(set(left.index) & set(right.index)):
            diff = float(left.loc[seed, "macro_f1"] - right.loc[seed, "macro_f1"])
            records.append(
                {
                    "comparison": f"{better} - {baseline}",
                    "seed": int(seed),
                    "macro_f1_diff": diff,
                    "winner": better if diff > 0 else baseline if diff < 0 else "tie",
                }
            )
    return pd.DataFrame(records)


def read_hapt_class_names(root: Path = Path("data/HAPT Dataset")) -> list[str]:
    labels: dict[int, str] = {}
    path = root / "activity_labels.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                labels[int(parts[0]) - 1] = parts[1]
    return [labels.get(class_id, f"class_{class_id}") for class_id in range(12)]


def sequence_support(config_path: str, split: str = "test") -> pd.DataFrame:
    config = load_config(config_path)
    loaders, meta = create_hapt_dataloaders(config, smoke_test=False)
    if split not in loaders:
        raise ValueError(f"Unknown split: {split}")
    dataset = loaders[split].dataset
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    labels: list[int] = []
    for indices in dataset.indices:
        if target_mode == "last":
            labels.append(int(dataset.y[indices[-1]]))
        else:
            labels.extend(int(label) for label in dataset.y[indices])
    counts = pd.Series(labels, dtype="int64").value_counts().to_dict()
    records = []
    for class_id in range(meta.num_classes):
        records.append({"class_id": class_id, "support": int(counts.get(class_id, 0))})
    return pd.DataFrame(records)


def test_sequence_support(config_path: str) -> pd.DataFrame:
    return sequence_support(config_path, split="test")


def hapt12_k1_k2_coverage() -> pd.DataFrame:
    class_names = read_hapt_class_names()
    k1_test = sequence_support("configs/hapt12_k1_last.yaml", split="test").rename(
        columns={"support": "K1_test_support"}
    )
    k2_test = sequence_support("configs/hapt12_k2_last.yaml", split="test").rename(
        columns={"support": "K2_test_support"}
    )
    k1_train = sequence_support("configs/hapt12_k1_last.yaml", split="train").rename(
        columns={"support": "K1_train_support"}
    )
    k2_train = sequence_support("configs/hapt12_k2_last.yaml", split="train").rename(
        columns={"support": "K2_train_support"}
    )
    coverage = (
        k1_test.merge(k2_test, on="class_id", how="outer")
        .merge(k1_train, on="class_id", how="outer")
        .merge(k2_train, on="class_id", how="outer")
        .fillna(0)
    )
    coverage["class_name"] = coverage["class_id"].map(lambda class_id: class_names[int(class_id)])
    coverage["lost_in_K2"] = (coverage["K1_test_support"] > 0) & (coverage["K2_test_support"] == 0)
    coverage["note"] = ""
    coverage.loc[coverage["lost_in_K2"], "note"] = "lost_in_K2_test_sequences"
    low_support = (coverage["K2_test_support"] > 0) & (coverage["K2_test_support"] < 20)
    coverage.loc[low_support & ~coverage["lost_in_K2"], "note"] = "low_K2_test_support"
    for result_path, prefix in (
        (Path("results/hapt12_k1_seed42_results.csv"), "k1"),
        (Path("results/hapt12_k2_seed42_results.csv"), "k2"),
    ):
        if result_path.exists():
            rows = pd.read_csv(result_path)
            for model in ("lif_snn", "cmg_lif_lite"):
                match = rows[rows["model"] == model]
                if not match.empty:
                    coverage[f"{prefix}_{model}_macro_f1"] = float(match.iloc[-1]["macro_f1"])
    coverage["effective_num_test_classes_k1"] = int((coverage["K1_test_support"] > 0).sum())
    coverage["effective_num_test_classes_k2"] = int((coverage["K2_test_support"] > 0).sum())
    ordered = [
        "class_id",
        "class_name",
        "K1_train_support",
        "K2_train_support",
        "K1_test_support",
        "K2_test_support",
        "lost_in_K2",
        "note",
        "k1_lif_snn_macro_f1",
        "k1_cmg_lif_lite_macro_f1",
        "k2_lif_snn_macro_f1",
        "k2_cmg_lif_lite_macro_f1",
        "effective_num_test_classes_k1",
        "effective_num_test_classes_k2",
    ]
    return coverage[ordered]


def write_report(
    hapt6_summary: pd.DataFrame | None,
    hapt12_summary: pd.DataFrame | None,
    hapt6_pairs: pd.DataFrame | None,
    hapt12_pairs: pd.DataFrame | None,
    coverage: pd.DataFrame,
) -> None:
    lines = [
        "# HAPT Readiness Report",
        "",
        "These results use raw HAPT inertial signals and official subject-level train/test splits.",
        "Efficiency-related values remain proxy-only; this report does not claim measured power.",
        "",
    ]
    if hapt6_summary is not None:
        lines.extend(["## HAPT-6 K=8", "", dataframe_to_markdown(hapt6_summary), ""])
    if hapt12_summary is not None:
        lines.extend(["## HAPT-12 K=2", "", dataframe_to_markdown(hapt12_summary), ""])
    if hapt6_pairs is not None and not hapt6_pairs.empty:
        lines.extend(["## HAPT-6 Pairwise Macro-F1 Differences", "", dataframe_to_markdown(hapt6_pairs), ""])
    if hapt12_pairs is not None and not hapt12_pairs.empty:
        lines.extend(["## HAPT-12 K=2 Pairwise Macro-F1 Differences", "", dataframe_to_markdown(hapt12_pairs), ""])
    lines.extend(
        [
            "## HAPT-12 K=1 vs K=2 Coverage",
            "",
            dataframe_to_markdown(coverage),
            "",
            "## Conservative Interpretation",
            "",
            "- HAPT-6 is suitable as a second main dataset with the current three-seed protocol.",
            "- On HAPT-6, CMG-LIF-Lite improves average Macro-F1 over LIF-SNN, but it does not win every seed.",
            "- On HAPT-6, MS-CMG-LIF beats MS-LIF-SNN across all three seeds in this run.",
            "- On HAPT-6, Window-GRU is the strongest baseline; the SNN variants should not be claimed to beat all ANN models.",
            "- On HAPT-6, MS-CMG-LIF remains competitive with CNN/MS-CNN while using an SNN-style spike representation.",
            "- HAPT-12 K=2 should be framed as a transition-aware diagnostic because transition supports are small.",
            "- On HAPT-12 K=2, CMG-LIF-Lite does not show a stable multi-seed advantage over LIF-SNN.",
            "- HAPT-12 K=8 should not be reported as a 12-class transition experiment under `sequence_within_segment=true`.",
        ]
    )
    Path("results/hapt_readiness_report.md").write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(format_markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def process_result_file(
    input_path: Path,
    summary_path: Path,
    latex_path: Path,
    pairs: list[tuple[str, str]],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not input_path.exists():
        print(f"Skipping missing {input_path}")
        return None, None
    rows = pd.read_csv(input_path)
    summary = summarize_multiseed(rows)
    pairwise = pairwise_differences(rows, pairs)
    summary.to_csv(summary_path, index=False)
    latex_summary(summary).to_latex(latex_path, index=False, escape=False)
    pairwise.to_csv(summary_path.with_name(summary_path.stem + "_pairwise.csv"), index=False)
    print(f"Saved {summary_path}")
    print(f"Saved {latex_path}")
    return summary, pairwise


def main() -> None:
    hapt6_summary, hapt6_pairs = process_result_file(
        Path("results/hapt6_multiseed_results.csv"),
        Path("results/hapt6_multiseed_summary.csv"),
        Path("results/table_hapt6_multiseed.tex"),
        [
            ("cmg_lif_lite", "lif_snn"),
            ("ms_cmg_lif", "ms_lif_snn"),
            ("ms_lif_snn", "ms_cnn1d"),
            ("ms_cmg_lif", "ms_cnn1d"),
            ("ms_lif_snn", "cnn1d"),
            ("ms_cmg_lif", "cnn1d"),
            ("ms_lif_snn", "window_gru"),
            ("ms_cmg_lif", "window_gru"),
        ],
    )
    hapt12_summary, hapt12_pairs = process_result_file(
        Path("results/hapt12_k2_multiseed_results.csv"),
        Path("results/hapt12_k2_multiseed_summary.csv"),
        Path("results/table_hapt12_k2_multiseed.tex"),
        [("cmg_lif_lite", "lif_snn")],
    )
    coverage = hapt12_k1_k2_coverage()
    coverage.to_csv("results/hapt12_k1_vs_k2_coverage.csv", index=False)
    write_report(hapt6_summary, hapt12_summary, hapt6_pairs, hapt12_pairs, coverage)
    print("Saved results/hapt12_k1_vs_k2_coverage.csv")
    print("Saved results/hapt_readiness_report.md")


if __name__ == "__main__":
    main()
