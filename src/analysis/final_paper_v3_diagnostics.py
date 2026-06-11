from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.final_paper_v3 import DATASET_ORDER, MODEL_ORDER, PAIRWISE, pairwise_statistics, summarize
from src.datasets.hapt import load_hapt_windows
from src.datasets.mhealth import MHEALTH_NUM_CLASSES, load_mhealth_windows, resolve_mhealth_root
from src.datasets.pamap2 import PAMAP2_ACTIVITY_IDS, load_pamap2_windows, resolve_pamap2_root
from src.datasets.sequence_dataset import SequenceWindowDataset
from src.datasets.ucihar import load_ucihar_arrays

V3_DIR = Path("results/final_paper_v3")
RAW_PATH = V3_DIR / "main_results_raw.csv"

PER_CLASS_PATH = V3_DIR / "per_class_metrics.csv"
WORST_CLASS_PATH = V3_DIR / "worst_class_summary.csv"
IMPROVEMENT_PATH = V3_DIR / "improvement_by_class_vs_ms_lif_snn_wide.csv"
PER_CLASS_REPORT_PATH = V3_DIR / "per_class_analysis_report.md"
CHECKPOINT_REPORT_PATH = V3_DIR / "main_run_checkpoint_report.md"
CLAIM_MATRIX_MD_PATH = V3_DIR / "v3_main_claim_matrix.md"
CONTEXT_SUPPORT_PATH = V3_DIR / "context_length_support_audit.csv"
CONTEXT_SUPPORT_REPORT_PATH = V3_DIR / "context_length_support_audit_report.md"
CONFUSION_DIR = V3_DIR / "confusion_matrices"

SELECTED_PER_CLASS_MODELS = ["window_gru", "ms_lif_snn", "ms_lif_snn_wide", "ms_ann_tcn", "ms_lif_tcn"]
CONTEXT_LENS = [1, 2, 4, 8, 16]
LOW_SUPPORT_THRESHOLD = 20

UCI_CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]
PAMAP2_CLASS_NAMES = [
    "lying",
    "sitting",
    "standing",
    "walking",
    "running",
    "cycling",
    "nordic_walking",
    "ascending_stairs",
    "descending_stairs",
    "vacuum_cleaning",
    "ironing",
    "rope_jumping",
]
MHEALTH_CLASS_NAMES = [
    "standing_still",
    "sitting_relaxing",
    "lying_down",
    "walking",
    "climbing_stairs",
    "waist_bends_forward",
    "frontal_arm_elevation",
    "knees_bending",
    "cycling",
    "jogging",
    "running",
    "jump_front_back",
]


def main() -> None:
    V3_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing v3 raw results: {RAW_PATH}")
    raw = pd.read_csv(RAW_PATH)
    if raw.empty:
        raise ValueError(f"Empty v3 raw results: {RAW_PATH}")

    summary = summarize(raw)
    pairwise = pairwise_statistics(raw)
    write_main_checkpoint_report(raw, summary, pairwise)
    write_claim_matrix_markdown(pairwise)
    per_class, worst, improvements, confusion_pairs = build_per_class_analysis(raw)
    per_class.to_csv(PER_CLASS_PATH, index=False)
    worst.to_csv(WORST_CLASS_PATH, index=False)
    improvements.to_csv(IMPROVEMENT_PATH, index=False)
    write_per_class_report(per_class, worst, improvements, confusion_pairs)
    support = build_context_support_audit()
    support.to_csv(CONTEXT_SUPPORT_PATH, index=False)
    write_context_support_report(support)
    print(f"Wrote v3 diagnostics under {V3_DIR}")


def write_main_checkpoint_report(raw: pd.DataFrame, summary: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    lines = [
        "# v3 Main Run Checkpoint Report",
        "",
        "Status: v3 10-seed main run complete; v3 paper package is not complete.",
        "",
        "This report locks the completed main experiment only. It does not lock the manuscript because context-length, TCN-depth, spike-regularization, per-class, transition, and layerwise resource analyses are still required.",
        "",
        "## Row Completeness",
        "",
    ]
    rows = []
    for dataset in DATASET_ORDER:
        dataset_rows = raw[raw["dataset_key"].astype(str).eq(dataset)]
        rows.append(
            {
                "dataset": dataset,
                "rows": len(dataset_rows),
                "expected_rows": len(MODEL_ORDER) * 10,
                "seeds": " ".join(str(int(seed)) for seed in sorted(dataset_rows["seed"].dropna().unique())),
                "models": " ".join(sorted(dataset_rows["model"].astype(str).unique())),
                "synthetic_rows": int(dataset_rows["synthetic_data"].astype(str).str.lower().eq("true").sum()),
                "smoke_rows": int(dataset_rows["smoke_test"].astype(str).str.lower().eq("true").sum()),
            }
        )
    lines.append(markdown_table(pd.DataFrame(rows)))
    lines.extend(["", "## Artifact Check", ""])
    artifact_rows = []
    for dataset in DATASET_ORDER:
        dataset_rows = raw[raw["dataset_key"].astype(str).eq(dataset)]
        artifact_rows.append(
            {
                "dataset": dataset,
                "missing_checkpoints": count_missing_paths(dataset_rows, "checkpoint"),
                "missing_epoch_logs": count_missing_paths(dataset_rows, "epoch_log"),
                "missing_confusion_matrices": count_missing_paths(dataset_rows, "confusion_matrix_path"),
            }
        )
    lines.append(markdown_table(pd.DataFrame(artifact_rows)))
    lines.extend(["", "## Main Macro-F1 Summary", ""])
    summary_view = summary[
        ["dataset_key", "model", "num_seeds", "macro_f1_mean", "macro_f1_std", "params", "spike_rate_mean"]
    ].copy()
    lines.append(markdown_table(summary_view, float_format="{:.4f}"))
    lines.extend(["", "## Core Paired Comparisons", ""])
    pair_view = pairwise[
        [
            "dataset_key",
            "comparison",
            "mean_delta_macro_f1",
            "ci95_low",
            "ci95_high",
            "paired_t_p",
            "wilcoxon_p",
            "win_count",
            "loss_count",
            "interpretation",
        ]
    ].copy()
    lines.append(markdown_table(pair_view, float_format="{:.4f}"))
    lines.extend(
        [
            "",
            "## Locked Interpretation",
            "",
            "- Supported: `ms_lif_tcn` has a higher ten-seed mean than compact `ms_lif_snn` on 4/4 datasets.",
            "- Supported: `ms_lif_tcn` has a higher ten-seed mean than parameter-matched `ms_lif_snn_wide` on 4/4 datasets.",
            "- Not supported: `ms_lif_tcn` universally outperforms `ms_ann_tcn`.",
            "- Not supported: `ms_lif_tcn` universally outperforms `window_gru`.",
            "- Not supported: any claim of measured neuromorphic hardware energy.",
            "",
            "Required before manuscript v3: per-class/confusion analysis, aligned or explicitly diagnostic context-length analysis, TCN-depth ablation, spike-regularization sweep, HAPT transition diagnostic, and layerwise resource proxy.",
            "",
        ]
    )
    CHECKPOINT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_claim_matrix_markdown(pairwise: pd.DataFrame) -> None:
    rows: list[dict[str, str]] = []
    claim_specs = [
        (
            "MS-LIF-TCN improves over compact MS-LIF-SNN across datasets.",
            "ms_lif_tcn - ms_lif_snn",
        ),
        (
            "MS-LIF-TCN improves over parameter-matched MS-LIF-SNN-wide across datasets.",
            "ms_lif_tcn - ms_lif_snn_wide",
        ),
        (
            "MS-LIF-TCN universally outperforms the non-spiking MS-ANN-TCN counterpart.",
            "ms_lif_tcn - ms_ann_tcn",
        ),
        (
            "MS-LIF-TCN universally outperforms Window-GRU.",
            "ms_lif_tcn - window_gru",
        ),
    ]
    for claim, comparison in claim_specs:
        subset = pairwise[pairwise["comparison"].astype(str).eq(comparison)]
        positive = int((subset["mean_delta_macro_f1"] > 0).sum()) if not subset.empty else 0
        statistically_supported = int((subset["interpretation"].astype(str).eq("statistically_supported")).sum()) if not subset.empty else 0
        support = "supported" if positive == 4 and comparison in {"ms_lif_tcn - ms_lif_snn", "ms_lif_tcn - ms_lif_snn_wide"} else "not_supported"
        if support == "supported" and statistically_supported < 2:
            support = "higher_mean_supported_conservatively"
        evidence = "; ".join(
            f"{row.dataset_key}: Δ={row.mean_delta_macro_f1:.4f}, wins={int(row.win_count)}/{int(row.num_pairs)}, {row.interpretation}"
            for row in subset.itertuples()
        )
        rows.append({"claim": claim, "support": support, "evidence": evidence})
    rows.extend(
        [
            {
                "claim": "MS-LIF-TCN is competitive with ANN temporal baselines.",
                "support": "supported",
                "evidence": "The SNN is close to MS-ANN-TCN and Window-GRU on several datasets, but universal superiority is not supported.",
            },
            {
                "claim": "SNN models have measured neuromorphic low power.",
                "support": "not_supported",
                "evidence": "No neuromorphic hardware was measured; resource analysis is proxy/software-stack only.",
            },
            {
                "claim": "v3 paper package is complete.",
                "support": "not_supported",
                "evidence": "Main runs are complete, but ablation, per-class/confusion, transition, and layerwise resource analyses remain.",
            },
        ]
    )
    text = ["# v3 Main Claim Matrix", "", markdown_table(pd.DataFrame(rows)), ""]
    CLAIM_MATRIX_MD_PATH.write_text("\n".join(text), encoding="utf-8")


def build_per_class_analysis(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    selected = raw[raw["model"].astype(str).isin(SELECTED_PER_CLASS_MODELS)].copy()
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    matrix_accumulator: dict[tuple[str, str], list[np.ndarray]] = {}
    for record in selected.to_dict("records"):
        matrix_path = Path(str(record.get("confusion_matrix_path", "")))
        if not matrix_path.exists():
            continue
        matrix = np.asarray(json.loads(matrix_path.read_text(encoding="utf-8")), dtype=np.float64)
        dataset_key = str(record.get("dataset_key", ""))
        model = str(record.get("model", ""))
        seed = int(record.get("seed", 0))
        class_names = class_names_for(dataset_key, matrix.shape[0])
        matrix_accumulator.setdefault((dataset_key, model), []).append(matrix)
        for row in per_class_from_matrix(matrix, class_names).to_dict("records"):
            rows.append(
                {
                    "dataset": record.get("dataset", dataset_key),
                    "dataset_key": dataset_key,
                    "model": model,
                    "seed": seed,
                    **row,
                }
            )
        pair = worst_confusion_pair(matrix, class_names)
        if pair:
            pair_rows.append(
                {
                    "dataset": record.get("dataset", dataset_key),
                    "dataset_key": dataset_key,
                    "model": model,
                    "seed": seed,
                    **pair,
                }
            )
    for (dataset_key, model), matrices in matrix_accumulator.items():
        mean_matrix = np.mean(np.stack(matrices, axis=0), axis=0)
        class_names = class_names_for(dataset_key, mean_matrix.shape[0])
        write_confusion_outputs(dataset_key, model, mean_matrix, class_names)

    seed_level = pd.DataFrame(rows)
    if seed_level.empty:
        return seed_level, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(pair_rows)
    grouped = (
        seed_level.groupby(["dataset", "dataset_key", "model", "class_id", "class_name"], dropna=False)
        .agg(
            num_seeds=("seed", "nunique"),
            support_mean=("support", "mean"),
            support_min=("support", "min"),
            support_max=("support", "max"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
    )
    grouped["low_support_flag"] = grouped["support_min"] < LOW_SUPPORT_THRESHOLD
    grouped["note"] = np.where(grouped["low_support_flag"], "low_support_under_k8_protocol", "")
    worst = (
        grouped.sort_values(["dataset_key", "model", "f1_mean"], ascending=[True, True, True])
        .groupby(["dataset_key", "model"], group_keys=False)
        .head(3)
        .reset_index(drop=True)
    )
    improvements = class_improvements(grouped)
    return grouped, worst, improvements, pd.DataFrame(pair_rows)


def class_improvements(grouped: pd.DataFrame) -> pd.DataFrame:
    tcn = grouped[grouped["model"].eq("ms_lif_tcn")].copy()
    wide = grouped[grouped["model"].eq("ms_lif_snn_wide")].copy()
    merged = tcn.merge(
        wide,
        on=["dataset", "dataset_key", "class_id", "class_name"],
        suffixes=("_ms_lif_tcn", "_ms_lif_snn_wide"),
    )
    if merged.empty:
        return merged
    merged["delta_f1"] = merged["f1_mean_ms_lif_tcn"] - merged["f1_mean_ms_lif_snn_wide"]
    merged["delta_recall"] = merged["recall_mean_ms_lif_tcn"] - merged["recall_mean_ms_lif_snn_wide"]
    merged["delta_precision"] = merged["precision_mean_ms_lif_tcn"] - merged["precision_mean_ms_lif_snn_wide"]
    return merged[
        [
            "dataset",
            "dataset_key",
            "class_id",
            "class_name",
            "support_mean_ms_lif_tcn",
            "low_support_flag_ms_lif_tcn",
            "f1_mean_ms_lif_snn_wide",
            "f1_mean_ms_lif_tcn",
            "delta_f1",
            "recall_mean_ms_lif_snn_wide",
            "recall_mean_ms_lif_tcn",
            "delta_recall",
            "precision_mean_ms_lif_snn_wide",
            "precision_mean_ms_lif_tcn",
            "delta_precision",
        ]
    ].sort_values(["dataset_key", "delta_f1"], ascending=[True, False])


def build_context_support_audit() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_key in DATASET_ORDER:
        train, test, num_classes = load_support_arrays(dataset_key)
        for context_len in CONTEXT_LENS:
            for split, arrays in [("train", train), ("test", test)]:
                x, y, subjects, group_ids = arrays
                labels = final_sequence_labels(x, y, subjects, group_ids, context_len)
                for class_id in range(num_classes):
                    support = int((labels == class_id).sum())
                    rows.append(
                        {
                            "dataset_key": dataset_key,
                            "dataset": display_dataset_name(dataset_key),
                            "context_len": context_len,
                            "split": split,
                            "class_id": class_id,
                            "class_name": class_names_for(dataset_key, num_classes)[class_id],
                            "final_label_sequence_support": support,
                            "low_support_flag": bool(split == "test" and support < LOW_SUPPORT_THRESHOLD),
                            "zero_support_flag": bool(split == "test" and support == 0),
                        }
                    )
    return pd.DataFrame(rows)


def write_context_support_report(support: pd.DataFrame) -> None:
    test = support[support["split"].eq("test")].copy()
    low = test[test["low_support_flag"]].copy()
    zero = test[test["zero_support_flag"]].copy()
    k_summary = (
        test.groupby(["dataset_key", "context_len"], dropna=False)
        .agg(
            min_test_support=("final_label_sequence_support", "min"),
            low_support_classes=("low_support_flag", "sum"),
            zero_support_classes=("zero_support_flag", "sum"),
            total_test_sequences=("final_label_sequence_support", "sum"),
        )
        .reset_index()
    )
    lines = [
        "# v3 Context-Length Support Audit",
        "",
        "This audit counts final-window class support before running any context-length ablation. It is not a training result.",
        "",
        "## K-Level Test Support Summary",
        "",
        markdown_table(k_summary),
        "",
        "## Low-Support Test Classes",
        "",
        markdown_table(
            low[
                [
                    "dataset_key",
                    "context_len",
                    "class_id",
                    "class_name",
                    "final_label_sequence_support",
                    "zero_support_flag",
                ]
            ]
            if not low.empty
            else low
        ),
        "",
        "## Protocol Decision Notes",
        "",
        "- K=8 remains usable for the v3 main protocol, with MHEALTH class 11 flagged as low support.",
        "- K=16 should not be used as a main context-length comparison for HAPT-6 or MHEALTH because it creates zero-support test classes.",
        "- The strict K sweep should use an aligned final-window protocol, e.g. `--aligned_kmax 8`, and should treat K=16 as diagnostic only if support is adequate.",
        "",
    ]
    if not zero.empty:
        lines.extend(["## Zero-Support Test Classes", "", markdown_table(zero), ""])
    CONTEXT_SUPPORT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def load_support_arrays(dataset_key: str) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None], int]:
    if dataset_key == "ucihar":
        train_x, train_y, train_subjects = load_ucihar_arrays("data/UCI HAR Dataset", "train")
        test_x, test_y, test_subjects = load_ucihar_arrays("data/UCI HAR Dataset", "test")
        return (train_x, train_y, train_subjects, None), (test_x, test_y, test_subjects, None), 6
    if dataset_key == "hapt6":
        train_x, train_y, train_subjects, train_meta = load_hapt_windows("data/HAPT Dataset", "train", task="hapt6")
        test_x, test_y, test_subjects, test_meta = load_hapt_windows("data/HAPT Dataset", "test", task="hapt6")
        return (
            train_x,
            train_y,
            train_subjects,
            train_meta["segment_ids"],
        ), (
            test_x,
            test_y,
            test_subjects,
            test_meta["segment_ids"],
        ), 6
    if dataset_key == "pamap2":
        root = resolve_pamap2_root("data/PAMAP2_Dataset")
        train_x, train_y, train_subjects, train_meta = load_pamap2_windows(root, "train")
        test_x, test_y, test_subjects, test_meta = load_pamap2_windows(root, "test")
        return (
            train_x,
            train_y,
            train_subjects,
            train_meta["segment_ids"],
        ), (
            test_x,
            test_y,
            test_subjects,
            test_meta["segment_ids"],
        ), len(PAMAP2_ACTIVITY_IDS)
    if dataset_key == "mhealth":
        root = resolve_mhealth_root("data/MHEALTHDATASET")
        train_x, train_y, train_subjects, train_meta = load_mhealth_windows(root, "train")
        test_x, test_y, test_subjects, test_meta = load_mhealth_windows(root, "test")
        return (
            train_x,
            train_y,
            train_subjects,
            train_meta["segment_ids"],
        ), (
            test_x,
            test_y,
            test_subjects,
            test_meta["segment_ids"],
        ), MHEALTH_NUM_CLASSES
    raise ValueError(f"Unsupported dataset: {dataset_key}")


def final_sequence_labels(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    group_ids: np.ndarray | None,
    context_len: int,
) -> np.ndarray:
    dataset = SequenceWindowDataset(x, y, subjects, context_len=context_len, group_ids=group_ids)
    if len(dataset.indices) == 0:
        return np.asarray([], dtype=np.int64)
    return np.asarray([int(y[indices[-1]]) for indices in dataset.indices], dtype=np.int64)


def per_class_from_matrix(matrix: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    rows = []
    for idx, class_name in enumerate(class_names):
        support = float(matrix[idx].sum()) if idx < matrix.shape[0] else 0.0
        predicted = float(matrix[:, idx].sum()) if idx < matrix.shape[1] else 0.0
        tp = float(matrix[idx, idx]) if idx < matrix.shape[0] and idx < matrix.shape[1] else 0.0
        precision = tp / predicted if predicted > 0 else np.nan
        recall = tp / support if support > 0 else np.nan
        f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0 else np.nan
        rows.append(
            {
                "class_id": idx,
                "class_name": class_name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def worst_confusion_pair(matrix: np.ndarray, class_names: list[str]) -> dict[str, Any] | None:
    if matrix.size == 0:
        return None
    off = matrix.copy()
    for idx in range(min(off.shape)):
        off[idx, idx] = 0
    if off.max() <= 0:
        return None
    src, dst = np.unravel_index(np.argmax(off), off.shape)
    return {
        "true_class_id": int(src),
        "true_class": class_names[src] if src < len(class_names) else str(src),
        "predicted_class_id": int(dst),
        "predicted_class": class_names[dst] if dst < len(class_names) else str(dst),
        "count": int(off[src, dst]),
    }


def write_confusion_outputs(dataset_key: str, model: str, matrix: np.ndarray, class_names: list[str]) -> None:
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    base = f"{dataset_key}_{model}_mean_confusion"
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(CONFUSION_DIR / f"{base}.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"{display_dataset_name(dataset_key)} {model} mean confusion")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CONFUSION_DIR / f"{base}.png", dpi=180)
    plt.close(fig)


def write_per_class_report(per_class: pd.DataFrame, worst: pd.DataFrame, improvements: pd.DataFrame, confusion_pairs: pd.DataFrame) -> None:
    lines = [
        "# v3 Per-Class Analysis Report",
        "",
        "This analysis is computed only from existing v3 main-run confusion matrices. No models were retrained.",
        "",
        "## Key Cautions",
        "",
        "- MHEALTH class 11 has only 16 K=8 test sequences and must be treated as a low-support class.",
        "- Per-class metrics should be used alongside Macro-F1 to avoid over-interpreting low-support classes.",
        "",
        "## Worst Three Classes Per Dataset/Model",
        "",
        markdown_table(
            worst[
                [
                    "dataset_key",
                    "model",
                    "class_id",
                    "class_name",
                    "support_mean",
                    "f1_mean",
                    "recall_mean",
                    "low_support_flag",
                ]
            ],
            float_format="{:.4f}",
        )
        if not worst.empty
        else "(empty)",
        "",
        "## Largest Positive Class-Level MS-LIF-TCN Gains vs MS-LIF-SNN-wide",
        "",
    ]
    if not improvements.empty:
        top = improvements.sort_values(["dataset_key", "delta_f1"], ascending=[True, False]).groupby("dataset_key", group_keys=False).head(5)
        lines.append(
            markdown_table(
                top[
                    [
                        "dataset_key",
                        "class_id",
                        "class_name",
                        "support_mean_ms_lif_tcn",
                        "f1_mean_ms_lif_snn_wide",
                        "f1_mean_ms_lif_tcn",
                        "delta_f1",
                    ]
                ],
                float_format="{:.4f}",
            )
        )
        lines.extend(["", "## Largest Negative Class-Level MS-LIF-TCN Changes vs MS-LIF-SNN-wide", ""])
        bottom = improvements.sort_values(["dataset_key", "delta_f1"], ascending=[True, True]).groupby("dataset_key", group_keys=False).head(5)
        lines.append(
            markdown_table(
                bottom[
                    [
                        "dataset_key",
                        "class_id",
                        "class_name",
                        "support_mean_ms_lif_tcn",
                        "f1_mean_ms_lif_snn_wide",
                        "f1_mean_ms_lif_tcn",
                        "delta_f1",
                    ]
                ],
                float_format="{:.4f}",
            )
        )
    else:
        lines.append("(empty)")
    lines.extend(["", "## Frequent Confusion Pairs", ""])
    if not confusion_pairs.empty:
        summary = (
            confusion_pairs.groupby(["dataset_key", "model", "true_class", "predicted_class"], dropna=False)
            .agg(mean_count=("count", "mean"), max_count=("count", "max"), num_seeds=("seed", "nunique"))
            .reset_index()
            .sort_values(["dataset_key", "model", "mean_count"], ascending=[True, True, False])
            .groupby(["dataset_key", "model"], group_keys=False)
            .head(3)
        )
        lines.append(markdown_table(summary, float_format="{:.2f}"))
    else:
        lines.append("(empty)")
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- If improvements are concentrated in only one or two classes, the Results section should say so explicitly.",
            "- If MHEALTH class 11 is among the worst classes, report the low support rather than treating the mean Macro-F1 as definitive.",
            "- UCI-HAR and HAPT-6 static-pose confusions such as SITTING/STANDING should be discussed if they remain among the dominant confusion pairs.",
            "",
        ]
    )
    PER_CLASS_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def class_names_for(dataset_key: str, num_classes: int) -> list[str]:
    if dataset_key == "ucihar":
        names = UCI_CLASS_NAMES
    elif dataset_key == "hapt6":
        names = UCI_CLASS_NAMES
    elif dataset_key == "pamap2":
        names = PAMAP2_CLASS_NAMES
    elif dataset_key == "mhealth":
        names = MHEALTH_CLASS_NAMES
    else:
        names = []
    if len(names) < num_classes:
        names = list(names) + [f"class_{idx}" for idx in range(len(names), num_classes)]
    return names[:num_classes]


def display_dataset_name(dataset_key: str) -> str:
    return {"ucihar": "UCI-HAR", "hapt6": "HAPT-6", "pamap2": "PAMAP2", "mhealth": "MHEALTH"}.get(dataset_key, dataset_key)


def count_missing_paths(frame: pd.DataFrame, column: str) -> int:
    count = 0
    for value in frame.get(column, pd.Series(dtype=str)).fillna("").astype(str):
        if not value or not Path(value).exists():
            count += 1
    return count


def markdown_table(frame: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    if frame.empty:
        return "(empty)"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else float_format.format(float(value)))
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.astype(object).itertuples(index=False, name=None)]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


if __name__ == "__main__":
    main()
