from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasets.sequence_dataset import SequenceWindowDataset
from src.datasets.ucihar import UCIHAR_CHANNELS, load_ucihar_arrays


RESULTS = Path("results")
OUT = RESULTS / "final_paper"

MODEL_LABELS = {
    "cnn1d": "CNN1D",
    "gru": "GRU",
    "window_gru": "Window-GRU",
    "ms_lif_snn": "MS-LIF-SNN",
    "ms_lif_tcn": "MS-LIF-TCN",
    "ms_lif_tcn_attn": "MS-LIF-TCN+",
    "ms_lif_tcn_plus": "MS-LIF-TCN+",
    "lif_snn": "LIF-SNN",
    "cmg_lif_lite": "CMG-LIF-Lite",
}

MAIN_MODEL_ORDER = ["CNN1D", "GRU", "Window-GRU", "MS-LIF-SNN", "MS-LIF-TCN", "MS-LIF-TCN+"]
MAIN_DATASET_ORDER = ["UCI-HAR", "HAPT-6"]
NON_SPIKING_MODEL_LABELS = {"CNN1D", "GRU", "Window-GRU"}

UCI_CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_all_core_records()
    main_results = build_main_results(records)
    gpu = build_gpu_table()
    main_results.to_csv(OUT / "main_results.csv", index=False)
    write_latex(main_results, OUT / "table_main_results.tex", MAIN_MODEL_ORDER)

    ablation = build_ablation_results(records)
    ablation.to_csv(OUT / "ablation_results.csv", index=False)
    write_latex(ablation, OUT / "table_ablation.tex")

    gpu.to_csv(OUT / "gpu_benchmark.csv", index=False)
    write_latex(gpu, OUT / "table_gpu_benchmark.tex")
    copy_gpu_report(gpu)

    per_class, worst, pairs = build_failure_analysis(records)
    per_class.to_csv(OUT / "per_class_metrics.csv", index=False)
    worst.to_csv(OUT / "worst_classes_summary.csv", index=False)
    write_failure_report(worst, pairs)

    claims = build_claim_matrix(main_results, ablation, gpu)
    claims.to_csv(OUT / "claim_support_matrix.csv", index=False)
    write_latex(claims, OUT / "table_claim_support_matrix.tex")

    dataset_stats = build_dataset_statistics()
    dataset_stats.to_csv(OUT / "dataset_statistics.csv", index=False)
    write_latex(dataset_stats, OUT / "table_dataset_statistics.tex")

    write_readiness_report(main_results, ablation, gpu, claims)
    write_manifest()
    print(f"Saved final paper artifacts under {OUT}")


def load_all_core_records() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    uci = read_csv_if_exists(RESULTS / "ucihar_formal_multiseed_results.csv")
    if not uci.empty:
        uci = uci[uci["model"].isin(["cnn1d", "gru", "window_gru", "ms_lif_snn", "lif_snn"])].copy()
        uci["dataset_key"] = "ucihar"
        uci["dataset"] = "UCI-HAR"
        uci["task"] = "ucihar"
        frames.append(uci)

    hapt = read_csv_if_exists(RESULTS / "hapt6_multiseed_results.csv")
    if not hapt.empty:
        hapt = hapt[hapt["model"].isin(["cnn1d", "gru", "window_gru", "ms_lif_snn", "lif_snn"])].copy()
        hapt["dataset_key"] = "hapt6"
        hapt["dataset"] = "HAPT-6"
        frames.append(hapt)

    tcn = read_csv_if_exists(RESULTS / "ms_tcn_multiseed_results.csv")
    if not tcn.empty:
        tcn = tcn[tcn["model"].eq("ms_lif_tcn")].copy()
        tcn.loc[tcn["dataset_key"].eq("hapt6"), "dataset"] = "HAPT-6"
        frames.append(tcn)

    plus = read_csv_if_exists(RESULTS / "ms_lif_tcn_plus_multiseed.csv")
    if not plus.empty:
        plus = plus[plus["variant"].eq("attn_supcon_0.1")].copy()
        plus.loc[plus["dataset_key"].eq("hapt6"), "dataset"] = "HAPT-6"
        plus["model"] = "ms_lif_tcn_plus"
        frames.append(plus)

    if not frames:
        return pd.DataFrame()
    records = pd.concat(frames, ignore_index=True, sort=False)
    records["model_label"] = records["model"].map(MODEL_LABELS).fillna(records["model"])
    if "balanced_accuracy" not in records.columns:
        records["balanced_accuracy"] = np.nan
    records["balanced_accuracy"] = records.apply(fill_balanced_accuracy, axis=1)
    return records


def build_main_results(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = records[records["model_label"].isin(MAIN_MODEL_ORDER)].copy()
    for (dataset, model), group in selected.groupby(["dataset", "model_label"], dropna=False):
        rows.append(summary_row(dataset, model, group, status="available"))
    result = pd.DataFrame(rows)
    for dataset in MAIN_DATASET_ORDER:
        for model in MAIN_MODEL_ORDER:
            if result.empty or not ((result["dataset"] == dataset) & (result["model"] == model)).any():
                result = pd.concat(
                    [
                        result,
                        pd.DataFrame(
                            [
                                {
                                    "dataset": dataset,
                                    "model": model,
                                    "num_seeds": 0,
                                    "status": "not_available",
                                    "note": "No matching final result row found.",
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                    sort=False,
                )
    result["dataset_order"] = result["dataset"].map({name: i for i, name in enumerate(MAIN_DATASET_ORDER)})
    result["model_order"] = result["model"].map({name: i for i, name in enumerate(MAIN_MODEL_ORDER)})
    return result.sort_values(["dataset_order", "model_order"]).drop(columns=["dataset_order", "model_order"])


def build_gpu_table() -> pd.DataFrame:
    verified = OUT / "gpu_benchmark_verified.csv"
    path = verified if verified.exists() else RESULTS / "gpu_benchmark" / "gpu_inference_summary.csv"
    gpu = read_csv_if_exists(path)
    if gpu.empty:
        return pd.DataFrame()
    keep_models = {
        "cnn1d": "CNN1D",
        "window_gru": "Window-GRU",
        "ms_lif_snn": "MS-LIF-SNN",
        "ms_lif_tcn": "MS-LIF-TCN",
        "ms_lif_tcn_plus": "MS-LIF-TCN+",
    }
    gpu = gpu[gpu["model"].isin(keep_models)].copy()
    gpu["model_label"] = gpu["model"].map(keep_models)
    columns = [
        "dataset",
        "model_label",
        "batch_size",
        "repeats",
        "macro_f1_mean",
        "params_mean",
        "spike_rate_mean",
        "latency_ms_per_sample_mean",
        "latency_ms_per_sample_std",
        "throughput_samples_per_s_mean",
        "avg_power_w_mean",
        "net_energy_mj_per_sample_mean",
        "net_energy_mj_per_sample_std",
        "peak_memory_mb_mean",
        "energy_note",
        "note",
    ]
    gpu = gpu[[col for col in columns if col in gpu.columns]]
    return gpu.rename(columns={"model_label": "model", "params_mean": "params"})


def add_batch1_gpu_metrics(main_results: pd.DataFrame, gpu: pd.DataFrame) -> pd.DataFrame:
    merged = main_results.copy()
    if gpu.empty:
        merged["gpu_latency_ms_sample_b1"] = np.nan
        merged["gpu_net_energy_mj_sample_b1"] = np.nan
        return merged
    batch1 = gpu[pd.to_numeric(gpu["batch_size"], errors="coerce").eq(1)].copy()
    batch1 = batch1[
        [
            "dataset",
            "model",
            "latency_ms_per_sample_mean",
            "net_energy_mj_per_sample_mean",
            "peak_memory_mb_mean",
        ]
    ].rename(
        columns={
            "latency_ms_per_sample_mean": "gpu_latency_ms_sample_b1",
            "net_energy_mj_per_sample_mean": "gpu_net_energy_mj_sample_b1",
            "peak_memory_mb_mean": "gpu_peak_memory_mb_b1",
        }
    )
    return merged.merge(batch1, on=["dataset", "model"], how="left")


def build_ablation_results(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in MAIN_DATASET_ORDER:
        key = "ucihar" if dataset == "UCI-HAR" else "hapt6"
        rows.extend(ablation_rows_for_dataset(dataset, key, records))
    return pd.DataFrame(rows)


def ablation_rows_for_dataset(dataset: str, dataset_key: str, records: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_from_records(label: str, model_label: str, note: str = "") -> None:
        group = records[(records["dataset"].eq(dataset)) & (records["model_label"].eq(model_label))]
        if group.empty:
            rows.append(missing_ablation(dataset, label, note or "No matching result."))
            return
        row = summary_row(dataset, label, group, status="available")
        row["note"] = note
        rows.append(row)

    add_from_records("MS-LIF-SNN", "MS-LIF-SNN", "No window-level TCN.")
    add_from_records("MS-LIF-TCN / TCN-2", "MS-LIF-TCN", "Main model; two causal TCN layers.")

    plus_diag = read_csv_if_exists(RESULTS / "ms_lif_tcn_plus_diagnostic.csv")
    stability = read_csv_if_exists(RESULTS / "uci_ms_lif_tcn_plus_stability.csv")

    add_variant_row(rows, dataset, dataset_key, plus_diag, "attn_ce", "MS-LIF-TCN attention only")
    if dataset_key == "ucihar":
        add_variant_row(rows, dataset, dataset_key, stability, "supcon_only_0.05", "MS-LIF-TCN SupCon only")
        add_variant_row(rows, dataset, dataset_key, stability, "attn_supcon_0.05_tcn1", "MS-LIF-TCN TCN-1")
    else:
        add_variant_row(
            rows,
            dataset,
            dataset_key,
            plus_diag,
            "tcn_supcon_0.05_aug",
            "MS-LIF-TCN SupCon only",
            note="Available HAPT row includes training augmentation; pure SupCon-only was not run.",
        )
        rows.append(missing_ablation(dataset, "MS-LIF-TCN TCN-1", "TCN-1 ablation not run for HAPT-6."))

    plus = read_csv_if_exists(RESULTS / "ms_lif_tcn_plus_multiseed.csv")
    add_variant_row(rows, dataset, dataset_key, plus, "attn_supcon_0.1", "MS-LIF-TCN attention + SupCon")
    add_single_scale_row(rows, dataset, dataset_key)
    add_from_records("MS-LIF-TCN TCN-2", "MS-LIF-TCN", "Same run as main MS-LIF-TCN.")
    return rows


def add_single_scale_row(rows: list[dict[str, Any]], dataset: str, dataset_key: str) -> None:
    source = read_csv_if_exists(OUT / "single_scale_ablation.csv")
    if source.empty:
        rows.append(missing_ablation(dataset, "MS-LIF-TCN single-scale-k5", "Single-scale encoder ablation not run."))
        return
    group = source[
        source["dataset_key"].astype(str).eq(dataset_key)
        & source["variant"].astype(str).eq("single_scale_k5")
    ].copy()
    if group.empty:
        rows.append(missing_ablation(dataset, "MS-LIF-TCN single-scale-k5", "Single-scale-k5 ablation not run for this dataset."))
        return
    group["dataset"] = dataset
    group["model_label"] = "MS-LIF-TCN single-scale-k5"
    row = summary_row(dataset, "MS-LIF-TCN single-scale-k5", group, status="available")
    row["note"] = "Single Conv1d kernel_size=5 encoder with same window TCN."
    rows.append(row)


def add_variant_row(
    rows: list[dict[str, Any]],
    dataset: str,
    dataset_key: str,
    source: pd.DataFrame,
    variant: str,
    label: str,
    note: str = "",
) -> None:
    if source.empty or "variant" not in source.columns:
        rows.append(missing_ablation(dataset, label, note or f"Variant {variant} not found."))
        return
    if "dataset_key" in source.columns:
        group = source[source["dataset_key"].astype(str).eq(dataset_key) & source["variant"].astype(str).eq(variant)].copy()
    else:
        want_dataset = "UCI-HAR" if dataset_key == "ucihar" else "HAPT-6"
        group = source[source["dataset"].astype(str).eq(want_dataset) & source["variant"].astype(str).eq(variant)].copy()
    if group.empty:
        rows.append(missing_ablation(dataset, label, note or f"Variant {variant} not run for {dataset}."))
        return
    group["dataset"] = dataset
    group["model_label"] = label
    if "balanced_accuracy" not in group.columns:
        group["balanced_accuracy"] = np.nan
    group["balanced_accuracy"] = group.apply(fill_balanced_accuracy, axis=1)
    row = summary_row(dataset, label, group, status="available")
    row["note"] = note
    rows.append(row)


def missing_ablation(dataset: str, label: str, note: str) -> dict[str, Any]:
    return {"dataset": dataset, "model": label, "num_seeds": 0, "status": "not_run", "note": note}


def build_failure_analysis(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    selected = records[records["model_label"].isin(["Window-GRU", "MS-LIF-TCN", "MS-LIF-TCN+", "LIF-SNN"])].copy()
    hapt12 = read_csv_if_exists(RESULTS / "hapt12_k2_multiseed_results.csv")
    if not hapt12.empty:
        hapt12 = hapt12[hapt12["model"].isin(["lif_snn", "cmg_lif_lite"])].copy()
        hapt12["dataset"] = "HAPT-12 K2"
        hapt12["task"] = "hapt12"
        hapt12["model_label"] = hapt12["model"].map(MODEL_LABELS).fillna(hapt12["model"])
        selected = pd.concat([selected, hapt12], ignore_index=True, sort=False)

    for row in selected.itertuples(index=False):
        matrix_path = Path(str(getattr(row, "confusion_matrix_path", "")))
        if not matrix_path.exists():
            continue
        matrix = np.asarray(json.loads(matrix_path.read_text(encoding="utf-8")), dtype=np.float64)
        class_names = class_names_for(str(getattr(row, "task", "")), str(getattr(row, "dataset", "")), matrix.shape[0])
        per_class = per_class_from_matrix(matrix, class_names)
        for item in per_class.to_dict("records"):
            rows.append(
                {
                    "dataset": getattr(row, "dataset", ""),
                    "task": getattr(row, "task", ""),
                    "model": getattr(row, "model_label", getattr(row, "model", "")),
                    "seed": int(getattr(row, "seed", 0)),
                    **item,
                }
            )
        pair = worst_confusion_pair(matrix, class_names)
        if pair:
            pair_rows.append(
                {
                    "dataset": getattr(row, "dataset", ""),
                    "model": getattr(row, "model_label", getattr(row, "model", "")),
                    "seed": int(getattr(row, "seed", 0)),
                    **pair,
                }
            )
        write_confusion_figure(matrix, class_names, getattr(row, "dataset", ""), getattr(row, "model_label", ""), int(getattr(row, "seed", 0)))

    per_class_df = pd.DataFrame(rows)
    if per_class_df.empty:
        return per_class_df, pd.DataFrame(), pd.DataFrame(pair_rows)
    grouped = (
        per_class_df.groupby(["dataset", "model", "class_id", "class_name"], dropna=False)
        .agg(support=("support", "mean"), precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"))
        .reset_index()
    )
    worst = (
        grouped.sort_values(["dataset", "model", "f1"], ascending=[True, True, True])
        .groupby(["dataset", "model"], group_keys=False)
        .head(3)
        .reset_index(drop=True)
    )
    return grouped, worst, pd.DataFrame(pair_rows)


def build_claim_matrix(main_results: pd.DataFrame, ablation: pd.DataFrame, gpu: pd.DataFrame) -> pd.DataFrame:
    def metric(dataset: str, model: str) -> float:
        match = main_results[main_results["dataset"].eq(dataset) & main_results["model"].eq(model)]
        if match.empty:
            return np.nan
        return float(match.iloc[0].get("macro_f1_mean", np.nan))

    rows = []
    uci_tcn = metric("UCI-HAR", "MS-LIF-TCN")
    uci_snn = metric("UCI-HAR", "MS-LIF-SNN")
    hapt_tcn = metric("HAPT-6", "MS-LIF-TCN")
    hapt_snn = metric("HAPT-6", "MS-LIF-SNN")
    hapt_wgru = metric("HAPT-6", "Window-GRU")
    uci_wgru = metric("UCI-HAR", "Window-GRU")
    uci_plus = metric("UCI-HAR", "MS-LIF-TCN+")
    hapt_plus = metric("HAPT-6", "MS-LIF-TCN+")

    rows.append(claim_row("MS-LIF-TCN improves over MS-LIF-SNN.", "supported", f"UCI +{uci_tcn - uci_snn:.4f}; HAPT-6 +{hapt_tcn - hapt_snn:.4f}."))
    rows.append(claim_row("MS-LIF-TCN beats Window-GRU on HAPT-6.", "supported" if hapt_tcn > hapt_wgru else "not_supported", f"MS-LIF-TCN {hapt_tcn:.4f} vs Window-GRU {hapt_wgru:.4f}."))
    rows.append(claim_row("MS-LIF-TCN beats Window-GRU on UCI-HAR.", "not_supported" if uci_tcn <= uci_wgru else "supported", f"MS-LIF-TCN {uci_tcn:.4f} vs Window-GRU {uci_wgru:.4f}."))
    rows.append(claim_row("MS-LIF-TCN+ is globally better than MS-LIF-TCN.", "not_supported", f"HAPT-6 improves ({hapt_plus:.4f} vs {hapt_tcn:.4f}), but UCI-HAR is seed-unstable ({uci_plus:.4f} mean with high std)."))
    rows.append(claim_row("Attention + SupCon improves HAPT-6.", "supported", f"MS-LIF-TCN+ {hapt_plus:.4f} vs MS-LIF-TCN {hapt_tcn:.4f}."))
    rows.append(claim_row("Attention + SupCon improves UCI-HAR.", "partially_supported", "Seed42 improves, but three-seed stability is weak; use as diagnostic, not a main claim."))
    rows.append(claim_row("CMG is the main contribution.", "not_supported", "Parameter-matched and MS-CMG diagnostics do not support CMG as the central claim."))
    rows.append(claim_row("SNN has measured neuromorphic low power.", "not_supported", "No neuromorphic hardware was measured."))
    single = read_csv_if_exists(OUT / "single_scale_ablation.csv")
    if single.empty:
        rows.append(claim_row("Multi-scale encoder independently improves over single-scale.", "pending", "Single-scale ablation was not run."))
    else:
        evidence = []
        supported = True
        for dataset_key, dataset_name in (("ucihar", "UCI-HAR"), ("hapt6", "HAPT-6")):
            full = single[
                single["dataset_key"].astype(str).eq(dataset_key)
                & single["variant"].astype(str).eq("full_multi_scale")
            ]
            one = single[
                single["dataset_key"].astype(str).eq(dataset_key)
                & single["variant"].astype(str).eq("single_scale_k5")
            ]
            if full.empty or one.empty:
                supported = False
                evidence.append(f"{dataset_name}: missing pair")
                continue
            full_f1 = float(full.iloc[-1]["macro_f1"])
            one_f1 = float(one.iloc[-1]["macro_f1"])
            supported = supported and full_f1 > one_f1
            evidence.append(f"{dataset_name}: full {full_f1:.4f} vs k5 {one_f1:.4f}")
        rows.append(
            claim_row(
                "Multi-scale encoder independently improves over single-scale-k5.",
                "weakly_supported" if supported else "not_supported",
                "Seed-42 diagnostic only; " + "; ".join(evidence) + "; multi-seed confirmation not run.",
            )
        )
    rows.append(claim_row("RTX 4060 benchmark supports software-stack efficiency analysis.", "supported" if not gpu.empty else "pending", "GPU benchmark is PyTorch/NVML software-stack measurement only."))
    return pd.DataFrame(rows)


def claim_row(claim: str, support: str, evidence: str) -> dict[str, str]:
    return {"claim": claim, "support": support, "evidence": evidence}


def summary_row(dataset: str, model: str, group: pd.DataFrame, status: str) -> dict[str, Any]:
    spike_rate_mean = mean_numeric(group, "spike_rate")
    spike_rate_std = std_numeric(group, "spike_rate")
    if model in NON_SPIKING_MODEL_LABELS:
        spike_rate_mean = np.nan
        spike_rate_std = np.nan
    return {
        "dataset": dataset,
        "model": model,
        "num_seeds": int(group["seed"].nunique()) if "seed" in group.columns else len(group),
        "accuracy_mean": mean_numeric(group, "accuracy"),
        "accuracy_std": std_numeric(group, "accuracy"),
        "macro_f1_mean": mean_numeric(group, "macro_f1"),
        "macro_f1_std": std_numeric(group, "macro_f1"),
        "weighted_f1_mean": mean_numeric(group, "weighted_f1"),
        "weighted_f1_std": std_numeric(group, "weighted_f1"),
        "balanced_accuracy_mean": mean_numeric(group, "balanced_accuracy"),
        "balanced_accuracy_std": std_numeric(group, "balanced_accuracy"),
        "params": int(round(mean_numeric(group, "params"))) if pd.notna(mean_numeric(group, "params")) else np.nan,
        "spike_rate_mean": spike_rate_mean,
        "spike_rate_std": spike_rate_std,
        "status": status,
    }


def mean_numeric(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns:
        return np.nan
    values = pd.to_numeric(group[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def std_numeric(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns:
        return np.nan
    values = pd.to_numeric(group[column], errors="coerce")
    return float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0


def fill_balanced_accuracy(row: pd.Series) -> float:
    current = row.get("balanced_accuracy", np.nan)
    if pd.notna(current):
        return float(current)
    matrix_path = Path(str(row.get("confusion_matrix_path", "")))
    if not matrix_path.exists():
        return np.nan
    matrix = np.asarray(json.loads(matrix_path.read_text(encoding="utf-8")), dtype=np.float64)
    recalls = []
    for idx in range(matrix.shape[0]):
        support = matrix[idx].sum()
        if support > 0:
            recalls.append(matrix[idx, idx] / support)
    return float(np.mean(recalls)) if recalls else np.nan


def class_names_for(task: str, dataset: str, num_classes: int) -> list[str]:
    if "UCI" in dataset:
        return UCI_CLASS_NAMES[:num_classes]
    labels_path = Path("data/HAPT Dataset/activity_labels.txt")
    labels: dict[int, str] = {}
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                labels[int(parts[0]) - 1] = parts[1]
    return [labels.get(idx, f"class_{idx}") for idx in range(num_classes)]


def per_class_from_matrix(matrix: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    rows = []
    for idx, class_name in enumerate(class_names):
        support = float(matrix[idx].sum()) if idx < matrix.shape[0] else 0.0
        predicted = float(matrix[:, idx].sum()) if idx < matrix.shape[1] else 0.0
        tp = float(matrix[idx, idx]) if idx < matrix.shape[0] and idx < matrix.shape[1] else 0.0
        precision = tp / predicted if predicted > 0 else np.nan
        recall = tp / support if support > 0 else np.nan
        f1 = 2 * precision * recall / (precision + recall) if pd.notna(precision) and pd.notna(recall) and precision + recall > 0 else np.nan
        rows.append(
            {
                "class_id": idx,
                "class_name": class_name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "note": "zero_support_after_sequence_filtering" if support == 0 else "",
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
        "true_class": class_names[src] if src < len(class_names) else str(src),
        "predicted_class": class_names[dst] if dst < len(class_names) else str(dst),
        "count": int(off[src, dst]),
    }


def write_confusion_figure(matrix: np.ndarray, class_names: list[str], dataset: str, model: str, seed: int) -> None:
    if matrix.size == 0:
        return
    fig_dir = OUT / "confusion_matrices"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"{dataset} {model} seed{seed}")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    safe_dataset = str(dataset).replace(" ", "_").replace("/", "_")
    safe_model = str(model).replace(" ", "_").replace("/", "_")
    fig.savefig(fig_dir / f"{safe_dataset}_{safe_model}_seed{seed}.png", dpi=180)
    plt.close(fig)


def write_failure_report(worst: pd.DataFrame, pairs: pd.DataFrame) -> None:
    lines = [
        "# Per-Class Failure Analysis",
        "",
        "This report summarizes per-class F1 and confusion pairs from existing confusion matrices.",
        "HAPT-12 K2 is diagnostic only because sequence filtering reduces transition-class support.",
        "",
        "## Worst Classes",
        "",
        dataframe_to_markdown(worst[["dataset", "model", "class_name", "support", "f1"]] if not worst.empty else worst),
        "",
        "## Most Frequent Confusion Pairs",
        "",
        dataframe_to_markdown(pairs if not pairs.empty else pairs),
    ]
    (OUT / "failure_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_readiness_report(main_results: pd.DataFrame, ablation: pd.DataFrame, gpu: pd.DataFrame, claims: pd.DataFrame) -> None:
    lines = [
        "# Final Readiness Report",
        "",
        "## Recommended Title",
        "",
        "Window-Temporal Multi-Scale Spiking Neural Network for Wearable Human Activity Recognition",
        "",
        "## Final Main Model",
        "",
        "`MS-LIF-TCN` should be the global main model. `MS-LIF-TCN+` should be reported as an enhanced diagnostic variant, strongest on HAPT-6 but unstable on UCI-HAR.",
        "",
        "## Safe Claims",
        "",
        "- MS-LIF-TCN improves over MS-LIF-SNN on UCI-HAR and HAPT-6.",
        "- MS-LIF-TCN beats Window-GRU on HAPT-6 in the current three-seed mean.",
        "- MS-LIF-TCN+ with attention and SupCon strongly improves HAPT-6.",
        "- RTX 4060 measurements are GPU software-stack inference benchmarks, not neuromorphic energy measurements.",
        "",
        "## Unsafe Claims",
        "",
        "- Do not claim SOTA.",
        "- Do not claim MS-LIF-TCN beats Window-GRU on UCI-HAR.",
        "- Do not claim MS-LIF-TCN+ is globally better than MS-LIF-TCN.",
        "- Do not claim CMG is the central contribution.",
        "- Do not claim measured neuromorphic low power.",
        "",
        "## Remaining Limitations",
        "",
        "- Single-scale-k5 ablation is seed-42 only and supports only a weak, diagnostic multi-scale contribution claim.",
        "- HAPT-12 K2 remains diagnostic because sequence filtering affects transition-class coverage.",
        "- GPU benchmark reflects PyTorch/CUDA/NVML behavior on RTX 4060, not event-driven neuromorphic deployment.",
        "",
        "## Target Journal Direction",
        "",
        "- Prioritize applied wearable sensing, sensor analytics, edge AI, and biomedical/health informatics venues.",
        "- Verify current journal scope, indexing, APC, and review expectations before submission; do not rely on this repository for current journal metrics.",
        "",
        "## Claim Matrix",
        "",
        dataframe_to_markdown(claims),
    ]
    (OUT / "final_readiness_report.md").write_text("\n".join(lines), encoding="utf-8")


def copy_gpu_report(gpu: pd.DataFrame) -> None:
    source = OUT / "gpu_benchmark_verification_report.md"
    if not source.exists():
        source = RESULTS / "gpu_benchmark" / "gpu_benchmark_report.md"
    if source.exists():
        text = source.read_text(encoding="utf-8")
    elif gpu.empty:
        text = "# GPU Benchmark Report\n\nNo GPU benchmark rows were available.\n"
    else:
        text = "# GPU Benchmark Report\n\nGenerated from existing RTX 4060 PyTorch/NVML benchmark rows.\n"
    text += "\n\nFinal paper note: these are RTX 4060 software-stack measurements, not neuromorphic hardware power.\n"
    (OUT / "gpu_benchmark_report.md").write_text(text, encoding="utf-8")


def build_dataset_statistics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    uci_root = Path("data/UCI HAR Dataset")
    if uci_root.exists():
        train_x, train_y, train_subjects = load_ucihar_arrays(uci_root, "train")
        test_x, test_y, test_subjects = load_ucihar_arrays(uci_root, "test")
        rows.append(
            {
                "dataset": "UCI-HAR",
                "task": "ucihar",
                "channels": len(UCIHAR_CHANNELS),
                "window_size": 128,
                "context_len": 8,
                "target_mode": "last",
                "split": "official train/test; subject-aware validation from train only",
                "train_subjects": len(np.unique(train_subjects)),
                "test_subjects": len(np.unique(test_subjects)),
                "train_windows": len(train_y),
                "test_windows": len(test_y),
                "train_sequences_k8": len(SequenceWindowDataset(train_x, train_y, train_subjects, context_len=8)),
                "test_sequences_k8": len(SequenceWindowDataset(test_x, test_y, test_subjects, context_len=8)),
                "classes": int(max(train_y.max(), test_y.max()) + 1),
                "note": "Official inertial-signal windows; no random window-level test split.",
            }
        )

    hapt_inspection = RESULTS / "hapt_dataset_inspection.json"
    if hapt_inspection.exists():
        data = json.loads(hapt_inspection.read_text(encoding="utf-8"))
        hapt6 = data.get("sequence_counts", {}).get("hapt6", {})
        train = hapt6.get("train", {})
        test = hapt6.get("test", {})
        rows.append(
            {
                "dataset": "HAPT-6",
                "task": "hapt6",
                "channels": 6,
                "window_size": 128,
                "context_len": 8,
                "target_mode": "last",
                "split": "official train/test subjects; sequence_within_segment=true",
                "train_subjects": len(train.get("subjects", [])),
                "test_subjects": len(test.get("subjects", [])),
                "train_windows": train.get("windows", np.nan),
                "test_windows": test.get("windows", np.nan),
                "train_sequences_k8": train.get("k8_sequence_count_within_segment", np.nan),
                "test_sequences_k8": test.get("k8_sequence_count_within_segment", np.nan),
                "classes": 6,
                "note": "Derived from HAPT raw acc/gyro segments, activities 1-6 only.",
            }
        )
    return pd.DataFrame(rows)


def write_manifest() -> None:
    commit = current_git_commit()
    lines = [
        f"repo_commit_at_generation: {commit}",
        "final_result_lock: results/final_paper/LOCKED_README.md",
        "datasets:",
        "  ucihar:",
        "    root: data/UCI HAR Dataset",
        "    split: official subject train/test split",
        "    context_len: 8",
        "    target_mode: last",
        "    channels: 9",
        "  hapt6:",
        "    root: data/HAPT Dataset",
        "    task: activity_id 1-6",
        "    split: official subject train/test split",
        "    sequence_within_segment: true",
        "    context_len: 8",
        "    target_mode: last",
        "    channels: 6",
        "seeds: [42, 43, 44]",
        "main_models:",
        "  - CNN1D",
        "  - GRU",
        "  - Window-GRU",
        "  - MS-LIF-SNN",
        "  - MS-LIF-TCN",
        "  - MS-LIF-TCN+",
        "main_tables:",
        "  - results/final_paper/main_results.csv",
        "  - results/final_paper/table_main_results.tex",
        "  - results/final_paper/dataset_statistics.csv",
        "  - results/final_paper/gpu_benchmark_verified.csv",
        "excluded_from_main_claims:",
        "  - CMG variants",
        "  - HAPT-12 K2",
        "  - GPU neuromorphic energy claims",
        "  - state-of-the-art claims",
        "notes:",
        "  - GPU benchmark is RTX 4060 PyTorch/CUDA/NVML software-stack measurement only.",
        "  - Non-spiking models have spike_rate reported as N/A in paper tables.",
    ]
    (OUT / "MANIFEST.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_latex(df: pd.DataFrame, output_path: Path, order: list[str] | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        output_path.write_text("", encoding="utf-8")
        return
    table = df.copy()
    if order and "model" in table.columns:
        table["model_order"] = table["model"].map({name: i for i, name in enumerate(order)}).fillna(999)
        table = table.sort_values([col for col in ["dataset", "model_order"] if col in table.columns])
        table = table.drop(columns=["model_order"])
    table.to_latex(output_path, index=False, escape=False, na_rep="N/A", float_format=lambda value: f"{value:.4f}")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    columns = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("N/A" if pd.isna(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


if __name__ == "__main__":
    main()
