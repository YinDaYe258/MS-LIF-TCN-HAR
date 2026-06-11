from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_v3_context_length_ablation import aligned_indices
from src.analysis.context_length_aligned import build_model_comparison, summarize as summarize_context_aligned
from src.analysis.context_length_msliftcn import summarize_context_results, trend_table
from src.analysis.final_paper_v3_diagnostics import final_sequence_labels, per_class_from_matrix
from scripts.run_v3_context_length_ablation import row_exists as context_row_exists
from scripts.run_v3_spike_reg_ablation import row_exists as spike_reg_row_exists
from scripts.run_v3_single_scale_ablation import row_exists as single_scale_row_exists
from scripts.run_v3_hapt_transition_diagnostic import (
    binary_transition_metrics,
    hapt12_context_is_supported,
    row_exists as hapt_transition_row_exists,
)
from src.analysis.single_scale_ablation import summarize as summarize_single_scale
from src.analysis.spike_reg_ablation import summarize as summarize_spike_reg
from src.analysis.layerwise_resource_proxy import build_layerwise_proxy, summarize_layerwise
from scripts.profile_v3_context_runtime import markdown_table, mean, std
from src.datasets.sequence_dataset import SequenceWindowDataset
from src.analysis.final_paper_v3 import energy_proxy, pairwise_statistics, summarize


def _fake_v3_raw() -> pd.DataFrame:
    rows = []
    for seed, lif_tcn, lif_snn, ann_tcn in [
        (42, 0.91, 0.88, 0.90),
        (43, 0.925, 0.89, 0.91),
        (44, 0.93, 0.90, 0.92),
    ]:
        for model, macro_f1, spike_rate in [
            ("ms_lif_tcn", lif_tcn, 0.25),
            ("ms_lif_snn", lif_snn, 0.22),
            ("ms_ann_tcn", ann_tcn, np.nan),
        ]:
            rows.append(
                {
                    "dataset": "Toy-HAR",
                    "dataset_key": "toy",
                    "model": model,
                    "seed": seed,
                    "context_len": 8,
                    "window_size": 128,
                    "hidden_dim": 128,
                    "accuracy": macro_f1,
                    "macro_f1": macro_f1,
                    "weighted_f1": macro_f1,
                    "balanced_accuracy": macro_f1,
                    "params": 1000 if model != "ms_ann_tcn" else 900,
                    "spike_rate": spike_rate,
                }
            )
    return pd.DataFrame(rows)


def test_v3_summary_computes_seed_count_and_spike_rate() -> None:
    summary = summarize(_fake_v3_raw())
    row = summary[(summary["dataset_key"].eq("toy")) & (summary["model"].eq("ms_lif_tcn"))].iloc[0]
    assert row["num_seeds"] == 3
    assert abs(row["macro_f1_mean"] - 0.9216666666666667) < 1e-12
    assert abs(row["spike_rate_mean"] - 0.25) < 1e-12


def test_v3_pairwise_statistics_are_seed_paired() -> None:
    stats = pairwise_statistics(_fake_v3_raw())
    row = stats[stats["comparison"].eq("ms_lif_tcn - ms_lif_snn")].iloc[0]
    assert row["num_pairs"] == 3
    assert abs(row["mean_delta_macro_f1"] - 0.0316666666666667) < 1e-12
    assert row["win_count"] == 3
    assert row["loss_count"] == 0


def test_v3_energy_proxy_marks_non_spiking_as_nan() -> None:
    proxy = energy_proxy(_fake_v3_raw())
    ann = proxy[proxy["model"].eq("ms_ann_tcn")].iloc[0]
    snn = proxy[proxy["model"].eq("ms_lif_tcn")].iloc[0]
    assert pd.isna(ann["spike_rate"])
    assert pd.isna(ann["synops_proxy"])
    assert snn["spike_count_per_sample_proxy"] > 0
    assert snn["note"] == "proxy_only_not_measured_power"


def test_context_length_row_exists_includes_k(tmp_path) -> None:
    path = tmp_path / "context.csv"
    pd.DataFrame(
        [
            {
                "dataset_key": "ucihar",
                "model": "ms_lif_tcn",
                "seed": 42,
                "context_len": 8,
                "training_budget": "alignedK8_e20_p5_b64",
            },
            {
                "dataset_key": "ucihar",
                "model": "ms_lif_tcn",
                "seed": 42,
                "context_len": 4,
                "training_budget": "alignedK8_e20_p5_b64",
            },
        ]
    ).to_csv(path, index=False)
    assert context_row_exists(path, "ucihar", "ms_lif_tcn", 42, 8)
    assert context_row_exists(path, "ucihar", "ms_lif_tcn", 42, 4)
    assert not context_row_exists(path, "ucihar", "ms_lif_tcn", 42, 16)
    assert context_row_exists(path, "ucihar", "ms_lif_tcn", 42, 8, training_budget="alignedK8_e20_p5_b64")
    assert not context_row_exists(path, "ucihar", "ms_lif_tcn", 42, 8, training_budget="alignedK8_e15_p4_b64")


def test_v3_per_class_from_matrix_handles_unpredicted_class() -> None:
    matrix = np.array([[3, 0], [2, 0]])
    rows = per_class_from_matrix(matrix, ["a", "b"])
    assert rows.loc[0, "precision"] == 3 / 5
    assert rows.loc[0, "recall"] == 1.0
    assert pd.isna(rows.loc[1, "precision"])
    assert rows.loc[1, "recall"] == 0.0
    assert pd.isna(rows.loc[1, "f1"])


def test_v3_final_sequence_labels_respect_group_boundaries() -> None:
    x = np.zeros((6, 4, 2), dtype=np.float32)
    y = np.array([0, 1, 2, 3, 4, 5])
    subjects = np.ones(6, dtype=np.int64)
    groups = np.array([1, 1, 1, 2, 2, 2])
    labels = final_sequence_labels(x, y, subjects, groups, context_len=2)
    assert labels.tolist() == [1, 2, 4, 5]


def test_context_length_aligned_indices_use_same_final_targets() -> None:
    x = np.zeros((6, 4, 2), dtype=np.float32)
    y = np.arange(6)
    subjects = np.ones(6, dtype=np.int64)
    groups = np.ones(6, dtype=np.int64)
    k1 = SequenceWindowDataset(x, y, subjects, context_len=1, group_ids=groups)
    k3 = SequenceWindowDataset(x, y, subjects, context_len=3, group_ids=groups)
    k1_indices = aligned_indices(k1, aligned_kmax=3)
    k3_indices = aligned_indices(k3, aligned_kmax=3)
    assert [int(item[-1]) for item in k1_indices] == [2, 3, 4, 5]
    assert [int(item[-1]) for item in k3_indices] == [2, 3, 4, 5]
    assert [item.tolist() for item in k3_indices] == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_context_runtime_profile_helpers_format_tables() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert round(std([1.0, 2.0, 3.0]), 6) == 1.0
    table = markdown_table(pd.DataFrame([{"context_len": 1, "iter_ms": 12.34567}]))
    assert "| context_len | iter_ms |" in table
    assert "12.346" in table


def test_context_length_msliftcn_summary_marks_best_context() -> None:
    rows = []
    for dataset_key, values in {
        "ucihar": {1: 0.90, 2: 0.89, 4: 0.92, 8: 0.91},
        "hapt6": {1: 0.80, 2: 0.84, 4: 0.87, 8: 0.89},
        "pamap2": {1: 0.70, 2: 0.73, 4: 0.75, 8: 0.77},
        "mhealth": {1: 0.76, 2: 0.78, 4: 0.79, 8: 0.82},
    }.items():
        for context_len, macro_f1 in values.items():
            for seed in [42, 43, 44]:
                rows.append(
                    {
                        "dataset": dataset_key.upper(),
                        "dataset_key": dataset_key,
                        "model": "ms_lif_tcn",
                        "context_len": context_len,
                        "sequence_protocol": "aligned_kmax_8",
                        "training_budget": "alignedK8_e20_p5_b64",
                        "seed": seed,
                        "accuracy": macro_f1,
                        "macro_f1": macro_f1,
                        "weighted_f1": macro_f1,
                        "balanced_accuracy": macro_f1,
                        "spike_rate": 0.25,
                        "best_epoch": 1,
                    }
                )
    summary = summarize_context_results(pd.DataFrame(rows))
    trend = trend_table(summary)
    uci = trend[trend["dataset_key"].eq("ucihar")].iloc[0]
    assert int(uci["best_context_len"]) == 4
    assert int((trend["best_context_len"] > 1).sum()) == 4
    assert int(trend["all_k_gt_1_above_k1"].sum()) == 3


def test_context_length_aligned_summary_compares_models() -> None:
    rows = []
    for model, offset in [("ms_lif_tcn", 0.0), ("ms_ann_tcn", 0.01)]:
        for dataset_key in ["ucihar", "hapt6", "pamap2", "mhealth"]:
            for context_len, macro_f1 in {1: 0.80, 2: 0.82, 4: 0.84, 8: 0.86}.items():
                for seed in [42, 43, 44]:
                    rows.append(
                        {
                            "dataset": dataset_key.upper(),
                            "dataset_key": dataset_key,
                            "model": model,
                            "context_len": context_len,
                            "sequence_protocol": "aligned_kmax_8",
                            "training_budget": "alignedK8_e20_p5_b64",
                            "seed": seed,
                            "accuracy": macro_f1 + offset,
                            "macro_f1": macro_f1 + offset,
                            "weighted_f1": macro_f1 + offset,
                            "balanced_accuracy": macro_f1 + offset,
                            "spike_rate": 0.25 if model == "ms_lif_tcn" else np.nan,
                            "best_epoch": 1,
                        }
                    )
    summary = summarize_context_aligned(pd.DataFrame(rows))
    comparison = build_model_comparison(summary)
    row = comparison[(comparison["dataset_key"].eq("ucihar")) & (comparison["context_len"].eq(8))].iloc[0]
    assert abs(row["lif_minus_ann_macro_f1"] + 0.01) < 1e-12
    assert abs(summary["delta_vs_k1_macro_f1"].max() - 0.06) < 1e-12


def test_spike_reg_row_exists_includes_lambda_and_budget(tmp_path) -> None:
    path = tmp_path / "spike_reg.csv"
    pd.DataFrame(
        [
            {
                "dataset_key": "ucihar",
                "model": "ms_lif_tcn",
                "spike_reg_lambda": 0.0001,
                "seed": 42,
                "training_budget": "fixedK8_e20_p5_b64",
            }
        ]
    ).to_csv(path, index=False)
    assert spike_reg_row_exists(path, "ucihar", "ms_lif_tcn", 0.0001, 42)
    assert not spike_reg_row_exists(path, "ucihar", "ms_lif_tcn", 0.001, 42)
    assert spike_reg_row_exists(path, "ucihar", "ms_lif_tcn", 0.0001, 42, "fixedK8_e20_p5_b64")
    assert not spike_reg_row_exists(path, "ucihar", "ms_lif_tcn", 0.0001, 42, "fixedK8_e15_p4_b64")


def test_spike_reg_summary_computes_tradeoff_columns() -> None:
    rows = []
    for dataset_key in ["ucihar", "pamap2"]:
        for model in ["ms_lif_snn_wide", "ms_lif_tcn"]:
            for spike_lambda, macro_f1, spike_rate in [
                (0.0, 0.90, 0.30),
                (1e-5, 0.905, 0.28),
                (1e-4, 0.895, 0.20),
                (1e-3, 0.82, 0.10),
            ]:
                for seed in [42, 43, 44]:
                    rows.append(
                        {
                            "dataset": dataset_key.upper(),
                            "dataset_key": dataset_key,
                            "model": model,
                            "spike_reg_lambda": spike_lambda,
                            "sequence_protocol": "fixed_k8",
                            "training_budget": "fixedK8_e20_p5_b64",
                            "seed": seed,
                            "context_len": 8,
                            "window_size": 128,
                            "hidden_dim": 128,
                            "accuracy": macro_f1,
                            "macro_f1": macro_f1,
                            "weighted_f1": macro_f1,
                            "balanced_accuracy": macro_f1,
                            "spike_rate": spike_rate,
                            "params": 1000,
                            "best_epoch": 1,
                        }
                    )
    summary = summarize_spike_reg(pd.DataFrame(rows))
    row = summary[
        summary["dataset_key"].eq("ucihar")
        & summary["model"].eq("ms_lif_tcn")
        & np.isclose(summary["spike_reg_lambda"].astype(float), 1e-5)
    ].iloc[0]
    assert row["num_seeds"] == 3
    assert abs(row["delta_macro_f1_vs_lambda0"] - 0.005) < 1e-12
    assert abs(row["spike_rate_reduction_pct_vs_lambda0"] - (0.02 / 0.30 * 100.0)) < 1e-12
    assert row["synops_proxy_mean"] > 0


def test_single_scale_row_exists_includes_variant_and_budget(tmp_path) -> None:
    path = tmp_path / "single_scale.csv"
    pd.DataFrame(
        [
            {
                "dataset_key": "ucihar",
                "variant_key": "single_k5",
                "seed": 42,
                "training_budget": "fixedK8_e20_p5_b64",
            }
        ]
    ).to_csv(path, index=False)
    assert single_scale_row_exists(path, "ucihar", "single_k5", 42)
    assert not single_scale_row_exists(path, "ucihar", "single_k3", 42)
    assert single_scale_row_exists(path, "ucihar", "single_k5", 42, "fixedK8_e20_p5_b64")
    assert not single_scale_row_exists(path, "ucihar", "single_k5", 42, "fixedK8_e15_p4_b64")


def test_single_scale_summary_computes_delta_vs_multiscale() -> None:
    rows = []
    values = {
        "multi_scale_full": (0.91, "multi", 5),
        "single_k3": (0.89, "single", 3),
        "single_k5": (0.90, "single", 5),
        "single_k9": (0.88, "single", 9),
    }
    for dataset_key in ["ucihar", "hapt6", "pamap2", "mhealth"]:
        for variant, (macro_f1, encoder_mode, kernel_size) in values.items():
            for seed in [42, 43, 44]:
                rows.append(
                    {
                        "dataset": dataset_key.upper(),
                        "dataset_key": dataset_key,
                        "ablation": "single_scale",
                        "variant_key": variant,
                        "variant": variant,
                        "encoder_mode": encoder_mode,
                        "single_kernel_size": kernel_size,
                        "sequence_protocol": "fixed_k8",
                        "training_budget": "fixedK8_e20_p5_b64",
                        "seed": seed,
                        "accuracy": macro_f1,
                        "macro_f1": macro_f1,
                        "weighted_f1": macro_f1,
                        "balanced_accuracy": macro_f1,
                        "spike_rate": 0.25,
                        "params": 1000,
                        "best_epoch": 1,
                    }
                )
    summary = summarize_single_scale(pd.DataFrame(rows))
    row = summary[summary["dataset_key"].eq("ucihar") & summary["variant_key"].eq("single_k5")].iloc[0]
    assert abs(row["delta_vs_multi_scale_macro_f1"] + 0.01) < 1e-12
    full = summary[summary["variant_key"].eq("multi_scale_full")]
    assert (full["delta_vs_multi_scale_macro_f1"] == 0).all()


def test_hapt_transition_metrics_from_binary_confusion() -> None:
    metrics = binary_transition_metrics(np.array([[8, 2], [3, 7]]))
    assert metrics["transition_support"] == 10
    assert abs(metrics["transition_precision"] - 7 / 9) < 1e-12
    assert abs(metrics["transition_recall"] - 7 / 10) < 1e-12
    expected_f1 = 2 * (7 / 9) * (7 / 10) / ((7 / 9) + (7 / 10))
    assert abs(metrics["transition_f1"] - expected_f1) < 1e-12


def test_hapt12_support_rejects_low_transition_class() -> None:
    rows = []
    for class_id in range(12):
        rows.append(
            {
                "context_len": 2,
                "split": "test",
                "class_id": class_id,
                "class_name": f"class_{class_id}",
                "is_transition": class_id >= 6,
                "sequence_support": 25 if class_id != 7 else 0,
                "low_support_flag": class_id == 7,
                "zero_support_flag": class_id == 7,
            }
        )
    assert not hapt12_context_is_supported(pd.DataFrame(rows), 2)


def test_hapt_transition_row_exists_includes_task_k_and_budget(tmp_path) -> None:
    path = tmp_path / "transition.csv"
    pd.DataFrame(
        [
            {
                "task": "binary",
                "context_len": 2,
                "model": "ms_lif_tcn",
                "seed": 42,
                "training_budget": "fixedK2_e20_p5_b64",
            }
        ]
    ).to_csv(path, index=False)
    assert hapt_transition_row_exists(path, "binary", 2, "ms_lif_tcn", 42)
    assert not hapt_transition_row_exists(path, "binary", 4, "ms_lif_tcn", 42)
    assert hapt_transition_row_exists(path, "binary", 2, "ms_lif_tcn", 42, "fixedK2_e20_p5_b64")
    assert not hapt_transition_row_exists(path, "binary", 2, "ms_lif_tcn", 42, "fixedK2_e15_p4_b64")


def test_layerwise_resource_proxy_splits_dense_and_synops() -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset": "Toy",
                "dataset_key": "toy",
                "model": "ms_lif_tcn",
                "num_seeds": 3,
                "macro_f1_mean": 0.9,
                "macro_f1_std": 0.01,
                "params": 1000,
                "spike_rate_mean": 0.25,
                "context_len": 2,
                "window_size": 4,
                "num_channels": 3,
                "num_classes": 5,
                "hidden_dim": 8,
                "branch_dim": 2,
                "tcn_layers": 2,
            },
            {
                "dataset": "Toy",
                "dataset_key": "toy",
                "model": "ms_ann_tcn",
                "num_seeds": 3,
                "macro_f1_mean": 0.91,
                "macro_f1_std": 0.01,
                "params": 900,
                "spike_rate_mean": np.nan,
                "context_len": 2,
                "window_size": 4,
                "num_channels": 3,
                "num_classes": 5,
                "hidden_dim": 8,
                "branch_dim": 2,
                "tcn_layers": 2,
            },
        ]
    )
    layerwise = build_layerwise_proxy(summary)
    tcn_dense = layerwise[
        layerwise["model"].eq("ms_lif_tcn")
        & layerwise["component"].eq("window_tcn")
        & layerwise["op_type"].eq("dense_mac")
    ].iloc[0]
    assert tcn_dense["ops_per_sample_proxy"] == 2 * (2 * 8 * 3 + 2 * 8 * 8)
    synops = layerwise[layerwise["model"].eq("ms_lif_tcn") & layerwise["component"].eq("lif_synops_proxy")].iloc[0]
    assert synops["ops_per_sample_proxy"] == 0.25 * 2 * 4 * 8 * 8
    resource = summarize_layerwise(layerwise)
    ann = resource[resource["model"].eq("ms_ann_tcn")].iloc[0]
    assert pd.isna(ann["synops_per_sample_proxy"])
