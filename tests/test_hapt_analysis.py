from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hapt_multiseed import row_mask
from src.analysis.per_class_metrics import per_class_from_confusion
from src.analysis.summarize_hapt_results import pairwise_differences, summarize_multiseed


def test_per_class_metrics_zero_support() -> None:
    matrix = np.asarray([[3, 1, 0], [0, 2, 0], [0, 0, 0]])
    rows = per_class_from_confusion(matrix, ["a", "b", "c"])
    zero = rows[rows["class_id"] == 2].iloc[0]
    assert int(zero["support"]) == 0
    assert zero["note"] == "filtered_by_sequence_protocol"
    assert pd.isna(zero["f1"])


def test_hapt_summary_mean_std_and_pairwise() -> None:
    rows = pd.DataFrame(
        [
            {"model": "lif_snn", "seed": 1, "accuracy": 0.7, "macro_f1": 0.6, "weighted_f1": 0.65, "balanced_accuracy": 0.61, "params": 10, "spike_rate": 0.2, "best_epoch": 3},
            {"model": "lif_snn", "seed": 2, "accuracy": 0.8, "macro_f1": 0.7, "weighted_f1": 0.75, "balanced_accuracy": 0.71, "params": 10, "spike_rate": 0.3, "best_epoch": 5},
            {"model": "cmg_lif_lite", "seed": 1, "accuracy": 0.8, "macro_f1": 0.65, "weighted_f1": 0.7, "balanced_accuracy": 0.66, "params": 12, "spike_rate": 0.2, "best_epoch": 4},
            {"model": "cmg_lif_lite", "seed": 2, "accuracy": 0.75, "macro_f1": 0.68, "weighted_f1": 0.72, "balanced_accuracy": 0.67, "params": 12, "spike_rate": 0.25, "best_epoch": 6},
        ]
    )
    summary = summarize_multiseed(rows)
    lif = summary[summary["model"].astype(str) == "lif_snn"].iloc[0]
    assert lif["num_seeds"] == 2
    assert np.isclose(lif["macro_f1_mean"], 0.65)
    diffs = pairwise_differences(rows, [("cmg_lif_lite", "lif_snn")])
    assert len(diffs) == 2
    assert set(diffs["winner"]) == {"cmg_lif_lite", "lif_snn"}


def test_row_mask_deduplicates_hapt_rows() -> None:
    rows = pd.DataFrame(
        [
            {"task": "hapt6", "model": "lif_snn", "seed": 42, "context_len": 8, "target_mode": "last"},
            {"task": "hapt6", "model": "lif_snn", "seed": 43, "context_len": 8, "target_mode": "last"},
        ]
    )
    config = {"seed": 42, "dataset": {"task": "hapt6", "context_len": 8}, "training": {"target_mode": "last"}}
    mask = row_mask(rows, "lif_snn", config)
    assert mask.tolist() == [True, False]


def test_hapt12_coverage_detects_missing_classes() -> None:
    coverage = pd.DataFrame(
        [
            {"class_id": 0, "class_name": "A", "K1_test_support": 5, "K2_test_support": 5},
            {"class_id": 1, "class_name": "B", "K1_test_support": 4, "K2_test_support": 0},
        ]
    )
    coverage["lost_in_K2"] = (coverage["K1_test_support"] > 0) & (coverage["K2_test_support"] == 0)
    assert coverage.loc[coverage["class_id"] == 1, "lost_in_K2"].iloc[0]
