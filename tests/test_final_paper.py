from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.final_paper import build_claim_matrix, per_class_from_matrix, summary_row


def test_summary_row_computes_mean_std() -> None:
    group = pd.DataFrame(
        [
            {"seed": 1, "accuracy": 0.8, "macro_f1": 0.7, "weighted_f1": 0.75, "balanced_accuracy": 0.72, "params": 10, "spike_rate": 0.2},
            {"seed": 2, "accuracy": 0.9, "macro_f1": 0.9, "weighted_f1": 0.85, "balanced_accuracy": 0.82, "params": 10, "spike_rate": 0.3},
        ]
    )
    row = summary_row("D", "M", group, "available")
    assert row["num_seeds"] == 2
    assert abs(row["macro_f1_mean"] - 0.8) < 1e-12
    assert row["params"] == 10


def test_summary_row_marks_non_spiking_spike_rate_as_nan() -> None:
    group = pd.DataFrame(
        [
            {"seed": 1, "accuracy": 0.8, "macro_f1": 0.7, "weighted_f1": 0.75, "balanced_accuracy": 0.72, "params": 10, "spike_rate": 0.0},
            {"seed": 2, "accuracy": 0.9, "macro_f1": 0.9, "weighted_f1": 0.85, "balanced_accuracy": 0.82, "params": 10, "spike_rate": 0.0},
        ]
    )
    row = summary_row("D", "Window-GRU", group, "available")
    assert pd.isna(row["spike_rate_mean"])
    assert pd.isna(row["spike_rate_std"])


def test_per_class_from_matrix_handles_zero_prediction() -> None:
    matrix = np.asarray([[2, 0], [1, 0]])
    rows = per_class_from_matrix(matrix, ["a", "b"])
    assert rows.loc[0, "support"] == 2
    assert pd.isna(rows.loc[1, "precision"])
    assert rows.loc[1, "support"] == 1


def test_claim_matrix_marks_uci_window_gru_not_supported() -> None:
    main = pd.DataFrame(
        [
            {"dataset": "UCI-HAR", "model": "MS-LIF-TCN", "macro_f1_mean": 0.92},
            {"dataset": "UCI-HAR", "model": "MS-LIF-SNN", "macro_f1_mean": 0.90},
            {"dataset": "UCI-HAR", "model": "Window-GRU", "macro_f1_mean": 0.93},
            {"dataset": "UCI-HAR", "model": "MS-LIF-TCN+", "macro_f1_mean": 0.91},
            {"dataset": "HAPT-6", "model": "MS-LIF-TCN", "macro_f1_mean": 0.95},
            {"dataset": "HAPT-6", "model": "MS-LIF-SNN", "macro_f1_mean": 0.89},
            {"dataset": "HAPT-6", "model": "Window-GRU", "macro_f1_mean": 0.94},
            {"dataset": "HAPT-6", "model": "MS-LIF-TCN+", "macro_f1_mean": 0.97},
        ]
    )
    claims = build_claim_matrix(main, pd.DataFrame(), pd.DataFrame([{"model": "x"}]))
    target = claims[claims["claim"].str.contains("UCI-HAR", regex=False)].iloc[0]
    assert target["support"] == "not_supported"
