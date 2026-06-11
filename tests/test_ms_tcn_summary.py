from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.summarize_ms_tcn_results import compare_to_baselines, summarize_ms_tcn


def test_ms_tcn_summary_mean_std() -> None:
    rows = pd.DataFrame(
        [
            {
                "dataset": "UCI-HAR",
                "dataset_key": "ucihar",
                "task": "ucihar",
                "model": "ms_lif_tcn",
                "seed": 42,
                "accuracy": 0.90,
                "macro_f1": 0.88,
                "weighted_f1": 0.89,
                "balanced_accuracy": 0.87,
                "params": 100,
                "spike_rate": 0.20,
                "best_epoch": 3,
            },
            {
                "dataset": "UCI-HAR",
                "dataset_key": "ucihar",
                "task": "ucihar",
                "model": "ms_lif_tcn",
                "seed": 43,
                "accuracy": 0.94,
                "macro_f1": 0.92,
                "weighted_f1": 0.93,
                "balanced_accuracy": 0.91,
                "params": 100,
                "spike_rate": 0.30,
                "best_epoch": 5,
            },
        ]
    )
    summary = summarize_ms_tcn(rows)
    row = summary.iloc[0]
    assert row["num_seeds"] == 2
    assert row["seeds"] == "42,43"
    assert np.isclose(row["macro_f1_mean"], 0.90)
    assert np.isclose(row["spike_rate_mean"], 0.25)


def test_ms_tcn_baseline_comparison() -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset_key": "ucihar",
                "model": "ms_lif_tcn",
                "macro_f1_mean": 0.93,
                "params": 100,
            }
        ]
    )
    baselines = pd.DataFrame(
        [
            {
                "dataset_key": "ucihar",
                "model": "window_gru",
                "macro_f1_mean": 0.91,
                "params": 200,
                "source": "fake.csv",
            }
        ]
    )
    comparisons = compare_to_baselines(summary, baselines)
    assert len(comparisons) == 1
    assert comparisons.iloc[0]["baseline_model"] == "window_gru"
    assert np.isclose(comparisons.iloc[0]["macro_f1_diff"], 0.02)
