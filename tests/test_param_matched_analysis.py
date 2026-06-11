from __future__ import annotations

import pandas as pd

from src.analysis.summarize_param_matched import pairwise, summarize


def test_param_matched_summary_and_pairwise() -> None:
    rows = pd.DataFrame(
        [
            {"dataset": "UCI-HAR", "task": "ucihar", "model": "lif_snn", "seed": 1, "context_len": 8, "target_mode": "last", "params": 10, "accuracy": 0.7, "macro_f1": 0.60, "weighted_f1": 0.62, "spike_rate": 0.2},
            {"dataset": "UCI-HAR", "task": "ucihar", "model": "lif_snn", "seed": 2, "context_len": 8, "target_mode": "last", "params": 10, "accuracy": 0.8, "macro_f1": 0.70, "weighted_f1": 0.72, "spike_rate": 0.3},
            {"dataset": "UCI-HAR", "task": "ucihar", "model": "lif_snn_h192", "seed": 1, "context_len": 8, "target_mode": "last", "params": 14, "accuracy": 0.71, "macro_f1": 0.61, "weighted_f1": 0.63, "spike_rate": 0.21},
            {"dataset": "UCI-HAR", "task": "ucihar", "model": "lif_snn_h192", "seed": 2, "context_len": 8, "target_mode": "last", "params": 14, "accuracy": 0.79, "macro_f1": 0.69, "weighted_f1": 0.71, "spike_rate": 0.29},
            {"dataset": "UCI-HAR", "task": "ucihar", "model": "cmg_lif_lite", "seed": 1, "context_len": 8, "target_mode": "last", "params": 13, "accuracy": 0.72, "macro_f1": 0.64, "weighted_f1": 0.65, "spike_rate": 0.22},
            {"dataset": "UCI-HAR", "task": "ucihar", "model": "cmg_lif_lite", "seed": 2, "context_len": 8, "target_mode": "last", "params": 13, "accuracy": 0.81, "macro_f1": 0.71, "weighted_f1": 0.73, "spike_rate": 0.31},
        ]
    )
    summary = summarize(rows)
    assert set(summary["model"].astype(str)) == {"lif_snn", "lif_snn_h192", "cmg_lif_lite"}
    detailed, pair_summary = pairwise(rows)
    assert len(detailed) == 4
    match = pair_summary[pair_summary["comparison"] == "cmg_lif_lite - lif_snn_h192"].iloc[0]
    assert match["wins"] == 2
    assert round(float(match["mean_macro_f1_diff"]), 4) == 0.025
