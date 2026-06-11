from __future__ import annotations

import pandas as pd

from src.analysis.gpu_benchmark import format_optional, model_for_inference, summarize_gpu_benchmark
from src.analysis.gpu_monitor import PowerRecord, integrate_energy_j


def test_energy_integration_trapezoid() -> None:
    records = [
        PowerRecord(timestamp=0.0, power_w=10.0),
        PowerRecord(timestamp=1.0, power_w=20.0),
        PowerRecord(timestamp=2.0, power_w=20.0),
    ]
    assert integrate_energy_j(records) == 35.0


def test_gpu_benchmark_summary_mean_std() -> None:
    raw = pd.DataFrame(
        [
            {
                "dataset": "UCI-HAR",
                "model": "cnn1d",
                "batch_size": 1,
                "macro_f1": 0.9,
                "params": 10,
                "latency_ms_per_sample": 2.0,
                "net_energy_mj_per_sample": 4.0,
            },
            {
                "dataset": "UCI-HAR",
                "model": "cnn1d",
                "batch_size": 1,
                "macro_f1": 0.9,
                "params": 10,
                "latency_ms_per_sample": 4.0,
                "net_energy_mj_per_sample": 8.0,
            },
        ]
    )
    summary = summarize_gpu_benchmark(raw)
    row = summary.iloc[0]
    assert row["repeats"] == 2
    assert row["latency_ms_per_sample_mean"] == 3.0
    assert row["net_energy_mj_per_sample_mean"] == 6.0


def test_non_spiking_spike_rate_formats_as_na() -> None:
    assert format_optional(None) == "N/A"
    assert format_optional(float("nan")) == "N/A"


def test_distilled_model_uses_student_only_for_inference() -> None:
    assert model_for_inference("ms_lif_snn_distill") == "ms_lif_snn"
    assert model_for_inference("ms_cmg_lif_distill") == "ms_cmg_lif"
    assert model_for_inference("window_gru") == "window_gru"

