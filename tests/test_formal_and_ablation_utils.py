from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_ucihar_cmg_lite_ablation import ablation_specs, config_for_spec
from scripts.run_ucihar_formal_multiseed import append_unique_formal_row
from src.analysis.summarize_formal_multiseed import latex_table, summarize_formal


def test_formal_append_unique_does_not_duplicate(tmp_path: Path) -> None:
    result_path = tmp_path / "formal.csv"
    row = {
        "dataset": "UCI-HAR",
        "model": "lif_snn",
        "seed": 42,
        "context_len": 8,
        "target_mode": "last",
        "synthetic_data": False,
        "params": 1,
        "accuracy": 0.1,
        "macro_f1": 0.2,
        "weighted_f1": 0.3,
        "loss": 1.0,
        "spike_rate": 0.4,
        "best_epoch": 1,
        "best_val_macro_f1": 0.5,
        "checkpoint": "x.pt",
        "epoch_log": "x.csv",
        "confusion_matrix_path": "x.json",
    }

    assert append_unique_formal_row(result_path, row) is True
    assert append_unique_formal_row(result_path, row) is False
    assert len(pd.read_csv(result_path)) == 1


def test_formal_summary_mean_std_and_latex_nonempty() -> None:
    rows = pd.DataFrame(
        [
            {"model": "lif_snn", "accuracy": 0.8, "macro_f1": 0.7, "weighted_f1": 0.75, "params": 2, "spike_rate": 0.1, "best_epoch": 3, "seed": 1},
            {"model": "lif_snn", "accuracy": 0.9, "macro_f1": 0.8, "weighted_f1": 0.85, "params": 2, "spike_rate": 0.2, "best_epoch": 5, "seed": 2},
        ]
    )

    summary = summarize_formal(rows)
    table = latex_table(summary)

    assert summary.iloc[0]["macro_f1_mean"] == 0.75
    assert summary.iloc[0]["num_seeds"] == 2
    assert not table.empty
    assert "±" in table.iloc[0]["Macro-F1"]


def test_ablation_config_modifications() -> None:
    base_config = {
        "seed": 42,
        "dataset": {"context_len": 8},
        "model": {"alpha": 0.8, "num_groups": 8, "context_memory": True, "threshold_modulation": True},
        "training": {"target_mode": "last"},
    }
    spec = {"ablation_name": "alpha_0.5", "model": "cmg_lif_lite", "context_len": 4, "alpha": 0.5, "num_groups": 16}
    config = config_for_spec(base_config, spec)

    assert config["dataset"]["context_len"] == 4
    assert config["model"]["alpha"] == 0.5
    assert config["model"]["num_groups"] == 16
    assert any(item["ablation_name"] == "without_context_memory" for item in ablation_specs())
