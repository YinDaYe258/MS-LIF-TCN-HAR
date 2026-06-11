from __future__ import annotations

import pandas as pd

from scripts.run_uci_ms_lif_tcn_plus_stability import VARIANTS, apply_variant_config, build_variant_summary, row_exists


def test_stability_variant_config_overrides_tcn_layers_and_attention() -> None:
    config = {
        "model": {"attention_hidden_dim": 64, "tcn_layers": 2},
        "training": {"aux_loss": {"supervised_contrastive": {}}, "augmentation": {}},
    }
    apply_variant_config(config, VARIANTS["attn_supcon_0.05_smallattn_tcn1"])
    assert config["model"]["attention_hidden_dim"] == 16
    assert config["model"]["tcn_layers"] == 1
    assert config["training"]["aux_loss"]["supervised_contrastive"]["enabled"]
    assert config["training"]["aux_loss"]["supervised_contrastive"]["weight"] == 0.05


def test_stability_row_exists(tmp_path) -> None:
    path = tmp_path / "stability.csv"
    pd.DataFrame([{"variant": "attn_only", "seed": 42}]).to_csv(path, index=False)
    assert row_exists(path, "attn_only", 42)
    assert not row_exists(path, "attn_only", 43)


def test_stability_summary_groups_seeds() -> None:
    rows = pd.DataFrame(
        [
            {"variant": "attn", "seed": 42, "macro_f1": 0.9, "params": 10, "spike_rate": 0.2},
            {"variant": "attn", "seed": 43, "macro_f1": 0.8, "params": 10, "spike_rate": 0.3},
        ]
    )
    summary = build_variant_summary(rows)
    assert summary.iloc[0]["num_seeds"] == 2
    assert summary.iloc[0]["seeds"] == "42,43"
    assert abs(summary.iloc[0]["macro_f1_mean"] - 0.85) < 1e-12
