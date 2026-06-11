from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.utils import build_model, count_parameters, load_config

DATASETS = {
    "ucihar": {"display": "UCI-HAR", "config": "configs/ucihar_ms_tcn_snn.yaml", "channels": 9, "classes": 6},
    "hapt6": {"display": "HAPT-6", "config": "configs/hapt6_ms_tcn_snn.yaml", "channels": 6, "classes": 6},
    "pamap2": {"display": "PAMAP2", "config": "configs/pamap2_k8_last.yaml", "channels": 18, "classes": 12},
    "mhealth": {"display": "MHEALTH", "config": "configs/mhealth_k8_last.yaml", "channels": 15, "classes": 12},
}

MODELS = ["cnn1d", "window_gru", "ms_lif_snn", "ms_lif_snn_wide", "ms_ann_tcn", "ms_lif_tcn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count v3 model parameters without training.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=sorted(DATASETS))
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--output", default="results/final_paper_v3/param_dry_run.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for dataset_key in args.datasets:
        spec = DATASETS[dataset_key]
        base_config = load_config(spec["config"])
        for model_label in args.models:
            config = copy.deepcopy(base_config)
            build_name = build_name_for(model_label)
            model_cfg = config.setdefault("model", {})
            if model_label == "ms_lif_snn_wide":
                model_cfg["hidden_dim"] = 224
                model_cfg["branch_dim"] = 64
            model = build_model(build_name, int(spec["channels"]), int(spec["classes"]), model_cfg)
            rows.append(
                {
                    "dataset": spec["display"],
                    "dataset_key": dataset_key,
                    "model": model_label,
                    "build_model": build_name,
                    "input_channels": int(spec["channels"]),
                    "num_classes": int(spec["classes"]),
                    "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
                    "branch_dim": int(model_cfg.get("branch_dim", 32)),
                    "tcn_layers": int(model_cfg.get("tcn_layers", 0)),
                    "params": int(count_parameters(model)),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    tcn_params = frame[frame["model"].eq("ms_lif_tcn")][["dataset_key", "params"]].rename(columns={"params": "ms_lif_tcn_params"})
    frame = frame.merge(tcn_params, on="dataset_key", how="left")
    frame["params_ratio_to_ms_lif_tcn"] = frame["params"] / frame["ms_lif_tcn_params"]
    frame = frame.drop(columns=["ms_lif_tcn_params"])
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"Saved parameter dry run to {output}")


def build_name_for(model_label: str) -> str:
    normalized = model_label.lower().replace("-", "_")
    if normalized == "ms_lif_snn_wide":
        return "ms_lif_snn"
    return normalized


if __name__ == "__main__":
    main()
