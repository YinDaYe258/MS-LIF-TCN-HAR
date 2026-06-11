from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.ucihar import create_ucihar_dataloaders
from src.training.trainer import Trainer
from src.training.utils import append_csv_row, build_model, count_parameters, get_device, load_config, set_seed


SOURCE_RESULT_FILES = [
    "ucihar_cmg_lite_ablation_results.csv",
    "ucihar_matched_protocol_results.csv",
    "ucihar_cmg_diagnostic_results.csv",
    "ucihar_strong_baseline_results.csv",
]


def ablation_specs() -> list[dict[str, Any]]:
    return [
        {"ablation_name": "lif_snn_k8_last", "model": "lif_snn", "context_len": 8},
        {"ablation_name": "full_k8_last", "model": "cmg_lif_lite", "context_len": 8},
        {"ablation_name": "without_context_memory", "model": "cmg_lif_lite", "context_len": 8, "context_memory": False},
        {
            "ablation_name": "without_threshold_modulation",
            "model": "cmg_lif_lite",
            "context_len": 8,
            "threshold_modulation": False,
        },
        {"ablation_name": "alpha_0.5", "model": "cmg_lif_lite", "context_len": 8, "alpha": 0.5},
        {"ablation_name": "alpha_0.8", "model": "cmg_lif_lite", "context_len": 8, "alpha": 0.8},
        {"ablation_name": "alpha_0.9", "model": "cmg_lif_lite", "context_len": 8, "alpha": 0.9},
        {"ablation_name": "num_groups_1", "model": "cmg_lif_lite", "context_len": 8, "num_groups": 1},
        {"ablation_name": "num_groups_4", "model": "cmg_lif_lite", "context_len": 8, "num_groups": 4},
        {"ablation_name": "num_groups_8", "model": "cmg_lif_lite", "context_len": 8, "num_groups": 8},
        {"ablation_name": "num_groups_16", "model": "cmg_lif_lite", "context_len": 8, "num_groups": 16},
        {"ablation_name": "context_len_1", "model": "cmg_lif_lite", "context_len": 1},
        {"ablation_name": "context_len_2", "model": "cmg_lif_lite", "context_len": 2},
        {"ablation_name": "context_len_4", "model": "cmg_lif_lite", "context_len": 4},
        {"ablation_name": "context_len_8", "model": "cmg_lif_lite", "context_len": 8},
    ]


def config_for_spec(base_config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["seed"] = 42
    config.setdefault("dataset", {})["context_len"] = int(spec.get("context_len", 8))
    config.setdefault("training", {})["target_mode"] = "last"
    model_cfg = config.setdefault("model", {})
    for key in ["alpha", "num_groups", "context_memory", "threshold_modulation"]:
        if key in spec:
            model_cfg[key] = spec[key]
    return config


def ablation_exists(result_path: Path, ablation_name: str) -> bool:
    if not result_path.exists():
        return False
    rows = pd.read_csv(result_path)
    return bool((rows["ablation_name"] == ablation_name).any()) if not rows.empty else False


def reusable_existing_row(model: str, config: dict[str, Any], results_dir: Path) -> dict[str, Any] | None:
    context_len = int(config.get("dataset", {}).get("context_len", 8))
    target_mode = str(config.get("training", {}).get("target_mode", "last"))
    model_cfg = config.get("model", {})
    is_default_cmg = (
        model == "cmg_lif_lite"
        and float(model_cfg.get("alpha", 0.8)) == 0.8
        and int(model_cfg.get("num_groups", 8)) == 8
        and bool(model_cfg.get("context_memory", True))
        and bool(model_cfg.get("threshold_modulation", True))
    )
    is_default_lif = model == "lif_snn"
    if not (is_default_cmg or is_default_lif):
        return None
    for file_name in SOURCE_RESULT_FILES:
        path = results_dir / file_name
        if not path.exists():
            continue
        rows = pd.read_csv(path)
        for _, row in rows.iterrows():
            if str(row.get("model", "")) != model:
                continue
            if int(row.get("seed", -1)) != 42:
                continue
            if int(row.get("context_len", -1)) != context_len:
                continue
            if str(row.get("target_mode", "last")) != target_mode:
                continue
            if bool(row.get("synthetic_data", False)) or bool(row.get("smoke_test", False)):
                continue
            return row.to_dict()
    return None


def row_from_metrics(
    ablation_name: str,
    model_name: str,
    config: dict[str, Any],
    params: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    return {
        "ablation_name": ablation_name,
        "model": model_name,
        "seed": int(config.get("seed", 42)),
        "context_len": int(config.get("dataset", {}).get("context_len", 8)),
        "target_mode": str(config.get("training", {}).get("target_mode", "last")),
        "alpha": float(model_cfg.get("alpha", 0.8)),
        "num_groups": int(model_cfg.get("num_groups", 8)),
        "context_memory": bool(model_cfg.get("context_memory", True)),
        "threshold_modulation": bool(model_cfg.get("threshold_modulation", True)),
        "params": params,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "spike_rate": metrics.get("spike_rate", 0.0),
        "best_epoch": metrics["best_epoch"],
        "checkpoint": metrics["checkpoint"],
    }


def row_from_existing(ablation_name: str, model_name: str, config: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    return row_from_metrics(
        ablation_name,
        model_name,
        config,
        int(existing.get("params", 0)),
        {
            "accuracy": float(existing.get("accuracy", 0.0)),
            "macro_f1": float(existing.get("macro_f1", 0.0)),
            "weighted_f1": float(existing.get("weighted_f1", 0.0)),
            "spike_rate": float(existing.get("spike_rate", 0.0)),
            "best_epoch": int(existing.get("best_epoch", 0)),
            "checkpoint": existing.get("checkpoint", ""),
        },
    )


def run_variant(spec: dict[str, Any], base_config: dict[str, Any]) -> dict[str, Any]:
    config = config_for_spec(base_config, spec)
    model_name = str(spec["model"])
    set_seed(42)
    loaders, meta = create_ucihar_dataloaders(config, model_name=model_name, smoke_test=False)
    device = get_device(config.get("device", "auto"))
    model = build_model(model_name, meta.num_channels, meta.num_classes, config.get("model", {}))
    run_name = f"ucihar_ablate_{spec['ablation_name']}_seed42"
    trainer = Trainer(
        model,
        loaders,
        config,
        device,
        run_name,
        results_dir=config.get("results", {}).get("dir", "results"),
        num_classes=meta.num_classes,
    )
    metrics = trainer.fit()
    return row_from_metrics(str(spec["ablation_name"]), model_name, config, count_parameters(model), metrics)


def main() -> None:
    base_config = load_config("configs/ucihar_k8_last.yaml")
    results_dir = Path(base_config.get("results", {}).get("dir", "results"))
    result_path = results_dir / "ucihar_cmg_lite_ablation_results.csv"
    for spec in ablation_specs():
        ablation_name = str(spec["ablation_name"])
        if ablation_exists(result_path, ablation_name):
            print(f"Skipping existing ablation: {ablation_name}")
            continue
        config = config_for_spec(base_config, spec)
        model_name = str(spec["model"])
        existing = reusable_existing_row(model_name, config, results_dir)
        row = row_from_existing(ablation_name, model_name, config, existing) if existing else run_variant(spec, base_config)
        append_csv_row(result_path, row)
        print(row)


if __name__ == "__main__":
    main()
