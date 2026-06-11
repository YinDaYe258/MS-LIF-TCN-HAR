from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .utils import build_model


CHECKPOINT_RESULT_FILES = (
    "ucihar_strong_baseline_results.csv",
    "ucihar_main_results.csv",
    "ucihar_matched_protocol_results.csv",
)


def _normal_target_mode(value: Any) -> str:
    if value is None or pd.isna(value) or value == "":
        return "all"
    return str(value)


def _normal_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def find_checkpoint(
    results_dir: str | Path,
    model: str,
    context_len: int,
    target_mode: str,
    seed: int,
) -> Path:
    results_dir = Path(results_dir)
    matches: list[Path] = []
    for file_name in CHECKPOINT_RESULT_FILES:
        path = results_dir / file_name
        if not path.exists():
            continue
        rows = pd.read_csv(path)
        for _, row in rows.iterrows():
            if str(row.get("model", "")) != model:
                continue
            if int(row.get("seed", -1)) != int(seed):
                continue
            if int(row.get("context_len", -1)) != int(context_len):
                continue
            if _normal_target_mode(row.get("target_mode")) != str(target_mode):
                continue
            if _normal_bool(row.get("synthetic_data", False)) or _normal_bool(row.get("smoke_test", False)):
                continue
            checkpoint = row.get("checkpoint", "")
            if not isinstance(checkpoint, str) or not checkpoint:
                continue
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = results_dir.parent / checkpoint_path
            if checkpoint_path.exists():
                matches.append(checkpoint_path)
                break
        if matches:
            break
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for model={model}, seed={seed}, context_len={context_len}, target_mode={target_mode}"
        )
    return matches[0]


def load_model_from_checkpoint(
    model_name: str,
    config: dict[str, Any],
    input_channels: int,
    num_classes: int,
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config")
    model_cfg = config.get("model", {})
    if isinstance(checkpoint_config, dict):
        model_cfg = checkpoint_config.get("model", model_cfg)
    model = build_model(model_name, input_channels, num_classes, model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
