from __future__ import annotations

from typing import Any

import pandas as pd
import torch


def count_params(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_size_mb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024.0 * 1024.0)


def estimate_spike_count_per_sample(
    spike_rate: float,
    context_len: int,
    window_size: int,
    hidden_dim: int,
) -> float:
    return float(spike_rate) * int(context_len) * int(window_size) * int(hidden_dim)


def estimate_encoder_macs(
    context_len: int,
    window_size: int,
    input_channels: int,
    hidden_dim: int,
) -> int:
    return int(context_len) * int(window_size) * int(input_channels) * int(hidden_dim)


def estimate_multiscale_encoder_macs(
    context_len: int,
    window_size: int,
    input_channels: int,
    hidden_dim: int,
    branch_dim: int,
    kernel_sizes: tuple[int, ...] = (3, 5, 9),
) -> int:
    branch_macs = sum(int(input_channels) * int(branch_dim) * int(kernel) for kernel in kernel_sizes)
    projection_macs = int(branch_dim) * len(kernel_sizes) * int(hidden_dim)
    return int(context_len) * int(window_size) * (branch_macs + projection_macs)


def estimate_window_gru_ops(
    context_len: int,
    window_size: int,
    input_channels: int,
    hidden_dim: int,
    num_classes: int,
    target_mode: str,
) -> dict[str, int]:
    window_encoder_macs = int(context_len) * int(window_size) * int(input_channels) * int(hidden_dim) * 5
    gru_ops_proxy = int(context_len) * 3 * (
        int(hidden_dim) * int(hidden_dim) + int(hidden_dim) * int(hidden_dim)
    )
    classifier_ops = estimate_classifier_ops(context_len, hidden_dim, num_classes, target_mode)
    return {
        "encoder_macs": window_encoder_macs,
        "recurrent_ops": gru_ops_proxy,
        "classifier_ops": classifier_ops,
        "total_ops_proxy": window_encoder_macs + gru_ops_proxy + classifier_ops,
    }


def estimate_gate_macs(
    model_name: str,
    context_len: int,
    hidden_dim: int,
    num_groups: int | None = None,
) -> int:
    normalized = model_name.lower().replace("-", "_")
    if normalized in {
        "lif",
        "lif_snn",
        "vanilla_lif",
        "vanilla_lif_snn",
        "ms_lif",
        "ms_lif_snn",
        "cnn1d",
        "cnn",
        "gru",
        "ms_cnn",
        "ms_cnn1d",
        "window_gru",
    }:
        return 0
    if normalized in {"cmg_lif_lite", "cmg_lif_lite_snn", "cmglif_lite", "ms_cmg_lif", "ms_cmg_lif_snn"}:
        if num_groups is None:
            raise ValueError("num_groups is required for group-wise CMG gate MAC estimate")
        return int(context_len) * int(hidden_dim) * int(num_groups)
    if normalized in {"cmg_lif", "cmg_lif_snn", "cmglif"}:
        return int(context_len) * int(hidden_dim) * int(hidden_dim)
    return 0


def estimate_classifier_ops(
    context_len: int,
    hidden_dim: int,
    num_classes: int,
    target_mode: str,
) -> int:
    if target_mode == "all":
        return int(context_len) * int(hidden_dim) * int(num_classes)
    if target_mode == "last":
        return int(hidden_dim) * int(num_classes)
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def summarize_efficiency(row: pd.Series | dict[str, Any], config: dict[str, Any], model_name: str) -> dict[str, Any]:
    row_dict = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    context_len = int(row_dict.get("context_len") or dataset_cfg.get("context_len", 1))
    target_mode = str(row_dict.get("target_mode") or training_cfg.get("target_mode", "all") or "all")
    if target_mode == "nan":
        target_mode = "all"

    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    window_size = int(dataset_cfg.get("window_size", 128))
    input_channels = int(dataset_cfg.get("input_channels", 9))
    num_classes = int(dataset_cfg.get("num_classes", 6))
    num_groups = int(model_cfg.get("num_groups", 8))
    branch_dim = int(model_cfg.get("branch_dim", 32))
    params = int(row_dict.get("params", 0) or 0)
    spike_rate = float(row_dict.get("spike_rate", 0.0) or 0.0)

    normalized_model = model_name.lower().replace("-", "_")
    spike_count = estimate_spike_count_per_sample(spike_rate, context_len, window_size, hidden_dim)
    recurrent_ops = 0
    if normalized_model in {"ms_cmg_lif", "ms_cmg_lif_snn", "ms_lif", "ms_lif_snn", "ms_cnn", "ms_cnn1d"}:
        encoder_macs = estimate_multiscale_encoder_macs(
            context_len,
            window_size,
            input_channels,
            hidden_dim,
            branch_dim,
        )
    elif normalized_model == "window_gru":
        ops = estimate_window_gru_ops(context_len, window_size, input_channels, hidden_dim, num_classes, target_mode)
        encoder_macs = ops["encoder_macs"]
        recurrent_ops = ops["recurrent_ops"]
    else:
        encoder_macs = estimate_encoder_macs(context_len, window_size, input_channels, hidden_dim)
    gate_macs = estimate_gate_macs(model_name, context_len, hidden_dim, num_groups=num_groups)
    classifier_ops = (
        ops["classifier_ops"] if normalized_model == "window_gru" else estimate_classifier_ops(context_len, hidden_dim, num_classes, target_mode)
    )
    total_ops_proxy = encoder_macs + recurrent_ops + gate_macs + classifier_ops

    note_parts = ["proxy_only_not_measured_power"]
    if normalized_model in {"cnn", "cnn1d", "gru", "ms_cnn", "ms_cnn1d", "window_gru"}:
        note_parts.append("non_spiking_model")
    if bool(row_dict.get("smoke_test", False)):
        note_parts.append("smoke_test")
    if bool(row_dict.get("synthetic_data", False)):
        note_parts.append("synthetic_data")

    return {
        "dataset": row_dict.get("dataset", dataset_cfg.get("name", "")),
        "model": model_name,
        "seed": int(row_dict.get("seed", config.get("seed", 0)) or 0),
        "context_len": context_len,
        "target_mode": target_mode,
        "accuracy": float(row_dict.get("accuracy", 0.0) or 0.0),
        "macro_f1": float(row_dict.get("macro_f1", 0.0) or 0.0),
        "weighted_f1": float(row_dict.get("weighted_f1", 0.0) or 0.0),
        "params": params,
        "model_size_mb": params * 4.0 / (1024.0 * 1024.0),
        "spike_rate": spike_rate,
        "spike_count_per_sample": spike_count,
        "encoder_macs": encoder_macs,
        "recurrent_ops": recurrent_ops,
        "gate_macs": gate_macs,
        "classifier_ops": classifier_ops,
        "total_ops_proxy": total_ops_proxy,
        "note": ";".join(note_parts),
    }
