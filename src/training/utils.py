from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.models import (
    CMGLIFLiteSNN,
    CMGLIFSNN,
    CNN1D,
    GRUClassifier,
    LIFSNN,
    MSANNTCN,
    MSCMGLIFSNN,
    MSCMGTCNLIFSNN,
    MSCNN1D,
    MSLIFSNN,
    MSLIFTCNAttnSNN,
    MSLIFTCNGateSNN,
    MSLIFTCNSNN,
    WindowGRU,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model(model_name: str, input_channels: int, num_classes: int, model_cfg: dict[str, Any]) -> torch.nn.Module:
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    dropout = float(model_cfg.get("dropout", 0.2))
    normalized = model_name.lower().replace("-", "_")
    if normalized in {"cnn", "cnn1d"}:
        return CNN1D(input_channels, num_classes, hidden_dim=hidden_dim, dropout=dropout)
    if normalized in {"gru", "rnn_gru"}:
        return GRUClassifier(input_channels, num_classes, hidden_dim=hidden_dim, dropout=dropout)
    if normalized in {"ms_cnn", "ms_cnn1d", "multiscale_cnn"}:
        return MSCNN1D(
            input_channels,
            num_classes,
            hidden_dim=hidden_dim,
            branch_dim=int(model_cfg.get("branch_dim", 32)),
            dropout=dropout,
        )
    if normalized in {"window_gru", "cross_window_gru"}:
        return WindowGRU(input_channels, num_classes, hidden_dim=hidden_dim, dropout=dropout)
    if normalized in {"lif", "lif_snn", "vanilla_lif", "vanilla_lif_snn"}:
        return LIFSNN(
            input_channels,
            num_classes,
            hidden_dim=hidden_dim,
            beta=float(model_cfg.get("beta", 0.9)),
            theta_init=float(model_cfg.get("theta_init", 1.0)),
            dropout=dropout,
        )
    if normalized in {"cmg_lif", "cmg_lif_snn", "cmglif"}:
        return CMGLIFSNN(
            input_channels,
            num_classes,
            hidden_dim=hidden_dim,
            beta=float(model_cfg.get("beta", 0.9)),
            alpha=float(model_cfg.get("alpha", 0.8)),
            theta_init=float(model_cfg.get("theta_init", 1.0)),
            theta_scale=float(model_cfg.get("theta_scale", 0.5)),
            dropout=dropout,
            threshold_modulation=bool(model_cfg.get("threshold_modulation", True)),
            context_memory=bool(model_cfg.get("context_memory", True)),
        )
    if normalized in {"cmg_lif_lite", "cmg_lif_lite_snn", "cmglif_lite"}:
        return CMGLIFLiteSNN(
            input_channels,
            num_classes,
            hidden_dim=hidden_dim,
            beta=float(model_cfg.get("beta", 0.9)),
            alpha=float(model_cfg.get("alpha", 0.8)),
            theta_init=float(model_cfg.get("theta_init", 1.0)),
            theta_scale=float(model_cfg.get("theta_scale", 0.1)),
            num_groups=int(model_cfg.get("num_groups", 8)),
            dropout=dropout,
            threshold_modulation=bool(model_cfg.get("threshold_modulation", True)),
            context_memory=bool(model_cfg.get("context_memory", True)),
        )
    if normalized in {"ms_cmg_lif", "ms_cmg_lif_snn", "mscmg_lif"}:
        return MSCMGLIFSNN(
            input_channels,
            num_classes,
            hidden_dim=hidden_dim,
            branch_dim=int(model_cfg.get("branch_dim", 32)),
            beta=float(model_cfg.get("beta", 0.9)),
            alpha=float(model_cfg.get("alpha", 0.8)),
            theta_init=float(model_cfg.get("theta_init", 1.0)),
            theta_scale=float(model_cfg.get("theta_scale", 0.1)),
            num_groups=int(model_cfg.get("num_groups", 8)),
            dropout=dropout,
            threshold_modulation=bool(model_cfg.get("threshold_modulation", True)),
            context_memory=bool(model_cfg.get("context_memory", True)),
        )
    if normalized in {"ms_lif", "ms_lif_snn", "multiscale_lif_snn"}:
        return MSLIFSNN(
            input_channels,
            num_classes,
            hidden_dim=hidden_dim,
            branch_dim=int(model_cfg.get("branch_dim", 32)),
            beta=float(model_cfg.get("beta", 0.9)),
            theta_init=float(model_cfg.get("theta_init", 1.0)),
            dropout=dropout,
        )
    if normalized in {"ms_lif_tcn", "ms_lif_tcn_snn", "wt_mslif"}:
        return MSLIFTCNSNN(
            input_channels,
            num_classes,
            hidden_dim=hidden_dim,
            branch_dim=int(model_cfg.get("branch_dim", 32)),
            beta=float(model_cfg.get("beta", 0.9)),
            theta_init=float(model_cfg.get("theta_init", 1.0)),
            dropout=dropout,
            tcn_kernel_size=int(model_cfg.get("tcn_kernel_size", 3)),
            tcn_dropout=float(model_cfg.get("tcn_dropout", dropout)),
            tcn_layers=int(model_cfg.get("tcn_layers", 2)),
            encoder_mode=str(model_cfg.get("encoder_mode", "multi")),
            single_kernel_size=int(model_cfg.get("single_kernel_size", 5)),
        )
    if normalized in {"ms_lif_tcn_attn", "ms_lif_tcn_attention", "wt_mslif_attn"}:
        return MSLIFTCNAttnSNN(
            input_channels,
            num_classes,
            hidden_dim=hidden_dim,
            branch_dim=int(model_cfg.get("branch_dim", 32)),
            beta=float(model_cfg.get("beta", 0.9)),
            theta_init=float(model_cfg.get("theta_init", 1.0)),
            dropout=dropout,
            tcn_kernel_size=int(model_cfg.get("tcn_kernel_size", 3)),
            tcn_dropout=float(model_cfg.get("tcn_dropout", dropout)),
            tcn_layers=int(model_cfg.get("tcn_layers", 2)),
            attention_hidden_dim=int(model_cfg.get("attention_hidden_dim", 64)),
            encoder_mode=str(model_cfg.get("encoder_mode", "multi")),
            single_kernel_size=int(model_cfg.get("single_kernel_size", 5)),
        )
    if normalized in {"ms_lif_tcn_gate", "ms_lif_tcn_gated", "ms_lif_tcn_resgate", "ms_lif_tcn_gate_scalar"}:
        return MSLIFTCNGateSNN(
            input_channels,
            num_classes,
            hidden_dim=hidden_dim,
            branch_dim=int(model_cfg.get("branch_dim", 32)),
            beta=float(model_cfg.get("beta", 0.9)),
            theta_init=float(model_cfg.get("theta_init", 1.0)),
            dropout=dropout,
            tcn_kernel_size=int(model_cfg.get("tcn_kernel_size", 3)),
            tcn_dropout=float(model_cfg.get("tcn_dropout", dropout)),
            tcn_layers=int(model_cfg.get("tcn_layers", 2)),
            gate_hidden_dim=int(model_cfg.get("gate_hidden_dim", 64)),
            gate_dropout=float(model_cfg.get("gate_dropout", 0.1)),
            gate_mode="scalar",
            encoder_mode=str(model_cfg.get("encoder_mode", "multi")),
            single_kernel_size=int(model_cfg.get("single_kernel_size", 5)),
        )
    if normalized == "ms_lif_tcn_gate_channel":
        return MSLIFTCNGateSNN(
            input_channels,
            num_classes,
            hidden_dim=hidden_dim,
            branch_dim=int(model_cfg.get("branch_dim", 32)),
            beta=float(model_cfg.get("beta", 0.9)),
            theta_init=float(model_cfg.get("theta_init", 1.0)),
            dropout=dropout,
            tcn_kernel_size=int(model_cfg.get("tcn_kernel_size", 3)),
            tcn_dropout=float(model_cfg.get("tcn_dropout", dropout)),
            tcn_layers=int(model_cfg.get("tcn_layers", 2)),
            gate_hidden_dim=int(model_cfg.get("gate_hidden_dim", 64)),
            gate_dropout=float(model_cfg.get("gate_dropout", 0.1)),
            gate_mode="channel",
            encoder_mode=str(model_cfg.get("encoder_mode", "multi")),
            single_kernel_size=int(model_cfg.get("single_kernel_size", 5)),
        )
    if normalized in {"ms_ann_tcn", "ms_cnn_tcn", "ann_tcn", "ms_tcn_ann"}:
        return MSANNTCN(
            input_channels,
            num_classes,
            hidden_dim=hidden_dim,
            branch_dim=int(model_cfg.get("branch_dim", 32)),
            dropout=dropout,
            tcn_kernel_size=int(model_cfg.get("tcn_kernel_size", 3)),
            tcn_dropout=float(model_cfg.get("tcn_dropout", dropout)),
            tcn_layers=int(model_cfg.get("tcn_layers", 2)),
            encoder_mode=str(model_cfg.get("encoder_mode", "multi")),
            single_kernel_size=int(model_cfg.get("single_kernel_size", 5)),
        )
    if normalized in {"ms_cmg_tcn", "ms_cmg_tcn_lif", "ms_cmg_tcn_snn"}:
        return MSCMGTCNLIFSNN(
            input_channels,
            num_classes,
            hidden_dim=hidden_dim,
            branch_dim=int(model_cfg.get("branch_dim", 32)),
            beta=float(model_cfg.get("beta", 0.9)),
            alpha=float(model_cfg.get("alpha", 0.8)),
            theta_init=float(model_cfg.get("theta_init", 1.0)),
            theta_scale=float(model_cfg.get("theta_scale", 0.1)),
            num_groups=int(model_cfg.get("num_groups", 8)),
            dropout=dropout,
            threshold_modulation=bool(model_cfg.get("threshold_modulation", True)),
            context_memory=bool(model_cfg.get("context_memory", True)),
            tcn_kernel_size=int(model_cfg.get("tcn_kernel_size", 3)),
            tcn_dropout=float(model_cfg.get("tcn_dropout", dropout)),
            tcn_layers=int(model_cfg.get("tcn_layers", 2)),
            encoder_mode=str(model_cfg.get("encoder_mode", "multi")),
            single_kernel_size=int(model_cfg.get("single_kernel_size", 5)),
        )
    raise ValueError(f"Unknown model name: {model_name}")


def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if path.exists():
        existing = pd.read_csv(path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True, sort=False)
    else:
        updated = pd.DataFrame([row])
    updated.to_csv(path, index=False)
