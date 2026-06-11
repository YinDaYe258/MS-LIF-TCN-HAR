from __future__ import annotations

from typing import Any

import numpy as np


def fit_train_preprocessor(
    train_x: np.ndarray,
    normalize: str = "none",
    impute_missing: str = "none",
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Fit channel-wise preprocessing statistics from the train split only."""
    normalize = normalize.lower()
    impute_missing = impute_missing.lower()
    if normalize not in {"none", "train_zscore"}:
        raise ValueError(f"Unsupported normalize mode: {normalize}")
    if impute_missing not in {"none", "train_channel_mean"}:
        raise ValueError(f"Unsupported impute_missing mode: {impute_missing}")

    channel_mean = np.nanmean(train_x, axis=(0, 1)).astype(np.float32)
    channel_mean = np.where(np.isnan(channel_mean), 0.0, channel_mean).astype(np.float32)
    filled = _fill_with_channel_mean(train_x, channel_mean) if impute_missing == "train_channel_mean" else train_x
    channel_std = np.nanstd(filled, axis=(0, 1)).astype(np.float32)
    channel_std = np.where((np.isnan(channel_std)) | (channel_std < eps), 1.0, channel_std).astype(np.float32)
    return {
        "normalize": normalize,
        "impute_missing": impute_missing,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "num_train_windows_for_stats": int(train_x.shape[0]),
        "eps": float(eps),
    }


def apply_train_preprocessor(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Apply train-only preprocessing statistics to a split."""
    transformed = x.astype(np.float32, copy=True)
    mean = np.asarray(stats["channel_mean"], dtype=np.float32)
    std = np.asarray(stats["channel_std"], dtype=np.float32)
    if stats.get("impute_missing") == "train_channel_mean":
        transformed = _fill_with_channel_mean(transformed, mean)
    if stats.get("normalize") == "train_zscore":
        transformed = (transformed - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def stats_to_serializable(stats: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy arrays in preprocessing stats into JSON-friendly lists."""
    serializable: dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.astype(float).tolist()
        else:
            serializable[key] = value
    return serializable


def _fill_with_channel_mean(x: np.ndarray, channel_mean: np.ndarray) -> np.ndarray:
    filled = x.astype(np.float32, copy=True)
    if not np.isnan(filled).any():
        return filled
    rows = np.where(np.isnan(filled))
    channel_indices = rows[-1]
    filled[rows] = channel_mean[channel_indices]
    return filled
