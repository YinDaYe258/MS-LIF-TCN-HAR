from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceDatasetMeta:
    context_len: int
    num_channels: int
    window_size: int
    num_classes: int
    channel_names: tuple[str, ...]
    synthetic: bool = False


class SequenceWindowDataset(Dataset):
    """Build chronological K-window sequences within each subject."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        subjects: np.ndarray,
        context_len: int,
        channel_names: Iterable[str] | None = None,
        max_sequences: int | None = None,
        noise_std: float = 0.0,
        channel_dropout_prob: float = 0.0,
        modality_dropout: str | None = None,
        group_ids: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [N,T,C], got {x.shape}")
        if len(x) != len(y) or len(x) != len(subjects):
            raise ValueError("x, y, and subjects must have the same first dimension")
        if context_len < 1:
            raise ValueError("context_len must be >= 1")

        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.subjects = subjects.astype(np.int64, copy=False)
        if group_ids is not None and len(group_ids) != len(x):
            raise ValueError("group_ids must have the same first dimension as x")
        self.group_ids = group_ids.astype(np.int64, copy=False) if group_ids is not None else None
        self.context_len = context_len
        self.channel_names = tuple(channel_names or [f"ch_{i}" for i in range(x.shape[-1])])
        self.noise_std = float(noise_std)
        self.channel_dropout_prob = float(channel_dropout_prob)
        self.modality_dropout = modality_dropout
        self.seed = int(seed)
        self.indices = self._build_indices()

        if max_sequences is not None:
            self.indices = self.indices[: max(0, int(max_sequences))]

    def _build_indices(self) -> list[np.ndarray]:
        indices: list[np.ndarray] = []
        if self.group_ids is None:
            for subject_id in np.unique(self.subjects):
                subject_indices = np.flatnonzero(self.subjects == subject_id)
                self._append_subject_indices(indices, subject_indices)
        else:
            pairs = np.stack([self.subjects, self.group_ids], axis=1)
            for subject_id, group_id in np.unique(pairs, axis=0):
                group_indices = np.flatnonzero((self.subjects == subject_id) & (self.group_ids == group_id))
                self._append_subject_indices(indices, group_indices)
        return indices

    def _append_subject_indices(self, indices: list[np.ndarray], subject_indices: np.ndarray) -> None:
        if len(subject_indices) < self.context_len:
            return
        for start in range(0, len(subject_indices) - self.context_len + 1):
            indices.append(subject_indices[start : start + self.context_len])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        window_indices = self.indices[idx]
        x_seq = self.x[window_indices].copy()
        y_seq = self.y[window_indices]
        subject_id = self.subjects[window_indices[-1]]
        x_seq = self._apply_perturbations(x_seq, idx)
        sample = {
            "x": torch.from_numpy(x_seq),
            "y": torch.from_numpy(y_seq.copy()),
            "subject": torch.tensor(subject_id, dtype=torch.long),
        }
        if self.group_ids is not None:
            sample["group"] = torch.from_numpy(self.group_ids[window_indices].copy())
        return sample

    def _apply_perturbations(self, x_seq: np.ndarray, idx: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + idx)

        if self.noise_std > 0:
            x_seq = x_seq + rng.normal(0.0, self.noise_std, size=x_seq.shape).astype(np.float32)

        if self.channel_dropout_prob > 0:
            mask = rng.random(x_seq.shape[-1]) >= self.channel_dropout_prob
            x_seq[..., ~mask] = 0.0

        if self.modality_dropout:
            channels = self._modality_channels(self.modality_dropout)
            if channels:
                x_seq[..., channels] = 0.0

        return x_seq.astype(np.float32, copy=False)

    def _modality_channels(self, modality: str) -> list[int]:
        modality = modality.lower()
        selected: list[int] = []
        for idx, name in enumerate(self.channel_names):
            lowered = name.lower()
            if modality in {"acc", "accelerometer"} and "acc" in lowered:
                selected.append(idx)
            if modality in {"gyro", "gyroscope"} and "gyro" in lowered:
                selected.append(idx)
        return selected

    @property
    def meta(self) -> SequenceDatasetMeta:
        return SequenceDatasetMeta(
            context_len=self.context_len,
            num_channels=self.x.shape[-1],
            window_size=self.x.shape[1],
            num_classes=int(self.y.max()) + 1 if len(self.y) else 0,
            channel_names=self.channel_names,
        )
