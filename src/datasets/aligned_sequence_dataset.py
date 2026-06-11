from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .sequence_dataset import SequenceWindowDataset


class AlignedSequenceWindowDataset(SequenceWindowDataset):
    """Build K-window sequences aligned to a shared Kmax final target set."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        subjects: np.ndarray,
        context_len: int,
        aligned_context_max: int,
        channel_names: Iterable[str] | None = None,
        max_sequences: int | None = None,
        noise_std: float = 0.0,
        channel_dropout_prob: float = 0.0,
        modality_dropout: str | None = None,
        group_ids: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        self.aligned_context_max = int(aligned_context_max)
        if self.aligned_context_max < 1:
            raise ValueError("aligned_context_max must be >= 1")
        if int(context_len) > self.aligned_context_max:
            raise ValueError("context_len must be <= aligned_context_max")
        super().__init__(
            x=x,
            y=y,
            subjects=subjects,
            context_len=context_len,
            channel_names=channel_names,
            max_sequences=max_sequences,
            noise_std=noise_std,
            channel_dropout_prob=channel_dropout_prob,
            modality_dropout=modality_dropout,
            group_ids=group_ids,
            seed=seed,
        )

    def _append_subject_indices(self, indices: list[np.ndarray], subject_indices: np.ndarray) -> None:
        if len(subject_indices) < self.aligned_context_max:
            return
        for final_offset in range(self.aligned_context_max - 1, len(subject_indices)):
            start = final_offset - self.context_len + 1
            indices.append(subject_indices[start : final_offset + 1])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        sample["aligned_context_max"] = torch.tensor(self.aligned_context_max, dtype=torch.long)
        return sample
