from __future__ import annotations

import torch


def apply_sensor_augmentation(
    x: torch.Tensor,
    *,
    jitter_std: float = 0.0,
    scaling_std: float = 0.0,
    channel_dropout_prob: float = 0.0,
    temporal_shift_max: int = 0,
) -> torch.Tensor:
    """Apply lightweight training-only sensor augmentation to [B,K,T,C] tensors."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B,K,T,C], got {tuple(x.shape)}")
    out = x
    if jitter_std > 0:
        out = out + torch.randn_like(out) * float(jitter_std)
    if scaling_std > 0:
        scale = 1.0 + torch.randn((out.shape[0], 1, 1, out.shape[-1]), device=out.device, dtype=out.dtype) * float(
            scaling_std
        )
        out = out * scale
    if channel_dropout_prob > 0:
        keep = torch.rand((out.shape[0], 1, 1, out.shape[-1]), device=out.device) >= float(channel_dropout_prob)
        out = out * keep.to(out.dtype)
    if temporal_shift_max > 0:
        max_shift = int(temporal_shift_max)
        shifts = torch.randint(-max_shift, max_shift + 1, (out.shape[0],), device=out.device)
        shifted = torch.empty_like(out)
        for batch_idx, shift in enumerate(shifts.tolist()):
            shifted[batch_idx] = torch.roll(out[batch_idx], shifts=int(shift), dims=1)
        out = shifted
    return out
