from __future__ import annotations

import torch


class FastSigmoidSpike(torch.autograd.Function):
    """Hard spike in forward pass, fast-sigmoid surrogate in backward pass."""

    @staticmethod
    def forward(ctx, membrane_minus_threshold: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
        ctx.save_for_backward(membrane_minus_threshold)
        ctx.slope = slope
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        grad = grad_output / (slope * x.abs() + 1.0).pow(2)
        return grad, None


def surrogate_spike(x: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    return FastSigmoidSpike.apply(x, slope)


def inverse_softplus(value: float) -> float:
    tensor = torch.tensor(float(value))
    return float(torch.log(torch.expm1(tensor)))
