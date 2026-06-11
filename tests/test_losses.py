from __future__ import annotations

import torch

from src.training.losses import sequence_classification_loss, supervised_contrastive_loss


def test_sequence_loss_target_mode_all() -> None:
    logits = torch.randn(2, 3, 6, requires_grad=True)
    targets = torch.tensor([[0, 1, 2], [3, 4, 5]])
    loss, details = sequence_classification_loss({"logits": logits}, targets, target_mode="all")

    assert loss.ndim == 0
    assert details["ce_loss"] > 0
    loss.backward()
    assert logits.grad is not None


def test_sequence_loss_target_mode_last() -> None:
    logits = torch.randn(2, 3, 6, requires_grad=True)
    targets = torch.tensor([[0, 1, 2], [3, 4, 5]])
    loss, details = sequence_classification_loss({"logits": logits}, targets, target_mode="last")

    assert loss.ndim == 0
    assert details["ce_loss"] > 0


def test_supervised_contrastive_loss_has_gradient() -> None:
    features = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = supervised_contrastive_loss(features, labels, temperature=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None


def test_sequence_loss_with_supcon_auxiliary() -> None:
    logits = torch.randn(4, 3, 5, requires_grad=True)
    features = torch.randn(4, 3, 8, requires_grad=True)
    targets = torch.tensor([[0, 1, 1], [1, 1, 1], [2, 3, 3], [2, 3, 3]])
    loss, details = sequence_classification_loss(
        {"logits": logits, "features": features},
        targets,
        target_mode="last",
        aux_loss_cfg={
            "supervised_contrastive": {
                "enabled": True,
                "weight": 0.05,
                "temperature": 0.2,
            }
        },
    )
    assert torch.isfinite(loss)
    assert details["supcon_loss"] >= 0.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[:, :-1].abs().sum().item() == 0.0
