from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


def sequence_classification_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    spike_reg_lambda: float = 0.0,
    target_mode: str = "all",
    loss_type: str = "ce",
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    aux_loss_cfg: dict | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["logits"]
    ce_logits, ce_targets = select_target_mode(logits, targets, target_mode)
    ce_loss = supervised_loss(ce_logits, ce_targets, loss_type, class_weights, focal_gamma)
    spike_rate = outputs.get("spike_rate")
    total_loss = ce_loss
    supcon_value = 0.0
    if aux_loss_cfg:
        supcon_cfg = aux_loss_cfg.get("supervised_contrastive", {})
        if bool(supcon_cfg.get("enabled", False)):
            if "features" not in outputs:
                raise ValueError("supervised_contrastive loss requires model outputs['features']")
            features, feature_targets = select_target_mode(outputs["features"], targets, target_mode)
            supcon_loss = supervised_contrastive_loss(
                features,
                feature_targets,
                temperature=float(supcon_cfg.get("temperature", 0.2)),
            )
            total_loss = total_loss + float(supcon_cfg.get("weight", 0.05)) * supcon_loss
            supcon_value = float(supcon_loss.detach().cpu())
        center_cfg = aux_loss_cfg.get("center_loss", {})
        if bool(center_cfg.get("enabled", False)):
            raise NotImplementedError("center_loss is configured but not implemented in this training path")
    spike_value = 0.0
    if spike_rate is not None and spike_reg_lambda > 0:
        total_loss = total_loss + float(spike_reg_lambda) * spike_rate
        spike_value = float(spike_rate.detach().cpu())
    return total_loss, {
        "ce_loss": float(ce_loss.detach().cpu()),
        "supcon_loss": supcon_value,
        "spike_rate": spike_value,
    }


def supervised_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "ce",
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    normalized = loss_type.lower()
    weights = class_weights.to(logits.device) if class_weights is not None else None
    if normalized == "ce":
        return F.cross_entropy(logits, targets)
    if normalized == "weighted_ce":
        return F.cross_entropy(logits, targets, weight=weights)
    if normalized in {"focal", "weighted_focal"}:
        focal_weights = weights if normalized == "weighted_focal" else None
        ce = F.cross_entropy(logits, targets, weight=focal_weights, reduction="none")
        pt = torch.exp(-ce.detach())
        return (((1.0 - pt) ** float(focal_gamma)) * ce).mean()
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def select_target_mode(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_mode == "all":
        return logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    if target_mode == "last":
        return logits[:, -1, :], targets[:, -1]
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def supervised_contrastive_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    if targets.ndim != 1:
        targets = targets.reshape(-1)
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same batch dimension")
    if features.shape[0] < 2:
        return features.sum() * 0.0

    features = F.normalize(features, dim=1)
    logits = features @ features.T / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive_mask = targets[:, None].eq(targets[None, :]) & ~eye
    logits_mask = ~eye

    exp_logits = torch.exp(logits) * logits_mask.to(logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.sum() * 0.0
    mean_log_prob_pos = (positive_mask.to(log_prob.dtype) * log_prob).sum(dim=1)[valid] / positive_count[valid]
    return -mean_log_prob_pos.mean()


def compute_class_weights_from_loader(
    loader: DataLoader,
    num_classes: int,
    target_mode: str = "all",
) -> torch.Tensor:
    counts = torch.zeros(int(num_classes), dtype=torch.float32)
    for batch in loader:
        labels = batch["y"]
        if target_mode == "last":
            labels = labels[:, -1]
        else:
            labels = labels.reshape(-1)
        values = torch.bincount(labels.to(torch.long), minlength=int(num_classes)).to(torch.float32)
        counts += values[: int(num_classes)]
    weights = torch.zeros_like(counts)
    valid = counts > 0
    if valid.any():
        weights[valid] = counts[valid].sum() / (valid.sum().to(torch.float32) * counts[valid])
    return weights
