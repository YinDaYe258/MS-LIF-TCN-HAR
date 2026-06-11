from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .augmentation import apply_sensor_augmentation
from .losses import compute_class_weights_from_loader, sequence_classification_loss
from .metrics import classification_metrics
from .utils import ensure_dir


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loaders: dict[str, torch.utils.data.DataLoader],
        config: dict[str, Any],
        device: torch.device,
        run_name: str,
        results_dir: str | Path = "results",
        num_classes: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.loaders = loaders
        self.config = config
        self.device = device
        self.run_name = run_name
        self.results_dir = ensure_dir(results_dir)
        self.training_cfg = config.get("training", {})
        self.target_mode = str(self.training_cfg.get("target_mode", "all"))
        self.num_classes = num_classes
        self.loss_type = str(self.training_cfg.get("loss_type", "ce"))
        self.focal_gamma = float(self.training_cfg.get("focal_gamma", 2.0))
        self.aux_loss_cfg = self.training_cfg.get("aux_loss", {})
        self.augmentation_cfg = self.training_cfg.get("augmentation", {})
        self.class_weights = self._build_class_weights()
        self.checkpoint_path = self.results_dir / f"{run_name}_best.pt"
        self.epoch_log_path = self.results_dir / f"{run_name}_epoch_log.csv"

    def fit(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(self.training_cfg.get("weight_decay", 1e-4)),
        )
        epochs = int(self.training_cfg.get("epochs", 30))
        patience = int(self.training_cfg.get("patience", 8))
        best_macro_f1 = -1.0
        best_epoch = 0
        stale_epochs = 0
        logs: list[dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_train_epoch(optimizer)
            val_metrics = self.evaluate("val")
            row = {"epoch": epoch, **prefix_keys(train_metrics, "train"), **prefix_keys(val_metrics, "val")}
            logs.append(row)
            pd.DataFrame(logs).to_csv(self.epoch_log_path, index=False)

            if float(val_metrics["macro_f1"]) > best_macro_f1:
                best_macro_f1 = float(val_metrics["macro_f1"])
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": self.config,
                        "run_name": self.run_name,
                        "best_epoch": best_epoch,
                        "best_val_macro_f1": best_macro_f1,
                    },
                    self.checkpoint_path,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = self.evaluate("test")
        test_metrics.update(
            {
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_macro_f1,
                "checkpoint": str(self.checkpoint_path),
                "epoch_log": str(self.epoch_log_path),
            }
        )
        return test_metrics

    def _run_train_epoch(self, optimizer: torch.optim.Optimizer) -> dict[str, float]:
        self.model.train()
        grad_clip = float(self.training_cfg.get("grad_clip", 1.0))
        spike_reg_lambda = float(self.training_cfg.get("spike_reg_lambda", 0.0))
        losses = []
        ce_losses = []
        spike_rates = []
        gate_means = []
        gate_stds = []
        y_true = []
        y_pred = []

        disable_tqdm = bool(self.training_cfg.get("disable_tqdm", False))
        for batch in tqdm(self.loaders["train"], desc=f"{self.run_name}:train", leave=False, disable=disable_tqdm):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            x = self._augment_train_batch(x)
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model(x)
            loss, details = sequence_classification_loss(
                outputs,
                y,
                spike_reg_lambda,
                self.target_mode,
                self.loss_type,
                self.class_weights,
                self.focal_gamma,
                self.aux_loss_cfg,
            )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            ce_losses.append(details["ce_loss"])
            if "spike_rate" in outputs:
                spike_rates.append(float(outputs["spike_rate"].detach().cpu()))
            if "gate_mean" in outputs:
                gate_means.append(float(outputs["gate_mean"].detach().cpu()))
            if "gate_std" in outputs:
                gate_stds.append(float(outputs["gate_std"].detach().cpu()))
            true_batch, pred_batch = select_predictions_for_metrics(outputs["logits"], y, self.target_mode)
            y_true.append(true_batch.detach().cpu().numpy())
            y_pred.append(pred_batch.detach().cpu().numpy())

        metrics = classification_metrics(
            np.concatenate(y_true).reshape(-1),
            np.concatenate(y_pred).reshape(-1),
            self._num_classes(),
        )
        metrics.pop("confusion_matrix", None)
        metrics.update(
            {
                "loss": float(np.mean(losses)),
                "ce_loss": float(np.mean(ce_losses)),
                "spike_rate": float(np.mean(spike_rates)) if spike_rates else 0.0,
                "gate_mean": float(np.mean(gate_means)) if gate_means else np.nan,
                "gate_std": float(np.mean(gate_stds)) if gate_stds else np.nan,
            }
        )
        return metrics

    @torch.no_grad()
    def evaluate(self, split: str) -> dict[str, Any]:
        self.model.eval()
        spike_reg_lambda = float(self.training_cfg.get("spike_reg_lambda", 0.0))
        losses = []
        ce_losses = []
        spike_rates = []
        gate_means = []
        gate_stds = []
        y_true = []
        y_pred = []

        for batch in self.loaders[split]:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            outputs = self.model(x)
            loss, details = sequence_classification_loss(
                outputs,
                y,
                spike_reg_lambda,
                self.target_mode,
                self.loss_type,
                self.class_weights,
                self.focal_gamma,
                self.aux_loss_cfg,
            )
            losses.append(float(loss.detach().cpu()))
            ce_losses.append(details["ce_loss"])
            if "spike_rate" in outputs:
                spike_rates.append(float(outputs["spike_rate"].detach().cpu()))
            if "gate_mean" in outputs:
                gate_means.append(float(outputs["gate_mean"].detach().cpu()))
            if "gate_std" in outputs:
                gate_stds.append(float(outputs["gate_std"].detach().cpu()))
            true_batch, pred_batch = select_predictions_for_metrics(outputs["logits"], y, self.target_mode)
            y_true.append(true_batch.detach().cpu().numpy())
            y_pred.append(pred_batch.detach().cpu().numpy())

        metrics = classification_metrics(
            np.concatenate(y_true).reshape(-1),
            np.concatenate(y_pred).reshape(-1),
            self._num_classes(),
        )
        metrics.update(
            {
                "loss": float(np.mean(losses)),
                "ce_loss": float(np.mean(ce_losses)),
                "spike_rate": float(np.mean(spike_rates)) if spike_rates else 0.0,
                "gate_mean": float(np.mean(gate_means)) if gate_means else np.nan,
                "gate_std": float(np.mean(gate_stds)) if gate_stds else np.nan,
            }
        )
        if split == "test":
            confusion_path = self.results_dir / f"{self.run_name}_confusion_matrix.json"
            confusion_path.write_text(json.dumps(metrics["confusion_matrix"], indent=2), encoding="utf-8")
            metrics["confusion_matrix_path"] = str(confusion_path)
        return metrics

    def _num_classes(self) -> int:
        if self.num_classes is None:
            raise ValueError("Trainer requires num_classes for metric computation")
        return int(self.num_classes)

    def _augment_train_batch(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.augmentation_cfg.get("enabled", False)):
            return x
        return apply_sensor_augmentation(
            x,
            jitter_std=float(self.augmentation_cfg.get("jitter_std", 0.0)),
            scaling_std=float(self.augmentation_cfg.get("scaling_std", 0.0)),
            channel_dropout_prob=float(self.augmentation_cfg.get("channel_dropout_prob", 0.0)),
            temporal_shift_max=int(self.augmentation_cfg.get("temporal_shift_max", 0)),
        )

    def _build_class_weights(self) -> torch.Tensor | None:
        if self.num_classes is None:
            return None
        if self.loss_type not in {"weighted_ce", "weighted_focal"}:
            return None
        source = str(self.training_cfg.get("class_weight_source", "train_labels"))
        if source != "train_labels":
            raise ValueError("Only class_weight_source=train_labels is supported")
        return compute_class_weights_from_loader(
            self.loaders["train"],
            int(self.num_classes),
            self.target_mode,
        ).to(self.device)


def prefix_keys(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items() if key != "confusion_matrix"}


def select_predictions_for_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_mode == "all":
        return targets, logits.argmax(dim=-1)
    if target_mode == "last":
        return targets[:, -1], logits[:, -1, :].argmax(dim=-1)
    raise ValueError(f"Unsupported target_mode: {target_mode}")
