from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .losses import select_target_mode, sequence_classification_loss
from .metrics import classification_metrics
from .trainer import prefix_keys, select_predictions_for_metrics
from .utils import ensure_dir


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temperature = float(temperature)
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)


class DistillationTrainer:
    def __init__(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        loaders: dict[str, torch.utils.data.DataLoader],
        config: dict[str, Any],
        device: torch.device,
        run_name: str,
        results_dir: str | Path = "results",
        num_classes: int | None = None,
    ) -> None:
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.loaders = loaders
        self.config = config
        self.device = device
        self.run_name = run_name
        self.results_dir = ensure_dir(results_dir)
        self.training_cfg = config.get("training", {})
        self.target_mode = str(self.training_cfg.get("target_mode", "last"))
        distill_cfg = self.training_cfg.get("distillation", {})
        self.temperature = float(distill_cfg.get("temperature", 4.0))
        self.kd_weight = float(distill_cfg.get("kd_weight", 0.5))
        self.num_classes = num_classes
        self.checkpoint_path = self.results_dir / f"{run_name}_best.pt"
        self.epoch_log_path = self.results_dir / f"{run_name}_epoch_log.csv"

    def fit(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
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
                        "model_state_dict": self.student.state_dict(),
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
            self.student.load_state_dict(checkpoint["model_state_dict"])

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
        self.student.train()
        self.teacher.eval()
        grad_clip = float(self.training_cfg.get("grad_clip", 1.0))
        spike_reg_lambda = float(self.training_cfg.get("spike_reg_lambda", 0.0))
        losses: list[float] = []
        ce_losses: list[float] = []
        kd_losses: list[float] = []
        spike_rates: list[float] = []
        y_true = []
        y_pred = []

        for batch in tqdm(self.loaders["train"], desc=f"{self.run_name}:train", leave=False):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            optimizer.zero_grad(set_to_none=True)
            student_outputs = self.student(x)
            with torch.no_grad():
                teacher_outputs = self.teacher(x)
            ce_loss, details = sequence_classification_loss(
                student_outputs,
                y,
                spike_reg_lambda,
                self.target_mode,
            )
            student_logits, _ = select_target_mode(student_outputs["logits"], y, self.target_mode)
            teacher_logits, _ = select_target_mode(teacher_outputs["logits"], y, self.target_mode)
            kd_loss = distillation_kl_loss(student_logits, teacher_logits, self.temperature)
            loss = ce_loss + self.kd_weight * kd_loss
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), grad_clip)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            ce_losses.append(details["ce_loss"])
            kd_losses.append(float(kd_loss.detach().cpu()))
            if "spike_rate" in student_outputs:
                spike_rates.append(float(student_outputs["spike_rate"].detach().cpu()))
            true_batch, pred_batch = select_predictions_for_metrics(student_outputs["logits"], y, self.target_mode)
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
                "kd_loss": float(np.mean(kd_losses)),
                "spike_rate": float(np.mean(spike_rates)) if spike_rates else 0.0,
            }
        )
        return metrics

    @torch.no_grad()
    def evaluate(self, split: str) -> dict[str, Any]:
        self.student.eval()
        spike_reg_lambda = float(self.training_cfg.get("spike_reg_lambda", 0.0))
        losses: list[float] = []
        ce_losses: list[float] = []
        spike_rates: list[float] = []
        y_true = []
        y_pred = []

        for batch in self.loaders[split]:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            outputs = self.student(x)
            loss, details = sequence_classification_loss(outputs, y, spike_reg_lambda, self.target_mode)
            losses.append(float(loss.detach().cpu()))
            ce_losses.append(details["ce_loss"])
            if "spike_rate" in outputs:
                spike_rates.append(float(outputs["spike_rate"].detach().cpu()))
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
            }
        )
        if split == "test":
            confusion_path = self.results_dir / f"{self.run_name}_confusion_matrix.json"
            confusion_path.write_text(json.dumps(metrics["confusion_matrix"], indent=2), encoding="utf-8")
            metrics["confusion_matrix_path"] = str(confusion_path)
        return metrics

    def _num_classes(self) -> int:
        if self.num_classes is None:
            raise ValueError("DistillationTrainer requires num_classes for metric computation")
        return int(self.num_classes)
