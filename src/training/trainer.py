"""Generic PyTorch trainer with early stopping, checkpointing, and metric logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer handling optimization, validation, checkpointing, and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device = "cpu",
        checkpoint_dir: Optional[str | Path] = None,
        model_name: str = "model",
        early_stopping_patience: int = 5,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.model_name = model_name
        self.early_stopping_patience = early_stopping_patience

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(
        self,
        dataloader: DataLoader,
        target_label_key: Optional[str] = None,
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                inputs, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch["inputs"] if "inputs" in batch else batch["faces"]
                labels = batch[target_label_key] if target_label_key else batch["overall_label"]
            else:
                raise ValueError("Unsupported batch structure")

            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            # If model returns (logits, embeddings), take logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        return running_loss / total_samples if total_samples > 0 else 0.0

    @torch.no_grad()
    def validate_epoch(
        self,
        dataloader: DataLoader,
        target_label_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run validation loop and calculate loss, accuracy, and F1."""
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                inputs, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch["inputs"] if "inputs" in batch else batch["faces"]
                labels = batch[target_label_key] if target_label_key else batch["overall_label"]
            else:
                raise ValueError("Unsupported batch structure")

            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float()

            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = self.criterion(logits, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())

        val_loss = running_loss / total_samples if total_samples > 0 else 0.0
        val_acc = float(accuracy_score(all_labels, all_preds)) if all_labels else 0.0
        val_f1 = float(f1_score(all_labels, all_preds, zero_division=0)) if all_labels else 0.0

        return {"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        target_label_key: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Fit model across epochs with early stopping on validation loss."""
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting training '{self.model_name}' for up to {epochs} epochs on {self.device}.")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, target_label_key=target_label_key)
            val_metrics = self.validate_epoch(val_loader, target_label_key=target_label_key)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_acc"].append(val_metrics["val_acc"])
            self.history["val_f1"].append(val_metrics["val_f1"])

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc']:.4f} | Val F1: {val_metrics['val_f1']:.4f}"
            )

            # Checkpoint on improvement
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                if self.checkpoint_dir:
                    save_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"Saved new best model checkpoint to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (patience: {self.early_stopping_patience}).")
                    break

        return self.history
