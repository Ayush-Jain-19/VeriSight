"""Evaluation runner for deepfake detection models on test datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import AudioDataset, MultimodalDataset, VideoDataset
from src.evaluation.metrics import (
    compute_binary_metrics,
    compute_bootstrap_confidence_intervals,
    compute_metrics_with_abstention,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model_on_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str | torch.device = "cpu",
    tau: Optional[float] = None,
) -> Dict[str, Any]:
    """Run evaluation on a DataLoader and compute complete benchmark metrics.

    Returns dictionary containing standard binary metrics, bootstrap CIs, and abstention metrics.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    all_probs = []
    all_labels = []

    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            inputs, labels = batch[0], batch[1]
        elif isinstance(batch, dict):
            inputs = batch["inputs"] if "inputs" in batch else batch["faces"]
            labels = batch["overall_label"]
        else:
            raise ValueError("Unsupported batch format")

        inputs = inputs.to(dev)
        outputs = model(inputs)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    # Standard metrics
    metrics = compute_binary_metrics(y_true, y_prob)

    # Bootstrap CIs (F1 & ROC-AUC)
    if len(np.unique(y_true)) > 1:
        ci = compute_bootstrap_confidence_intervals(y_true, y_prob)
        metrics.update(ci)

    # Calibrated abstention metrics if tau is provided
    if tau is not None:
        abstention_info = compute_metrics_with_abstention(y_true, y_prob, tau=tau)
        metrics["abstention"] = abstention_info

    metrics["probabilities"] = y_prob.tolist()
    metrics["ground_truth"] = y_true.tolist()

    return metrics
