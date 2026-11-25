"""Evaluation metrics, confidence intervals, and confusion matrices for deepfake detection."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_binary_metrics(
    y_true: np.ndarray | List[float],
    y_prob: np.ndarray | List[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute standard deepfake detection benchmark metrics.

    Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)

    # Predictions at threshold
    y_pred = (y_prob_arr >= threshold).astype(np.int32)

    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    # ROC-AUC and PR-AUC require at least 2 distinct classes
    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) > 1:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).tolist()
    # tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel().tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total_samples": int(len(y_true_arr)),
    }


def compute_metrics_with_abstention(
    y_true: np.ndarray | List[float],
    y_prob: np.ndarray | List[float],
    tau: float = 0.65,
) -> Dict[str, Any]:
    """Evaluate predictions with calibrated abstention gate.

    Rule: If max(P(real), P(fake)) < tau -> UNCERTAIN (abstain).
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)

    p_real = 1.0 - y_prob_arr
    p_fake = y_prob_arr
    confidence = np.maximum(p_real, p_fake)

    retained_mask = confidence >= tau
    abstain_mask = ~retained_mask

    n_total = len(y_true_arr)
    n_retained = int(np.sum(retained_mask))
    n_abstained = int(np.sum(abstain_mask))

    abstention_rate = float(n_abstained / n_total) if n_total > 0 else 0.0
    coverage = float(n_retained / n_total) if n_total > 0 else 0.0

    if n_retained > 0:
        retained_metrics = compute_binary_metrics(y_true_arr[retained_mask], y_prob_arr[retained_mask])
    else:
        retained_metrics = {}

    if n_abstained > 0:
        abstained_metrics = compute_binary_metrics(y_true_arr[abstain_mask], y_prob_arr[abstain_mask])
    else:
        abstained_metrics = {}

    return {
        "tau": tau,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "n_retained": n_retained,
        "n_abstained": n_abstained,
        "total_samples": n_total,
        "retained_metrics": retained_metrics,
        "abstained_metrics": abstained_metrics,
    }


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray | List[float],
    y_prob: np.ndarray | List[float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute empirical bootstrap confidence intervals for F1 and ROC-AUC."""
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)

    rng = np.random.RandomState(seed)
    n_samples = len(y_true_arr)

    bootstrapped_f1 = []
    bootstrapped_auc = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        sample_true = y_true_arr[indices]
        sample_prob = y_prob_arr[indices]

        if len(np.unique(sample_true)) < 2:
            continue

        sample_pred = (sample_prob >= 0.5).astype(np.int32)
        bootstrapped_f1.append(f1_score(sample_true, sample_pred, zero_division=0))
        bootstrapped_auc.append(roc_auc_score(sample_true, sample_prob))

    alpha = (1.0 - confidence_level) / 2.0
    f1_ci = (
        float(np.percentile(bootstrapped_f1, alpha * 100)),
        float(np.percentile(bootstrapped_f1, (1.0 - alpha) * 100)),
    )
    auc_ci = (
        float(np.percentile(bootstrapped_auc, alpha * 100)),
        float(np.percentile(bootstrapped_auc, (1.0 - alpha) * 100)),
    )

    return {"f1_ci": f1_ci, "roc_auc_ci": auc_ci}
