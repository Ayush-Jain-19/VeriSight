"""Unit tests for metric computation, confusion matrix, and bootstrap confidence intervals."""

import numpy as np
import pytest
from src.evaluation.metrics import (
    compute_binary_metrics,
    compute_bootstrap_confidence_intervals,
    compute_metrics_with_abstention,
)


def test_compute_binary_metrics_known():
    """Verify metrics computation against known ground truth."""
    y_true = [1, 1, 0, 0]
    y_prob = [0.9, 0.8, 0.2, 0.1]

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0


def test_compute_metrics_with_abstention():
    """Verify that samples with confidence < tau are counted as abstained."""
    y_true = [1, 1, 0, 0]
    # Two confident (0.95, 0.05) and two uncertain (0.55, 0.52)
    y_prob = [0.95, 0.55, 0.52, 0.05]

    res = compute_metrics_with_abstention(y_true, y_prob, tau=0.80)

    assert res["total_samples"] == 4
    assert res["n_retained"] == 2  # 0.95 and 0.05 have confidence >= 0.80
    assert res["n_abstained"] == 2  # 0.55 and 0.52 have confidence < 0.80
    assert res["coverage"] == 0.50
    assert res["abstention_rate"] == 0.50
    assert res["retained_metrics"]["accuracy"] == 1.0


def test_bootstrap_confidence_intervals():
    """Verify that bootstrap confidence intervals produce valid bounds."""
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_prob = [0.9, 0.85, 0.8, 0.75, 0.2, 0.15, 0.1, 0.05]

    ci = compute_bootstrap_confidence_intervals(y_true, y_prob, n_bootstraps=100, confidence_level=0.95)

    assert "f1_ci" in ci
    assert "roc_auc_ci" in ci
    low_f1, high_f1 = ci["f1_ci"]
    assert 0.0 <= low_f1 <= high_f1 <= 1.0
    low_auc, high_auc = ci["roc_auc_ci"]
    assert 0.0 <= low_auc <= high_auc <= 1.0
