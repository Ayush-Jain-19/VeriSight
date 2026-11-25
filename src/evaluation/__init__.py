"""Evaluation, calibration, and benchmarking package."""

from src.evaluation.calibration import calibrate_all_paths, calibrate_path_threshold
from src.evaluation.compare import generate_markdown_comparison_report
from src.evaluation.evaluate import evaluate_model_on_dataset
from src.evaluation.metrics import (
    compute_binary_metrics,
    compute_bootstrap_confidence_intervals,
    compute_metrics_with_abstention,
)

__all__ = [
    "compute_binary_metrics",
    "compute_metrics_with_abstention",
    "compute_bootstrap_confidence_intervals",
    "calibrate_path_threshold",
    "calibrate_all_paths",
    "evaluate_model_on_dataset",
    "generate_markdown_comparison_report",
]
