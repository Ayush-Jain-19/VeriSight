"""Unit tests for selective prediction and calibrated abstention threshold logic."""

import numpy as np
import pytest
from src.evaluation.calibration import (
    calibrate_all_paths,
    calibrate_path_threshold,
    compute_coverage_error_curve,
)


def test_coverage_error_curve_properties():
    """Verify monotonic relationship: higher tau -> lower coverage -> higher abstention."""
    rng = np.random.RandomState(42)
    labels = rng.randint(0, 2, size=100)
    # Generate noisy probabilities
    probs = np.clip(labels * 0.8 + rng.normal(0, 0.2, size=100), 0.05, 0.95)

    curve_df = compute_coverage_error_curve(probs, labels, step=0.05)

    # 1. Coverage should be monotonically decreasing or non-increasing with tau
    coverages = curve_df["coverage"].values
    assert np.all(np.diff(coverages) <= 0.0001), "Coverage must not increase as threshold tau increases"

    # 2. Abstention rate should be monotonically increasing or non-decreasing with tau
    abstentions = curve_df["abstention_rate"].values
    assert np.all(np.diff(abstentions) >= -0.0001), "Abstention must not decrease as threshold tau increases"

    # 3. Check boundary conditions
    row_at_05 = curve_df[curve_df["threshold"] == 0.50].iloc[0]
    assert row_at_05["coverage"] == 1.0  # At 0.50, all samples are covered
    assert row_at_05["abstention_rate"] == 0.0


def test_calibrate_path_threshold():
    """Verify threshold calibration finds optimal tau near target coverage."""
    probs = np.array([0.95, 0.90, 0.85, 0.70, 0.55, 0.10, 0.05, 0.15, 0.45, 0.30])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])

    tau, curve = calibrate_path_threshold(probs, labels, target_coverage=0.80)
    assert 0.50 <= tau <= 0.99
    assert isinstance(tau, float)


def test_calibrate_all_paths():
    """Verify independent threshold calibration for fusion, video, and audio paths."""
    val_labels = [1, 0, 1, 0, 1, 0, 1, 0]
    fusion_probs = [0.95, 0.05, 0.90, 0.10, 0.85, 0.15, 0.80, 0.20]
    video_probs = [0.85, 0.15, 0.75, 0.25, 0.70, 0.30, 0.65, 0.35]
    audio_probs = [0.90, 0.10, 0.80, 0.20, 0.60, 0.40, 0.70, 0.30]

    thresholds = calibrate_all_paths(
        fusion_val_probs=fusion_probs,
        video_val_probs=video_probs,
        audio_val_probs=audio_probs,
        val_labels=val_labels,
        target_coverage=0.85,
    )

    assert "tau_fusion" in thresholds
    assert "tau_video" in thresholds
    assert "tau_audio" in thresholds
    assert 0.50 <= thresholds["tau_fusion"] <= 1.0
    assert 0.50 <= thresholds["tau_video"] <= 1.0
    assert 0.50 <= thresholds["tau_audio"] <= 1.0
