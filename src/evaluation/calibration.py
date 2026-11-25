"""Threshold calibration for selective prediction and calibrated abstention (UNCERTAIN verdict).

Abstention Rule:
    Confidence = max(P(real), P(fake))
    If Confidence < tau -> Abstain (Verdict: UNCERTAIN)
    Else -> Predict argmax(P(real), P(fake))

Properties:
- Increasing tau WIDENS the UNCERTAIN region (more samples abstained, lower coverage).
- Decreasing tau NARROWS the UNCERTAIN region (fewer samples abstained, higher coverage).
- At tau = 0.50, abstention rate is 0% (full coverage).
- At tau = 1.00, abstention rate is 100% (zero coverage).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def compute_coverage_error_curve(
    val_probs: np.ndarray | List[float],
    val_labels: np.ndarray | List[float],
    step: float = 0.01,
) -> pd.DataFrame:
    """Compute empirical coverage vs. error rate across threshold range [0.50, 0.99]."""
    probs = np.asarray(val_probs, dtype=np.float32)
    labels = np.asarray(val_labels, dtype=np.int32)

    confidence = np.maximum(probs, 1.0 - probs)
    binary_preds = (probs >= 0.5).astype(np.int32)
    errors = (binary_preds != labels).astype(np.int32)

    thresholds = np.arange(0.50, 1.00, step)
    records = []
    n_total = len(labels)

    for tau in thresholds:
        retained = confidence >= tau
        n_retained = int(np.sum(retained))
        coverage = n_retained / n_total if n_total > 0 else 0.0
        abstention_rate = 1.0 - coverage

        if n_retained > 0:
            error_rate = float(np.sum(errors[retained])) / n_retained
            accuracy = 1.0 - error_rate
        else:
            error_rate = 0.0
            accuracy = 1.0

        records.append(
            {
                "threshold": round(float(tau), 4),
                "coverage": round(coverage, 4),
                "abstention_rate": round(abstention_rate, 4),
                "error_rate": round(error_rate, 4),
                "accuracy": round(accuracy, 4),
                "n_retained": n_retained,
            }
        )

    return pd.DataFrame(records)


def calibrate_path_threshold(
    val_probs: np.ndarray | List[float],
    val_labels: np.ndarray | List[float],
    target_coverage: float = 0.90,
    target_error_rate: Optional[float] = None,
    step: float = 0.01,
) -> Tuple[float, pd.DataFrame]:
    """Select optimal abstention threshold tau on validation data.

    If target_error_rate is provided, finds the highest coverage with error <= target_error_rate.
    Otherwise, selects the tau that yields closest coverage to target_coverage.
    """
    curve_df = compute_coverage_error_curve(val_probs, val_labels, step=step)

    if target_error_rate is not None:
        eligible = curve_df[curve_df["error_rate"] <= target_error_rate]
        if not eligible.empty:
            best_row = eligible.sort_values(by="coverage", ascending=False).iloc[0]
        else:
            best_row = curve_df.sort_values(by="error_rate").iloc[0]
    else:
        # Match closest target coverage
        diff = np.abs(curve_df["coverage"] - target_coverage)
        best_idx = diff.idxmin()
        best_row = curve_df.loc[best_idx]

    chosen_tau = float(best_row["threshold"])
    logger.info(
        f"Calibrated tau={chosen_tau:.2f} (Coverage: {best_row['coverage']:.3f}, Error Rate: {best_row['error_rate']:.3f})"
    )
    return chosen_tau, curve_df


def calibrate_all_paths(
    fusion_val_probs: np.ndarray | List[float],
    video_val_probs: np.ndarray | List[float],
    audio_val_probs: np.ndarray | List[float],
    val_labels: np.ndarray | List[float],
    target_coverage: float = 0.90,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Independently calibrate separate abstention thresholds for fusion, video, and audio paths."""
    tau_fusion, curve_fusion = calibrate_path_threshold(
        fusion_val_probs, val_labels, target_coverage=target_coverage
    )
    tau_video, curve_video = calibrate_path_threshold(
        video_val_probs, val_labels, target_coverage=target_coverage
    )
    tau_audio, curve_audio = calibrate_path_threshold(
        audio_val_probs, val_labels, target_coverage=target_coverage
    )

    thresholds_dict = {
        "tau_fusion": tau_fusion,
        "tau_video": tau_video,
        "tau_audio": tau_audio,
        "target_coverage": target_coverage,
        "calibration_metadata": {
            "fusion": curve_fusion[curve_fusion["threshold"] == tau_fusion].iloc[0].to_dict(),
            "video": curve_video[curve_video["threshold"] == tau_video].iloc[0].to_dict(),
            "audio": curve_audio[curve_audio["threshold"] == tau_audio].iloc[0].to_dict(),
        },
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            yaml.dump(thresholds_dict, f, default_flow_style=False)
        logger.info(f"Thresholds saved to {out_p}")

    return thresholds_dict
