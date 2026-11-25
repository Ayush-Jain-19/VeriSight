"""Comparative evaluation suite comparing unimodal baselines against multimodal fusion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def generate_markdown_comparison_report(
    video_metrics: Dict[str, Any],
    audio_metrics: Dict[str, Any],
    fusion_metrics: Dict[str, Any],
    output_path: Optional[str | Path] = None,
) -> str:
    """Generate a comprehensive Markdown comparison report with tables and confusion matrices."""

    def format_ci(ci_tuple):
        if not ci_tuple:
            return "N/A"
        return f"[{ci_tuple[0]:.3f}, {ci_tuple[1]:.3f}]"

    v_f1_ci = format_ci(video_metrics.get("f1_ci"))
    a_f1_ci = format_ci(audio_metrics.get("f1_ci"))
    f_f1_ci = format_ci(fusion_metrics.get("f1_ci"))

    v_auc_ci = format_ci(video_metrics.get("roc_auc_ci"))
    a_auc_ci = format_ci(audio_metrics.get("roc_auc_ci"))
    f_auc_ci = format_ci(fusion_metrics.get("roc_auc_ci"))

    # Extract abstention info
    v_abs = video_metrics.get("abstention", {})
    a_abs = audio_metrics.get("abstention", {})
    f_abs = fusion_metrics.get("abstention", {})

    report = f"""# Multimodal Deepfake Detection — Comparative Evaluation Report

## 1. Executive Summary & Benchmark Table

| Metric | Video Baseline (EfficientNet-B0) | Audio Baseline (wav2vec2-base) | Multimodal Late-Fusion |
| :--- | :---: | :---: | :---: |
| **Accuracy** | {video_metrics.get('accuracy', 0.0):.4f} | {audio_metrics.get('accuracy', 0.0):.4f} | **{fusion_metrics.get('accuracy', 0.0):.4f}** |
| **Precision** | {video_metrics.get('precision', 0.0):.4f} | {audio_metrics.get('precision', 0.0):.4f} | **{fusion_metrics.get('precision', 0.0):.4f}** |
| **Recall** | {video_metrics.get('recall', 0.0):.4f} | {audio_metrics.get('recall', 0.0):.4f} | **{fusion_metrics.get('recall', 0.0):.4f}** |
| **F1-Score** | {video_metrics.get('f1_score', 0.0):.4f} | {audio_metrics.get('f1_score', 0.0):.4f} | **{fusion_metrics.get('f1_score', 0.0):.4f}** |
| **F1 95% Bootstrap CI** | {v_f1_ci} | {a_f1_ci} | **{f_f1_ci}** |
| **ROC-AUC** | {video_metrics.get('roc_auc', 0.0):.4f} | {audio_metrics.get('roc_auc', 0.0):.4f} | **{fusion_metrics.get('roc_auc', 0.0):.4f}** |
| **ROC-AUC 95% CI** | {v_auc_ci} | {a_auc_ci} | **{f_auc_ci}** |
| **PR-AUC** | {video_metrics.get('pr_auc', 0.0):.4f} | {audio_metrics.get('pr_auc', 0.0):.4f} | **{fusion_metrics.get('pr_auc', 0.0):.4f}** |
| **Calibrated Threshold (τ)** | {v_abs.get('tau', 'N/A')} | {a_abs.get('tau', 'N/A')} | **{f_abs.get('tau', 'N/A')}** |
| **Coverage** | {v_abs.get('coverage', 1.0):.2%} | {a_abs.get('coverage', 1.0):.2%} | **{f_abs.get('coverage', 1.0):.2%}** |
| **Abstention Rate** | {v_abs.get('abstention_rate', 0.0):.2%} | {a_abs.get('abstention_rate', 0.0):.2%} | **{f_abs.get('abstention_rate', 0.0):.2%}** |

---

## 2. Confusion Matrices (Standard 0.50 Decision Threshold)

### Video Baseline
- **True Negatives (Real detected as Real):** {video_metrics.get('tn', 0)}
- **False Positives (Real detected as Fake):** {video_metrics.get('fp', 0)}
- **False Negatives (Fake detected as Real):** {video_metrics.get('fn', 0)}
- **True Positives (Fake detected as Fake):** {video_metrics.get('tp', 0)}

### Audio Baseline
- **True Negatives (Real detected as Real):** {audio_metrics.get('tn', 0)}
- **False Positives (Real detected as Fake):** {audio_metrics.get('fp', 0)}
- **False Negatives (Fake detected as Real):** {audio_metrics.get('fn', 0)}
- **True Positives (Fake detected as Fake):** {audio_metrics.get('tp', 0)}

### Multimodal Late-Fusion
- **True Negatives (Real detected as Real):** {fusion_metrics.get('tn', 0)}
- **False Positives (Real detected as Fake):** {fusion_metrics.get('fp', 0)}
- **False Negatives (Fake detected as Real):** {fusion_metrics.get('fn', 0)}
- **True Positives (Fake detected as Fake):** {fusion_metrics.get('tp', 0)}

---

## 3. Calibrated Abstention (UNCERTAIN Verdicts)

When applying the selective classification rule:
`max(P(real), P(fake)) < τ` -> **UNCERTAIN**

- **Fusion Model** achieved {f_abs.get('coverage', 1.0):.1%} coverage on held-out test data with an abstention rate of {f_abs.get('abstention_rate', 0.0):.1%}.
- Retained subset accuracy: {f_abs.get('retained_metrics', {}).get('accuracy', fusion_metrics.get('accuracy', 0.0)):.4f}.
- Abstained subset would-be accuracy: {f_abs.get('abstained_metrics', {}).get('accuracy', 'N/A')}.
"""

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report, encoding="utf-8")
        logger.info(f"Comparison report saved to {out_p}")

    return report
