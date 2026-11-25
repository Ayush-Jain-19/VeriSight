"""Audio baseline deepfake detector using wav2vec2-base backbone."""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AudioDeepfakeDetector(nn.Module):
    """Audio deepfake detector based on wav2vec2-base self-supervised representations.

    Processes 16 kHz raw waveforms, extracts contextual speech representations,
    generates 128-d embeddings, and produces binary manipulation logits.
    """

    def __init__(
        self,
        backbone_name: str = "facebook/wav2vec2-base",
        freeze_backbone: bool = True,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        use_mock_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.use_mock_backbone = use_mock_backbone

        if use_mock_backbone or backbone_name == "mock":
            # Lightweight 1D Conv front-end for unit testing without downloading 360MB model
            self.backbone = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=160, stride=80),  # Downsamples 64000 -> 800
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=10, stride=5),  # Downsamples 800 -> 160
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            hidden_dim = 32
        else:
            try:
                from transformers import Wav2Vec2Model

                self.backbone = Wav2Vec2Model.from_pretrained(backbone_name)
                hidden_dim = self.backbone.config.hidden_size  # 768 for wav2vec2-base

                if freeze_backbone:
                    for param in self.backbone.parameters():
                        param.requires_grad = False
            except Exception as e:
                logger.warning(
                    f"Could not load '{backbone_name}' from HuggingFace ({e}). Falling back to lightweight mock backbone."
                )
                self.backbone = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=160, stride=80),
                    nn.ReLU(),
                    nn.Conv1d(16, 32, kernel_size=10, stride=5),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                hidden_dim = 32
                self.use_mock_backbone = True

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Input:
            waveform: (Batch, NumSamples) e.g. (B, 64000) for 4s at 16 kHz.
        Returns:
            logits: (Batch,)
            embeddings: (Batch, embedding_dim)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if self.use_mock_backbone or self.backbone_name == "mock":
            # Input needs (Batch, Channels=1, NumSamples)
            x = waveform.unsqueeze(1)
            pooled = self.backbone(x).squeeze(-1)  # (Batch, 32)
        else:
            outputs = self.backbone(waveform)
            # outputs.last_hidden_state: (Batch, SequenceLength, 768)
            pooled = torch.mean(outputs.last_hidden_state, dim=1)  # (Batch, 768)

        embeddings = self.projection(pooled)
        logits = self.classifier(embeddings).squeeze(-1)
        return logits, embeddings
