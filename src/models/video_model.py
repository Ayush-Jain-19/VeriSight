"""Video baseline deepfake detector using EfficientNet-B0 backbone."""

from __future__ import annotations

import logging
from typing import Tuple
import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VideoDeepfakeDetector(nn.Module):
    """EfficientNet-B0 based video deepfake detector.

    Extracts spatial face representations, generates 128-d embeddings,
    and produces frame- and video-level manipulation logits.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim

        # num_classes=0 removes final classifier and returns pooled features
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        # Retrieve feature dimension
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        elif hasattr(self.backbone, "head") and hasattr(self.backbone.head, "fc"):
            in_features = self.backbone.head.fc.in_features
        else:
            in_features = 1280  # Standard for EfficientNet-B0

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head to 128-d embedding
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Classification head to single logit (for BCEWithLogitsLoss)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled feature vector from backbone."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Supports:
        - 4D tensor: (Batch, 3, H, W) -> logits: (Batch,), embeddings: (Batch, embedding_dim)
        - 5D tensor: (Batch, NumFrames, 3, H, W) -> mean-pooled video logits: (Batch,), video embeddings: (Batch, embedding_dim)
        """
        if x.dim() == 5:
            batch_size, num_frames, c, h, w = x.shape
            # Flatten batch and frame dimensions
            x_flat = x.view(batch_size * num_frames, c, h, w)
            features = self.extract_features(x_flat)
            embeddings_flat = self.projection(features)
            logits_flat = self.classifier(embeddings_flat).squeeze(-1)

            # Reshape back to (batch_size, num_frames)
            logits_per_frame = logits_flat.view(batch_size, num_frames)
            embeddings_per_frame = embeddings_flat.view(batch_size, num_frames, self.embedding_dim)

            # Mean pooling across frames for video-level score and embedding
            video_logits = torch.mean(logits_per_frame, dim=1)
            video_embeddings = torch.mean(embeddings_per_frame, dim=1)

            return video_logits, video_embeddings

        elif x.dim() == 4:
            features = self.extract_features(x)
            embeddings = self.projection(features)
            logits = self.classifier(embeddings).squeeze(-1)
            return logits, embeddings

        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {x.shape}")
