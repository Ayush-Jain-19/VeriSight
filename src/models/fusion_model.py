"""Late fusion deepfake detection MLP combining unimodal scores and embeddings."""

from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn


class FusionDeepfakeDetector(nn.Module):
    """Multimodal late-fusion model.

    Combines unimodal detection scores (scalar probabilities) and unimodal
    embeddings (128-d vectors from penultimate layers) into a 258-d feature
    vector, classifying the multimodal instance as REAL (0) or FAKE (1).
    """

    def __init__(
        self,
        input_dim: int = 258,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 64]

        self.input_dim = input_dim

        layers: List[nn.Module] = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim

        # Output single logit for BCEWithLogitsLoss
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def construct_fusion_input(
        video_score: torch.Tensor,
        audio_score: torch.Tensor,
        video_embedding: torch.Tensor,
        audio_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate unimodal scores and embeddings into unified fusion tensor.

        Shapes:
            video_score: (Batch, 1) or (Batch,)
            audio_score: (Batch, 1) or (Batch,)
            video_embedding: (Batch, 128)
            audio_embedding: (Batch, 128)
        Output:
            (Batch, 258)
        """
        if video_score.dim() == 1:
            video_score = video_score.unsqueeze(1)
        if audio_score.dim() == 1:
            audio_score = audio_score.unsqueeze(1)

        return torch.cat([video_score, audio_score, video_embedding, audio_embedding], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input:
            x: (Batch, 258)
        Returns:
            logits: (Batch,)
        """
        return self.mlp(x).squeeze(-1)
