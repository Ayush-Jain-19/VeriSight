"""Unit tests for Video, Audio, and Fusion model forward passes and tensor shapes."""

import pytest
import torch

from src.models.audio_model import AudioDeepfakeDetector
from src.models.fusion_model import FusionDeepfakeDetector
from src.models.video_model import VideoDeepfakeDetector


def test_video_model_forward():
    """Verify VideoDeepfakeDetector forward pass with 4D and 5D inputs on CPU."""
    # pretrained=False for fast offline unit test execution
    model = VideoDeepfakeDetector(
        backbone_name="efficientnet_b0",
        pretrained=False,
        freeze_backbone=False,
        embedding_dim=128,
    )
    model.eval()

    # 1. Test single frame per sample (4D: Batch, Channels, Height, Width)
    dummy_4d = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits_4d, emb_4d = model(dummy_4d)

    assert logits_4d.shape == (2,)
    assert emb_4d.shape == (2, 128)

    # 2. Test multi-frame video input (5D: Batch, NumFrames, Channels, Height, Width)
    dummy_5d = torch.randn(2, 4, 3, 224, 224)
    with torch.no_grad():
        logits_5d, emb_5d = model(dummy_5d)

    assert logits_5d.shape == (2,)
    assert emb_5d.shape == (2, 128)


def test_audio_model_forward():
    """Verify AudioDeepfakeDetector forward pass with raw waveform input on CPU."""
    model = AudioDeepfakeDetector(
        backbone_name="mock",
        freeze_backbone=False,
        embedding_dim=128,
        use_mock_backbone=True,
    )
    model.eval()

    # (Batch=2, Samples=64000) representing 4 seconds at 16kHz
    dummy_audio = torch.randn(2, 64000)
    with torch.no_grad():
        logits, emb = model(dummy_audio)

    assert logits.shape == (2,)
    assert emb.shape == (2, 128)


def test_fusion_model_forward():
    """Verify FusionDeepfakeDetector input construction and forward pass."""
    model = FusionDeepfakeDetector(input_dim=258, hidden_dims=[64, 32], dropout=0.1)
    model.eval()

    # Construct inputs
    v_scores = torch.tensor([0.8, 0.2])
    a_scores = torch.tensor([0.9, 0.1])
    v_emb = torch.randn(2, 128)
    a_emb = torch.randn(2, 128)

    fusion_input = FusionDeepfakeDetector.construct_fusion_input(v_scores, a_scores, v_emb, a_emb)
    assert fusion_input.shape == (2, 258)

    with torch.no_grad():
        fused_logits = model(fusion_input)

    assert fused_logits.shape == (2,)
