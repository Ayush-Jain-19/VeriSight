"""Unit tests for the end-to-end CPU inference pipeline with mocked models."""

import pytest
import torch
import torch.nn as nn

from src.config import load_config
from src.inference.pipeline import InferencePipeline
from src.models.audio_model import AudioDeepfakeDetector
from src.models.fusion_model import FusionDeepfakeDetector
from src.models.video_model import VideoDeepfakeDetector


@pytest.fixture
def test_pipeline(mock_face_detector):
    """Create an InferencePipeline with lightweight mock models and mock face detector."""
    config = load_config()

    video_model = VideoDeepfakeDetector(
        backbone_name="efficientnet_b0",
        pretrained=False,
        freeze_backbone=False,
        embedding_dim=128,
    )
    audio_model = AudioDeepfakeDetector(
        backbone_name="mock",
        freeze_backbone=False,
        embedding_dim=128,
        use_mock_backbone=True,
    )
    fusion_model = FusionDeepfakeDetector(input_dim=258, hidden_dims=[32, 16], dropout=0.1)

    return InferencePipeline(
        config=config,
        video_model=video_model,
        audio_model=audio_model,
        fusion_model=fusion_model,
        face_detector=mock_face_detector,
    )


def test_pipeline_both_modalities(test_pipeline, synthetic_video_mp4):
    """Test standard pipeline execution on a video with face and audio."""
    result = test_pipeline.predict(synthetic_video_mp4)

    assert result["prediction"] in {"REAL", "FAKE", "UNCERTAIN"}
    assert 0.50 <= result["confidence"] <= 1.0
    assert result["video"]["available"] is True
    assert result["video"]["frames_analyzed"] > 0
    assert result["video"]["faces_detected"] > 0
    assert "timing" in result
    assert result["timing"]["total_pipeline_sec"] > 0


def test_pipeline_video_only(synthetic_video_mp4, mock_face_detector):
    """Test fallback to video_only when audio extraction fails or audio is missing."""
    config = load_config()
    video_model = VideoDeepfakeDetector(backbone_name="efficientnet_b0", pretrained=False)
    audio_model = AudioDeepfakeDetector(backbone_name="mock", use_mock_backbone=True)
    fusion_model = FusionDeepfakeDetector(input_dim=258, hidden_dims=[32])

    pipeline = InferencePipeline(
        config=config,
        video_model=video_model,
        audio_model=audio_model,
        fusion_model=fusion_model,
        face_detector=mock_face_detector,
    )

    # Force audio unavailable by using non-existent audio
    result = pipeline.predict(synthetic_video_mp4)
    # Since synthetic_video_mp4 generated via OpenCV has no audio stream, path should be video_only!
    assert result["prediction_path"] == "video_only"
    assert result["video"]["available"] is True
    assert result["audio"]["available"] is False
    assert result["video"]["score"] is not None


def test_pipeline_neither_modality(synthetic_video_mp4, failing_face_detector):
    """Test explicit UNCERTAIN when both face detection and audio fail."""
    config = load_config()
    video_model = VideoDeepfakeDetector(backbone_name="efficientnet_b0", pretrained=False)
    audio_model = AudioDeepfakeDetector(backbone_name="mock", use_mock_backbone=True)

    pipeline = InferencePipeline(
        config=config,
        video_model=video_model,
        audio_model=audio_model,
        fusion_model=None,
        face_detector=failing_face_detector,
    )

    result = pipeline.predict(synthetic_video_mp4)
    assert result["prediction"] == "UNCERTAIN"
    assert result["prediction_path"] == "no_modalities"
    assert result["video"]["available"] is False
    assert result["audio"]["available"] is False


def test_pipeline_readiness_and_mock_flags(test_pipeline, synthetic_video_mp4):
    """Verify that unvalidated/mock models are properly flagged in readiness and prediction output."""
    readiness = test_pipeline.get_readiness()
    assert readiness["models_ready"] is False
    assert readiness["video_model"]["checkpoint_loaded"] is False
    assert readiness["audio_model"]["is_mock"] is True
    assert readiness["fusion_model"]["checkpoint_loaded"] is False

    result = test_pipeline.predict(synthetic_video_mp4)
    assert result["is_mock"] is True
    assert result["models_ready"] is False
    assert any("[UNVALIDATED]" in w for w in result["warnings"])


def test_pipeline_strict_checkpoints_raises_on_missing():
    """Verify InferencePipeline(strict_checkpoints=True) raises FileNotFoundError when checkpoints are missing."""
    config = load_config()
    with pytest.raises(FileNotFoundError, match="not found"):
        InferencePipeline(config=config, strict_checkpoints=True)


def test_pipeline_corrupted_checkpoint_handling(tmp_path):
    """Verify corrupted checkpoint handling: records error without strict mode, raises ValueError with strict mode."""
    config = load_config()
    config.paths.checkpoints_dir = str(tmp_path)

    # Write corrupted checkpoint bytes for video
    corrupt_ckpt = tmp_path / "video_efficientnet_b0_best.pt"
    corrupt_ckpt.write_bytes(b"CORRUPTED_NON_PICKLE_DATA_00000000")

    # In non-strict mode: should log error and continue with untrained model
    pipe = InferencePipeline(config=config, strict_checkpoints=False)
    assert pipe.video_checkpoint_loaded is False
    assert pipe.video_error is not None
    assert "Corrupted or invalid checkpoint" in pipe.video_error

    # In strict mode: should raise ValueError
    with pytest.raises(ValueError, match="Invalid video checkpoint"):
        InferencePipeline(config=config, strict_checkpoints=True)
