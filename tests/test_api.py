"""Unit tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient
import pytest

from src.api.main import create_app
from src.inference.pipeline import InferencePipeline
from src.models.audio_model import AudioDeepfakeDetector
from src.models.fusion_model import FusionDeepfakeDetector
from src.models.video_model import VideoDeepfakeDetector


@pytest.fixture
def api_client(mock_face_detector):
    """Create a FastAPI test client with lightweight mocked models attached to app state."""
    app = create_app()

    video_model = VideoDeepfakeDetector(backbone_name="efficientnet_b0", pretrained=False)
    audio_model = AudioDeepfakeDetector(backbone_name="mock", use_mock_backbone=True)
    fusion_model = FusionDeepfakeDetector(input_dim=258, hidden_dims=[32])

    pipeline = InferencePipeline(
        config=app.state.config,
        video_model=video_model,
        audio_model=audio_model,
        fusion_model=fusion_model,
        face_detector=mock_face_detector,
    )
    app.state.pipeline = pipeline

    with TestClient(app) as client:
        yield client


def test_health_endpoint(api_client):
    """Verify /api/health returns 200 and expected diagnostic and model readiness fields."""
    response = api_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "ffmpeg_available" in data
    assert data["models_loaded"] is True
    # Readiness verification for uncalibrated / mock models
    assert data["models_ready"] is False
    assert data["thresholds_calibrated"] is False
    assert data["video_model"]["loaded"] is True
    assert data["video_model"]["checkpoint_loaded"] is False
    assert data["video_model"]["is_mock"] is True
    assert data["audio_model"]["loaded"] is True
    assert data["audio_model"]["is_mock"] is True
    assert data["fusion_model"]["loaded"] is True
    assert data["fusion_model"]["checkpoint_loaded"] is False


def test_predict_endpoint_valid_video(api_client, synthetic_video_mp4):
    """Verify /api/predict accepts multipart video file and marks unvalidated mock predictions."""
    with open(synthetic_video_mp4, "rb") as f:
        response = api_client.post(
            "/api/predict",
            files={"file": ("test_clip.mp4", f, "video/mp4")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in {"REAL", "FAKE", "UNCERTAIN"}
    assert "video" in data
    assert "audio" in data
    assert "fusion" in data
    assert "timing" in data
    assert data["timing"]["total_pipeline_sec"] > 0
    # Verification that mock predictions are never presented as real
    assert data["is_mock"] is True
    assert data["models_ready"] is False
    assert any("[UNVALIDATED]" in w for w in data["warnings"])
    assert data["video"]["checkpoint_loaded"] is False
    assert data["video"]["is_mock"] is True


def test_predict_endpoint_strict_mode_rejects_mock(api_client, synthetic_video_mp4):
    """Verify /api/predict?strict=true rejects unvalidated mock models with 503."""
    with open(synthetic_video_mp4, "rb") as f:
        response = api_client.post(
            "/api/predict?strict=true",
            files={"file": ("test_clip.mp4", f, "video/mp4")},
        )

    assert response.status_code == 503
    assert "Trained checkpoints are not loaded" in response.json()["detail"]


def test_predict_endpoint_invalid_extension(api_client, tmp_path):
    """Verify /api/predict rejects non-video files with 400 Bad Request."""
    dummy_txt = tmp_path / "notes.txt"
    dummy_txt.write_text("not a video")

    with open(dummy_txt, "rb") as f:
        response = api_client.post(
            "/api/predict",
            files={"file": ("notes.txt", f, "text/plain")},
        )

    assert response.status_code == 400
    assert "Unsupported video format" in response.json()["detail"]


def test_predict_text_endpoint(api_client):
    """Verify /api/predict/text analyzes input text and produces linguistic metrics."""
    response = api_client.post(
        "/api/predict/text",
        json={"text": "This is a detailed and carefully articulated sample text designed to verify the linguistic analysis engine."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in {"REAL", "FAKE", "UNCERTAIN"}
    assert data["prediction_path"] == "text_only"
    assert "text" in data
    assert data["text"]["word_count"] > 10
    assert "burstiness_score" in data["text"]


def test_predict_audio_endpoint(api_client, tmp_path):
    """Verify /api/predict/audio accepts audio files."""
    import numpy as np
    import soundfile as sf

    sr = 16000
    audio_data = np.zeros(sr * 2, dtype=np.float32)
    wav_path = tmp_path / "test_sample.wav"
    sf.write(str(wav_path), audio_data, sr)

    with open(wav_path, "rb") as f:
        response = api_client.post(
            "/api/predict/audio",
            files={"file": ("test_sample.wav", f, "audio/wav")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_path"] == "audio_only"
    assert data["audio"]["available"] is True


def test_predict_image_endpoint(api_client, tmp_path):
    """Verify /api/predict/image accepts image files."""
    import cv2
    import numpy as np

    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img_path = tmp_path / "face.jpg"
    cv2.imwrite(str(img_path), img)

    with open(img_path, "rb") as f:
        response = api_client.post(
            "/api/predict/image",
            files={"file": ("face.jpg", f, "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_path"] == "image_only"
    assert "image" in data
    assert data["image"]["available"] is True


def test_predict_video_text_endpoint(api_client, synthetic_video_mp4):
    """Verify /api/predict/video-text analyzes video with textual content."""
    with open(synthetic_video_mp4, "rb") as f:
        response = api_client.post(
            "/api/predict/video-text",
            files={"file": ("test_clip.mp4", f, "video/mp4")},
            data={"text": "Authentic on-screen caption test."},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_path"] == "video_text_fusion"
    assert "cross_modal" in data


def test_predict_endpoint_zero_byte_video_returns_400(api_client, tmp_path):
    """Verify 0-byte video upload returns HTTP 400 Bad Request."""
    empty_video = tmp_path / "empty.mp4"
    empty_video.write_bytes(b"")

    with open(empty_video, "rb") as f:
        response = api_client.post(
            "/api/predict",
            files={"file": ("empty.mp4", f, "video/mp4")},
        )
    assert response.status_code == 400
    assert "empty (0 bytes)" in response.json()["detail"]


def test_predict_audio_endpoint_zero_byte_returns_400(api_client, tmp_path):
    """Verify 0-byte audio upload returns HTTP 400 Bad Request."""
    empty_audio = tmp_path / "empty.wav"
    empty_audio.write_bytes(b"")

    with open(empty_audio, "rb") as f:
        response = api_client.post(
            "/api/predict/audio",
            files={"file": ("empty.wav", f, "audio/wav")},
        )
    assert response.status_code == 400
    assert "empty (0 bytes)" in response.json()["detail"]


def test_predict_image_endpoint_zero_byte_returns_400(api_client, tmp_path):
    """Verify 0-byte image upload returns HTTP 400 Bad Request."""
    empty_img = tmp_path / "empty.jpg"
    empty_img.write_bytes(b"")

    with open(empty_img, "rb") as f:
        response = api_client.post(
            "/api/predict/image",
            files={"file": ("empty.jpg", f, "image/jpeg")},
        )
    assert response.status_code == 400
    assert "empty (0 bytes)" in response.json()["detail"]


def test_predict_image_endpoint_corrupted_file_returns_400(api_client, tmp_path):
    """Verify corrupted non-image file upload returns HTTP 400 instead of 500."""
    corrupt_img = tmp_path / "corrupt.jpg"
    corrupt_img.write_bytes(b"NOT_A_VALID_JPEG_HEADER_RANDOM_DATA_HERE")

    with open(corrupt_img, "rb") as f:
        response = api_client.post(
            "/api/predict/image",
            files={"file": ("corrupt.jpg", f, "image/jpeg")},
        )
    assert response.status_code == 400
    assert "could not be decoded" in response.json()["detail"]


def test_predict_text_endpoint_oversized_returns_400(api_client):
    """Verify text exceeding 50,000 characters is rejected with HTTP 400."""
    huge_text = "word " * 15_000  # ~75,000 characters
    response = api_client.post(
        "/api/predict/text",
        json={"text": huge_text},
    )
    assert response.status_code == 400
    assert "exceeds maximum allowed length" in response.json()["detail"]


def test_predict_text_endpoint_form_and_raw_body(api_client):
    """Verify text endpoint supports multipart form-data and raw body payloads."""
    # Test multipart/form-data
    resp_form = api_client.post(
        "/api/predict/text",
        data={"text": "Authentic statement sent via multipart form data for testing."},
    )
    assert resp_form.status_code == 200
    assert resp_form.json()["prediction_path"] == "text_only"

    # Test raw text body
    resp_raw = api_client.post(
        "/api/predict/text",
        content="Authentic statement sent as raw body payload for testing.",
        headers={"Content-Type": "text/plain"},
    )
    assert resp_raw.status_code == 200
    assert resp_raw.json()["prediction_path"] == "text_only"


