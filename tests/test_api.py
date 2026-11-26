"""Unit tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient
import pytest

from src.api.main import create_app
from src.inference.pipeline import InferencePipeline
from src.inference import pipeline as serving


@pytest.fixture
def api_client(monkeypatch):
    app = create_app()
    pipe = InferencePipeline()
    # Offline adapter tests, with real decoding and the production pipeline.
    class Face:
        def predict_video_frames(self, frames):
            return dict(score=.1, has_face=True, faces_detected=len(frames),
                        frames_analyzed=len(frames), reason=None, model=serving.FACE)

        def predict_image(self, image):
            return dict(score=0., has_face=True, reason=None)

    monkeypatch.setattr(serving, "load_visual", lambda _: Face())
    def offline_text(text, slot, name):
        return dict(score=None, available=False, word_count=len(text.split()),
                    model=name, reason="Offline text fixture", spans=[])
    monkeypatch.setattr(serving,"analyze_text",offline_text)
    app.state.pipeline = pipe
    with TestClient(app) as client:
        yield client


def test_health_endpoint(api_client):
    data = api_client.get("/api/health").json()
    assert data["status"] == "ok"
    assert not data["models_loaded"]
    assert not data["models_ready"]
    assert not data["thresholds_calibrated"]
    assert not data["video_model"]["checkpoint_loaded"]


def test_predict_endpoint_valid_video(api_client, synthetic_video_mp4):
    with synthetic_video_mp4.open("rb") as f:
        response = api_client.post("/api/predict", files={"file": ("clip.mp4", f)})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "UNCERTAIN"
    assert data["confidence"] is None
    assert data["video"]["score"] == .1
    assert data["audio"]["score"] is None
    assert data["fusion"]["score"] is None
    assert data["timing"]["total_pipeline_sec"] > 0


def test_predict_endpoint_strict_mode_rejects_missing_stream(api_client, synthetic_video_mp4):
    with synthetic_video_mp4.open("rb") as f:
        response = api_client.post("/api/predict?strict=true", files={"file": ("clip.mp4", f)})
    assert response.status_code == 503


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
    assert data["audio"]["available"] is False
    assert data["audio"]["score"] is None


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



def test_subtitle_upload_and_strategy(api_client, synthetic_video_mp4):
    with synthetic_video_mp4.open("rb") as f:
        response=api_client.post("/api/predict/video-text?strategy=confidence_weighted_fusion",
                  files={"file":("clip.mp4",f),"subtitles":("ref.srt",b"example subtitle")})
    assert response.status_code==200
    result=response.json()
    assert result["fusion"]["method"]=="confidence_weighted_fusion"
    assert result["cross_modal"]["extracted_text_detected"]
    assert result["cross_modal"]["mismatch"] is None
