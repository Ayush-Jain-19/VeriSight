"""Shared pytest fixtures: synthetic video, audio, dummy tensors, and mocked face detector."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytest
import soundfile as sf
import torch


class MockFaceDetector:
    """Deterministic mock face detector for unit tests."""

    def __init__(self, image_size: int = 224, should_detect: bool = True):
        self.image_size = image_size
        self.should_detect = should_detect

    def __call__(self, img: Image.Image) -> Image.Image | None:
        if not self.should_detect:
            return None
        # Return a deterministic synthetic cropped face image (RGB 224x224)
        arr = np.full((self.image_size, self.image_size, 3), 128, dtype=np.uint8)
        # Draw some arbitrary features to emulate a face crop
        cv2.circle(arr, (112, 112), 60, (200, 180, 160), -1)
        cv2.circle(arr, (90, 95), 10, (50, 50, 50), -1)
        cv2.circle(arr, (134, 95), 10, (50, 50, 50), -1)
        return Image.fromarray(arr)


@pytest.fixture
def mock_face_detector():
    """Returns a mock face detector that deterministically produces 224x224 crops."""
    return MockFaceDetector(image_size=224, should_detect=True)


@pytest.fixture
def failing_face_detector():
    """Returns a mock detector that simulates detection failure (no faces found)."""
    return MockFaceDetector(image_size=224, should_detect=False)


@pytest.fixture
def synthetic_audio_wav(tmp_path) -> Path:
    """Generate a valid 2-second mono 16 kHz sine wave WAV file."""
    wav_path = tmp_path / "test_sine.wav"
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz sine wave
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(wav_path), sine_wave.astype(np.float32), sr)
    return wav_path


@pytest.fixture
def synthetic_video_mp4(tmp_path) -> Path:
    """Generate a small 2-second 30fps MP4 video file using OpenCV."""
    video_path = tmp_path / "test_clip.mp4"
    width, height = 320, 240
    fps = 30
    duration_sec = 2
    total_frames = fps * duration_sec

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Simple dynamic pattern across frames
        cv2.circle(frame, (160 + int(20 * np.sin(i / 5)), 120), 40, (0, 255, 0), -1)
        out.write(frame)

    out.release()
    return video_path
