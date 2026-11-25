"""Unit tests for video and audio preprocessing pipelines with mocked face detection."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from src.data.audio_preprocessing import load_waveform
from src.data.dataset import AudioDataset, MultimodalDataset, VideoDataset
from src.data.video_preprocessing import detect_and_crop_faces, extract_frames


def test_extract_frames(synthetic_video_mp4):
    """Verify uniform frame sampling returns PIL images."""
    frames = extract_frames(synthetic_video_mp4, max_frames=8)
    assert len(frames) == 8
    assert frames[0].size == (320, 240)


def test_detect_and_crop_faces_mocked(synthetic_video_mp4, mock_face_detector):
    """Verify that mocked detector produces normalized tensors of shape (3, 224, 224)."""
    frames = extract_frames(synthetic_video_mp4, max_frames=4)
    tensors, valid_indices = detect_and_crop_faces(frames, detector=mock_face_detector, image_size=224)

    assert len(tensors) == 4
    assert valid_indices == [0, 1, 2, 3]
    assert tensors[0].shape == (3, 224, 224)
    assert isinstance(tensors[0], torch.Tensor)


def test_detect_and_crop_faces_failing(synthetic_video_mp4, failing_face_detector):
    """Verify graceful handling when face detector detects zero faces."""
    frames = extract_frames(synthetic_video_mp4, max_frames=4)
    tensors, valid_indices = detect_and_crop_faces(frames, detector=failing_face_detector, image_size=224)

    assert len(tensors) == 0
    assert len(valid_indices) == 0


def test_load_waveform_padding_and_truncation(synthetic_audio_wav, tmp_path):
    """Verify that 2s audio is padded to 4.0s (64,000 samples at 16kHz)."""
    waveform = load_waveform(
        synthetic_audio_wav,
        target_sample_rate=16000,
        max_duration_seconds=4.0,
        min_duration_seconds=0.5,
    )

    assert waveform is not None
    assert waveform.shape == (64000,)
    assert isinstance(waveform, torch.Tensor)

    # Test longer audio (6 seconds) -> should truncate to 64,000 samples
    long_wav = tmp_path / "long.wav"
    sr = 16000
    long_data = np.zeros(sr * 6, dtype=np.float32)
    sf.write(str(long_wav), long_data, sr)

    truncated = load_waveform(long_wav, target_sample_rate=16000, max_duration_seconds=4.0)
    assert truncated.shape == (64000,)


def test_load_waveform_too_short(tmp_path):
    """Verify that audio shorter than min_duration returns None."""
    short_wav = tmp_path / "too_short.wav"
    sr = 16000
    short_data = np.zeros(int(sr * 0.2), dtype=np.float32)  # 0.2s < 0.5s
    sf.write(str(short_wav), short_data, sr)

    waveform = load_waveform(short_wav, target_sample_rate=16000, min_duration_seconds=0.5)
    assert waveform is None


def test_datasets_with_synthetic_fixtures(synthetic_video_mp4, synthetic_audio_wav, mock_face_detector):
    """Verify VideoDataset, AudioDataset, and MultimodalDataset return expected shapes and labels."""
    df = pd.DataFrame(
        [
            {
                "video_path": str(synthetic_video_mp4),
                "identity_id": "id001",
                "category": "RealVideo-RealAudio",
                "video_label": "REAL",
                "audio_label": "REAL",
                "overall_label": "REAL",
            },
            {
                "video_path": str(synthetic_audio_wav),  # AudioDataset supports wav directly
                "identity_id": "id002",
                "category": "FakeVideo-FakeAudio",
                "video_label": "FAKE",
                "audio_label": "FAKE",
                "overall_label": "FAKE",
            },
        ]
    )

    # 1. VideoDataset
    video_ds = VideoDataset(df.iloc[:1], max_frames=4, face_detector=mock_face_detector)
    faces, v_label = video_ds[0]
    assert faces.shape == (4, 3, 224, 224)
    assert v_label.item() == 0.0  # REAL

    # 2. AudioDataset
    audio_ds = AudioDataset(df.iloc[1:2], sample_rate=16000, max_duration_seconds=4.0)
    wav, a_label = audio_ds[0]
    assert wav.shape == (64000,)
    assert a_label.item() == 1.0  # FAKE
