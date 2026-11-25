"""Unit tests for dataset manifest scanning and path parsing."""

from pathlib import Path
import pytest
from src.data.manifest import parse_video_path, scan_dataset, DEFAULT_CATEGORY_MAPPING


def test_parse_video_path_standard_layout():
    """Test parsing path with standard FakeAVCeleb directory structure."""
    rel_path = Path("RealVideo-RealAudio/African/men/id00076/00109.mp4")
    parsed = parse_video_path(rel_path, DEFAULT_CATEGORY_MAPPING)

    assert parsed is not None
    assert parsed["category"] == "RealVideo-RealAudio"
    assert parsed["abbreviation"] == "RARV"
    assert parsed["ethnicity"] == "African"
    assert parsed["gender"] == "men"
    assert parsed["identity_id"] == "id00076"
    assert parsed["video_label"] == "REAL"
    assert parsed["audio_label"] == "REAL"
    assert parsed["overall_label"] == "REAL"
    assert parsed["filename"] == "00109.mp4"


def test_parse_video_path_farv():
    """Test parsing path for RealVideo-FakeAudio (FARV)."""
    rel_path = Path("RealVideo-FakeAudio/Caucasian/women/id00120/00042_wav2lip.mp4")
    parsed = parse_video_path(rel_path, DEFAULT_CATEGORY_MAPPING)

    assert parsed is not None
    assert parsed["abbreviation"] == "FARV"
    assert parsed["video_label"] == "REAL"
    assert parsed["audio_label"] == "FAKE"
    assert parsed["overall_label"] == "FAKE"
    assert parsed["method"] == "wav2lip"


def test_parse_video_path_rafv():
    """Test parsing path for FakeVideo-RealAudio (RAFV)."""
    rel_path = Path("FakeVideo-RealAudio/Asian (Eastern)/men/id00300/00001_faceswap.mp4")
    parsed = parse_video_path(rel_path, DEFAULT_CATEGORY_MAPPING)

    assert parsed is not None
    assert parsed["abbreviation"] == "RAFV"
    assert parsed["video_label"] == "FAKE"
    assert parsed["audio_label"] == "REAL"
    assert parsed["overall_label"] == "FAKE"
    assert parsed["method"] == "faceswap"


def test_scan_dataset_on_synthetic_tree(tmp_path):
    """Test scanning a small synthetic directory tree on disk."""
    # Create fake directory structure
    rarv_dir = tmp_path / "RealVideo-RealAudio" / "African" / "men" / "id001"
    rarv_dir.mkdir(parents=True)
    (rarv_dir / "0001.mp4").write_text("dummy video")

    fafv_dir = tmp_path / "FakeVideo-FakeAudio" / "Caucasian" / "women" / "id002"
    fafv_dir.mkdir(parents=True)
    (fafv_dir / "0002.mp4").write_text("dummy video")

    df = scan_dataset(tmp_path)
    assert len(df) == 2
    assert set(df["abbreviation"].unique()) == {"RARV", "FAFV"}
    assert set(df["identity_id"].unique()) == {"id001", "id002"}
