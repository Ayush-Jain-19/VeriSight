"""Automated tests for FakeAVCeleb category to modality label mapping.

Categories follow the convention: Audio-then-Video:
- RARV: Real Audio, Real Video
- FARV: Fake Audio, Real Video
- RAFV: Real Audio, Fake Video
- FAFV: Fake Audio, Fake Video
"""

import pytest
from src.config import load_config


@pytest.fixture
def category_mapping():
    config = load_config()
    return config.dataset.category_mapping


def test_label_mapping_rarv(category_mapping):
    """RARV (RealVideo-RealAudio): video REAL, audio REAL, overall REAL."""
    mapping = category_mapping["RealVideo-RealAudio"]
    assert mapping.abbreviation == "RARV"
    assert mapping.video_label == "REAL"
    assert mapping.audio_label == "REAL"
    assert mapping.overall_label == "REAL"


def test_label_mapping_farv(category_mapping):
    """FARV (RealVideo-FakeAudio): video REAL, audio FAKE, overall FAKE."""
    mapping = category_mapping["RealVideo-FakeAudio"]
    assert mapping.abbreviation == "FARV"
    assert mapping.video_label == "REAL"
    assert mapping.audio_label == "FAKE"
    assert mapping.overall_label == "FAKE"


def test_label_mapping_rafv(category_mapping):
    """RAFV (FakeVideo-RealAudio): video FAKE, audio REAL, overall FAKE."""
    mapping = category_mapping["FakeVideo-RealAudio"]
    assert mapping.abbreviation == "RAFV"
    assert mapping.video_label == "FAKE"
    assert mapping.audio_label == "REAL"
    assert mapping.overall_label == "FAKE"


def test_label_mapping_fafv(category_mapping):
    """FAFV (FakeVideo-FakeAudio): video FAKE, audio FAKE, overall FAKE."""
    mapping = category_mapping["FakeVideo-FakeAudio"]
    assert mapping.abbreviation == "FAFV"
    assert mapping.video_label == "FAKE"
    assert mapping.audio_label == "FAKE"
    assert mapping.overall_label == "FAKE"
