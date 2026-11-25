"""Unit tests for configuration loading and validation."""

from pathlib import Path
import pytest
from src.config import load_config, AppConfig


def test_load_default_config():
    """Verify that default config loads and contains valid defaults."""
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.project_name == "multimodal-deepfake-detection"
    assert config.device == "cpu"
    assert config.preprocessing.video.max_frames == 8
    assert config.preprocessing.audio.sample_rate == 16000
    assert config.models.fusion.input_dim == 258


def test_config_overrides(tmp_path):
    """Verify that overriding values updates nested config keys correctly."""
    override_file = tmp_path / "custom.yaml"
    override_file.write_text(
        """
device: "cuda"
preprocessing:
  video:
    max_frames: 12
training:
  batch_size: 16
""",
        encoding="utf-8",
    )

    config = load_config(overrides_path=override_file)
    assert config.device == "cuda"
    assert config.preprocessing.video.max_frames == 12
    assert config.training.batch_size == 16
    # Other un-overridden values should remain default
    assert config.preprocessing.audio.sample_rate == 16000


def test_extra_dict_overrides():
    """Verify programmatic dict overrides take precedence."""
    config = load_config(extra_overrides={"inference": {"tau_fusion": 0.85}})
    assert config.inference.tau_fusion == 0.85
