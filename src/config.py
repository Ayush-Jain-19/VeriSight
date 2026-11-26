"""Configuration management using Pydantic and PyYAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
import yaml
from pydantic import BaseModel, Field, field_validator


class CategoryMapping(BaseModel):
    abbreviation: str
    video_label: str
    audio_label: str
    overall_label: str


class PathsConfig(BaseModel):
    data_root: str = "./data/FakeAVCeleb"
    manifest_path: str = "./data/manifest.csv"
    train_split: str = "./data/train.csv"
    val_split: str = "./data/val.csv"
    test_split: str = "./data/test.csv"
    checkpoints_dir: str = "./checkpoints"
    logs_dir: str = "./logs"
    preprocessed_cache_dir: str = "./data/cache"


class DatasetConfig(BaseModel):
    category_regex: str = r"(RealVideo-RealAudio|RealVideo-FakeAudio|FakeVideo-RealAudio|FakeVideo-FakeAudio)"
    category_mapping: Dict[str, CategoryMapping] = Field(
        default_factory=lambda: {
            "RealVideo-RealAudio": CategoryMapping(
                abbreviation="RARV", video_label="REAL", audio_label="REAL", overall_label="REAL"
            ),
            "RealVideo-FakeAudio": CategoryMapping(
                abbreviation="FARV", video_label="REAL", audio_label="FAKE", overall_label="FAKE"
            ),
            "FakeVideo-RealAudio": CategoryMapping(
                abbreviation="RAFV", video_label="FAKE", audio_label="REAL", overall_label="FAKE"
            ),
            "FakeVideo-FakeAudio": CategoryMapping(
                abbreviation="FAFV", video_label="FAKE", audio_label="FAKE", overall_label="FAKE"
            ),
        }
    )


class SplitsConfig(BaseModel):
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    group_column: str = "identity_id"
    stratify_columns: List[str] = Field(default_factory=lambda: ["ethnicity", "gender"])
    min_samples_per_stratum_warning: int = 3


class VideoPreprocessingConfig(BaseModel):
    sample_rate_fps: float = 1.0
    max_frames: int = 8
    hard_max_frames: int = 16
    face_image_size: int = 224
    min_face_size: int = 40
    face_margin: int = 20


class AudioPreprocessingConfig(BaseModel):
    sample_rate: int = 16000
    max_duration_seconds: float = 4.0
    min_duration_seconds: float = 0.5


class PreprocessingConfig(BaseModel):
    video: VideoPreprocessingConfig = Field(default_factory=VideoPreprocessingConfig)
    audio: AudioPreprocessingConfig = Field(default_factory=AudioPreprocessingConfig)


class VideoModelConfig(BaseModel):
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone: bool = True
    embedding_dim: int = 128
    dropout: float = 0.3


class AudioModelConfig(BaseModel):
    backbone: str = "facebook/wav2vec2-base"
    freeze_backbone: bool = True
    embedding_dim: int = 128
    dropout: float = 0.3


class FusionModelConfig(BaseModel):
    input_dim: int = 258
    hidden_dims: List[int] = Field(default_factory=lambda: [256, 64])
    dropout: float = 0.3


class ModelsConfig(BaseModel):
    video: VideoModelConfig = Field(default_factory=VideoModelConfig)
    audio: AudioModelConfig = Field(default_factory=AudioModelConfig)
    fusion: FusionModelConfig = Field(default_factory=FusionModelConfig)


class TrainingConfig(BaseModel):
    batch_size: int = 8
    num_workers: int = 0
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    epochs: int = 20
    early_stopping_patience: int = 5
    oof_folds: int = 5


class CalibrationConfig(BaseModel):
    target_metric: str = "coverage"
    target_value: float = 0.90
    threshold_step: float = 0.01


class InferenceConfig(BaseModel):
    audio_model: str = "pranjal-pravesh/wav2vec2-large-xlsr-deepfake-audio-classification"
    text_model: str = "fakespot-ai/roberta-base-ai-text-detection-v1"
    asr_enabled: bool = False
    strategy: Literal["disagreement_max", "confidence_weighted_fusion"] = "disagreement_max"
    text_band: tuple[float, float] = (.35, .65)
    audio_band: tuple[float, float] = (.45, .55)
    video_band: tuple[float, float] = (.45, .55)
    fusion_band: tuple[float, float] = (.40, .60)
    window_seconds: Literal[4, 5] = 5

    @field_validator("text_band", "audio_band", "video_band", "fusion_band")
    @classmethod
    def validate_band(cls, band):
        if not 0 <= band[0] < .5 < band[1] <= 1:
            raise ValueError("Each decision band must straddle 0.5 inside [0,1]")
        return band

    tau_fusion: float = 0.65
    tau_video: float = 0.70
    tau_audio: float = 0.68
    low_face_count_threshold: int = 3
    batch_size: int = 1


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])


class AppConfig(BaseModel):
    project_name: str = "multimodal-deepfake-detection"
    seed: int = 42
    device: str = "cpu"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)


def _deep_update(base: dict, update: dict) -> dict:
    """Recursively update a dictionary."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    config_path: Optional[str | Path] = None,
    overrides_path: Optional[str | Path] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """Load application configuration from YAML files with optional overrides."""
    base_dict: Dict[str, Any] = {}

    # 1. Load default config
    if config_path is None:
        # Search relative to this file
        current_dir = Path(__file__).resolve().parent
        potential_default = current_dir.parent / "configs" / "default.yaml"
        if potential_default.exists():
            config_path = potential_default

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if loaded:
                base_dict = loaded

    # 2. Apply override file if specified
    if overrides_path and Path(overrides_path).exists():
        with open(overrides_path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f)
            if overrides:
                base_dict = _deep_update(base_dict, overrides)

    # 3. Apply extra dict overrides
    if extra_overrides:
        base_dict = _deep_update(base_dict, extra_overrides)

    return AppConfig(**base_dict)
