"""Model architectures for video, audio, and multimodal fusion deepfake detection."""

from src.models.audio_model import AudioDeepfakeDetector
from src.models.fusion_model import FusionDeepfakeDetector
from src.models.video_model import VideoDeepfakeDetector

__all__ = ["VideoDeepfakeDetector", "AudioDeepfakeDetector", "FusionDeepfakeDetector"]
