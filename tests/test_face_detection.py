"""Integration test for MTCNN face detection.

Skipped automatically in environments without a designated test fixture.
"""

from pathlib import Path
import pytest
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    HAS_FACENET = True
except ImportError:
    HAS_FACENET = False

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests/fixtures/real_face_test.jpg"
if not FIXTURE_PATH.is_file():
    FIXTURE_PATH = ROOT / "demo_samples/test_face_real.jpg"


@pytest.mark.skipif(not HAS_FACENET, reason="facenet-pytorch is not installed")
@pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="No real face test fixture provided")
def test_mtcnn_real_detection():
    """Integration test verifying MTCNN detection on a permitted real face fixture."""
    img = Image.open(FIXTURE_PATH).convert("RGB")
    detector = MTCNN(image_size=224, margin=20, keep_all=False, device="cpu")
    crop = detector(img)
    assert crop is not None
