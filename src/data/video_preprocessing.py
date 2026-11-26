"""Video preprocessing: uniform frame sampling and face extraction with MTCNN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

logger = logging.getLogger(__name__)


class FaceDetectorProtocol(Protocol):
    """Protocol for injectable face detectors (enables easy mocking in unit tests)."""

    def __call__(self, img: Image.Image | List[Image.Image]) -> Any:
        ...


def extract_frames(
    video_path: str | Path,
    max_frames: int = 8,
    sample_rate_fps: Optional[float] = None,
    max_side: Optional[int] = None,
) -> List[Image.Image]:
    """Uniformly sample up to max_frames frames from a video file using OpenCV.

    Returns a list of PIL Images in RGB format.
    """
    def as_rgb(frame):
        if max_side and max(frame.shape[:2]) > max_side:
            scale = max_side / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    path_str = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {path_str}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0 or np.isnan(total_frames):
        # Fallback: sequential read
        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(as_rgb(frame))
        cap.release()
        return frames

    # Uniform sampling indices
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

    sampled_images: List[Image.Image] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            sampled_images.append(as_rgb(frame))

    cap.release()
    return sampled_images


def detect_and_crop_faces(
    frames: List[Image.Image],
    detector: Optional[Any] = None,
    image_size: int = 224,
    margin: int = 20,
    min_face_size: int = 40,
    device: str = "cpu",
) -> Tuple[List[torch.Tensor], List[int]]:
    """Detect and crop faces from sampled frames.

    Args:
        frames: List of PIL images in RGB.
        detector: Optional callable/MTCNN instance. If None, initializes MTCNN lazily.
        image_size: Target square crop size (default 224).
        margin: Margin added to bounding box.
        min_face_size: Minimum face detection dimension.
        device: 'cpu' or 'cuda'.

    Returns:
        cropped_faces: List of normalized (3, image_size, image_size) torch Tensors.
        valid_indices: Frame indices where a face was successfully detected.
    """
    if not frames:
        return [], []

    if detector is None:
        try:
            from facenet_pytorch import MTCNN

            detector = MTCNN(
                image_size=image_size,
                margin=margin,
                min_face_size=min_face_size,
                keep_all=False,
                device=device,
                post_process=False,  # We handle normalization explicitly
            )
        except ImportError:
            logger.error("facenet-pytorch not installed. Cannot initialize MTCNN.")
            return [], []

    # Standard ImageNet normalization for EfficientNet backbones
    normalize_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cropped_tensors: List[torch.Tensor] = []
    valid_frame_indices: List[int] = []

    for i, frame in enumerate(frames):
        try:
            # detector may return PIL Image, Tensor, or numpy array
            face_result = detector(frame)
            if face_result is not None:
                if isinstance(face_result, torch.Tensor):
                    # Already a tensor
                    if face_result.dim() == 4 and face_result.shape[0] == 1:
                        face_result = face_result.squeeze(0)
                    if face_result.shape[0] != 3:
                        face_result = face_result.permute(2, 0, 1)
                    # Normalize if values in [0, 255]
                    if face_result.max() > 1.0:
                        face_result = face_result.float() / 255.0
                    # Apply normalization
                    norm_tensor = T.functional.normalize(
                        face_result.float(),
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )
                    cropped_tensors.append(norm_tensor)
                elif isinstance(face_result, Image.Image):
                    cropped_tensors.append(normalize_transform(face_result))
                else:
                    # Array or PIL
                    img = Image.fromarray(np.uint8(face_result))
                    cropped_tensors.append(normalize_transform(img))

                valid_frame_indices.append(i)
        except Exception as e:
            logger.debug(f"Face extraction failed for frame {i}: {e}")
            continue

    return cropped_tensors, valid_frame_indices
