"""Audio preprocessing: FFmpeg audio extraction and waveform normalization."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg executable is available on PATH."""
    return shutil.which("ffmpeg") is not None


def extract_audio(
    video_path: str | Path,
    output_wav_path: Optional[str | Path] = None,
    sample_rate: int = 16000,
) -> Path:
    """Extract audio track from video file as 16 kHz mono PCM WAV using FFmpeg.

    Returns:
        Path to extracted WAV file.
    """
    video_p = Path(video_path).resolve()
    if not video_p.exists():
        raise FileNotFoundError(f"Video file not found: {video_p}")

    if not is_ffmpeg_available():
        raise RuntimeError(
            "FFmpeg executable not found on PATH. Please ensure FFmpeg is installed and added to PATH."
        )

    if output_wav_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_p = Path(temp_file.name)
        temp_file.close()
    else:
        out_p = Path(output_wav_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

    # Command: ffmpeg -y -i <video> -vn -acodec pcm_s16le -ar <sample_rate> -ac 1 <output.wav>
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_p),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(out_p),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error("FFmpeg audio extraction failed with return code %d: %s", result.returncode, result.stderr)
        if out_p.exists() and output_wav_path is None:
            out_p.unlink(missing_ok=True)
        clean_err = "Invalid or unsupported media container"
        if result.stderr:
            clean_lines = [
                line.strip()
                for line in result.stderr.strip().splitlines()
                if line.strip()
                and not line.startswith("ffmpeg version")
                and not line.startswith("configuration:")
                and not line.startswith("lib")
            ]
            if clean_lines:
                last_line = clean_lines[-1]
                if ":" in last_line:
                    last_line = last_line.split(":")[-1].strip()
                if last_line:
                    clean_err = last_line
        raise RuntimeError(f"FFmpeg audio extraction failed: {clean_err}")

    return out_p


def load_waveform(
    wav_path: str | Path,
    target_sample_rate: int = 16000,
    max_duration_seconds: float = 4.0,
    min_duration_seconds: float = 0.5,
) -> Optional[torch.Tensor]:
    """Load and format audio waveform to a fixed length 1D torch Tensor.

    Target length: target_sample_rate * max_duration_seconds (e.g. 64,000 samples).
    Returns None if audio is shorter than min_duration_seconds.
    """
    p = Path(wav_path).resolve()
    if not p.exists():
        logger.error(f"WAV file does not exist: {p}")
        return None

    try:
        waveform, sr = torchaudio.load(str(p))
    except Exception as e:
        # Fallback to soundfile if torchaudio backend has issue
        try:
            data, sr = sf.read(str(p))
            waveform = torch.from_numpy(data).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()  # channels first
        except Exception as ex:
            logger.error(f"Failed to load audio {p}: {e} / {ex}")
            return None

    # Convert to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Flatten to 1D
    waveform = waveform.squeeze(0)

    target_length = int(target_sample_rate * max_duration_seconds)
    min_length = int(target_sample_rate * min_duration_seconds)

    if waveform.numel() < min_length:
        logger.warning(
            f"Audio duration ({waveform.numel() / target_sample_rate:.2f}s) is shorter than minimum threshold ({min_duration_seconds}s)"
        )
        return None

    # Pad or truncate to target_length
    if waveform.numel() < target_length:
        # Zero padding
        padding = target_length - waveform.numel()
        waveform = torch.nn.functional.pad(waveform, (0, padding), mode="constant", value=0.0)
    elif waveform.numel() > target_length:
        # Truncate
        waveform = waveform[:target_length]

    return waveform
