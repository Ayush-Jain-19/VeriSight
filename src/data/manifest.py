"""Dataset manifest scanner and metadata parser for FakeAVCeleb and related layouts."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import CategoryMapping, load_config

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}

DEFAULT_CATEGORY_MAPPING = {
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


def parse_video_path(
    relative_path: Path,
    category_mapping: Dict[str, CategoryMapping],
) -> Optional[Dict[str, str]]:
    """Parse relative path parts into metadata fields.

    Standard FakeAVCeleb layout:
    <Category>/<Ethnicity>/<Gender>/<Identity_ID>/<filename>

    Example:
    RealVideo-RealAudio/African/men/id00076/00109.mp4
    """
    parts = relative_path.parts
    if len(parts) < 2:
        return None

    # Find which part matches known categories
    category_idx = -1
    matched_category = None
    for idx, part in enumerate(parts):
        for cat_name in category_mapping.keys():
            if cat_name.lower() in part.lower():
                category_idx = idx
                matched_category = cat_name
                break
        if matched_category:
            break

    if not matched_category or category_idx == -1:
        # Fallback: check first part
        first_part = parts[0]
        if first_part in category_mapping:
            matched_category = first_part
            category_idx = 0
        else:
            return None

    mapping = category_mapping[matched_category]

    # Defaults
    ethnicity = "unknown"
    gender = "unknown"
    identity_id = "unknown"
    method = "none" if mapping.overall_label == "REAL" else "synthetic"

    # Inspect remaining parts after category
    sub_parts = parts[category_idx + 1 :]
    # If standard layout: [ethnicity, gender, identity_id, filename]
    if len(sub_parts) >= 4:
        ethnicity = sub_parts[0]
        gender = sub_parts[1]
        identity_id = sub_parts[2]
        filename = sub_parts[-1]
    elif len(sub_parts) == 3:
        ethnicity = sub_parts[0]
        identity_id = sub_parts[1]
        filename = sub_parts[2]
    elif len(sub_parts) == 2:
        identity_id = sub_parts[0]
        filename = sub_parts[1]
    elif len(sub_parts) == 1:
        filename = sub_parts[0]
        # Try extracting identity from filename (e.g., id00076_001.mp4)
        match = re.search(r"(id\d+)", filename, re.IGNORECASE)
        if match:
            identity_id = match.group(1).lower()
    else:
        filename = parts[-1]

    # Check method in filename or parts (e.g., wav2lip, faceswap, fsgan)
    full_str = "/".join(parts).lower()
    for potential_method in ["wav2lip", "faceswap", "fsgan", "sv2tts"]:
        if potential_method in full_str:
            method = potential_method
            break

    return {
        "category": matched_category,
        "abbreviation": mapping.abbreviation,
        "ethnicity": ethnicity,
        "gender": gender,
        "identity_id": identity_id,
        "method": method,
        "video_label": mapping.video_label,
        "audio_label": mapping.audio_label,
        "overall_label": mapping.overall_label,
        "filename": filename,
    }


def scan_dataset(
    data_root: str | Path,
    category_mapping: Optional[Dict[str, CategoryMapping]] = None,
) -> pd.DataFrame:
    """Recursively scan data root directory and build a DataFrame manifest."""
    root_path = Path(data_root).resolve()
    if not root_path.exists():
        logger.warning(f"Data root path does not exist: {root_path}")
        return pd.DataFrame(
            columns=[
                "video_path",
                "category",
                "abbreviation",
                "ethnicity",
                "gender",
                "identity_id",
                "method",
                "video_label",
                "audio_label",
                "overall_label",
                "filename",
            ]
        )

    if category_mapping is None:
        category_mapping = DEFAULT_CATEGORY_MAPPING

    records = []
    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        for file_path in root_path.rglob(f"*{ext}"):
            try:
                rel_path = file_path.relative_to(root_path)
            except ValueError:
                rel_path = Path(file_path.name)

            parsed = parse_video_path(rel_path, category_mapping)
            if parsed:
                parsed["video_path"] = str(file_path.resolve())
                records.append(parsed)

    df = pd.DataFrame(records)
    if not df.empty:
        # Sort for deterministic output
        df = df.sort_values(by=["identity_id", "filename"]).reset_index(drop=True)

    logger.info(f"Discovered {len(df)} video entries from {root_path}")
    return df


def build_manifest_from_directory(
    data_root: str | Path,
    output_path: Optional[str | Path] = None,
    category_mapping: Optional[Dict[str, CategoryMapping]] = None,
) -> pd.DataFrame:
    """Scan dataset directory, save manifest to CSV if requested, and return DataFrame."""
    df = scan_dataset(data_root, category_mapping)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"Manifest saved to {out}")
    return df
