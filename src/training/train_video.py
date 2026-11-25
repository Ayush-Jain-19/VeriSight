"""Video baseline model training script."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import VideoDataset
from src.models.video_model import VideoDeepfakeDetector
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def train_video_model(config_path: str | None = None, overrides: dict | None = None):
    """Train EfficientNet-B0 video classifier on video labels."""
    config = load_config(config_path, extra_overrides=overrides)
    device = torch.device(config.device)

    train_csv = Path(config.paths.train_split)
    val_csv = Path(config.paths.val_split)

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Split files not found at {train_csv} or {val_csv}. Please run build_manifest.py first."
        )

    # 1. Datasets and Loaders
    train_dataset = VideoDataset(
        manifest_or_csv=train_csv,
        max_frames=config.preprocessing.video.max_frames,
        image_size=config.preprocessing.video.face_image_size,
    )
    val_dataset = VideoDataset(
        manifest_or_csv=val_csv,
        max_frames=config.preprocessing.video.max_frames,
        image_size=config.preprocessing.video.face_image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    # 2. Model, Loss, Optimizer
    model = VideoDeepfakeDetector(
        backbone_name=config.models.video.backbone,
        pretrained=config.models.video.pretrained,
        freeze_backbone=config.models.video.freeze_backbone,
        embedding_dim=config.models.video.embedding_dim,
        dropout=config.models.video.dropout,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # 3. Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config.paths.checkpoints_dir,
        model_name="video_efficientnet_b0",
        early_stopping_patience=config.training.early_stopping_patience,
    )

    return trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train video deepfake detection baseline.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()
    train_video_model(config_path=args.config)
