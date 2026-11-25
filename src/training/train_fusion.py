"""Fusion model training script using out-of-fold extracted unimodal features."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.models.fusion_model import FusionDeepfakeDetector
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def train_fusion_model(
    features_npz_path: str | Path | None = None,
    val_features_npz_path: str | Path | None = None,
    config_path: str | None = None,
    overrides: dict | None = None,
):
    """Train late-fusion MLP on out-of-fold unimodal features to predict overall manipulation."""
    config = load_config(config_path, extra_overrides=overrides)
    device = torch.device(config.device)

    # Load pre-extracted OOF features
    if features_npz_path is None:
        features_npz_path = Path(config.paths.data_root).parent / "fusion_train_oof_features.npz"

    npz_p = Path(features_npz_path)
    if not npz_p.exists():
        raise FileNotFoundError(
            f"OOF features NPZ file not found at {npz_p}. Please run extract_oof_embeddings.py first."
        )

    data = np.load(npz_p)
    v_scores = torch.from_numpy(data["video_scores"]).float().unsqueeze(1)
    a_scores = torch.from_numpy(data["audio_scores"]).float().unsqueeze(1)
    v_embs = torch.from_numpy(data["video_embeddings"]).float()
    a_embs = torch.from_numpy(data["audio_embeddings"]).float()
    y_train = torch.from_numpy(data["overall_labels"]).float()

    X_train = FusionDeepfakeDetector.construct_fusion_input(v_scores, a_scores, v_embs, a_embs)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)

    # Optional validation features
    if val_features_npz_path and Path(val_features_npz_path).exists():
        val_data = np.load(val_features_npz_path)
        X_val = FusionDeepfakeDetector.construct_fusion_input(
            torch.from_numpy(val_data["video_scores"]).float().unsqueeze(1),
            torch.from_numpy(val_data["audio_scores"]).float().unsqueeze(1),
            torch.from_numpy(val_data["video_embeddings"]).float(),
            torch.from_numpy(val_data["audio_embeddings"]).float(),
        )
        y_val = torch.from_numpy(val_data["overall_labels"]).float()
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    else:
        # Use simple split for validation
        n_val = max(1, int(len(train_dataset) * 0.15))
        n_tr = len(train_dataset) - n_val
        tr_ds, val_ds = torch.utils.data.random_split(train_dataset, [n_tr, n_val])
        train_loader = DataLoader(tr_ds, batch_size=config.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = FusionDeepfakeDetector(
        input_dim=config.models.fusion.input_dim,
        hidden_dims=config.models.fusion.hidden_dims,
        dropout=config.models.fusion.dropout,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config.paths.checkpoints_dir,
        model_name="fusion_mlp",
        early_stopping_patience=config.training.early_stopping_patience,
    )

    return trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fusion MLP on out-of-fold features.")
    parser.add_argument("--features", type=str, default=None, help="Path to OOF features NPZ.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()
    train_fusion_model(features_npz_path=args.features, config_path=args.config)
