"""CLI script to extract identity-wise out-of-fold embeddings and scores for fusion training."""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import MultimodalDataset
from src.models.audio_model import AudioDeepfakeDetector
from src.models.video_model import VideoDeepfakeDetector
from src.training.oof_embeddings import assemble_oof_dataset, generate_identity_folds
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Extract out-of-fold features for fusion training.")
    parser.add_argument("--train-split", type=str, default=None, help="Path to train.csv split.")
    parser.add_argument("--output-npz", type=str, default=None, help="Path to save OOF features NPZ.")
    parser.add_argument("--folds", type=int, default=5, help="Number of identity folds.")
    parser.add_argument("--epochs-per-fold", type=int, default=5, help="Epochs to train per fold.")
    parser.add_argument("--config", type=str, default=None, help="Config file path.")

    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(config.device)

    train_csv = Path(args.train_split or config.paths.train_split)
    if not train_csv.exists():
        print(f"Error: Training split CSV not found at {train_csv}.")
        sys.exit(1)

    output_path = Path(
        args.output_npz or (Path(config.paths.data_root).parent / "fusion_train_oof_features.npz")
    )

    df_train = pd.read_csv(train_csv)
    print(f"Loaded {len(df_train)} samples across {df_train[config.splits.group_column].nunique()} identities.")

    folds = generate_identity_folds(
        df_train,
        n_folds=args.folds,
        group_column=config.splits.group_column,
        stratify_columns=config.splits.stratify_columns,
        seed=config.seed,
    )

    extracted_folds = []

    for k, (train_idx, held_out_idx) in enumerate(folds):
        print(f"\n--- Processing Fold {k+1}/{len(folds)} ---")
        fold_train_df = df_train.iloc[train_idx].reset_index(drop=True)
        fold_held_out_df = df_train.iloc[held_out_idx].reset_index(drop=True)

        # Datasets
        train_ds = MultimodalDataset(fold_train_df, max_frames=config.preprocessing.video.max_frames)
        held_out_ds = MultimodalDataset(fold_held_out_df, max_frames=config.preprocessing.video.max_frames)

        train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
        held_out_loader = DataLoader(held_out_ds, batch_size=config.training.batch_size, shuffle=False)

        # 1. Train Fold Video Model
        print(f"Training Fold {k+1} Video Model...")
        v_model = VideoDeepfakeDetector(
            backbone_name=config.models.video.backbone,
            pretrained=config.models.video.pretrained,
            freeze_backbone=config.models.video.freeze_backbone,
            embedding_dim=config.models.video.embedding_dim,
        ).to(device)
        v_opt = torch.optim.AdamW(v_model.parameters(), lr=config.training.learning_rate)
        v_trainer = Trainer(v_model, nn.BCEWithLogitsLoss(), v_opt, device=device)
        for ep in range(args.epochs_per_fold):
            v_trainer.train_epoch(train_loader, target_label_key="video_label")

        # 2. Train Fold Audio Model
        print(f"Training Fold {k+1} Audio Model...")
        a_model = AudioDeepfakeDetector(
            backbone_name=config.models.audio.backbone,
            freeze_backbone=config.models.audio.freeze_backbone,
            embedding_dim=config.models.audio.embedding_dim,
            use_mock_backbone=(config.models.audio.backbone == "mock"),
        ).to(device)
        a_opt = torch.optim.AdamW(a_model.parameters(), lr=config.training.learning_rate)
        a_trainer = Trainer(a_model, nn.BCEWithLogitsLoss(), a_opt, device=device)
        for ep in range(args.epochs_per_fold):
            a_trainer.train_epoch(train_loader, target_label_key="audio_label")

        # 3. Extract Features for Held-Out Fold
        print(f"Extracting un-optimistic features for Fold {k+1} held-out identities...")
        v_model.eval()
        a_model.eval()

        v_scores_list, a_scores_list = [], []
        v_embs_list, a_embs_list = [], []
        overall_labels_list = []

        with torch.no_grad():
            for batch in held_out_loader:
                faces = batch["faces"].to(device)
                wav = batch["waveform"].to(device)
                overall = batch["overall_label"].numpy()

                v_logits, v_emb = v_model(faces)
                a_logits, a_emb = a_model(wav)

                v_probs = torch.sigmoid(v_logits).cpu().numpy()
                a_probs = torch.sigmoid(a_logits).cpu().numpy()

                v_scores_list.extend(v_probs.tolist())
                a_scores_list.extend(a_probs.tolist())
                v_embs_list.append(v_emb.cpu().numpy())
                a_embs_list.append(a_emb.cpu().numpy())
                overall_labels_list.extend(overall.tolist())

        extracted_folds.append(
            {
                "video_scores": np.array(v_scores_list, dtype=np.float32),
                "audio_scores": np.array(a_scores_list, dtype=np.float32),
                "video_embeddings": np.concatenate(v_embs_list, axis=0),
                "audio_embeddings": np.concatenate(a_embs_list, axis=0),
                "overall_labels": np.array(overall_labels_list, dtype=np.float32),
            }
        )

    # Assemble and save
    assemble_oof_dataset(extracted_folds, output_path=output_path)
    print(f"\nAll {len(folds)} folds successfully extracted and saved to {output_path}")


if __name__ == "__main__":
    main()
