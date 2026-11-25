"""CLI script to evaluate models and produce comparative evaluation reports."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import AudioDataset, MultimodalDataset, VideoDataset
from src.evaluation.compare import generate_markdown_comparison_report
from src.evaluation.evaluate import evaluate_model_on_dataset
from src.models.audio_model import AudioDeepfakeDetector
from src.models.fusion_model import FusionDeepfakeDetector
from src.models.video_model import VideoDeepfakeDetector


def main():
    parser = argparse.ArgumentParser(description="Evaluate unimodal and fusion deepfake detectors.")
    parser.add_argument("--test-split", type=str, default=None, help="Path to test.csv split.")
    parser.add_argument("--checkpoints-dir", type=str, default=None, help="Directory containing .pt checkpoints.")
    parser.add_argument("--output-report", type=str, default="evaluation_report.md", help="Path to save report.")
    parser.add_argument("--config", type=str, default=None, help="Config file path.")

    args = parser.parse_args()
    config = load_config(args.config)

    test_csv = Path(args.test_split or config.paths.test_split)
    ckpt_dir = Path(args.checkpoints_dir or config.paths.checkpoints_dir)

    if not test_csv.exists():
        print(f"Error: Test split CSV not found at {test_csv}.")
        sys.exit(1)

    print(f"Loading test dataset from {test_csv}...")
    df = pd.read_csv(test_csv)

    v_ckpt = ckpt_dir / "video_efficientnet_b0_best.pt"
    a_ckpt = ckpt_dir / "audio_wav2vec2_base_best.pt"
    f_ckpt = ckpt_dir / "fusion_mlp_best.pt"

    if not (v_ckpt.exists() and a_ckpt.exists() and f_ckpt.exists()):
        print(f"Warning: Checkpoints not found in {ckpt_dir}. Please place trained .pt models.")
        print(f"Looking for: {v_ckpt.name}, {a_ckpt.name}, {f_ckpt.name}")
        sys.exit(1)

    device = torch.device(config.device)

    # 1. Evaluate Video Model
    print("Evaluating Video Model...")
    v_model = VideoDeepfakeDetector(backbone_name=config.models.video.backbone, pretrained=False)
    v_model.load_state_dict(torch.load(v_ckpt, map_location=device))
    v_ds = VideoDataset(df, max_frames=config.preprocessing.video.max_frames)
    v_loader = DataLoader(v_ds, batch_size=config.training.batch_size, shuffle=False)
    v_metrics = evaluate_model_on_dataset(v_model, v_loader, device=device, tau=config.inference.tau_video)

    # 2. Evaluate Audio Model
    print("Evaluating Audio Model...")
    a_model = AudioDeepfakeDetector(backbone_name=config.models.audio.backbone)
    a_model.load_state_dict(torch.load(a_ckpt, map_location=device))
    a_ds = AudioDataset(df, sample_rate=config.preprocessing.audio.sample_rate)
    a_loader = DataLoader(a_ds, batch_size=config.training.batch_size, shuffle=False)
    a_metrics = evaluate_model_on_dataset(a_model, a_loader, device=device, tau=config.inference.tau_audio)

    # 3. Evaluate Fusion Model
    print("Evaluating Multimodal Fusion Model...")
    f_model = FusionDeepfakeDetector(input_dim=config.models.fusion.input_dim)
    f_model.load_state_dict(torch.load(f_ckpt, map_location=device))
    m_ds = MultimodalDataset(df)
    m_loader = DataLoader(m_ds, batch_size=config.training.batch_size, shuffle=False)
    f_metrics = evaluate_model_on_dataset(f_model, m_loader, device=device, tau=config.inference.tau_fusion)

    # 4. Generate Report
    report = generate_markdown_comparison_report(
        video_metrics=v_metrics,
        audio_metrics=a_metrics,
        fusion_metrics=f_metrics,
        output_path=args.output_report,
    )
    print(f"Evaluation report successfully generated and saved to {args.output_report}")


if __name__ == "__main__":
    main()
