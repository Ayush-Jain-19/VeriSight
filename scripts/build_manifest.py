"""CLI script to scan dataset directory, generate manifest, and create identity splits."""

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.data.manifest import build_manifest_from_directory
from src.data.splits import create_identity_splits


def main():
    parser = argparse.ArgumentParser(description="Build FakeAVCeleb dataset manifest and identity splits.")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory containing FakeAVCeleb videos.")
    parser.add_argument("--output-manifest", type=str, default=None, help="Path to save manifest CSV.")
    parser.add_argument("--output-splits-dir", type=str, default=None, help="Directory to save train/val/test CSVs.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Ratio for training identities.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Ratio for validation identities.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Ratio for test identities.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")

    args = parser.parse_args()
    config = load_config()

    data_root = args.data_root or config.paths.data_root
    output_manifest = args.output_manifest or config.paths.manifest_path
    output_splits_dir = args.output_splits_dir or str(Path(config.paths.train_split).parent)

    print(f"Scanning dataset from: {data_root}")
    manifest_df = build_manifest_from_directory(
        data_root=data_root,
        output_path=output_manifest,
        category_mapping=config.dataset.category_mapping,
    )

    if manifest_df.empty:
        print(f"Warning: No video files found in {data_root}. Check directory path or file extensions.")
        sys.exit(0)

    print(f"Manifest created with {len(manifest_df)} samples. Generating splits...")
    create_identity_splits(
        df=manifest_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        group_column=config.splits.group_column,
        stratify_columns=config.splits.stratify_columns,
        seed=args.seed,
        output_dir=output_splits_dir,
    )
    print("Dataset manifest and identity splits successfully generated.")


if __name__ == "__main__":
    main()
