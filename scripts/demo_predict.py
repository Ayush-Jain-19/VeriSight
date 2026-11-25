"""Quick CLI demo prediction script."""

import json
import sys
from pathlib import Path
from src.inference.pipeline import InferencePipeline

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "demo_samples/test_pattern_with_audio.mp4"
    print(f"[DEMO] Analyzing video: {video_path}")
    pipeline = InferencePipeline()
    result = pipeline.predict(video_path)
    print("\n--- INFERENCE RESULTS ---")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
