# FakeAVCeleb Dataset Setup

FakeAVCeleb is a multimodal benchmark dataset for deepfake video and audio detection.

## Licensing & Access
Access is restricted for academic and research purposes.
1. Fill out the official [FakeAVCeleb Request Form](https://bit.ly/38prlVO).
2. Follow the instructions received from the dataset maintainers to download the archive.
3. Extract the contents into this directory:
   `multimodal-deepfake-detection/data/FakeAVCeleb/`

## Expected Folder Structure
The dataset directory scanner supports configurable layouts. The standard layout is:
```
data/
└── FakeAVCeleb/
    ├── RealVideo-RealAudio/   (RARV: Video Real, Audio Real, Overall Real)
    ├── RealVideo-FakeAudio/   (FARV: Video Real, Audio Fake, Overall Fake)
    ├── FakeVideo-RealAudio/   (RAFV: Video Fake, Audio Real, Overall Fake)
    └── FakeVideo-FakeAudio/   (FAFV: Video Fake, Audio Fake, Overall Fake)
        ├── African/
        │   ├── men/
        │   └── women/
        ├── Caucasian/
        ├── Asian (Eastern)/
        └── Asian (Southern)/
```

## Building Manifest & Splits
Run the scanner script after placing the data:
```bash
python scripts/build_manifest.py --data-root ./data/FakeAVCeleb --output ./data/manifest.csv
```
This generates `manifest.csv`, `train.csv`, `val.csv`, and `test.csv` using identity-stratified splitting.
