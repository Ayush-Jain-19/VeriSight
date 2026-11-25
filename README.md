# VeriSight — Multimodal Deepfake Detection System

A production-grade, CPU-optimized multimodal deepfake detection system running in-place on Windows (Intel i3, 8 GB RAM, no CUDA). Analyzes video, audio, image, and text modalities using task-trained open-source deep learning checkpoints with strict zero-mock reporting and evidence-based verdicts (`REAL`, `FAKE`, or `UNCERTAIN`).

---

## Modality Detectors & Model Credits

All models run sequentially on CPU with memory bounding (no GPU required):

1. **Visual Face-Swap Detection (Video Frames & Image Tab)**:
   - **Model**: [`Samson5827/deepfakeguard-v2`](https://huggingface.co/Samson5827/deepfakeguard-v2)
   - **Architecture**: ONNX export running on ONNXRuntime CPU (`CPUExecutionProvider`).
   - **Preprocessing**: Frontal face detection (MTCNN / Haar Cascade fallback), 18% margin crop (`margin = 0.18`), resized to `224×224 RGB`, unnormalized float32 pixels `[0, 255]`, input layout `(B, 224, 224, 3)`.
   - **Scoring**: Scalar output is REAL probability; face-swap score = `1.0 - real_output`.
   - **Handling**: If no detectable face is present, returns `"No usable face"` with `score: null`.

2. **General AI-Image Detection (Selectable Image Check)**:
   - **Model**: [`Organika/sdxl-detector`](https://huggingface.co/Organika/sdxl-detector)
   - **Architecture**: Swin Transformer for Image Classification.
   - **Preprocessing**: `AutoImageProcessor` native normalization and square patch tokenization.
   - **Scoring**: Read from `id2label` (`0: 'artificial', 1: 'human'`). Reports synthetic AI generation probability.

3. **Acoustic Speech Deepfake Detection (Audio Tab & Video Soundtrack)**:
   - **Model**: [`MelodyMachine/Deepfake-audio-detection`](https://huggingface.co/MelodyMachine/Deepfake-audio-detection)
   - **Architecture**: Wav2Vec2 audio classification.
   - **Preprocessing**: Audio decoded to mono 16 kHz WAV via FFmpeg / PySoundFile.
   - **CPU Memory Bounding**: Audio is segmented into short 5-second analysis windows (80,000 samples) up to 30 seconds max to prevent RAM spikes on 8 GB systems.
   - **Silence Gating**: Zero signal, silence, or duration < 0.5s returns `"Silence or insufficient speech detected"` with `score: null`.
   - **Scoring**: Read from `id2label` (`0: 'fake', 1: 'real'`).

4. **Linguistic AI-Writing Style Estimation (Text Tab)**:
   - **Model**: [`Hello-SimpleAI/chatgpt-detector-roberta`](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta)
   - **Architecture**: RoBERTa sequence classification with safetensors weights.
   - **Label**: `"Experimental AI-writing style estimate"`.
   - **Handling**: Texts shorter than 50 characters or 10 words, or unsupported languages (> 35% non-Latin), return an explicit limitation reason with `score: null`.

5. **Video Modality Inspection**:
   - Exactly 8 evenly spaced frames sampled across the clip.
   - Visual branch runs `deepfakeguard-v2` on usable face crops; reports median score and face count (`X/8`).
   - Acoustic branch extracts soundtrack and evaluates speech with `Deepfake-audio-detection`.
   - Separate visual and acoustic findings are reported independently.
   - If visual and audio branches disagree, the cross-modal discrepancy is flagged and explained.
   - No synthetic/fused probabilities are fabricated without a genuine trained fusion checkpoint.
   - Optional subtitles never vote on facial or acoustic manipulation; reports `"Transcript comparison unavailable"`.

---

## Directory Structure

```
multimodal-deepfake-detection/
├── checkpoints/            # Local cached checkpoints (chatgpt_detector_roberta safetensors)
├── demo_samples/           # Test fixtures: real face, fake face, speech, silence, silent video
├── frontend/               # React 18 + TypeScript + Vite UI ("The Screening Room" theme)
│   ├── src/components/     # SourceMonitor, FindingsSheet
│   ├── src/styles/         # global.css design tokens
│   └── src/types/          # TypeScript API response definitions
├── src/
│   ├── api/                # FastAPI application, REST endpoints, Pydantic schemas
│   ├── data/               # Frame extraction, face detection, audio loading, splits
│   ├── inference/          # InferencePipeline & pretrained_models wrappers
│   └── models/             # PyTorch model definitions
├── scripts/
│   └── verify_all_modalities.py # 13-case end-to-end API verification suite
└── tests/                  # Automated pytest unit test suite (50 tests)
```

---

## Windows Launch Commands

Run these commands in Windows PowerShell:

### 1. Start the FastAPI Backend (Port 8000)
```powershell
cd "C:\Users\hp\.gemini\antigravity\scratch\multimodal-deepfake-detection"
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Start the React Frontend (Port 5173)
```powershell
cd "C:\Users\hp\.gemini\antigravity\scratch\multimodal-deepfake-detection\frontend"
npm run dev
```

Open your browser to: **http://localhost:5173**

---

## Running Automated Verification

### Run End-to-End Multimodal Matrix (13 Tests Across All 4 Tabs)
```powershell
cd "C:\Users\hp\.gemini\antigravity\scratch\multimodal-deepfake-detection"
.\.venv\Scripts\python.exe scripts/verify_all_modalities.py
```

### Run PyTest Suite (50 Tests)
```powershell
cd "C:\Users\hp\.gemini\antigravity\scratch\multimodal-deepfake-detection"
.\.venv\Scripts\pytest.exe tests/ -v
```

---

## Model Credits & Citations

- **Face-Swap Detector**: Samson5827 (`Samson5827/deepfakeguard-v2`), Hugging Face Model Hub.
- **Audio Deepfake Detector**: MelodyMachine (`MelodyMachine/Deepfake-audio-detection`), Hugging Face Model Hub.
- **General AI-Image Detector**: Organika (`Organika/sdxl-detector`), Hugging Face Model Hub.
- **ChatGPT Writing Style Detector**: Hello-SimpleAI (`Hello-SimpleAI/chatgpt-detector-roberta`), Hugging Face Model Hub.

---

## License
Educational and research use only.
