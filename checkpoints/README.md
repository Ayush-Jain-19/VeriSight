# Model Checkpoints

This directory stores trained model weights, configurations, and calibration thresholds.
Checkpoints are ignored by git to keep the repository lightweight.

## Expected Files
When downloading models trained in Google Colab or locally, place them here:
- `video_efficientnet_b0_best.pt`
- `audio_wav2vec2_base_best.pt`
- `fusion_mlp_best.pt`
- `thresholds.yaml` (stores calibrated thresholds τ_fusion, τ_video, and τ_audio)

## Format of `thresholds.yaml`
```yaml
tau_fusion: 0.65
tau_video: 0.70
tau_audio: 0.68
calibration_metadata:
  fusion_coverage: 0.912
  video_coverage: 0.885
  audio_coverage: 0.897
```
