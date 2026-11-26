# VeriSight inference audit
25 November 2026 · CSET 346 PJ-07

## Reproduced before changing inference
The repository fixture `demo_samples/test_speech_synthesis.wav` has 6.7380045 s of audio. Calling `load_waveform(path,16000,30,0.1)` returned 480,000 samples (30 s). Transcoding the same file with FFmpeg's AAC encoder into an M4A container and calling the same loader produced “Format not recognised” in both decoder attempts and returned None. The standalone route allowed that suffix but skipped the FFmpeg extraction used for video. This reproduces the mechanism reported for WhatsApp. The original voice note was unavailable.

## Labels and decision rules
| Source of evidence | Index 0 | Index 1 | Finding |
|---|---|---|---|
| Cached Hello-SimpleAI/chatgpt-detector-roberta config, d2b342c | Human | ChatGPT | Old code selected 1 correctly. |
| Cached MelodyMachine/Deepfake-audio-detection config, 8fa1269 | fake | real | Old code selected 0 correctly. |
| Cached Organika/sdxl-detector config, b37fede | artificial | human | Old code selected 0 correctly. |
| Samson5827/deepfakeguard-v2 publisher labels and model card | FAKE | REAL | Single sigmoid for class 1. Complement is correct. ONNX has no Transformers id2label. |
| fakespot-ai/roberta-base-ai-text-detection-v1 public config | Human | AI | Semantically labeled binary head. |
| MelodyMachine/Deepfake-audio-detection-V2 public config | fake | real | Semantically labeled binary head. |
| rasbt/ai-text-detector-modernbert public config | HUMAN | AI | Supported alternate text head. |
| GeorgeDrayson/modernbert-ai-detection public config | unspecified | unspecified | No id2label in config. Generic LABEL_0/1 cannot establish direction. |
| Systran/faster-whisper-tiny | — | — | Speech-to-text decoder, no binary classification head. |

## Failure mechanisms and limits of the evidence
- **Audio** · The legacy training helper first decoded whole files, mixed channels, resampled, and padded/truncated to exactly the requested duration. Resampling itself had the expected source/target order. Padding distorted duration, whole-clip RMS, window boundaries and the minimum-duration check. The detector already discarded wholly quiet windows by RMS. A partially quiet five-second window, a half-second tail and five seconds of active speech could receive equal median weight. Energy alone admits noise. Window length and codec sensitivity require paired measurements. No causal error rate can be inferred from one clip.
- **Text** · The HC3-era classifier used the first 512 tokens, including special tokens. Consequently later sentences could never affect the score. The ASCII fraction accepted Latin-script non-English prose and treated punctuation/script as language evidence. No sentence attribution was computed. Domain shift from early ChatGPT is plausible, but the checkout contains no held-out text set establishing a degradation rate.
- **Verdicts** · The production pipeline applied 0.50 to raw scores independent of the historical calibration config. A reported 0.5001 synthetic-speech result was sufficient for FAKE. Missing evidence produced a constant 0.50 confidence. Direct routes marked thresholds calibrated and production readiness returned true before loading weights. The video code allowed a single available stream to clear a video, contradicting the reported silent-video abstention. Visual model errors were sometimes relabeled as missing faces.
- **Memory** · Four module-level detector singletons retained model references across requests. Local historical checkpoints could be initialized at startup even though normal prediction used different detectors. Loading eight full-resolution frames and decoding entire files also affected peak memory.
- **UI and tests** · The Findings Sheet attributed decisions to localized blending seams, perplexity or entropy without corresponding model outputs. Bar colors used the hard binary threshold. Several API tests explicitly injected untrained networks, including an assertion that silence was available. The old integration harness largely asserted HTTP status, then printed expected behavior. Neither constitutes held-out accuracy.
- **Benchmark** · The checkout contained a data README, demo fixtures and manifest/split tooling. No identity-disjoint labeled test CSV was present. Functional examples and upstream metrics do not establish team accuracy, calibration, explanation faithfulness or generalization.

Publisher sources: [Fakespot](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1/blob/main/config.json), [V2 audio](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2/blob/main/config.json), [SDXL](https://huggingface.co/Organika/sdxl-detector/blob/main/config.json), [face model](https://huggingface.co/Samson5827/deepfakeguard-v2), [GeorgeDrayson](https://huggingface.co/GeorgeDrayson/modernbert-ai-detection/blob/main/config.json), [rasbt](https://huggingface.co/rasbt/ai-text-detector-modernbert/blob/main/config.json).

## Verification after the upgrade

| Run | Observed result |
|---|---|
| Python 3.12 unit and integration suite | 69 passed, 1 skipped. The existing MTCNN integration test requires tests/fixtures/real_face_test.jpg, which was absent. The real API face branch ran on the bundled portrait. |
| React TypeScript and Vite production build | Succeeded, 1,486 modules transformed. |
| Existing live HTTP harness, updated to the new contract | 13/13 checks passed. The synthetic-speech case checks score availability. Its classification was incorrect. |
| Windows real-weight directional check, Fakespot | Austen 0.7637245, assistant-generated paragraph 0.9999372. Direction passed. Austen crossed the operational FAKE boundary. |
| Windows real-weight directional check, MelodyMachine V2 | LibriSpeech 0.00005353, synthetic fixture 0.00001846. Direction failed. Both classified REAL. The command exited 1. |
| AAC and M4A re-encoding of the same fixture | Both produced finite model scores. M4A 6.7380625 s, raw AAC 6.8266875 s. |
| Acoustic window sensitivity | Four-second weights gave 0.00129411 on LibriSpeech and 0.00004023 on the synthetic fixture. Five-second weights gave the scores above. Both window settings missed the known synthetic recording. |
| Real ASR plus deliberately different caption | Whisper transcribed 15 words. Compared with “The quick brown fox jumped over a fence.”, WER was 1.875, CER 2.17949, lexical cosine 0.04851. Insertions account for edit rates above one. The transcript style score was 0.24701. |
| Benchmark smoke run | Two bundled video fixtures produced ten assessments and the requested five report files. Both text-only baselines abstained. This verifies the report path, including an entire modality with zero coverage. |
| Working set | Final default text/audio run 932.5 MB; earlier run 944.4 MB. Separate face + whole-image API run 809.9 MB. ModernBERT with Torch 2.5.1 CPU and BF16 weights 715.0 MB on a single passage. An earlier FP32 ModernBERT load reached 1,393.3 MB. These are Windows process peaks on the exercised cases. |

The V2 label direction is confirmed by its config. Treat its missed synthetic recording as a real error. Reversing the labels or selecting decision boundaries on these fixtures would invalidate the check. A benchmark with known sources, speaker-disjoint splits and representative generators is still needed to characterize the error.

Default text and acoustic checkpoints were exercised from snapshots f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db and de3cde5a29c449bb5268814e421b46bf6ebdcd72. The alternate text snapshot was a6efd4bf6ecd852f332b2569ae0f98cae13b7985. The alternate was run with the CPU Torch 2.5.1 / torchvision 0.20.1 / torchaudio 2.5.1 package set on the existing Python 3.12 environment. The default checkout virtual environment retained Torch 2.2.2 CPU.

### Reproduction

Run the following in PowerShell from the repository root. The benchmark smoke manifest used the bundled test_video_realface_with_audio.mp4 (REAL) and test_video_faceswap_with_audio.mp4 (FAKE), with no supplied caption. The supplied project-spec file is ignored by Git.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m scripts.sanity_check_live --download
& $python -m pytest tests/ -v
& $python -m scripts.verify_all_modalities
& $python -m scripts.sanity_check_live
@(
    [pscustomobject]@{ path = 'demo_samples/test_video_realface_with_audio.mp4'; label = 'REAL' }
    [pscustomobject]@{ path = 'demo_samples/test_video_faceswap_with_audio.mp4'; label = 'FAKE' }
) | Export-Csv data/functional.csv -NoTypeInformation
& $python -m scripts.evaluate_benchmark data/functional.csv --root . --asr --output reports/smoke
Push-Location frontend
npm.cmd run build
Pop-Location
```

For a held-out evaluation replace the smoke manifest with your labeled test CSV. The default live command fetches the same LibriSpeech excerpt if you omit the local audio path.

## Checkpoint selection, 26 November 2026

Five clips, fixed five-second windows and the existing 0.45–0.55 abstention band. Scores below are synthetic-class scores. Peaks are Windows process working sets, in decimal MB.

| Clip | MelodyMachine V2 | Hemgg | Gary Stafford streamed INT8 | Gustking INT8 ONNX |
|---|---:|---:|---:|---:|
| LibriSpeech 5703 | 5.352878361009061e-05 | 0.9063324332237244 | 0.9785716533660889 | 0.15581493079662323 |
| LibriSpeech 3436 | 1.6812982721603476e-05 | 0.7858244180679321 | 0.9839158654212952 | 0.34307795763015747 |
| Repository synthesis | 1.8461825675331056e-05 | 0.9369385838508606 | 0.995600700378418 | 0.9152333736419678 |
| Flite SLT | 0.9999802112579346 | 0.9771676659584045 | 0.9938790798187256 | 0.8943591713905334 |
| SAPI Zira | 1.6891055565793067e-05 | 0.9772592782974243 | 0.9914347529411316 | 0.8929392695426941 |
| Peak MB | 871.264256 | 870.445056 | 878.809088 | 797.814784 |

Gary Stafford used 1362.35 MB with staged conversion. Streaming individual layers reduced its peak to 878.81 MB. The BF16-only trial was stopped for latency. Gustking classified 5/5 correctly and became the default. The other detectors scored 3/5. Both LibriSpeech excerpts were human. The positive clips were the bundled synthetic fixture, FFmpeg Flite SLT and Windows SAPI Zira.

| Passage | Fakespot | ModernBERT |
|---|---:|---:|
| modern news | 0.7440916299819946 | 0.9763652682304382 |
| generated news | 0.9250738024711609 | 0.8849467039108276 |
| casual writing | 0.14268870651721954 | 0.0045091803185641766 |
| generated casual | 0.5057657957077026 | 0.02143876999616623 |
| classic Austen | 0.7637245059013367 | 0.0014650047523900867 |
| generated essay | 0.9999371767044067 | 0.98674076795578 |
| Peak MB | 787.4 | 717.9 |

Fakespot retained as default. ModernBERT improved from 3/6 to 4/6 on the fixed passages, but reversed the news pair. The small selection set cannot establish generalization.

Final pytest: 71 passed in 68.26 seconds, zero skips. Final combined sanity check: PASS, peak 933.3 MB. M4A and AAC both decoded and scored.
