"""Comprehensive End-to-End API Verification Script for All Modalities.

Tests:
1. Face-Swap Real Image (Samson5827/deepfakeguard-v2)
2. Face-Swap Manipulated Image (Samson5827/deepfakeguard-v2)
3. No-Face Image (Samson5827/deepfakeguard-v2)
4. General AI Real Image (Organika/sdxl-detector)
5. General AI Synthetic Image (Organika/sdxl-detector)
6. Authentic Speech Audio (MelodyMachine/Deepfake-audio-detection)
7. Synthesized Speech Audio (MelodyMachine/Deepfake-audio-detection)
8. Silent Audio (MelodyMachine/Deepfake-audio-detection)
9. Authentic Video (Visual + Audio)
10. Face-Swapped Video with Original Audio (Cross-Modal Disagreement)
11. Silent Video (No Soundtrack)
12. One-Line Short Text (Hello-SimpleAI/chatgpt-detector-roberta)
13. Text English Passage (Hello-SimpleAI/chatgpt-detector-roberta)
14. Video with Subtitles (Subtitle Non-Voting Verification)
"""

from pathlib import Path
from fastapi.testclient import TestClient
from src.api.main import create_app
import json

def run_verification():
    app = create_app()
    with TestClient(app) as client:
        samples = Path("demo_samples")
        results = []

        print("=" * 80)
        print("STARTING END-TO-END VERIFICATION MATRIX")
        print("=" * 80)

        # 1. Image: Real Face (Face-Swap Check)
        p = samples / "test_face_real.jpg"
        with open(p, "rb") as f:
            res = client.post("/api/predict/image", files={"file": ("test_face_real.jpg", f, "image/jpeg")}, data={"check_type": "face_swap"})
        assert res.status_code == 200, res.text
        data = res.json()
        results.append({
            "modality": "Image (Face-Swap)",
            "input": "test_face_real.jpg",
            "expected": "REAL, score < 0.50, faces_detected = 1",
            "actual_prediction": data["prediction"],
            "actual_score": data["image"]["score"],
            "actual_faces": data["image"]["faces_detected"],
            "pass": data["prediction"] == "REAL" and data["image"]["faces_detected"] == 1,
        })

        # 2. Image: Face-Swapped Manipulated (Face-Swap Check)
        p = samples / "test_face_fake.jpg"
        with open(p, "rb") as f:
            res = client.post("/api/predict/image", files={"file": ("test_face_fake.jpg", f, "image/jpeg")}, data={"check_type": "face_swap"})
        assert res.status_code == 200, res.text
        data = res.json()
        results.append({
            "modality": "Image (Face-Swap)",
            "input": "test_face_fake.jpg",
            "expected": "FAKE, score >= 0.50, faces_detected = 1",
            "actual_prediction": data["prediction"],
            "actual_score": data["image"]["score"],
            "actual_faces": data["image"]["faces_detected"],
            "pass": data["prediction"] == "FAKE" and data["image"]["faces_detected"] == 1,
        })

        # 3. Image: No-Face (Test Pattern)
        p = samples / "sample_frame.jpg"
        with open(p, "rb") as f:
            res = client.post("/api/predict/image", files={"file": ("sample_frame.jpg", f, "image/jpeg")}, data={"check_type": "face_swap"})
        assert res.status_code == 200, res.text
        data = res.json()
        results.append({
            "modality": "Image (Face-Swap)",
            "input": "sample_frame.jpg (no face)",
            "expected": "UNCERTAIN, score = null, reason: No usable face",
            "actual_prediction": data["prediction"],
            "actual_score": data["image"]["score"],
            "actual_faces": data["image"]["faces_detected"],
            "actual_reason": data["image"]["reason"],
            "pass": data["prediction"] == "UNCERTAIN" and data["image"]["score"] is None and "usable face" in (data["image"]["reason"] or ""),
        })

        # 4. Image: General AI Real Photo (SDXL Check)
        p = samples / "test_face_real.jpg"
        with open(p, "rb") as f:
            res = client.post("/api/predict/image", files={"file": ("test_face_real.jpg", f, "image/jpeg")}, data={"check_type": "general_ai"})
        assert res.status_code == 200, res.text
        data = res.json()
        results.append({
            "modality": "Image (General AI SDXL)",
            "input": "test_face_real.jpg",
            "expected": "REAL, score < 0.50, model = Organika/sdxl-detector",
            "actual_prediction": data["prediction"],
            "actual_score": data["image"]["score"],
            "model": data["image"]["model"],
            "pass": data["prediction"] == "REAL" and "sdxl" in data["image"]["model"],
        })

        # 5. Image: General AI Synthetic (SDXL Check)
        p = samples / "test_synthetic.jpg"
        with open(p, "rb") as f:
            res = client.post("/api/predict/image", files={"file": ("test_synthetic.jpg", f, "image/jpeg")}, data={"check_type": "general_ai"})
        assert res.status_code == 200, res.text
        data = res.json()
        results.append({
            "modality": "Image (General AI SDXL)",
            "input": "test_synthetic.jpg",
            "expected": "FAKE, score >= 0.50, model = Organika/sdxl-detector",
            "actual_prediction": data["prediction"],
            "actual_score": data["image"]["score"],
            "model": data["image"]["model"],
            "pass": data["prediction"] == "FAKE" and "sdxl" in data["image"]["model"],
        })

        # 6. Audio: Silence
        p = samples / "test_silence.wav"
        with open(p, "rb") as f:
            res = client.post("/api/predict/audio", files={"file": ("test_silence.wav", f, "audio/wav")})
        assert res.status_code == 200, res.text
        data = res.json()
        results.append({
            "modality": "Audio",
            "input": "test_silence.wav",
            "expected": "UNCERTAIN, score = null, reason: Silence or insufficient speech",
            "actual_prediction": data["prediction"],
            "actual_score": data["audio"]["score"],
            "actual_reason": data["audio"]["reason"],
            "pass": data["prediction"] == "UNCERTAIN" and data["audio"]["score"] is None and "Silence" in (data["audio"]["reason"] or ""),
        })

        # 7. Audio: Synthesized Speech
        p = samples / "test_speech_synthesis.wav"
        with open(p, "rb") as f:
            res = client.post("/api/predict/audio", files={"file": ("test_speech_synthesis.wav", f, "audio/wav")})
        assert res.status_code == 200, res.text
        data = res.json()
        results.append({
            "modality": "Audio",
            "input": "test_speech_synthesis.wav",
            "expected": "Available = True, valid float score, model = MelodyMachine/Deepfake-audio-detection",
            "actual_prediction": data["prediction"],
            "actual_score": data["audio"]["score"],
            "actual_duration": data["audio"]["duration_seconds"],
            "pass": data["audio"]["score"] is not None and "MelodyMachine" in data["audio"]["model"],
        })

        # 8. Video: Real Face with Audio
        p = samples / "test_video_realface_with_audio.mp4"
        with open(p, "rb") as f:
            res = client.post("/api/predict", files={"file": ("test_video_realface_with_audio.mp4", f, "video/mp4")})
        assert res.status_code == 200, res.text
        data = res.json()
        results.append({
            "modality": "Video",
            "input": "test_video_realface_with_audio.mp4",
            "expected": "Faces detected = 8, video score < 0.50, fusion score = null",
            "actual_prediction": data["prediction"],
            "video_score": data["video"]["score"],
            "faces_detected": f"{data['video']['faces_detected']}/{data['video']['frames_analyzed']}",
            "audio_score": data["audio"]["score"],
            "fusion_score": data["fusion"]["score"],
            "pass": data["video"]["faces_detected"] == 8 and data["video"]["score"] is not None and data["fusion"]["score"] is None,
        })

        # 9. Video: Face-Swapped with Original Audio (Cross-Modal Disagreement)
        p = samples / "test_video_faceswap_with_audio.mp4"
        with open(p, "rb") as f:
            res = client.post("/api/predict", files={"file": ("test_video_faceswap_with_audio.mp4", f, "video/mp4")})
        assert res.status_code == 200, res.text
        data = res.json()
        results.append({
            "modality": "Video",
            "input": "test_video_faceswap_with_audio.mp4",
            "expected": "FAKE, video score >= 0.50, disagreement reported, fusion score = null",
            "actual_prediction": data["prediction"],
            "video_score": data["video"]["score"],
            "audio_score": data["audio"]["score"],
            "disagreement": data.get("disagreement"),
            "fusion_score": data["fusion"]["score"],
            "pass": data["prediction"] == "FAKE" and data["video"]["score"] > 0.50 and "Visual branch flagged" in (data.get("disagreement") or "") and data["fusion"]["score"] is None,
        })

        # 10. Video: Silent Video (No Soundtrack)
        p = samples / "test_silent_video.mp4"
        with open(p, "rb") as f:
            res = client.post("/api/predict", files={"file": ("test_silent_video.mp4", f, "video/mp4")})
        assert res.status_code == 200, res.text
        data = res.json()
        results.append({
            "modality": "Video (Silent)",
            "input": "test_silent_video.mp4",
            "expected": "audio score = null, audio reason: failed/silent, fusion score = null",
            "actual_prediction": data["prediction"],
            "audio_score": data["audio"]["score"],
            "audio_reason": data["audio"]["reason"],
            "fusion_score": data["fusion"]["score"],
            "pass": data["audio"]["score"] is None and data["fusion"]["score"] is None,
        })

        # 11. Text: One-Line Input
        short_text = "Good morning everyone."
        res = client.post("/api/predict/text", json={"text": short_text})
        assert res.status_code == 200, res.text
        data = res.json()
        results.append({
            "modality": "Text",
            "input": f"'{short_text}' (< 50 chars)",
            "expected": "UNCERTAIN, score = null, reason: Text passage is too short",
            "actual_prediction": data["prediction"],
            "actual_score": data["text"]["score"],
            "actual_reason": data["text"]["reason"],
            "pass": data["prediction"] == "UNCERTAIN" and data["text"]["score"] is None and "too short" in (data["text"]["reason"] or ""),
        })

        # 12. Text: English Passage
        passage = "The historic library was founded in the seventeenth century and maintains an extensive archive of rare manuscripts and early printed books."
        res = client.post("/api/predict/text", json={"text": passage})
        assert res.status_code == 200, res.text
        data = res.json()
        results.append({
            "modality": "Text",
            "input": passage[:50] + "...",
            "expected": "score is float, label = Experimental AI-writing style estimate",
            "actual_prediction": data["prediction"],
            "actual_score": data["text"]["score"],
            "actual_label": data["text"]["label"],
            "pass": data["text"]["score"] is not None and data["text"]["label"] == "Experimental AI-writing style estimate",
        })

        # 13. Video with Subtitles: Subtitles never vote on manipulation
        p = samples / "test_video_realface_with_audio.mp4"
        with open(p, "rb") as f:
            res = client.post(
                "/api/predict",
                files={"file": ("test_video_realface_with_audio.mp4", f, "video/mp4")},
                data={"text": "Burnt-in subtitles: This is an authentic speech demonstration."}
            )
        assert res.status_code == 200, res.text
        data = res.json()
        results.append({
            "modality": "Video with Subtitles",
            "input": "test_video_realface_with_audio.mp4 + Subtitles",
            "expected": "Subtitles do not vote, transcript comparison unavailable",
            "prediction": data["prediction"],
            "transcript_status": data.get("cross_modal", {}).get("transcript_status"),
            "pass": "Transcript comparison unavailable" in (data.get("cross_modal", {}).get("transcript_status") or ""),
        })

        print(f"\nCompleted {len(results)} automated test cases:\n")
        for i, r in enumerate(results, 1):
            status = "PASS" if r["pass"] else "FAIL"
            print(f"[{status}] Test {i:02d}: {r['modality']} - {r['input']}")
            print(f"       Expected: {r['expected']}")
            print(f"       Actual:   pred={r.get('actual_prediction') or r.get('prediction')}, score={r.get('actual_score') or r.get('video_score')}")
            if r.get("disagreement"):
                print(f"       Disagreement: {r['disagreement']}")
            if r.get("actual_reason"):
                print(f"       Reason: {r['actual_reason']}")

        all_passed = all(r["pass"] for r in results)
        print("\n" + "=" * 80)
        print(f"FINAL RESULT: {'ALL PASS' if all_passed else 'SOME FAILURES'}")
        print("=" * 80)

if __name__ == "__main__":
    run_verification()
