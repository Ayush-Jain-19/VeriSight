"""Test FastAPI live startup and prediction."""

from fastapi.testclient import TestClient
from src.api.main import create_app
import json

def test():
    print("[SERVER TEST] Initializing FastAPI app with lifespan...")
    app = create_app()
    with TestClient(app) as client:
        print("[SERVER TEST] Calling GET /api/health...")
        health = client.get("/api/health")
        print("Health status code:", health.status_code)
        print("Health JSON:", json.dumps(health.json(), indent=2))

        print("\n[SERVER TEST] Calling POST /api/predict with demo_samples/test_pattern_with_audio.mp4...")
        with open("demo_samples/test_pattern_with_audio.mp4", "rb") as f:
            res = client.post("/api/predict", files={"file": ("test.mp4", f, "video/mp4")})
        print("Predict status code:", res.status_code)
        print("Predict JSON:", json.dumps(res.json(), indent=2))

if __name__ == "__main__":
    test()
