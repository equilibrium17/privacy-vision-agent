import base64
from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)

# 1x1 transparent PNG data URL for testing image decoding
TEST_IMAGE_B64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "groq_configured" in data
    assert "ollama_host" in data


def test_providers_endpoint():
    response = client.get("/providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert len(data["providers"]) >= 2
    provider_ids = [p["id"] for p in data["providers"]]
    assert "groq" in provider_ids
    assert "ollama" in provider_ids


def test_analyze_invalid_image():
    response = client.post("/analyze", json={
        "image": "not-a-data-url",
        "pii_fields": [],
        "task": "Test task",
        "provider": "groq"
    })
    assert response.status_code == 400
    assert "base64 data URL" in response.json()["detail"]


def test_analyze_oversized_image_rejected():
    # Construct base64 string exceeding 14MB (~10MB decoded limit)
    huge_b64 = "data:image/png;base64," + ("A" * (15 * 1024 * 1024))
    response = client.post("/analyze", json={
        "image": huge_b64,
        "pii_fields": [],
        "task": "Oversized image test",
        "provider": "groq"
    })
    assert response.status_code in [413, 400]
    assert "Fail-Closed Error" in response.json()["detail"]


def test_analyze_ollama_fallback():
    # Test offline Ollama provider request format
    response = client.post("/analyze", json={
        "image": TEST_IMAGE_B64,
        "pii_fields": [
            {
                "selector": "#email",
                "rect": {"x": 10, "y": 20, "width": 100, "height": 30},
                "reason": "Email PII"
            }
        ],
        "task": "Fill email with user email",
        "page_context": [
            {
                "selector": "#email",
                "tag": "input",
                "type": "email",
                "placeholder": "Enter your email"
            }
        ],
        "provider": "ollama"
    })
    assert response.status_code == 200
    data = response.json()
    assert "reasoning" in data
    assert "actions" in data

