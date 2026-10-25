from fastapi.testclient import TestClient
from app.main import create_app
from app.config import Settings

def test_health_is_public():
    app = create_app(Settings(jwt_secret_key="x"*32, mongodb_uri=""))
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "adaptive-ai-nids-api"
    assert response.headers.get("x-request-id")

def test_validation_uses_contract_error_envelope():
    app = create_app(Settings(jwt_secret_key="x"*32, mongodb_uri=""))
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/login", json={"email":"not-an-email","password":"x"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "request_id" in response.json()["error"]
