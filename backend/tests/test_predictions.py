import asyncio
from types import SimpleNamespace

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from nids_ml import PredictionResult, validate_features
from nids_ml.contracts import ValidationIssue

from app import main
from app.config import Settings, get_settings
from app.security import create_access_token


def flow_record():
    return {
        "duration": 0.5,
        "protocol": "tcp",
        "source_port": 12345,
        "destination_port": 443,
        "flow_packets": 10,
        "flow_bytes": 500,
        "packets_per_second": 2,
        "bytes_per_second": 100,
        "connection_state": "established",
        "direction": "outbound",
        "observed_at": "2026-10-25T12:00:00Z",
    }


def prediction_result():
    return PredictionResult(
        row_index=0,
        prediction_label="port_scan",
        prediction_type="known_attack",
        confidence=0.91,
        anomaly_score=0.73,
        risk_score=76,
        severity="high",
        reason_codes=["KNOWN_CLASS_HIGH_SCORE", "ANOMALY_ELEVATED"],
        model_version="baseline-1",
        feature_schema_version="1.0.0",
        prediction_timestamp="2026-10-25T12:00:00Z",
    )


class FakeDatabase:
    def __init__(self, user_active=True):
        self.revoked_tokens = SimpleNamespace(
            create_index=lambda *args, **kwargs: None,
            find_one=lambda query: None,
        )
        self.users = SimpleNamespace(
            find_one=lambda query, projection=None: (
                {"is_active": True} if user_active else None
            )
        )


class FakeMongoClient:
    def __init__(self, user_active=True):
        self.admin = SimpleNamespace(command=lambda command: None)
        self.database = FakeDatabase(user_active)

    def __getitem__(self, name):
        return self.database


def make_client(
    monkeypatch,
    predictor=None,
    load_error=None,
    max_upload_mb=1,
    user_active=True,
):
    settings = Settings(
        jwt_secret_key="s" * 32,
        active_model_version="baseline-1",
        max_upload_mb=max_upload_mb,
    )

    def load_predictor(*args):
        if load_error is not None:
            raise load_error
        return predictor

    monkeypatch.setattr(main, "ensure_indexes", lambda db: None)
    monkeypatch.setattr(main, "load_active_predictor", load_predictor)
    app = main.create_app(
        settings,
        mongo_client=FakeMongoClient(user_active=user_active),
    )
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)
    token, _ = create_access_token(str(ObjectId()), "analyst", settings)
    return client, token


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def test_authenticated_prediction_calls_predictor_and_preserves_contract(monkeypatch):
    records = [flow_record()]
    result = prediction_result()
    calls = []

    class Predictor:
        def predict(self, observations):
            calls.append(observations)
            return [result]

    client, token = make_client(monkeypatch, predictor=Predictor())

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": records},
            headers=auth_headers(token),
        )

    assert response.status_code == 200
    assert calls == [records]
    assert len(response.json()) == 1
    assert set(response.json()[0]) == set(PredictionResult.model_fields)
    assert response.json()[0] == result.model_dump()


def test_prediction_requires_authentication(monkeypatch):
    client, _ = make_client(monkeypatch, predictor=SimpleNamespace(predict=lambda records: []))

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": [flow_record()]},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"


def test_inactive_user_cannot_use_prediction_endpoint(monkeypatch):
    client, token = make_client(
        monkeypatch,
        predictor=SimpleNamespace(predict=lambda records: [prediction_result()]),
        user_active=False,
    )

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": [flow_record()]},
            headers=auth_headers(token),
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"


def test_invalid_features_use_ml_validation_and_error_envelope(monkeypatch):
    class ValidatingPredictor:
        def predict(self, observations):
            report = validate_features(observations)
            report.raise_if_invalid()
            return [prediction_result()]

    client, token = make_client(monkeypatch, predictor=ValidatingPredictor())

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": [{"protocol": "tcp"}]},
            headers=auth_headers(token),
        )

    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == "VALIDATION_ERROR"
    assert body["details"]
    assert all("issue" in detail for detail in body["details"])


@pytest.mark.parametrize("observations", [[], [flow_record()] * 1001])
def test_empty_and_oversized_batches_are_rejected(monkeypatch, observations):
    client, token = make_client(monkeypatch, predictor=SimpleNamespace(predict=lambda records: []))

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": observations},
            headers=auth_headers(token),
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_missing_predictor_returns_service_unavailable(monkeypatch):
    from nids_ml import ModelUnavailableError

    client, token = make_client(
        monkeypatch,
        load_error=ModelUnavailableError("model not available"),
    )

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": [flow_record()]},
            headers=auth_headers(token),
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"
    assert response.json()["error"]["details"] is None


def test_ml_input_validation_error_is_mapped_to_validation_envelope(monkeypatch):
    from nids_ml import InputValidationError

    class InvalidPredictor:
        def predict(self, observations):
            raise InputValidationError(
                "invalid flow",
                [ValidationIssue(row_index=0, field="duration", code="NOT_NUMERIC", message="expected a number")],
            )

    client, token = make_client(monkeypatch, predictor=InvalidPredictor())

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": [flow_record()]},
            headers=auth_headers(token),
        )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["details"] == [{
        "field": "observations.0.duration",
        "issue": "expected a number",
    }]


def test_unexpected_inference_failure_does_not_leak_details(monkeypatch):
    class BrokenPredictor:
        def predict(self, observations):
            raise RuntimeError("secret path C:\\private\\model and raw flow data")

    client, token = make_client(monkeypatch, predictor=BrokenPredictor())

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": [flow_record()]},
            headers=auth_headers(token),
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"
    assert "secret path" not in response.text
    assert "raw flow data" not in response.text


def test_prediction_request_body_limit_uses_error_envelope(monkeypatch):
    client, token = make_client(
        monkeypatch,
        predictor=SimpleNamespace(predict=lambda records: [prediction_result()]),
        max_upload_mb=1,
    )

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": [flow_record()], "padding": "x" * (1024 * 1024)},
            headers=auth_headers(token),
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
    assert response.headers.get("x-request-id")
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


def test_prediction_openapi_documents_auth_schemas_and_error_responses():
    schema = main.create_app(Settings(jwt_secret_key="s" * 32)).openapi()
    operation = schema["paths"]["/api/v1/predictions"]["post"]
    request_schema = schema["components"]["schemas"]["PredictionRequest"]
    prediction_schema = schema["components"]["schemas"]["PredictionResult"]

    assert operation["security"] == [{"HTTPBearer": []}]
    assert operation["requestBody"]["content"]["application/json"]["schema"]["$ref"] == (
        "#/components/schemas/PredictionRequest"
    )
    assert request_schema["properties"]["observations"]["maxItems"] == 1000
    assert request_schema["properties"]["observations"]["minItems"] == 1
    request_description = request_schema["properties"]["observations"]["description"]
    assert "duration" in request_description
    assert "observed_at" in request_description
    assert "source_id is optional" in request_description
    assert set(prediction_schema["properties"]) == set(PredictionResult.model_fields)
    for status_code in ("401", "413", "422", "503"):
        assert operation["responses"][status_code]["content"]["application/json"]["schema"]["$ref"] == (
            "#/components/schemas/ErrorEnvelope"
        )


def test_streaming_body_limit_rejects_chunked_requests_without_content_length():
    async def exercise():
        pending = [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"456", "more_body": False},
        ]
        sent = []

        async def receive():
            return pending.pop(0)

        async def send(message):
            sent.append(message)

        async def consume_body(scope, receive, send):
            while (await receive()).get("more_body", False):
                pass

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/predictions",
            "raw_path": b"/api/v1/predictions",
            "query_string": b"",
            "headers": [],
            "server": ("test", 80),
            "client": ("test", 80),
            "root_path": "",
            "state": {"request_id": "req-test"},
        }

        middleware = main.PredictionRequestSizeLimitMiddleware(
            consume_body,
            path="/api/v1/predictions",
            max_bytes=5,
        )
        await middleware(scope, receive, send)
        status = next(
            message["status"]
            for message in sent
            if message["type"] == "http.response.start"
        )
        body = b"".join(
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        )
        return status, body

    status, body = asyncio.run(exercise())

    assert status == 413
    assert b'"code":"PAYLOAD_TOO_LARGE"' in body
