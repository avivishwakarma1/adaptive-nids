import asyncio
from types import SimpleNamespace

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from nids_ml import PredictionResult, validate_features
from nids_ml.contracts import ValidationIssue
from pymongo.errors import ConfigurationError, OperationFailure, PyMongoError

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


def prediction_result(
    row_index=0,
    prediction_type="known_attack",
    risk_score=76,
    severity="high",
):
    return PredictionResult(
        row_index=row_index,
        prediction_label=(
            {
                "known_attack": "port_scan",
                "benign": "benign",
                "anomaly": "unknown",
                "uncertain": "unknown",
            }[prediction_type]
        ),
        prediction_type=prediction_type,
        confidence=0.91,
        anomaly_score=0.73,
        risk_score=risk_score,
        severity=severity,
        reason_codes=["KNOWN_CLASS_HIGH_SCORE", "ANOMALY_ELEVATED"],
        model_version="baseline-1",
        feature_schema_version="1.0.0",
        prediction_timestamp="2026-10-25T12:00:00Z",
    )


class FakeWriteCollection:
    def __init__(self):
        self.records = []
        self.fail_insert = False
        self.fail_transaction = False
        self.fail_delete = False

    def create_index(self, *args, **kwargs):
        return None

    def insert_many(self, records, **kwargs):
        if self.fail_insert:
            raise PyMongoError("simulated database failure")
        if kwargs.get("session") is not None and self.fail_transaction:
            raise OperationFailure(
                "transactions are not supported",
                code=20,
            )
        self.records.extend(records)

    def delete_many(self, query):
        if self.fail_delete:
            raise PyMongoError("simulated cleanup failure")
        batch_id = query["prediction_batch_id"]
        self.records = [
            record for record in self.records
            if record.get("prediction_batch_id") != batch_id
        ]

    def count_documents(self, query):
        batch_id = query["prediction_batch_id"]
        return sum(
            record.get("prediction_batch_id") == batch_id
            for record in self.records
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
        self.standalone_predictions = FakeWriteCollection()
        self.alerts = FakeWriteCollection()
        self.client = None


class FakeTransaction:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeSession:
    def __init__(self, mongo_client):
        self.mongo_client = mongo_client

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def start_transaction(self):
        self.mongo_client.transaction_started = True
        return FakeTransaction()


class FakeMongoClient:
    def __init__(self, user_active=True, transaction_supported=False):
        self.admin = SimpleNamespace(command=lambda command: None)
        self.database = FakeDatabase(user_active)
        self.database.client = self
        self.transaction_supported = transaction_supported
        self.transaction_started = False

    def __getitem__(self, name):
        return self.database

    def start_session(self):
        if not self.transaction_supported:
            raise ConfigurationError("transactions are unavailable")
        return FakeSession(self)


def make_client(
    monkeypatch,
    predictor=None,
    load_error=None,
    max_upload_mb=1,
    user_active=True,
    database=None,
    transaction_supported=False,
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

    mongo_client = FakeMongoClient(
        user_active=user_active,
        transaction_supported=transaction_supported,
    )
    if database is not None:
        mongo_client.database = database
        mongo_client.database.client = mongo_client

    monkeypatch.setattr(main, "ensure_indexes", lambda db: None)
    monkeypatch.setattr(main, "load_active_predictor", load_predictor)
    app = main.create_app(
        settings,
        mongo_client=mongo_client,
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
    assert len(client.app.state.db.standalone_predictions.records) == 1
    assert len(client.app.state.db.alerts.records) == 1


def test_predictions_persist_with_utc_timestamps_and_row_mapping(monkeypatch):
    observations = [flow_record(), flow_record()]
    observations[0]["observed_at"] = "2026-10-25T12:00:00Z"
    observations[0]["source_id"] = "source-a"
    observations[1]["observed_at"] = "2026-10-25T17:30:00+05:30"
    results = [
        prediction_result(
            row_index=1,
            prediction_type="benign",
            risk_score=8,
            severity="info",
        ),
        prediction_result(row_index=0),
    ]
    predictor = SimpleNamespace(predict=lambda records: results)
    client, token = make_client(monkeypatch, predictor=predictor)

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": observations},
            headers=auth_headers(token),
        )

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json() == [result.model_dump() for result in results]
    assert set(response.json()[0]) == set(PredictionResult.model_fields)

    stored = sorted(
        client.app.state.db.standalone_predictions.records,
        key=lambda record: record["row_index"],
    )
    assert [record["row_index"] for record in stored] == [0, 1]
    assert [record["prediction_type"] for record in stored] == [
        "known_attack",
        "benign",
    ]
    assert [record["observed_at"].isoformat() for record in stored] == [
        "2026-10-25T12:00:00+00:00",
        "2026-10-25T12:00:00+00:00",
    ]
    assert all(
        record["prediction_timestamp"].tzinfo is not None
        for record in stored
    )
    assert all(
        record["created_at"].tzinfo is not None
        and record["updated_at"].tzinfo is not None
        for record in stored
    )
    assert len({record["prediction_batch_id"] for record in stored}) == 1
    assert "source_id" not in stored[0]
    assert len(client.app.state.db.alerts.records) == 1
    assert client.app.state.db.alerts.records[0]["row_index"] == 0


def test_alerts_only_include_known_attacks_over_conservative_threshold(monkeypatch):
    risk_cases = [
        ("known_attack", 60, "high"),
        ("known_attack", 80, "critical"),
        ("known_attack", 59, "medium"),
        ("benign", 99, "critical"),
        ("anomaly", 80, "critical"),
        ("uncertain", 80, "critical"),
    ]
    observations = [flow_record() for _ in risk_cases]
    results = [
        prediction_result(
            row_index=index,
            prediction_type=prediction_type,
            risk_score=risk_score,
            severity=severity,
        )
        for index, (prediction_type, risk_score, severity) in enumerate(risk_cases)
    ]
    client, token = make_client(
        monkeypatch,
        predictor=SimpleNamespace(predict=lambda records: results),
    )

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": observations},
            headers=auth_headers(token),
        )

    assert response.status_code == 200
    alerts = client.app.state.db.alerts.records
    assert [alert["row_index"] for alert in alerts] == [0, 1]
    assert [alert["status"] for alert in alerts] == ["open", "open"]
    assert [alert["severity"] for alert in alerts] == ["high", "critical"]


@pytest.mark.parametrize(
    "row_indices",
    [
        [0, 0],
        [0, 2],
        [1],
    ],
)
def test_invalid_result_mappings_are_rejected_before_persistence(
    monkeypatch, row_indices
):
    observations = [flow_record(), flow_record()]
    results = [
        prediction_result(row_index=index)
        for index in row_indices
    ]
    database = FakeDatabase()
    client, token = make_client(
        monkeypatch,
        predictor=SimpleNamespace(predict=lambda records: results),
        database=database,
    )

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": observations},
            headers=auth_headers(token),
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "INVALID_PREDICTION_OUTPUT"
    assert "observation" not in response.text
    assert database.standalone_predictions.records == []
    assert database.alerts.records == []


def test_prediction_persistence_uses_transaction_when_supported(monkeypatch):
    client, token = make_client(
        monkeypatch,
        predictor=SimpleNamespace(
            predict=lambda records: [prediction_result()]
        ),
        transaction_supported=True,
    )

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": [flow_record()]},
            headers=auth_headers(token),
        )

    assert response.status_code == 200
    assert client.app.state.mongo_client.transaction_started


def test_standalone_mongo_transaction_rejection_falls_back_safely(monkeypatch):
    database = FakeDatabase()
    database.standalone_predictions.fail_transaction = True
    client, token = make_client(
        monkeypatch,
        predictor=SimpleNamespace(
            predict=lambda records: [prediction_result()]
        ),
        database=database,
        transaction_supported=True,
    )

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": [flow_record()]},
            headers=auth_headers(token),
        )

    assert response.status_code == 200
    assert len(database.standalone_predictions.records) == 1
    assert len(database.alerts.records) == 1


def test_persistence_failure_returns_controlled_error_and_cleans_partial_writes(
    monkeypatch,
):
    database = FakeDatabase()
    database.alerts.fail_insert = True
    client, token = make_client(
        monkeypatch,
        predictor=SimpleNamespace(
            predict=lambda records: [prediction_result()]
        ),
        database=database,
    )

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": [flow_record()]},
            headers=auth_headers(token),
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "PERSISTENCE_UNAVAILABLE"
    assert "simulated database failure" not in response.text
    assert database.standalone_predictions.records == []
    assert database.alerts.records == []


def test_cleanup_failure_still_returns_controlled_persistence_error(monkeypatch):
    database = FakeDatabase()
    database.alerts.fail_insert = True
    database.standalone_predictions.fail_delete = True
    client, token = make_client(
        monkeypatch,
        predictor=SimpleNamespace(
            predict=lambda records: [prediction_result()]
        ),
        database=database,
    )

    with client:
        response = client.post(
            "/api/v1/predictions",
            json={"observations": [flow_record()]},
            headers=auth_headers(token),
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "PERSISTENCE_UNAVAILABLE"
    assert "simulated" not in response.text
    assert len(database.standalone_predictions.records) == 1


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
