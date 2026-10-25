from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import ml_adapter
from app import main
from app.config import Settings
from nids_ml import ModelUnavailableError


def test_active_model_load_uses_registry_checksum(monkeypatch):
    record = {"version": "baseline-1", "artifact_sha256": "a" * 64}
    db = SimpleNamespace(
        model_registry=SimpleNamespace(find_one=lambda query, projection: record)
    )
    loaded = SimpleNamespace(version="baseline-1")
    calls = []

    def fake_load_bundle(artifact_dir, version, *, expected_sha256):
        calls.append((artifact_dir, version, expected_sha256))
        return loaded

    monkeypatch.setattr(ml_adapter, "load_bundle", fake_load_bundle)

    predictor = ml_adapter.load_active_predictor(db, "models", "baseline-1")

    assert predictor.bundle is loaded
    assert calls == [("models", "baseline-1", "a" * 64)]


def test_missing_registry_checksum_prevents_model_loading(monkeypatch):
    db = SimpleNamespace(
        model_registry=SimpleNamespace(
            find_one=lambda query, projection: {"version": "baseline-1"}
        )
    )
    monkeypatch.setattr(
        ml_adapter,
        "load_bundle",
        lambda *args, **kwargs: pytest.fail("must not load without a trusted checksum"),
    )

    with pytest.raises(ModelUnavailableError):
        ml_adapter.load_active_predictor(db, "models", "baseline-1")


def test_predictor_preserves_ml_prediction_results(monkeypatch):
    result = object()
    bundle = object()
    calls = []

    def fake_predict(records, passed_bundle):
        calls.append((records, passed_bundle))
        return [result]

    monkeypatch.setattr(ml_adapter, "predict", fake_predict)
    predictor = ml_adapter.ModelPredictor(bundle)

    assert predictor.predict([{"protocol": "tcp"}]) == [result]
    assert calls == [([{"protocol": "tcp"}], bundle)]


@pytest.mark.parametrize(
    ("load_error", "allow_degraded", "expected_status", "expected_model_status"),
    [
        (None, True, 200, "ok"),
        (ModelUnavailableError("no verified model"), True, 200, "degraded"),
        (ModelUnavailableError("no verified model"), False, 503, None),
    ],
)
def test_readiness_reflects_verified_model_state(
    monkeypatch, load_error, allow_degraded, expected_status, expected_model_status
):
    class FakeCollection:
        def create_index(self, *args, **kwargs):
            pass

    class FakeDatabase:
        revoked_tokens = FakeCollection()

    class FakeMongoClient:
        admin = SimpleNamespace(command=lambda command: None)

        def __getitem__(self, name):
            return FakeDatabase()

    def load_predictor(*args):
        if load_error is not None:
            raise load_error
        return object()

    monkeypatch.setattr(main, "ensure_indexes", lambda db: None)
    monkeypatch.setattr(main, "load_active_predictor", load_predictor)
    app = main.create_app(
        Settings(
            jwt_secret_key="x" * 32,
            active_model_version="baseline-1",
            allow_degraded_model=allow_degraded,
        ),
        mongo_client=FakeMongoClient(),
    )

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == expected_status
    if expected_model_status is not None:
        assert response.json()["dependencies"]["model"] == expected_model_status
