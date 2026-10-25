from typing import Any

from nids_ml import (
    ModelBundle,
    ModelUnavailableError,
    PredictionResult,
    load_bundle,
    predict,
)


class ModelPredictor:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    def predict(self, records: Any) -> list[PredictionResult]:
        return predict(records, self.bundle)


def load_active_predictor(db: Any, artifact_dir: str, version: str) -> ModelPredictor:
    if version == "unavailable":
        raise ModelUnavailableError("no active model version is configured")

    registry_record = db.model_registry.find_one(
        {"version": version, "status": "active"},
        {"version": 1, "artifact_sha256": 1},
    )
    if registry_record is None:
        raise ModelUnavailableError("active model version is missing from the registry")

    checksum = registry_record.get("artifact_sha256")
    if not isinstance(checksum, str):
        raise ModelUnavailableError("active model registry record has no artifact checksum")

    bundle = load_bundle(artifact_dir, version, expected_sha256=checksum)
    return ModelPredictor(bundle)
