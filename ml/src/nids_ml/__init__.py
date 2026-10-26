"""nids_ml - offline, deterministic ML engine for the Adaptive AI-Driven NIDS.

Public interface (docs/ML_CONTRACT.md section 2): validate_features, preprocess, predict, evaluate,
validate_candidate. Advisory detection only: nothing in this package touches a network, a firewall,
a database or a live capture.
"""
__version__ = "0.1.0"

from .contracts import (  # noqa: E402
    BENIGN_LABEL, REASON_CODES, SCHEMA_VERSION, UNKNOWN_LABEL, ArtifactIntegrityError, ConfigError,
    EvaluationConfig, EvaluationReport, FeatureValidationReport, InputValidationError, LifecycleError,
    ModelLoadError, ModelUnavailableError, PredictionOptions, PredictionResult, PromotionGates,
    SchemaVersionError, ValidationResult, severity_for_score,
)
from .artifacts import ModelBundle, load_bundle, save_bundle  # noqa: E402
from .evaluation import evaluate  # noqa: E402
from .features import validate_features  # noqa: E402
from .inference import predict  # noqa: E402
from .lifecycle import validate_candidate  # noqa: E402
from .preprocessing import preprocess  # noqa: E402

__all__ = [
    "validate_features", "preprocess", "predict", "evaluate", "validate_candidate",
    "load_bundle", "save_bundle", "ModelBundle", "PredictionResult", "PredictionOptions", "EvaluationConfig",
    "EvaluationReport", "PromotionGates", "ValidationResult", "FeatureValidationReport", "severity_for_score",
    "SCHEMA_VERSION", "REASON_CODES", "BENIGN_LABEL", "UNKNOWN_LABEL",
    "InputValidationError", "SchemaVersionError", "ModelUnavailableError", "ModelLoadError",
    "ArtifactIntegrityError", "ConfigError", "LifecycleError",
]
