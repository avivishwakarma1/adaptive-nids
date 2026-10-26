"""Public contracts for the NIDS ML package (ML_CONTRACT.md v1.0.0).

Everything here mirrors docs/ML_CONTRACT.md. Nothing in this module adds or renames
a field of the *prediction output* (section 5 of the contract). Types that the
contract leaves unspecified (evaluation report, promotion gates, validation result)
are defined here as ML-internal types and are listed in README "Open questions".
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Constants (contract sections 3, 5, 7)
# --------------------------------------------------------------------------- #
SCHEMA_VERSION = "1.0.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})
DEFAULT_MAX_BATCH_SIZE = 1000  # INTEGRATION_GUIDE.md MAX_BATCH_SIZE default

BENIGN_LABEL = "benign"
UNKNOWN_LABEL = "unknown"

PROTOCOLS = ("tcp", "udp", "icmp", "other", "unknown")
DIRECTIONS = ("inbound", "outbound", "internal", "unknown")
PREDICTION_TYPES = ("known_attack", "benign", "anomaly", "uncertain")
SEVERITIES = ("critical", "high", "medium", "low", "info")
MODEL_KINDS = ("classifier_plus_anomaly", "classifier_only", "anomaly_only")

REASON_CODES = (
    "KNOWN_CLASS_HIGH_SCORE",
    "KNOWN_CLASS_LOW_CONFIDENCE",
    "ANOMALY_ELEVATED",
    "ANOMALY_HIGH",
    "FEATURES_INCOMPLETE",
    "OUT_OF_DISTRIBUTION_WARNING",
    "MODEL_COMPONENT_UNAVAILABLE",
    "RISK_POLICY_HIGH",
)

LABEL_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def normalize_label(raw: Any) -> str:
    """Normalize a class label to lowercase snake_case (contract section 5)."""
    if not isinstance(raw, str):
        raise ValueError("label must be a string")
    s = re.sub(r"[^0-9a-zA-Z]+", "_", raw.strip()).strip("_").lower()
    if not s or not LABEL_RE.match(s):
        raise ValueError(f"label {raw!r} cannot be normalized to snake_case")
    return s


def severity_for_score(score: int) -> str:
    """Required final mapping (contract section 7). The backend must not re-derive this."""
    if isinstance(score, bool) or not isinstance(score, int):
        raise TypeError("risk score must be an int")
    s = max(0, min(100, score))
    if s >= 80:
        return "critical"
    if s >= 60:
        return "high"
    if s >= 40:
        return "medium"
    if s >= 20:
        return "low"
    return "info"


def utc_now_iso(now: Optional[datetime] = None) -> str:
    dt = now or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        raise ValueError("clock must return timezone-aware datetimes")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Errors (all carry a stable machine code; none include raw input values)
# --------------------------------------------------------------------------- #
class NidsMLError(Exception):
    code = "ML_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    row_index: Optional[int] = None
    field: Optional[str] = None
    code: str
    message: str


class InputValidationError(NidsMLError):
    code = "VALIDATION_ERROR"

    def __init__(self, message: str, issues: Optional[list[ValidationIssue]] = None):
        super().__init__(message)
        self.issues = issues or []


class SchemaVersionError(NidsMLError):
    code = "SCHEMA_VERSION_MISMATCH"


class ModelUnavailableError(NidsMLError):
    """Maps to API error MODEL_UNAVAILABLE (503). Never swallowed into a 'benign' prediction."""
    code = "MODEL_UNAVAILABLE"


class ModelLoadError(ModelUnavailableError):
    code = "MODEL_UNAVAILABLE"


class ArtifactIntegrityError(ModelLoadError):
    code = "ARTIFACT_INTEGRITY_ERROR"


class ConfigError(NidsMLError):
    code = "CONFIG_ERROR"


class LifecycleError(NidsMLError):
    """Invalid state transition / blocked promotion. Backend maps to 409."""
    code = "LIFECYCLE_CONFLICT"


# --------------------------------------------------------------------------- #
# Feature validation result
# --------------------------------------------------------------------------- #
class FeatureValidationReport(BaseModel):
    """Result of `validate_features`.

    ``records`` is aligned with the input; entries for invalid rows are ``None``.
    Batch-level problems (empty, too large, wrong schema version, wrong container
    type) raise instead of being reported here.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    schema_version: str
    n_records: int
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    issues_truncated: bool = False
    records: list[Optional[dict]] = Field(default_factory=list, exclude=True)

    @property
    def invalid_row_indices(self) -> list[int]:
        return sorted({i.row_index for i in self.issues if i.row_index is not None})

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise InputValidationError(
                f"{len(self.invalid_row_indices)} invalid row(s) in batch of {self.n_records}",
                self.issues,
            )


# --------------------------------------------------------------------------- #
# Prediction output (contract section 5) - exactly these 11 fields
# --------------------------------------------------------------------------- #
class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    row_index: int = Field(ge=0)
    prediction_label: str
    prediction_type: Literal["known_attack", "benign", "anomaly", "uncertain"]
    confidence: Optional[float] = None
    anomaly_score: Optional[float] = None
    risk_score: int = Field(ge=0, le=100)
    severity: Literal["critical", "high", "medium", "low", "info"]
    reason_codes: list[str] = Field(default_factory=list)
    model_version: str = Field(min_length=1)
    feature_schema_version: str = Field(min_length=1)
    prediction_timestamp: str

    @field_validator("confidence", "anomaly_score")
    @classmethod
    def _unit_interval(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if not math.isfinite(v) or v < 0.0 or v > 1.0:
            raise ValueError("score must be finite and within [0, 1]")
        return float(v)

    @field_validator("prediction_label")
    @classmethod
    def _label(cls, v: str) -> str:
        if not LABEL_RE.match(v):
            raise ValueError("prediction_label must be lowercase snake_case")
        return v

    @field_validator("reason_codes")
    @classmethod
    def _codes(cls, v: list[str]) -> list[str]:
        bad = [c for c in v if c not in REASON_CODES]
        if bad:
            raise ValueError(f"reason codes outside allowlist: {bad}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate reason codes")
        return v

    @field_validator("prediction_timestamp")
    @classmethod
    def _ts(cls, v: str) -> str:
        if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", v):
            raise ValueError("prediction_timestamp must be UTC ISO 8601 ending in Z")
        datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> "PredictionResult":
        if self.severity != severity_for_score(self.risk_score):
            raise ValueError("severity does not match the contract's risk-score mapping")
        t, lab = self.prediction_type, self.prediction_label
        if t == "benign" and lab != BENIGN_LABEL:
            raise ValueError("benign predictions must carry label 'benign'")
        if t == "known_attack" and lab in (BENIGN_LABEL, UNKNOWN_LABEL):
            raise ValueError("known_attack requires a known attack class label")
        if t in ("anomaly", "uncertain") and lab != UNKNOWN_LABEL:
            raise ValueError("anomaly/uncertain predictions must carry label 'unknown'")
        return self


# --------------------------------------------------------------------------- #
# Prediction options (ML-internal; not part of the output contract)
# --------------------------------------------------------------------------- #
class PredictionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    max_batch_size: int = Field(default=DEFAULT_MAX_BATCH_SIZE, ge=1)
    extra_fields: Literal["reject", "strip"] = "reject"
    # Contract section 5: on component failure return a controlled failure OR explicit uncertain.
    component_failure_policy: Literal["fail", "uncertain"] = "fail"
    # Injectable clock for deterministic tests. Must return a timezone-aware datetime.
    clock: Optional[Callable[[], datetime]] = Field(default=None, exclude=True)


# --------------------------------------------------------------------------- #
# Evaluation / lifecycle types (ML-internal; contract leaves shape to implementation)
# --------------------------------------------------------------------------- #
class LabeledObservation(BaseModel):
    """One labelled evaluation row. Labels are kept apart from features (contract section 3)."""
    model_config = ConfigDict(extra="forbid")
    observation: dict[str, Any]
    label: str


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    dataset_name: str = "unspecified"
    dataset_fingerprint: Optional[str] = None  # computed from the records when omitted
    split_strategy: str = Field(
        default="unspecified",
        description="How the caller obtained these records (e.g. 'stratified_holdout').",
    )
    split_metadata: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=DEFAULT_MAX_BATCH_SIZE, ge=1)
    calibration_bins: int = Field(default=10, ge=2, le=50)
    min_class_support_warning: int = Field(default=30, ge=1)
    clock: Optional[Callable[[], datetime]] = Field(default=None, exclude=True)


class ClassMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    support: int
    false_positive_rate: Optional[float]


class EvaluationReport(BaseModel):
    """Measured results for one bundle on one labelled set. Every metric is ``None`` when
    undefined; nothing here is ever a placeholder (DATABASE_SCHEMA 2.9)."""
    model_config = ConfigDict(extra="forbid")

    evaluated: bool = True
    model_version: str
    artifact_sha256: Optional[str]
    feature_schema_version: str
    dataset_name: str
    dataset_fingerprint: str
    split_strategy: str
    split_metadata: dict[str, Any]
    n_samples: int
    class_support: dict[str, int]
    thresholds: dict[str, float]
    risk_policy_version: str
    package_versions: dict[str, str]
    evaluated_at: str

    # Names follow DATABASE_SCHEMA 2.9 `metrics`.
    metrics: dict[str, Optional[float]]
    metric_definitions: dict[str, str]
    class_metrics: list[ClassMetrics]
    confusion_labels_true: list[str]
    confusion_labels_pred: list[str]
    confusion_matrix: list[list[int]]
    type_by_truth: dict[str, dict[str, int]]
    alert_rate: Optional[float]
    anomaly_metrics: Optional[dict[str, Any]] = None
    classifier_closed_set: Optional[dict[str, Any]] = None
    calibration: Optional[dict[str, Any]] = None
    warnings: list[str] = Field(default_factory=list)


class PromotionGates(BaseModel):
    """Explicit, configured quality gates (contract section 10). ``None`` = gate not enforced.

    ``minimum_recall`` and ``maximum_false_positive_rate`` are the two gate names present in
    API_CONTRACT (POST /models/{version}/validate). The remaining fields are ML-internal
    extensions needed for the regression checks the contract asks for.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    minimum_recall: Optional[float] = Field(default=None, ge=0, le=1)
    maximum_false_positive_rate: Optional[float] = Field(default=None, ge=0, le=1)
    minimum_macro_f1: Optional[float] = Field(default=None, ge=0, le=1)
    minimum_per_class_recall: Optional[float] = Field(default=None, ge=0, le=1)
    minimum_samples: Optional[int] = Field(default=None, ge=1)
    # Regression gates against the previous model, scored on the SAME evaluation set.
    maximum_macro_f1_drop: Optional[float] = Field(default=None, ge=0, le=1)
    maximum_per_class_recall_drop: Optional[float] = Field(default=None, ge=0, le=1)
    maximum_false_positive_rate_increase: Optional[float] = Field(default=None, ge=0, le=1)
    baseline_report: Optional[EvaluationReport] = None

    def quantitative_gate_names(self) -> list[str]:
        names = [
            "minimum_recall", "maximum_false_positive_rate", "minimum_macro_f1",
            "minimum_per_class_recall", "minimum_samples", "maximum_macro_f1_drop",
            "maximum_per_class_recall_drop", "maximum_false_positive_rate_increase",
        ]
        return [n for n in names if getattr(self, n) is not None]


class GateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    passed: bool
    observed: Optional[float | str] = None
    required: Optional[float | str] = None
    detail: str = ""


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool
    candidate_version: str
    candidate_artifact_sha256: Optional[str]
    feature_schema_version: str
    evaluation_dataset_fingerprint: str
    baseline_version: Optional[str]
    checks: list[GateCheck]
    reasons: list[str]
    gates: dict[str, Any]
    validated_at: str


def validate_version_string(version: str) -> str:
    """Model versions become directory names: restrict to a safe charset (no traversal)."""
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        raise ConfigError("model version must match ^[a-z0-9][a-z0-9._-]{0,63}$")
    if ".." in version:
        raise ConfigError("model version must not contain '..'")
    return version


def jsonable_floats_ok(obj: Mapping[str, Any]) -> bool:
    """True when no NaN/Infinity occurs anywhere in a nested JSON-like structure."""
    stack: list[Any] = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, float) and not math.isfinite(cur):
            return False
        if isinstance(cur, Mapping):
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
    return True
