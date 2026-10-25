"""Canonical feature schema 1.0.0: validation, normalization, model-input frame, dataset mapping.

Nothing here fits anything: it is pure, deterministic, and has no I/O except the explicit
`load_mapping` / `apply_mapping` helpers used by offline scripts.
"""
from __future__ import annotations

import json
import math
import numbers
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import (
    DEFAULT_MAX_BATCH_SIZE, DIRECTIONS, PROTOCOLS, SUPPORTED_SCHEMA_VERSIONS,
    ConfigError, FeatureValidationReport, InputValidationError, SchemaVersionError,
    ValidationIssue, normalize_label,
)

# --------------------------------------------------------------------------- #
# Schema definition (contract section 3)
# --------------------------------------------------------------------------- #
CANONICAL_REQUIRED_KEYS: tuple[str, ...] = (
    "duration", "protocol", "source_port", "destination_port", "flow_packets", "flow_bytes",
    "packets_per_second", "bytes_per_second", "connection_state", "direction", "observed_at",
)
OPTIONAL_KEYS: tuple[str, ...] = ("source_id",)
ALL_KEYS = CANONICAL_REQUIRED_KEYS + OPTIONAL_KEYS

NULLABLE_NUMERIC = ("flow_packets", "flow_bytes", "packets_per_second", "bytes_per_second")
PORT_FIELDS = ("source_port", "destination_port")

# Model input columns (what the preprocessing pipeline sees). NOT used as features:
#   observed_at  - time artefacts are dataset-specific; only used for time-aware splitting
#   source_id    - opaque identifier, never predictive (contract section 3)
#   raw source_port - ephemeral; bucketed instead (see build_model_frame)
NUMERIC_COLUMNS: tuple[str, ...] = (
    "duration", "flow_packets", "flow_bytes", "packets_per_second", "bytes_per_second",
)
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "protocol", "connection_state", "direction", "source_port_bucket", "destination_port",
)
MODEL_INPUT_COLUMNS: tuple[str, ...] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

_STATE_RE = re.compile(r"^[a-z0-9_.:+\-]{1,32}$")
_MAX_ISSUES_DEFAULT = 100
_MIN_TS = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MAX_TS = datetime(2200, 1, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class _Collector:
    def __init__(self, cap: int):
        self.cap = cap
        self.issues: list[ValidationIssue] = []
        self.warnings: list[ValidationIssue] = []
        self.truncated = False

    def issue(self, row: int, field: Optional[str], code: str, msg: str) -> None:
        if len(self.issues) >= self.cap:
            self.truncated = True
            return
        self.issues.append(ValidationIssue(row_index=row, field=field, code=code, message=msg))

    def warn(self, row: int, field: Optional[str], code: str, msg: str) -> None:
        if len(self.warnings) < self.cap:
            self.warnings.append(ValidationIssue(row_index=row, field=field, code=code, message=msg))


def _is_number(v: Any) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def _clean_number(v: Any, *, nullable: bool, integer: bool = False,
                  hi: Optional[float] = None) -> tuple[Optional[float], Optional[tuple[str, str]]]:
    """Return (value, error). NaN is 'missing' for nullable fields and an error otherwise."""
    if v is None:
        return (None, None) if nullable else (None, ("NULL_NOT_ALLOWED", "value is required"))
    if not _is_number(v):
        return None, ("NOT_NUMERIC", "expected a number")
    f = float(v)
    if math.isnan(f):
        return (None, None) if nullable else (None, ("NON_FINITE", "NaN not allowed"))
    if math.isinf(f):
        return None, ("NON_FINITE", "Infinity not allowed")
    if f < 0:
        return None, ("OUT_OF_RANGE", "must be nonnegative")
    if integer and f != math.floor(f):
        return None, ("NOT_INTEGER", "must be an integer")
    if hi is not None and f > hi:
        return None, ("OUT_OF_RANGE", f"must be <= {hi:g}")
    return f, None


def _clean_category(v: Any, allowed: Optional[Sequence[str]], *, state: bool = False
                    ) -> tuple[Optional[str], Optional[tuple[str, str]], bool]:
    """Return (value, error, normalized_to_unknown)."""
    if v is None:
        return "unknown", None, True
    if not isinstance(v, str):
        return None, ("NOT_A_STRING", "expected a string"), False
    s = v.strip().lower()
    if s == "":
        return "unknown", None, True
    if allowed is not None and s not in allowed:
        return None, ("INVALID_ENUM", f"must be one of {list(allowed)}"), False
    if state and not _STATE_RE.match(s):
        return None, ("INVALID_FORMAT", "connection_state has unsupported characters or length"), False
    return s, None, False


def _clean_timestamp(v: Any) -> tuple[Optional[datetime], Optional[tuple[str, str]]]:
    if v is None:
        return None, None
    if isinstance(v, pd.Timestamp):
        v = v.to_pydatetime()
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v.strip())
        except ValueError:
            return None, ("INVALID_TIMESTAMP", "not ISO 8601")
    if not isinstance(v, datetime):
        return None, ("INVALID_TIMESTAMP", "expected ISO 8601 string or datetime")
    if v.tzinfo is None or v.utcoffset() is None:
        return None, ("TIMESTAMP_NOT_UTC", "timestamp must carry a UTC offset or 'Z'")
    v = v.astimezone(timezone.utc)
    if not (_MIN_TS <= v <= _MAX_TS):
        return None, ("OUT_OF_RANGE", "timestamp outside supported range")
    return v, None


def _check_batch(records: Any, schema_version: str, max_batch_size: Optional[int]) -> list:
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaVersionError(
            f"unsupported feature schema version {schema_version!r}; supported: "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    # Accept the contract's wrapper form: {"feature_schema_version": ..., "observations": [...]}
    if isinstance(records, Mapping):
        wrapped_version = records.get("feature_schema_version", schema_version)
        if wrapped_version != schema_version:
            raise SchemaVersionError("feature_schema_version in payload does not match requested version")
        if set(records) - {"feature_schema_version", "observations"} or "observations" not in records:
            raise InputValidationError("payload must contain only 'feature_schema_version' and 'observations'")
        records = records["observations"]
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise InputValidationError("records must be a list of mappings")
    if len(records) == 0:
        raise InputValidationError("batch is empty; at least one observation is required")
    if max_batch_size is not None and len(records) > max_batch_size:
        raise InputValidationError(f"batch of {len(records)} exceeds the maximum of {max_batch_size}")
    return list(records)


def validate_features(
    records: Any,
    schema_version: str = "1.0.0",
    *,
    max_batch_size: Optional[int] = DEFAULT_MAX_BATCH_SIZE,
    extra_fields: Literal["reject", "strip"] = "reject",
    max_issues: int = _MAX_ISSUES_DEFAULT,
) -> FeatureValidationReport:
    """Validate and normalize a batch of canonical observations.

    Batch-level problems raise (`InputValidationError`, `SchemaVersionError`). Row-level
    problems are collected in the report; call `report.raise_if_invalid()` to fail closed.
    Error messages never echo input values (they may contain sensitive data).
    """
    rows = _check_batch(records, schema_version, max_batch_size)
    col = _Collector(max_issues)
    out: list[Optional[dict]] = []

    for i, rec in enumerate(rows):
        if not isinstance(rec, Mapping):
            col.issue(i, None, "NOT_A_MAPPING", "each observation must be a mapping")
            out.append(None)
            continue
        n_before = len(col.issues) + (1 if col.truncated else 0)
        missing = [k for k in CANONICAL_REQUIRED_KEYS if k not in rec]
        for k in missing:
            col.issue(i, k, "MISSING_FIELD", "required field is absent")
        extras = [k for k in rec if k not in ALL_KEYS]
        for k in extras:
            if extra_fields == "reject":
                col.issue(i, str(k)[:64], "UNEXPECTED_FIELD", "field is not part of schema 1.0.0")
            else:
                col.warn(i, str(k)[:64], "FIELD_STRIPPED", "unexpected field removed before model entry")

        norm: dict[str, Any] = {}
        # duration: the only required non-nullable numeric
        if "duration" in rec:
            val, err = _clean_number(rec["duration"], nullable=False)
            if err:
                col.issue(i, "duration", *err)
            norm["duration"] = val
        for f in PORT_FIELDS:
            if f in rec:
                val, err = _clean_number(rec[f], nullable=True, integer=True, hi=65535)
                if err:
                    col.issue(i, f, *err)
                norm[f] = None if val is None else int(val)
        for f in NULLABLE_NUMERIC:
            if f in rec:
                val, err = _clean_number(rec[f], nullable=True)
                if err:
                    col.issue(i, f, *err)
                norm[f] = val
        for f, allowed, is_state in (("protocol", PROTOCOLS, False), ("direction", DIRECTIONS, False),
                                     ("connection_state", None, True)):
            if f in rec:
                val, err, was_null = _clean_category(rec[f], allowed, state=is_state)
                if err:
                    col.issue(i, f, *err)
                elif was_null:
                    col.warn(i, f, "NORMALIZED_TO_UNKNOWN", "null/empty category mapped to 'unknown'")
                norm[f] = val
        if "observed_at" in rec:
            ts, err = _clean_timestamp(rec["observed_at"])
            if err:
                col.issue(i, "observed_at", *err)
            norm["observed_at"] = ts
        sid = rec.get("source_id")
        if sid is not None:
            if isinstance(sid, bool) or not isinstance(sid, (str, int)) or len(str(sid)) > 128:
                col.issue(i, "source_id", "INVALID_FORMAT", "source_id must be a string/int up to 128 chars")
            else:
                norm["source_id"] = str(sid)
        norm.setdefault("source_id", None)

        row_failed = (len(col.issues) + (1 if col.truncated else 0)) > n_before
        out.append(None if row_failed else {k: norm.get(k) for k in ALL_KEYS})

    return FeatureValidationReport(
        schema_version=schema_version, n_records=len(rows), valid=not col.issues and not col.truncated,
        issues=col.issues, warnings=col.warnings, issues_truncated=col.truncated, records=out,
    )


# --------------------------------------------------------------------------- #
# Model-input frame
# --------------------------------------------------------------------------- #
def _port_bucket(p: Optional[int]) -> str:
    if p is None:
        return "missing"
    if p < 1024:
        return "well_known"
    if p < 49152:
        return "registered"
    return "dynamic"


def build_model_frame(normalized: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalized records -> DataFrame with exactly MODEL_INPUT_COLUMNS, in that order.

    source_id, observed_at and the raw source port never enter this frame.
    """
    def num(k: str) -> list[float]:
        return [np.nan if r[k] is None else float(r[k]) for r in normalized]

    data = {k: np.asarray(num(k), dtype="float64") for k in NUMERIC_COLUMNS}
    data["protocol"] = [r["protocol"] for r in normalized]
    data["connection_state"] = [r["connection_state"] for r in normalized]
    data["direction"] = [r["direction"] for r in normalized]
    data["source_port_bucket"] = [_port_bucket(r["source_port"]) for r in normalized]
    data["destination_port"] = ["missing" if r["destination_port"] is None else str(r["destination_port"])
                                for r in normalized]
    return pd.DataFrame(data, columns=list(MODEL_INPUT_COLUMNS))


def incomplete_mask(normalized: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """True where any nullable numeric/port feature is missing (reason code FEATURES_INCOMPLETE)."""
    keys = NULLABLE_NUMERIC + PORT_FIELDS
    return np.array([any(r[k] is None for k in keys) for r in normalized], dtype=bool)


# --------------------------------------------------------------------------- #
# Dataset mapping (contract section 3: explicit, reviewed mapping; no silent inference)
# --------------------------------------------------------------------------- #
class FieldSpec(BaseModel):
    """Exactly one source: `column`, `sum_of`, `constant`, or `derive_rate_of`. None of them = all-null."""
    model_config = ConfigDict(extra="forbid")
    column: Optional[str] = None
    sum_of: Optional[list[str]] = None
    constant: Optional[Any] = None
    derive_rate_of: Optional[Literal["flow_packets", "flow_bytes"]] = None  # numerator / duration
    scale: float = 1.0
    value_map: Optional[dict[str, str]] = None
    default: Optional[str] = None  # used with value_map for categories not listed

    @model_validator(mode="after")
    def _one_source(self) -> "FieldSpec":
        n = sum(x is not None for x in (self.column, self.sum_of, self.constant, self.derive_rate_of))
        if n > 1:
            raise ValueError("a field may use only one of column/sum_of/constant/derive_rate_of")
        return self


class ColumnMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mapping_version: str = "1"
    dataset_name: str
    reviewed: bool = Field(description="Must be true. Set only after a human compares this file with the real dataset.")
    review_notes: str = ""
    license_note: str = ""
    strip_header_whitespace: bool = True
    label_column: str
    benign_labels: list[str]
    label_map: dict[str, str] = Field(default_factory=dict)
    group_column: Optional[str] = None
    fields: dict[str, FieldSpec]

    @model_validator(mode="after")
    def _all_fields(self) -> "ColumnMapping":
        missing = [k for k in CANONICAL_REQUIRED_KEYS if k not in self.fields]
        if missing:
            raise ValueError(f"mapping must define every canonical field; missing {missing}")
        unknown = [k for k in self.fields if k not in ALL_KEYS]
        if unknown:
            raise ValueError(f"unknown canonical fields in mapping: {unknown}")
        return self


class MappingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str
    rows_in: int
    rows_out: int
    rows_dropped: int
    dropped_by_reason: dict[str, int]
    label_counts: dict[str, int]
    unmapped_labels: list[str]


def load_mapping(path: str | Path) -> ColumnMapping:
    try:
        m = ColumnMapping.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot load mapping file: {type(exc).__name__}") from None
    if not m.reviewed:
        raise ConfigError(
            f"mapping '{m.dataset_name}' has reviewed=false. Compare it with the real dataset header, "
            "fix it, and set reviewed=true before it can be used."
        )
    return m


def _apply_spec(df: pd.DataFrame, name: str, spec: FieldSpec, duration: Optional[pd.Series]) -> pd.Series:
    n = len(df)
    if spec.column is not None:
        if spec.column not in df.columns:
            raise ConfigError(f"mapped column for '{name}' not found in dataset header")
        s = df[spec.column]
    elif spec.sum_of is not None:
        for c in spec.sum_of:
            if c not in df.columns:
                raise ConfigError(f"mapped column for '{name}' not found in dataset header")
        s = sum(pd.to_numeric(df[c], errors="coerce") for c in spec.sum_of)
    elif spec.constant is not None:
        s = pd.Series([spec.constant] * n, index=df.index)
    elif spec.derive_rate_of is not None:
        raise ConfigError("derive_rate_of must be resolved after base fields")  # handled by caller
    else:
        s = pd.Series([None] * n, index=df.index, dtype="object")
    if spec.value_map is not None:
        key = s.astype("string").str.strip().str.lower()
        mapped = key.map({k.lower(): v for k, v in spec.value_map.items()})
        s = mapped.fillna(spec.default if spec.default is not None else "unknown")
    elif name in NULLABLE_NUMERIC or name in PORT_FIELDS or name == "duration":
        s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan) * spec.scale
    return s


def apply_mapping(raw: pd.DataFrame, mapping: ColumnMapping) -> tuple[pd.DataFrame, pd.Series, Optional[pd.Series], MappingReport]:
    """Raw dataset frame -> (canonical records frame, canonical labels, optional group key, report).

    Rows that fail canonical validation are dropped and *counted by reason*; they are never
    silently repaired. Labels stay in a separate Series (no label leakage into features).
    """
    df = raw.copy()
    if mapping.strip_header_whitespace:
        df.columns = [str(c).strip() for c in df.columns]
    for needed in [mapping.label_column] + ([mapping.group_column] if mapping.group_column else []):
        if needed not in df.columns:
            raise ConfigError("label/group column from mapping not found in dataset header")

    cols: dict[str, pd.Series] = {}
    for name, spec in mapping.fields.items():
        if spec.derive_rate_of is None:
            cols[name] = _apply_spec(df, name, spec, None)
    for name, spec in mapping.fields.items():
        if spec.derive_rate_of is not None:
            num = cols.get(spec.derive_rate_of)
            dur = cols["duration"]
            if num is None:
                raise ConfigError(f"derive_rate_of for '{name}' references an unmapped numerator")
            cols[name] = (num / dur.where(dur > 0)).replace([np.inf, -np.inf], np.nan) * spec.scale

    canon = pd.DataFrame({k: cols[k] for k in ALL_KEYS if k in cols})
    if "source_id" not in canon:
        canon["source_id"] = None
    benign = {normalize_label(b) for b in mapping.benign_labels}
    lab_raw = df[mapping.label_column].astype("string").str.strip()
    lm = {k.lower(): normalize_label(v) for k, v in mapping.label_map.items()}
    unmapped: set[str] = set()

    def canon_label(x: Any) -> Optional[str]:
        if x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA:
            return None
        key = str(x).strip().lower()
        if key in lm:
            return lm[key]
        try:
            lab = normalize_label(str(x))
        except ValueError:
            unmapped.add(str(x)[:40])
            return None
        return "benign" if lab in benign else lab

    labels = pd.Series([canon_label(x) for x in lab_raw], index=df.index, dtype="object")
    groups = df[mapping.group_column].astype("string") if mapping.group_column else None

    recs = canon.replace({np.nan: None}).to_dict(orient="records")
    rep = validate_features(recs, max_batch_size=None, max_issues=10_000_000)
    reasons: Counter = Counter()
    keep = np.ones(len(recs), dtype=bool)
    for iss in rep.issues:
        if iss.row_index is not None:
            keep[iss.row_index] = False
    for iss in rep.issues:
        reasons[f"{iss.field or 'row'}:{iss.code}"] += 1
    bad_label = labels.isna().to_numpy()
    reasons["label:MISSING_OR_UNUSABLE"] += int((bad_label & keep).sum())
    keep &= ~bad_label

    norm_rows = [r for r, k in zip(rep.records, keep) if k and r is not None]
    keep_idx = np.flatnonzero(keep)
    out = pd.DataFrame(norm_rows, index=raw.index[keep_idx])[list(ALL_KEYS)]
    out_labels = labels.iloc[keep_idx]
    out_groups = groups.iloc[keep_idx] if groups is not None else None
    report = MappingReport(
        dataset_name=mapping.dataset_name, rows_in=len(raw), rows_out=len(out),
        rows_dropped=len(raw) - len(out), dropped_by_reason=dict(reasons),
        label_counts={str(k): int(v) for k, v in out_labels.value_counts().items()},
        unmapped_labels=sorted(unmapped),
    )
    return out, out_labels, out_groups, report


def frame_to_records(frame: pd.DataFrame) -> list[dict]:
    """Canonical frame -> list of plain dicts (NaN -> None, Timestamps preserved)."""
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
