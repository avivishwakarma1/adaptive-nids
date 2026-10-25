"""Model bundle and safe artifact handling (contract section 8, SAFETY_POLICY section 9).

Layout on disk (one immutable directory per version, never overwritten in place):

    <base_dir>/<version>/bundle.pkl      restricted-pickle of the fitted bundle
    <base_dir>/<version>/manifest.json   human-readable metadata + sha256 of bundle.pkl

Integrity model
---------------
* `save_bundle` returns the SHA-256 of bundle.pkl. The caller (backend `model_registry`) stores it
  as `artifact_sha256`.
* `load_bundle` REQUIRES `expected_sha256` from that trusted record. A checksum read from the
  manifest sitting next to the artifact would not stop an attacker who can replace both files.
* Unpickling uses an allow-list `find_class`, so even a file with a *matching* checksum cannot
  import os/subprocess/builtins.eval etc. This is defence-in-depth, not a sandbox: only load
  artifacts produced by the controlled training pipeline, never user uploads.
* Scikit-learn / NumPy versions recorded at training time must match the runtime (fail closed).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import platform
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pydantic
import scipy
import sklearn
from pydantic import BaseModel, ConfigDict, Field

from . import __version__ as _pkg_version
from .anomaly import AnomalyDetector
from .classification import KnownClassClassifier
from .contracts import (
    MODEL_KINDS, SUPPORTED_SCHEMA_VERSIONS, ArtifactIntegrityError, ConfigError, ModelLoadError,
    validate_version_string,
)
from .preprocessing import PreprocessingArtifact
from .risk import RiskPolicy

BUNDLE_FILENAME = "bundle.pkl"
MANIFEST_FILENAME = "manifest.json"
_PICKLE_PROTOCOL = 5


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "pydantic": pydantic.VERSION,
        "nids_ml": _pkg_version,
    }


class Thresholds(BaseModel):
    """Key names follow DATABASE_SCHEMA 2.8 `thresholds` ({"known_confidence", "anomaly_threshold"})."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    known_confidence: float = Field(ge=0, le=1)
    anomaly_threshold: float = Field(ge=0, le=1)


class BundleMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_version: str
    model_kind: str
    feature_schema_version: str
    preprocessing_version: str
    input_features: list[str]
    transformed_feature_names: list[str]
    class_labels: list[str]
    label_mapping: dict[str, str] = Field(default_factory=dict)
    benign_label: str = "benign"
    thresholds: Thresholds
    threshold_selection: dict[str, str] = Field(default_factory=dict)
    risk_policy_version: str
    risk_policy: dict[str, Any]
    training_dataset_fingerprint: str
    split_metadata: dict[str, Any] = Field(default_factory=dict)
    training_config: dict[str, Any] = Field(default_factory=dict)
    library_versions: dict[str, str]
    trained_at: str
    evaluated_at: Optional[str] = None
    metrics: Optional[dict[str, Any]] = None  # None until an evaluation actually ran


@dataclass
class ModelBundle:
    metadata: BundleMetadata
    preprocessing: PreprocessingArtifact
    classifier: Optional[KnownClassClassifier]
    anomaly: Optional[AnomalyDetector]
    risk_policy: RiskPolicy
    # Runtime-only; set by save_bundle / load_bundle, never trusted from inside the pickle.
    artifact_sha256: Optional[str] = field(default=None, compare=False)

    @property
    def version(self) -> str:
        return self.metadata.model_version

    def components_available(self) -> dict[str, bool]:
        return {"classifier": self.classifier is not None, "anomaly": self.anomaly is not None}


def check_bundle_consistency(b: ModelBundle) -> None:
    m = b.metadata
    validate_version_string(m.model_version)
    if m.model_kind not in MODEL_KINDS:
        raise ConfigError(f"unknown model_kind {m.model_kind!r}")
    if m.feature_schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ConfigError("bundle feature schema version unsupported")
    if m.feature_schema_version != b.preprocessing.schema_version:
        raise ConfigError("bundle and preprocessing schema versions differ")
    if m.input_features != b.preprocessing.input_columns or m.transformed_feature_names != b.preprocessing.feature_names_out:
        raise ConfigError("bundle feature list does not match the preprocessing artifact")
    kind_ok = {"classifier_plus_anomaly": (True, True), "classifier_only": (True, False),
               "anomaly_only": (False, True)}[m.model_kind]
    if (b.classifier is not None, b.anomaly is not None) != kind_ok:
        raise ConfigError("model_kind does not match the components present")
    if b.classifier is not None and m.class_labels != b.classifier.classes_:
        raise ConfigError("class label list does not match the classifier")
    if m.risk_policy_version != b.risk_policy.policy_version:
        raise ConfigError("risk policy version mismatch")


# --------------------------------------------------------------------------- #
# Restricted unpickler
# --------------------------------------------------------------------------- #
_ALLOWED_BUILTINS = {
    "object", "set", "frozenset", "slice", "range", "complex", "bytearray", "int", "float", "str",
    "list", "dict", "tuple", "bool", "bytes",
}
# Only CLASSES are admitted from these prefixes. Functions such as sklearn.utils._testing helpers that
# can execute code are therefore unreachable even though they live under the same prefix.
_CLASS_ONLY_PREFIXES = ("sklearn.", "nids_ml.")
# Exact (module, name) globals that may be functions/constructors.
_ALLOWED_EXACT = {
    ("collections", "OrderedDict"), ("collections", "defaultdict"),
    ("numpy", "dtype"), ("numpy", "ndarray"),
    ("numpy", "float64"), ("numpy", "float32"), ("numpy", "int64"), ("numpy", "int32"), ("numpy", "bool_"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"), ("numpy.core.numeric", "_frombuffer"),
    ("nids_ml.preprocessing", "_log1p"),
    # Required by HistGradientBoostingClassifier (loss objects and its RNG); fixed, side-effect free helpers.
    ("_loss", "__pyx_unpickle_CyHalfMultinomialLoss"), ("_loss", "__pyx_unpickle_CyHalfBinomialLoss"),
    ("numpy.random._pickle", "__generator_ctor"), ("numpy.random._pickle", "__bit_generator_ctor"),
    ("numpy.random._pickle", "__randomstate_ctor"),
    ("numpy.random.bit_generator", "__pyx_unpickle_SeedSequence"), ("numpy.random.bit_generator", "SeedSequence"),
    ("_loss", "CyHalfMultinomialLoss"), ("_loss", "CyHalfBinomialLoss"),
    ("numpy.random._pcg64", "PCG64"), ("numpy.random._mt19937", "MT19937"), ("numpy.random._generator", "Generator"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "builtins" and name in _ALLOWED_BUILTINS:
            return super().find_class(module, name)
        if (module, name) in _ALLOWED_EXACT:
            return super().find_class(module, name)
        if module.startswith(_CLASS_ONLY_PREFIXES):
            obj = super().find_class(module, name)
            if isinstance(obj, type):
                return obj
        raise ArtifactIntegrityError("artifact references a disallowed global")


def restricted_loads(data: bytes) -> Any:
    try:
        return _RestrictedUnpickler(io.BytesIO(data)).load()
    except ArtifactIntegrityError:
        raise
    except Exception as exc:  # corrupted / truncated / incompatible
        raise ModelLoadError(f"artifact could not be deserialized ({type(exc).__name__})") from None


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SavedArtifact:
    version: str
    directory: Path
    sha256: str
    size_bytes: int


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _version_dir(base_dir: Path, version: str) -> Path:
    validate_version_string(version)
    base = Path(base_dir).resolve()
    d = (base / version).resolve()
    if d.parent != base:
        raise ConfigError("artifact path escapes the artifact directory")
    return d


def save_bundle(bundle: ModelBundle, base_dir: str | Path, *, saved_at: Optional[datetime] = None) -> SavedArtifact:
    """Write a new immutable version directory. Refuses to overwrite an existing version."""
    check_bundle_consistency(bundle)
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    target = _version_dir(base, bundle.version)
    if target.exists():
        raise ConfigError(f"version {bundle.version!r} already exists; artifacts are never overwritten in place")

    payload = pickle.dumps(
        {"format": "nids-ml-bundle/1", "bundle": ModelBundle(
            metadata=bundle.metadata, preprocessing=bundle.preprocessing, classifier=bundle.classifier,
            anomaly=bundle.anomaly, risk_policy=bundle.risk_policy, artifact_sha256=None)},
        protocol=_PICKLE_PROTOCOL,
    )
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "format": "nids-ml-bundle/1", "model_version": bundle.version,
        "model_kind": bundle.metadata.model_kind,
        "feature_schema_version": bundle.metadata.feature_schema_version,
        "preprocessing_version": bundle.metadata.preprocessing_version,
        "artifact_sha256": digest, "size_bytes": len(payload),
        "saved_at": (saved_at or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "library_versions": bundle.metadata.library_versions,
        "metadata": json.loads(bundle.metadata.model_dump_json()),
    }
    tmp = Path(tempfile.mkdtemp(prefix=f".{bundle.version}.", dir=base))
    try:
        (tmp / BUNDLE_FILENAME).write_bytes(payload)
        (tmp / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        os.replace(tmp, target)  # atomic on the same filesystem; fails if target appeared meanwhile
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ConfigError("could not write artifact (version may already exist)") from None
    bundle.artifact_sha256 = digest
    return SavedArtifact(bundle.version, target, digest, len(payload))


def load_bundle(base_dir: str | Path, version: str, *, expected_sha256: str, strict_runtime: bool = True) -> ModelBundle:
    """Load a trusted local artifact. Fails closed on any mismatch (raises `ModelLoadError` family).

    `expected_sha256` must come from the trusted model registry, not from the artifact directory.
    """
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ArtifactIntegrityError("expected_sha256 must be a 64-character hex digest from the model registry")
    try:
        d = _version_dir(Path(base_dir), version)
    except ConfigError as exc:
        raise ModelLoadError(exc.message) from None
    bundle_path, manifest_path = d / BUNDLE_FILENAME, d / MANIFEST_FILENAME
    if not bundle_path.is_file() or not manifest_path.is_file():
        raise ModelLoadError("artifact files not found")
    if bundle_path.is_symlink() or manifest_path.is_symlink():
        raise ArtifactIntegrityError("artifact files must not be symlinks")

    data = bundle_path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256.lower():
        raise ArtifactIntegrityError("artifact checksum does not match the registry value")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ModelLoadError("artifact manifest is unreadable") from None
    if manifest.get("artifact_sha256") != actual or manifest.get("model_version") != version:
        raise ArtifactIntegrityError("artifact manifest is inconsistent with the artifact")

    obj = restricted_loads(data)
    if not isinstance(obj, dict) or obj.get("format") != "nids-ml-bundle/1" or not isinstance(obj.get("bundle"), ModelBundle):
        raise ModelLoadError("artifact has an unexpected structure")
    bundle: ModelBundle = obj["bundle"]
    try:
        check_bundle_consistency(bundle)
    except ConfigError as exc:
        raise ModelLoadError(f"artifact failed schema validation: {exc.message}") from None
    if bundle.version != version:
        raise ArtifactIntegrityError("artifact version does not match the requested version")
    if strict_runtime:
        now = runtime_versions()
        for lib in ("scikit-learn", "numpy", "scipy", "pandas"):
            if bundle.metadata.library_versions.get(lib) != now[lib]:
                raise ModelLoadError(
                    f"{lib} version differs from training runtime "
                    f"({bundle.metadata.library_versions.get(lib)} vs {now[lib]}); retrain or load with strict_runtime=False"
                )
    bundle.artifact_sha256 = actual
    return bundle
