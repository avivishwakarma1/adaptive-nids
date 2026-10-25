import hashlib
import json
import os
import pickle
import shutil
from dataclasses import replace

import numpy as np
import pytest

from nids_ml import (ArtifactIntegrityError, ConfigError, ModelLoadError, PredictionOptions, load_bundle, predict,
                     save_bundle)
from nids_ml.artifacts import BUNDLE_FILENAME, MANIFEST_FILENAME, restricted_loads, runtime_versions
from conftest import fixed_clock
from test_features import GOOD


@pytest.fixture()
def saved(bundle, tmp_path):
    b = replace(bundle)  # shallow copy; save_bundle sets artifact_sha256 on this object
    s = save_bundle(b, tmp_path)
    return tmp_path, b, s


def test_round_trip_preserves_predictions_and_records_checksum(saved, test_records):
    base, b, s = saved
    loaded = load_bundle(base, b.version, expected_sha256=s.sha256)
    assert loaded.artifact_sha256 == s.sha256 == hashlib.sha256((base / b.version / BUNDLE_FILENAME).read_bytes()).hexdigest()
    recs, _ = test_records
    o = PredictionOptions(clock=fixed_clock)
    assert [p.model_dump() for p in predict(recs[:80], b, o)] == [p.model_dump() for p in predict(recs[:80], loaded, o)]
    assert loaded.metadata.library_versions == runtime_versions()
    assert loaded.metadata.metrics is None  # nothing evaluated yet: stays null


def test_manifest_is_valid_json_with_metadata_and_no_nan(saved):
    base, b, s = saved
    m = json.loads((base / b.version / MANIFEST_FILENAME).read_text())
    assert m["artifact_sha256"] == s.sha256 and m["metadata"]["thresholds"]["known_confidence"] >= 0
    assert m["metadata"]["training_dataset_fingerprint"] and m["metadata"]["split_metadata"]["strategy"] == "stratified"


def test_expected_checksum_is_mandatory_and_must_match(saved):
    base, b, s = saved
    with pytest.raises(ArtifactIntegrityError):
        load_bundle(base, b.version, expected_sha256="0" * 64)
    with pytest.raises(ArtifactIntegrityError):
        load_bundle(base, b.version, expected_sha256="short")
    with pytest.raises(TypeError):
        load_bundle(base, b.version)  # no way to skip the registry checksum


def test_tampered_artifact_is_rejected(saved):
    base, b, s = saved
    p = base / b.version / BUNDLE_FILENAME
    data = bytearray(p.read_bytes()); data[len(data) // 2] ^= 0xFF; p.write_bytes(bytes(data))
    with pytest.raises(ArtifactIntegrityError):
        load_bundle(base, b.version, expected_sha256=s.sha256)


def test_truncated_or_garbage_artifact_with_matching_checksum_fails_closed(saved):
    base, b, s = saved
    d = base / b.version
    for payload in (b"", b"not a pickle", (d / BUNDLE_FILENAME).read_bytes()[:200]):
        (d / BUNDLE_FILENAME).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        m = json.loads((d / MANIFEST_FILENAME).read_text()); m["artifact_sha256"] = digest
        (d / MANIFEST_FILENAME).write_text(json.dumps(m))
        with pytest.raises(ModelLoadError):
            load_bundle(base, b.version, expected_sha256=digest)


def test_missing_files_and_unknown_version(saved, tmp_path):
    base, b, s = saved
    with pytest.raises(ModelLoadError):
        load_bundle(base, "does-not-exist", expected_sha256=s.sha256)
    os.remove(base / b.version / MANIFEST_FILENAME)
    with pytest.raises(ModelLoadError):
        load_bundle(base, b.version, expected_sha256=s.sha256)


def test_manifest_inconsistent_with_artifact_is_rejected(saved):
    base, b, s = saved
    mp = base / b.version / MANIFEST_FILENAME
    m = json.loads(mp.read_text()); m["artifact_sha256"] = "f" * 64; mp.write_text(json.dumps(m))
    with pytest.raises(ArtifactIntegrityError):
        load_bundle(base, b.version, expected_sha256=s.sha256)


@pytest.mark.parametrize("version", ["../evil", "a/b", "A-upper", "", ".hidden", "x" * 100, "a..b"])
def test_version_strings_cannot_traverse(bundle, tmp_path, version):
    md = bundle.metadata.model_copy(update={"model_version": version})
    with pytest.raises(ConfigError):
        save_bundle(replace(bundle, metadata=md), tmp_path)
    with pytest.raises(ModelLoadError):
        load_bundle(tmp_path, version, expected_sha256="0" * 64)


def test_never_overwrites_an_existing_version(saved):
    base, b, s = saved
    before = (base / b.version / BUNDLE_FILENAME).read_bytes()
    with pytest.raises(ConfigError):
        save_bundle(replace(b), base)
    assert (base / b.version / BUNDLE_FILENAME).read_bytes() == before
    assert not [p for p in base.iterdir() if p.name.startswith(".")]  # no temp dirs left behind


def test_malicious_pickle_with_valid_checksum_cannot_execute(tmp_path):
    marker = tmp_path / "PWNED"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    payload = pickle.dumps({"format": "nids-ml-bundle/1", "bundle": Evil()})
    with pytest.raises(ArtifactIntegrityError):
        restricted_loads(payload)
    d = tmp_path / "evil-1"; d.mkdir()
    digest = hashlib.sha256(payload).hexdigest()
    (d / BUNDLE_FILENAME).write_bytes(payload)
    (d / MANIFEST_FILENAME).write_text(json.dumps({"artifact_sha256": digest, "model_version": "evil-1"}))
    with pytest.raises(ArtifactIntegrityError):
        load_bundle(tmp_path, "evil-1", expected_sha256=digest)
    assert not marker.exists()


@pytest.mark.parametrize("gadget", [("builtins", "eval"), ("builtins", "__import__"), ("subprocess", "Popen"),
                                    ("sklearn.utils._testing", "assert_run_python_script_without_output"),
                                    ("os", "system"), ("posix", "system")])
def test_restricted_unpickler_blocks_known_gadget_globals(gadget):
    mod, name = gadget
    payload = b"c" + mod.encode() + b"\n" + name.encode() + b"\n."
    with pytest.raises(ArtifactIntegrityError):
        restricted_loads(payload)


def test_runtime_version_mismatch_fails_closed_unless_relaxed(bundle, tmp_path):
    md = bundle.metadata.model_copy(update={"model_version": "old-sklearn-1", "library_versions": {
        **bundle.metadata.library_versions, "scikit-learn": "0.0.1"}})
    b = replace(bundle, metadata=md)
    s = save_bundle(b, tmp_path)
    with pytest.raises(ModelLoadError):
        load_bundle(tmp_path, b.version, expected_sha256=s.sha256)
    assert load_bundle(tmp_path, b.version, expected_sha256=s.sha256, strict_runtime=False).version == b.version


def test_unsupported_schema_version_in_bundle_is_rejected(bundle, tmp_path):
    md = bundle.metadata.model_copy(update={"model_version": "future-1", "feature_schema_version": "9.0.0"})
    with pytest.raises(ConfigError):
        save_bundle(replace(bundle, metadata=md), tmp_path)


def test_inconsistent_bundle_is_not_saved(bundle, tmp_path):
    md = bundle.metadata.model_copy(update={"model_version": "bad-kind-1", "model_kind": "classifier_only"})
    with pytest.raises(ConfigError):
        save_bundle(replace(bundle, metadata=md), tmp_path)


def test_symlinked_artifact_is_refused(saved, tmp_path_factory):
    base, b, s = saved
    p = base / b.version / BUNDLE_FILENAME
    real = tmp_path_factory.mktemp("elsewhere") / "x.pkl"
    shutil.copy(p, real); p.unlink(); p.symlink_to(real)
    with pytest.raises(ArtifactIntegrityError):
        load_bundle(base, b.version, expected_sha256=s.sha256)
