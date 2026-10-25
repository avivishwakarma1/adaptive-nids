# ml/artifacts

Trained model bundles live here, one **immutable** directory per version (`<version>/bundle.pkl` + `manifest.json`).
Everything except this file and `.gitkeep` is git-ignored: never commit trained artifacts, datasets or captures.

* Created only by `scripts/train.py` / the candidate cycle (`nids_ml.save_bundle`). A version is never overwritten.
* Loaded only with the SHA-256 recorded in the backend `model_registry` (`artifact_sha256`): `nids_ml.load_bundle(dir, version, expected_sha256=...)`.
* Never place user-uploaded or third-party model files here. Pickle-family formats can execute code when loaded.
* No artifact exists in this repository yet: no real dataset has been trained on. Until one is registered, the backend must report `MODEL_UNAVAILABLE`.
