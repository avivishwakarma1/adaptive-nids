# nids_ml — offline ML engine for the Adaptive AI-Driven NIDS

Implements `docs/ML_CONTRACT.md` v1.0.0 under the limits of `docs/SAFETY_POLICY.md`.
Advisory only: no network access, no packet capture, no enforcement, no database, no web framework.

> **Evidence status (PROJECT_SPEC §13).** All code below is `implemented`; the unit/contract/safety tests are `verified`
> in the environment listed at the bottom. **No real dataset has been obtained, inspected, trained on or evaluated.**
> No model-quality number exists. UNSW-NB15, CICIDS2017 and ToN-IoT work is `blocked` on the data files (see "Remaining dependencies").
> Tests use a clearly labelled synthetic fixture (`tests/synthetic_fixture.py`) whose results mean nothing about detection quality.

## 1. Layout

```text
ml/
├── pyproject.toml  requirements.txt  requirements-dev.txt  README.md  .gitignore
├── src/nids_ml/
│   ├── __init__.py        public interface re-exports
│   ├── contracts.py       enums, errors, strict PredictionResult, option/report/gate/result types, severity mapping
│   ├── features.py        schema 1.0.0 validation, model-input frame, REVIEWED dataset mapping loader
│   ├── preprocessing.py   fitted preprocessing artifact (impute/scale/one-hot + guaranteed "__other__"), preprocess()
│   ├── classification.py  known-class classifier wrapper (RandomForest baseline; LR / HistGB for comparison)
│   ├── anomaly.py         Isolation Forest on benign flows + validation-fitted rank normalisation
│   ├── risk.py            versioned risk policy -> score 0-100 (+ severity mapping)
│   ├── inference.py       decision policy + predict()                         [added module, see §9]
│   ├── training.py        dedupe, leakage checks, splits, threshold selection, fit_bundle  [added module]
│   ├── evaluation.py      metrics, calibration diagnostic, evaluate(), held-out-class protocol
│   ├── artifacts.py       ModelBundle, checksummed immutable save, restricted load
│   └── lifecycle.py       gates, validate_candidate, feedback->candidate set, LocalModelRegistry (promote/rollback)
├── scripts/               inspect_dataset.py train.py evaluate.py unknown_attack.py continual_experiment.py _common.py   [added dir]
├── configs/               train_default.json  gates.template.json  mappings/{canonical_identity, *.draft}.json           [added files]
├── artifacts/             (git-ignored) one immutable directory per model version
└── tests/                 213 tests; synthetic_fixture.py is the ONLY data source
```

## 2. Setup, tests, run commands

Python ≥ 3.11. From the repository root (PowerShell shown first, bash second):

```powershell
py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r ml/requirements-dev.txt ; pip install -e ml
python -m pytest ml/tests                      # or:  cd ml ; python -m pytest
```
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements-dev.txt && pip install -e ml
python -m pytest ml/tests
```
Runs in ~8 s, needs no frontend, MongoDB, network or admin rights. Single areas: `pytest ml/tests/test_inference.py -k anomaly`.

### Real-dataset workflow (runs only once you supply the files)

```bash
cd ml
# 1. Facts first: header, dtypes, missing, inf, duplicates, labels. Reads only; no modelling.
python scripts/inspect_dataset.py --csv /path/UNSW_NB15_training-set.csv --out runs/unsw_inspect.json
# 2. Review/fix configs/mappings/unsw_nb15.draft.json against the header, then set "reviewed": true
#    (the loader refuses reviewed=false). Re-run with --mapping to see rows dropped per reason + class balance.
python scripts/inspect_dataset.py --csv /path/UNSW.csv --mapping configs/mappings/unsw_nb15.json --out runs/unsw_inspect2.json
# 3. Train baseline -> immutable artifact + test-split evaluation (prints sha256 for model_registry.artifact_sha256)
python scripts/train.py --csv /path/UNSW.csv --mapping configs/mappings/unsw_nb15.json --dataset-name unsw-nb15 \
    --version baseline-1 --artifact-dir artifacts --run-dir runs/baseline-1 --config configs/train_default.json --compare-classifiers
# 4. Unknown-attack protocol (one fold per held-out class or family)
python scripts/unknown_attack.py --csv /path/UNSW.csv --mapping configs/mappings/unsw_nb15.json --dataset-name unsw-nb15 \
    --held-out backdoor dos --family dos=dos,ddos --out runs/unknown.json
# 5. Evaluate a saved model on a different labelled file (cross-dataset or fixed holdout)
python scripts/evaluate.py --csv /path/other.csv --mapping configs/mappings/other.json --dataset-name other \
    --artifact-dir artifacts --version baseline-1 --expected-sha256 <registry value> --out runs/eval_other.json
# 6. Continual-learning cycle (never promotes unless --promote --approved-by NAME and validation passed)
python scripts/continual_experiment.py --csv /path/UNSW.csv --mapping configs/mappings/unsw_nb15.json --dataset-name unsw-nb15 \
    --artifact-dir artifacts --registry runs/registry.json --feedback runs/reviewed_feedback.jsonl \
    --gates configs/gates.json --candidate-version candidate-2 --out-dir runs/cycle-2
```
Step 4: `--held-out` takes class names or `--family` aliases (one fold each); names must match the canonical snake_case labels produced by the mapping.
`configs/gates.template.json` is deliberately all-null: with no gate configured validation **fails**. Pick gate values only after a real baseline run, and record why.
The three `*.draft.json` mappings are **unverified** drafts from public documentation; they cannot be used until a human reviews them against the real header.

## 3. Public interface (ML_CONTRACT §2)

```python
from nids_ml import load_bundle, predict, PredictionOptions
bundle = load_bundle("artifacts", "baseline-1", expected_sha256="<model_registry.artifact_sha256>")   # checksum mandatory
results = predict(records_or_wrapper, bundle, PredictionOptions(max_batch_size=1000))                # -> list[PredictionResult]
```

| Function | Behaviour |
|---|---|
| `validate_features(records, schema_version="1.0.0")` | Returns `FeatureValidationReport` (`valid`, per-row `issues`, `warnings`, normalized `records` aligned with input, `None` for bad rows). Batch-level problems (empty, > max batch, wrong container, wrong schema version) **raise** `InputValidationError` / `SchemaVersionError`. Error text never echoes input values. Accepts a list of mappings or the contract's `{"feature_schema_version", "observations"}` wrapper. |
| `preprocess(records, artifact, schema_version)` | Validate → model frame → fitted transform. Returns float64 matrix whose columns are exactly `artifact.feature_names_out`. |
| `predict(records, bundle, options=None)` | Exactly one `PredictionResult` per input row, same order; only the 11 contract fields (`extra="forbid"`, severity/label/type consistency enforced by the model). Raises `ModelUnavailableError` (`.code == "MODEL_UNAVAILABLE"`) instead of ever returning a fabricated benign result. |
| `evaluate(bundle, labeled_records, config)` | `labeled_records` = list of `{"observation": {...canonical...}, "label": "..."}`. Returns `EvaluationReport` (see §6). Never tunes anything. |
| `validate_candidate(candidate_bundle, evaluation_report, gates)` | Returns `ValidationResult` with every check recorded (§8). |

Errors carry stable `.code`s: `VALIDATION_ERROR`, `SCHEMA_VERSION_MISMATCH`, `MODEL_UNAVAILABLE`, `ARTIFACT_INTEGRITY_ERROR`, `CONFIG_ERROR`, `LIFECYCLE_CONFLICT`.
The backend maps them to its envelope; nothing in `nids_ml` knows about HTTP.

### Input rules implemented
Required keys: `duration protocol source_port destination_port flow_packets flow_bytes packets_per_second bytes_per_second connection_state direction observed_at` (`source_id` optional, never a feature).
`duration` finite ≥ 0 and non-null; ports integer 0–65535 or null; flow/rate fields ≥ 0 or null (NaN = missing for nullable fields; ±Inf always rejected); booleans and numeric strings rejected;
`protocol` ∈ tcp/udp/icmp/other/unknown after `strip().lower()` (other values are **rejected** — ingestion mapping must normalise, see §9); `direction` ∈ inbound/outbound/internal/unknown;
`connection_state` lower-cased, ≤ 32 chars of `[a-z0-9_.:+-]`, null/empty → `unknown` (recorded as a warning); `observed_at` must be timezone-aware (naive rejected), converted to UTC;
extra keys are rejected by default (`extra_fields="strip"` removes them and records a warning).

## 4. Model, decision policy, risk (ML_CONTRACT §6–7)

**Features given to the model** (`MODEL_INPUT_COLUMNS`): `duration, flow_packets, flow_bytes, packets_per_second, bytes_per_second` (log1p → train-median impute → scale, plus a missing-indicator per numeric) and categorical `protocol, connection_state, direction, source_port_bucket, destination_port`.
Not used: `source_id`, `observed_at`, IPs, raw source port (bucketed to well_known/registered/dynamic/missing). Destination ports are one-hot by training frequency; rare and unseen values go to an always-present `__other__` column.

**Classifier**: `RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1)` baseline; `logistic_regression` and `hist_gradient_boosting` selectable and compared **on the validation split only** (`--compare-classifiers`). `confidence` = max class score; it is **not** a calibrated probability, and `evaluate` reports a reliability diagnostic (ECE + bins) beside it.

**Anomaly detector**: `IsolationForest` fitted on **benign training flows only**. Raw score → `anomaly_score ∈ [0,1]` by empirical CDF over held-out **benign validation** flows ("more anomalous than X% of held-out benign validation flows"): a rank, not a probability. `anomaly_threshold` is the validation-benign quantile for `target_false_positive_rate` (default 0.05, an operating-point choice recorded in `threshold_selection`).
`known_confidence` = smallest confidence whose *accepted* validation predictions reach `target_accepted_accuracy` (default 0.95); if unreachable it falls back to the highest observed confidence and says so in `threshold_selection`.

**Decision policy `decision-policy-1.0.0`** (code: `inference.decide_row`, tested independently of any model):

| Situation | `prediction_type` / `prediction_label` | codes |
|---|---|---|
| confidence ≥ threshold, attack class | `known_attack` / class | `KNOWN_CLASS_HIGH_SCORE` (+`ANOMALY_*` if anomalous; label is **never overwritten**) |
| confidence ≥ threshold, benign, anomaly below threshold | `benign` / `benign` | – |
| confidence ≥ threshold, benign, anomaly ≥ threshold | `anomaly` / `unknown` | `ANOMALY_*` |
| confidence < threshold, anomaly below threshold | `uncertain` / `unknown` | `KNOWN_CLASS_LOW_CONFIDENCE` |
| confidence < threshold, anomaly ≥ threshold | `anomaly` / `unknown` (explicit promotion) | `KNOWN_CLASS_LOW_CONFIDENCE`, `ANOMALY_*` |
| anomaly-only bundle | `anomaly` or `uncertain` / `unknown` — never `benign` | `MODEL_COMPONENT_UNAVAILABLE` |
| no component / component raises | `ModelUnavailableError` (default `component_failure_policy="fail"`) or explicit `uncertain` rows (`"uncertain"`) | `MODEL_COMPONENT_UNAVAILABLE` |

Additional codes: `FEATURES_INCOMPLETE` (any nullable numeric/port missing), `OUT_OF_DISTRIBUTION_WARNING` (a categorical value never seen in training; numeric range drift is **not** checked), `RISK_POLICY_HIGH` (risk ≥ 60). `ANOMALY_HIGH` = score in the upper half of `[threshold, 1]`.

**Risk `risk-policy-1.0.0`** (`risk.py`; a priority score, not a probability; weights are **unvalidated design defaults**):
known_attack `100·w_class·(0.7·conf + 0.3·anom)` (anom term dropped when no detector) · anomaly `100·0.75·anom` · uncertain `100·0.4·(anom or 0.5)` · benign `100·0.15·anom`; half-up rounding, clamp 0–100, then 0–19 info / 20–39 low / 40–59 medium / 60–79 high / 80–100 critical (`severity_for_score`, boundary-tested at 19/20, 39/40, 59/60, 79/80, and re-checked by `PredictionResult`).
Per-class weights go in `risk_policy.class_weights`; the backend must not recompute severity.

## 5. Training protocol and leakage controls (ML_CONTRACT §9)

1. Mapping (reviewed) → canonical records; invalid rows are **dropped and counted by reason**, never repaired. Labels stay in a separate Series.
2. Exact duplicates on model-input features (+label) are removed; feature-identical rows with conflicting labels are removed entirely (`split.dedupe`, reported before/after).
3. Split **before** fitting anything: `stratified` (seeded, per class; classes < 3 rows go to train with a warning), `time` (chronological, needs `observed_at`), or `group` (no group spans two partitions; needs `group_column` in the mapping).
4. Preprocessing, classifier and anomaly detector are fitted on **train** only; thresholds and anomaly normalisation on **validation** only; the **test** split is touched once by `evaluate`.
5. Recorded in the bundle/manifest: dataset fingerprint (train), full-dataset fingerprint, split metadata + per-split class counts, dedupe and cross-split overlap counts, seeds, library versions, thresholds + how they were chosen.

**Limits of this protocol.** A random stratified split can still put flows from one host/session/attack run on both sides (optimistic scores) — prefer `group` or `time` when the dataset provides a usable key. Removing feature-identical rows changes class balance (DoS-type classes lose many near-duplicates). The 5 canonical numeric fields discard most of each dataset's native features; datasets lacking ports/protocol/timestamps carry constant columns for those fields, so results are not comparable across datasets.

## 6. Evaluation output

`EvaluationReport` records dataset name + fingerprint, split strategy + metadata, sample count, class support, thresholds, risk-policy version, package versions, timestamp, and:
* `metrics` (names from DATABASE_SCHEMA §2.9): `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `false_positive_rate` — each defined in `metric_definitions`; **operational** (final label, `unknown` counts as a miss). Macro averages are over classes present in the evaluation set; a class never predicted contributes precision 0. Metrics that are mathematically undefined are `null` (e.g. FPR without benign rows), never 0.
* `class_metrics` (precision/recall/F1/support/one-vs-rest FPR), `confusion_matrix` (rows = true labels, columns = predicted labels + `unknown`), `type_by_truth` (how each true class was typed benign/known_attack/anomaly/uncertain), `alert_rate`.
* `anomaly_metrics`: thresholded confusion, precision/recall/F1, FPR, alert rate, **ROC-AUC and PR-AUC with the attack prevalence (the PR-AUC baseline)**, per-attack-class anomaly recall, threshold + selection method.
* `classifier_closed_set` (raw argmax, no rejection; for comparing classifiers) incl. attack-score ROC/PR-AUC; `calibration` (ECE/bins).
* `warnings` (small-support classes, classes the model never saw, missing benign rows…).
Imbalance handling: class-weighted training, per-class and macro metrics, PR-AUC next to ROC-AUC. Threshold selection never uses the test set.

## 7. Held-out class / family protocol (objective 6)

`run_unknown_attack_protocol` (CLI: `scripts/unknown_attack.py`). For each held-out class or family: remove it from train **and** validation, fit the full bundle, select thresholds without it, then score all its rows together with the benign/known **test** rows. Reported per fold: how held-out rows were typed (`anomaly`/`uncertain`/misattributed `known_attack`/passed as `benign`), the benign alert rate under the same policy, anomaly ROC/PR-AUC (held-out vs benign) with prevalence, and a control (macro recall of the remaining known classes). Each report embeds:
* **Establishes:** for this dataset/feature set/seed/thresholds, how often a never-seen class still raises an alert, is misattributed, or passes as benign, beside the benign false-alarm rate.
* **Does not establish:** detection of unknown attacks in general. One class from one lab-style dataset under one distribution shift; same-dataset classes share capture artefacts; family grouping reduces but does not remove near-duplicate leakage; not comparable across datasets; benign novelty raises the same alerts.

## 8. Artifacts and controlled continual learning (ML_CONTRACT §8, §10)

**Artifacts.** `save_bundle` writes `<dir>/<version>/{bundle.pkl,manifest.json}` atomically to a **new** directory (refuses existing versions; versions match `^[a-z0-9][a-z0-9._-]{0,63}$`, no traversal) and returns SHA-256. `load_bundle(dir, version, expected_sha256=…)` requires the registry's checksum, rejects symlinks, checks manifest consistency, schema version, bundle/feature-list consistency and that scikit-learn/numpy/scipy/pandas versions equal the training runtime (`strict_runtime=False` to override deliberately). Deserialisation uses an allow-list unpickler (classes from `sklearn.*`/`nids_ml.*` plus a short list of NumPy helpers; `os`, `subprocess`, `eval`, sklearn test helpers etc. are refused even when the checksum matches — tested). This is defence-in-depth, **not** a sandbox: only load artifacts made by the controlled training pipeline.

**Cycle** (`lifecycle.run_candidate_cycle`, `scripts/continual_experiment.py`):
1. `FeedbackExample` rows enter training only if `training_eligible` **and** reviewer + review time are recorded; label must be in the allowed set; observation must validate; exact duplicates, contradictory pairs, rows overlapping validation/holdout, and rows contradicting a training label are dropped and counted (`candidate_dataset_manifest.json`).
2. Replay: seeded stratified sample of the original train split (`replay_fraction`, `min_replay_per_class` protects minorities); optional `max_feedback_fraction` caps drift from feedback.
3. Candidate trained in-process, thresholds reselected on the **same** validation split, saved as a **new** version, registered `candidate`.
4. Candidate and active model scored on the **same untouched holdout** (identical dataset fingerprint).
5. `validate_candidate` applies structural checks (report is evaluated, bound to this exact artifact checksum and version, schema compatible, fingerprint present) and configured gates. **Zero quantitative gates ⇒ fail.** Gates: `minimum_recall` (= attack-detection recall: attack rows typed ≠ benign), `maximum_false_positive_rate`, `minimum_macro_f1`, `minimum_per_class_recall` (lowest attack-class recall), `minimum_samples`, and regression gates vs the previous model on the same fingerprint: `maximum_macro_f1_drop`, `maximum_per_class_recall_drop` (every class), `maximum_false_positive_rate_increase`. Aggregate accuracy is never a gate.
6. `LocalModelRegistry` (candidate → validated|rejected → active → retired) promotes only a `validated` candidate whose passing `ValidationResult` matches its artifact checksum, with `expected_active_version` (stale ⇒ conflict), a non-empty reason, and an identical feature schema. `rollback` restores a previously validated, schema-compatible version; checksums are re-verified on load. Every transition is appended to an audit log. Nothing promotes automatically, and `predict`/`evaluate` never train.
Production state lives in the backend's `model_registry`; `LocalModelRegistry` mirrors its state machine so the transitions are testable without MongoDB. Retention of ≥ 2 prior validated artifacts is a backend/ops duty and is not enforced here.

## 9. Interpretations and open questions for the contract owner (no contract text was changed)

1. `validate_features` return type is unspecified → `FeatureValidationReport` (batch-level failures raise).
2. Threshold key names differ: ML_CONTRACT §6 says `known_confidence_threshold`; DATABASE_SCHEMA §2.8 says `known_confidence` → bundle uses the DB names `known_confidence` / `anomaly_threshold`.
3. ML_CONTRACT §7 wants `risk_policy_version` in "internal prediction metadata", but the 11-field output has no slot → it is in `bundle.metadata.risk_policy_version` and `EvaluationReport`. Decide how the backend should persist it per prediction.
4. `model_kind`: only `classifier_plus_anomaly` is documented; `classifier_only` and `anomaly_only` were added for degraded bundles.
5. `protocol` values outside the five allowed are rejected (strict). Alternative: coerce to `other`. Dataset mappings currently coerce via `value_map` + `default`.
6. No "explicitly marked metadata" convention for extra fields exists → default reject / optional strip-with-warning.
7. ML-internal types not defined by any contract: `EvaluationReport` (adds `artifact_sha256` to bind a report to one artifact), `PromotionGates` (API names `minimum_recall`, `maximum_false_positive_rate` kept; regression fields and `baseline_report` added), `ValidationResult`, `FeedbackExample`, `PredictionOptions`. `false_positive_rate` in `metrics` is the binary benign-flagged-as-non-benign rate.
8. Structure additions beyond PROJECT_SPEC §4 (all inside `ml/`): `inference.py`, `training.py`, `scripts/`, `configs/`. Fold them into the layout or move them if preferred.
9. Mapping format cannot yet convert epoch timestamps (needed for ToN-IoT `ts`) or other transforms; add under review rather than silently.

## 10. Limitations (read before using any number)

* **No real data has been used.** Model quality, thresholds, risk weights and unknown-attack behaviour are unmeasured. Synthetic-fixture outputs are not results.
* Public IDS datasets are lab-generated and commonly criticised for label noise, artefacts and class imbalance; verify each dataset's documentation and licence before use. Flow-level canonical features only; no payloads.
* `anomaly_score` and `confidence` are uncalibrated scores; benign novelty alarms. With the default 5 % benign alert budget, alert volume on busy networks may be unacceptable — choose `target_false_positive_rate` with operators.
* Risk weights and the decision policy are documented defaults awaiting evaluation; `class_weights` are empty until real class names exist.
* Restricted unpickling is not a security boundary; strict runtime-version matching means upgrading scikit-learn requires retraining and re-registering.
* `OUT_OF_DISTRIBUTION_WARNING` only covers unseen categories. No drift monitoring. No throughput/latency measurements exist, so none are claimed.
* Training is single-threaded (`n_jobs=1`) for reproducibility; large datasets will be slow — use `max_train_rows` for experiments and say so in reports.

## 11. Verified environment

Python 3.12.3 · numpy 2.4.4 · pandas 3.0.2 · scikit-learn 1.8.0 · scipy 1.17.1 · pydantic 2.14.1 · pytest 9.1.1 (pinned in `requirements*.txt`; clean-venv install and full suite executed on Linux as a non-interactive root container user — re-run as an ordinary user and on Windows before release).
