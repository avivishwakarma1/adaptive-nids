# Machine Learning Contract
**Version:** 1.0.0  
**Package:** `ml/src/nids_ml`  
**Purpose:** deterministic, independently testable prediction and evaluation

## 1. Design goals and limitations

The ML package supports two complementary signals:
1. **Known-class classification:** assign a class from labels represented in the training data, subject to model confidence and configured rejection policy.
2. **Anomaly/novelty detection:** score deviation from a fitted baseline. An anomaly score does not prove malicious activity and can flag benign novelty.

The system must not claim that every unknown attack is detectable. Generalization depends on data quality, feature coverage, distribution shift, labels, thresholds, and evaluation design. The UI must distinguish a model's prediction from verified ground truth.

## 2. Public interface

The package exposes stable functions; no FastAPI, PyMongo, filesystem uploads, or HTTP code is permitted in this package.

```python
def validate_features(records, schema_version="1.0.0"): ...
def preprocess(records, artifact, schema_version="1.0.0"): ...
def predict(records, bundle, options=None) -> list[PredictionResult]: ...
def evaluate(bundle, labeled_records, config) -> EvaluationReport: ...
def validate_candidate(candidate_bundle, evaluation_report, gates) -> ValidationResult: ...
```

Use Pydantic models or equivalent strict validated dataclasses for public contracts. These names may be implemented as functions with typed signatures, but input/output semantics must not change without a contract version update.

## 3. Canonical feature schema

Canonical schema version: `1.0.0`. Each observation is a flow-level record, not an arbitrary raw packet payload.

Required logical fields:
- `duration`: finite nonnegative float, seconds.
- `protocol`: normalized lowercase string; allowed values `tcp`, `udp`, `icmp`, `other`, `unknown`.
- `source_port`: integer 0–65535 or null.
- `destination_port`: integer 0–65535 or null.
- `flow_packets`: finite nonnegative number or null.
- `flow_bytes`: finite nonnegative number or null.
- `packets_per_second`: finite nonnegative number or null.
- `bytes_per_second`: finite nonnegative number or null.
- `connection_state`: normalized string or `unknown`.
- `direction`: `inbound | outbound | internal | unknown`.
- `observed_at`: UTC timestamp or null.
- `source_id`: optional opaque row/flow identifier; never used as a predictive feature.

Optional supported fields may include TCP flag counts, source/destination byte counts, inter-arrival summaries, and Zeek-derived duration/state features only when they have documented definitions and versioned schema entries. Do not silently add features to v1. Raw source/destination IP addresses should not be used as model features by default; if retained for investigation, isolate them from model feature matrices and protect them as potentially sensitive data.

For generic CSV datasets with different feature names, ingestion must use an explicit, reviewed mapping to canonical fields. `auto` mapping may suggest mappings but must not silently infer an ambiguous mapping. Keep original labels separate from features to prevent label leakage.

### Numeric handling
- Reject NaN/Infinity in required numeric fields.
- Missing optional values are represented as null/NaN at the ingestion boundary and imputed using fitted training-set statistics.
- Do not fit imputers, scalers, encoders, feature selection, or dimensionality reduction on evaluation/test data.
- Use a persisted preprocessing pipeline bundled with the model.
- Transform must preserve exact feature order and names from the artifact.
- Unknown categorical values map to a documented `unknown`/other category or use encoder `handle_unknown`; they must not crash inference.

## 4. Input contract

Input is a batch of 1–configured maximum observations, each represented as a mapping of canonical feature names to primitive values.

Example:
```json
{
  "feature_schema_version": "1.0.0",
  "observations": [
    {
      "duration": 0.2,
      "protocol": "tcp",
      "source_port": 51515,
      "destination_port": 22,
      "flow_packets": 12,
      "flow_bytes": 1400,
      "packets_per_second": 60.0,
      "bytes_per_second": 7000.0,
      "connection_state": "S0",
      "direction": "unknown",
      "observed_at": "2026-10-24T12:30:00Z"
    }
  ]
}
```
This is the internal ML contract, not a public endpoint request. The backend adapts validated imported data into this form.

Input validation:
- schema version must match the model bundle or a declared compatible migration.
- required logical fields must exist after mapping.
- batch size is bounded by configuration.
- reject non-finite numeric values and invalid ranges.
- unknown extra fields are ignored only if explicitly marked metadata; otherwise reject or strip before model entry with a recorded warning.
- never pass raw PCAP bytes, packet payloads, file paths, or Mongo documents directly to ML.

## 5. Output contract

Each input observation yields exactly one output in the same order:

```json
{
  "row_index": 0,
  "prediction_label": "port_scan",
  "prediction_type": "known_attack",
  "confidence": 0.91,
  "anomaly_score": 0.73,
  "risk_score": 76,
  "severity": "high",
  "reason_codes": ["KNOWN_CLASS_HIGH_SCORE","ANOMALY_ELEVATED"],
  "model_version": "baseline-1",
  "feature_schema_version": "1.0.0",
  "prediction_timestamp": "2026-10-24T12:30:03Z"
}
```

Allowed `prediction_type`: `known_attack | benign | anomaly | uncertain`.  
`prediction_label`: a known class name, `benign`, or `unknown`. Normalize labels to lowercase snake_case.  
`confidence`: float `[0,1]` or null. Must not be called a probability unless calibration has been measured and documented.  
`anomaly_score`: float `[0,1]` or null; higher means more anomalous according to this detector's defined normalization.  
`risk_score`: integer `[0,100]`.  
`severity`: `critical | high | medium | low | info`.  
`reason_codes`: zero or more stable codes from an allowlisted vocabulary.  
`model_version`, `feature_schema_version`: exact identifiers used for inference.  
`prediction_timestamp`: UTC ISO 8601.

No NaN or Infinity may appear in serialized outputs. If a model component fails, the batch must return a controlled failure or explicit `uncertain` result per configured policy; it must not fabricate normal predictions.

## 6. Prediction decision policy

The implementation must document the exact estimator(s), thresholds, and policy version. Baseline behavior:
1. Run the fitted preprocessing pipeline.
2. Obtain classifier scores/label if a classifier is available.
3. Obtain anomaly score if an anomaly detector is available.
4. If classifier confidence is below configured `known_confidence_threshold`, return `prediction_label="unknown"` and `prediction_type="uncertain"` unless anomaly policy explicitly promotes the result to `anomaly`.
5. If anomaly score meets/exceeds configured `anomaly_threshold`, set or add anomaly signal. Do not overwrite a high-confidence known attack label without a documented arbitration rule.
6. For a predicted benign class with high anomaly score, use `prediction_type="anomaly"` and `prediction_label="unknown"` or preserve `benign` only in a separate classifier-output field; the canonical final label must not imply certainty.
7. If the known class is a non-benign attack and classifier policy accepts it, use `known_attack`.
8. If no model component is available, return `MODEL_UNAVAILABLE`; do not default to benign.

Exact thresholds are configuration/artifact metadata and must be learned or selected on a validation split, not guessed from test results.

## 7. Risk and severity contract

`risk_score` is a policy score, **not a probability**. Version the formula and record `risk_policy_version` in internal prediction metadata. A possible initial policy can combine class-risk weight, classifier confidence, and anomaly score, but the weights must be documented and evaluated before enabling production claims.

Required final mapping:
- 0–19: `info`
- 20–39: `low`
- 40–59: `medium`
- 60–79: `high`
- 80–100: `critical`

Clamp score to integer `[0,100]`. Boundary tests must cover 19/20, 39/40, 59/60, and 79/80. The backend must not independently recalculate severity using different thresholds.

Suggested reason-code vocabulary:
- `KNOWN_CLASS_HIGH_SCORE`
- `KNOWN_CLASS_LOW_CONFIDENCE`
- `ANOMALY_ELEVATED`
- `ANOMALY_HIGH`
- `FEATURES_INCOMPLETE`
- `OUT_OF_DISTRIBUTION_WARNING`
- `MODEL_COMPONENT_UNAVAILABLE`
- `RISK_POLICY_HIGH`
Codes are additive explanations, not causal proofs. Only emit codes supported by the actual path.

## 8. Model bundle and reproducibility

A bundle includes:
- model version and model kind;
- feature schema version and exact ordered feature list;
- preprocessing pipeline and fitted encoders/imputers/scalers;
- classifier and optional anomaly detector;
- class label mapping;
- thresholds and risk policy version;
- training dataset fingerprint and split metadata;
- library/runtime versions;
- artifact checksum;
- training/evaluation timestamps and metrics metadata.

Load only trusted local artifacts whose checksum and schema pass validation. Do not load arbitrary user-uploaded pickle/joblib artifacts. Scikit-learn pickle/joblib files can execute code during deserialization; model artifacts must be created by the controlled training pipeline and access-controlled. Consider safer formats where feasible.

## 9. Training and evaluation rules

- Split data before fitting preprocessing.
- Prefer stratified splits for classification where appropriate; use group/time-aware splits when flows from the same host/session or time periods could leak across splits.
- Keep a held-out test set untouched until final evaluation.
- Report per-class precision, recall, F1, macro-F1, confusion matrix, support, and false-positive rate where defined.
- For anomaly detection, report a thresholded confusion matrix when labels exist, precision/recall/F1, ROC-AUC or PR-AUC when mathematically appropriate, alert rate, and threshold selection method.
- Include class imbalance, missing data, duplicate records, leakage checks, and dataset provenance.
- Do not compare scores across datasets as if directly equivalent without caveats.
- Never fabricate accuracy, ROC-AUC, F1, recall, throughput, or latency.
- The UI must show `not evaluated` or null when metrics are unavailable.

## 10. Controlled continual learning

V1 uses **human-approved offline retraining**, not automatic online updates:
1. Feedback is stored with `training_eligible=false`.
2. A separate candidate dataset is reviewed, deduplicated, label-checked, and versioned.
3. Train a candidate in an isolated process.
4. Evaluate on a fixed validation protocol and an untouched holdout.
5. Validate schema compatibility, artifact checksum, and configured quality gates.
6. Admin explicitly promotes the candidate.
7. Keep previous validated artifacts for rollback.
8. Log candidate creation, evaluation, promotion, and rollback.
9. Never let a candidate overwrite the active artifact in place.
10. If validation fails, keep the active model unchanged.

Promotion gates must be explicit configuration, not hard-coded undocumented assumptions. A candidate must not be promoted solely because aggregate accuracy improves; assess false positives, minority-class recall, drift, and operational impact.

## 11. Required ML tests

- Input schema and range validation.
- Same input + same artifact/config gives deterministic output where estimator supports determinism.
- Output count/order matches input.
- Score bounds, severity thresholds, valid enums, and finite JSON values.
- Feature order and preprocessing artifact compatibility.
- Missing values and unseen categorical values.
- Label leakage prevention and split isolation.
- Serialization/deserialization with checksum verification.
- Known-class and anomaly paths tested independently and together.
- Low confidence yields documented uncertainty behavior.
- Missing model component does not silently produce benign.
- Candidate validation failure prevents promotion.
- Rollback restores prior compatible model.
- Evaluation output records dataset fingerprint, split strategy, sample count, class metrics, thresholds, and package versions.
