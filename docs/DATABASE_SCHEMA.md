# MongoDB Database Schema
**Version:** 1.0.0
**Access:** backend PyMongo repository layer only
**Database default:** `adaptive_nids`

## 1. Storage conventions

- Use BSON `ObjectId` internally; serialize as opaque string IDs in the API.
- Use BSON UTC `datetime` for timestamps. API converts to ISO 8601 UTC strings.
- Use `Decimal128` only if exact decimal financial-like values ever become necessary; NIDS scores use integer/double.
- Use `int` for counts and risk score, `double` for model scores, `bool` for flags, and arrays/embedded documents for bounded structured data.
- Never store password plaintext, access tokens, secrets, or raw packet payloads.
- Large files and model binaries live in controlled filesystem/object storage, not MongoDB documents. Store storage keys, size, checksum, and metadata. Do not return storage keys to clients.
- Every document has `_id`; most mutable business documents have `created_at` and `updated_at`.
- All timestamps are UTC.
- Repository methods are the only application code allowed to execute MongoDB queries.

## 2. Collections

### 2.1 `users`
```json
{
  "_id": "ObjectId",
  "email": "analyst@example.com",
  "email_normalized": "analyst@example.com",
  "display_name": "Analyst",
  "password_hash": "<argon2id hash>",
  "role": "analyst",
  "is_active": true,
  "created_at": "Date",
  "updated_at": "Date",
  "last_login_at": "Date|null"
}
```
Rules: lowercase/trim `email_normalized`; unique; password hash generated with Argon2id or approved secure alternative; roles restricted to `admin | analyst | viewer`. Do not return `password_hash`.

Indexes:
- unique `{email_normalized: 1}`
- `{role: 1, is_active: 1}`

### 2.2 `datasets`
```json
{
  "_id": "ObjectId",
  "dataset_name": "lab traffic",
  "description": "string|null",
  "source_type": "csv|pcap|pcapng|zeek_conn_log",
  "original_filename": "sanitized-name.csv",
  "storage_key": "opaque-internal-key",
  "file_size_bytes": 12345,
  "sha256": "hex checksum",
  "status": "uploaded|validating|ready|processing|completed|failed",
  "label_column": "Label|null",
  "timestamp_column": "Timestamp|null",
  "feature_profile": "auto|canonical_nids_v1",
  "row_count": "int|null",
  "validation_summary": {
    "valid_rows": 990,
    "invalid_rows": 10,
    "warnings": 1,
    "detected_columns": ["duration","protocol","destination_port"],
    "mapping": {"duration":"Duration","protocol":"Protocol"}
  },
  "error_summary": {"code":"SCHEMA_MISMATCH","message":"Safe user-facing summary"} ,
  "created_by": "ObjectId",
  "created_at": "Date",
  "updated_at": "Date",
  "processed_at": "Date|null"
}
```
`validation_summary` and `error_summary` may be null. Do not store all row-level errors in one document; store a capped sample (e.g. first 100) or a separate bounded report. `storage_key` is never a path supplied by the user.

Indexes:
- `{created_at: -1, _id: -1}`
- `{status: 1, created_at: -1}`
- `{created_by: 1, created_at: -1}`
- unique `{sha256: 1, created_by: 1, dataset_name: 1}` only if product explicitly wants duplicate prevention; otherwise use a non-unique lookup index.

### 2.3 `jobs`
```json
{
  "_id": "ObjectId",
  "job_type": "validate_dataset|process_dataset|validate_model|generate_report",
  "resource_id": "ObjectId",
  "status": "queued|running|completed|failed|cancelled",
  "stage": "string|null",
  "progress_percent": 40,
  "processed_count": 400,
  "total_count": 1000,
  "idempotency_key": "string",
  "created_by": "ObjectId",
  "error_code": "string|null",
  "error_summary": "string|null",
  "created_at": "Date",
  "started_at": "Date|null",
  "updated_at": "Date",
  "finished_at": "Date|null"
}
```
Progress is 0–100 or null. Unique `{job_type: 1, idempotency_key: 1}` for idempotent submission. Indexes `{status: 1, created_at: 1}`, `{resource_id: 1, created_at: -1}`.

### 2.4 `predictions`
```json
{
  "_id": "ObjectId",
  "dataset_id": "ObjectId",
  "job_id": "ObjectId",
  "row_index": 12,
  "observed_at": "Date|null",
  "prediction_label": "port_scan",
  "prediction_type": "known_attack|benign|anomaly|uncertain",
  "confidence": 0.91,
  "anomaly_score": 0.73,
  "risk_score": 76,
  "severity": "high",
  "reason_codes": ["KNOWN_CLASS_HIGH_SCORE","ANOMALY_ELEVATED"],
  "model_version": "baseline-1",
  "feature_schema_version": "1.0.0",
  "feature_summary": {"duration":0.2,"protocol":"tcp","destination_port":22},
  "created_at": "Date"
}
```
`confidence` and `anomaly_score` are nullable doubles in `[0,1]`. `risk_score` integer `[0,100]`. `feature_summary` must be allowlisted, bounded, and free of raw payloads. Unique `{dataset_id: 1, row_index: 1, model_version: 1}` if one prediction per row/model is the chosen behavior; use a separate prediction run ID if multiple scoring runs per model are needed.

Indexes:
- `{dataset_id: 1, row_index: 1}`
- `{created_at: -1, _id: -1}`
- `{severity: 1, created_at: -1}`
- `{prediction_type: 1, created_at: -1}`
- `{risk_score: -1, created_at: -1}`
- `{prediction_label: 1, created_at: -1}`

### 2.5 `alerts`
```json
{
  "_id": "ObjectId",
  "prediction_id": "ObjectId",
  "dataset_id": "ObjectId",
  "title": "High-risk network observation",
  "status": "open|acknowledged|investigating|resolved|false_positive",
  "severity": "high",
  "risk_score": 76,
  "prediction_label": "port_scan",
  "assigned_to": "ObjectId|null",
  "reason_codes": ["KNOWN_CLASS_HIGH_SCORE"],
  "created_at": "Date",
  "updated_at": "Date",
  "resolved_at": "Date|null"
}
```
One alert per prediction is the default; unique index on `prediction_id`. Alerts are generated only if the configured policy threshold is met. Indexes: `{status: 1, severity: 1, created_at: -1}`, `{assigned_to: 1, status: 1}`, `{dataset_id: 1, created_at: -1}`, `{risk_score: -1, created_at: -1}`.

### 2.6 `investigations`
```json
{
  "_id": "ObjectId",
  "alert_id": "ObjectId",
  "title": "Investigate suspected port scan",
  "status": "open|in_progress|resolved|closed",
  "priority": "critical|high|medium|low",
  "assigned_to": "ObjectId|null",
  "notes": [
    {"_id":"ObjectId","author_id":"ObjectId","body":"Note text","created_at":"Date"}
  ],
  "resolution": "string|null",
  "created_by": "ObjectId",
  "created_at": "Date",
  "updated_at": "Date",
  "closed_at": "Date|null"
}
```
For long-running investigations, prefer a separate `investigation_notes` collection rather than unbounded embedded notes. V1 may embed notes only with a hard cap (e.g. 200 notes and 5000 chars each); beyond the cap, migrate to separate collection. Unique partial index on `{alert_id: 1}` for active statuses if duplicate active investigations are prohibited. Indexes `{status: 1, updated_at: -1}`, `{assigned_to: 1, status: 1}`, `{alert_id: 1}`.

### 2.7 `feedback`
```json
{
  "_id": "ObjectId",
  "prediction_id": "ObjectId",
  "analyst_label": "benign",
  "disposition": "true_positive|false_positive|benign|needs_review",
  "comment": "string|null",
  "submitted_by": "ObjectId",
  "training_eligible": false,
  "training_eligibility_reviewed_by": "ObjectId|null",
  "training_eligibility_reviewed_at": "Date|null",
  "created_at": "Date"
}
```
Feedback is immutable after submission except for an explicit reviewed eligibility transition in a future approved workflow. Never automatically train on feedback. Indexes `{prediction_id: 1, created_at: -1}`, `{submitted_by: 1, created_at: -1}`, `{disposition: 1, created_at: -1}`.

### 2.8 `model_registry`
```json
{
  "_id": "ObjectId",
  "version": "candidate-2",
  "status": "candidate|validated|active|retired|rejected",
  "model_kind": "classifier_plus_anomaly",
  "feature_schema_version": "1.0.0",
  "artifact_storage_key": "opaque-internal-key",
  "artifact_sha256": "hex checksum",
  "training_dataset_fingerprint": "sha256",
  "preprocessing_version": "1.0.0",
  "thresholds": {"known_confidence":0.7,"anomaly_threshold":0.8},
  "metrics": null,
  "validation_report_id": "ObjectId|null",
  "created_by": "ObjectId",
  "created_at": "Date",
  "validated_at": "Date|null",
  "activated_at": "Date|null"
}
```
`metrics` remains null until an evaluation actually runs. Store named metrics with definitions, sample counts, split/fold metadata, dataset fingerprint, and evaluation timestamp. Unique `{version: 1}`; `{status: 1, created_at: -1}`.

### 2.9 `model_evaluations`
```json
{
  "_id": "ObjectId",
  "model_version": "candidate-2",
  "dataset_id": "ObjectId",
  "dataset_fingerprint": "sha256",
  "feature_schema_version": "1.0.0",
  "split_strategy": "holdout",
  "metrics": {"accuracy":0.0,"precision_macro":0.0,"recall_macro":0.0,"f1_macro":0.0,"false_positive_rate":0.0},
  "class_metrics": [],
  "gates": {"passed":false,"reasons":["not evaluated"]},
  "created_by": "ObjectId",
  "created_at": "Date"
}
```
Example numbers above are schema placeholders, not reported results. Do not persist placeholder metrics as real evaluations. Indexes `{model_version: 1, created_at: -1}`, `{dataset_id: 1, created_at: -1}`.

### 2.10 `reports`
```json
{
  "_id": "ObjectId",
  "report_type": "alerts_summary",
  "format": "json|csv",
  "status": "generating|completed|failed|expired",
  "filters": {},
  "storage_key": "opaque-internal-key|null",
  "created_by": "ObjectId",
  "created_at": "Date",
  "expires_at": "Date",
  "error_summary": "string|null"
}
```
Indexes `{created_by: 1, created_at: -1}`, `{status: 1, expires_at: 1}`. Delete report files after expiry and update status.

### 2.11 `audit_events`
```json
{
  "_id": "ObjectId",
  "actor_id": "ObjectId|null",
  "action": "dataset.process_requested",
  "resource_type": "dataset",
  "resource_id": "string|null",
  "outcome": "success|failure|denied",
  "request_id": "string",
  "metadata": {"job_id":"job_def456"},
  "created_at": "Date"
}
```
Append-only through application APIs. No update/delete endpoint. Do not store secrets, access tokens, raw payloads, or unrestricted user-supplied objects. Indexes `{created_at: -1, _id: -1}`, `{actor_id: 1, created_at: -1}`, `{resource_type: 1, resource_id: 1, created_at: -1}`, `{action: 1, created_at: -1}`.

### 2.12 `simulated_actions`
```json
{
  "_id": "ObjectId",
  "alert_id": "ObjectId",
  "action": "mark_host_for_review|create_ticket_simulation|notify_team_simulation",
  "status": "simulated",
  "reason": "string",
  "performed_by": "ObjectId",
  "created_at": "Date"
}
```
This collection is for audit/reporting only. No execution adapter to real enforcement tools may be connected. Indexes `{alert_id: 1, created_at: -1}`, `{performed_by: 1, created_at: -1}`.

## 3. Validation rules

- Use Pydantic validation before repository writes and MongoDB collection validators where practical.
- Configure MongoDB JSON Schema validators for required fields, BSON types, enums, numeric ranges, and bounded arrays.
- Normalize email; reject unknown role/status/enum values.
- Score bounds: confidence/anomaly score `[0,1]`, risk integer `[0,100]`.
- Strings have explicit maximum lengths; never permit unbounded notes, error details, or metadata.
- Enforce references in service logic. MongoDB does not provide foreign-key constraints; verify referenced records exist and handle deletion races.
- Use unique indexes for email, model version, and one-alert-per-prediction. Handle duplicate-key exceptions as `409 CONFLICT`.
- Use transactions only where supported and required. Atlas replica sets support transactions; where transactions are unavailable, use explicit state transitions and compensating actions.
- Avoid embedding unbounded arrays. Predictions are one document per observation; never put millions of predictions inside a dataset document.

## 4. Pagination and query safety

- Default limit 25; maximum 100.
- Prefer keyset pagination on `{created_at, _id}`. Cursor must be opaque, signed or otherwise validated, and encode only approved sort/filter state.
- Sort fields are server-defined; never interpolate user-supplied field names into Mongo queries.
- Allowlisted filters only. Escape/normalize search strings and use bounded prefix/text search if added.
- Apply time-range limits and query timeouts to expensive analytics.
- Dashboard aggregations must use indexed date filters and bounded buckets.

## 5. Retention recommendations

These are initial configurable recommendations, not legal requirements:
- Raw uploaded artifacts: 30 days by default after processing; allow admins to configure longer retention.
- Predictions and alerts: 180 days, subject to project requirements.
- Audit events: 365 days or longer where required by policy.
- Reports: 24 hours by default, then delete generated file and mark `expired`.
- Failed upload temp files: delete within 24 hours.
- Model artifacts and evaluation reports: retain active model plus at least two prior validated versions; retain evaluation metadata for reproducibility.
- Never use TTL on data that must be retained for audit/compliance. TTL deletion is asynchronous and should not be treated as exact-time deletion.
- Deletion jobs must log counts and outcomes without logging sensitive data.

## 6. Database operational requirements

- Use a least-privilege MongoDB Atlas database user, TLS, network access controls, and secrets from environment/secret manager.
- Do not log `MONGODB_URI`.
- Add startup ping and readiness checks.
- Use connection pooling and close clients on shutdown.
- Set bounded server selection/socket timeouts.
- Create indexes through a controlled initialization/migration step, not on every request.
- Test repositories against a dedicated test database or isolated fakes; never run test cleanup against production.
