# REST API Contract
**Version:** 1.0.0
**Prefix:** `/api/v1`
**Format:** JSON except multipart uploads and binary report export
**Auth:** `Authorization: Bearer <access_token>` unless endpoint is marked public

## 1. Global rules

- All timestamps are ISO 8601 UTC strings, e.g. `2026-10-24T12:30:00Z`.
- IDs are opaque strings. MongoDB ObjectIds are serialized as 24-character hex strings but clients must not construct them.
- Unknown JSON fields should be rejected on write schemas where practical (`extra="forbid"`).
- JSON response objects use the exact fields shown. Nullable fields are explicitly `null`; omitted optional query parameters are not errors.
- Lists use `{ "items": [...], "next_cursor": "..." | null, "limit": 25 }`.
- Pagination: `limit` default 25, range 1–100; `cursor` opaque URL-safe token. Sort order is deterministic: `created_at` descending then `_id` descending unless endpoint states otherwise.
- API must never accept raw MongoDB filters, aggregation pipelines, filesystem paths, model artifact paths, shell commands, or arbitrary parser flags.
- `request_id` is generated per request and returned in response header `X-Request-ID` and error envelope.
- Authentication errors do not disclose whether an email exists.
- Rate-limit login and upload endpoints. Exact limits are deployment configuration, and `429` responses include `Retry-After` when applicable.

## 2. Common error envelope

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed.",
    "details": [
      {"field": "limit", "issue": "must be between 1 and 100"}
    ],
    "request_id": "req_01J..."
  }
}
```

`details` is either an array of safe field issues or `null`. Standard codes:
- `VALIDATION_ERROR` — 422
- `UNAUTHENTICATED` — 401
- `FORBIDDEN` — 403
- `NOT_FOUND` — 404
- `CONFLICT` — 409
- `UNSUPPORTED_MEDIA_TYPE` — 415
- `PAYLOAD_TOO_LARGE` — 413
- `RATE_LIMITED` — 429
- `PROCESSING_ERROR` — 422 for artifact-specific validation; 500 for unexpected server failure
- `MODEL_UNAVAILABLE` — 503
- `INTERNAL_ERROR` — 500

Example:
```json
{"error":{"code":"NOT_FOUND","message":"Resource not found.","details":null,"request_id":"req_01J..."}}
```

## 3. Shared enums and objects

### Roles
`admin | analyst | viewer`

### Severity
`critical | high | medium | low | info`

### Prediction type
`known_attack | benign | anomaly | uncertain`

### Prediction object
```json
{
  "id": "pred_abc123",
  "dataset_id": "ds_abc123",
  "row_index": 12,
  "observed_at": "2026-10-24T12:30:00Z",
  "prediction_label": "port_scan",
  "prediction_type": "known_attack",
  "confidence": 0.91,
  "anomaly_score": 0.73,
  "risk_score": 76,
  "severity": "high",
  "reason_codes": ["KNOWN_CLASS_HIGH_SCORE", "ANOMALY_ELEVATED"],
  "model_version": "baseline-1",
  "feature_schema_version": "1.0.0",
  "created_at": "2026-10-24T12:30:03Z"
}
```
`confidence` and `anomaly_score` may be `null` if unavailable. Scores are not probabilities unless explicitly documented as calibrated. `reason_codes` are stable machine-readable strings; UI text is mapped separately.

### User object
```json
{"id":"usr_abc123","email":"analyst@example.com","display_name":"Analyst","role":"analyst","is_active":true,"created_at":"2026-10-24T12:00:00Z"}
```
Never return password hash, JWT secret, or internal auth metadata.

## 4. Endpoints

### 4.1 Health

#### `GET /health`
Auth: none.
200:
```json
{"status":"ok","service":"adaptive-ai-nids-api","api_version":"1.0.0","timestamp":"2026-10-24T12:30:00Z"}
```
500 only if process cannot serve. This endpoint must not expose database URI, filesystem paths, installed packages, or secrets.

#### `GET /ready`
Auth: none.
200:
```json
{"status":"ready","dependencies":{"database":"ok","model":"ok"},"timestamp":"2026-10-24T12:30:00Z"}
```
503 if a required dependency is unavailable. For offline dev, model may be `degraded` only if the API is configured to start without predictions; the readiness policy must be explicit in config.

### 4.2 Authentication

#### `POST /auth/login`
Auth: none. Content-Type JSON.
Request:
```json
{"email":"analyst@example.com","password":"correct horse battery staple"}
```
Constraints: valid email; password length 1–128 at login.
200:
```json
{"access_token":"<jwt>","token_type":"bearer","expires_in":1800,"user":{"id":"usr_abc123","email":"analyst@example.com","display_name":"Analyst","role":"analyst","is_active":true,"created_at":"2026-10-24T12:00:00Z"}}
```
401 `UNAUTHENTICATED` for bad credentials or inactive user; 422 invalid schema; 429 rate limit. Access token claims: `sub` user ID, `role`, `iat`, `exp`, `iss`, `aud`. Use a strong secret and a modern password hash (Argon2id preferred; bcrypt acceptable with safe parameters). No refresh token in v1.

#### `POST /auth/logout`
Auth: any authenticated role.
Request: empty JSON `{}`.
204 No Content. Client must discard token. Because v1 access tokens are stateless, logout does not guarantee server-side immediate token revocation. Audit event required. 401 if invalid token.

#### `GET /auth/me`
Auth: any role.
200: `User object`.
401 invalid/expired token.

### 4.3 Datasets

#### `POST /datasets`
Auth: admin, analyst. `multipart/form-data` with:
- `file`: one file, required.
- `source_type`: `csv | pcap | pcapng | zeek_conn_log`, required.
- `dataset_name`: string 1–120 chars, required.
- `description`: string 0–1000 chars, optional.
- `label_column`: string 1–128 chars, optional; CSV only.
- `timestamp_column`: string 1–128 chars, optional; CSV only.
- `feature_profile`: `auto | canonical_nids_v1`, default `auto`.

Constraints: maximum file size from `MAX_UPLOAD_MB`; filename is sanitized and never used as a filesystem path. Only approved extensions/content are accepted.
201:
```json
{
  "id":"ds_abc123",
  "dataset_name":"lab traffic",
  "source_type":"csv",
  "status":"uploaded",
  "file_size_bytes":12345,
  "row_count":null,
  "validation_summary":null,
  "created_by":"usr_abc123",
  "created_at":"2026-10-24T12:30:00Z"
}
```
413 too large; 415 unsupported type; 422 invalid metadata/file; 403 viewer; 503 storage unavailable.

#### `GET /datasets`
Auth: all roles. Query: `limit`, `cursor`, optional `status`, `source_type`, `created_by`.
200:
```json
{"items":[{"id":"ds_abc123","dataset_name":"lab traffic","source_type":"csv","status":"ready","file_size_bytes":12345,"row_count":1000,"validation_summary":{"valid_rows":990,"invalid_rows":10,"warnings":1},"created_by":"usr_abc123","created_at":"2026-10-24T12:30:00Z"}],"next_cursor":null,"limit":25}
```
422 invalid filters.

#### `GET /datasets/{dataset_id}`
Auth: all roles.
200: dataset object as above, including `description`, `label_column`, `timestamp_column`, `feature_profile`, `error_summary` (nullable).
404 missing or inaccessible dataset.

#### `POST /datasets/{dataset_id}/validate`
Auth: admin, analyst. Request:
```json
{"strict":false,"sample_rows":20}
```
`strict` default false; `sample_rows` 0–100, default 20.
202:
```json
{"dataset_id":"ds_abc123","job_id":"job_abc123","status":"validating","message":"Validation queued."}
```
409 if validation/processing already running; 422 if unsupported source or invalid state.

#### `POST /datasets/{dataset_id}/process`
Auth: admin, analyst. Request:
```json
{"create_alerts":true,"max_rows":100000}
```
`create_alerts` default true; `max_rows` 1–configured max.
202:
```json
{"dataset_id":"ds_abc123","job_id":"job_def456","status":"processing","message":"Dataset processing queued."}
```
Requires validation status `ready`; 409 otherwise. Processing must be bounded and idempotent per dataset/job key.

#### `GET /datasets/{dataset_id}/status`
Auth: all roles.
200:
```json
{"dataset_id":"ds_abc123","status":"processing","stage":"scoring","processed_rows":400,"total_rows":1000,"progress_percent":40,"error_code":null,"updated_at":"2026-10-24T12:35:00Z"}
```
Statuses: `uploaded | validating | ready | processing | completed | failed`. `progress_percent` 0–100 or null if unknown. 404 if absent.

#### `GET /datasets/{dataset_id}/predictions`
Auth: all roles. Query: `limit`, `cursor`, optional `severity`, `prediction_type`, `prediction_label`, `min_risk_score` 0–100.
200: paginated list of `Prediction object`. 422 invalid filters.

### 4.4 Predictions

#### `GET /predictions`
Auth: all roles. Query: `limit`, `cursor`, optional `dataset_id`, `severity`, `prediction_type`, `prediction_label`, `min_risk_score`, `from`, `to`. `from`/`to` must be ISO 8601; `from <= to`.
200: paginated list of `Prediction object`.

#### `GET /predictions/{prediction_id}`
Auth: all roles.
200:
```json
{
  "id":"pred_abc123",
  "dataset_id":"ds_abc123",
  "row_index":12,
  "observed_at":"2026-10-24T12:30:00Z",
  "prediction_label":"port_scan",
  "prediction_type":"known_attack",
  "confidence":0.91,
  "anomaly_score":0.73,
  "risk_score":76,
  "severity":"high",
  "reason_codes":["KNOWN_CLASS_HIGH_SCORE","ANOMALY_ELEVATED"],
  "model_version":"baseline-1",
  "feature_schema_version":"1.0.0",
  "feature_summary":{"duration":0.2,"protocol":"tcp","destination_port":22},
  "created_at":"2026-10-24T12:30:03Z"
}
```
`feature_summary` is a safe allowlisted subset; never expose packet payloads or secrets. 404 missing.

### 4.5 Alerts

#### `GET /alerts`
Auth: all roles. Query: `limit`, `cursor`, optional `status` (`open | acknowledged | investigating | resolved | false_positive`), `severity`, `min_risk_score`, `dataset_id`, `assigned_to`, `from`, `to`.
200:
```json
{"items":[{"id":"alt_abc123","prediction_id":"pred_abc123","dataset_id":"ds_abc123","title":"High-risk network observation","status":"open","severity":"high","risk_score":76,"prediction_label":"port_scan","assigned_to":null,"created_at":"2026-10-24T12:30:03Z","updated_at":"2026-10-24T12:30:03Z"}],"next_cursor":null,"limit":25}
```

#### `GET /alerts/{alert_id}`
Auth: all roles.
200 includes alert object plus `reason_codes`, `model_version`, `notes_count`, `investigation_id` nullable. 404 missing.

#### `PATCH /alerts/{alert_id}`
Auth: admin, analyst. Request may include one or more:
```json
{"status":"acknowledged","assigned_to":"usr_abc123","title":"Reviewed suspicious scan"}
```
Allowed status: `open | acknowledged | investigating | resolved | false_positive`. `assigned_to` is user ID or null; title 1–160 chars. Unknown fields rejected.
200: updated alert object. 404 missing; 409 invalid state transition; 422 invalid field. Every change creates an audit event.

#### `POST /alerts/{alert_id}/simulate-response`
Auth: admin, analyst. Request:
```json
{"action":"mark_host_for_review","reason":"Analyst-approved tabletop simulation"}
```
Allowed actions only: `mark_host_for_review | create_ticket_simulation | notify_team_simulation`. This endpoint **only records a simulation**; it must not call operating-system networking tools, firewalls, routers, switches, endpoint agents, or third-party enforcement APIs.
201:
```json
{"id":"sim_abc123","alert_id":"alt_abc123","action":"mark_host_for_review","status":"simulated","reason":"Analyst-approved tabletop simulation","performed_by":"usr_abc123","created_at":"2026-10-24T12:40:00Z"}
```
422 unsupported action; 403 viewer. Audit event required.

### 4.6 Investigations

#### `POST /investigations`
Auth: admin, analyst. Request:
```json
{"alert_id":"alt_abc123","title":"Investigate suspected port scan","priority":"high","assigned_to":"usr_abc123"}
```
`title` 1–160 chars; `priority`: `critical | high | medium | low`; `assigned_to` nullable.
201:
```json
{"id":"inv_abc123","alert_id":"alt_abc123","title":"Investigate suspected port scan","status":"open","priority":"high","assigned_to":"usr_abc123","notes":[],"created_by":"usr_abc123","created_at":"2026-10-24T12:45:00Z","updated_at":"2026-10-24T12:45:00Z"}
```
404 alert/user; 409 if policy disallows duplicate active investigation for same alert.

#### `GET /investigations`
Auth: all roles. Query: `limit`, `cursor`, optional `status` (`open | in_progress | resolved | closed`), `priority`, `assigned_to`, `alert_id`.
200: paginated investigation objects. Viewer notes may be redacted according to deployment policy; v1 default permits reading investigation notes to authenticated roles.

#### `GET /investigations/{investigation_id}`
Auth: all roles.
200: investigation object. 404 missing.

#### `PATCH /investigations/{investigation_id}`
Auth: admin, analyst. Request:
```json
{"status":"in_progress","priority":"medium","assigned_to":"usr_abc123","resolution":"Needs additional dataset review"}
```
Allowed status `open | in_progress | resolved | closed`; `resolution` 0–2000 chars or null.
200 updated object; 409 invalid transition; 404 missing. Audit event required.

#### `POST /investigations/{investigation_id}/notes`
Auth: admin, analyst. Request:
```json
{"body":"Compared the observation with baseline traffic; follow-up required."}
```
Body length 1–5000 chars.
201:
```json
{"id":"note_abc123","author_id":"usr_abc123","body":"Compared the observation with baseline traffic; follow-up required.","created_at":"2026-10-24T12:50:00Z"}
```
404 investigation; 422 invalid body.

### 4.7 Dashboard

#### `GET /dashboard/summary`
Auth: all roles. Query: optional `from`, `to`, `dataset_id`.
200:
```json
{
  "range":{"from":"2026-10-01T00:00:00Z","to":"2026-10-24T23:59:59Z"},
  "totals":{"datasets":4,"observations":1000,"alerts":120,"open_alerts":35},
  "severity_counts":{"critical":2,"high":18,"medium":45,"low":40,"info":15},
  "prediction_type_counts":{"known_attack":60,"benign":800,"anomaly":110,"uncertain":30},
  "risk_series":[{"bucket_start":"2026-10-24T12:00:00Z","count":12,"mean_risk_score":47.5}],
  "generated_at":"2026-10-24T23:00:00Z"
}
```
All counts are nonnegative integers. `mean_risk_score` is 0–100 or null. Empty data returns zero counts and an empty series. 422 invalid date range.

### 4.8 Reports

#### `POST /reports`
Auth: all roles. Request:
```json
{"report_type":"alerts_summary","format":"json","filters":{"from":"2026-10-01T00:00:00Z","to":"2026-10-24T23:59:59Z","severity":["high","critical"],"dataset_id":null}}
```
`report_type`: `alerts_summary | dataset_analysis | model_evaluation | audit_summary`; `format`: `json | csv`. Filter fields are allowlisted by report type.
202:
```json
{"id":"rpt_abc123","report_type":"alerts_summary","format":"json","status":"generating","created_at":"2026-10-24T13:00:00Z"}
```
403 if role cannot read source records; 422 invalid filters.

#### `GET /reports/{report_id}`
Auth: all roles, with authorization checked against report scope.
200:
```json
{"id":"rpt_abc123","report_type":"alerts_summary","format":"json","status":"completed","download_url":"/api/v1/reports/rpt_abc123/download","expires_at":"2026-10-25T13:00:00Z","created_at":"2026-10-24T13:00:00Z"}
```
`download_url` is a same-origin API path, not a public storage URL. Status: `generating | completed | failed | expired`. 404 missing.

#### `GET /reports/{report_id}/download`
Auth: all roles with report authorization.
200: JSON or CSV file bytes with correct `Content-Type` and safe `Content-Disposition`. 404 absent/expired. No arbitrary path access.

### 4.9 Feedback

#### `POST /feedback`
Auth: admin, analyst. Request:
```json
{"prediction_id":"pred_abc123","analyst_label":"benign","disposition":"false_positive","comment":"Confirmed as expected backup traffic."}
```
`analyst_label` is a known class or `benign | unknown`; `disposition`: `true_positive | false_positive | benign | needs_review`; comment nullable, max 2000 chars.
201:
```json
{"id":"fb_abc123","prediction_id":"pred_abc123","analyst_label":"benign","disposition":"false_positive","comment":"Confirmed as expected backup traffic.","submitted_by":"usr_abc123","training_eligible":false,"created_at":"2026-10-24T13:10:00Z"}
```
`training_eligible` is always false on creation in v1. Feedback must not automatically change the active model. 404 prediction.

#### `GET /feedback`
Auth: admin, analyst. Query: `limit`, `cursor`, optional `prediction_id`, `disposition`, `submitted_by`.
200: paginated feedback objects. Viewers receive 403.

### 4.10 Model lifecycle

#### `GET /models`
Auth: all roles.
200:
```json
{"active_model_version":"baseline-1","items":[{"version":"baseline-1","status":"active","model_kind":"classifier_plus_anomaly","feature_schema_version":"1.0.0","created_at":"2026-10-20T10:00:00Z","metrics_summary":null}],"next_cursor":null,"limit":25}
```
`metrics_summary` is null until actual metrics exist.

#### `GET /models/{version}`
Auth: all roles.
200 includes version, status (`candidate | validated | active | retired | rejected`), model kind, feature schema version, artifact checksum, training dataset fingerprint, metrics, thresholds, creator, timestamps, and validation summary. Never return artifact bytes or server paths. 404 unknown version.

#### `POST /models/{version}/validate`
Auth: admin. Request:
```json
{"evaluation_dataset_id":"ds_eval123","minimum_recall":null,"maximum_false_positive_rate":null}
```
Thresholds may be null; when supplied must be within `[0,1]`. Actual configured gates and results must be stored.
202:
```json
{"version":"candidate-2","status":"validation_queued","job_id":"job_eval123"}
```
409 wrong state; 422 incompatible schema or missing evaluation dataset.

#### `POST /models/{version}/promote`
Auth: admin. Request:
```json
{"expected_active_version":"baseline-1","validation_report_id":"eval_abc123","reason":"Passed approved evaluation gates"}
```
201:
```json
{"active_model_version":"candidate-2","previous_active_version":"baseline-1","promoted_by":"usr_abc123","promoted_at":"2026-10-24T14:00:00Z"}
```
Promotion requires passing validation, artifact checksum verification, feature schema compatibility, and optimistic concurrency using `expected_active_version`. 409 on stale active version or failed gates. Must be atomic from the application's perspective and audited.

#### `POST /models/rollback`
Auth: admin. Request:
```json
{"target_version":"baseline-1","expected_active_version":"candidate-2","reason":"Regression observed during controlled review"}
```
201:
```json
{"active_model_version":"baseline-1","previous_active_version":"candidate-2","rolled_back_by":"usr_abc123","rolled_back_at":"2026-10-24T14:10:00Z"}
```
Target must be a previously validated compatible version. 409 stale version or incompatible target. Audit event required.

### 4.11 Audit logs

#### `GET /audit`
Auth: admin; analyst may query only their own events. Query: `limit`, `cursor`, optional `actor_id`, `action`, `resource_type`, `from`, `to`.
200:
```json
{"items":[{"id":"aud_abc123","actor_id":"usr_abc123","action":"dataset.process_requested","resource_type":"dataset","resource_id":"ds_abc123","outcome":"success","request_id":"req_01J...","metadata":{"job_id":"job_def456"},"created_at":"2026-10-24T12:35:00Z"}],"next_cursor":null,"limit":25}
```
Audit metadata must not contain passwords, tokens, secrets, raw packet payloads, or arbitrary unbounded user input. Audit records are append-only through application interfaces. 403 viewer.

## 5. Status codes by operation

- `200`: successful read or update.
- `201`: resource created.
- `202`: accepted asynchronous job.
- `204`: successful no-content operation.
- `400`: malformed request syntax if not mapped to 422.
- `401`: missing, invalid, or expired authentication.
- `403`: authenticated but role is insufficient.
- `404`: resource not found or deliberately concealed.
- `409`: invalid state transition, duplicate conflict, stale model version.
- `413`: upload too large.
- `415`: unsupported media type/file format.
- `422`: schema or domain validation failure.
- `429`: rate limited.
- `500`: generic unexpected error.
- `503`: required dependency/model unavailable.

## 6. OpenAPI and compatibility requirements

FastAPI-generated OpenAPI is the machine-readable API source, but this document defines product semantics. Add tests that:
1. compare route paths/methods to this contract;
2. validate response models and enum values;
3. assert the standard error envelope for handled errors;
4. verify every mutating endpoint has role enforcement and appropriate audit coverage;
5. reject undocumented query filters and request fields;
6. ensure no endpoint exposes MongoDB credentials, local paths, raw payloads, or model artifact paths.
