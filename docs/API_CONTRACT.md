# REST API Contract
**Version:** 1.0.0
**Prefix:** `/api/v1`
**Format:** JSON except multipart uploads and binary report export
**Auth:** `Authorization: Bearer <access_token>` unless endpoint is marked public

## Implementation status

This document includes both implemented behavior and planned API design. The
planned endpoint specifications below preserve product requirements but are not
available in the current backend.

Implemented HTTP routes:

| Method | Path | Status |
|---|---|---|
| `GET` | `/health` | Implemented |
| `GET` | `/ready` | Implemented |
| `POST` | `/api/v1/auth/login` | Implemented |
| `POST` | `/api/v1/auth/logout` | Implemented |
| `GET` | `/api/v1/auth/me` | Implemented |
| `POST` | `/api/v1/predictions` | Implemented |
| `GET` | `/api/v1/dashboard/summary` | Implemented |

All other endpoint specifications in this document are **planned and not
currently available**, including dataset upload/parsing and processing,
prediction list/detail reads, alert management, investigations, reports,
feedback, general model lifecycle operations, and audit-log retrieval.

## 1. Global rules

- Contract-wide requirements below are intended semantics, not guarantees that
  every planned endpoint is implemented.
- Implemented APIs return timestamps as ISO 8601 UTC strings.
- IDs are opaque strings. MongoDB ObjectIds are serialized as 24-character hex strings but clients must not construct them.
- The implemented login request rejects extra fields, as does the top-level
  prediction request. Prediction observation fields are validated against the
  ML feature schema. Other write-schema behavior below is a planned
  requirement.
- Implemented response fields follow their declared response models or route
  return values. Nullable fields are explicitly `null` when defined that way;
  omitted optional dashboard query parameters are supported.
- The `items`/`next_cursor`/`limit` list shape and pagination rules below are
  planned; no paginated list endpoint is currently implemented.
- Security design requirement: APIs must never accept raw MongoDB filters,
  aggregation pipelines, filesystem paths, model artifact paths, shell
  commands, or arbitrary parser flags.
- The implemented request middleware accepts a printable `X-Request-ID` (truncated
  to 100 characters) or generates one, and returns it in the response header.
  The handled error responses use the common envelope with a request ID.
- Implemented login errors do not disclose whether an email exists.
- Rate limiting for login and upload endpoints is a planned deployment
  requirement; the current backend does not implement rate limiting or
  `Retry-After`.

## 2. Common error envelope

The backend currently uses this envelope for handled HTTP exceptions,
request-validation failures, and unexpected application exceptions. This does
not imply that errors generated outside the application (for example, by a
proxy) use it. The codes and status mappings in this section are the intended
contract; not every listed operation or status is currently implemented.

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

`details` is either an array of field issues or `null`. The HTTP exception
handler currently maps 401 to `UNAUTHENTICATED`, 403 to `FORBIDDEN`, 404 to
`NOT_FOUND`, 409 to `CONFLICT`, 413 to `PAYLOAD_TOO_LARGE`, 415 to
`UNSUPPORTED_MEDIA_TYPE`, 429 to `RATE_LIMITED`, and 503 to
`MODEL_UNAVAILABLE`. Request-schema validation and the prediction feature
validator return `VALIDATION_ERROR` with status 422. A directly raised HTTP
422 currently falls through to `INTERNAL_ERROR`; 500 errors use
`INTERNAL_ERROR`. The common envelope is implemented, but error codes are not
yet fully consistent across all 422 and 503 paths.

Other intended codes:
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
This endpoint does not check database or model readiness. It must not expose a
database URI, filesystem paths, installed packages, or secrets.

#### `GET /ready`
Auth: none.
200:
```json
{"status":"ready","dependencies":{"database":"ok","model":"ok"},"timestamp":"2026-10-24T12:30:00Z"}
```
503 if the database is unavailable or the model is unavailable while
`ALLOW_DEGRADED_MODEL=false`. Otherwise the model dependency reports `degraded`
when no predictor is loaded and degraded mode is allowed.

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
401 for bad credentials or inactive user; 422 invalid request schema; 503 when
authentication storage or token configuration is unavailable. No rate limiter
is currently implemented. Access token claims: `sub` user ID, `role`, `jti`,
`iat`, `exp`, `iss`, `aud`. Passwords are hashed with Argon2id. No refresh
token is implemented.

#### `POST /auth/logout`
Auth: any authenticated role.
Request: no request body is required.
204 No Content. The backend records the token identifier (`jti`) in
`revoked_tokens` until its expiry; authentication checks this collection, so a
successfully revoked token is rejected on subsequent authenticated requests.
The client should also discard its token. 401 if authentication is invalid or
the token is revoked; 503 if revocation persistence fails. Logout does not
currently create a general audit event.

#### `GET /auth/me`
Auth: any role.
200: `User object`.
401 for invalid/expired/revoked tokens, inactive users, or when the user's
current database role is missing, invalid, or differs from the role in the
token. Authentication checks the current database role on each request. 503
if authentication storage is unavailable.

### 4.3 Datasets — Planned (not currently available)

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

#### `POST /predictions` — Implemented
Auth: `admin` or `analyst` only. The user's current database role must match
the role in their token; otherwise authentication returns 401. An authenticated
viewer is denied with 403, and the denial is best-effort recorded in
`audit_events` with the UTC timestamp, actor subject and role, action,
request method and path, and a generic denial reason. If the audit write fails,
the request remains denied; this limited best-effort event is not general
audit coverage.

Request: `application/json`, with exactly one top-level field:

```json
{"observations":[{"duration":0.5,"protocol":"tcp","source_port":12345,"destination_port":443,"flow_packets":10,"flow_bytes":500,"packets_per_second":2,"bytes_per_second":100,"connection_state":"established","direction":"outbound","observed_at":"2026-10-24T12:30:00Z"}]}
```

`observations` must contain 1–1,000 canonical records. Required fields for
each observation: `duration`, `protocol`, `source_port`, `destination_port`,
`flow_packets`, `flow_bytes`, `packets_per_second`, `bytes_per_second`,
`connection_state`, `direction`, and `observed_at`. `source_id` is optional.
The ML feature schema validates field values, rejects unknown observation
fields, and requires `observed_at` to be an ISO 8601 timestamp carrying a UTC
offset or `Z`. Invalid feature data returns 422. The configured request body
limit is controlled by `MAX_UPLOAD_MB`.

200 response: a JSON array of `PredictionResult` objects, in predictor result
order. Each object has exactly these fields:

| Field | Type / constraint |
|---|---|
| `row_index` | Integer, at least 0 |
| `prediction_label` | String, lowercase snake_case |
| `prediction_type` | `known_attack \| benign \| anomaly \| uncertain` |
| `confidence` | Number in `[0,1]` or `null` |
| `anomaly_score` | Number in `[0,1]` or `null` |
| `risk_score` | Integer in `[0,100]` |
| `severity` | `critical \| high \| medium \| low \| info` |
| `reason_codes` | Array of allowlisted strings |
| `model_version` | Non-empty string |
| `feature_schema_version` | Non-empty string |
| `prediction_timestamp` | UTC ISO 8601 timestamp ending in `Z` |

`confidence` and `anomaly_score` are not asserted to be calibrated
probabilities. Result rows must map one-to-one to submitted observation indices.

Errors: 401 missing/invalid/revoked authentication or stale user role; 403
authenticated role is not permitted; 413 request exceeds configured body
limit; 422 request schema or ML feature validation failure; 503 when
authentication checks, prediction service/inference, result mapping, or
database persistence is unavailable. Unexpected application failures use 500.
Handled errors use the common envelope.

Successful standalone predictions are stored in `standalone_predictions`,
separately from dataset predictions. This endpoint does not create dataset
records. An alert is created only for a `known_attack` prediction with
`risk_score >= 60` and severity `high` or `critical`; it starts with status
`open`. This is an implemented conservative rule, not a claim about model
accuracy or real-world detection quality.

#### `GET /predictions` — Planned (not currently available)
Auth: all roles. Query: `limit`, `cursor`, optional `dataset_id`, `severity`, `prediction_type`, `prediction_label`, `min_risk_score`, `from`, `to`. `from`/`to` must be ISO 8601; `from <= to`.
200: paginated list of `Prediction object`.

#### `GET /predictions/{prediction_id}` — Planned (not currently available)
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

### 4.5 Alerts — Planned (not currently available)

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

### 4.6 Investigations — Planned (not currently available)

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

#### `GET /dashboard/summary` — Implemented
Auth: all roles. Query: optional `from`, `to`, `dataset_id`.

`from` and `to` are optional ISO 8601 datetime query values. Naive datetime
values are interpreted as UTC; offset-aware values are converted to UTC. If
both are omitted, the range is the preceding 24 hours ending at request time.
If only `to` is supplied, `from` defaults to 24 hours before it; if only
`from` is supplied, `to` defaults to request time. `from` later than `to`
returns 422. `dataset_id`, when supplied, must be 1–128 characters.

Dataset totals count dataset records by `created_at`. Observation totals count
legacy predictions by `observed_at`; standalone predictions are included only
when no `dataset_id` filter is supplied. Alerts and open alerts are counted
from `alerts` by `created_at` (and the dataset filter when supplied).
`severity_counts` counts alert severities, while prediction-type counts and
risk series aggregate prediction records. Risk buckets are UTC hours for
ranges up to 48 hours and UTC days for longer ranges. Risk aggregation includes
scores from 0 through 100; a null mean remains null.

200:
The response shape below is illustrative; counts are computed from database
data and are never hardcoded. `generated_at` is the time the summary was
generated; it does not indicate model training or model health.

```json
{
  "range":{"from":"<UTC datetime>","to":"<UTC datetime>"},
  "totals":{"datasets":0,"observations":0,"alerts":0,"open_alerts":0},
  "severity_counts":{"critical":0,"high":0,"medium":0,"low":0,"info":0},
  "prediction_type_counts":{"known_attack":0,"benign":0,"anomaly":0,"uncertain":0},
  "risk_series":[],
  "generated_at":"<UTC datetime>"
}
```
The response follows `DashboardSummaryResponse`. Counts are nonnegative
integers; each risk point has `bucket_start`, `count`, and `mean_risk_score`
(0–100 or null). Empty data returns zero counts and an empty series. Database
failure returns 503; missing or invalid authentication returns 401.

### 4.8 Reports — Planned (not currently available)

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

### 4.9 Feedback — Planned (not currently available)

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

### 4.10 Model lifecycle — Planned (not currently available)

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

### 4.11 Audit logs — Planned (not currently available)

#### `GET /audit`
Auth: admin; analyst may query only their own events. Query: `limit`, `cursor`, optional `actor_id`, `action`, `resource_type`, `from`, `to`.
200:
```json
{"items":[{"id":"aud_abc123","actor_id":"usr_abc123","action":"dataset.process_requested","resource_type":"dataset","resource_id":"ds_abc123","outcome":"success","request_id":"req_01J...","metadata":{"job_id":"job_def456"},"created_at":"2026-10-24T12:35:00Z"}],"next_cursor":null,"limit":25}
```
Audit metadata must not contain passwords, tokens, secrets, raw packet payloads, or arbitrary unbounded user input. Audit records are append-only through application interfaces. 403 viewer.

## 5. Status codes by operation

This is the intended status-code vocabulary for the API contract, not a claim
that every operation listed below exists. Implemented endpoints use only the
statuses described in their endpoint sections.

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
- `429`: rate limited (planned; no rate limiter is implemented).
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
