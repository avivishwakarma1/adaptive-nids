# Adaptive AI-Driven NIDS — Backend Foundation

Python 3.11+, FastAPI, Pydantic v2, PyMongo, MongoDB Atlas, Argon2id password hashing, JWT, pytest.

## Run locally (PowerShell)

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env: set MONGODB_URI and replace JWT_SECRET_KEY with a random secret of at least 32 bytes.
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`.

## Test

```powershell
cd backend
pytest -q
```

## Notes

- MongoDB is accessed only through the backend PyMongo layer. Never put Atlas credentials in `VITE_*` variables.
- The backend installs the sibling `ml/` package and loads the configured active version only when its `model_registry` record has status `active`. The registry's `artifact_sha256` is passed to `nids_ml.load_bundle`; the local manifest is never treated as the trust source. The loaded predictor is available as `app.state.predictor` for backend processing workflows.
- When no verified active model is available, readiness reports the configured degraded state; with `ALLOW_DEGRADED_MODEL=false`, readiness returns `503`. Prediction calls through the predictor preserve the ML package's `PredictionResult` contract and failures are not replaced with benign results.
- `POST /api/v1/predictions` accepts `{"observations":[...]}` and returns a JSON array of the ML package's 11-field `PredictionResult` objects. Input is validated by the ML package, with a maximum of 1,000 observations and a request body limit from `MAX_UPLOAD_MB`. Successful standalone requests persist mapped rows in `standalone_predictions`; they do not create dataset records or set `dataset_id`. The existing unique `(dataset_id, row_index)` index on `predictions` remains unchanged. The new collection uses a unique `(prediction_batch_id, row_index)` index and an `observed_at` index.
- Standalone alerts are created only when `prediction_type` is `known_attack`, `risk_score` is at least 60, and severity is `high` or `critical`; they start with status `open`. This is a conservative alerting rule, not a claim about model accuracy or real-world detection quality. Dashboard summaries include standalone prediction counts and risk data only without a `dataset_id` filter; dataset counts continue to come from real dataset records.
- Persistence uses a MongoDB transaction when supported. If transactions are unavailable, writes use compensating deletion on failure; if compensation itself fails, partial records may remain and the request still returns a controlled error. The prediction request has no documented idempotency key: retrying it may create a duplicate prediction batch and duplicate alerts. `X-Request-ID` is not treated as an idempotency key.
- Create the first admin user through a controlled seed/admin procedure; public registration is intentionally absent.
- The current code is a runnable foundation: health/readiness, Mongo lifecycle/index setup, verified ML predictor loading, error envelope, Argon2id helpers, JWT issue/validation, login/logout/me, standalone prediction persistence, and prediction-triggered alerts.
- Dataset upload/parsing and processing, investigations, feedback, model lifecycle, reports, and general audit persistence and retrieval are not yet implemented. Authorization-denial events are best-effort written to `audit_events`; this is not general audit coverage. Standalone prediction persistence is not a substitute for dataset processing.
- JWT logout requires authentication and records the token identifier in `revoked_tokens` through its expiry. Subsequent authentication checks reject that token; logout returns 503 if revocation persistence fails. Clients should also discard the access token.
- The ML package and alert rule have not been demonstrated to provide accurate detection on real-world traffic; no real-world detection-quality claim is made.
- Add deployment-grade login/upload rate limiting before exposure beyond local development.
