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
- `POST /api/v1/predictions` accepts `{"observations":[...]}` and returns a JSON array of the ML package's 11-field `PredictionResult` objects. Input is validated by the ML package, with a maximum of 1,000 observations and a request body limit from `MAX_UPLOAD_MB`. This endpoint performs inference only and does not persist results: the documented prediction records require dataset/job context, which this standalone request does not supply. Dataset-linked persistence should be added with the dataset-processing contract rather than inferred here.
- Create the first admin user through a controlled seed/admin procedure; public registration is intentionally absent.
- The current code is a runnable foundation: health/readiness, Mongo lifecycle/index setup, verified ML predictor loading, error envelope, Argon2id helpers, JWT issue/validation, and login/logout/me route skeleton.
- Dataset processing, alerts, investigations, feedback, model lifecycle, reports, audit persistence, and upload parsing are not yet implemented. Do not treat the predictor adapter as an implemented dataset-processing endpoint.
- JWT logout is stateless in v1; client must discard the access token.
- Add deployment-grade login/upload rate limiting before exposure beyond local development.
