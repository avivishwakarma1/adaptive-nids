# Integration Guide
**Version:** 1.0.0
**Target:** Windows + VS Code local development; compatible with macOS/Linux
**Status:** instructions for assembling modules, not evidence the application already runs

## 1. Development prerequisites

Install:
- Git
- VS Code
- Python 3.11 or newer
- Node.js LTS and npm
- MongoDB Atlas account and a database user with least privilege
- Optional: a supported offline PCAP parser library, added only by the ingestion owner

Do not install Docker as a requirement. Atlas is the default database. A local MongoDB instance may be used for isolated development if the team chooses, but it is not required by this contract.

## 2. Environment setup

Copy `.env.example` to `backend/.env` or load it through the chosen local environment manager. Do not commit `.env`.

Backend variables:
```dotenv
NIDS_ENV=development
NIDS_API_PREFIX=/api/v1
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>/<database>?retryWrites=true&w=majority
MONGODB_DATABASE=adaptive_nids
JWT_SECRET_KEY=<generate-a-long-random-secret>
JWT_ACCESS_TOKEN_MINUTES=30
JWT_ISSUER=adaptive-ai-nids
JWT_AUDIENCE=adaptive-ai-nids-frontend
CORS_ORIGINS=http://localhost:5173
MAX_UPLOAD_MB=100
MAX_ROWS_PER_DATASET=250000
MAX_BATCH_SIZE=1000
ARTIFACT_STORAGE_DIR=./var/artifacts
MODEL_ARTIFACT_DIR=../ml/artifacts
ACTIVE_MODEL_VERSION=baseline-1
LOG_LEVEL=INFO
```
The MongoDB URI above is a placeholder, not a real credential. URL-encode special characters in the password. Use Atlas TLS and restrict network access.

Frontend `frontend/.env.local`:
```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1
VITE_APP_NAME=Adaptive AI NIDS
```
Never place JWT secrets, MongoDB URIs, database credentials, or private API keys in `VITE_` variables; Vite exposes them to browser code.

## 3. Python environment

From repository root in PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .\ml
pip install -e .\backend
```
If editable installs are not configured yet, use the backend and ML owners' documented development install commands. Do not silently create divergent dependencies in separate modules. Lock dependencies once the initial package manifests exist.

Run tests:
```powershell
python -m pytest backend/app/tests ml/tests
```
On macOS/Linux, create/activate `.venv` with `python3.11 -m venv .venv` and `source .venv/bin/activate`.

## 4. Frontend environment

```powershell
cd frontend
npm ci
npm run dev
```
Expected default dev URL: `http://localhost:5173`. The frontend should show a clear API connection error if backend is down; it must not attempt database access.

Suggested `package.json` scripts:
- `dev`: Vite development server
- `build`: production build
- `preview`: preview built bundle
- `test`: React Testing Library test runner
- `test:e2e`: Playwright

## 5. Local startup order

1. **Configure Atlas** and verify the database user and network allowlist.
2. **Set environment variables** for backend and frontend.
3. **Create indexes/validators** using the backend's controlled initialization command.
4. **Bootstrap the first admin** using a one-time script that reads credentials securely from environment or prompts interactively. Never hard-code a default password.
5. **Verify ML artifacts**. If no real model exists, the backend may start in a documented degraded mode, but prediction endpoints must return `503 MODEL_UNAVAILABLE`; it must not return fake predictions.
6. **Start backend**:
   ```powershell
   uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
   ```
   API docs in development: `http://127.0.0.1:8000/docs`.
7. **Check health/readiness**:
   ```powershell
   Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
   Invoke-RestMethod http://127.0.0.1:8000/api/v1/ready
   ```
8. **Start frontend** in another terminal with `npm run dev`.
9. **Run the end-to-end flow** with a safe fixture dataset.
10. **Review logs and audit events** and confirm no sensitive values were emitted.

Basic operation must not require Administrator PowerShell, sudo, or root privileges.

## 6. Contract-first GitHub workflow

Recommended branches:
- `main`: protected, passing CI.
- `feature/contracts`: architecture/contract work.
- `feature/backend-*`
- `feature/ml-*`
- `feature/frontend-*`
- `feature/ingestion-*`
- `feature/tests-*`

Before implementation begins:
1. Commit all six specification documents.
2. Create GitHub issues from the module ownership list.
3. Assign one owner per module and one integration owner.
4. Require every issue to reference the relevant contract section.
5. Require pull requests to include tests and documentation updates.
6. Do not let agents add new endpoint names or schema fields without a contract change.

PR checklist:
- [ ] No undocumented endpoint, field, enum, status, or environment variable.
- [ ] Pydantic schemas match API contract.
- [ ] Frontend uses API client only.
- [ ] Repository layer contains all MongoDB access.
- [ ] ML imports do not depend on backend/FastAPI/PyMongo.
- [ ] Unit/contract tests added and pass.
- [ ] Audit and RBAC behavior covered.
- [ ] No secrets, captures, or large artifacts committed.
- [ ] Safety policy and non-admin operation preserved.

## 7. Parallel AI agent work allocation

### Agent A — Architecture/contracts (ChatGPT or architect owner)
Owns the six docs, shared enums, OpenAPI review, integration checklist, and contract tests. Must not independently invent features outside scope.

### Agent B — ML engine (Claude or ML-focused owner)
Owns `ml/src/nids_ml/` and `ml/tests/`. Delivers strict input/output models, preprocessing, known-class model interface, anomaly interface, risk mapping, evaluation, and candidate lifecycle. Must provide deterministic fixture tests and must not invent metrics.

### Agent C — Backend/API (ChatGPT or Claude)
Owns `backend/app/`. Implements auth, services, repositories, upload jobs, API schemas, audit, reports, and model lifecycle endpoints. Imports the ML public interface only. No direct Mongo queries in route files.

### Agent D — Frontend (GitHub Copilot in VS Code, guided by contract owner)
Owns `frontend/src/`. Builds login, dashboard, datasets, predictions, alerts, investigations, feedback, reports, and model pages. Uses only `/api/v1`; uses exact contract field names and enums. No fake API success data outside clearly marked storybook/test fixtures.

### Agent E — Integration/testing (shared owner)
Owns API/ML contract tests, Playwright journey, CI, seed/test fixtures, and integration bug triage. May modify module code only through reviewed PRs.

AI allocation is a recommendation, not a claim that any particular model is intrinsically best. Use each agent for bounded tasks, review generated code, and test it locally.

## 8. Interface integration sequence

1. **Freeze schemas:** common enums, Prediction, User, Dataset, Alert, error envelope, ML input/output.
2. **Build ML in isolation:** `pytest ml/tests`; demonstrate inference against a small synthetic fixture or return explicit unavailable state.
3. **Build backend skeleton:** health, readiness, error envelope, auth, dependency injection.
4. **Build repositories:** indexes, serializers, pagination, and tests against a test database/fakes.
5. **Connect backend to ML interface:** inject a predictor object; never import internal estimators from route modules.
6. **Implement ingestion:** CSV first, then PCAP/PCAPNG and optional Zeek adapter. All adapters emit canonical flow records.
7. **Implement jobs and persistence:** idempotent dataset processing and bounded batches.
8. **Implement alerts/investigations/feedback/audit.**
9. **Build frontend against OpenAPI:** common Axios client, auth interceptor, typed-by-convention JS schema constants, loading/empty/error states.
10. **Run Playwright end-to-end.**
11. **Perform safety review and dependency scan.**
12. **Record test results and known limitations** in README/release notes.

## 9. End-to-end acceptance criteria

Using a synthetic CSV fixture with known schema:
1. Unauthenticated access to a protected endpoint returns 401.
2. Login returns a token and `/auth/me` returns the expected role.
3. Viewer upload/mutation is rejected with 403.
4. Analyst uploads a CSV within limits and receives 201 with `uploaded`.
5. Validation returns a job and reaches `ready` or a documented validation failure.
6. Processing queues a job and reaches `completed` or a safe, explicit failure.
7. Each accepted row yields one prediction with valid enums, finite bounded scores, consistent severity, model version, and feature schema version.
8. Alerts are created only according to configured threshold policy.
9. Dashboard totals agree with stored fixture records.
10. Analyst can create an investigation, add a note, update status, and submit feedback.
11. Feedback does not automatically change model version or training eligibility.
12. Simulated response creates an audit record and makes no external or host action.
13. Report export returns valid JSON/CSV and respects access control/expiry.
14. Admin can validate/promote a passing candidate and rollback to a compatible prior version; failed validation blocks promotion.
15. Audit records exist for sensitive mutations and contain no secrets.
16. Malformed, oversized, traversal-named, and unsupported files are rejected safely.
17. Frontend works without direct MongoDB connectivity or credentials.
18. All tests pass on a normal non-admin user account.

## 10. Troubleshooting

- `503 MODEL_UNAVAILABLE`: verify configured artifact, checksum, feature schema, and startup logs. Do not bypass by returning benign.
- Atlas connection failure: check URI encoding, network access list, DB user permissions, TLS, and local environment loading. Do not print the URI.
- CORS failure: align `CORS_ORIGINS` with the exact frontend origin.
- `422` upload/validation: inspect safe validation summary, expected columns, label mapping, and file limits.
- Empty dashboard: verify time range, dataset filter, and whether processing has completed.
- Contract mismatch: update the source contract and tests first; then update all dependent modules in one reviewed change.
- Flaky ML results: set random seeds where appropriate, record dependency versions, and use deterministic fixtures; do not confuse reproducibility with real-world detection quality.

## 11. Release gate

A release candidate requires green backend, ML, frontend, contract, and Playwright tests; no known critical safety issue; a clean secret scan; verified model artifact provenance; and documented limitations. Any unavailable or unimplemented feature must be marked as such in the UI and documentation.
